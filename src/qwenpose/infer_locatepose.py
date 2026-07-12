from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional for minimal envs.
    tqdm = None

from qwenpose.data import (
    ALL_POSE_PROMPT,
    PoseRecord,
    PoseRecordDataset,
    load_refhuman_records,
    pose_collate,
)
from qwenpose.eval_pose import load_eval_model, resolve_checkpoint, tensor_to_prediction_rows
from qwenpose.schemas import SCHEMA_INDICES, SCHEMA_KEYPOINTS, SCHEMA_TO_ID, UNION_KEYPOINTS
from qwenpose.train_pose import (
    LocatePoseUnifiedConfig,
    LocatePoseUnifiedRuntime,
    _context_scale_for_indices,
    expand_boxes_xyxy_per_box,
    move_batch_to_device,
    parse_locate_bbox_response,
    parse_locate_boxes_for_task,
    prepare_box_conditioning,
    save_pose_visualization,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

FORMAT_TO_SCHEMA = {
    "coco": "COCO17",
    "refhuman": "COCO17",
    "mpii": "MPII16",
    "crowdpose": "CrowdPose14",
    "aic": "AIC14",
}
VISION_LORA_ENV = "QWENPOSE_VLLM_LOCATE_VISION_LORA_ADAPTER"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LocatePose 图片/文件夹/RefHuman train-val 推理入口；默认使用 transformers 单次特征复用。"
    )

    # --checkpoint：必填，LocatePose checkpoint 路径；可以是 run 目录、stage 目录、checkpoint-* 目录或 qwenpose_checkpoint.pt 文件。
    parser.add_argument("--checkpoint", type=Path, required=True, help="LocatePose checkpoint 路径：run/stage/checkpoint 目录或 qwenpose_checkpoint.pt。")
    # --output_dir：必填，本次推理输出目录；会写入 predictions、manifest、summary 和 visualizations。
    parser.add_argument("--output_dir", type=Path, required=True, help="推理输出目录。")
    # --input：输入图片文件或图片文件夹；可重复传入；RefHuman 可传数据集根目录或 images 目录。
    parser.add_argument("--input", dest="inputs", type=Path, action="append", default=[], help="输入图片文件或目录，可重复指定。")
    # --image：显式指定单张图片；可重复传入，适合少量图片逐个列出。
    parser.add_argument("--image", dest="images", type=Path, action="append", default=[], help="显式图片路径，可重复指定。")
    # --image_paths：一次性传入多张图片路径；支持逗号/分号/系统 pathsep 分隔。
    parser.add_argument("--image_paths", type=str, default="", help="多图片路径字符串，支持逗号、分号或系统 pathsep 分隔。")
    # --recursive：输入为目录时是否递归扫描子目录；默认只扫当前目录一层。
    parser.add_argument("--recursive", action="store_true", help="递归扫描输入目录下的图片。")
    # --format：输出/关键点格式；refhuman 会自动使用 COCO17 schema 但任务为 REF_POSE。
    parser.add_argument("--format", choices=sorted(FORMAT_TO_SCHEMA), default="coco", help="推理输出格式：coco/mpii/crowdpose/refhuman/aic。")
    # --split：RefHuman 标注 split；仅在从 RefHuman_train.json/RefHuman_val.json 读取 caption 时使用。
    parser.add_argument("--split", choices=["train", "val"], default="val", help="RefHuman 标注 split：train 或 val。")
    # --refhuman_root：RefHuman 数据集根目录；目录下应有 RefHuman_train.json、RefHuman_val.json 和 images/。
    parser.add_argument("--refhuman_root", type=Path, default=None, help="RefHuman 根目录，包含 RefHuman_train/val.json 和 images/。")
    # --refhuman_max_captions_per_image：RefHuman 标注模式下每张图片最多保留多少条 caption；0 表示全部保留。
    parser.add_argument("--refhuman_max_captions_per_image", type=int, default=0, help="RefHuman 每张图最多使用多少条 caption；0 表示全部。")
    # --start_index：排序后的图片起始下标，包含该下标；用于文件夹 0-n 或任意范围切片。
    parser.add_argument("--start_index", type=int, default=0, help="排序后图片起始下标，包含。")
    # --end_index：排序后的图片结束下标，不包含该下标；例如 start=0,end=10 表示前 10 张。
    parser.add_argument("--end_index", type=int, default=None, help="排序后图片结束下标，不包含；为空表示直到末尾。")
    # --num_images：在 start/end/random 选择后最多保留多少张图片；适合快速 smoke test。
    parser.add_argument("--num_images", type=int, default=None, help="最多保留多少张图片。")
    # --random_n：在 start/end 切片后随机抽取 n 张图片；和 seed 配合保证可复现。
    parser.add_argument("--random_n", type=int, default=None, help="随机抽取 n 张图片。")
    # --seed：随机种子；控制 random_n 和其他可复现随机选择。
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--num_parts", type=int, default=1, help="split selected records into N parts。")
    parser.add_argument("--worker_index", type=int, default=0, help="0-based part id for this run。")
    # --caption：RefHuman 任意图片模式下所有图片共用的 caption；不依赖 RefHuman 标注文件。
    parser.add_argument("--caption", type=str, default="", help="RefHuman 任意图片模式的统一 caption。")
    # --caption_file：RefHuman 任意图片模式下的 caption 映射文件；支持 JSON/JSONL/TSV。
    parser.add_argument("--caption_file", type=Path, default=None, help="RefHuman 任意图片 caption 映射文件，支持 JSON/JSONL/TSV。")
    # --interactive_captions：当 RefHuman 任意图片没有 caption 时，在交互终端逐张询问操作人员。
    parser.add_argument("--interactive_captions", dest="interactive_captions", action="store_true", default=True, help="缺少 caption 时在 TTY 中交互输入。")
    # --no_interactive_captions：禁用交互输入；缺少 caption 时直接报错，适合批处理。
    parser.add_argument("--no_interactive_captions", dest="interactive_captions", action="store_false", help="禁用 RefHuman caption 交互输入。")

    # --backbone：推理骨干名称；locatepose 是脚本层别名，Python 内部会转换为 eagle。
    parser.add_argument("--backbone", choices=["locatepose", "eagle"], default="locatepose", help="骨干名称；locatepose 会映射到 eagle。")
    # --locate_model_path：LocateAnything-3B 权重目录；需与训练脚本 locatepose.sh 保持一致。
    parser.add_argument("--locate_model_path", "--eagle_model_path", dest="eagle_model_path", type=str, default="weights/LocateAnything-3B", help="LocateAnything-3B 权重目录。")
    # --locate_dtype：LocateAnything PyTorch 模型 dtype；4090 推荐 bfloat16。
    parser.add_argument("--locate_dtype", "--eagle_dtype", dest="eagle_dtype", choices=["bfloat16", "float16", "float32", "auto", "none"], default="bfloat16", help="LocateAnything PyTorch dtype。")
    # --locate_attn_implementation：Locate vision tower attention 实现；默认 flash_attention_2 更省显存更快。
    parser.add_argument("--locate_attn_implementation", "--eagle_attn_implementation", dest="eagle_attn_implementation", type=str, default="flash_attention_2", help="Locate vision attention 实现。")
    # --locate_min_pixels：兼容参数；LocateAnything processor 通常不使用 Qwen-style min_pixels。
    parser.add_argument("--locate_min_pixels", "--eagle_min_pixels", dest="eagle_min_pixels", type=int, default=None, help="Locate 最小像素预算兼容参数。")
    # --locate_max_pixels：Locate 图像最大像素预算；会转成 processor 支持的图像限制逻辑。
    parser.add_argument("--locate_max_pixels", "--eagle_max_pixels", dest="eagle_max_pixels", type=int, default=None, help="Locate 最大像素预算。")
    # --locate_image_token_limit：Locate 原生图像 token 上限；和训练/eval 入口保持同名同义。
    parser.add_argument("--locate_image_token_limit", "--eagle_image_token_limit", dest="eagle_image_token_limit", type=int, default=None, help="Locate 原生图像 token 上限；空表示不覆盖 processor 默认。")
    # --locate_feature_size：Locate token 特征投影后的空间特征图边长；需与 checkpoint 配置一致。
    parser.add_argument("--locate_feature_size", "--eagle_feature_size", dest="eagle_feature_size", type=int, default=64, help="Locate 特征图输出边长。")
    # --locate_feature_refiner_layers：Locate feature refiner 层数；加载 checkpoint 时会优先使用保存配置。
    parser.add_argument("--locate_feature_refiner_layers", "--eagle_feature_refiner_layers", dest="eagle_feature_refiner_layers", type=int, default=2, help="Locate feature refiner 层数。")
    # --locate_feature_refiner_bottleneck_dim：feature refiner bottleneck 维度；需与训练配置匹配。
    parser.add_argument("--locate_feature_refiner_bottleneck_dim", "--eagle_feature_refiner_bottleneck_dim", dest="eagle_feature_refiner_bottleneck_dim", type=int, default=256, help="Locate feature refiner bottleneck 维度。")
    # --locate_feature_refiner_init_scale：feature refiner 残差初始化尺度；推理时主要用于构建结构。
    parser.add_argument("--locate_feature_refiner_init_scale", "--eagle_feature_refiner_init_scale", dest="eagle_feature_refiner_init_scale", type=float, default=0.1, help="Locate feature refiner 初始化尺度。")
    # --locate_lora_r：Locate 语言/主干 LoRA rank；需与训练配置一致以正确构建 adapter。
    parser.add_argument("--locate_lora_r", "--eagle_lora_r", dest="eagle_lora_r", type=int, default=32, help="Locate 主干 LoRA rank。")
    # --locate_lora_alpha：Locate 语言/主干 LoRA alpha；需与训练配置一致。
    parser.add_argument("--locate_lora_alpha", "--eagle_lora_alpha", dest="eagle_lora_alpha", type=int, default=64, help="Locate 主干 LoRA alpha。")
    # --locate_lora_dropout：Locate 主干 LoRA dropout；推理时结构参数需与训练一致。
    parser.add_argument("--locate_lora_dropout", "--eagle_lora_dropout", dest="eagle_lora_dropout", type=float, default=0.05, help="Locate 主干 LoRA dropout。")
    # --locate_vision_lora_r：Locate vision tower LoRA rank；需与训练配置一致。
    parser.add_argument("--locate_vision_lora_r", "--eagle_vision_lora_r", dest="eagle_vision_lora_r", type=int, default=16, help="Locate vision LoRA rank。")
    # --locate_vision_lora_alpha：Locate vision tower LoRA alpha；需与训练配置一致。
    parser.add_argument("--locate_vision_lora_alpha", "--eagle_vision_lora_alpha", dest="eagle_vision_lora_alpha", type=int, default=32, help="Locate vision LoRA alpha。")
    # --locate_vision_lora_dropout：Locate vision LoRA dropout；推理时结构参数需与训练一致。
    parser.add_argument("--locate_vision_lora_dropout", "--eagle_vision_lora_dropout", dest="eagle_vision_lora_dropout", type=float, default=0.05, help="Locate vision LoRA dropout。")

    # --locate_generation_backend：Locate 生成框后端；默认 transformers，可复用 PoseHead 特征。
    parser.add_argument("--locate_generation_backend", choices=["vllm", "transformers", "auto"], default="transformers", help="Locate 生成框后端；默认 transformers，可复用 PoseHead 特征。")
    # --box_source：PoseHead 条件框来源；默认闭环使用 Locate 生成框，gt 用于诊断 stage1 的 GT-box 上限。
    parser.add_argument("--box_source", choices=["locate_generate", "gt"], default="locate_generate", help="PoseHead 条件框来源：locate_generate 或 gt。")
    # --disable_vllm_fallback：vLLM 初始化/生成失败时不回退 transformers，直接抛错。
    parser.add_argument("--disable_vllm_fallback", action="store_true", help="禁用 vLLM 失败后的 transformers 回退。")
    # --gpu：指定可见 GPU，例如 0、1 或 0,1；会设置 CUDA_VISIBLE_DEVICES。
    parser.add_argument("--gpu", type=str, default="", help="指定推理 GPU，如 0 或 0,1。")
    # --vllm_tensor_parallel_size：vLLM tensor parallel 大小；单 4090 通常设 1。
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1, help="vLLM tensor parallel size。")
    # --vllm_gpu_memory_utilization：vLLM 可使用的 GPU 显存比例；越高越激进。
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.85, help="vLLM GPU 显存占用比例。")
    # --vllm_cpu_offload_gb：vLLM 权重 CPU offload 大小；GPU 显存紧张时可设 8/16/20。
    parser.add_argument("--vllm_cpu_offload_gb", type=float, default=0.0, help="vLLM CPU offload GiB。")
    # --vllm_enforce_eager：强制 vLLM eager 执行；自定义模型调试或显存紧张时更稳。
    parser.add_argument("--vllm_enforce_eager", action="store_true", help="强制 vLLM eager 执行。")
    # --vllm_max_model_len：vLLM 最大上下文长度；0 表示使用 vLLM/模型默认值。
    parser.add_argument("--vllm_max_model_len", type=int, default=0, help="vLLM 最大上下文长度；0 为默认。")
    # --vllm_batch_size：vLLM 预生成 Locate 框时每批请求数；0 表示自动同步 batch_size。
    parser.add_argument("--vllm_batch_size", type=int, default=0, help="vLLM 生成框请求 batch size；0 表示同步 batch_size。")
    # --vllm_max_num_seqs：vLLM scheduler 最大并发序列数；0 表示同步 vllm_batch_size。
    parser.add_argument("--vllm_max_num_seqs", type=int, default=0, help="vLLM max_num_seqs；0 同步 vllm_batch_size。")
    # --vllm_max_num_batched_tokens：vLLM prefill/profile token 预算；0 表示按 max_model_len 或 2048 自动设置。
    parser.add_argument("--vllm_max_num_batched_tokens", type=int, default=0, help="vLLM max_num_batched_tokens；0 自动。")
    # --vllm_model_impl：vLLM 模型实现；LocateAnything 已注册项目内 custom model，默认 auto。
    parser.add_argument("--vllm_model_impl", choices=["auto", "transformers", "vllm"], default="auto", help="vLLM model_impl；LocateAnything custom model 请保持 auto/vllm。")
    # --vllm_lora_adapter：vLLM LoRA adapter 路径；auto 会从 checkpoint 附近查找，none 表示不用。
    parser.add_argument("--vllm_lora_adapter", type=str, default="auto", help="vLLM LoRA adapter 路径：auto/none/具体路径。")
    # --vllm_max_lora_rank：vLLM LoRA 最大 rank；需覆盖训练时 locate_lora_r/vision_lora_r。
    parser.add_argument("--vllm_max_lora_rank", type=int, default=64, help="vLLM LoRA 最大 rank。")
    # --vllm_trust_remote_code：vLLM 加载 LocateAnything 自定义代码；本地 custom model 通常必须开启。
    parser.add_argument("--vllm_trust_remote_code", action="store_true", default=True, help="vLLM trust_remote_code。")
    # --no_vllm_trust_remote_code：关闭 vLLM trust_remote_code；一般不建议。
    parser.add_argument("--no_vllm_trust_remote_code", dest="vllm_trust_remote_code", action="store_false", help="关闭 vLLM trust_remote_code。")

    # --hidden_dim：PoseHead hidden dimension；加载旧 checkpoint 缺少 pose_config 时使用。
    parser.add_argument("--hidden_dim", type=int, default=448, help="PoseHead hidden dimension。")
    # --pose_decoder_layers：Pose decoder 层数；加载旧 checkpoint 缺少 pose_config 时使用。
    parser.add_argument("--pose_decoder_layers", type=int, default=3, help="Pose decoder 层数。")
    # --refinement_steps：关键点 refinement 步数；加载旧 checkpoint 缺少 pose_config 时使用。
    parser.add_argument("--refinement_steps", type=int, default=3, help="关键点 refinement 步数。")
    # --decoder_heads：Pose decoder attention head 数；加载旧 checkpoint 缺少 pose_config 时使用。
    parser.add_argument("--decoder_heads", type=int, default=8, help="Pose decoder attention head 数。")
    # --box_condition_scale：PoseHead 使用 box 前的扩框比例；影响关键点上下文。
    parser.add_argument("--box_condition_scale", type=float, default=1.2, help="PoseHead 条件框扩展比例。")
    # --pose_roi_size：每个 box 的 ROI 特征采样边长；越大越耗显存。
    parser.add_argument("--pose_roi_size", type=int, default=16, help="Pose ROI 特征边长。")
    # --disable_refinement：关闭关键点 refinement 分支；用于兼容无 refinement checkpoint 或加速。
    parser.add_argument("--disable_refinement", action="store_true", help="关闭关键点 refinement。")

    # --batch_size：PyTorch PoseHead/Locate 特征前向 batch size；vLLM 生成框 batch 单独由 vllm_batch_size 控制。
    parser.add_argument("--batch_size", type=int, default=1, help="PyTorch 姿态推理 batch size。")
    # --num_workers：DataLoader worker 数；本地图片调试默认 0 更稳。
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker 数。")
    # --prefetch_factor：DataLoader worker 预取因子；仅 num_workers>0 生效。
    parser.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader prefetch factor。")
    # --max_instances：每张图最多保留/推理的人体框数量。
    parser.add_argument("--max_instances", type=int, default=80, help="每张图最多人体实例数。")
    # --locate_generation_mode：LocateAnything generate 模式；hybrid 兼顾速度和稳定性。
    parser.add_argument("--locate_generation_mode", choices=["fast", "slow", "hybrid"], default="hybrid", help="LocateAnything 生成模式。")
    parser.add_argument("--single_pass_prompt", choices=["locate", "pose"], default="locate", help="单次特征复用时使用的 prompt：locate 复用纯定位 prompt，pose 复用 PoseHead prompt。")
    parser.add_argument("--disable_single_pass_features", action="store_true", help="禁用 transformers backend 的 Locate/PoseHead 特征复用，回退到两次前向。")
    # --locate_box_max_new_tokens：Locate 生成框文本最大新 token 数；多人图可适当增大。
    parser.add_argument("--locate_box_max_new_tokens", type=int, default=8192, help="Locate 生成框最大新 token 数。")
    # --box_nms_iou_thresh：仅在显式启用 PoseHead 前 NMS 时使用。
    parser.add_argument("--box_nms_iou_thresh", type=float, default=0.70, help="可选的 PoseHead 前 NMS IoU 阈值。")
    parser.add_argument(
        "--disable_pre_pose_nms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认保留全部 Locate 框进入 PoseHead。",
    )
    parser.add_argument("--post_pose_nms_iou_thresh", type=float, default=0.95, help="PoseHead 输出后的高阈值去重。")
    # --score_threshold：可视化和结果筛选的关键点/person score 阈值。
    parser.add_argument("--score_threshold", type=float, default=0.05, help="预测分数阈值。")
    # --max_predictions_per_image：每张图最多写出多少个预测实例。
    parser.add_argument("--max_predictions_per_image", type=int, default=100, help="每张图最多输出预测数。")
    # --visualize_max_samples：最多保存多少张可视化；-1 表示全部，0 表示关闭。
    parser.add_argument("--visualize_max_samples", type=int, default=-1, help="最多保存多少张可视化；-1 全部，0 关闭。")
    # --visualize_max_instances：每张可视化最多绘制多少个人体实例。
    parser.add_argument("--visualize_max_instances", type=int, default=8, help="每张可视化最多绘制实例数。")
    # --disable_progress：关闭 tqdm 进度条；适合写日志或非交互环境。
    parser.add_argument("--disable_progress", action="store_true", help="关闭进度条。")
    # --device：PyTorch PoseHead/Locate 特征推理设备；指定 gpu 时通常保持 cuda 即可。
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="PyTorch 推理设备。")
    return parser.parse_args()


def split_image_paths(value: str) -> list[Path]:
    if not value:
        return []
    normalized = value.replace(";", ",")
    parts: list[str] = []
    for chunk in normalized.split(","):
        parts.extend(piece for piece in chunk.split(":" if sys.platform != "win32" else ";") if piece)
    return [Path(piece).expanduser() for piece in parts if piece.strip()]


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_images_from_paths(paths: list[Path], recursive: bool) -> list[Path]:
    images: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = raw_path.expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {path}")
        candidates: list[Path]
        if is_image_file(path):
            candidates = [path]
        elif path.is_dir():
            iterator = path.rglob("*") if recursive else path.iterdir()
            candidates = [item for item in iterator if is_image_file(item)]
        else:
            continue
        for image_path in sorted(candidates, key=lambda p: str(p)):
            resolved = image_path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                images.append(resolved)
    return images


def apply_image_selection(image_paths: list[Path], args: argparse.Namespace) -> list[Path]:
    start = max(int(args.start_index), 0)
    end = args.end_index
    if end is not None and int(end) < start:
        raise ValueError("--end_index must be >= --start_index.")
    selected = image_paths[start:end]
    if args.random_n is not None:
        n = max(int(args.random_n), 0)
        rng = random.Random(int(args.seed))
        selected = rng.sample(selected, k=min(n, len(selected))) if selected else []
        selected.sort(key=lambda p: str(p))
    if args.num_images is not None:
        selected = selected[: max(int(args.num_images), 0)]
    return selected


def load_caption_map(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Caption file not found: {path}")
    caption_map: dict[str, list[str]] = defaultdict(list)

    def add_row(image_key: object, caption: object) -> None:
        key = str(image_key or "").strip()
        text = str(caption or "").strip()
        if key and text:
            caption_map[key].append(text)

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, list):
                    for item in value:
                        add_row(key, item)
                else:
                    add_row(key, value)
        elif isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict):
                    add_row(row.get("image") or row.get("image_path") or row.get("file_name"), row.get("caption") or row.get("text") or row.get("ref_text"))
    elif path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            add_row(row.get("image") or row.get("image_path") or row.get("file_name"), row.get("caption") or row.get("text") or row.get("ref_text"))
    else:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if "\t" in line:
                key, caption = line.split("\t", 1)
            elif "," in line:
                key, caption = line.split(",", 1)
            else:
                continue
            add_row(key, caption)
    return dict(caption_map)


def caption_candidates_for_image(path: Path) -> list[str]:
    resolved = path.resolve()
    return [
        str(resolved),
        str(path),
        path.name,
        path.stem,
    ]


def captions_for_image(path: Path, args: argparse.Namespace, caption_map: dict[str, list[str]]) -> list[str]:
    if args.caption.strip():
        return [args.caption.strip()]
    for key in caption_candidates_for_image(path):
        captions = [text.strip() for text in caption_map.get(key, []) if text.strip()]
        if captions:
            return captions
    if args.interactive_captions and sys.stdin.isatty():
        caption = input(f"Caption for {path}: ").strip()
        if caption:
            return [caption]
    raise ValueError(
        "RefHuman arbitrary-image inference requires a caption. "
        "Set --caption, provide --caption_file, or run in a TTY with interactive captions."
    )


def prompt_for_format(format_name: str, caption: str = "") -> tuple[str, str, str]:
    schema = FORMAT_TO_SCHEMA[format_name]
    if format_name == "refhuman":
        text = caption.strip()
        if not text:
            raise ValueError("RefHuman prompt requires a non-empty caption.")
        return (
            "REF_POSE",
            schema,
            f'Locate a single person that matches the following description: "{text}".',
        )
    return ("ALL_POSE", schema, ALL_POSE_PROMPT)


def empty_target_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(0, 4, dtype=torch.float32),
        torch.zeros(0, len(UNION_KEYPOINTS), 3, dtype=torch.float32),
        torch.zeros(0, len(UNION_KEYPOINTS), dtype=torch.bool),
    )


def make_arbitrary_records(image_paths: list[Path], args: argparse.Namespace) -> list[PoseRecord]:
    caption_map = load_caption_map(args.caption_file)
    records: list[PoseRecord] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size
        captions = [""]
        if args.format == "refhuman":
            captions = captions_for_image(image_path, args, caption_map)
        for caption_idx, caption in enumerate(captions):
            task, schema, prompt = prompt_for_format(args.format, caption=caption)
            boxes, keypoints, valid = empty_target_tensors()
            image_id = image_path.stem if len(captions) == 1 else f"{image_path.stem}:{caption_idx}"
            records.append(
                PoseRecord(
                    image_path=image_path,
                    width=int(width),
                    height=int(height),
                    boxes_xyxy=boxes,
                    loss_boxes_xyxy=boxes.clone(),
                    loss_areas=torch.zeros(0, dtype=torch.float32),
                    keypoints=keypoints,
                    keypoint_valid=valid,
                    visibility_valid=valid.clone(),
                    box_context_scale=torch.zeros(0, dtype=torch.float32),
                    box_jitter_scale=torch.zeros(0, dtype=torch.float32),
                    box_jitter_shift=torch.zeros(0, dtype=torch.float32),
                    schema=schema,
                    task=task,
                    prompt=prompt,
                    ref_text=caption,
                    ref_target=-1,
                    dataset_name=args.format,
                    image_id=image_id,
                )
            )
    return records


def find_refhuman_root(args: argparse.Namespace) -> Path | None:
    candidates: list[Path] = []
    if args.refhuman_root is not None:
        candidates.append(args.refhuman_root.expanduser())
    for path in list(args.inputs) + list(args.images):
        expanded = path.expanduser()
        if expanded.is_dir():
            candidates.append(expanded)
            if expanded.name == "images":
                candidates.append(expanded.parent)
        elif expanded.exists():
            candidates.append(expanded.parent)
            candidates.append(expanded.parent.parent)
    for candidate in candidates:
        if (candidate / f"RefHuman_{args.split}.json").is_file() and (candidate / "images").is_dir():
            return candidate.resolve()
    return None


def input_is_refhuman_root_selection(args: argparse.Namespace, root: Path) -> bool:
    if args.images or args.image_paths:
        return False
    if not args.inputs:
        return True
    resolved_inputs = {path.expanduser().resolve() for path in args.inputs if path.expanduser().exists()}
    return bool(resolved_inputs) and resolved_inputs.issubset({root, root / "images"})


def limit_refhuman_records_per_image(records: list[PoseRecord], limit: int) -> list[PoseRecord]:
    if limit <= 0:
        return records
    counts: dict[Path, int] = defaultdict(int)
    selected: list[PoseRecord] = []
    for record in records:
        key = record.image_path.resolve()
        if counts[key] >= limit:
            continue
        counts[key] += 1
        selected.append(record)
    return selected


def make_refhuman_annotation_records(args: argparse.Namespace, selected_images: list[Path]) -> list[PoseRecord] | None:
    if args.format != "refhuman":
        return None
    root = find_refhuman_root(args)
    if root is None:
        return None
    records = load_refhuman_records(root, split=args.split, max_samples=None)
    grouped: dict[Path, list[PoseRecord]] = defaultdict(list)
    for record in records:
        grouped[record.image_path.resolve()].append(record)
    if input_is_refhuman_root_selection(args, root):
        image_paths = sorted(grouped.keys(), key=lambda p: str(p))
        selected_image_paths = apply_image_selection(image_paths, args)
    else:
        selected_set = {path.resolve() for path in selected_images}
        selected_image_paths = [path for path in sorted(grouped.keys(), key=lambda p: str(p)) if path in selected_set]
        selected_image_paths = apply_image_selection(selected_image_paths, args)
    selected_records: list[PoseRecord] = []
    for path in selected_image_paths:
        selected_records.extend(grouped[path])
    selected_records = limit_refhuman_records_per_image(selected_records, int(args.refhuman_max_captions_per_image))
    if not selected_records:
        raise ValueError(f"No RefHuman {args.split} annotation records matched the selected inputs under {root}.")
    return selected_records


def build_records(args: argparse.Namespace) -> list[PoseRecord]:
    explicit_paths = list(args.inputs) + list(args.images) + split_image_paths(args.image_paths)
    selected_images = collect_images_from_paths(explicit_paths, recursive=bool(args.recursive)) if explicit_paths else []
    if args.format == "refhuman":
        annotation_records = make_refhuman_annotation_records(args, selected_images)
        if annotation_records is not None:
            return annotation_records
    if not selected_images:
        raise ValueError("No input images found. Set --input, --image, or --image_paths.")
    selected_images = apply_image_selection(sorted(selected_images, key=lambda p: str(p)), args)
    if not selected_images:
        raise ValueError("Image selection is empty after applying start/end/random/limit options.")
    return make_arbitrary_records(selected_images, args)


def apply_record_parts(records: list[PoseRecord], args: argparse.Namespace) -> list[PoseRecord]:
    total_parts = int(getattr(args, "num_parts", 1) or 1)
    run_part = int(getattr(args, "worker_index", 0) or 0)
    if total_parts <= 1:
        return records
    selected = [record for idx, record in enumerate(records) if idx % total_parts == run_part]
    print(
        f"[Record split] worker_index={run_part}, num_parts={total_parts}, "
        f"selected={len(selected)} / total={len(records)}",
        flush=True,
    )
    if not selected:
        raise ValueError("No records selected for this part. Check --num_parts/--worker_index.")
    return selected


def enrich_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    schema = str(row.get("schema") or "")
    if schema not in SCHEMA_INDICES:
        return row
    indices = [int(v) for v in SCHEMA_INDICES[schema].tolist()]
    names = SCHEMA_KEYPOINTS[schema]
    row["schema_keypoint_names"] = names
    for pred in row.get("predictions", []):
        all_keypoints = pred.get("keypoints", [])
        schema_keypoints = []
        schema_keypoints_flat = []
        for name, union_idx in zip(names, indices):
            if union_idx >= len(all_keypoints):
                x = y = score = 0.0
            else:
                x, y, score = [float(value) for value in all_keypoints[union_idx][:3]]
            schema_keypoints.append({"name": name, "x": x, "y": y, "score": score})
            schema_keypoints_flat.extend([x, y, score])
        pred["schema_keypoints"] = schema_keypoints
        pred["schema_keypoints_flat"] = schema_keypoints_flat
    return row


def export_schema_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    compact_rows: list[dict[str, Any]] = []
    for row in rows:
        compact_preds = []
        for pred in row.get("predictions", []):
            bbox = [float(v) for v in pred.get("bbox_2d", [])]
            bbox_xywh = []
            if len(bbox) == 4:
                bbox_xywh = [bbox[0], bbox[1], max(bbox[2] - bbox[0], 0.0), max(bbox[3] - bbox[1], 0.0)]
            compact_preds.append(
                {
                    "score": float(pred.get("person_score", 0.0)),
                    "ref_score": float(pred.get("ref_score", 0.0)),
                    "bbox_xyxy": bbox,
                    "bbox_xywh": bbox_xywh,
                    "keypoints": pred.get("schema_keypoints", []),
                    "keypoints_flat": pred.get("schema_keypoints_flat", []),
                }
            )
        compact_rows.append(
            {
                "dataset": row.get("dataset"),
                "format": row.get("dataset"),
                "schema": row.get("schema"),
                "schema_keypoint_names": row.get("schema_keypoint_names", []),
                "image_id": row.get("image_id"),
                "image_path": row.get("image_path"),
                "width": row.get("width"),
                "height": row.get("height"),
                "task_id": row.get("task_id"),
                "caption": row.get("caption", ""),
                "prompt": row.get("prompt", ""),
                "predictions": compact_preds,
            }
        )
    output_path.write_text(json.dumps(compact_rows, ensure_ascii=False, indent=2), encoding="utf-8")


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def apply_gpu_selection(args: argparse.Namespace) -> None:
    """Apply --gpu before vLLM/PyTorch initializes CUDA contexts."""
    gpu = str(getattr(args, "gpu", "") or "").strip()
    if not gpu:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    if str(args.device).startswith("cuda"):
        # After CUDA_VISIBLE_DEVICES is narrowed, cuda:0 means the first visible GPU.
        args.device = "cuda"


def locate_generation_prompt_for_record(record: PoseRecord) -> str:
    """Build the same Locate bbox prompt as train_pose.build_locate_generation_prompts."""
    if record.task == "REF_POSE":
        ref_text = str(record.ref_text or "").strip() or "person"
        return f'Locate a single person that matches the following description: "{ref_text}".'
    return "Locate all the instances that match the following description: person."


def vllm_dtype_from_locate_dtype(dtype: str) -> str:
    dtype = str(dtype or "auto").lower()
    if dtype in {"none", "auto"}:
        return "auto"
    return dtype


def resolve_vllm_lora_adapter(args: argparse.Namespace, checkpoint_path: Path) -> Path | None:
    """Resolve PEFT adapter directory for vLLM LoRA, if available.

    Training checkpoints may save adapters as backbone_lora_adapter beside an
    intermediate checkpoint, or eagle_lora_adapter at the final stage output.
    vLLM support for this custom multimodal model is environment-dependent, so
    missing adapters do not fail here; generation can still fall back to the
    transformers path if vLLM cannot run correctly.
    """
    requested = str(getattr(args, "vllm_lora_adapter", "auto") or "auto").strip()
    if requested.lower() in {"", "none", "off", "0", "false"}:
        return None
    if requested.lower() != "auto":
        adapter = Path(requested).expanduser().resolve()
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"vLLM LoRA adapter_config.json not found under: {adapter}")
        return adapter

    bases: list[Path] = []
    payload_dir = checkpoint_path.parent if checkpoint_path.is_file() else checkpoint_path
    bases.extend([payload_dir, payload_dir.parent])
    if payload_dir.name.startswith("checkpoint-") or payload_dir.name.startswith("checkpoint_step_"):
        bases.append(payload_dir.parent)
        bases.append(payload_dir.parent.parent)
    seen: set[Path] = set()
    for base in bases:
        if not base or base in seen:
            continue
        seen.add(base)
        for name in ("backbone_lora_adapter", "eagle_lora_adapter", "locatepose_lora_adapter"):
            candidate = base / name
            if (candidate / "adapter_config.json").is_file():
                return candidate.resolve()
    return None


def prepare_vllm_lora_adapter(args: argparse.Namespace, adapter: Path) -> Path:
    """Split mixed LocateAnything LoRA for vLLM.

    vLLM 0.11 can apply LoRA only to the language model of a multimodal model.
    LocatePose stage-1 adapters also contain MoonViT vision fc1 LoRA weights, so
    we pass a filtered language-only adapter to vLLM and let the custom model
    merge the vision LoRA from the original adapter during base weight loading.
    """
    tensors_path = adapter / "adapter_model.safetensors"
    config_path = adapter / "adapter_config.json"
    if not tensors_path.is_file() or not config_path.is_file():
        return adapter
    from safetensors.torch import load_file, save_file

    tensors = load_file(str(tensors_path), device="cpu")
    has_vision_lora = any("vision_model." in key for key in tensors)
    if not has_vision_lora:
        os.environ.pop(VISION_LORA_ENV, None)
        return adapter

    language_tensors = {
        key: value
        for key, value in tensors.items()
        if "vision_model." not in key
    }
    if not language_tensors:
        raise ValueError(f"LoRA adapter has no language weights for vLLM: {adapter}")

    os.environ[VISION_LORA_ENV] = str(adapter)
    out_dir = Path(args.output_dir).expanduser() / "vllm_lora" / "language_only_adapter"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(language_tensors, str(out_dir / "adapter_model.safetensors"))

    config = json.loads(config_path.read_text(encoding="utf-8"))
    target_modules = config.get("target_modules")
    if isinstance(target_modules, list):
        config["target_modules"] = [item for item in target_modules if item != "fc1"]
    for pattern_key in ("rank_pattern", "alpha_pattern"):
        pattern = config.get(pattern_key)
        if isinstance(pattern, dict):
            config[pattern_key] = {
                key: value
                for key, value in pattern.items()
                if "vision_model." not in key
            }
    (out_dir / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "[vLLM] split mixed LocatePose LoRA: "
        f"language adapter={out_dir}, vision adapter={adapter}",
        flush=True,
    )
    return out_dir.resolve()


class VLLMLocateBBoxGenerator:
    """vLLM LocateAnything bbox generator.

    LocateAnything image-text bbox generation runs inside the project custom
    vLLM multimodal model. The same vLLM model instance also owns the PoseHead
    and reuses the captured prefill features to predict poses batch-by-batch.
    """

    def __init__(self, args: argparse.Namespace, checkpoint_path: Path) -> None:
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        os.environ["QWENPOSE_VLLM_FEATURE_SIZE"] = str(int(getattr(args, "eagle_feature_size", 64)))
        os.environ["QWENPOSE_VLLM_FEATURE_REFINER_LAYERS"] = str(int(getattr(args, "eagle_feature_refiner_layers", 2)))
        os.environ["QWENPOSE_VLLM_FEATURE_REFINER_BOTTLENECK_DIM"] = str(int(getattr(args, "eagle_feature_refiner_bottleneck_dim", 256)))
        os.environ["QWENPOSE_VLLM_FEATURE_REFINER_INIT_SCALE"] = str(float(getattr(args, "eagle_feature_refiner_init_scale", 0.1)))
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        from qwenpose.vllm_locateanything import enable_locateanything_vllm_transformers_backend

        self.args = args
        patch_info = enable_locateanything_vllm_transformers_backend(args.eagle_model_path)
        if patch_info.get("registered"):
            print(f"[vLLM] LocateAnything backend patch enabled: {patch_info}", flush=True)
        self.processor = AutoProcessor.from_pretrained(args.eagle_model_path, trust_remote_code=True)
        if str(args.vllm_model_impl) == "transformers":
            print(
                "[vLLM] LocateAnything uses qwenpose custom vLLM model; "
                "--vllm_model_impl=transformers is overridden to auto.",
                flush=True,
            )
            args.vllm_model_impl = "auto"
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max(1, int(args.locate_box_max_new_tokens)),
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
        )
        self.lora_request = None
        lora_adapter = resolve_vllm_lora_adapter(args, checkpoint_path)
        if lora_adapter is not None:
            lora_adapter = prepare_vllm_lora_adapter(args, lora_adapter)
        else:
            os.environ.pop(VISION_LORA_ENV, None)
        llm_kwargs: dict[str, Any] = {
            "model": args.eagle_model_path,
            "trust_remote_code": bool(args.vllm_trust_remote_code),
            "dtype": vllm_dtype_from_locate_dtype(args.eagle_dtype),
            "tensor_parallel_size": max(int(args.vllm_tensor_parallel_size), 1),
            "gpu_memory_utilization": float(args.vllm_gpu_memory_utilization),
            "cpu_offload_gb": float(getattr(args, "vllm_cpu_offload_gb", 0.0) or 0.0),
            "enforce_eager": bool(getattr(args, "vllm_enforce_eager", False)),
            "limit_mm_per_prompt": {"image": 1},
            "model_impl": str(args.vllm_model_impl),
            "enable_chunked_prefill": False,
        }
        if getattr(args, "eagle_image_token_limit", None) is not None:
            llm_kwargs["mm_processor_kwargs"] = {
                "locate_image_token_limit": int(args.eagle_image_token_limit)
            }
        if int(getattr(args, "vllm_max_num_seqs", 0) or 0) > 0:
            llm_kwargs["max_num_seqs"] = int(args.vllm_max_num_seqs)
        if int(getattr(args, "vllm_max_num_batched_tokens", 0) or 0) > 0:
            llm_kwargs["max_num_batched_tokens"] = int(args.vllm_max_num_batched_tokens)
        if int(args.vllm_max_model_len) > 0:
            llm_kwargs["max_model_len"] = int(args.vllm_max_model_len)
        if lora_adapter is not None:
            from vllm.lora.request import LoRARequest

            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = max(int(args.vllm_max_lora_rank), 1)
            self.lora_request = LoRARequest("locatepose", 1, str(lora_adapter))
            print(f"[vLLM] using LoRA adapter: {lora_adapter}", flush=True)
        else:
            print("[vLLM] no LoRA adapter found; using base LocateAnything for bbox generation.", flush=True)
        self.llm = LLM(**llm_kwargs)
        self.vllm_model = self._resolve_qwenpose_vllm_model()
        self.vllm_model.load_qwenpose_checkpoint(str(checkpoint_path))
        self.vllm_model.set_qwenpose_feature_capture(True)

    def _resolve_qwenpose_vllm_model(self):
        candidates: list[Any] = []
        engine = getattr(self.llm, "llm_engine", None)
        executor = getattr(engine, "model_executor", None)
        worker = getattr(executor, "driver_worker", None)
        if worker is not None:
            try:
                candidates.append(worker.apply_model(lambda model: model))
            except Exception:
                pass
            for path in (
                ("worker", "model_runner", "model"),
                ("worker", "model_runner"),
                ("model_runner", "model"),
                ("model_runner",),
            ):
                obj = worker
                for attr in path:
                    obj = getattr(obj, attr, None)
                    if obj is None:
                        break
                if obj is not None:
                    candidates.append(obj)
        for obj in candidates:
            resolved = self._unwrap_qwenpose_vllm_model(obj)
            if resolved is not None:
                return resolved
        raise RuntimeError("Unable to resolve qwenpose LocateAnything custom model from vLLM worker.")

    @staticmethod
    def _unwrap_qwenpose_vllm_model(obj: Any):
        seen: set[int] = set()
        stack = [obj]
        while stack:
            current = stack.pop()
            if current is None or id(current) in seen:
                continue
            seen.add(id(current))
            if hasattr(current, "run_qwenpose_pose") and hasattr(current, "pop_qwenpose_feature_cache"):
                return current
            if hasattr(current, "get_model"):
                try:
                    stack.append(current.get_model())
                except Exception:
                    pass
            for attr in ("module", "model", "base_model", "_model"):
                child = getattr(current, attr, None)
                if child is not None:
                    stack.append(child)
        return None

    def _format_prompt(self, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"<image>\n{prompt}"

    def generate(self, records: list[PoseRecord]) -> list[str]:
        requests = []
        for record in records:
            with Image.open(record.image_path) as image:
                pil_image = image.convert("RGB").copy()
            requests.append(
                {
                    "prompt": self._format_prompt(locate_generation_prompt_for_record(record)),
                    "multi_modal_data": {"image": pil_image},
                }
            )
        outputs = self.llm.generate(
            requests,
            self.sampling_params,
            lora_request=self.lora_request,
            use_tqdm=not bool(getattr(self.args, "disable_progress", False)),
        )
        responses: list[str] = []
        for output in outputs:
            if getattr(output, "outputs", None):
                responses.append(str(output.outputs[0].text).strip())
            else:
                responses.append("")
        return responses

    def generate_with_features(self, records: list[PoseRecord]) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        self.vllm_model.reset_qwenpose_feature_cache()
        responses = self.generate(records)
        feature_map, text_embed = self.vllm_model.pop_qwenpose_feature_cache()
        if int(feature_map.shape[0]) != len(records) or int(text_embed.shape[0]) != len(records):
            raise RuntimeError(
                "vLLM qwenpose feature cache batch mismatch: "
                f"features={int(feature_map.shape[0])}, texts={int(text_embed.shape[0])}, records={len(records)}."
            )
        return responses, feature_map, text_embed

    @torch.inference_mode()
    def run_pose(
        self,
        batch: dict,
        target_boxes: torch.Tensor,
        target_box_mask: torch.Tensor,
        feature_map: torch.Tensor,
        text_embed: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.vllm_model.run_qwenpose_pose(
            schema_ids=batch["schema_ids"],
            task_ids=batch["task_ids"],
            target_boxes=target_boxes,
            target_box_mask=target_box_mask,
            images=batch.get("images"),
            external_feature_map=feature_map,
            external_text_embed=text_embed,
        )

    def close(self) -> None:
        llm = getattr(self, "llm", None)
        if llm is not None:
            try:
                engine = getattr(llm, "llm_engine", None)
                engine_core = getattr(engine, "engine_core", None)
                if engine_core is not None and hasattr(engine_core, "shutdown"):
                    engine_core.shutdown()
            except Exception as exc:
                print(f"[vLLM] shutdown warning: {type(exc).__name__}: {exc}", flush=True)
            del self.llm
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def precompute_vllm_locate_responses(
    records: list[PoseRecord],
    args: argparse.Namespace,
    checkpoint_path: Path,
) -> list[str]:
    generator = VLLMLocateBBoxGenerator(args, checkpoint_path)
    responses: list[str] = []
    progress_bar = None
    try:
        total_batches = (len(records) + max(int(args.vllm_batch_size), 1) - 1) // max(int(args.vllm_batch_size), 1)
        if not args.disable_progress and tqdm is not None:
            progress_bar = tqdm(total=total_batches, desc="vLLM locate boxes", unit="batch", dynamic_ncols=True)
        batch_size = max(int(args.vllm_batch_size), 1)
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            responses.extend(generator.generate(chunk))
            if progress_bar is not None:
                progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()
        generator.close()
    return responses


def validate_vllm_locate_responses(records: list[PoseRecord], responses: list[str]) -> None:
    if not responses:
        raise RuntimeError("vLLM LocateAnything returned no responses.")
    parsed_counts: list[int] = []
    malformed: list[str] = []
    for response in responses:
        boxes = parse_locate_bbox_response(response, max_instances=100)
        parsed_counts.append(int(boxes.shape[0]))
        if "<box" not in response and "</box>" not in response:
            malformed.append(response[:120])
    if sum(parsed_counts) == 0 and malformed:
        example = malformed[0]
        raise RuntimeError(
            "vLLM LocateAnything generated no parseable boxes. "
            "LocateAnything requires its custom MTP/box decoding loop; "
            f"standard vLLM token generation returned e.g. {example!r}."
        )


def target_boxes_to_abs_lists(
    target_boxes: torch.Tensor,
    target_box_mask: torch.Tensor,
    batch: dict,
) -> list[list[list[float]]]:
    boxes_cpu = target_boxes.detach().cpu()
    mask_cpu = target_box_mask.detach().cpu().bool()
    locate_boxes_abs: list[list[list[float]]] = []
    for sample_idx, target in enumerate(batch["targets"]):
        width = float(target["width"])
        height = float(target["height"])
        sample_boxes: list[list[float]] = []
        valid_indices = torch.nonzero(mask_cpu[sample_idx], as_tuple=False).flatten().tolist()
        for box_idx in valid_indices:
            x1, y1, x2, y2 = boxes_cpu[sample_idx, int(box_idx)].tolist()
            sample_boxes.append([x1 * width, y1 * height, x2 * width, y2 * height])
        locate_boxes_abs.append(sample_boxes)
    return locate_boxes_abs


def condition_inference_from_locate_responses(
    responses: list[str],
    batch: dict,
    config: LocatePoseUnifiedConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[list[list[float]]]]:
    selected_boxes: list[torch.Tensor] = []
    raw_locate_boxes_abs: list[list[list[float]]] = []
    for sample_idx, response in enumerate(responses):
        task_id = int(batch["task_ids"][sample_idx].detach().cpu().item())
        boxes = parse_locate_boxes_for_task(
            response,
            task_id=task_id,
            max_instances=config.max_instances,
            nms_iou_thresh=config.box_nms_iou_thresh,
            disable_pre_pose_nms=config.disable_pre_pose_nms,
        )
        target = batch["targets"][sample_idx]
        width = float(target["width"])
        height = float(target["height"])
        raw_locate_boxes_abs.append(
            [
                [
                    float(box[0]) * width,
                    float(box[1]) * height,
                    float(box[2]) * width,
                    float(box[3]) * height,
                ]
                for box in boxes.detach().cpu().tolist()
            ]
        )
        context_scale = _context_scale_for_indices(target, [], int(boxes.shape[0]))
        selected_boxes.append(expand_boxes_xyxy_per_box(boxes, context_scale))

    max_boxes = max([int(boxes.shape[0]) for boxes in selected_boxes] + [1])
    box_tensor = torch.zeros(len(selected_boxes), max_boxes, 4, dtype=torch.float32, device=device)
    box_mask = torch.zeros(len(selected_boxes), max_boxes, dtype=torch.bool, device=device)
    for sample_idx, boxes in enumerate(selected_boxes):
        n = int(boxes.shape[0])
        if n <= 0:
            continue
        box_tensor[sample_idx, :n] = boxes.to(device=device, dtype=torch.float32)
        box_mask[sample_idx, :n] = True
    return box_tensor, box_mask, raw_locate_boxes_abs


def maybe_precompute_locate_responses(
    records: list[PoseRecord],
    args: argparse.Namespace,
    checkpoint_path: Path,
) -> list[str] | None:
    if args.box_source == "gt":
        print("[Locate generation] skipped because BOX_SOURCE=gt.", flush=True)
        return None
    backend = str(args.locate_generation_backend or "transformers").lower()
    if backend == "vllm" and not bool(args.disable_single_pass_features):
        print(
            "[Locate generation] backend=vllm uses the qwenpose custom vLLM LocateAnything runner for boxes; "
            "PoseHead consumes those boxes in the synchronized PyTorch batch.",
            flush=True,
        )
    if backend == "auto" and not bool(args.disable_single_pass_features):
        print("[Locate generation] backend=auto -> transformers, because single-pass feature reuse is enabled.", flush=True)
        return None
    if backend == "transformers":
        print("[Locate generation] backend=transformers", flush=True)
        print("[Locate generation] single process uses one CUDA device.", flush=True)
        return None
    try:
        print("[Locate generation] backend=vllm", flush=True)
        responses = precompute_vllm_locate_responses(records, args, checkpoint_path)
        validate_vllm_locate_responses(records, responses)
        return responses
    except Exception as exc:
        if args.disable_vllm_fallback or backend == "vllm" and args.disable_vllm_fallback:
            raise
        print(
            "[Locate generation] vLLM failed; falling back to transformers. "
            f"Reason: {type(exc).__name__}: {exc}",
            flush=True,
        )
        print("[Locate generation] fallback is single-process single-CUDA-device.", flush=True)
        return None


def main() -> None:
    args = parse_args()
    if args.backbone == "locatepose":
        args.backbone = "eagle"
    if args.max_instances <= 0:
        raise ValueError("--max_instances must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if int(getattr(args, "num_parts", 1)) <= 0:
        raise ValueError("--num_parts must be positive.")
    wi = int(getattr(args, "worker_index", 0))
    if wi < 0 or wi >= int(args.num_parts):
        raise ValueError("--worker_index is out of range.")
    if not 0.0 <= args.box_nms_iou_thresh <= 1.0:
        raise ValueError("--box_nms_iou_thresh must be in [0, 1].")
    if not 0.0 <= args.post_pose_nms_iou_thresh <= 1.0:
        raise ValueError("--post_pose_nms_iou_thresh must be in [0, 1].")
    if args.locate_box_max_new_tokens <= 0:
        raise ValueError("--locate_box_max_new_tokens must be positive.")
    if args.vllm_tensor_parallel_size <= 0:
        raise ValueError("--vllm_tensor_parallel_size must be positive.")
    if not 0.0 < args.vllm_gpu_memory_utilization <= 1.0:
        raise ValueError("--vllm_gpu_memory_utilization must be in (0, 1].")
    if args.vllm_batch_size <= 0:
        args.vllm_batch_size = int(args.batch_size)
    if args.vllm_max_num_seqs <= 0:
        args.vllm_max_num_seqs = int(args.vllm_batch_size)
    if args.vllm_max_num_batched_tokens <= 0:
        args.vllm_max_num_batched_tokens = int(args.vllm_max_model_len) if int(args.vllm_max_model_len) > 0 else 2048
    if args.vllm_max_lora_rank <= 0:
        raise ValueError("--vllm_max_lora_rank must be positive.")
    if args.vllm_max_num_seqs <= 0:
        raise ValueError("--vllm_max_num_seqs must be positive after auto sync.")
    if args.vllm_max_num_batched_tokens <= 0:
        raise ValueError("--vllm_max_num_batched_tokens must be positive after auto sync.")
    if args.eagle_min_pixels is not None and args.eagle_min_pixels <= 0:
        raise ValueError("--locate_min_pixels must be positive when set.")
    if args.eagle_max_pixels is not None and args.eagle_max_pixels <= 0:
        raise ValueError("--locate_max_pixels must be positive when set.")
    if args.eagle_image_token_limit is not None and args.eagle_image_token_limit <= 0:
        raise ValueError("--locate_image_token_limit must be positive when set.")
    if (
        args.eagle_min_pixels is not None
        and args.eagle_max_pixels is not None
        and args.eagle_max_pixels < args.eagle_min_pixels
    ):
        raise ValueError("--locate_max_pixels must be >= --locate_min_pixels.")

    apply_gpu_selection(args)
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = apply_record_parts(build_records(args), args)
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    backend = str(args.locate_generation_backend or "transformers").lower()
    use_vllm_integrated = args.box_source == "locate_generate" and backend == "vllm"
    precomputed_locate_responses = None if use_vllm_integrated else maybe_precompute_locate_responses(records, args, checkpoint_path)
    dataset = PoseRecordDataset(records, max_instances=args.max_instances, load_image_tensors=True)
    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": pose_collate,
        "drop_last": False,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)

    device = torch.device(args.device)
    vllm_generator: VLLMLocateBBoxGenerator | None = None
    model = None
    backbone_processor = None
    if use_vllm_integrated:
        print("[Locate generation] backend=vllm integrated: LocateAnything+PoseHead in the custom vLLM model.", flush=True)
        try:
            vllm_generator = VLLMLocateBBoxGenerator(args, checkpoint_path)
        except Exception as exc:
            if args.disable_vllm_fallback:
                raise
            print(
                "[Locate generation] integrated vLLM failed during initialization; falling back to transformers. "
                f"Reason: {type(exc).__name__}: {exc}",
                flush=True,
            )
            use_vllm_integrated = False
            precomputed_locate_responses = None
    if not use_vllm_integrated:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model, backbone_processor = load_eval_model(args, checkpoint, device)
        if backbone_processor is None:
            raise ValueError("LocatePose inference requires a LocateAnything processor.")
    use_single_pass_features = (
        args.box_source == "locate_generate"
        and not use_vllm_integrated
        and precomputed_locate_responses is None
        and not bool(args.disable_single_pass_features)
        and str(args.locate_generation_backend or "transformers").lower() in {"transformers", "auto"}
    )
    if use_single_pass_features:
        print(
            f"[Locate generation] transformers single-pass features enabled; prompt={args.single_pass_prompt}",
            flush=True,
        )
    unified_config = LocatePoseUnifiedConfig.from_args(
        args,
        use_single_pass_features=use_single_pass_features,
    )
    unified_runtime = None
    if not use_vllm_integrated:
        unified_runtime = LocatePoseUnifiedRuntime(
            model,
            backbone_processor,
            device,
            backbone_name=args.backbone,
        )

    predictions_jsonl = args.output_dir / "predictions.jsonl"
    predictions_json = args.output_dir / "predictions.json"
    schema_json = args.output_dir / f"predictions_{args.format}.json"
    manifest_json = args.output_dir / "manifest.json"
    summary_json = args.output_dir / "summary.json"
    visualization_dir = args.output_dir / "visualizations"
    save_visualizations = args.visualize_max_samples != 0
    if save_visualizations:
        visualization_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    response_cursor = 0
    visualized = 0
    samples = 0
    batches = 0
    actual_single_pass_features_used = False
    vllm_integrated_features_used = False
    max_visualizations = len(records) if args.visualize_max_samples < 0 else int(args.visualize_max_samples)

    try:
        with predictions_jsonl.open("w", encoding="utf-8") as writer:
            progress_bar = None
            if not args.disable_progress and tqdm is not None:
                progress_bar = tqdm(total=len(loader), desc="infer locatepose", unit="batch", dynamic_ncols=True)
            with torch.inference_mode():
                for batch in loader:
                    batch = move_batch_to_device(batch, device)
                    responses_for_batch = None
                    if precomputed_locate_responses is not None:
                        batch_size_now = len(batch["image_paths"])
                        responses_for_batch = precomputed_locate_responses[response_cursor : response_cursor + batch_size_now]
                        response_cursor += batch_size_now
                    if args.box_source == "gt":
                        if unified_runtime is None:
                            raise RuntimeError("BOX_SOURCE=gt is not supported by the integrated vLLM path.")
                        target_boxes, target_box_mask, _ = prepare_box_conditioning(
                            batch["targets"],
                            batch["task_ids"],
                            device,
                            max_instances=args.max_instances,
                        )
                        outputs, _ = unified_runtime.forward_pose(
                            batch,
                            target_boxes,
                            target_box_mask,
                            unified_config,
                        )
                        responses = ["<gt_box_conditioning>"] * len(batch["image_paths"])
                        locate_boxes_abs = target_boxes_to_abs_lists(target_boxes, target_box_mask, batch)
                    elif use_vllm_integrated:
                        if vllm_generator is None:
                            raise RuntimeError("Integrated vLLM generator is not initialized.")
                        batch_size_now = len(batch["image_paths"])
                        records_for_batch = records[response_cursor : response_cursor + batch_size_now]
                        response_cursor += batch_size_now
                        responses, feature_map, text_embed = vllm_generator.generate_with_features(records_for_batch)
                        validate_vllm_locate_responses(records_for_batch, responses)
                        target_boxes, target_box_mask, locate_boxes_abs = condition_inference_from_locate_responses(
                            responses,
                            batch,
                            unified_config,
                            device,
                        )
                        outputs = vllm_generator.run_pose(
                            batch,
                            target_boxes,
                            target_box_mask,
                            feature_map,
                            text_embed,
                        )
                        vllm_integrated_features_used = True
                    else:
                        if unified_runtime is None:
                            raise RuntimeError("LocatePose runtime is not initialized.")
                        unified_result = unified_runtime.infer_batch(
                            batch,
                            unified_config,
                            precomputed_locate_responses=responses_for_batch,
                        )
                        outputs = unified_result.outputs
                        responses = unified_result.locate_responses or []
                        locate_boxes_abs = unified_result.locate_boxes_abs or []
                        actual_single_pass_features_used = (
                            actual_single_pass_features_used or unified_result.used_single_pass_features
                        )
                    rows = tensor_to_prediction_rows(
                        outputs,
                        batch,
                        args,
                        raw_boxes_abs=locate_boxes_abs,
                    )
                    for local_idx, row in enumerate(rows):
                        row["caption"] = batch.get("ref_texts", [""] * len(rows))[local_idx]
                        row["locate_response"] = responses[local_idx] if local_idx < len(responses) else ""
                        row["locate_boxes_2d"] = locate_boxes_abs[local_idx] if local_idx < len(locate_boxes_abs) else []
                        row = enrich_prediction_row(row)
                        writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                        all_rows.append(row)
                    if save_visualizations and visualized < max_visualizations:
                        batch_size = len(batch["schema_ids"])
                        for local_idx in range(batch_size):
                            if visualized >= max_visualizations:
                                break
                            source_name = Path(batch["image_paths"][local_idx]).stem
                            safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in source_name)[:80]
                            vis_path = visualization_dir / f"infer_{samples + local_idx:06d}_{safe_name}.jpg"
                            save_pose_visualization(
                                outputs,
                                batch,
                                vis_path,
                                sample_idx=local_idx,
                                max_instances=args.visualize_max_instances,
                                score_threshold=args.score_threshold,
                            )
                            visualized += 1
                    samples += len(rows)
                    batches += 1
                    if progress_bar is not None:
                        progress_bar.set_postfix(samples=samples, visualized=visualized, refresh=False)
                        progress_bar.update(1)
            if progress_bar is not None:
                progress_bar.close()
    finally:
        if vllm_generator is not None:
            vllm_generator.close()

    predictions_json.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    export_schema_results(all_rows, schema_json)
    manifest = [
        {
            "image_path": str(record.image_path),
            "image_id": record.image_id,
            "dataset": record.dataset_name,
            "schema": record.schema,
            "task": record.task,
            "caption": record.ref_text,
            "prompt": record.prompt,
            "width": record.width,
            "height": record.height,
        }
        for record in records
    ]
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "checkpoint": str(checkpoint_path),
        "box_source": args.box_source,
        "locate_generation_backend": args.locate_generation_backend,
        "vllm_used_for_locate_boxes": precomputed_locate_responses is not None or use_vllm_integrated,
        "vllm_integrated_features_used": vllm_integrated_features_used,
        "single_pass_features_used": actual_single_pass_features_used,
        "single_pass_prompt": args.single_pass_prompt,
        "gpu": str(args.gpu or os.environ.get("CUDA_VISIBLE_DEVICES", "")),
        "format": args.format,
        "schema": FORMAT_TO_SCHEMA[args.format],
        "split": args.split if args.format == "refhuman" else None,
        "samples": samples,
        "batches": batches,
        "num_parts": int(args.num_parts),
        "worker_index": int(args.worker_index),
        "output_dir": str(args.output_dir),
        "predictions_jsonl": str(predictions_jsonl),
        "predictions_json": str(predictions_json),
        "schema_predictions_json": str(schema_json),
        "manifest_json": str(manifest_json),
        "visualizations_dir": str(visualization_dir) if save_visualizations else None,
        "visualized_samples": visualized,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
