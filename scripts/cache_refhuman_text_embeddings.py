#!/usr/bin/env python3
"""Cache frozen LocateAnything prompt tokens for vision-only Stage-1 training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwenpose.data import (  # noqa: E402
    ALL_POSE_PROMPT,
    REFHUMAN_TEXT_CACHE_VERSION,
    build_refhuman_locate_prompt,
    locate_prompt_embedding_key,
    normalize_refhuman_text,
)
from qwenpose.eagle_lora import get_eagle_base_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache frozen LocateAnything prompt tokens and pooled embeddings."
    )
    parser.add_argument("--model_path", type=Path, default=Path("weights/LocateAnything-3B"))
    parser.add_argument("--refhuman_root", type=Path, default=Path("datasets/refhuman"))
    parser.add_argument("--splits", default="train,val")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/qwenpose_text/locateanything_prompt_tokens.pt"),
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=160)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument(
        "--save_dtype", choices=("float32", "float16", "bfloat16"), default="float16"
    )
    parser.add_argument(
        "--max_instances_per_split",
        "--max_records_per_split",
        dest="max_instances_per_split",
        type=int,
        default=0,
        help=(
            "Debug limit on RefHuman person instances per split. Every caption for "
            "each retained instance is cached so epoch-wise caption rotation remains valid."
        ),
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def collect_records(root: Path, splits: list[str], limit: int):
    captions: dict[str, str] = {}
    prompts: dict[str, str] = {}
    split_keys: dict[str, list[str]] = {}
    for split in splits:
        annotation_path = root / f"RefHuman_{split}.json"
        print(
            f"Loading RefHuman {split} annotations from {annotation_path} ...",
            flush=True,
        )
        with annotation_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        images = payload.get("images", [])
        annotations = payload.get("annotations", [])
        row_count = min(len(images), len(annotations))
        selected_instances: set[tuple[object, ...]] = set()
        keys: list[str] = []
        rows = zip(images, annotations)
        for image, annotation in tqdm(
            rows,
            total=row_count,
            desc=f"Scan RefHuman {split}",
            unit="row",
            dynamic_ncols=True,
        ):
            instance_key = (
                str(image.get("file_name", "")),
                str(
                    image.get(
                        "original_id",
                        image.get("origin_id", image.get("file_name", "")),
                    )
                ),
                str(
                    annotation.get(
                        "original_id",
                        annotation.get("origin_id", annotation.get("id", "")),
                    )
                ),
                tuple(round(float(value), 2) for value in annotation.get("bbox", [])),
            )
            if limit > 0 and instance_key not in selected_instances:
                if len(selected_instances) >= limit:
                    continue
                selected_instances.add(instance_key)
            caption = normalize_refhuman_text(image.get("caption", ""))
            if not caption:
                continue
            prompt = build_refhuman_locate_prompt(caption)
            key = locate_prompt_embedding_key(prompt)
            captions.setdefault(key, caption)
            prompts.setdefault(key, prompt)
            keys.append(key)
        split_keys[split] = sorted(set(keys))
        del payload, images, annotations
    all_pose_key = locate_prompt_embedding_key(ALL_POSE_PROMPT)
    prompts[all_pose_key] = ALL_POSE_PROMPT
    captions[all_pose_key] = "person"
    split_keys["all_pose"] = [all_pose_key]
    if not captions:
        raise RuntimeError("No RefHuman captions were found.")
    return captions, prompts, split_keys


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_length <= 0:
        raise ValueError("--batch_size and --max_length must be positive.")
    splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    captions, prompts, split_keys = collect_records(
        args.refhuman_root.expanduser(), splits, args.max_instances_per_split
    )
    keys = sorted(captions)
    print(f"Collected {len(keys):,} unique LocateAnything prompts.")

    from transformers import AutoConfig, AutoModel, AutoTokenizer

    model_path = str(args.model_path.expanduser())
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config._attn_implementation = "sdpa"
    if hasattr(config, "text_config"):
        config.text_config._attn_implementation = "sdpa"
    model = AutoModel.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=resolve_dtype(args.dtype),
        attn_implementation="sdpa",
    ).to(torch.device(args.device))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, fix_mistral_regex=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = get_eagle_base_model(model)
    decoder = getattr(base.language_model, "model", None)
    if decoder is None:
        raise RuntimeError("LocateAnything language_model has no decoder core.")
    image_token_index = int(base.image_token_index)
    device = torch.device(args.device)
    save_dtype = resolve_dtype(args.save_dtype)
    pooled_embeddings: dict[str, torch.Tensor] = {}
    token_embeddings: dict[str, torch.Tensor] = {}

    def prompt_only_chat_text(prompt: str) -> str:
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": str(prompt)}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return str(prompt)

    with tqdm(
        total=len(keys),
        desc="Cache Locate prompts",
        unit="prompt",
        dynamic_ncols=True,
    ) as progress, torch.inference_mode():
        for start in range(0, len(keys), args.batch_size):
            batch_keys = keys[start : start + args.batch_size]
            encoded_texts = [prompt_only_chat_text(prompts[key]) for key in batch_keys]
            encoded = tokenizer(
                encoded_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            hidden = decoder(
                input_ids=input_ids,
                visual_features=None,
                image_token_index=image_token_index,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state
            mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pooled = pooled.detach().to(device="cpu", dtype=save_dtype)
            hidden_cpu = hidden.detach().to(device="cpu", dtype=save_dtype)
            mask_cpu = attention_mask.detach().to(device="cpu", dtype=torch.bool)
            for row, key in enumerate(batch_keys):
                pooled_embeddings[key] = pooled[row].contiguous()
                token_embeddings[key] = hidden_cpu[row, mask_cpu[row]].contiguous()
            progress.update(len(batch_keys))

    hidden_dim = int(next(iter(pooled_embeddings.values())).numel())
    token_count = sum(int(value.shape[0]) for value in token_embeddings.values())
    payload = {
        "version": REFHUMAN_TEXT_CACHE_VERSION,
        "model_path": str(args.model_path),
        "hidden_dim": hidden_dim,
        "dtype": str(save_dtype).replace("torch.", ""),
        "splits": splits,
        "split_keys": split_keys,
        "captions": captions,
        "prompts": prompts,
        "cache_kind": "prompt_only_last_hidden_state",
        "pooling": "attention_mask_mean",
        "max_length": int(args.max_length),
        "chat_template": str(getattr(tokenizer, "chat_template", "") or ""),
        "pooled_embeddings": pooled_embeddings,
        "token_embeddings": token_embeddings,
    }
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    print(
        f"Saved {len(pooled_embeddings):,} prompts, {token_count:,} tokens, hidden_dim={hidden_dim}, "
        f"to {output} ({output.stat().st_size / 1024**2:.1f} MB)."
    )


if __name__ == "__main__":
    main()
