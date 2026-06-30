from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwenpose.metrics import compute_pose_metrics, load_prediction_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score LocatePose/QwenPose prediction JSON/JSONL by dataset type.")
    parser.add_argument("--predictions", type=Path, required=True, help="predictions.jsonl 或 predictions.json。")
    parser.add_argument("--dataset_root", type=Path, default=Path("datasets"), help="数据集根目录。")
    parser.add_argument("--split", type=str, default="val", help="数据集 split，默认 val。")
    parser.add_argument("--output", type=Path, default=None, help="可选指标 JSON 输出路径。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_prediction_rows(args.predictions)
    metrics = compute_pose_metrics(rows, dataset_root=args.dataset_root, split=args.split)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
