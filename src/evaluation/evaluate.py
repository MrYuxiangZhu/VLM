from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data.datasets import build_dataset
from ..inference.engine import VisionPredictor
from .metrics import match_detections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--media-root", default="")
    parser.add_argument("--media-type", choices=["image", "video"], default="video")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output-json", default="", help="Optional metrics JSON output")
    args = parser.parse_args()
    dataset = build_dataset({"type": "jsonl", "annotations": args.data, "media_root": args.media_root, "media_type": args.media_type, "fps": args.fps})
    predictor = VisionPredictor(args.model)
    scores, cls_correct, valid_json, mean_ious = [], 0, 0, []
    for item in dataset:
        target = json.loads(item["messages"][-1]["content"][0]["text"])
        instruction = item["messages"][1]["content"][1]["text"]
        try:
            prediction = predictor.predict(item["media"], instruction, item["media_type"], args.fps)
            valid_json += 1
        except (ValueError, TypeError, json.JSONDecodeError):
            prediction = {"video_class": "", "detections": []}
        detection_metrics = match_detections(prediction, target, args.iou_threshold)
        scores.append(detection_metrics["f1"])
        mean_ious.append(detection_metrics["mean_iou"])
        cls_correct += prediction.get("video_class") == target.get("video_class")
    result = {
        "model": args.model,
        "samples": len(dataset),
        "valid_json": valid_json,
        "json_valid_rate": valid_json / max(len(dataset), 1),
        "iou_threshold": args.iou_threshold,
        "detection_f1": sum(scores) / max(len(scores), 1),
        "mean_iou": sum(mean_ious) / max(len(mean_ious), 1),
        "classification_accuracy": cls_correct / max(len(dataset), 1),
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
