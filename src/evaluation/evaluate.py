from __future__ import annotations

import argparse
import json

from ..data.datasets import build_dataset
from ..inference.infer import predict
from .metrics import detection_f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--media-root", default="")
    parser.add_argument("--media-type", choices=["image", "video"], default="video")
    parser.add_argument("--fps", type=float, default=2.0)
    args = parser.parse_args()
    dataset = build_dataset({"type": "jsonl", "annotations": args.data, "media_root": args.media_root, "media_type": args.media_type, "fps": args.fps})
    scores, cls_correct = [], 0
    for item in dataset:
        target = json.loads(item["messages"][-1]["content"][0]["text"])
        instruction = item["messages"][1]["content"][1]["text"]
        prediction = predict(args.model, item["media"], instruction, item["media_type"], args.fps)
        scores.append(detection_f1(prediction, target))
        cls_correct += prediction.get("video_class") == target.get("video_class")
    result = {"samples": len(dataset), "detection_f1": sum(scores) / max(len(scores), 1), "classification_accuracy": cls_correct / max(len(dataset), 1)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
