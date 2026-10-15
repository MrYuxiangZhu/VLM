from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import VisionPredictor


def predict(model_path: str, media: str, instruction: str, media_type: str = "video", fps: float = 2.0) -> dict:
    return VisionPredictor(model_path).predict(media, instruction, media_type, fps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--media", required=True)
    parser.add_argument("--media-type", choices=["image", "video"], default="video")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--instruction", default="检测所有目标，并输出视频级类别和归一化边界框。")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = predict(args.model, args.media, args.instruction, args.media_type, args.fps)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
