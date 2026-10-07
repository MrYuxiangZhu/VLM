#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-outputs/qwen3-coco-det-cls}"
VIDEO="${1:?用法: scripts/video_infer.sh <video_path> [output_json] [fps] [instruction]}"
OUTPUT="${2:-outputs/video_prediction.json}"
FPS="${3:-2}"
INSTRUCTION="${4:-检测视频中的所有目标，并输出视频级类别、置信度和归一化边界框。}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$(dirname "$OUTPUT")"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m src.inference.infer \
  --model "$MODEL" \
  --media "$VIDEO" \
  --media-type video \
  --fps "$FPS" \
  --instruction "$INSTRUCTION" \
  --output "$OUTPUT"

echo "视频推理结果已保存到: $OUTPUT"
