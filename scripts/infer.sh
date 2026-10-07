#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-outputs/qwen3-coco-det-cls}"
MEDIA="${1:?用法: scripts/infer.sh <image_path> [output_json] [instruction]}"
OUTPUT="${2:-outputs/prediction.json}"
INSTRUCTION="${3:-检测图片中的所有目标，并输出类别、置信度和归一化边界框。}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$(dirname "$OUTPUT")"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m src.inference.infer \
  --model "$MODEL" \
  --media "$MEDIA" \
  --media-type image \
  --instruction "$INSTRUCTION" \
  --output "$OUTPUT"

echo "推理结果已保存到: $OUTPUT"
