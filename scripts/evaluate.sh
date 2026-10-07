#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-outputs/qwen3-coco-det-cls}"
DATA="${1:-data/coco_val.jsonl}"
MEDIA_ROOT="${2:-/home/yuxiangzhu/volume/animal_det/data/coco/val2017}"
MEDIA_TYPE="${MEDIA_TYPE:-image}"
FPS="${FPS:-2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m src.evaluation.evaluate \
  --model "$MODEL" \
  --data "$DATA" \
  --media-root "$MEDIA_ROOT" \
  --media-type "$MEDIA_TYPE" \
  --fps "$FPS"
