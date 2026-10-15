#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASELINE_MODEL="${1:?用法: scripts/report.sh <baseline_model> <finetuned_model> [data] [media_root] [output_html] [media_type] [fps]}"
FINETUNED_MODEL="${2:?用法: scripts/report.sh <baseline_model> <finetuned_model> [data] [media_root] [output_html] [media_type] [fps]}"
DATA="${3:-data/coco_val.jsonl}"
MEDIA_ROOT="${4:-/home/yuxiangzhu/volume/animal_det/data/coco/val2017}"
OUTPUT_HTML="${5:-outputs/reports/finetune_comparison.html}"
MEDIA_TYPE="${6:-image}"
FPS="${7:-2}"
IOU_THRESHOLD="${IOU_THRESHOLD:-0.5}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
BASELINE_JSON="${OUTPUT_HTML%.html}.baseline.json"
FINETUNED_JSON="${OUTPUT_HTML%.html}.finetuned.json"

mkdir -p "$(dirname "$OUTPUT_HTML")"

"$PYTHON_BIN" -m src.evaluation.evaluate \
  --model "$BASELINE_MODEL" \
  --data "$DATA" \
  --media-root "$MEDIA_ROOT" \
  --media-type "$MEDIA_TYPE" \
  --fps "$FPS" \
  --iou-threshold "$IOU_THRESHOLD" \
  --output-json "$BASELINE_JSON"

"$PYTHON_BIN" -m src.evaluation.evaluate \
  --model "$FINETUNED_MODEL" \
  --data "$DATA" \
  --media-root "$MEDIA_ROOT" \
  --media-type "$MEDIA_TYPE" \
  --fps "$FPS" \
  --iou-threshold "$IOU_THRESHOLD" \
  --output-json "$FINETUNED_JSON"

"$PYTHON_BIN" -m src.evaluation.report \
  --before "$BASELINE_JSON" \
  --after "$FINETUNED_JSON" \
  --before-name "Before fine-tuning" \
  --after-name "After fine-tuning" \
  --output "$OUTPUT_HTML"

echo "报告已生成: $OUTPUT_HTML"
