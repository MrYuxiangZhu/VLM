#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-${FINETUNE_MODE:-lora}}"
CONFIG="${2:-${CONFIG:-configs/train.yaml}}"
OUTPUT_DIR="${3:-${OUTPUT_DIR:-}}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

case "$MODE" in
  lora|qlora|dora|full) ;;
  *)
    echo "用法: $0 [lora|qlora|dora|full] [config] [output_dir]" >&2
    exit 2
    ;;
esac

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p outputs/logs

ARGS=("-m" "src.training.train" "--config" "$CONFIG" "--mode" "$MODE")
if [[ -n "$OUTPUT_DIR" ]]; then
  ARGS+=("--output-dir" "$OUTPUT_DIR")
fi

if [[ "$MODE" == "qlora" ]]; then
  echo "启动 QLoRA：需要 NVIDIA GPU 和 bitsandbytes。"
fi
if [[ "$MODE" == "full" ]]; then
  echo "启动 Full SFT：显存需求显著高于 LoRA/QLoRA。"
fi

echo "finetune_mode=$MODE config=$CONFIG"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" "${ARGS[@]}"
