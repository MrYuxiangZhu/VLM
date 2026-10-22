#!/usr/bin/env bash
# GPU_ID=0,1,2,3 bash scripts/finetune.sh lora configs/train.yaml outputs/qwen3-coco-lora
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 默认使用本地模型/不联网
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

MODE="${1:-${FINETUNE_MODE:-lora}}"
CONFIG="${2:-${CONFIG:-configs/train.yaml}}"
OUTPUT_DIR="${3:-${OUTPUT_DIR:-outputs/qwen3-${MODE}}}"
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
