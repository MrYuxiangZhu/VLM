#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PYTHON_BIN="${PYTHON_BIN:-/home/users/yuxiang.zhu/miniconda3/envs/agent/bin/python3}"
GPU_ID="${GPU_ID:-1}"
MODE="${MODE:-qlora}"

echo "Starting smoke training mode=$MODE gpu=$GPU_ID"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m src.training.train \
  --config configs/train_smoke.yaml \
  --mode "$MODE" \
  --output-dir outputs/qwen3-coco-smoke
