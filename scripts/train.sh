#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${1:-configs/train.yaml}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p outputs/logs
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m src.training.train \
  --config "$CONFIG"
