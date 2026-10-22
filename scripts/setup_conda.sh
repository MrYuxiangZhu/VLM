#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${ENV_NAME:-vlm-qwen}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
CONDA_BIN="${CONDA_BIN:-conda}"

if ! command -v "$CONDA_BIN" >/dev/null 2>&1; then
  echo "错误: 未找到 conda，请先安装 Miniconda 或 Anaconda。" >&2
  exit 1
fi

if "$CONDA_BIN" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "复用已有 conda 环境: $ENV_NAME"
else
  echo "创建 conda 环境: $ENV_NAME (Python $PYTHON_VERSION)"
  "$CONDA_BIN" create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
fi

echo "安装 PyTorch: $TORCH_INDEX_URL"
"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python -m pip install --upgrade \
  torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"

echo "安装项目依赖"
"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python -m pip install --upgrade -r requirements.txt

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python -m compileall -q src scripts

echo
echo "环境搭建完成。"
echo "激活环境: conda activate $ENV_NAME"
echo "训练命令: bash scripts/train.sh"
echo "如需 CPU 版 PyTorch，可执行:"
echo "TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu bash scripts/setup_conda.sh"
