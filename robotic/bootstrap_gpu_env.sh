#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
TORCH_CUDA="${TORCH_CUDA:-cu121}"
if [[ "$TORCH_CUDA" != "cu121" && "$TORCH_CUDA" != "cu118" ]]; then
  echo "TORCH_CUDA must be cu121 or cu118" >&2
  exit 2
fi

if command -v python3.10 >/dev/null 2>&1; then
  python3.10 -m venv --clear .venv_gpu
else
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
  else
    python3 -m venv .uv_bootstrap
    .uv_bootstrap/bin/python -m pip install uv
    UV_BIN="$PWD/.uv_bootstrap/bin/uv"
  fi
  "$UV_BIN" venv --seed --clear --python 3.10 .venv_gpu
fi

.venv_gpu/bin/python -m pip install --upgrade pip
.venv_gpu/bin/python -m pip install 'torch==2.5.1' \
  --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
PYTHON_BIN="$PWD/.venv_gpu/bin/python" ./setup_gpu_env.sh
