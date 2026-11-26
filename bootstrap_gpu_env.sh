#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

TORCH_CUDA="${TORCH_CUDA:-cu121}"
if [[ "$TORCH_CUDA" != "cu121" && "$TORCH_CUDA" != "cu118" ]]; then
  echo "TORCH_CUDA must be cu121 or cu118" >&2
  exit 2
fi
if [[ -n "${PYTHON_BASE:-}" ]]; then
  "$PYTHON_BASE" -c 'import sys; assert (3, 9) <= sys.version_info[:2] <= (3, 12), "PYTHON_BASE must point to Python 3.9–3.12"'
else
  for candidate in python3.12 python3.11 python3.10 python3.9; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BASE="$candidate"
      break
    fi
  done
fi

if [[ -n "${PYTHON_BASE:-}" ]]; then
  echo "Using $("$PYTHON_BASE" --version)"
  "$PYTHON_BASE" -m venv --clear .venv_gpu
else
  echo "No compatible system Python found; installing a local Python 3.12 with uv"
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
  else
    python3 -m venv .uv_bootstrap
    .uv_bootstrap/bin/python -m pip install uv
    UV_BIN="$PWD/.uv_bootstrap/bin/uv"
  fi
  "$UV_BIN" venv --seed --clear --python 3.12 .venv_gpu
fi

.venv_gpu/bin/python --version
.venv_gpu/bin/python -m pip install --upgrade pip
.venv_gpu/bin/python -m pip install 'torch==2.5.1' \
  --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
PYTHON_BIN="$PWD/.venv_gpu/bin/python" ./setup_gpu_env.sh
