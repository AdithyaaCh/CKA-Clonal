#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "Install a CUDA-enabled PyTorch in this Python environment first"; print("CUDA:", torch.cuda.get_device_name(0))'
"$PYTHON_BIN" -m pip install -r requirements-gpu.txt
"$PYTHON_BIN" -c 'import sys, AutoROM; sys.argv = ["AutoROM", "--accept-license"]; AutoROM.cli()'
"$PYTHON_BIN" - <<'PY'
import gymnasium as gym
for game, last in (("Freeway", 7), ("SpaceInvaders", 9)):
    env = gym.make(f"ALE/{game}-v5", mode=last)
    env.reset(seed=0)
    env.step(env.action_space.sample())
    env.close()
    print(f"{game}: mode {last} ready")
PY
