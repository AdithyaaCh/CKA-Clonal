#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA PyTorch is required"; print("CUDA:", torch.cuda.get_device_name(0))'
"$PYTHON_BIN" -m pip install -r requirements-gpu.txt
"$PYTHON_BIN" - <<'PY'
from tasks import get_task, robotic_sequences
sequence = 'robotic-push'
for task_id, name in enumerate(robotic_sequences[sequence]):
    env = get_task(task_id, sequence=sequence)
    env.reset(seed=0)
    env.step(env.action_space.sample())
    env.close()
    print(sequence, name, 'ready')
PY
