# Robotic arm GPU comparisons

This folder contains everything needed to run a focused Meta-World v2 comparison on a Linux NVIDIA GPU server:

`push-v2` → `reach-wall-v2` → **`push-wall-v2`**

The two source tasks train once for 100,000 environment steps each. The target is trained three times—random α initialization, argmax, and CLONALG—with the same 100,000-step budget. All three spend 256 of those target steps probing, so the comparison isolates α initialization. Initial, periodic, and final evaluations record episode return and success. This is seed 0 and an exploratory budget, not an official Continual World result.

## Set up and run

```bash
cd robotic
./bootstrap_gpu_env.sh
export PYTHON_BIN="$PWD/.venv_gpu/bin/python"
./start_gpu_runs.sh
./status_gpu_runs.sh
```

Bootstrap makes a separate Python 3.10 environment because this repo's pinned Meta-World and MuJoCo versions need it. It uses a system Python 3.10 if present, otherwise downloads one with `uv`. It installs CUDA PyTorch 2.5.1, Meta-World, MuJoCo, and the remaining Python packages. If CUDA 12.1 is unavailable on the server, use `TORCH_CUDA=cu118 ./bootstrap_gpu_env.sh` instead. The setup script resets and steps all three tasks before training begins.

`./status_gpu_runs.sh` shows the job state. The main log is `logs/robotic_suite.log`, and the comparison writes `agents/robotic_push_seed0_100k/comparison_seed_0.json`. To stop it, run `./stop_gpu_runs.sh`.

The [included seed-0 results](agents/robotic_push_seed0_300k_cpu/comparison_seed_0.json) used 300,000 steps per stage on CPU. To reproduce that profile, run `"$PYTHON_BIN" run_robotic_suite.py --cpu-300k` from a fresh checkout with a Python environment containing the robotics dependencies.
