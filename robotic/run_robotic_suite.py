"""Run the focused push-wall transfer comparison."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-300k", action="store_true",
                        help="Use one CPU thread and 300,000 steps per stage")
    args = parser.parse_args()
    steps = "300000" if args.cpu_300k else "100000"
    suffix = "300k_cpu" if args.cpu_300k else "100k"
    env = os.environ.copy()
    device_args = []
    if args.cpu_300k:
        env.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1")
    else:
        device_args.append("--require-cuda")

    for sequence in ("robotic-push",):
        tag = sequence.replace("-", "_") + f"_seed0_{suffix}"
        cmd = [sys.executable, str(HERE / "compare_robotic_push.py"),
               "--sequence", sequence, "--seed", "0", *device_args,
               "--source-steps", steps, "--target-steps", steps,
               "--probe-steps", "256", "--eval-every", "25000",
               "--num-evals", "5", "--tag", tag]
        print("START", sequence, flush=True)
        subprocess.run(cmd, cwd=HERE, env=env, check=True)
        print("DONE", sequence, flush=True)


if __name__ == "__main__":
    main()
