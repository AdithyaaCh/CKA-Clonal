"""Start, inspect, and stop detached Atari GPU comparison jobs."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys


HERE = Path(__file__).resolve().parent
LOGS = HERE / "long_runs"
GAMES = ("Freeway", "SpaceInvaders")
FULL_TAG = "gpu100k"
ONE_HOUR_TAG = "gpu1h"
SEQUENCE_TAG = "gpu500k_seq10"
SEQUENCES = {
    "Freeway": (0, 1, 4, 6, 0, 3, 4, 7, 1, 3),
    "SpaceInvaders": (0, 1, 5, 8, 0, 3, 5, 9, 1, 3),
}
ONE_HOUR_STEPS = 18 * 1024 + 256
COMPLETED_VISITS = {"Freeway": 4, "SpaceInvaders": 5}


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def paths(game, tag):
    stem = f"{tag}_seed0_{game}"
    return LOGS / f"{stem}.pid", LOGS / f"{stem}.log"


def start(tag):
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is missing. Run ./bootstrap_gpu_env.sh, then set PYTHON_BIN to .venv_gpu/bin/python") from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU unavailable in this Python environment; no jobs started")
    LOGS.mkdir(exist_ok=True)
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    for game in GAMES:
        pid_file, log_file = paths(game, tag)
        if pid_file.exists() and alive(int(pid_file.read_text())):
            print(f"{game}: already running, PID {pid_file.read_text().strip()}")
            continue
        command = [sys.executable, str(HERE / "run_long_alpha_compare.py"),
                   "--env", f"ALE/{game}-v5", "--seed", "0", "--tag", tag, "--require-cuda"]
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        if tag == ONE_HOUR_TAG:
            command += ["--steps", str(ONE_HOUR_STEPS), "--reuse-tag", FULL_TAG,
                        "--reuse-visits", str(COMPLETED_VISITS[game])]
        elif tag == SEQUENCE_TAG:
            command += ["--steps", "500000", "--include-random", "--complete-step-budget",
                        "--modes", *map(str, SEQUENCES[game])]
            if game == "Freeway":
                command.append("--async-envs")
            env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        else:
            command += ["--steps", "100000"]
        with log_file.open("a") as log:
            process = subprocess.Popen(command, cwd=HERE, stdout=log,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       start_new_session=True,
                                       env=env)
        pid_file.write_text(f"{process.pid}\n")
        print(f"{game}: started PID {process.pid}, log {log_file}")


def status(tag):
    for game in GAMES:
        pid_file, log_file = paths(game, tag)
        if not pid_file.exists():
            print(f"{game}: never started")
            continue
        pid = int(pid_file.read_text())
        print(f"{game}: {'running' if alive(pid) else 'stopped'} PID {pid}; log {log_file}")
        if log_file.exists():
            lines = [line.strip() for line in log_file.read_text(errors="replace").splitlines()
                     if line.startswith(("START ", "DONE ", "FINISHED COMPARISON ", "Traceback", "RuntimeError"))]
            for line in lines[-3:]:
                print("  " + line)


def stop(tag):
    for game in GAMES:
        pid_file, _ = paths(game, tag)
        if not pid_file.exists():
            continue
        pid = int(pid_file.read_text())
        if alive(pid):
            os.killpg(pid, signal.SIGTERM)
            print(f"{game}: sent stop signal to process group {pid}")
        else:
            print(f"{game}: already stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "status", "stop"))
    profiles = parser.add_mutually_exclusive_group()
    profiles.add_argument("--one-hour", dest="tag", action="store_const", const=ONE_HOUR_TAG,
                          help="Resume the shortened seed-0 comparison")
    profiles.add_argument("--sequence-500k", dest="tag", action="store_const", const=SEQUENCE_TAG,
                          help="Run the ten-visit, three-method comparison")
    parser.set_defaults(tag=FULL_TAG)
    args = parser.parse_args()
    {"start": start, "status": status, "stop": stop}[args.action](args.tag)
