"""Manage the detached push-wall GPU comparison."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys


HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"
PID_FILE = LOGS / "robotic_suite.pid"
LOG_FILE = LOGS / "robotic_suite.log"


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def start():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch missing; run ./bootstrap_gpu_env.sh first") from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU unavailable; no job started")
    LOGS.mkdir(exist_ok=True)
    if PID_FILE.exists() and alive(int(PID_FILE.read_text())):
        print(f"Already running: PID {PID_FILE.read_text().strip()}")
        return
    with LOG_FILE.open("a") as log:
        process = subprocess.Popen([sys.executable, str(HERE / "run_robotic_suite.py")],
                                   cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True,
                                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    PID_FILE.write_text(f"{process.pid}\n")
    print(f"Started PID {process.pid}; log {LOG_FILE}")


def status():
    if not PID_FILE.exists():
        print("Never started")
        return
    pid = int(PID_FILE.read_text())
    print(f"{'Running' if alive(pid) else 'Stopped'} PID {pid}; log {LOG_FILE}")
    if LOG_FILE.exists():
        lines = [line.strip() for line in LOG_FILE.read_text(errors="replace").splitlines()
                 if line.startswith(("START ", "DONE ", "Results saved", "Traceback", "RuntimeError"))]
        for line in lines[-5:]:
            print(line)


def stop():
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text())
        if alive(pid):
            os.killpg(pid, signal.SIGTERM)
            print(f"Sent stop signal to process group {pid}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "status", "stop"))
    {"start": start, "status": status, "stop": stop}[parser.parse_args().action]()
