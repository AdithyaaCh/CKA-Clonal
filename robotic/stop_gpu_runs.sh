#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec "${PYTHON_BIN:-python3}" robotic_jobs.py stop
