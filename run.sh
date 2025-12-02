#!/usr/bin/env bash
set -euo pipefail

# run from script directory
cd "$(dirname "$0")"
RUNFILE="camera_orient_detection.py"

# create a local venv if neither .venv nor venv exist
if [ ! -d "./venv" ]; then
  if command -v python3 >/dev/null 2>&1; then
    python3 -m venv venv
    ./venv/bin/pip install -r requirements.txt
  elif command -v python >/dev/null 2>&1; then
    python -m venv venv
    ./venv/bin/pip install -r requirements.txt
  else
    echo "No Python interpreter found to create virtualenv" >&2
    exit 1
  fi
fi

# pick Python: prefer a local venv, then system python3, then python
if [ -x "./venv/bin/python3" ]; then
  PYTHON="./venv/bin/python3"
  $PYTHON $RUNFILE "$@"

elif [ -x "./venv/bin/python" ]; then
  PYTHON="./venv/bin/python"
  $PYTHON $RUNFILE "$@"

else
  echo "No Python interpreter found to run the script" >&2
  exit 1
fi

