#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-<name>}"

python -m venv "$ENV_NAME"
source "$ENV_NAME/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .

if [ -f local_env_requirements.txt ]; then
  python -m pip install -r local_env_requirements.txt
fi