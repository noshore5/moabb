#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="${1:-moabb-env}"
ENV_PATH="$REPO_ROOT/$ENV_NAME"
REQS="$SCRIPT_DIR/local_env_requirements.txt"
ACTIVATE="$ENV_PATH/bin/activate"

if [[ -e "$ENV_PATH" ]]; then
  if [[ ! -f "$ACTIVATE" ]]; then
    echo "error: $ENV_PATH exists but is not a venv (missing bin/activate)" >&2
    exit 1
  fi
  echo "Reusing existing env at $ENV_PATH"
else
  if ! command -v python3.11 >/dev/null 2>&1; then
    echo "error: python3.11 not found on PATH" >&2
    exit 1
  fi
  python3.11 -m venv "$ENV_PATH"
  echo "Created $ENV_PATH (Python 3.11)."
fi

# shellcheck disable=SC1091
source "$ACTIVATE"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "$REPO_ROOT"

if [[ -f "$REQS" ]]; then
  python -m pip install -r "$REQS"
fi

echo "Packages installed into $ENV_PATH."
echo "Activate with: source $ACTIVATE"
