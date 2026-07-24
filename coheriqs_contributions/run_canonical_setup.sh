#!/usr/bin/env bash
# Canonical experimental setup (subject 1, WCT-Evidence-GNN).
# Lives in coheriqs_contributions/ next to run_wct_gnn.py.
#
# Keep this script aligned with the current canonical experimental CLI.
# When the active run profile changes, update the arguments below to match
# run_wct_gnn.py.
#
# Safe to invoke from repo root or coheriqs_contributions/ (cwd does not matter).
# Usage (with moabb-env-win active):
#   bash coheriqs_contributions/run_canonical_setup.sh
#   bash run_canonical_setup.sh   # when cwd is coheriqs_contributions/

set -euo pipefail

CONTRIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$CONTRIB_DIR/.." && pwd)"
RUNNER="$CONTRIB_DIR/run_wct_gnn.py"

if [[ ! -f "$RUNNER" ]]; then
  echo "error: expected runner at $RUNNER" >&2
  exit 1
fi

cd "$REPO_ROOT"

exec python "$RUNNER" \
  --subjects 1 \
  --pipeline WCT-Evidence-GNN \
  --run-id canonical \
  --console-all \
  --no-console-train-steps \
  --console-selector-every 0
