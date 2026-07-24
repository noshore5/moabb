"""Run the canonical WCT-Evidence-GNN setup without IDE parameters.

Keep ``CANONICAL_ARGS`` aligned with ``run_canonical_setup.sh``.
"""

import os
from pathlib import Path


CONTRIB_DIR = Path(__file__).resolve().parent
REPO_ROOT = CONTRIB_DIR.parent

CANONICAL_ARGS = [
    "--subjects",
    "1",
    "--pipeline",
    "WCT-Evidence-GNN",
    "--run-id",
    "canonical",
    "--param-names",
    "window_compute_mode",
    "--param-values",
    "chunked",
    "--console-all",
    "--no-console-train-steps",
    "--console-selector-every",
    "0",
]


def main() -> None:
    """Run the canonical configuration in the current Python process."""
    os.chdir(REPO_ROOT)
    from run_wct_gnn import main as run_wct_gnn
    from run_wct_gnn import parse_parameters

    run_wct_gnn(parse_parameters(CANONICAL_ARGS))


if __name__ == "__main__":
    main()
