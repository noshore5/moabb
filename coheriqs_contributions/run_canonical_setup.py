"""Run the canonical evidence-GNN setup without IDE parameters.

Keep ``CANONICAL_ARGS`` aligned with ``run_canonical_setup.sh``.
"""

import os
from pathlib import Path
from importlib import import_module


CONTRIB_DIR = Path(__file__).resolve().parent
REPO_ROOT = CONTRIB_DIR.parent

# Flip this between "msc", "wct", "eegnet", and "sparse" to switch the canonical pipeline.
CANONICAL_VARIANT = "sparse"

CANONICAL_CONFIG = {
    "msc": {
        "runner": "run_msc_gnn",
        "pipeline": "MSC-Evidence-GNN",
        "args": [
            "--subjects",
            "11",
            "--pipeline",
            "MSC-Evidence-GNN",
            "--run-id",
            "canonical",
            "--console-all",
            "--no-console-train-steps",
            "--console-selector-every",
            "0",
        ],
    },
    "wct": {
        "runner": "run_wct_gnn",
        "pipeline": "WCT-Evidence-GNN",
        "args": [
            "--subjects",
            "3",
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
        ],
    },
    "eegnet": {
        "runner": "run_wct_gnn",
        "pipeline": "EEGNet",
        "args": [
            "--subjects",
            "3",
            "--pipeline",
            "EEGNet",
            "--run-id",
            "canonical-eegnet",
            "--console-all",
            "--no-console-train-steps",
            "--console-selector-every",
            "0",
        ],
    },
    "sparse": {
        "runner": "run_wct_gnn",
        "pipeline": "Sparse-Evidence-GNN",
        "args": [
            "--subjects",
            "1",
            "2",
            "3",
            "4",
            "--pipeline",
            "Sparse-Evidence-GNN",
            "--run-id",
            "canonical-sparse",
            "--console-all",
            "--no-console-train-steps",
            "--console-selector-every",
            "0",
        ],
    },
}


def _canonical_config() -> dict[str, object]:
    try:
        return CANONICAL_CONFIG[CANONICAL_VARIANT]
    except KeyError as exc:
        raise ValueError(
            "CANONICAL_VARIANT must be one of 'msc', 'wct', 'eegnet', or 'sparse'."
        ) from exc


CANONICAL_ARGS = _canonical_config()["args"]


def main() -> None:
    """Run the canonical configuration in the current Python process."""
    os.chdir(REPO_ROOT)
    config = _canonical_config()
    runner_name = config["runner"]
    parse_module = import_module(runner_name)

    parse_module.main(parse_module.parse_parameters(CANONICAL_ARGS))


if __name__ == "__main__":
    main()
