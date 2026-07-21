"""Validate the Python runtime used by WCT local orchestration."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path


def _required_environment_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Required environment variable is missing: {name}")
    return Path(value).resolve()


def _require_under(label: str, value: str, root: Path) -> None:
    path = Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"{label} resolved outside {root}: {path}") from exc
    print(f"{label}: {path}")


def check_dependencies() -> None:
    """Import the shared scientific stack without changing it."""
    for name in ("mne", "numpy", "pandas", "sklearn", "torch"):
        module = importlib.import_module(name)
        print(f"{name}: {getattr(module, '__version__', 'unknown')}")


def check_sources() -> None:
    """Require project imports to resolve from their configured sources."""
    import moabb
    import utils.coherence_utils as coherence_utils

    from coheriqs_contributions.moabb_pipelines import common

    worktree = _required_environment_path("LOCAL_ORCHESTRATION_WORKTREE_ROOT")
    coherent_root = _required_environment_path("WCT_COHERENT_MULTIPLEX_ROOT")
    _require_under("moabb", moabb.__file__, worktree)
    _require_under("coheriqs common", common.__file__, worktree)
    _require_under("coherence utils", coherence_utils.__file__, coherent_root)


def check_mne_configuration() -> None:
    """Confirm MNE observes the paths configured by the managed launcher."""
    import mne

    for key in ("MNE_DATA", "MNE_DATASETS_BNCI_PATH", "MOABB_RESULTS"):
        configured = mne.get_config(key)
        expected = _required_environment_path(key)
        if configured is None or Path(configured).resolve() != expected:
            raise SystemExit(
                f"MNE configuration mismatch for {key}: "
                f"{configured!r} != {str(expected)!r}"
            )
        print(f"{key}: {configured}")


def check_compute() -> None:
    """Confirm the requested managed compute policy is effective."""
    import torch

    resource = os.environ.get("LOCAL_ORCHESTRATION_RESOURCE")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if resource == "gpu":
        if not torch.cuda.is_available():
            raise SystemExit(
                "Exclusive GPU execution was requested, but PyTorch cannot see "
                f"CUDA with CUDA_VISIBLE_DEVICES={visible!r}."
            )
        print(f"compute: gpu-exclusive ({torch.cuda.get_device_name(0)})")
    elif resource == "none":
        if torch.cuda.is_available():
            raise SystemExit(
                "CPU-only managed execution unexpectedly retained CUDA visibility: "
                f"CUDA_VISIBLE_DEVICES={visible!r}."
            )
        print("compute: cpu-only")
    else:
        raise SystemExit(f"Unsupported managed resource for WCT execution: {resource!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("dependencies", "sources", "execution"),
        help="Preflight stage to validate.",
    )
    args = parser.parse_args()

    if args.mode == "dependencies":
        check_dependencies()
    elif args.mode == "sources":
        check_sources()
    else:
        check_sources()
        check_mne_configuration()
        check_compute()


if __name__ == "__main__":
    main()
