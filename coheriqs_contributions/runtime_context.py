"""Small, best-effort runtime provenance collector for experiment outputs."""

from __future__ import annotations

import importlib
import importlib.metadata
from datetime import datetime
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping


_PACKAGE_NAMES = ("torch", "moabb", "mne", "numpy", "scipy", "scikit-learn", "fcwt")


def _git_repository(path: Path, role: str, failures: list[str]) -> dict[str, Any]:
    repository = {"role": role, "name": path.name, "path": str(path.resolve())}
    try:
        root = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
        repository["path"] = str(Path(root).resolve())
        repository["name"] = Path(root).name
        repository["commit"] = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
        repository["dirty"] = bool(
            subprocess.check_output(
                ["git", "-C", root, "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2,
            ).strip()
        )
    except (OSError, subprocess.SubprocessError) as exc:
        failures.append(f"repository probe for {role}: {type(exc).__name__}")
    return repository


def _package_versions(failures: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in _PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except Exception as exc:  # Optional package probes must not block a run.
            versions[package] = None
            failures.append(
                f"package version unavailable: {package} ({type(exc).__name__})"
            )
    return versions


def _resolve_devices(
    requested_devices: Mapping[str, list[str]], failures: list[str]
) -> dict[str, Any]:
    """Resolve configured devices with the runner's torch-device semantics."""

    resolved_pipelines: dict[str, dict[str, Any]] = {}
    try:
        torch = importlib.import_module("torch")
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        torch = None
        cuda_available = False
        failures.append(f"device probe unavailable: {type(exc).__name__}")

    for pipeline, requested_values in sorted(requested_devices.items()):
        resolutions: dict[str, str] = {}
        for requested in sorted(set(requested_values)):
            try:
                if requested == "auto":
                    candidate = "cuda" if cuda_available else "cpu"
                elif requested.startswith("cuda") and not cuda_available:
                    candidate = "cpu"
                else:
                    candidate = requested
                resolutions[requested] = str(torch.device(candidate))
            except Exception as exc:
                resolutions[requested] = "invalid"
                failures.append(
                    f"device request unavailable: {requested} ({type(exc).__name__})"
                )
        resolved_pipelines[pipeline] = {
            "requested": sorted(set(requested_values)),
            "resolved": resolutions,
        }

    device_context: dict[str, Any] = {"pipelines": resolved_pipelines}
    cuda_devices = sorted(
        {
            resolved
            for pipeline in resolved_pipelines.values()
            for resolved in pipeline["resolved"].values()
            if resolved.startswith("cuda")
        }
    )
    if cuda_devices and torch is not None:
        try:
            device_context["cuda"] = {
                "runtime": torch.version.cuda,
                "devices": {
                    device: torch.cuda.get_device_name(
                        int(device.split(":", 1)[1]) if ":" in device else 0
                    )
                    for device in cuda_devices
                },
            }
        except Exception as exc:
            failures.append(f"CUDA detail probe unavailable: {type(exc).__name__}")
    return device_context


def collect_runtime_context(
    *,
    run_id: str,
    parsed_arguments: Mapping[str, Any],
    repository_root: Path,
    pipeline_devices: Mapping[str, list[str]],
    source_repositories: Mapping[str, Path] | None = None,
    source_probe_failures: list[str] | None = None,
) -> dict[str, Any]:
    """Collect lightweight local facts; optional probe failures stay non-fatal."""

    failures = list(source_probe_failures or [])
    repositories = [_git_repository(repository_root, "main", failures)]
    for role, path in sorted((source_repositories or {}).items()):
        repositories.append(
            _git_repository(Path(path), role, failures)
        )

    return {
        "schema_version": 1,
        "run_id": run_id,
        "start_time": datetime.now().astimezone().isoformat(),
        "invocation": list(sys.argv),
        "parsed_arguments": dict(parsed_arguments),
        "working_directory": str(Path.cwd()),
        "repositories": repositories,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
        },
        "package_versions": _package_versions(failures),
        "device": _resolve_devices(pipeline_devices, failures),
        "probe_failures": failures,
    }
