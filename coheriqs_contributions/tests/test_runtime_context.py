import subprocess
from pathlib import Path
from types import SimpleNamespace

from coheriqs_contributions.runtime_context import collect_runtime_context


def test_runtime_context_keeps_optional_probe_failures_nonfatal(monkeypatch, tmp_path) -> None:
    def unavailable(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("git", 2)

    monkeypatch.setattr(subprocess, "check_output", unavailable)
    monkeypatch.setattr("importlib.metadata.version", unavailable)

    context = collect_runtime_context(
        run_id="run-1",
        parsed_arguments={"subjects": [1]},
        repository_root=Path(tmp_path),
        pipeline_devices={"pipeline": ["auto"]},
    )

    assert context["run_id"] == "run-1"
    assert context["parsed_arguments"] == {"subjects": [1]}
    assert context["repositories"][0]["role"] == "main"
    assert context["package_versions"]["torch"] is None
    assert any("repository probe" in failure for failure in context["probe_failures"])
    assert any("package version unavailable" in failure for failure in context["probe_failures"])


def test_runtime_context_registers_explicit_source_and_resolves_devices(
    monkeypatch, tmp_path
) -> None:
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_name(index):
            assert index == 0
            return "Test GPU"

    fake_torch = SimpleNamespace(
        cuda=FakeCuda(), version=SimpleNamespace(cuda="12.1"), device=lambda value: value
    )
    monkeypatch.setattr("importlib.import_module", lambda _name: fake_torch)
    monkeypatch.setattr("importlib.metadata.version", lambda name: f"{name}-version")
    monkeypatch.setattr(subprocess, "check_output", lambda *_args, **_kwargs: str(tmp_path))

    context = collect_runtime_context(
        run_id="run-1",
        parsed_arguments={},
        repository_root=tmp_path,
        pipeline_devices={"WCT": ["auto", "cpu"]},
        source_repositories={"coherent_multiplex": tmp_path},
    )

    assert [item["role"] for item in context["repositories"]] == [
        "main",
        "coherent_multiplex",
    ]
    assert context["device"]["pipelines"]["WCT"] == {
        "requested": ["auto", "cpu"],
        "resolved": {"auto": "cuda", "cpu": "cpu"},
    }
    assert context["device"]["cuda"] == {
        "runtime": "12.1",
        "devices": {"cuda": "Test GPU"},
    }


def test_runtime_context_resolves_auto_to_cpu_without_cuda(monkeypatch, tmp_path) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        version=SimpleNamespace(cuda=None),
        device=lambda value: value,
    )
    monkeypatch.setattr("importlib.import_module", lambda _name: fake_torch)

    context = collect_runtime_context(
        run_id="run-1",
        parsed_arguments={},
        repository_root=tmp_path,
        pipeline_devices={"WCT": ["auto", "cuda:0", "cpu"]},
    )

    assert context["device"]["pipelines"]["WCT"]["resolved"] == {
        "auto": "cpu",
        "cpu": "cpu",
        "cuda:0": "cpu",
    }
    assert "cuda" not in context["device"]


def test_runtime_context_records_invalid_cuda_request_without_aborting(
    monkeypatch, tmp_path
) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        version=SimpleNamespace(cuda="12.1"),
        device=lambda value: (_ for _ in ()).throw(ValueError("bad device")),
    )
    monkeypatch.setattr("importlib.import_module", lambda _name: fake_torch)

    context = collect_runtime_context(
        run_id="run-1",
        parsed_arguments={},
        repository_root=tmp_path,
        pipeline_devices={"WCT": ["cuda:not-an-index"]},
    )

    assert context["device"]["pipelines"]["WCT"]["resolved"] == {
        "cuda:not-an-index": "invalid"
    }
    assert any("device request unavailable" in failure for failure in context["probe_failures"])
