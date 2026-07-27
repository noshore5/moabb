from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import pytest

from coheriqs_contributions.moabb_pipelines import common
from coheriqs_contributions.wct_cache import (
    TensorCache,
    cleanup_cache,
    inspect_cache,
)


class CountingTransform:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        values,
        sampling_rate,
        highest,
        lowest,
        *,
        nfreqs,
    ):
        self.calls += 1
        values = np.asarray(values, dtype=np.float32)
        factors = np.arange(1, nfreqs + 1, dtype=np.float32)
        coeffs = (
            values[None, :] * factors[:, None]
            + 1j * values[None, :] * (factors[:, None] + 0.25)
        ).astype(np.complex64)
        return coeffs, np.linspace(lowest, highest, nfreqs, dtype=np.float32)


def _tensor_producer(value: float):
    return {"value": np.asarray([value], dtype=np.float32)}


def test_tensor_cache_stable_keys_hits_parameter_misses_and_immutable_values(
    tmp_path: Path,
) -> None:
    cache = TensorCache(tmp_path, kind="unit", namespace="tests")
    first = cache.load_or_compute(
        key_fields={"parameter": 1, "nested": {"b": 2, "a": 1}},
        expected_names=("value",),
        producer=lambda: _tensor_producer(3.0),
    )
    second = cache.load_or_compute(
        key_fields={"nested": {"a": 1, "b": 2}, "parameter": 1},
        expected_names=("value",),
        producer=lambda: pytest.fail("cache hit unexpectedly recomputed"),
    )
    changed = cache.load_or_compute(
        key_fields={"parameter": 2, "nested": {"a": 1, "b": 2}},
        expected_names=("value",),
        producer=lambda: _tensor_producer(4.0),
    )

    assert first.hit is False
    assert second.hit is True
    assert changed.hit is False
    assert first.key == second.key != changed.key
    assert second.tensors["value"].flags.writeable is False


def test_corrupt_and_incomplete_entries_are_misses_and_republished(
    tmp_path: Path,
) -> None:
    cache = TensorCache(tmp_path, kind="unit", namespace="tests")
    first = cache.load_or_compute(
        key_fields={"case": "corrupt"},
        expected_names=("value",),
        producer=lambda: _tensor_producer(1.0),
    )
    entry = next(path.parent for path in tmp_path.rglob("value.npy"))
    (entry / "value.npy").unlink()

    repaired = cache.load_or_compute(
        key_fields={"case": "corrupt"},
        expected_names=("value",),
        producer=lambda: _tensor_producer(2.0),
    )
    hit = cache.load_or_compute(
        key_fields={"case": "corrupt"},
        expected_names=("value",),
        producer=lambda: pytest.fail("repaired entry unexpectedly recomputed"),
    )

    assert repaired.hit is False
    assert hit.hit is True
    assert hit.key == first.key
    np.testing.assert_array_equal(hit.tensors["value"], [2.0])
    assert inspect_cache(tmp_path).invalid_entries >= 1


def test_simultaneous_writers_publish_once(tmp_path: Path) -> None:
    calls = 0

    def produce():
        nonlocal calls
        calls += 1
        time.sleep(0.1)
        return _tensor_producer(7.0)

    def access():
        return TensorCache(tmp_path, kind="unit", namespace="tests").load_or_compute(
            key_fields={"same": True},
            expected_names=("value",),
            producer=produce,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: access(), range(2)))

    assert calls == 1
    assert sorted(result.hit for result in results) == [False, True]
    for result in results:
        np.testing.assert_array_equal(result.tensors["value"], [7.0])


def test_dirty_source_bypasses_without_namespace(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        common,
        "implementation_identity",
        lambda *sources: {"dirty": True, "sources": [{"dirty": True}]},
    )
    transform = CountingTransform()
    X = np.arange(8, dtype=np.float32).reshape(1, 1, 8)

    _, first = common.compute_cached_cwt_real_imag_tensors(
        X,
        sampling_rate=8,
        highest=4.0,
        lowest=1.0,
        nfreqs=2,
        cwt_resample_n_time=None,
        transform_fn=transform,
        cache_root=tmp_path,
        cache_namespace=None,
        verbose=0,
    )
    _, second = common.compute_cached_cwt_real_imag_tensors(
        X,
        sampling_rate=8,
        highest=4.0,
        lowest=1.0,
        nfreqs=2,
        cwt_resample_n_time=None,
        transform_fn=transform,
        cache_root=tmp_path,
        cache_namespace=None,
        verbose=0,
    )

    assert first.bypasses == second.bypasses == 1
    assert first.hits == second.hits == 0
    assert not (tmp_path / ".wct-cache-root.json").exists()
    assert inspect_cache(tmp_path).entries == 0


def test_input_cache_reuses_per_trial_across_order_and_misses_on_changes(
    tmp_path: Path,
) -> None:
    transform = CountingTransform()
    X = np.arange(16, dtype=np.float32).reshape(2, 1, 8)
    first, first_stats = common.compute_cached_cwt_real_imag_tensors(
        X,
        sampling_rate=8,
        highest=4.0,
        lowest=1.0,
        nfreqs=2,
        cwt_resample_n_time=6,
        transform_fn=transform,
        cache_root=tmp_path,
        cache_namespace="tests",
        verbose=0,
    )
    reversed_features, reversed_stats = common.compute_cached_cwt_real_imag_tensors(
        X[::-1],
        sampling_rate=8,
        highest=4.0,
        lowest=1.0,
        nfreqs=2,
        cwt_resample_n_time=6,
        transform_fn=transform,
        cache_root=tmp_path,
        cache_namespace="tests",
        verbose=0,
    )
    changed = X.copy()
    changed[0, 0, 0] += 1
    _, changed_stats = common.compute_cached_cwt_real_imag_tensors(
        changed,
        sampling_rate=8,
        highest=4.0,
        lowest=1.0,
        nfreqs=2,
        cwt_resample_n_time=6,
        transform_fn=transform,
        cache_root=tmp_path,
        cache_namespace="tests",
        verbose=0,
    )
    _, parameter_stats = common.compute_cached_cwt_real_imag_tensors(
        X,
        sampling_rate=8,
        highest=4.0,
        lowest=1.0,
        nfreqs=3,
        cwt_resample_n_time=6,
        transform_fn=transform,
        cache_root=tmp_path,
        cache_namespace="tests",
        verbose=0,
    )

    assert first_stats.misses == 2
    assert reversed_stats.hits == 2
    assert changed_stats.hits == 1 and changed_stats.misses == 1
    assert parameter_stats.misses == 2
    for original, reversed_value in zip(first[:3], reversed_features[:3]):
        np.testing.assert_array_equal(original.numpy()[::-1], reversed_value.numpy())


def test_noise_cache_preserves_raw_real_imag_pairing(tmp_path: Path) -> None:
    transform = CountingTransform()
    first = common.compute_paired_cwt_noise_bank(
        bank_size=4,
        segment_length=8,
        sampling_rate=8,
        highest=4.0,
        lowest=1.0,
        nfreqs=2,
        cwt_resample_n_time=None,
        transform_fn=transform,
        seed=17,
        verbose=0,
        cache_root=tmp_path,
        cache_namespace="tests",
    )
    calls_after_first = transform.calls
    second = common.compute_paired_cwt_noise_bank(
        bank_size=4,
        segment_length=8,
        sampling_rate=8,
        highest=4.0,
        lowest=1.0,
        nfreqs=2,
        cwt_resample_n_time=None,
        transform_fn=transform,
        seed=17,
        verbose=0,
        cache_root=tmp_path,
        cache_namespace="tests",
    )

    assert transform.calls == calls_after_first
    for left, right in zip(first, second):
        assert left.is_contiguous()
        assert np.array_equal(left.numpy(), right.numpy())
    raw, real, imag = (tensor.numpy() for tensor in second)
    np.testing.assert_allclose(real[..., 0], raw)
    np.testing.assert_allclose(imag[..., 0], raw * 1.25)


def test_real_transform_and_noise_cache_smoke(tmp_path: Path) -> None:
    transform, _ = common.resolve_coherence_utils()
    input_root = tmp_path / "input"
    noise_root = tmp_path / "noise"
    X = np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(1, 2, 16)

    first, first_stats = common.compute_cached_cwt_real_imag_tensors(
        X,
        sampling_rate=16,
        highest=7.0,
        lowest=2.0,
        nfreqs=3,
        cwt_resample_n_time=12,
        transform_fn=transform,
        cache_root=input_root,
        cache_namespace=None,
        verbose=0,
    )
    second, second_stats = common.compute_cached_cwt_real_imag_tensors(
        X,
        sampling_rate=16,
        highest=7.0,
        lowest=2.0,
        nfreqs=3,
        cwt_resample_n_time=12,
        transform_fn=transform,
        cache_root=input_root,
        cache_namespace=None,
        verbose=0,
    )
    first_noise = common.compute_paired_cwt_noise_bank(
        bank_size=2,
        segment_length=16,
        sampling_rate=16,
        highest=7.0,
        lowest=2.0,
        nfreqs=3,
        cwt_resample_n_time=12,
        transform_fn=transform,
        seed=23,
        verbose=0,
        cache_root=noise_root,
        cache_namespace=None,
    )
    second_noise = common.compute_paired_cwt_noise_bank(
        bank_size=2,
        segment_length=16,
        sampling_rate=16,
        highest=7.0,
        lowest=2.0,
        nfreqs=3,
        cwt_resample_n_time=12,
        transform_fn=transform,
        seed=23,
        verbose=0,
        cache_root=noise_root,
        cache_namespace=None,
    )

    assert first_stats.misses == 1
    assert second_stats.hits == 1
    for left, right in zip(first, second):
        assert np.array_equal(left.numpy(), right.numpy())
    for left, right in zip(first_noise, second_noise):
        assert np.array_equal(left.numpy(), right.numpy())


def test_affine_cached_features_match_normalize_before_linear_transform(
    tmp_path: Path,
) -> None:
    transform = CountingTransform()
    X = np.linspace(-3.0, 4.0, 32, dtype=np.float32).reshape(2, 2, 8)
    mean, std = common.fit_global_zscore_stats(X[:1])
    normalized = common.apply_global_zscore(X, mean, std)
    reference = common.compute_cwt_real_imag_tensors(
        normalized,
        sampling_rate=8,
        highest=4.0,
        lowest=1.0,
        nfreqs=2,
        cwt_resample_n_time=6,
        transform_fn=transform,
        verbose=0,
    )
    cached, _ = common.compute_cached_cwt_real_imag_tensors(
        X,
        sampling_rate=8,
        highest=4.0,
        lowest=1.0,
        nfreqs=2,
        cwt_resample_n_time=6,
        transform_fn=transform,
        cache_root=tmp_path,
        cache_namespace="tests",
        verbose=0,
    )
    ones, _ = common.compute_cached_cwt_real_imag_tensors(
        np.ones((1, 1, 8), dtype=np.float32),
        sampling_rate=8,
        highest=4.0,
        lowest=1.0,
        nfreqs=2,
        cwt_resample_n_time=6,
        transform_fn=transform,
        cache_root=tmp_path,
        cache_namespace="tests",
        verbose=0,
    )
    denominator = std + 1e-8
    affine = (
        (cached[0] - mean * ones[0]) / denominator,
        (cached[1] - mean * ones[1]) / denominator,
        (cached[2] - mean * ones[2]) / denominator,
    )

    for expected, actual in zip(reference[:3], affine):
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=2e-5, atol=5e-5)


def test_inspection_and_explicit_cleanup(tmp_path: Path) -> None:
    cache = TensorCache(tmp_path, kind="unit", namespace="tests")
    cache.load_or_compute(
        key_fields={"entry": 1},
        expected_names=("value",),
        producer=lambda: _tensor_producer(1.0),
    )
    inspection = inspect_cache(tmp_path)
    assert inspection.entries == 1
    assert inspection.bytes > 0

    cleanup_cache(tmp_path, kind="unit")
    assert inspect_cache(tmp_path).entries == 0
    with pytest.raises(ValueError, match="unmarked"):
        cleanup_cache(tmp_path / "not-a-cache")


def test_metadata_is_json_and_contains_versioned_key_fields(tmp_path: Path) -> None:
    TensorCache(tmp_path, kind="unit", namespace="tests").load_or_compute(
        key_fields={"entry": 1},
        expected_names=("value",),
        producer=lambda: _tensor_producer(1.0),
    )
    metadata_path = next(tmp_path.rglob("metadata.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["key_fields"]["cache_schema_version"] == 1
    assert metadata["key_fields"]["development_namespace"] == "tests"
