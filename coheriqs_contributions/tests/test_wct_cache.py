from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from coheriqs_contributions.moabb_pipelines import common
from coheriqs_contributions.wct_cache import refresh_wct_cache


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
        coefficients = (
            values[None, :] * factors[:, None]
            + 1j * values[None, :] * (factors[:, None] + 0.25)
        ).astype(np.complex64)
        frequencies = np.linspace(lowest, highest, nfreqs, dtype=np.float32)
        return coefficients, frequencies


def _metadata(rows, *, session_size=3):
    return pd.DataFrame(
        {
            "dataset": ["dataset"] * len(rows),
            "subject": ["1"] * len(rows),
            "session": ["0"] * len(rows),
            "session_trial_index": rows,
            "session_trial_count": [session_size] * len(rows),
        }
    )


def _cached(X, metadata, transform, root, *, nfreqs=2):
    return common.compute_cached_cwt_real_imag_tensors(
        X,
        metadata=metadata,
        sampling_rate=8,
        highest=4.0,
        lowest=1.0,
        nfreqs=nfreqs,
        cwt_resample_n_time=6,
        transform_fn=transform,
        cache_root=root,
        verbose=0,
    )


def test_session_cache_incrementally_fills_and_selects_rows(tmp_path: Path) -> None:
    transform = CountingTransform()
    full = np.arange(24, dtype=np.float32).reshape(3, 1, 8)

    first, first_stats = _cached(
        full[[0, 2]], _metadata([0, 2]), transform, tmp_path
    )
    middle, middle_stats = _cached(
        full[[1]], _metadata([1]), transform, tmp_path
    )
    reordered, reordered_stats = _cached(
        full[[2, 0]], _metadata([2, 0]), transform, tmp_path
    )

    assert (first_stats.hits, first_stats.misses) == (0, 2)
    assert (middle_stats.hits, middle_stats.misses) == (0, 1)
    assert (reordered_stats.hits, reordered_stats.misses) == (2, 0)
    for original, selected in zip(first[:3], reordered[:3]):
        np.testing.assert_array_equal(original.numpy()[::-1], selected.numpy())
    present_path = next(tmp_path.rglob("present.npy"))
    np.testing.assert_array_equal(np.load(present_path), [True, True, True])


def test_session_cache_parameters_select_separate_files(tmp_path: Path) -> None:
    transform = CountingTransform()
    X = np.arange(8, dtype=np.float32).reshape(1, 1, 8)

    _, first = _cached(X, _metadata([0], session_size=1), transform, tmp_path)
    _, changed = _cached(
        X, _metadata([0], session_size=1), transform, tmp_path, nfreqs=3
    )

    assert first.misses == changed.misses == 1
    assert len(list(tmp_path.rglob("xcwt.npy"))) == 2


def test_noise_cache_is_packed_mapped_and_preserves_pairing(tmp_path: Path) -> None:
    transform = CountingTransform()
    arguments = dict(
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
    )

    first, first_handle = common.compute_paired_cwt_noise_bank(**arguments)
    calls_after_first = transform.calls
    second, second_handle = common.compute_paired_cwt_noise_bank(**arguments)

    assert first_handle.hit is False
    assert second_handle.hit is True
    assert isinstance(second_handle.packed, np.memmap)
    assert transform.calls == calls_after_first
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left.numpy(), right.numpy())
    raw, real, imag = (tensor.numpy() for tensor in second)
    np.testing.assert_allclose(real[..., 0], raw)
    np.testing.assert_allclose(imag[..., 0], raw * 1.25)
    assert len(list((tmp_path / "noise-bank").rglob("*.npy"))) == 1


def test_refresh_removes_only_wct_cache_namespaces(tmp_path: Path) -> None:
    input_dir = tmp_path / "input-cwt" / "v1"
    noise_dir = tmp_path / "noise-bank" / "v1"
    unrelated = tmp_path / "keep.txt"
    input_dir.mkdir(parents=True)
    noise_dir.mkdir(parents=True)
    unrelated.write_text("keep", encoding="utf-8")

    refresh_wct_cache(tmp_path)

    assert not input_dir.exists()
    assert not noise_dir.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"
