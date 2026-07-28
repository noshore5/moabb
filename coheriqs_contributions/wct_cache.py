"""Packed local caches for reusable input and noise-bank CWT tensors."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Callable, Mapping, Sequence
from urllib.parse import quote
from uuid import uuid4

from filelock import FileLock
import numpy as np


CACHE_SCHEMA_VERSION = 1
PACKED_LAYOUT = "raw-real-slot-0-cwt-complex-slots-1+"
PACKED_DTYPE = np.dtype(np.complex64)


def stable_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def digest_fields(fields: Mapping[str, object]) -> str:
    payload = stable_json(dict(fields)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_digest(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _safe_component(label: str, value: object) -> str:
    return f"{label}-{quote(str(value), safe='-_.')}"


def _entry_key(kind: str, fields: Mapping[str, object]) -> str:
    return digest_fields(
        {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "kind": kind,
            "packed_layout": PACKED_LAYOUT,
            **dict(fields),
        }
    )


def _load_memmap(
    path: Path,
    *,
    mode: str,
    expected_shape: Sequence[int],
) -> np.memmap:
    try:
        value = np.load(path, mmap_mode=mode, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Cannot load cache file {path}; refresh the WCT cache."
        ) from exc
    if value.dtype != PACKED_DTYPE or tuple(value.shape) != tuple(expected_shape):
        raise ValueError(
            f"Cache file {path} has shape/dtype {value.shape}/{value.dtype}; "
            f"expected {tuple(expected_shape)}/{PACKED_DTYPE}. Refresh the WCT cache."
        )
    return value


@dataclass(frozen=True)
class PackedCacheRows:
    values: np.ndarray
    hits: int
    misses: int
    key: str


@dataclass(frozen=True)
class NoiseBankHandle:
    packed: np.ndarray
    hit: bool
    key: str
    path: Path | None


class SessionCWTCache:
    """Sparse memory-mapped packed CWT storage keyed by recording session."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        lock_timeout_seconds: float = 120.0,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.kind_root = self.root / "input-cwt" / f"v{CACHE_SCHEMA_VERSION}"
        self.lock_timeout_seconds = float(lock_timeout_seconds)

    def _entry_path(
        self,
        *,
        dataset: object,
        subject: object,
        session: object,
        parameter_fields: Mapping[str, object],
    ) -> tuple[Path, str]:
        key = _entry_key("input-cwt-session", parameter_fields)
        entry = (
            self.kind_root
            / _safe_component("dataset", dataset)
            / _safe_component("subject", subject)
            / _safe_component("session", session)
            / key
        )
        return entry, key

    @staticmethod
    def _requested_rows(
        rows: Sequence[int],
        *,
        session_size: int,
    ) -> np.ndarray:
        requested = np.asarray(rows, dtype=np.int64)
        if requested.ndim != 1:
            raise ValueError("Session trial indices must be one-dimensional.")
        if requested.size and (
            int(requested.min()) < 0 or int(requested.max()) >= session_size
        ):
            raise ValueError(
                f"Session trial indices must be in [0, {session_size}); "
                f"got {requested.tolist()}."
            )
        if np.unique(requested).size != requested.size:
            raise ValueError("Session trial indices must be unique within a request.")
        return requested

    def _load_present(self, path: Path, *, session_size: int, mode: str):
        try:
            present = np.load(path, mmap_mode=mode, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Cannot load cache presence mask {path}; refresh the WCT cache."
            ) from exc
        if present.dtype != np.dtype(bool) or present.shape != (session_size,):
            raise ValueError(
                f"Cache presence mask {path} has shape/dtype "
                f"{present.shape}/{present.dtype}; expected "
                f"{(session_size,)}/{np.dtype(bool)}. Refresh the WCT cache."
            )
        return present

    def _initialize_entry(
        self,
        entry: Path,
        *,
        packed_shape: tuple[int, ...],
        session_size: int,
    ) -> None:
        packed_path = entry / "xcwt.npy"
        present_path = entry / "present.npy"
        if packed_path.exists() != present_path.exists():
            raise ValueError(
                f"Incomplete cache entry {entry}; refresh the WCT cache."
            )
        if packed_path.exists():
            return
        entry.mkdir(parents=True, exist_ok=True)
        packed = np.lib.format.open_memmap(
            packed_path,
            mode="w+",
            dtype=PACKED_DTYPE,
            shape=packed_shape,
        )
        packed.flush()
        del packed
        present = np.lib.format.open_memmap(
            present_path,
            mode="w+",
            dtype=bool,
            shape=(session_size,),
        )
        present[:] = False
        present.flush()
        del present

    def load_or_fill(
        self,
        *,
        dataset: object,
        subject: object,
        session: object,
        parameter_fields: Mapping[str, object],
        session_size: int,
        requested_session_rows: Sequence[int],
        packed_row_shape: Sequence[int],
        producer: Callable[[np.ndarray], np.ndarray],
    ) -> PackedCacheRows:
        session_size = int(session_size)
        if session_size <= 0:
            raise ValueError("session_size must be positive.")
        requested = self._requested_rows(
            requested_session_rows,
            session_size=session_size,
        )
        row_shape = tuple(int(value) for value in packed_row_shape)
        packed_shape = (session_size, *row_shape)
        entry, key = self._entry_path(
            dataset=dataset,
            subject=subject,
            session=session,
            parameter_fields=parameter_fields,
        )
        packed_path = entry / "xcwt.npy"
        present_path = entry / "present.npy"

        if packed_path.is_file() and present_path.is_file():
            present = self._load_present(
                present_path,
                session_size=session_size,
                mode="r",
            )
            complete = bool(np.all(present[requested]))
            del present
            if complete:
                packed = _load_memmap(
                    packed_path,
                    mode="r",
                    expected_shape=packed_shape,
                )
                values = np.array(packed[requested], copy=True)
                del packed
                return PackedCacheRows(values, int(requested.size), 0, key)

        lock_path = entry.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(lock_path), timeout=self.lock_timeout_seconds):
            self._initialize_entry(
                entry,
                packed_shape=packed_shape,
                session_size=session_size,
            )
            packed = _load_memmap(
                packed_path,
                mode="r+",
                expected_shape=packed_shape,
            )
            present = self._load_present(
                present_path,
                session_size=session_size,
                mode="r+",
            )
            missing_positions = np.flatnonzero(~present[requested])
            if missing_positions.size:
                produced = np.ascontiguousarray(
                    producer(missing_positions),
                    dtype=PACKED_DTYPE,
                )
                expected = (int(missing_positions.size), *row_shape)
                if produced.shape != expected:
                    raise ValueError(
                        f"Packed CWT producer returned {produced.shape}; "
                        f"expected {expected}."
                    )
                missing_rows = requested[missing_positions]
                packed[missing_rows] = produced
                packed.flush()
                present[missing_rows] = True
                present.flush()
            values = np.array(packed[requested], copy=True)
            misses = int(missing_positions.size)
            del present
            del packed
        return PackedCacheRows(
            values,
            hits=int(requested.size) - misses,
            misses=misses,
            key=key,
        )


def load_or_compute_noise_bank(
    root: str | os.PathLike[str],
    *,
    key_fields: Mapping[str, object],
    expected_shape: Sequence[int],
    producer: Callable[[], np.ndarray],
    lock_timeout_seconds: float = 120.0,
) -> NoiseBankHandle:
    cache_root = Path(root).expanduser().resolve()
    kind_root = cache_root / "noise-bank" / f"v{CACHE_SCHEMA_VERSION}"
    key = _entry_key("noise-bank", key_fields)
    path = kind_root / f"{key}.npy"
    shape = tuple(int(value) for value in expected_shape)
    if path.is_file():
        return NoiseBankHandle(
            _load_memmap(path, mode="c", expected_shape=shape),
            True,
            key,
            path,
        )

    kind_root.mkdir(parents=True, exist_ok=True)
    lock_path = kind_root / ".locks" / f"{key}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path), timeout=float(lock_timeout_seconds)):
        if path.is_file():
            return NoiseBankHandle(
                _load_memmap(path, mode="c", expected_shape=shape),
                True,
                key,
                path,
            )
        packed = np.ascontiguousarray(producer(), dtype=PACKED_DTYPE)
        if packed.shape != shape:
            raise ValueError(
                f"Noise-bank producer returned {packed.shape}; expected {shape}."
            )
        temporary = kind_root / f".{key}.tmp-{uuid4().hex}.npy"
        try:
            with temporary.open("wb") as stream:
                np.save(stream, packed, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return NoiseBankHandle(
        _load_memmap(path, mode="c", expected_shape=shape),
        False,
        key,
        path,
    )


def refresh_wct_cache(root: str | os.PathLike[str]) -> tuple[Path, ...]:
    """Remove both active WCT cache namespaces beneath an explicit root."""

    cache_root = Path(root).expanduser().resolve()
    targets = (
        cache_root / "input-cwt" / f"v{CACHE_SCHEMA_VERSION}",
        cache_root / "noise-bank" / f"v{CACHE_SCHEMA_VERSION}",
    )
    removed = []
    for target in targets:
        resolved = target.resolve(strict=False)
        try:
            resolved.relative_to(cache_root)
        except ValueError as exc:
            raise ValueError(f"Cache refresh target escaped its root: {resolved}") from exc
        if resolved.exists():
            shutil.rmtree(resolved)
            removed.append(resolved)
    return tuple(removed)
