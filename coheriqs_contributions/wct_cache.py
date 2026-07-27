"""Safe reusable storage for deterministic WCT tensor computations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Callable, Mapping, Sequence
from uuid import uuid4

import numpy as np


CACHE_SCHEMA_VERSION = 1
_ROOT_MARKER = ".wct-cache-root.json"
_ENTRY_METADATA = "metadata.json"
_LOCK_POLL_SECONDS = 0.05


@dataclass(frozen=True)
class CacheResult:
    tensors: dict[str, np.ndarray]
    hit: bool
    key: str


@dataclass(frozen=True)
class CacheInspection:
    entries: int
    invalid_entries: int
    bytes: int


def stable_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_digest(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return _digest_bytes(memoryview(contiguous).cast("B"))


def exact_array_identity(value: np.ndarray) -> dict[str, object]:
    array = np.asarray(value)
    return {
        "sha256": _array_digest(array),
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "byte_order": array.dtype.byteorder,
    }


class _PerKeyLock:
    def __init__(self, path: Path, *, timeout_seconds: float) -> None:
        self.path = path
        self.timeout_seconds = timeout_seconds
        self._owned = False

    def __enter__(self) -> "_PerKeyLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_seconds
        payload = stable_json({"pid": os.getpid(), "created_ns": time.time_ns()})
        while True:
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for cache lock: {self.path}")
                time.sleep(_LOCK_POLL_SECONDS)
                continue
            try:
                os.write(descriptor, payload.encode("utf-8"))
            finally:
                os.close(descriptor)
            self._owned = True
            return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._owned:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            self._owned = False


class TensorCache:
    """Canonical immutable-entry cache used by narrow WCT adapters."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        kind: str,
        lock_timeout_seconds: float = 120.0,
    ) -> None:
        if not kind or any(character in kind for character in "/\\"):
            raise ValueError("Cache kind must be one path component.")
        self.root = Path(root).expanduser().resolve()
        self.kind = kind
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self.kind_root = self.root / kind / f"v{CACHE_SCHEMA_VERSION}"

    def _initialize_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        marker = self.root / _ROOT_MARKER
        if not marker.exists():
            marker.write_text(
                stable_json({"format": "wct-cache", "schema_version": CACHE_SCHEMA_VERSION}),
                encoding="utf-8",
            )

    def _key(self, fields: Mapping[str, object]) -> tuple[str, dict[str, object]]:
        key_fields = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "kind": self.kind,
            **dict(fields),
        }
        return _digest_bytes(stable_json(key_fields).encode("utf-8")), key_fields

    def _entry_path(self, key: str) -> Path:
        return self.kind_root / key[:2] / key

    @staticmethod
    def _load_valid_entry(
        path: Path,
        *,
        expected_key: str,
        expected_names: Sequence[str],
    ) -> dict[str, np.ndarray] | None:
        try:
            metadata = json.loads((path / _ENTRY_METADATA).read_text(encoding="utf-8"))
            if metadata["schema_version"] != CACHE_SCHEMA_VERSION:
                return None
            if metadata["key"] != expected_key:
                return None
            arrays = metadata["arrays"]
            if set(arrays) != set(expected_names):
                return None
            loaded: dict[str, np.ndarray] = {}
            for name in expected_names:
                manifest = arrays[name]
                value = np.load(path / f"{name}.npy", allow_pickle=False)
                if list(value.shape) != manifest["shape"]:
                    return None
                if value.dtype.str != manifest["dtype"]:
                    return None
                if _array_digest(value) != manifest["sha256"]:
                    return None
                value.flags.writeable = False
                loaded[name] = value
            return loaded
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            return None

    def _write_temporary(
        self,
        path: Path,
        *,
        key: str,
        key_fields: Mapping[str, object],
        tensors: Mapping[str, np.ndarray],
    ) -> None:
        path.mkdir(parents=False, exist_ok=False)
        manifests: dict[str, object] = {}
        for name in sorted(tensors):
            if not name or any(character in name for character in "/\\"):
                raise ValueError(f"Invalid tensor name: {name!r}")
            value = np.ascontiguousarray(tensors[name])
            np.save(path / f"{name}.npy", value, allow_pickle=False)
            manifests[name] = {
                "shape": list(value.shape),
                "dtype": value.dtype.str,
                "sha256": _array_digest(value),
            }
        metadata = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "key": key,
            "key_fields": key_fields,
            "arrays": manifests,
        }
        (path / _ENTRY_METADATA).write_text(stable_json(metadata), encoding="utf-8")

    def load_or_compute(
        self,
        *,
        key_fields: Mapping[str, object],
        expected_names: Sequence[str],
        producer: Callable[[], Mapping[str, np.ndarray]],
    ) -> CacheResult:
        key, complete_key_fields = self._key(key_fields)
        self._initialize_root()
        entry = self._entry_path(key)
        loaded = self._load_valid_entry(
            entry,
            expected_key=key,
            expected_names=expected_names,
        )
        if loaded is not None:
            return CacheResult(loaded, True, key)

        lock_path = self.root / ".locks" / self.kind / f"{key}.lock"
        with _PerKeyLock(lock_path, timeout_seconds=self.lock_timeout_seconds):
            loaded = self._load_valid_entry(
                entry,
                expected_key=key,
                expected_names=expected_names,
            )
            if loaded is not None:
                return CacheResult(loaded, True, key)

            tensors = {
                name: np.ascontiguousarray(value)
                for name, value in producer().items()
            }
            if set(tensors) != set(expected_names):
                raise ValueError(
                    f"Producer returned {sorted(tensors)}; expected {sorted(expected_names)}."
                )
            entry.parent.mkdir(parents=True, exist_ok=True)
            if entry.exists():
                quarantine = entry.with_name(f"{entry.name}.corrupt-{uuid4().hex}")
                os.replace(entry, quarantine)
            temporary = entry.with_name(f".{entry.name}.tmp-{uuid4().hex}")
            try:
                self._write_temporary(
                    temporary,
                    key=key,
                    key_fields=complete_key_fields,
                    tensors=tensors,
                )
                validated = self._load_valid_entry(
                    temporary,
                    expected_key=key,
                    expected_names=expected_names,
                )
                if validated is None:
                    raise RuntimeError("Temporary cache entry failed validation.")
                os.replace(temporary, entry)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            for value in tensors.values():
                value.flags.writeable = False
            return CacheResult(tensors, False, key)


def inspect_cache(root: str | os.PathLike[str]) -> CacheInspection:
    cache_root = Path(root).expanduser().resolve()
    if not (cache_root / _ROOT_MARKER).is_file():
        return CacheInspection(entries=0, invalid_entries=0, bytes=0)
    entries = 0
    invalid = 0
    total_bytes = 0
    for path in cache_root.rglob("*"):
        if path.is_file():
            total_bytes += path.stat().st_size
        if path.is_dir() and ".corrupt-" in path.name:
            invalid += 1
        elif path.is_dir() and (path / _ENTRY_METADATA).is_file():
            entries += 1
            try:
                json.loads((path / _ENTRY_METADATA).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                invalid += 1
    return CacheInspection(entries=entries, invalid_entries=invalid, bytes=total_bytes)


def cleanup_cache(
    root: str | os.PathLike[str],
    *,
    kind: str | None = None,
) -> None:
    """Explicitly remove cache entries, guarded by the cache-root marker."""

    cache_root = Path(root).expanduser().resolve()
    if not (cache_root / _ROOT_MARKER).is_file():
        raise ValueError(f"Refusing to clean an unmarked cache root: {cache_root}")
    target = cache_root if kind is None else cache_root / kind
    try:
        target.relative_to(cache_root)
    except ValueError as exc:
        raise ValueError("Cleanup target escaped the cache root.") from exc
    if kind is None:
        for child in cache_root.iterdir():
            if child.name != _ROOT_MARKER:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
    elif target.exists():
        shutil.rmtree(target)
