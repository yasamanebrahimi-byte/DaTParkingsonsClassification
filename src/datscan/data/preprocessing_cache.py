"""Persistent cache for deterministic NIfTI preprocessing results."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

from ..utils.config import PreprocessConfig, ROIConfig
from .preprocessing import preprocessing_fingerprint

logger = logging.getLogger(__name__)


class PreprocessingCache:
    """Lazily cache deterministic preprocessed volumes as validated ``.npy`` files."""

    def __init__(
        self,
        cache_dir: str | Path,
        config: PreprocessConfig,
        data_view: str = "global",
        roi_config: ROIConfig | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.data_view = data_view
        self.roi_config = roi_config
        if data_view not in {"global", "roi"}:
            raise ValueError(f"Unknown data_view: {data_view}")
        if data_view == "roi" and (roi_config is None or not roi_config.enabled):
            raise ValueError("ROI cache entries require roi.enabled=true")
        shape = roi_config.roi_shape if data_view == "roi" and roi_config is not None else config.output_shape
        self._expected_shape = (1, *(int(size) for size in shape))
        self.hits = 0
        self.misses = 0
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Preprocessing cache enabled at %s", self.cache_dir)

    def cache_path(self, source_path: str | Path, uid: str) -> Path:
        """Return the deterministic cache path for one source scan and config."""
        source = Path(source_path)
        stat = source.stat()
        identity = {
            "source_path": str(source.resolve()),
            "uid": str(uid),
            "source_size": int(stat.st_size),
            "source_mtime_ns": int(stat.st_mtime_ns),
            "preprocessing": preprocessing_fingerprint(self.config, self.data_view, self.roi_config),
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        return self.cache_dir / f"{digest}.npy"

    def get_or_create(self, source_path: str | Path, uid: str, processor: Callable[[], np.ndarray]) -> np.ndarray:
        """Load a valid cached array or preprocess and atomically cache a new one."""
        path = self.cache_path(source_path, uid)
        cached = self._load_valid(path)
        if cached is not None:
            self.hits += 1
            return cached

        self.misses += 1
        array = self._validate_array(processor())
        self._save_atomic(path, array)
        return array

    def _load_valid(self, path: Path) -> np.ndarray | None:
        if not path.is_file():
            return None
        try:
            loaded = np.load(path, allow_pickle=False)
            return self._validate_array(loaded)
        except (OSError, ValueError, EOFError, TypeError):
            logger.warning("Ignoring invalid preprocessing cache entry: %s", path)
            return None

    def _validate_array(self, array: np.ndarray) -> np.ndarray:
        value = np.asarray(array)
        if value.shape != self._expected_shape:
            raise ValueError(f"Cached preprocessing shape {value.shape} does not match {self._expected_shape}")
        if not np.issubdtype(value.dtype, np.number):
            raise ValueError(f"Cached preprocessing dtype {value.dtype} is not numeric")
        value = np.asarray(value, dtype=np.float32)
        if not np.isfinite(value).all():
            raise ValueError("Cached preprocessing contains non-finite values")
        return np.ascontiguousarray(value)

    def _save_atomic(self, path: Path, array: np.ndarray) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.cache_dir,
                prefix=f".{path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                np.save(handle, array, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass
