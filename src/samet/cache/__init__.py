"""Experiment cache identity, namespaces, manifests, coverage, and provenance."""

from __future__ import annotations

from pathlib import Path

from samet.cache.coverage import GRID_SIGNATURE
from samet.cache.coverage import summarize_coverage
from samet.cache.identity import CACHE_SCHEMA_VERSION
from samet.cache.identity import CACHE_SOURCE
from samet.cache.identity import experiment_cache_key
from samet.cache.identity import normalize_dataclass
from samet.cache.identity import normalize_value
from samet.cache.identity import request_fingerprint
from samet.cache.manifest import ExperimentManifest
from samet.cache.manifest import StoreRecord
from samet.cache.namespace import ExperimentNamespace
from samet.cache.namespace import default_namespace
from samet.cache.namespace import validate_slug
from samet.cache.provenance import LEGEND
from samet.cache.provenance import SINGLE
from samet.cache.provenance import legend_payload


def experiment_cache_dir(cache_root: Path, cache_key: str) -> Path:
    """Return the legacy isolated cache directory for one experiment."""
    return Path(cache_root) / CACHE_SOURCE / cache_key


def experiment_store_path(data_root: Path, cache_key: str) -> Path:
    """Return the legacy canonical Zarr path for one experiment."""
    return Path(data_root) / CACHE_SOURCE / f"{cache_key}.zarr"


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CACHE_SOURCE",
    "GRID_SIGNATURE",
    "LEGEND",
    "SINGLE",
    "ExperimentManifest",
    "ExperimentNamespace",
    "StoreRecord",
    "default_namespace",
    "experiment_cache_dir",
    "experiment_cache_key",
    "experiment_store_path",
    "legend_payload",
    "normalize_dataclass",
    "normalize_value",
    "request_fingerprint",
    "summarize_coverage",
    "validate_slug",
]
