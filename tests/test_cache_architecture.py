"""Tests for the cache namespace, fingerprint, and manifest contract."""

from __future__ import annotations

from datetime import date
from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from samet.cache import ExperimentNamespace
from samet.cache import request_fingerprint
from samet.cache import validate_slug
from samet.exceptions import SametValidationError
from samet.models import DailyRequest
from samet.models import HourlyRequest


if TYPE_CHECKING:
    from pathlib import Path


def _daily(day: date = date(2026, 9, 1)) -> DailyRequest:
    return DailyRequest(day=day)


def _hourly(hour: datetime = datetime(2026, 9, 1, 12)) -> HourlyRequest:
    return HourlyRequest(start=hour, end=hour)


def test_fingerprint_excludes_name_and_tracks_fields() -> None:
    same = request_fingerprint({"daily": _daily()})
    assert same == request_fingerprint({"daily": _daily()})
    assert same != request_fingerprint({"daily": _daily(date(2026, 9, 2))})
    assert same != request_fingerprint({"surface": _daily()})


def test_fingerprint_separates_daily_and_hourly() -> None:
    assert request_fingerprint({"x": _daily()}) != request_fingerprint({"x": _hourly()})


def test_slug_validation() -> None:
    assert validate_slug("samet-brasil") == "samet-brasil"
    for bad in ("", " ", "../escape", "a/b", ".hidden"):
        with pytest.raises(SametValidationError):
            validate_slug(bad)


def test_namespace_paths(tmp_path: Path) -> None:
    namespace = ExperimentNamespace(tmp_path, "samet", "daily")
    assert namespace.pool_dir == tmp_path / ".cache" / "fragments" / "samet" / "v2"
    assert namespace.store_path("abc") == (
        tmp_path / ".cache" / "stores" / "samet" / "daily" / "abc.zarr"
    )


def test_pool_is_shared_across_namespaces(tmp_path: Path) -> None:
    first = ExperimentNamespace(tmp_path, "samet", "daily")
    second = ExperimentNamespace(tmp_path, "samet", "hourly")
    assert first.pool_dir == second.pool_dir
    assert first.data_dir != second.data_dir


def test_store_path_tracks_fingerprint(tmp_path: Path) -> None:
    namespace = ExperimentNamespace(tmp_path, "samet", "daily")
    first = request_fingerprint({"daily": _daily()})
    second = request_fingerprint({"daily": _daily(date(2026, 9, 2))})
    assert namespace.store_path(first) != namespace.store_path(second)


def test_manifest_round_trip(tmp_path: Path) -> None:
    namespace = ExperimentNamespace(tmp_path, "samet", "daily")
    assert namespace.load_manifest() is None
    namespace.record_store(
        "fp1",
        requests={"daily": {"type": "DailyRequest"}},
        coverage={"time": ["2026-09-01", "2026-09-01"]},
        provenance={"0": "samet"},
        now="2026-09-18T12:00:00+00:00",
    )
    manifest = namespace.load_manifest()
    assert manifest is not None
    assert manifest.current == "fp1"
    assert manifest.stores["fp1"].coverage["time"] == ["2026-09-01", "2026-09-01"]
    assert manifest.stores["fp1"].provenance == {"0": "samet"}
