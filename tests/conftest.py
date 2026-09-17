"""Shared offline fixtures: synthetic SAMeT datasets and a fake fetcher."""

from __future__ import annotations

from datetime import date
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from samet.retrieval import HeadInfo


if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


LATITUDES = np.array([-30.0, -15.0, 0.0, 5.0])
LONGITUDES = np.array([-75.0, -60.0, -45.0, -34.05])

SAMPLE_FILE_SIZE = 4096


def probe_day() -> date:
    """Return a fixed, valid day for tests."""
    return date(2026, 9, 15)


def probe_hour() -> datetime:
    """Return a fixed, valid hour for tests."""
    return datetime(2026, 9, 15, 12)


def sample_samet_dataset(
    *,
    variable: str = "tmax",
    day: date | None = None,
    hour: datetime | None = None,
    constant: bool = False,
) -> xr.Dataset:
    """Build a synthetic dataset shaped like a decoded SAMeT hypercube."""
    generator = np.random.default_rng(seed=7)
    shape = (len(LATITUDES), len(LONGITUDES))
    block = (
        np.full(shape, 20.0) if constant else generator.normal(20.0, 8.0, size=shape)
    )
    nobs_name = "nobs" if variable == "temperature" else f"nobs_{variable}"
    nobs = generator.integers(1, 4, size=shape).astype("float32")
    reference: object = (
        hour if hour is not None else np.datetime64(f"{day or probe_day()}T00:00:00")
    )
    return xr.Dataset(
        data_vars={
            variable: (("lat", "lon"), block),
            nobs_name: (("lat", "lon"), nobs),
        },
        coords={
            "lat": LATITUDES,
            "lon": LONGITUDES,
            "time": reference,
        },
    )


def write_source_file(
    path: Path,
    *,
    variable: str = "tt2m",
    day: date | None = None,
    hour: datetime | None = None,
) -> Path:
    """Write a synthetic NetCDF in the source vocabulary (tmed/tt2m + nobs)."""
    generator = np.random.default_rng(seed=7)
    shape = (1, len(LATITUDES), len(LONGITUDES))
    reference: object = (
        hour if hour is not None else np.datetime64(f"{day or probe_day()}T00:00:00")
    )
    dataset = xr.Dataset(
        data_vars={
            variable: (("time", "lat", "lon"), generator.normal(20.0, 8.0, shape)),
            "nobs": (
                ("time", "lat", "lon"),
                generator.integers(1, 4, shape).astype("float32"),
            ),
        },
        coords={
            "time": [reference],
            "lat": LATITUDES,
            "lon": LONGITUDES,
        },
    )
    dataset.to_netcdf(path)
    return path


class FakeFetcher:
    """In-memory transport that records every request it serves."""

    def __init__(
        self,
        *,
        file_size: int = SAMPLE_FILE_SIZE,
        status: int = 200,
        payload: bytes | None = None,
    ) -> None:
        self.file_size = file_size
        self.status = status
        self.payload = payload or b"S" * 4096
        self.heads: list[str] = []
        self.ranges: list[tuple[str, int, int]] = []

    def head(self, url: str) -> HeadInfo:
        """Return the configured status and content length."""
        self.heads.append(url)
        return HeadInfo(
            status_code=self.status,
            headers={"content-length": str(self.file_size)},
        )

    def stream_range(self, url: str, start: int, end: int) -> Iterable[bytes]:
        """Yield one block per requested range and record it."""
        self.ranges.append((url, start, end))
        block = self.payload[: max(1, min(len(self.payload), end - start + 1))]
        yield block
