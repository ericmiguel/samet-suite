"""Live smoke test against the INPE open-data server (marked ``network``)."""

from __future__ import annotations

from datetime import date
from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
import pytest

from samet import DailyRequest
from samet import Experiment
from samet import HourlyRequest


if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.network


def published_day() -> date:
    """Return a day INPE has certainly published already."""
    return date.today() - timedelta(days=3)


def published_hour() -> datetime:
    """Return an hour INPE has certainly published already."""
    moment = datetime.now() - timedelta(days=3)
    return moment.replace(minute=0, second=0, microsecond=0)


def test_daily_download_and_store(tmp_path: Path) -> None:
    """One published day becomes a plausible three-variable store."""
    experiment = Experiment(
        name="samet_network_daily",
        day=DailyRequest(day=published_day()),
        root_dir=tmp_path,
    )
    paths = experiment.download()
    assert len(paths) == 3
    store = experiment.to_zarr()
    with experiment.open() as dataset:
        assert store.exists()
        assert {"tmax", "tmin", "tmean"} <= set(dataset.data_vars)
        tmax = dataset["tmax"].values
        finite = tmax[np.isfinite(tmax)]
        assert finite.size > 0
        assert -60.0 < float(finite.min()) < float(finite.max()) < 60.0
        above = dataset["tmax"] - dataset["tmin"]
        valid = above.values[np.isfinite(above.values)]
        # The blended fields disagree marginally at a few grid points; the
        # extremes must still order correctly almost everywhere.
        assert float((valid >= -1.0).mean()) > 0.99


def test_hourly_download_and_store(tmp_path: Path) -> None:
    """Two published hours become a plausible temperature store."""
    first = published_hour()
    experiment = Experiment(
        name="samet_network_hourly",
        hours=HourlyRequest(start=first, end=first + timedelta(hours=1)),
        root_dir=tmp_path,
    )
    paths = experiment.download()
    assert len(paths) == 2
    experiment.to_zarr()
    with experiment.open() as dataset:
        assert {"temperature", "nobs"} <= set(dataset.data_vars)
        assert dataset.sizes["time"] == 2
        temperature = dataset["temperature"].values
        finite = temperature[np.isfinite(temperature)]
        assert finite.size > 0
        assert -60.0 < float(finite.min()) < float(finite.max()) < 60.0
