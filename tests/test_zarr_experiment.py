"""Store conversion and experiment orchestration, offline."""

from __future__ import annotations

from datetime import date
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

from conftest import LONGITUDES
from conftest import probe_day
from conftest import probe_hour
from conftest import sample_samet_dataset
from samet import Area
from samet import DailyRequest
from samet import Experiment
from samet import HourlyRequest
from samet import SametValidationError
from samet import apply_area
from samet import files_to_zarr
from samet import normalize_dataset
from samet import write_zarr


if TYPE_CHECKING:
    from pathlib import Path


def daily_request(**overrides: object) -> DailyRequest:
    """Build a daily request, overridable per test."""
    fields: dict[str, object] = {"day": probe_day(), "area": None}
    fields.update(overrides)
    return DailyRequest(**fields)  # type: ignore[arg-type]


def hourly_request(**overrides: object) -> HourlyRequest:
    """Build a single-hour request, overridable per test."""
    fields: dict[str, object] = {
        "start": probe_hour(),
        "end": probe_hour(),
        "area": None,
    }
    fields.update(overrides)
    return HourlyRequest(**fields)  # type: ignore[arg-type]


class RecordingDownloader:
    """Stand-in downloader that touches the filesystem and records calls."""

    def __init__(self) -> None:
        self.calls: list[Path] = []
        self.listeners: list[object] = []
        self.skipped: list[bool] = []

    def download_chunk(self, chunk, *, listener=None, skip_missing=False):  # type: ignore[no-untyped-def]
        """Create the cache file the store step will decode."""
        self.listeners.append(listener)
        self.skipped.append(skip_missing)
        chunk.path.parent.mkdir(parents=True, exist_ok=True)
        chunk.path.write_bytes(b"S" * 64)
        self.calls.append(chunk.path)
        return chunk.path


def daily_decoder(dataset: xr.Dataset | None = None):
    """Return a decoder replacement keyed by the file's variable label."""

    def decode(path: Path) -> xr.Dataset:
        if dataset is not None:
            return dataset
        name = path.name
        if "TMAX" in name:
            return sample_samet_dataset(variable="tmax")
        if "TMIN" in name:
            return sample_samet_dataset(variable="tmin")
        return sample_samet_dataset(variable="tmean")

    return decode


def test_normalize_keeps_canonical_names_and_promotes_time() -> None:
    """SAMeT coordinates are already canonical; time becomes a dimension."""
    normalized = normalize_dataset(sample_samet_dataset())
    assert "lat" in normalized.coords
    assert "lon" in normalized.coords
    assert normalized.sizes["time"] == 1
    assert float(normalized["lon"].min()) >= -180.0


def test_normalize_rejects_out_of_range_longitudes() -> None:
    """The SAMeT convention is [-180, 180]; unsigned longitudes are refused."""
    dataset = sample_samet_dataset()
    unsigned = dataset.assign_coords(lon=np.array([240.0, 260.0, 280.0, 300.0]))
    with pytest.raises(SametValidationError, match="outside"):
        normalize_dataset(unsigned)


def test_two_hours_concatenate_along_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multiple hours land on one time axis, oldest first."""
    hours = {
        "a.nc": datetime(2026, 9, 15, 12),
        "b.nc": datetime(2026, 9, 15, 13),
    }

    def decode(path: Path) -> xr.Dataset:
        return sample_samet_dataset(variable="temperature", hour=hours[path.name])

    monkeypatch.setattr("samet.zarr.open_samet_dataset", decode)
    store = files_to_zarr((tmp_path / "a.nc", tmp_path / "b.nc"), tmp_path / "h.zarr")
    with xr.open_zarr(store, consolidated=False) as dataset:
        assert dataset.sizes["time"] == 2
        assert {"temperature", "nobs"} <= set(dataset.data_vars)
        times = dataset["time"].values
        assert times[0] < times[1]


def test_two_days_merge_the_three_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two days times three variables become one cube per variable."""
    monkeypatch.setattr("samet.zarr.open_samet_dataset", daily_decoder())
    files = tuple(
        tmp_path / f"SAMeT_CPTEC_{label}_2026091{day}.nc"
        for day in (4, 5)
        for label in ("TMAX", "TMIN", "TMED")
    )
    store = files_to_zarr(files, tmp_path / "days.zarr")
    with xr.open_zarr(store, consolidated=False) as dataset:
        assert dataset.sizes["time"] == 2
        assert {
            "tmax",
            "tmin",
            "tmean",
            "nobs_tmax",
            "nobs_tmin",
            "nobs_tmean",
        } <= set(dataset.data_vars)


def test_partial_day_aligns_variables_with_nan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A day missing one variable still merges, padded with NaN."""
    datasets = {
        "SAMeT_CPTEC_TMAX_20260915.nc": sample_samet_dataset(
            variable="tmax", day=date(2026, 9, 15)
        ),
        "SAMeT_CPTEC_TMAX_20260916.nc": sample_samet_dataset(
            variable="tmax", day=date(2026, 9, 16)
        ),
        "SAMeT_CPTEC_TMIN_20260915.nc": sample_samet_dataset(
            variable="tmin", day=date(2026, 9, 15)
        ),
    }

    def decode(path: Path) -> xr.Dataset:
        return datasets[path.name]

    monkeypatch.setattr("samet.zarr.open_samet_dataset", decode)
    files = tuple(tmp_path / name for name in datasets)
    store = files_to_zarr(files, tmp_path / "partial.zarr")
    with xr.open_zarr(store, consolidated=False) as dataset:
        assert dataset.sizes["time"] == 2
        assert int(dataset["tmax"].notnull().any("time").sum()) > 0
        assert int(dataset["tmin"].notnull().any("time").sum()) > 0
        assert bool(dataset["tmin"].isel(time=1).isnull().all())


def test_write_zarr_refuses_to_overwrite_by_default(tmp_path: Path) -> None:
    """A store is never replaced silently."""
    destination = tmp_path / "store.zarr"
    write_zarr(sample_samet_dataset(), destination)
    assert destination.exists()
    with pytest.raises(FileExistsError):
        write_zarr(sample_samet_dataset(), destination)
    write_zarr(sample_samet_dataset(), destination, overwrite=True)


def test_experiment_downloads_and_writes_an_openable_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full offline path produces a store that opens as a dataset."""
    monkeypatch.setattr("samet.zarr.open_samet_dataset", daily_decoder())
    downloader = RecordingDownloader()
    experiment = Experiment(
        name="probe",
        day=daily_request(),
        downloader=downloader,  # type: ignore[arg-type]
        root_dir=tmp_path,
    )
    events: list[object] = []
    paths = experiment.download(listener=events.append)
    assert len(paths) == 3
    store = experiment.to_zarr(listener=events.append)
    assert store == experiment.store_path
    with experiment.open() as dataset:
        assert {"tmax", "tmin", "tmean"} <= set(dataset.data_vars)
        assert "lat" in dataset.coords
    kinds = {type(event).__name__ for event in events}
    assert {"RequestPlanned", "FileResolved", "StorePlanned", "ItemWritten"} <= kinds


def test_hourly_experiment_writes_an_openable_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An hourly experiment runs the same pipeline end to end."""
    monkeypatch.setattr(
        "samet.zarr.open_samet_dataset",
        daily_decoder(sample_samet_dataset(variable="temperature", hour=probe_hour())),
    )
    experiment = Experiment(
        name="probe",
        hours=hourly_request(),
        downloader=RecordingDownloader(),  # type: ignore[arg-type]
        root_dir=tmp_path,
    )
    experiment.download()
    store = experiment.to_zarr()
    with experiment.open() as dataset:
        assert {"temperature", "nobs"} <= set(dataset.data_vars)
    assert store == experiment.store_path


def test_download_skips_unpublished_cycles_when_asked(tmp_path: Path) -> None:
    """A cycle INPE has not published is skipped, not fatal, when requested."""
    from samet import NoDataAvailableError

    class MissingDownloader:
        def __init__(self) -> None:
            self.listeners: list[object] = []
            self.skipped: list[bool] = []

        def download_chunk(self, chunk, *, listener=None, skip_missing=False):  # type: ignore[no-untyped-def]
            self.listeners.append(listener)
            self.skipped.append(skip_missing)
            raise NoDataAvailableError(f"not published: {chunk.label}")

    experiment = Experiment(
        name="probe",
        day=daily_request(),
        downloader=MissingDownloader(),  # type: ignore[arg-type]
        root_dir=tmp_path,
    )
    with pytest.raises(NoDataAvailableError):
        experiment.download()
    assert experiment.download(skip_missing=True) == ()


def test_experiment_requires_download_before_writing(tmp_path: Path) -> None:
    """Writing a store before downloading is a programming error."""
    experiment = Experiment(
        name="probe",
        day=daily_request(),
        downloader=RecordingDownloader(),  # type: ignore[arg-type]
        root_dir=tmp_path,
    )
    with pytest.raises(RuntimeError, match="download"):
        experiment.to_zarr()
    with pytest.raises(RuntimeError, match="to_zarr"):
        experiment.open()


def test_experiment_cache_key_is_isolated_per_request(tmp_path: Path) -> None:
    """Changing the day yields a different store and cache."""
    base = Experiment(
        name="probe",
        day=daily_request(),
        downloader=RecordingDownloader(),  # type: ignore[arg-type]
        root_dir=tmp_path,
    )
    other = Experiment(
        name="probe",
        day=daily_request(day=date(2026, 9, 16)),
        downloader=RecordingDownloader(),  # type: ignore[arg-type]
        root_dir=tmp_path,
    )
    assert base.cache_key != other.cache_key
    assert base.store_path != other.store_path


def test_experiment_rejects_empty_and_mistyped_requests(tmp_path: Path) -> None:
    """Names and request types are validated at construction."""
    with pytest.raises(SametValidationError, match="empty"):
        Experiment(name="  ", day=daily_request(), root_dir=tmp_path)
    with pytest.raises(SametValidationError, match="named request"):
        Experiment(name="probe", root_dir=tmp_path)
    with pytest.raises(SametValidationError, match="DailyRequests"):
        Experiment(
            name="probe",
            day={"not": "a request"},  # type: ignore[arg-type]
            root_dir=tmp_path,
        )


def test_experiment_rejects_mixed_daily_and_hourly_requests(tmp_path: Path) -> None:
    """Daily and hourly products never share one store."""
    with pytest.raises(SametValidationError, match="all DailyRequests"):
        Experiment(
            name="probe",
            day=daily_request(),
            hour=hourly_request(),
            root_dir=tmp_path,
        )


def test_area_crops_latitudes_and_crosses_the_antimeridian() -> None:
    """Cropping honours the [-180, 180] grid and the seam crossing."""
    dataset = normalize_dataset(sample_samet_dataset())
    inner = Area(south=-20.0, north=0.0, west=-60.0, east=-45.0)
    cropped = apply_area(dataset, inner)
    assert float(cropped["lat"].min()) >= -20.0
    assert float(cropped["lat"].max()) <= 0.0
    assert set(np.round(cropped["lon"].values, 3)) <= {-60.0, -45.0}
    crossing = Area(south=-90.0, north=90.0, west=-45.0, east=-60.0)
    assert crossing.crosses_antimeridian
    wrapped = apply_area(dataset, crossing)
    assert set(np.round(wrapped["lon"].values, 3)) <= set(np.round(LONGITUDES, 3))
