"""NetCDF decoding: source vocabulary translation and verification."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

from conftest import LATITUDES
from conftest import LONGITUDES
from conftest import probe_day
from conftest import probe_hour
from conftest import write_source_file
from samet import DownloadError
from samet import open_samet_dataset


if TYPE_CHECKING:
    from pathlib import Path


def test_hourly_file_decodes_to_canonical_names(tmp_path: Path) -> None:
    """``tt2m`` becomes ``temperature`` and ``nobs`` keeps its name."""
    path = write_source_file(tmp_path / "hourly.nc", variable="tt2m", hour=probe_hour())
    dataset = open_samet_dataset(path)
    assert set(dataset.data_vars) == {"temperature", "nobs"}
    assert dataset["temperature"].attrs["units"] == "degC"
    assert dataset["nobs"].attrs["units"] == "1"
    assert dataset.sizes["time"] == 1


def test_daily_mean_file_decodes_to_canonical_names(tmp_path: Path) -> None:
    """``tmed`` becomes ``tmean`` and ``nobs`` is suffixed per variable."""
    path = write_source_file(tmp_path / "tmed.nc", variable="tmed", day=probe_day())
    dataset = open_samet_dataset(path)
    assert set(dataset.data_vars) == {"tmean", "nobs_tmean"}


def test_daily_extreme_files_keep_the_variable_name(tmp_path: Path) -> None:
    """``tmax`` and ``tmin`` are canonical already; only ``nobs`` is suffixed."""
    tmax = open_samet_dataset(write_source_file(tmp_path / "tmax.nc", variable="tmax"))
    tmin = open_samet_dataset(write_source_file(tmp_path / "tmin.nc", variable="tmin"))
    assert set(tmax.data_vars) == {"tmax", "nobs_tmax"}
    assert set(tmin.data_vars) == {"tmin", "nobs_tmin"}


def test_decoder_rejects_files_without_a_temperature_variable(tmp_path: Path) -> None:
    """A NetCDF with no SAMeT temperature field is refused."""
    dataset = xr.Dataset(
        data_vars={"nobs": (("lat", "lon"), np.ones((4, 4)))},
        coords={"lat": LATITUDES, "lon": LONGITUDES},
    )
    path = tmp_path / "empty.nc"
    dataset.to_netcdf(path)
    with pytest.raises(DownloadError, match="temperature variable"):
        open_samet_dataset(path)


def test_decoder_rejects_unexpected_variables(tmp_path: Path) -> None:
    """A NetCDF carrying unknown fields is refused."""
    path = write_source_file(tmp_path / "hourly.nc", variable="tt2m")
    with xr.open_dataset(path) as dataset:
        dataset.assign(sst=dataset["tt2m"]).to_netcdf(tmp_path / "extra.nc")
    with pytest.raises(DownloadError, match="Unexpected variables"):
        open_samet_dataset(tmp_path / "extra.nc")


def test_decoder_rejects_an_undecodable_file(tmp_path: Path) -> None:
    """Garbage bytes surface as a download error, not a decode traceback."""
    path = tmp_path / "garbage.nc"
    path.write_bytes(b"not a netcdf")
    with pytest.raises(DownloadError, match="decode"):
        open_samet_dataset(path)
