"""NetCDF decoding of SAMeT files.

Decoding lives here so both the downloader (which verifies what it fetched)
and the store writer (which converts it) use exactly one code path.

INPE's file vocabulary differs from the canonical store vocabulary: the
hourly temperature decodes as ``tt2m`` and the daily mean as ``tmed``, and
every file carries an ``nobs`` field (number of stations feeding the
analysis) that differs between the daily variables. This module translates
to the canonical names on the way out: ``temperature``, ``tmean`` and one
``nobs_<variable>`` per daily variable so the three daily fields never
collide in a store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

from samet.exceptions import DownloadError


if TYPE_CHECKING:
    from pathlib import Path

#: Fields larger than this are treated as undecoded garbage rather than data.
PLAUSIBLE_MAGNITUDE = 1.0e30

#: Plausible 2 m temperature bounds in degrees Celsius. The observed SAMeT
#: range is roughly -40..48; the bounds stay generous on purpose.
MIN_TEMPERATURE = -100.0
MAX_TEMPERATURE = 70.0

#: Source temperature names mapped onto the canonical store vocabulary.
_TEMPERATURE_MAP = {
    "tmax": "tmax",
    "tmin": "tmin",
    "tmed": "tmean",
    "tt2m": "temperature",
}

_LONG_NAMES = {
    "tmax": "SAMeT daily maximum 2 m temperature",
    "tmin": "SAMeT daily minimum 2 m temperature",
    "tmean": "SAMeT daily mean 2 m temperature",
    "temperature": "SAMeT hourly 2 m temperature",
}


def open_samet_dataset(path: Path) -> xr.Dataset:
    """Return the dataset of one SAMeT NetCDF file, decoded and renamed.

    Parameters
    ----------
    path : pathlib.Path
        A SAMeT NetCDF file (daily variable or hourly).

    Returns
    -------
    xarray.Dataset
        The loaded dataset with canonical variable names.

    Raises
    ------
    DownloadError
        If the file cannot be decoded or holds no temperature variable.
    """
    try:
        dataset = xr.open_dataset(path)
    except (OSError, ValueError, RuntimeError) as error:
        raise DownloadError(f"Could not decode SAMeT file {path}") from error
    try:
        renamed = _rename(dataset)
        return _versioned(renamed).load()
    except (OSError, ValueError, RuntimeError) as error:
        raise DownloadError(f"Could not read SAMeT values from {path}") from error
    finally:
        dataset.close()


def _rename(dataset: xr.Dataset) -> xr.Dataset:
    """Translate source variable names to the canonical vocabulary.

    Every SAMeT file carries exactly one temperature variable plus an
    ``nobs`` station count. The hourly ``nobs`` keeps its name; a daily
    ``nobs`` is suffixed with its variable so tmax, tmin and tmean can share
    one store.
    """
    temperatures = [
        str(name) for name in dataset.data_vars if str(name) in _TEMPERATURE_MAP
    ]
    unexpected = (
        {str(name) for name in dataset.data_vars} - set(_TEMPERATURE_MAP) - {"nobs"}
    )
    if unexpected:
        raise DownloadError(f"Unexpected variables {sorted(unexpected)} in SAMeT file.")
    if len(temperatures) != 1:
        raise DownloadError(
            f"SAMeT file must carry one temperature variable; "
            f"found {sorted(temperatures)}."
        )
    (source,) = temperatures
    target = _TEMPERATURE_MAP[source]
    mapping: dict[str, str] = {}
    if source != target:
        mapping[source] = target
    if "nobs" in dataset.data_vars:
        nobs_name = "nobs" if target == "temperature" else f"nobs_{target}"
        mapping["nobs"] = nobs_name
    return dataset.rename(mapping) if mapping else dataset


def _versioned(dataset: xr.Dataset) -> xr.Dataset:
    """Attach canonical units and names to the SAMeT variables."""
    for raw_name in dataset.data_vars:
        name = str(raw_name)
        if name in _LONG_NAMES:
            dataset[name].attrs["units"] = "degC"
            dataset[name].attrs["long_name"] = _LONG_NAMES[name]
        elif name.startswith("nobs"):
            dataset[name].attrs["units"] = "1"
            dataset[name].attrs["long_name"] = (
                "Number of stations per grid point in the SAMeT analysis"
            )
    return dataset
