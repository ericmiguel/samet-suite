"""Canonical filtering and atomic Zarr v3 writing for SAMeT stores."""

from __future__ import annotations

from pathlib import Path
from shutil import rmtree
from typing import TYPE_CHECKING

import xarray as xr

from samet.exceptions import MissingCoordinateError
from samet.exceptions import SametValidationError
from samet.netcdf import open_samet_dataset


if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from samet.models import Area
    from samet.models import SametRequest


def normalize_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """Normalize coordinate order and the time axis.

    SAMeT files already carry one-dimensional ``lat``/``lon`` coordinates on
    the source-native ``[-180, 180]`` convention and a length-one ``time``
    axis; a scalar time is still promoted so one store serves one cycle or a
    decade.
    """
    normalized = _promote_time(dataset)
    _require_spatial_coordinates(normalized)
    for coordinate in ("lat", "lon", "time"):
        coordinate_variable = normalized.coords.get(coordinate)
        if coordinate_variable is not None and coordinate_variable.ndim == 1:
            normalized = normalized.sortby(coordinate)
    return normalized


def _promote_time(dataset: xr.Dataset) -> xr.Dataset:
    """Turn the scalar reference time into a length-one dimension."""
    if "time" in dataset.coords and dataset["time"].ndim == 0:
        return dataset.expand_dims("time")
    return dataset


def apply_area(dataset: xr.Dataset, area: Area) -> xr.Dataset:
    """Crop a normalized dataset to an area on the ``[-180, 180]`` grid."""
    cropped = dataset.sel(lat=slice(area.south, area.north))
    longitude = cropped["lon"]
    if area.east - area.west >= 360:
        return cropped
    if area.west <= area.east:
        mask = (longitude >= area.west) & (longitude <= area.east)
    else:
        mask = (longitude >= area.west) | (longitude <= area.east)
    return cropped.where(mask, drop=True)


def write_zarr(
    dataset: xr.Dataset,
    destination: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a normalized dataset to an atomic Zarr v3 directory."""
    destination = Path(destination)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    backup = destination.with_name(f".{destination.name}.old")
    if temporary.exists():
        _remove_path(temporary)
    if backup.exists():
        _remove_path(backup)
    try:
        normalized = normalize_dataset(dataset)
        normalized.to_zarr(
            temporary,
            mode="w",
            zarr_format=3,
            consolidated=False,
            encoding=_encoding(normalized),
        )
        if destination.exists():
            destination.rename(backup)
        try:
            temporary.rename(destination)
        except OSError:
            if backup.exists():
                backup.rename(destination)
            raise
        _remove_path(backup)
    finally:
        if temporary.exists():
            _remove_path(temporary)
    return destination


def files_to_zarr(
    sources: Iterable[Path],
    destination: Path,
    *,
    overwrite: bool = False,
    request_sources: Sequence[tuple[SametRequest, Iterable[Path]]] | None = None,
) -> Path:
    """Decode, crop, combine, and write SAMeT files as one Zarr store.

    Parameters
    ----------
    sources : iterable of pathlib.Path
        SAMeT NetCDF files, used when no request grouping is supplied.
    destination : pathlib.Path
        Directory of the Zarr v3 store.
    overwrite : bool, default=False
        Whether an existing store may be replaced.
    request_sources : sequence of tuple, optional
        ``(request, paths)`` pairs. Each request's own area filter is applied
        before the files are combined.
    """
    groups = tuple(request_sources or ())
    if not groups:
        paths = tuple(sources)
        if not paths:
            raise ValueError("At least one SAMeT file is required.")
        return write_zarr(_combine(_decode(paths)), destination, overwrite=overwrite)
    datasets: list[xr.Dataset] = []
    for request, paths in groups:
        request_paths = tuple(paths)
        if not request_paths:
            raise ValueError("A request group must carry at least one SAMeT file.")
        for dataset in _decode(request_paths):
            cropped = (
                apply_area(dataset, request.area)
                if request.area is not None
                else dataset
            )
            datasets.append(cropped)
    return write_zarr(_combine(tuple(datasets)), destination, overwrite=overwrite)


def _decode(paths: Iterable[Path]) -> tuple[xr.Dataset, ...]:
    """Decode every SAMeT file, normalized."""
    decoded = tuple(normalize_dataset(open_samet_dataset(path)) for path in paths)
    if not decoded:
        raise ValueError("At least one SAMeT file is required.")
    return decoded


def _combine(datasets: Sequence[xr.Dataset]) -> xr.Dataset:
    """Combine decoded cycles along the time axis, then by variables.

    SAMeT cycles are independent files carrying the same variables, so
    datasets with the same variable signature are concatenated along ``time``
    and the resulting cubes are merged.
    """
    if not datasets:
        raise ValueError("At least one SAMeT dataset is required.")
    if len(datasets) == 1:
        return datasets[0]
    groups: dict[tuple[str, ...], list[xr.Dataset]] = {}
    for dataset in datasets:
        signature = tuple(sorted(str(name) for name in dataset.data_vars))
        groups.setdefault(signature, []).append(dataset)
    combined: list[xr.Dataset] = []
    for members in groups.values():
        if len(members) == 1:
            combined.append(members[0])
            continue
        stacked = xr.concat(
            members,
            dim="time",
            compat="override",
            coords="minimal",
            combine_attrs="override",
            join="outer",
        )
        combined.append(stacked.sortby("time"))
    if len(combined) == 1:
        return combined[0]
    return xr.merge(combined, compat="override", combine_attrs="override", join="outer")


def _require_spatial_coordinates(dataset: xr.Dataset) -> None:
    """Require one-dimensional canonical latitude and longitude coordinates."""
    for name, minimum, maximum in (("lat", -90, 90), ("lon", -180, 180)):
        if name not in dataset.coords:
            raise MissingCoordinateError(f"Dataset is missing {name!r} coordinate.")
        coordinate = dataset[name]
        if coordinate.ndim != 1:
            raise SametValidationError(f"Coordinate {name!r} must be one-dimensional.")
        if float(coordinate.min()) < minimum or float(coordinate.max()) > maximum:
            raise SametValidationError(
                f"Coordinate {name!r} is outside the SAMeT longitude convention."
            )


def _encoding(dataset: xr.Dataset) -> dict[str, dict[str, tuple[int, ...]]]:
    """Derive conservative time and spatial chunk sizes."""
    chunks: dict[str, int] = {}
    for raw_dimension, size in dataset.sizes.items():
        dimension = str(raw_dimension)
        if dimension in {"lat", "lon"}:
            chunks[dimension] = size or 1
        elif dimension == "time":
            chunks[dimension] = 1
        else:
            chunks[dimension] = min(size, 8) or 1
    return {
        str(name): {
            "chunks": tuple(chunks[str(dimension)] for dimension in variable.dims)
        }
        for name, variable in dataset.data_vars.items()
        if variable.dims
    }


def _remove_path(path: Path) -> None:
    """Remove a temporary or replaced Zarr path."""
    if path.is_dir():
        rmtree(path)
    else:
        path.unlink(missing_ok=True)
