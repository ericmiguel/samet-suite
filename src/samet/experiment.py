"""Experiment orchestration for named SAMeT requests."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import xarray as xr

from samet.cache import experiment_cache_dir
from samet.cache import experiment_cache_key
from samet.cache import experiment_store_path
from samet.chunking import plan_chunks
from samet.events import FileResolved
from samet.events import ItemWritten
from samet.events import PipelineListener
from samet.events import RequestPlanned
from samet.events import StorePlanned
from samet.exceptions import NoDataAvailableError
from samet.exceptions import SametValidationError
from samet.models import DailyRequest
from samet.models import HourlyRequest
from samet.retrieval import SametDownloader
from samet.root import resolve_project_root
from samet.zarr import files_to_zarr


if TYPE_CHECKING:
    from pathlib import Path

    from samet.models import SametRequest


class Experiment:
    """Combine compatible named SAMeT requests into one Zarr v3 store.

    An experiment is homogeneous: every named request is either a
    ``DailyRequest`` or every one is an ``HourlyRequest``. Mixing the two
    products in one store is rejected, since their grids, time axes and
    variables do not line up.

    Parameters
    ----------
    name : str
        Experiment name used in progress descriptions and store metadata.
    downloader : SametDownloader or None, default=None
        Optional downloader, normally injected with a fake in tests.
    root_dir : pathlib.Path or None, default=None
        Project root used for the hidden cache and data directories. When
        omitted, it is discovered from ``.git``, ``.venv``, or ``README.md``.
    requests : DailyRequest or HourlyRequest
        One or more named requests supplied as keyword arguments.
    """

    def __init__(
        self,
        *,
        name: str,
        downloader: SametDownloader | None = None,
        root_dir: Path | None = None,
        **requests: SametRequest,
    ) -> None:
        if not name.strip():
            raise SametValidationError("Experiment name cannot be empty.")
        if not requests:
            raise SametValidationError("At least one named request is required.")
        kinds = {type(request) for request in requests.values()}
        if not kinds <= {DailyRequest, HourlyRequest} or len(kinds) != 1:
            raise SametValidationError(
                "Experiment requests must be all DailyRequests or all HourlyRequests."
            )
        self.name = name
        self.requests = dict(requests)
        self._cache_key = experiment_cache_key(name, self.requests)
        self.root_dir = resolve_project_root(root_dir)
        self.downloader = downloader or SametDownloader()
        self._paths: tuple[Path, ...] = ()
        self._request_paths: dict[str, tuple[Path, ...]] = {}
        self._store_path: Path | None = None
        self._logger = logging.getLogger(__name__)

    @property
    def cache_key(self) -> str:
        """Return the isolated cache key for this experiment."""
        return self._cache_key

    @property
    def cache_path(self) -> Path:
        """Return the isolated cache directory for this experiment."""
        return experiment_cache_dir(self.root_dir / ".cache", self.cache_key)

    @property
    def store_path(self) -> Path:
        """Return the canonical Zarr v3 path for this experiment."""
        return experiment_store_path(self.root_dir / "data", self.cache_key)

    def plan(self, request_name: str) -> tuple[str, int, int]:
        """Return ``(files, messages, days)`` for one named request."""
        request = self.requests[request_name]
        plan = plan_chunks(request, self.cache_path, request_name=request_name)
        return (request_name, len(plan.chunks), plan.message_count)

    def download(
        self,
        *,
        listener: PipelineListener | None = None,
        skip_missing: bool = False,
    ) -> tuple[Path, ...]:
        """Download every planned file and return deduplicated paths.

        Parameters
        ----------
        listener : PipelineListener or None, default=None
            Optional subscriber receiving planned, resolved and byte-transfer
            events.
        skip_missing : bool, default=False
            Whether a cycle INPE has not published yet is skipped with a
            warning instead of raising.

        Returns
        -------
        tuple of pathlib.Path
            Verified SAMeT files in request and cycle order.
        """
        paths: list[Path] = []
        seen: set[Path] = set()
        for request_name, request in self.requests.items():
            plan = plan_chunks(request, self.cache_path, request_name=request_name)
            if listener is not None:
                listener(
                    RequestPlanned(
                        name=request_name,
                        files=len(plan.chunks),
                        messages=plan.message_count,
                    )
                )
            request_paths: list[Path] = []
            for chunk in plan.chunks:
                if chunk.path not in seen:
                    try:
                        self.downloader.download_chunk(
                            chunk, listener=listener, skip_missing=skip_missing
                        )
                    except NoDataAvailableError as error:
                        if not skip_missing:
                            raise
                        self._logger.warning("Skipping %s: %s", chunk.label, error)
                        continue
                    paths.append(chunk.path)
                    seen.add(chunk.path)
                request_paths.append(chunk.path)
                if listener is not None:
                    listener(FileResolved(request=request_name, path=chunk.path))
            self._request_paths[request_name] = tuple(request_paths)
        self._paths = tuple(paths)
        self._logger.info("Downloaded %d files for %s.", len(paths), self.name)
        return self._paths

    def to_zarr(
        self,
        *,
        overwrite: bool = False,
        listener: PipelineListener | None = None,
    ) -> Path:
        """Decode, crop, and write the experiment's Zarr store.

        Parameters
        ----------
        overwrite : bool, default=False
            Whether an existing destination may be replaced.
        listener : PipelineListener or None, default=None
            Optional subscriber receiving store-writing events.

        Returns
        -------
        pathlib.Path
            The written Zarr directory.
        """
        if not self._paths:
            raise RuntimeError("Call download() before to_zarr().")
        if listener is not None:
            listener(StorePlanned(items=1))
        groups = tuple(
            (self.requests[name], self._request_paths[name]) for name in self.requests
        )
        destination = files_to_zarr(
            self._paths,
            self.store_path,
            overwrite=overwrite,
            request_sources=groups,
        )
        self._store_path = destination
        if listener is not None:
            listener(ItemWritten(description=str(destination)))
        self._logger.info("Wrote Zarr store %s.", destination)
        return destination

    def open(self) -> xr.Dataset:
        """Open the experiment's Zarr v3 store lazily."""
        if self._store_path is None:
            raise RuntimeError("Call to_zarr() before open().")
        return xr.open_zarr(self._store_path, consolidated=False)
