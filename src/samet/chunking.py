"""Deterministic file planning for the CPTEC/INPE SAMeT products."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from samet.exceptions import SametValidationError
from samet.models import DAILY_VARIABLES
from samet.models import DailyRequest
from samet.models import DailyVariable
from samet.models import HourlyRequest


if TYPE_CHECKING:
    from samet.models import Area
    from samet.models import DailyCycle
    from samet.models import HourlyCycle
    from samet.models import SametRequest

SAMET_DAILY_BASE = "https://data.inpe.br/bdc/data/samet_daily/DAILY"
SAMET_HOURLY_BASE = "https://data.inpe.br/bdc/data/samet_hourly/HOURLY"

#: A cycle published within this window can still be receiving corrections.
MUTABLE_WINDOW = timedelta(days=2)


@dataclass(frozen=True, kw_only=True)
class Chunk:
    """One SAMeT remote file and its cache location.

    Parameters
    ----------
    url : str
        Remote NetCDF file.
    path : pathlib.Path
        Cache destination of the downloaded file.
    request_name : str
        Name of the request the chunk belongs to.
    cycle : DailyCycle or HourlyCycle
        The cycle the file describes.
    variables : tuple of str
        Canonical variable names the file must decode to (e.g. ``tmax`` and
        ``nobs_tmax``); download verification checks them.
    area : Area or None
        Area filter, applied when the store is written.
    mutable : bool
        Whether the remote file may still be replaced by INPE.
    """

    url: str
    path: Path
    request_name: str
    cycle: DailyCycle | HourlyCycle
    variables: tuple[str, ...]
    area: Area | None = None
    mutable: bool = False

    @property
    def label(self) -> str:
        """Return a compact label for events and logs."""
        primary = self.variables[0] if self.variables else "file"
        return f"samet {primary} {self.cycle}"


@dataclass(frozen=True, kw_only=True)
class ChunkPlan:
    """Immutable deterministic chunks for one request."""

    request: SametRequest
    cache_dir: Path
    chunks: tuple[Chunk, ...]

    @property
    def message_count(self) -> int:
        """Return how many files the plan extracts."""
        return len(self.chunks)


def plan_chunks(
    request: SametRequest,
    cache_dir: Path,
    *,
    request_name: str = "request",
    now: datetime | None = None,
) -> ChunkPlan:
    """Plan the SAMeT files for one request.

    Parameters
    ----------
    request : SametRequest
        Validated daily or hourly request.
    cache_dir : pathlib.Path
        Cache directory of the owning experiment.
    request_name : str, default="request"
        Name used in events and logs.
    now : datetime.datetime, optional
        Reference instant for the mutability window; ``datetime.now()`` when
        omitted.
    """
    reference = now or datetime.now()
    if isinstance(request, DailyRequest):
        chunks = _daily_chunks(request, Path(cache_dir), request_name, reference)
    elif isinstance(request, HourlyRequest):
        chunks = _hourly_chunks(request, Path(cache_dir), request_name, reference)
    else:
        raise SametValidationError(
            "plan_chunks requires a DailyRequest or HourlyRequest."
        )
    return ChunkPlan(request=request, cache_dir=Path(cache_dir), chunks=chunks)


def _daily_chunks(
    request: DailyRequest,
    cache_dir: Path,
    request_name: str,
    now: datetime,
) -> tuple[Chunk, ...]:
    """Plan one file per daily variable for the request's day."""
    cycle = request.cycle
    return tuple(
        Chunk(
            url=_daily_url(cycle, variable),
            path=cache_dir / f"SAMeT_CPTEC_{variable.file_label}_{cycle.stamp}.nc",
            request_name=request_name,
            cycle=cycle,
            variables=(variable.value, f"nobs_{variable.value}"),
            area=request.area,
            mutable=(now.date() - request.day) <= MUTABLE_WINDOW,
        )
        for variable in DAILY_VARIABLES
    )


def _hourly_chunks(
    request: HourlyRequest,
    cache_dir: Path,
    request_name: str,
    now: datetime,
) -> tuple[Chunk, ...]:
    """Plan one file per hour in the request's range."""
    return tuple(
        Chunk(
            url=_hourly_url(cycle),
            path=cache_dir / f"SAMeT_CPTEC_{cycle.stamp}.nc",
            request_name=request_name,
            cycle=cycle,
            variables=("temperature", "nobs"),
            area=request.area,
            mutable=(now - cycle.instant) <= MUTABLE_WINDOW,
        )
        for cycle in request.cycles
    )


def _daily_url(cycle: DailyCycle, variable: DailyVariable) -> str:
    """Return the published URL of one daily variable file."""
    day = cycle.day
    return (
        f"{SAMET_DAILY_BASE}/{variable.file_label}/{day.year:04d}/{day.month:02d}"
        f"/SAMeT_CPTEC_{variable.file_label}_{cycle.stamp}.nc"
    )


def _hourly_url(cycle: HourlyCycle) -> str:
    """Return the published URL of one hourly file."""
    instant = cycle.instant
    return (
        f"{SAMET_HOURLY_BASE}/{instant.year:04d}/{instant.month:02d}"
        f"/{instant.day:02d}/SAMeT_CPTEC_{cycle.stamp}.nc"
    )


def plan_summary(plan: ChunkPlan) -> str:
    """Return a one-line summary of a chunk plan."""
    return f"{len(plan.chunks)} files, {plan.message_count} messages"
