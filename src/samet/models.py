"""Typed request objects for the CPTEC/INPE SAMeT temperature products.

SAMeT (South American Mapping of Temperature, Rozante et al., 2021) blends
station observations with the ERA5 reanalysis, corrected by a regional,
seasonal lapse rate. INPE publishes two collections through its open-data
server: a *daily* product (tmax, tmin, tmean, one file per variable per day,
0.05 degrees, history from 2000-01-01) and an *hourly* product (2 m
temperature, one file per hour, 0.05 degrees, history from 2022-05-01). The
suite's vocabulary mirrors that split: a ``DailyRequest`` names one day, an
``HourlyRequest`` names an inclusive range of whole hours.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from datetime import datetime
from datetime import timedelta
from enum import Enum
from math import isfinite
from typing import TYPE_CHECKING

from samet.exceptions import SametValidationError


if TYPE_CHECKING:
    from collections.abc import Iterable

#: The daily SAMeT history starts 2000-01-01.
SAMET_DAILY_HISTORY_START = date(2000, 1, 1)

#: The hourly SAMeT history starts at the 00Z file of 2022-05-01.
SAMET_HOURLY_HISTORY_START = datetime(2022, 5, 1, 0)


class DailyVariable(Enum):
    """One daily SAMeT variable, in the canonical store vocabulary."""

    TMAX = "tmax"
    TMIN = "tmin"
    TMEAN = "tmean"

    @property
    def file_label(self) -> str:
        """Return the label INPE uses in paths and file names."""
        if self is DailyVariable.TMEAN:
            return "TMED"
        return self.name

    @property
    def long_name(self) -> str:
        """Return a descriptive name for the variable."""
        return {
            DailyVariable.TMAX: "SAMeT daily maximum 2 m temperature",
            DailyVariable.TMIN: "SAMeT daily minimum 2 m temperature",
            DailyVariable.TMEAN: "SAMeT daily mean 2 m temperature",
        }[self]


#: Every variable one daily request downloads.
DAILY_VARIABLES = tuple(DailyVariable)


@dataclass(frozen=True, kw_only=True)
class DailyCycle:
    """One SAMeT daily cycle.

    Parameters
    ----------
    day : datetime.date
        The calendar day in UTC.
    """

    day: date

    def __post_init__(self) -> None:
        """Reject days outside the published history."""
        if not isinstance(self.day, date) or isinstance(self.day, datetime):
            raise SametValidationError("DailyCycle day must be a date.")
        if self.day < SAMET_DAILY_HISTORY_START:
            raise SametValidationError(
                f"SAMeT daily history starts {SAMET_DAILY_HISTORY_START.isoformat()}."
            )
        if self.day > date.today():
            raise SametValidationError("DailyCycle day cannot be in the future.")

    @property
    def stamp(self) -> str:
        """Return the ``YYYYMMDD`` file label."""
        return f"{self.day.year:04d}{self.day.month:02d}{self.day.day:02d}"

    @property
    def reference(self) -> datetime:
        """Return the file's reference instant: 00Z of the calendar day.

        Daily SAMeT files carry their ``time`` coordinate at 00:00 of the
        day the extremes and mean describe. This is the source convention,
        kept as published.
        """
        return datetime(self.day.year, self.day.month, self.day.day)

    def __str__(self) -> str:
        """Return a compact human label such as ``2026-09-15``."""
        return self.day.isoformat()


@dataclass(frozen=True, kw_only=True)
class HourlyCycle:
    """One SAMeT hourly cycle.

    Parameters
    ----------
    instant : datetime.datetime
        The hour the file is stamped with, naive and in UTC. Hourly files
        hold the hour ending at the stamp: the file ``...2026091715.nc``
        describes 14:01-15:00 UTC and is stamped 15:00.
    """

    instant: datetime

    def __post_init__(self) -> None:
        """Reject off-hour or out-of-history instants."""
        if not isinstance(self.instant, datetime):
            raise SametValidationError("HourlyCycle instant must be a datetime.")
        if self.instant.tzinfo is not None:
            raise SametValidationError(
                "HourlyCycle instant must be naive; SAMeT hours are UTC."
            )
        if (
            self.instant.minute != 0
            or self.instant.second != 0
            or self.instant.microsecond != 0
        ):
            raise SametValidationError("HourlyCycle instant must be a whole hour.")
        if self.instant < SAMET_HOURLY_HISTORY_START:
            raise SametValidationError(
                f"SAMeT hourly history starts {SAMET_HOURLY_HISTORY_START.isoformat()}."
            )
        if self.instant > datetime.now():
            raise SametValidationError("HourlyCycle instant cannot be in the future.")

    @property
    def stamp(self) -> str:
        """Return the ``YYYYMMDDHH`` file label."""
        return (
            f"{self.instant.year:04d}{self.instant.month:02d}"
            f"{self.instant.day:02d}{self.instant.hour:02d}"
        )

    @property
    def reference(self) -> datetime:
        """Return the file's reference instant: the stamped hour itself."""
        return self.instant

    def __str__(self) -> str:
        """Return a compact human label such as ``2026-09-17T15``."""
        return f"{self.instant.date().isoformat()}T{self.instant.hour:02d}"


@dataclass(frozen=True, kw_only=True)
class Area:
    """A geographic bounding box applied when the store is written.

    SAMeT files are whole-domain fields; the box is applied locally, at store
    time, and it never changes which bytes are downloaded.

    Parameters
    ----------
    south, north : float
        Latitude bounds in degrees.
    west, east : float
        Longitude bounds in degrees on the source ``[-180, 180]`` convention
        (e.g. Brazil is roughly -75..-34). ``west`` greater than ``east``
        spans the antimeridian.
    """

    south: float
    north: float
    west: float
    east: float

    def __post_init__(self) -> None:
        """Validate finite, ordered bounds on the source convention."""
        values = (self.south, self.north, self.west, self.east)
        if not all(isfinite(value) for value in values):
            raise SametValidationError("Area bounds must be finite.")
        if not -90 <= self.south < self.north <= 90:
            raise SametValidationError(
                "Latitude bounds must satisfy -90 <= south < north <= 90."
            )
        if not -180 <= self.west <= 180 or not -180 <= self.east <= 180:
            raise SametValidationError(
                "Longitude bounds must lie in -180 <= bound <= 180."
            )
        if self.west == self.east:
            raise SametValidationError("Longitude bounds must span some longitude.")

    @property
    def crosses_antimeridian(self) -> bool:
        """Return whether the box spans the ``180`` meridian."""
        return self.west > self.east


@dataclass(frozen=True, kw_only=True)
class DailyRequest:
    """Validated parameters for one SAMeT daily download.

    One request covers one day and always downloads the three daily
    variables (tmax, tmin, tmean), one file each.

    Parameters
    ----------
    day : date
        The calendar day to read.
    area : Area or None, default=None
        Optional spatial subset applied when the store is written.
    """

    day: date
    area: Area | None = None

    def __post_init__(self) -> None:
        """Validate the day against the published history."""
        DailyCycle(day=self.day)

    @property
    def cycle(self) -> DailyCycle:
        """Return the validated daily cycle for the request."""
        return DailyCycle(day=self.day)

    @property
    def stamp(self) -> str:
        """Return the ``YYYYMMDD`` file label."""
        return self.cycle.stamp


@dataclass(frozen=True, kw_only=True)
class HourlyRequest:
    """Validated parameters for an inclusive range of SAMeT hourly files.

    Parameters
    ----------
    start : datetime.datetime
        First hour to read, naive UTC, whole hour.
    end : datetime.datetime
        Last hour to read, naive UTC, whole hour. Equal to ``start`` for a
        single hour.
    area : Area or None, default=None
        Optional spatial subset applied when the store is written.
    """

    start: datetime
    end: datetime
    area: Area | None = None

    def __post_init__(self) -> None:
        """Validate both instants and their ordering."""
        HourlyCycle(instant=self.start)
        HourlyCycle(instant=self.end)
        if self.end < self.start:
            raise SametValidationError("HourlyRequest end must not precede start.")

    @property
    def cycles(self) -> tuple[HourlyCycle, ...]:
        """Return every hourly cycle in the inclusive range, oldest first."""
        return tuple(
            HourlyCycle(instant=hour) for hour in hours_between(self.start, self.end)
        )


def days_between(start: date, end: date) -> Iterable[date]:
    """Yield every day in the inclusive range, oldest first."""
    step = timedelta(days=1)
    day = start
    while day <= end:
        yield day
        day += step


def hours_between(start: datetime, end: datetime) -> Iterable[datetime]:
    """Yield every whole hour in the inclusive range, oldest first."""
    step = timedelta(hours=1)
    hour = start
    while hour <= end:
        yield hour
        hour += step


type SametRequest = DailyRequest | HourlyRequest
"""Union of every request type an experiment accepts."""
