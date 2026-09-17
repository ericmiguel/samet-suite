"""Boundary validation of the typed request objects."""

from __future__ import annotations

from datetime import date
from datetime import datetime
from datetime import timedelta

import pytest

from samet import SAMET_DAILY_HISTORY_START
from samet import SAMET_HOURLY_HISTORY_START
from samet import Area
from samet import DailyCycle
from samet import DailyRequest
from samet import DailyVariable
from samet import HourlyCycle
from samet import HourlyRequest
from samet import SametValidationError
from samet import days_between
from samet import hours_between


def test_daily_cycle_stamps_and_references_the_day() -> None:
    """The file label is YYYYMMDD and the reference sits at 00Z."""
    cycle = DailyCycle(day=date(2026, 9, 15))
    assert cycle.stamp == "20260915"
    assert cycle.reference == datetime(2026, 9, 15, 0)
    assert str(cycle) == "2026-09-15"


def test_daily_cycle_rejects_days_outside_the_history() -> None:
    """The daily product starts 2000-01-01 and never reaches the future."""
    with pytest.raises(SametValidationError, match="history starts"):
        DailyCycle(day=SAMET_DAILY_HISTORY_START - timedelta(days=1))
    with pytest.raises(SametValidationError, match="future"):
        DailyCycle(day=date.today() + timedelta(days=1))


def test_hourly_cycle_stamps_and_references_the_hour() -> None:
    """The file label is YYYYMMDDHH and the reference is the stamp itself."""
    cycle = HourlyCycle(instant=datetime(2026, 9, 17, 15))
    assert cycle.stamp == "2026091715"
    assert cycle.reference == datetime(2026, 9, 17, 15)
    assert str(cycle) == "2026-09-17T15"


def test_hourly_cycle_requires_whole_naive_hours() -> None:
    """Minutes, seconds and time zones are rejected at the boundary."""
    from datetime import UTC

    with pytest.raises(SametValidationError, match="whole hour"):
        HourlyCycle(instant=datetime(2026, 9, 17, 15, 30))
    with pytest.raises(SametValidationError, match="naive"):
        HourlyCycle(instant=datetime(2026, 9, 17, 15, tzinfo=UTC))


def test_hourly_cycle_rejects_instants_outside_the_history() -> None:
    """The hourly product starts 2022-05-01 00Z and never reaches the future."""
    with pytest.raises(SametValidationError, match="history starts"):
        HourlyCycle(instant=SAMET_HOURLY_HISTORY_START - timedelta(hours=1))
    with pytest.raises(SametValidationError, match="future"):
        future = datetime.now().replace(minute=0, second=0, microsecond=0)
        HourlyCycle(instant=future + timedelta(days=1))


def test_area_validates_bounds_on_the_signed_convention() -> None:
    """SAMeT longitudes live in [-180, 180]; the antimeridian can be crossed."""
    with pytest.raises(SametValidationError, match="finite"):
        Area(south=float("nan"), north=0.0, west=-75.0, east=-34.0)
    with pytest.raises(SametValidationError, match="south < north"):
        Area(south=10.0, north=0.0, west=-75.0, east=-34.0)
    with pytest.raises(SametValidationError, match="-180 <= bound <= 180"):
        Area(south=-30.0, north=0.0, west=240.0, east=300.0)
    with pytest.raises(SametValidationError, match="span some longitude"):
        Area(south=-30.0, north=0.0, west=-50.0, east=-50.0)
    assert not Area(south=-30.0, north=0.0, west=-75.0, east=-34.0).crosses_antimeridian
    assert Area(south=-30.0, north=0.0, west=170.0, east=-170.0).crosses_antimeridian


def test_daily_request_wraps_a_validated_cycle() -> None:
    """The request validates the day and exposes the stamp."""
    request = DailyRequest(day=date(2026, 9, 15))
    assert request.stamp == "20260915"
    assert request.cycle.day == date(2026, 9, 15)
    with pytest.raises(SametValidationError, match="history starts"):
        DailyRequest(day=date(1999, 12, 31))


def test_hourly_request_expands_to_inclusive_cycles() -> None:
    """A range request covers every whole hour, oldest first."""
    request = HourlyRequest(
        start=datetime(2026, 9, 15, 22), end=datetime(2026, 9, 16, 1)
    )
    stamps = [cycle.stamp for cycle in request.cycles]
    assert stamps == ["2026091522", "2026091523", "2026091600", "2026091601"]
    single = HourlyRequest(
        start=datetime(2026, 9, 15, 12), end=datetime(2026, 9, 15, 12)
    )
    assert len(single.cycles) == 1


def test_hourly_request_rejects_an_inverted_range() -> None:
    """The end hour must not precede the start hour."""
    with pytest.raises(SametValidationError, match="must not precede"):
        HourlyRequest(start=datetime(2026, 9, 16, 0), end=datetime(2026, 9, 15, 23))


def test_days_and_hours_between_yield_inclusive_ranges() -> None:
    """Both generators walk the inclusive range, oldest first."""
    days = list(days_between(date(2026, 9, 14), date(2026, 9, 16)))
    assert days == [date(2026, 9, 14), date(2026, 9, 15), date(2026, 9, 16)]
    hours = list(hours_between(datetime(2026, 9, 15, 23), datetime(2026, 9, 16, 1)))
    assert hours == [
        datetime(2026, 9, 15, 23),
        datetime(2026, 9, 16, 0),
        datetime(2026, 9, 16, 1),
    ]


def test_daily_variable_file_labels_follow_the_source_vocabulary() -> None:
    """The daily mean is TMED in file names and tmean in the store."""
    assert DailyVariable.TMAX.file_label == "TMAX"
    assert DailyVariable.TMIN.file_label == "TMIN"
    assert DailyVariable.TMEAN.file_label == "TMED"
    assert DailyVariable.TMEAN.value == "tmean"
