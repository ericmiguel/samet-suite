"""Guided tour of the samet-suite public API, against the live INPE server.

Run from the repository root:

    uv run scripts/demo.py

The script downloads a handful of small files (a few MB in total) into
``.cache/samet/`` and writes Zarr v3 stores under ``data/samet/``. Every step
prints what it does and why, so the file doubles as extended documentation
for the package.
"""

from __future__ import annotations

from datetime import date
from datetime import datetime
from datetime import timedelta

from samet import Area
from samet import DailyRequest
from samet import Experiment
from samet import HourlyRequest
from samet import PipelineEvent
from samet import days_between
from samet import hours_between


def report(event: PipelineEvent) -> None:
    """Print pipeline events; a real caller would render progress bars."""
    print(f"    [event] {event}")


def heading(title: str) -> None:
    """Separate the tour stops."""
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> None:
    """Walk through the daily and hourly products end to end."""
    # ------------------------------------------------------------------
    heading("1. One daily day: tmax, tmin, tmean in a single store")
    # A DailyRequest names one calendar day (UTC) and always downloads the
    # three daily variables, one NetCDF file each (~2.2 MB per file). The
    # Experiment combines any number of named requests into one store; the
    # request names are just labels for events and logs.
    day = date.today() - timedelta(days=3)  # certain to be published
    daily = Experiment(
        name="demo_daily",
        day=DailyRequest(day=day),
    )
    print(f"Cache key: {daily.cache_key[:16]}...  (isolated per request set)")
    print(f"Cache dir: {daily.cache_path}")
    print(f"Store:     {daily.store_path}")
    paths = daily.download(listener=report)
    print(f"Downloaded {len(paths)} files (tmax, tmin, tmean).")
    daily.to_zarr(listener=report)
    dataset = daily.open()
    print(dataset)
    dataset.close()

    # ------------------------------------------------------------------
    heading("2. A daily range with days_between, cropped to an Area")
    # days_between walks an inclusive day range. Area crops at store time —
    # the download is always the whole South America field. Longitudes use
    # the source-native [-180, 180] convention; west > east would cross the
    # antimeridian. This box covers southeastern Brazil.
    southeast = Area(south=-25.0, north=-15.0, west=-50.0, east=-39.0)
    start, end = day - timedelta(days=1), day
    ranged = Experiment(
        name="demo_daily_range",
        **{
            f"day_{stamp}": DailyRequest(day=each, area=southeast)
            for each in days_between(start, end)
            for stamp in [each.strftime("%Y%m%d")]
        },
    )
    ranged.download()
    ranged.to_zarr()
    dataset = ranged.open()
    print(
        "Cropped store:",
        dict(dataset.sizes),
        "lon",
        float(dataset["lon"].min()),
        "..",
        float(dataset["lon"].max()),
    )
    dataset.close()

    # ------------------------------------------------------------------
    heading("3. One hourly day: 24 files, one temperature time axis")
    # An HourlyRequest names an inclusive range of whole hours, naive UTC.
    # Hourly files are stamped at the END of the hour: the file stamped
    # 15:00 covers 14:01-15:00 UTC, and that stamp is the store's time.
    hour = datetime.now().replace(minute=0, second=0, microsecond=0) - timedelta(days=3)
    hourly = Experiment(
        name="demo_hourly",
        day=HourlyRequest(start=hour, end=hour + timedelta(hours=5)),
    )
    hourly.download()
    hourly.to_zarr()
    dataset = hourly.open()
    print(dataset)
    print("Hours:", [str(t)[:13] for t in dataset["time"].values])
    dataset.close()

    # ------------------------------------------------------------------
    heading("4. Ranges with hours_between and rolling updates")
    # hours_between is the hourly analogue of days_between. For a rolling
    # product that includes recent cycles, skip_missing=True turns a
    # not-yet-published hour into a warning instead of an error.
    print(
        "hours_between:",
        [h.isoformat() for h in hours_between(hour, hour + timedelta(hours=2))],
    )
    rolling = Experiment(
        name="demo_rolling",
        window=HourlyRequest(start=hour, end=hour + timedelta(hours=72)),
    )
    paths = rolling.download(skip_missing=True)
    print(f"Downloaded {len(paths)} published hours (unpublished ones skipped).")

    # ------------------------------------------------------------------
    heading("5. Cache identity: change any field, get a new isolated store")
    # The cache key hashes the experiment name plus every request field.
    # Same name, different day -> different store; no cross-contamination.
    other = Experiment(name="demo_daily", day=DailyRequest(day=day - timedelta(days=1)))
    print(f"Original key: {daily.cache_key[:16]}...")
    print(f"Other key:    {other.cache_key[:16]}...  (different day)")


if __name__ == "__main__":
    main()
