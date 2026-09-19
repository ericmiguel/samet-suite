# samet-suite

Data suite (leaf). Downloads the CPTEC/INPE **SAMeT** temperature products —
daily and hourly — from the INPE open-data server, decodes them, and
materializes canonical **Zarr v3** stores. Package `samet`.

SAMeT (South American Mapping of Temperature, Rozante et al., 2021) blends
station observations with the ERA5 reanalysis, corrected by a regional,
seasonal lapse rate estimated over four South American regions. INPE
publishes it in two collections (also browsable through the
[BDC STAC catalog](https://data.inpe.br/stac/browser)):

- **daily** — `tmax`, `tmin`, `tmean`, one NetCDF file per variable per day,
  0.05 degrees (lat -56..13, lon -83..-33), history from 2000-01-01;
- **hourly** — 2 m `temperature`, one NetCDF file per hour, 0.05 degrees
  (lat -35.05..5.95, lon -75.05..-34.05), history from 2022-05-01.

```python
from datetime import date, datetime

from samet import DailyRequest, Experiment, HourlyRequest

daily = Experiment(
    name="samet_daily_sep_2026",
    september_15=DailyRequest(day=date(2026, 9, 15)),
    september_16=DailyRequest(day=date(2026, 9, 16)),
)
daily.download()
daily.to_zarr()
dataset = daily.open()  # lazy xarray dataset: time, lat, lon

hourly = Experiment(
    name="samet_hourly_sep_15",
    day=HourlyRequest(start=datetime(2026, 9, 15, 0), end=datetime(2026, 9, 15, 23)),
)
hourly.download()
hourly.to_zarr()
dataset = hourly.open()  # hourly time axis: temperature, nobs
```

## Data and vocabulary

| concept | values | note |
| --- | --- | --- |
| `DailyRequest` | `day` + optional `Area` | one request is one day and downloads all three daily variables |
| `HourlyRequest` | inclusive `start`/`end` hours + optional `Area` | one request is a range of whole hours, naive UTC |
| daily variables | `tmax`, `tmin`, `tmean`, `nobs_tmax`, `nobs_tmin`, `nobs_tmean` | temperatures in degC, station counts per grid point |
| hourly variables | `temperature`, `nobs` | 2 m temperature in degC, station count per grid point |

The NetCDF vocabulary differs from the store vocabulary: the daily mean
decodes as `tmed` and the hourly temperature as `tt2m`; every file also
carries an `nobs` field (stations feeding the analysis) that differs between
the daily variables, so the suite renames them `nobs_tmax` / `nobs_tmin` /
`nobs_tmean` to keep the three daily fields in one store without collisions.
Daily files are stamped 00Z of the calendar day; hourly files are stamped at
the *end* of the hour they describe (file `...1715.nc` covers 14:01-15:00
UTC). Both conventions are kept as published.

Longitude stays source-native `[-180, 180]`; `Area` with `west > east` spans
the antimeridian and is applied at store time only — never on the server.
The fields are masked where no analysis exists (`_FillValue` -9.99e8), which
xarray surfaces as NaN.

An experiment is homogeneous: every named request is either a
`DailyRequest` or every one is an `HourlyRequest`. The daily and hourly
grids, time axes and variables do not line up, so mixing them in one store
is rejected at construction.

## The download economics

A daily file is ~2.2 MB and an hourly file ~1 MB, so the suite downloads
whole files — there is nothing to extract. A cycle is immutable once
published, but a cycle inside the last two days is re-checked against the
remote `Content-Length` before the cache is trusted.

Verification refuses a file unless it decodes, carries the planned
variables, has finite values, keeps station counts non-negative, and stays
inside the temperatures a 2 m field can reach (-100..70 degC, far wider than
the observed -40..48).

## Missing data

A cycle INPE has not published yet is `NoDataAvailableError`; with
`download(skip_missing=True)` it is skipped with a warning, which is how a
rolling range (`days_between(...)`, `hours_between(...)`) can include today
without failing.

## Store shape

- Daily store: coordinates `time` (00Z of the day, the source convention),
  `lat`, `lon`; variables `tmax`, `tmin`, `tmean` (degC) plus one
  `nobs_<variable>` station count each.
- Hourly store: coordinates `time` (hour end, the source convention), `lat`,
  `lon`; variables `temperature` (degC) and `nobs`.
- Fragments: source-global pool `.cache/fragments/samet/v2/`, shared by every
  experiment that plans the same cycle.
- Store: `.cache/stores/samet/<name>/<fingerprint>.zarr` with a
  `manifest.json` beside it. The `name` is a validated slug and the
  `fingerprint` hashes the request fields (never the name), so changing any
  request field (day, hours, area) yields a new store over the same fragments.

## Guided demo

`scripts/demo.py` is a runnable tour of the public API — daily and hourly
experiments, day/hour ranges, `Area` cropping, rolling updates with
`skip_missing`, and cache-key isolation — against the live INPE server:

```bash
uv run scripts/demo.py
```

## Quality

```bash
uv run ruff check . ; uv run ruff format . --check ; uv run pyrefly check ; uv run pytest ; uv run pytest -m network
```

The network smoke test downloads one published daily day and one published
hourly hour (both certain to exist) and verifies physically plausible
stores.
