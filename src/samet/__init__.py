"""Typed access to the CPTEC/INPE SAMeT temperature products.

SAMeT (South American Mapping of Temperature, Rozante et al., 2021) blends
station observations with the ERA5 reanalysis, corrected by a regional,
seasonal lapse rate. INPE publishes a daily product (tmax, tmin, tmean, one
NetCDF file per variable per day, 0.05 degrees, history from 2000-01-01) and
an hourly product (2 m temperature, one NetCDF file per hour, 0.05 degrees,
history from 2022-05-01).

```python
from datetime import date
from datetime import datetime

from samet import DailyRequest, Experiment, HourlyRequest

daily = Experiment(
    name="samet_daily_sep_2026",
    september_15=DailyRequest(day=date(2026, 9, 15)),
    september_16=DailyRequest(day=date(2026, 9, 16)),
)
daily.download()
daily.to_zarr()
dataset = daily.open()  # time, lat, lon; tmax, tmin, tmean + nobs_*

hourly = Experiment(
    name="samet_hourly_sep_15",
    day=HourlyRequest(start=datetime(2026, 9, 15, 0), end=datetime(2026, 9, 15, 23)),
)
hourly.download()
hourly.to_zarr()
dataset = hourly.open()  # hourly time axis; temperature + nobs
```
"""

from samet.cache import experiment_cache_dir
from samet.cache import experiment_cache_key
from samet.cache import experiment_store_path
from samet.chunking import SAMET_DAILY_BASE
from samet.chunking import SAMET_HOURLY_BASE
from samet.chunking import Chunk
from samet.chunking import ChunkPlan
from samet.chunking import plan_chunks
from samet.chunking import plan_summary
from samet.events import BytesTransferred
from samet.events import FileResolved
from samet.events import ItemWritten
from samet.events import PipelineEvent
from samet.events import PipelineListener
from samet.events import RequestPlanned
from samet.events import StorePlanned
from samet.exceptions import DownloadError
from samet.exceptions import MissingCoordinateError
from samet.exceptions import NoDataAvailableError
from samet.exceptions import SametError
from samet.exceptions import SametValidationError
from samet.experiment import Experiment
from samet.models import DAILY_VARIABLES
from samet.models import SAMET_DAILY_HISTORY_START
from samet.models import SAMET_HOURLY_HISTORY_START
from samet.models import Area
from samet.models import DailyCycle
from samet.models import DailyRequest
from samet.models import DailyVariable
from samet.models import HourlyCycle
from samet.models import HourlyRequest
from samet.models import SametRequest
from samet.models import days_between
from samet.models import hours_between
from samet.netcdf import MAX_TEMPERATURE
from samet.netcdf import MIN_TEMPERATURE
from samet.netcdf import open_samet_dataset
from samet.retrieval import Fetcher
from samet.retrieval import HttpxFetcher
from samet.retrieval import SametDownloader
from samet.root import resolve_project_root
from samet.zarr import apply_area
from samet.zarr import files_to_zarr
from samet.zarr import normalize_dataset
from samet.zarr import write_zarr


__all__ = [
    "DAILY_VARIABLES",
    "MAX_TEMPERATURE",
    "MIN_TEMPERATURE",
    "SAMET_DAILY_BASE",
    "SAMET_DAILY_HISTORY_START",
    "SAMET_HOURLY_BASE",
    "SAMET_HOURLY_HISTORY_START",
    "Area",
    "BytesTransferred",
    "Chunk",
    "ChunkPlan",
    "DailyCycle",
    "DailyRequest",
    "DailyVariable",
    "DownloadError",
    "Experiment",
    "Fetcher",
    "FileResolved",
    "HourlyCycle",
    "HourlyRequest",
    "HttpxFetcher",
    "ItemWritten",
    "MissingCoordinateError",
    "NoDataAvailableError",
    "PipelineEvent",
    "PipelineListener",
    "RequestPlanned",
    "SametDownloader",
    "SametError",
    "SametRequest",
    "SametValidationError",
    "StorePlanned",
    "apply_area",
    "days_between",
    "experiment_cache_dir",
    "experiment_cache_key",
    "experiment_store_path",
    "files_to_zarr",
    "hours_between",
    "normalize_dataset",
    "open_samet_dataset",
    "plan_chunks",
    "plan_summary",
    "resolve_project_root",
    "write_zarr",
]
