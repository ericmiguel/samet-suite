"""Downloader behaviour: whole-file retrieval, verification, idempotence."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from typing import override

import pytest


if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr

    from samet.chunking import Chunk


from conftest import SAMPLE_FILE_SIZE
from conftest import FakeFetcher
from conftest import probe_day
from conftest import probe_hour
from conftest import sample_samet_dataset
from samet import DailyRequest
from samet import DownloadError
from samet import HourlyRequest
from samet import NoDataAvailableError
from samet import SametDownloader
from samet import plan_chunks


def daily_chunk(tmp_path: Path, *, now: datetime | None = None) -> Chunk:
    """Plan the first (tmax) chunk of a probe daily request."""
    return plan_chunks(
        DailyRequest(day=probe_day()),
        tmp_path,
        request_name="probe",
        now=now,
    ).chunks[0]


def hourly_chunk(tmp_path: Path) -> Chunk:
    """Plan the chunk of a single-hour probe request."""
    return plan_chunks(
        HourlyRequest(start=probe_hour(), end=probe_hour()),
        tmp_path,
        request_name="probe",
    ).chunks[0]


def stub_decoder(dataset: xr.Dataset):
    """Return a decoder replacement yielding the given dataset."""

    def decode(_path: Path) -> xr.Dataset:
        return dataset

    return decode


def test_daily_plan_covers_the_three_variables(tmp_path: Path) -> None:
    """One daily request plans one file per variable with canonical names."""
    plan = plan_chunks(DailyRequest(day=probe_day()), tmp_path, request_name="probe")
    assert len(plan.chunks) == 3
    variables = [chunk.variables for chunk in plan.chunks]
    assert variables == [
        ("tmax", "nobs_tmax"),
        ("tmin", "nobs_tmin"),
        ("tmean", "nobs_tmean"),
    ]
    urls = [chunk.url for chunk in plan.chunks]
    assert urls[0] == (
        "https://data.inpe.br/bdc/data/samet_daily/DAILY/TMAX/2026/09"
        "/SAMeT_CPTEC_TMAX_20260915.nc"
    )
    assert urls[2] == (
        "https://data.inpe.br/bdc/data/samet_daily/DAILY/TMED/2026/09"
        "/SAMeT_CPTEC_TMED_20260915.nc"
    )


def test_hourly_plan_covers_every_hour(tmp_path: Path) -> None:
    """One hourly request plans one file per hour in the range."""
    from datetime import datetime

    plan = plan_chunks(
        HourlyRequest(start=datetime(2026, 9, 15, 22), end=datetime(2026, 9, 16, 0)),
        tmp_path,
        request_name="probe",
    )
    assert len(plan.chunks) == 3
    assert plan.chunks[0].url == (
        "https://data.inpe.br/bdc/data/samet_hourly/HOURLY/2026/09/15"
        "/SAMeT_CPTEC_2026091522.nc"
    )
    assert plan.chunks[0].variables == ("temperature", "nobs")
    assert plan.chunks[2].url.endswith("/2026/09/16/SAMeT_CPTEC_2026091600.nc")


def test_download_writes_and_verifies_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole file lands in the cache and the request is recorded."""
    chunk = daily_chunk(tmp_path)
    fetcher = FakeFetcher()
    monkeypatch.setattr(
        "samet.retrieval.open_samet_dataset",
        stub_decoder(sample_samet_dataset(variable="tmax")),
    )
    path = SametDownloader(fetcher=fetcher).download_chunk(chunk)
    assert path == chunk.path
    assert path.stat().st_size == SAMPLE_FILE_SIZE
    assert fetcher.ranges == [
        (
            "https://data.inpe.br/bdc/data/samet_daily/DAILY/TMAX/2026/09"
            "/SAMeT_CPTEC_TMAX_20260915.nc",
            0,
            SAMPLE_FILE_SIZE - 1,
        )
    ]
    assert not chunk.path.with_name(f".{chunk.path.name}.part").exists()


def test_download_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A verified file is reused; fresh cycles re-check the remote size."""
    chunk = daily_chunk(tmp_path, now=datetime(2026, 9, 16, 12))
    fetcher = FakeFetcher()
    monkeypatch.setattr(
        "samet.retrieval.open_samet_dataset",
        stub_decoder(sample_samet_dataset(variable="tmax")),
    )
    downloader = SametDownloader(fetcher=fetcher)
    downloader.download_chunk(chunk)
    heads = list(fetcher.heads)
    ranges = list(fetcher.ranges)
    downloader.download_chunk(chunk)
    # No bytes are re-transferred; the recent-cycle freshness probe issues
    # one HEAD to compare the advertised size.
    assert fetcher.ranges == ranges
    assert len(fetcher.heads) == len(heads) + 1
    assert chunk.path.stat().st_size == SAMPLE_FILE_SIZE


def test_download_emits_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Byte events reach the listener."""
    chunk = hourly_chunk(tmp_path)
    events: list[object] = []
    monkeypatch.setattr(
        "samet.retrieval.open_samet_dataset",
        stub_decoder(sample_samet_dataset(variable="temperature", hour=probe_hour())),
    )
    SametDownloader(fetcher=FakeFetcher()).download_chunk(chunk, listener=events.append)
    assert {type(event).__name__ for event in events} >= {"BytesTransferred"}


def test_download_rejects_implausible_temperatures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A field beyond any 2 m temperature is refused."""
    chunk = daily_chunk(tmp_path)
    dataset = sample_samet_dataset(variable="tmax")
    dataset["tmax"].values[:] = 500.0
    monkeypatch.setattr("samet.retrieval.open_samet_dataset", stub_decoder(dataset))
    with pytest.raises(DownloadError, match="implausible"):
        SametDownloader(fetcher=FakeFetcher()).download_chunk(chunk)
    assert not chunk.path.exists()


def test_download_rejects_negative_station_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A station count cannot be negative; a wrong file is refused."""
    chunk = daily_chunk(tmp_path)
    dataset = sample_samet_dataset(variable="tmax")
    dataset["nobs_tmax"].values[:] = -2.0
    monkeypatch.setattr("samet.retrieval.open_samet_dataset", stub_decoder(dataset))
    with pytest.raises(DownloadError, match="negative"):
        SametDownloader(fetcher=FakeFetcher()).download_chunk(chunk)


def test_download_rejects_files_without_the_planned_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file that does not carry the planned fields is refused."""
    chunk = daily_chunk(tmp_path)
    dataset = sample_samet_dataset(variable="tmin")
    monkeypatch.setattr("samet.retrieval.open_samet_dataset", stub_decoder(dataset))
    with pytest.raises(DownloadError, match="missing"):
        SametDownloader(fetcher=FakeFetcher()).download_chunk(chunk)


def test_download_rejects_a_size_mismatch(tmp_path: Path) -> None:
    """A truncated transfer is refused by the byte accounting."""
    chunk = daily_chunk(tmp_path)

    class ShortFetcher(FakeFetcher):
        @override
        def stream_range(self, url: str, start: int, end: int):
            self.ranges.append((url, start, end))
            yield b"X" * 10  # far fewer bytes than advertised

    with pytest.raises(DownloadError, match="advertised"):
        SametDownloader(fetcher=ShortFetcher()).download_chunk(chunk)
    assert not chunk.path.exists()


def test_download_reports_an_unpublished_cycle(tmp_path: Path) -> None:
    """A 404 on HEAD means INPE has not published the cycle yet."""
    chunk = daily_chunk(tmp_path)
    with pytest.raises(NoDataAvailableError, match="not published"):
        SametDownloader(fetcher=FakeFetcher(status=404)).download_chunk(chunk)


def test_download_reports_an_unusable_status(tmp_path: Path) -> None:
    """A server failure on HEAD surfaces as a download error."""
    chunk = daily_chunk(tmp_path)
    with pytest.raises(DownloadError, match="HTTP 500"):
        SametDownloader(fetcher=FakeFetcher(status=500)).download_chunk(chunk)
