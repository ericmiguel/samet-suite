"""HTTP retrieval of SAMeT files with structural verification."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Protocol

import httpx
import numpy as np
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from samet.events import BytesTransferred
from samet.exceptions import DownloadError
from samet.exceptions import NoDataAvailableError
from samet.netcdf import MAX_TEMPERATURE
from samet.netcdf import MIN_TEMPERATURE
from samet.netcdf import PLAUSIBLE_MAGNITUDE
from samet.netcdf import open_samet_dataset


if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator
    from pathlib import Path

    from samet.chunking import Chunk
    from samet.events import PipelineListener

LOGGER = logging.getLogger(__name__)


def _log_retry(state: object) -> None:
    """Log a tenacity retry without exposing response contents."""
    LOGGER.warning("Retrying transient INPE request: %s", state)


@dataclass(frozen=True, kw_only=True)
class HeadInfo:
    """Minimal HEAD response information used by the downloader."""

    status_code: int
    headers: Mapping[str, str]


class Fetcher(Protocol):
    """Offline-testable transport protocol for INPE files."""

    def head(self, url: str) -> HeadInfo:
        """Return status and headers for a URL."""

    def stream_range(self, url: str, start: int, end: int) -> Iterable[bytes]:
        """Yield the bytes of an inclusive range of a remote file."""


class TransientHTTPError(httpx.HTTPError):
    """Internal marker for retryable HTTP status responses."""


class HttpxFetcher:
    """Streaming HTTPS fetcher with explicit timeouts and transient retries."""

    def __init__(
        self,
        *,
        connect_timeout: float = 10.0,
        read_timeout: float = 120.0,
        pool_timeout: float = 10.0,
        write_timeout: float = 120.0,
    ) -> None:
        timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            pool=pool_timeout,
            write=write_timeout,
        )
        self._client = httpx.Client(timeout=timeout, follow_redirects=True)

    def head(self, url: str) -> HeadInfo:
        """Issue a HEAD request with transient retry handling."""
        response = self._request_head(url)
        return HeadInfo(status_code=response.status_code, headers=response.headers)

    def stream_range(self, url: str, start: int, end: int) -> Iterator[bytes]:
        """Stream an inclusive byte range in blocks."""
        response = self._stream_response(url, start, end)
        try:
            yield from response.iter_bytes()
        finally:
            response.close()

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, TransientHTTPError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        before_sleep=_log_retry,
        reraise=True,
    )
    def _request_head(self, url: str) -> httpx.Response:
        """Request HEAD and mark only 429/5xx as retryable."""
        response = self._client.head(url)
        _raise_retryable_status(response)
        return response

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, TransientHTTPError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        before_sleep=_log_retry,
        reraise=True,
    )
    def _stream_response(self, url: str, start: int, end: int) -> httpx.Response:
        """Open a ranged response and mark only transient statuses for retry."""
        request = self._client.build_request(
            "GET", url, headers={"Range": f"bytes={start}-{end}"}
        )
        response = self._client.send(request, stream=True)
        try:
            _raise_retryable_status(response)
            response.raise_for_status()
        except httpx.HTTPError:
            response.close()
            raise
        return response

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()


def _raise_retryable_status(response: httpx.Response) -> None:
    """Raise retry markers for 429 and server errors only."""
    if response.status_code == 429 or response.status_code >= 500:
        raise TransientHTTPError(f"Transient HTTP status {response.status_code}")


class SametDownloader:
    """Download, cache, and verify SAMeT NetCDF files."""

    def __init__(self, fetcher: Fetcher | None = None) -> None:
        self.fetcher = fetcher or HttpxFetcher()

    def download_chunk(
        self,
        chunk: Chunk,
        *,
        listener: PipelineListener | None = None,
        skip_missing: bool = False,
    ) -> Path:
        """Download one SAMeT file into the cache, atomically.

        Parameters
        ----------
        chunk : Chunk
            Planned remote file and cache destination.
        listener : PipelineListener or None, default=None
            Optional subscriber receiving byte-transfer events.
        skip_missing : bool, default=False
            Whether a cycle INPE has not published is left out instead of
            failing. Handled by the experiment; kept for contract parity.

        Returns
        -------
        pathlib.Path
            The verified file.

        Raises
        ------
        NoDataAvailableError
            If INPE has not published the cycle yet.
        DownloadError
            If the transfer or the decoded file cannot be verified.
        """
        del skip_missing  # the experiment handles skipping
        if self._cache_is_usable(chunk):
            return chunk.path
        file_size = self._remote_size(chunk.url)
        chunk_path = chunk.path
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        part = chunk_path.with_name(f".{chunk_path.name}.part")
        part.unlink(missing_ok=True)
        try:
            self._download(chunk, file_size, part, listener=listener)
            self._verify(part, chunk.variables)
            part.replace(chunk_path)
        except NoDataAvailableError:
            raise
        except DownloadError:
            raise
        except (OSError, ValueError, RuntimeError, httpx.HTTPError) as error:
            raise DownloadError(f"Could not download {chunk.url}") from error
        finally:
            part.unlink(missing_ok=True)
        LOGGER.info("Stored %s (%d bytes).", chunk_path.name, file_size)
        return chunk_path

    def _remote_size(self, url: str) -> int:
        """Return Content-Length and reject all failed HEAD responses."""
        try:
            response = self.fetcher.head(url)
        except (OSError, httpx.HTTPError) as error:
            raise DownloadError(f"HEAD failed for {url}") from error
        status = _status_code(response)
        if status == 404:
            raise NoDataAvailableError(f"INPE has not published {url} yet.")
        if status >= 400:
            raise DownloadError(f"HEAD returned HTTP {status} for {url}")
        value = _headers(response).get("content-length")
        if value is None:
            raise DownloadError(f"HEAD omitted Content-Length for {url}")
        try:
            return int(value)
        except ValueError as error:
            raise DownloadError(f"Invalid Content-Length for {url}") from error

    def _download(
        self,
        chunk: Chunk,
        file_size: int,
        target: Path,
        *,
        listener: PipelineListener | None,
    ) -> None:
        """Stream the whole file into the target path."""
        written = 0
        with target.open("wb") as output:
            for block in self.fetcher.stream_range(chunk.url, 0, file_size - 1):
                if not isinstance(block, bytes):
                    raise DownloadError("Fetcher yielded a non-byte block.")
                output.write(block)
                written += len(block)
                if listener is not None:
                    listener(
                        BytesTransferred(
                            name=chunk.label, total=file_size, amount=len(block)
                        )
                    )
        if written != file_size:
            raise DownloadError(
                f"Downloaded {written} of {file_size} advertised bytes for {chunk.url}."
            )
        if listener is not None:
            listener(
                BytesTransferred(name=chunk.label, total=written, amount=0, done=True)
            )

    def _cache_is_usable(self, chunk: Chunk) -> bool:
        """Verify an existing cache file, re-checking recent cycles."""
        if not chunk.path.is_file():
            return False
        if chunk.mutable:
            try:
                if self._remote_size(chunk.url) != chunk.path.stat().st_size:
                    return False
            except (NoDataAvailableError, DownloadError):
                return True
        try:
            self._verify(chunk.path, chunk.variables)
        except (DownloadError, OSError, ValueError, RuntimeError):
            chunk.path.unlink(missing_ok=True)
            return False
        return True

    @staticmethod
    def _verify(path: Path, variables: tuple[str, ...]) -> None:
        """Verify that a file decodes and carries plausible SAMeT values.

        The check refuses files that do not decode, are missing the planned
        variables, carry no finite values, or hold magnitudes no 2 m
        temperature field reaches.
        """
        if not path.is_file() or path.stat().st_size == 0:
            raise DownloadError(f"Downloaded file is empty: {path}")
        dataset = open_samet_dataset(path)
        try:
            missing = set(variables) - {str(name) for name in dataset.data_vars}
            if missing:
                raise DownloadError(
                    f"SAMeT file {path.name} is missing {sorted(missing)}; "
                    f"it decoded {sorted(str(n) for n in dataset.data_vars)} instead."
                )
            for raw_name, variable in dataset.data_vars.items():
                name = str(raw_name)
                values = np.asarray(variable.values, dtype="float64")
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    raise DownloadError(
                        f"SAMeT file {path.name} has no finite values for {name}."
                    )
                if name.startswith("nobs"):
                    if float(np.nanmin(finite)) < 0.0:
                        raise DownloadError(
                            f"SAMeT file {path.name} carries negative station counts."
                        )
                elif (
                    float(np.nanmin(finite)) < MIN_TEMPERATURE
                    or float(np.nanmax(finite)) > MAX_TEMPERATURE
                    or float(np.max(np.abs(finite))) >= PLAUSIBLE_MAGNITUDE
                ):
                    raise DownloadError(
                        f"SAMeT file {path.name} carries implausible "
                        f"temperature magnitudes for {name}."
                    )
        finally:
            dataset.close()


def _status_code(response: object) -> int:
    """Extract an HTTP status from a protocol response."""
    value = getattr(response, "status_code", None)
    if not isinstance(value, int):
        raise DownloadError("Fetcher HEAD response has no integer status code.")
    return value


def _headers(response: object) -> dict[str, str]:
    """Normalize protocol response headers to lowercase strings."""
    raw = getattr(response, "headers", None)
    if not isinstance(raw, Mapping):
        raise DownloadError("Fetcher HEAD response has no headers.")
    return {str(key).lower(): str(value) for key, value in raw.items()}


__all__ = [
    "Fetcher",
    "HeadInfo",
    "HttpxFetcher",
    "SametDownloader",
    "TransientHTTPError",
]
