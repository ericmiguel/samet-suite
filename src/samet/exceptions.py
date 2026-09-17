"""Exceptions raised by :mod:`samet`."""

from __future__ import annotations


class SametError(Exception):
    """Base class for all SAMeT errors."""


class SametValidationError(SametError, ValueError):
    """Raised when a request or dataset violates a library contract."""


class NoDataAvailableError(SametError):
    """Raised when INPE publishes no file for a planned cycle."""


class DownloadError(SametError):
    """Raised when an HTTP transfer or a downloaded file cannot be verified."""


class MissingCoordinateError(SametValidationError):
    """Raised when a dataset lacks a required spatial or temporal coordinate."""
