"""The public API surface is importable and complete."""

from __future__ import annotations

import samet


def test_every_exported_name_exists() -> None:
    """``__all__`` never drifts from the module namespace."""
    missing = [name for name in samet.__all__ if not hasattr(samet, name)]
    assert missing == []


def test_core_contract_is_importable() -> None:
    """The Experiment contract imports from the package root."""
    from samet import DailyRequest
    from samet import Experiment
    from samet import HourlyRequest
    from samet import SametDownloader

    assert Experiment is not None
    assert DailyRequest is not None
    assert HourlyRequest is not None
    assert SametDownloader is not None
