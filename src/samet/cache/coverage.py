"""Coverage summaries for SAMeT stores.

SAMeT fragments are whole-domain South American grids, so area is a read-time
view and the coverage axes that matter are time and variables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from samet.models import DAILY_VARIABLES
from samet.models import DailyRequest
from samet.models import HourlyRequest


if TYPE_CHECKING:
    from collections.abc import Mapping


GRID_SIGNATURE = "latlon:0.05:south-america"


def summarize_coverage(requests: Mapping[str, object]) -> dict[str, object]:
    """Return the time, variable, and grid coverage of a request set."""
    starts: list[str] = []
    ends: list[str] = []
    variables: dict[str, None] = {}
    for request in requests.values():
        if isinstance(request, DailyRequest):
            starts.append(request.day.isoformat())
            ends.append(request.day.isoformat())
            for variable in DAILY_VARIABLES:
                variables.setdefault(variable.value, None)
        elif isinstance(request, HourlyRequest):
            starts.append(request.start.isoformat())
            ends.append(request.end.isoformat())
            variables.setdefault("temperature", None)
    return {
        "time": [min(starts), max(ends)] if starts else [],
        "variables": sorted(variables),
        "grid": GRID_SIGNATURE,
    }
