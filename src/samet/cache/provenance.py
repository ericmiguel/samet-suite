"""Provenance ranks for SAMeT fragments.

SAMeT publishes a single blended product per collection, so every value shares
one provenance code and there is no precedence between fragments.
"""

from __future__ import annotations


SINGLE = 0

LEGEND: dict[int, str] = {
    SINGLE: "samet",
}


def legend_payload() -> dict[str, str]:
    """Return the manifest-ready provenance legend."""
    return {str(code): label for code, label in LEGEND.items()}
