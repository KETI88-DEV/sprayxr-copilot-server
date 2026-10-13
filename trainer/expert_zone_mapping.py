"""Canonical Zone IDs for known expert-reference conversion corrections."""

from __future__ import annotations

from typing import Any


_EXPERT_ZONE_REMAPS: dict[tuple[str, str, int], dict[str, str]] = {
    ("spray", "learning", 2): {"1": "3", "3": "1"},
}


def canonical_expert_zone_id(
    process: str,
    content_type: str,
    mode_id: int,
    zone_id: Any,
) -> str | None:
    """Return the design-document Zone ID without modifying raw source data."""

    if zone_id is None:
        return None
    value = str(zone_id).strip()
    if not value:
        return None
    try:
        number = float(value)
        if number.is_integer():
            value = str(int(number))
    except ValueError:
        pass
    return _EXPERT_ZONE_REMAPS.get(
        (process.casefold(), content_type.casefold(), int(mode_id)), {}
    ).get(value, value)
