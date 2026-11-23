"""Region registry: look up a RegionPack by code."""

from __future__ import annotations

from app.regions.base import RegionPack
from app.regions.india.pack import INDIA_PACK
from app.regions.us.pack import US_PACK

DEFAULT_REGION = "us"

_REGISTRY: dict[str, RegionPack] = {pack.code: pack for pack in (US_PACK, INDIA_PACK)}


class UnknownRegionError(ValueError):
    """Raised when a region code has no registered pack."""


def get_pack(code: str | None = None) -> RegionPack:
    """Return the pack for ``code`` (case-insensitive); the default region when ``None``."""
    key = (code or DEFAULT_REGION).lower()
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        raise UnknownRegionError(
            f"Unknown region '{code}'. Available: {', '.join(sorted(_REGISTRY))}"
        ) from exc


def available_regions() -> list[RegionPack]:
    """All registered packs, ordered by code (for signup dropdowns)."""
    return [_REGISTRY[code] for code in sorted(_REGISTRY)]