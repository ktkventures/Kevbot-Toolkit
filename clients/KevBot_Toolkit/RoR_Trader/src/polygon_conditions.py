"""Polygon sale-condition eligibility rules.

Centralized so the live engine (ralph_engine.py) and the flat-file
ingestion (flat_file_ingestion.py) apply IDENTICAL filtering — that
parity is the entire point of Phase G. Any bug in one applies to both
by construction.

Public API:
    _classify_eligibility(conditions) -> (ohlc_ok, vol_ok)
        Returns whether a trade's conditions allow it to update OHLC
        and/or volume per Polygon's canonical rules.

    _load_polygon_condition_rules() -> (ohlc_excluded, vol_excluded)
        Fetches Polygon's /v3/reference/conditions endpoint at first
        call, caches for process lifetime. Falls back to hardcoded
        snapshot if API unreachable.

    _parse_conditions(value) -> set[int] | None
        Normalizes any of: comma-separated string, JSON-array string,
        list of ints, list of strings, None, empty → set[int] | None.
        Returns None on unparseable input so callers can fail-open.

Phase G (2026-05-15): extracted from flat_file_ingestion.py so
ralph_engine.py can share the exact same eligibility logic. The
hardcoded fallback sets match the snapshot we captured 2026-05-14
when first running the Polygon API.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# Snapshot from Polygon's /v3/reference/conditions (asset_class=stocks)
# captured 2026-05-14. Used only when the live API fetch fails at
# startup. Each ID below has consolidated.updates_high_low=False OR
# consolidated.updates_open_close=False per Polygon's official rules.
#
#   2  Average Price Trade
#   5  Bunched Sold Trade
#   7  Cash Sale
#   10 Derivatively Priced
#   12 Form T / Extended Hours
#   13 Extended Hours (Sold Out Of Sequence)
#   15 Market Center Official Close
#   16 Market Center Official Open
#   20 Next Day
#   21 Price Variation Trade
#   22 Prior Reference Price
#   29 Seller
#   32 Sold (Out Of Sequence)
#   33 Sold (Out of Sequence) and Stopped Stock
#   37 Odd Lot Trade
#   52 Contingent Trade
#   53 Qualified Contingent Trade
FALLBACK_OHLC_EXCLUDED: set[int] = {
    2, 5, 7, 10, 12, 13, 15, 16, 20, 21, 22, 29, 32, 33, 37, 52, 53,
}

# Conditions where consolidated.updates_volume=False (rare).
FALLBACK_VOL_EXCLUDED: set[int] = {15, 16, 38}


# Module-level caches — populated on first call to _load_polygon_condition_rules().
_OHLC_EXCLUDED_CACHE: set[int] | None = None
_VOL_EXCLUDED_CACHE: set[int] | None = None


def _load_polygon_condition_rules() -> tuple[set[int], set[int]]:
    """Return (ohlc_excluded, vol_excluded) sets of Polygon condition codes.

    Fetched from /v3/reference/conditions on first call, cached
    per-process. Falls back to hardcoded snapshot if the API is
    unreachable.
    """
    global _OHLC_EXCLUDED_CACHE, _VOL_EXCLUDED_CACHE
    if _OHLC_EXCLUDED_CACHE is not None and _VOL_EXCLUDED_CACHE is not None:
        return _OHLC_EXCLUDED_CACHE, _VOL_EXCLUDED_CACHE

    api_key = os.environ.get("POLYGON_API_KEY", "").strip()
    if not api_key:
        logger.warning(
            "POLYGON_API_KEY not set — using fallback condition exclusion list")
        _OHLC_EXCLUDED_CACHE = set(FALLBACK_OHLC_EXCLUDED)
        _VOL_EXCLUDED_CACHE = set(FALLBACK_VOL_EXCLUDED)
        return _OHLC_EXCLUDED_CACHE, _VOL_EXCLUDED_CACHE

    try:
        import httpx
        resp = httpx.get(
            "https://api.polygon.io/v3/reference/conditions",
            params={"asset_class": "stocks", "limit": 1000, "apiKey": api_key},
            timeout=30.0,
        )
        resp.raise_for_status()
        rules = resp.json().get("results", [])
        ohlc_excl: set[int] = set()
        vol_excl: set[int] = set()
        for c in rules:
            cid = c.get("id")
            if not isinstance(cid, int):
                continue
            consolidated = (c.get("update_rules") or {}).get("consolidated") or {}
            if (consolidated.get("updates_high_low") is False
                    or consolidated.get("updates_open_close") is False):
                ohlc_excl.add(cid)
            if consolidated.get("updates_volume") is False:
                vol_excl.add(cid)
        if not ohlc_excl:
            logger.warning(
                "Polygon conditions API returned empty exclusion set — using fallback")
            ohlc_excl = set(FALLBACK_OHLC_EXCLUDED)
            vol_excl = set(FALLBACK_VOL_EXCLUDED)
        else:
            logger.info(
                "Loaded canonical Polygon condition rules: %d OHLC-excluded, %d vol-excluded",
                len(ohlc_excl), len(vol_excl))
            logger.info("OHLC-excluded codes: %s", sorted(ohlc_excl))
        _OHLC_EXCLUDED_CACHE = ohlc_excl
        _VOL_EXCLUDED_CACHE = vol_excl
    except Exception as e:
        logger.warning(
            "Polygon conditions API fetch failed (%s) — using fallback", e)
        _OHLC_EXCLUDED_CACHE = set(FALLBACK_OHLC_EXCLUDED)
        _VOL_EXCLUDED_CACHE = set(FALLBACK_VOL_EXCLUDED)
    return _OHLC_EXCLUDED_CACHE, _VOL_EXCLUDED_CACHE


def _parse_conditions(value: Any) -> set[int] | None:
    """Normalize a trade's conditions field into a set of int codes.

    Accepts:
      - None or empty list/string → empty set (= "regular trade")
      - Comma-separated string: "12,37" or '"12,37"'
      - List of ints: [12, 37]
      - List of strings: ['12', '37']  (legacy compat)
      - JSON-array-looking string: "[12, 37]"

    Returns None if the input can't be normalized to a set of ints
    (caller should treat as fail-open).
    """
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        out: set[int] = set()
        for c in value:
            if c is None:
                continue
            try:
                out.add(int(c))
            except (ValueError, TypeError):
                # Letter codes (legacy CTA/UTP) — can't map without
                # a translation table. Return None so caller fails
                # open. ralph_engine should not see letter codes when
                # using Polygon WS, but we defend against it.
                return None
        return out
    if isinstance(value, str):
        s = value.strip().strip('"').strip('[').strip(']')
        if not s:
            return set()
        try:
            return {int(c.strip()) for c in s.split(',') if c.strip()}
        except ValueError:
            return None
    return None


def _classify_eligibility(conditions: Any) -> tuple[bool, bool]:
    """Return (ohlc_eligible, volume_eligible) for a trade.

    Mirrors Polygon REST aggregates' own filtering rules:
      - OHLC-ineligible (e.g. Form T, Odd Lot, Average Price Trade)
        don't move O/H/L/C but do count for volume
      - Vol-ineligible (rare — Market Center Open/Close, Corrected
        Consolidated Close) don't add to volume but may set O/C
      - Both-ineligible: skip entirely (handful of rare cases)
      - Both-eligible (the common case): full contribution

    Unparseable conditions → (True, True) fail-open.
    """
    codes = _parse_conditions(conditions)
    if codes is None:
        return True, True
    ohlc_excluded, vol_excluded = _load_polygon_condition_rules()
    return (
        not (codes & ohlc_excluded),
        not (codes & vol_excluded),
    )
