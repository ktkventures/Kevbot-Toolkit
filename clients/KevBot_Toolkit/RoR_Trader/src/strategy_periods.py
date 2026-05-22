"""Strategy time-period resolution — in-sample / OOS-historical / post-creation.

Single source of truth for the Out-of-Sample feature (see
docs/Spec_OOS_Test_Periods.md). Every KPI split, chart band, and the
Mass Builder OOS gate must bucket trades through `classify_trade_period`
rather than comparing dates ad hoc.

The model — three time bands on the backtest-model lane:

    in_sample      [in_sample_start, in_sample_end)
    oos_hist       [in_sample_end,   forward_test_start)
    post_creation  [forward_test_start, now]

`in_sample_start` / `in_sample_end` live in `strategy['config']` (JSONB,
additive — no schema migration). When unset they DEFAULT such that the
strategy behaves byte-identically to pre-OOS:

    in_sample_end   unset -> forward_test_start   (zero-width OOS band)
    in_sample_start unset -> None (caller uses the first trade's date)

`forward_test_start` is UNCHANGED and immutable — this module only reads
it; it never moves it (see feedback_forward_test_start_immutable).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

# Period labels — stable strings; used as dict keys / DataFrame values.
IN_SAMPLE = 'in_sample'
OOS_HIST = 'oos_hist'
POST_CREATION = 'post_creation'

VALID_PERIODS = (IN_SAMPLE, OOS_HIST, POST_CREATION)


def _parse_dt(value: Any) -> Optional[datetime]:
    """Parse an ISO timestamp (str or datetime) to a datetime, or None.

    Tolerant of trailing 'Z' and of values already being datetimes.
    """
    if value is None or value == '':
        return None
    if isinstance(value, datetime):
        return value
    try:
        s = str(value).strip()
        if s.endswith('Z'):
            s = s[:-1] + '+00:00'
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _config(strategy: dict) -> dict:
    cfg = strategy.get('config')
    return cfg if isinstance(cfg, dict) else {}


def resolve_forward_test_start(strategy: dict) -> Optional[datetime]:
    """The OOS-historical -> post-creation boundary (strategy creation).

    Plain read of `forward_test_start`. Unchanged from pre-OOS — exposed
    here only so callers route every boundary through one module.
    """
    return _parse_dt(strategy.get('forward_test_start'))


def resolve_in_sample_end(strategy: dict) -> Optional[datetime]:
    """The in-sample -> OOS-historical boundary (the KPI divider).

    Returns `config.in_sample_end` when set, else falls back to
    `forward_test_start`. The fallback is what makes the whole feature
    behavior-neutral for every pre-OOS strategy: with no explicit
    in-sample window, the OOS band collapses to zero width and the
    Backtest|Forward split lands exactly where it always has.
    """
    explicit = _parse_dt(_config(strategy).get('in_sample_end'))
    if explicit is not None:
        return explicit
    return resolve_forward_test_start(strategy)


def resolve_in_sample_start(strategy: dict) -> Optional[datetime]:
    """Start of the in-sample band, if explicitly recorded.

    Returns `config.in_sample_start` when set, else None — the caller
    should fall back to the first backtest-lane trade's entry date for
    charting. There is no meaningful default for KPI bucketing (anything
    at or after the start is in-sample regardless), so None is fine.
    """
    return _parse_dt(_config(strategy).get('in_sample_start'))


def _normalize_pair(a: datetime, b: datetime) -> tuple[datetime, datetime]:
    """Make two datetimes comparable — both naive or both aware.

    Alpaca/Polygon timestamps are UTC-aware; config ISO strings may be
    naive. Comparing the two raises TypeError, so coerce: if exactly one
    is aware, drop its tzinfo (see the known timezone-mismatch issue).
    """
    a_aware = a.tzinfo is not None
    b_aware = b.tzinfo is not None
    if a_aware and not b_aware:
        return a.replace(tzinfo=None), b
    if b_aware and not a_aware:
        return a, b.replace(tzinfo=None)
    return a, b


def classify_trade_period(entry_ts: Any, strategy: dict) -> str:
    """Bucket a trade into one of the three time bands by its entry time.

    `entry_ts` may be an ISO string or a datetime. A trade whose entry
    time can't be parsed is treated as `in_sample` (conservative — it
    will not inflate the out-of-sample KPIs that gate trust).
    """
    entry = _parse_dt(entry_ts)
    if entry is None:
        return IN_SAMPLE

    in_sample_end = resolve_in_sample_end(strategy)
    fwd_start = resolve_forward_test_start(strategy)

    # No boundaries at all -> everything is in-sample (a pure backtest
    # with no forward period, e.g. an unsaved strategy).
    if in_sample_end is None and fwd_start is None:
        return IN_SAMPLE

    if in_sample_end is not None:
        e, b = _normalize_pair(entry, in_sample_end)
        if e < b:
            return IN_SAMPLE

    # At/after in_sample_end. Split OOS-historical vs post-creation.
    if fwd_start is not None:
        e, b = _normalize_pair(entry, fwd_start)
        if e < b:
            return OOS_HIST
        return POST_CREATION

    # in_sample_end known but no forward_test_start — unsaved strategy
    # with an explicit in-sample window. Everything past it is OOS.
    return OOS_HIST


def has_oos_window(strategy: dict) -> bool:
    """True when the strategy has a non-zero-width OOS-historical band.

    i.e. an explicit `in_sample_end` strictly before `forward_test_start`.
    Used to decide whether to render the amber band / OOS KPI columns.
    """
    in_sample_end = _parse_dt(_config(strategy).get('in_sample_end'))
    fwd_start = resolve_forward_test_start(strategy)
    if in_sample_end is None or fwd_start is None:
        return False
    e, b = _normalize_pair(in_sample_end, fwd_start)
    return e < b
