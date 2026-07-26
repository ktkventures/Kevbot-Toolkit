"""US equity market calendar — weekends, extended-hours window, and the
NYSE/Nasdaq full-day holiday table (board #140, 2026-07-26).

Why this exists: several background loops (gap_healer's sweeper, and
potentially the data-worker sweeps) only need one predicate — "is there
an active trading session right now where fresh bars are expected?".
The weekend + extended-hours half of that already lives in
`data_worker_ingest.is_market_window`; what was missing is the
market-HOLIDAY half. On a full-day holiday (e.g. Juneteenth) the market
is closed all day, so every window looks like a WS-missed gap and the
gap_healer keeps REST-confirming genuinely-tradeless windows — pure log
noise + REST/Supabase load, zero bars actually healed (board #15/#51).

Dependency-light on purpose (stdlib `zoneinfo` only — no
pandas_market_calendars / exchange_calendars): this is imported by the
live-worker gap_healer and must stay cheap and side-effect-free.

The holiday table is HARDCODED and fail-SAFE: a date in a year past the
table's coverage returns `is_market_holiday() == False`, i.e. callers
fall back to their current (pre-#140) behavior — the worst case is the
benign log noise this module removes, never a missed heal on a real
trading day. Extend `_HOLIDAYS_BY_YEAR` each December (see
_LAST_KNOWN_YEAR).
"""
from __future__ import annotations

import datetime as _dt
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")

# Extended-hours window (ET), mirrors data_worker_ingest.is_market_window:
# 4:00 AM – 8:00 PM ET covers every pre/regular/post session a strategy
# might trade. Kept in sync deliberately — gap_healer should still heal
# real WS-missed bars during pre/post-market (sparse but real trades).
_SESSION_OPEN_MIN = 4 * 60    # 04:00 ET
_SESSION_CLOSE_MIN = 20 * 60  # 20:00 ET

# Full-day NYSE/Nasdaq equity-market closures. Half-days (day after
# Thanksgiving, Christmas Eve, July 3 when it's a trading day) are NOT
# listed — the market IS open then, so gap-healing should proceed.
# Verified 2026-07-26: every date lands on the expected weekday.
_HOLIDAYS_BY_YEAR: dict[int, tuple[tuple[int, int], ...]] = {
    2025: ((1, 1), (1, 20), (2, 17), (4, 18), (5, 26), (6, 19),
           (7, 4), (9, 1), (11, 27), (12, 25)),
    2026: ((1, 1), (1, 19), (2, 16), (4, 3), (5, 25), (6, 19),
           (7, 3), (9, 7), (11, 26), (12, 25)),
    2027: ((1, 1), (1, 18), (2, 15), (3, 26), (5, 31), (6, 18),
           (7, 5), (9, 6), (11, 25), (12, 24)),
}

# Extend the table before this year rolls over. Past this year the
# holiday check returns False (fail-safe: revert to always-on behavior).
_LAST_KNOWN_YEAR = max(_HOLIDAYS_BY_YEAR)

US_EQUITY_HOLIDAYS: frozenset[_dt.date] = frozenset(
    _dt.date(y, m, d)
    for y, days in _HOLIDAYS_BY_YEAR.items()
    for (m, d) in days
)


def _to_et(now: _dt.datetime | None) -> _dt.datetime:
    now = now or _dt.datetime.now(_dt.timezone.utc)
    if now.tzinfo is None:  # treat naive as UTC (repo convention)
        now = now.replace(tzinfo=_dt.timezone.utc)
    return now.astimezone(_ET)


def is_market_holiday(when: _dt.datetime | _dt.date | None = None) -> bool:
    """True if `when` falls on a US equity full-day market holiday.

    Accepts a UTC/aware datetime (converted to its ET calendar date),
    a naive datetime (assumed UTC), or a plain `date`. Fail-safe False
    for any date whose year is outside the hardcoded table.
    """
    if isinstance(when, _dt.datetime):
        d = _to_et(when).date()
    elif isinstance(when, _dt.date):
        d = when
    else:
        d = _to_et(None).date()
    return d in US_EQUITY_HOLIDAYS


def is_extended_session(now: _dt.datetime | None = None) -> bool:
    """True during a weekday extended-hours session (04:00–20:00 ET) that
    is NOT a market holiday — i.e. a window where fresh bars are expected.
    """
    et = _to_et(now)
    if et.weekday() >= 5:  # Saturday / Sunday
        return False
    if et.date() in US_EQUITY_HOLIDAYS:
        return False
    minutes = et.hour * 60 + et.minute
    return _SESSION_OPEN_MIN <= minutes < _SESSION_CLOSE_MIN


def is_market_closed(now: _dt.datetime | None = None) -> bool:
    """Inverse of `is_extended_session` — True when there is no active
    trading session (overnight, weekend, or full-day holiday). This is
    the "don't expect bars" predicate the gap_healer sweeper gates on.
    """
    return not is_extended_session(now)


def coverage_note() -> str:
    """One-line diagnostic for logs: the holiday table's coverage."""
    return (f"US equity holidays loaded {min(_HOLIDAYS_BY_YEAR)}–"
            f"{_LAST_KNOWN_YEAR} ({len(US_EQUITY_HOLIDAYS)} dates)")
