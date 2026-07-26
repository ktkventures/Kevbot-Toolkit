"""Tests for market_calendar (board #140 / #15 / #51).

Verifies the holiday table + the extended-session / market-closed
predicates the gap_healer gate relies on. Run from src/:
    python -m pytest test_market_calendar.py -q
"""
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, '.')

import market_calendar as mc  # noqa: E402

ET = ZoneInfo("America/New_York")


def _et(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=ET)


# --- holiday table --------------------------------------------------------

def test_juneteenth_2026_is_holiday():
    # The exact example in board #15 (Juneteenth = Fri 2026-06-19).
    assert mc.is_market_holiday(date(2026, 6, 19)) is True


def test_july4_observed_2026_is_holiday():
    # 2026-07-04 is a Saturday → observed Friday 2026-07-03.
    assert mc.is_market_holiday(date(2026, 7, 3)) is True
    assert mc.is_market_holiday(date(2026, 7, 4)) is False  # the Saturday


def test_normal_trading_day_not_holiday():
    assert mc.is_market_holiday(date(2026, 7, 6)) is False  # Mon


def test_holiday_accepts_datetime_in_utc():
    # 14:00Z on Juneteenth is 10:00 ET, same calendar day → holiday.
    assert mc.is_market_holiday(
        datetime(2026, 6, 19, 14, 0, tzinfo=ZoneInfo("UTC"))) is True


def test_future_year_failsafe_false():
    # Past the hardcoded table → False (fail-safe to always-on behavior).
    assert mc.is_market_holiday(date(2099, 6, 19)) is False


# --- session / closed predicates -----------------------------------------

def test_weekday_rth_is_open():
    dt = _et(2026, 7, 6, 10, 0)  # Mon 10:00 ET
    assert mc.is_extended_session(dt) is True
    assert mc.is_market_closed(dt) is False


def test_premarket_and_postmarket_are_open():
    assert mc.is_extended_session(_et(2026, 7, 6, 5, 0)) is True   # 05:00
    assert mc.is_extended_session(_et(2026, 7, 6, 19, 30)) is True  # 19:30


def test_overnight_after_hours_is_closed():
    assert mc.is_market_closed(_et(2026, 7, 6, 21, 0)) is True   # 21:00
    assert mc.is_market_closed(_et(2026, 7, 6, 3, 0)) is True    # 03:00


def test_weekend_is_closed():
    assert mc.is_market_closed(_et(2026, 7, 4, 10, 0)) is True   # Sat RTH


def test_holiday_rth_is_closed():
    # Weekday + RTH clock time, but it's Juneteenth → closed all day.
    dt = _et(2026, 6, 19, 10, 0)
    assert mc.is_extended_session(dt) is False
    assert mc.is_market_closed(dt) is True


def test_session_boundaries_exact():
    assert mc.is_extended_session(_et(2026, 7, 6, 4, 0)) is True    # 04:00 open
    assert mc.is_extended_session(_et(2026, 7, 6, 3, 59)) is False  # just before
    assert mc.is_extended_session(_et(2026, 7, 6, 20, 0)) is False  # 20:00 closed
    assert mc.is_extended_session(_et(2026, 7, 6, 19, 59)) is True  # just before
