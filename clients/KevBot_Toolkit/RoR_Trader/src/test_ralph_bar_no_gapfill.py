"""Lock the no-gap-fill behavior of Ralph's BarBuilder (2026-06-10).

The backtest builds bars via data_loader.resample_to_timeframe(...).dropna(
subset=['open']) — empty/no-trade periods produce NO bar (TradingView
convention). The live engine previously GAP-FILLED empty periods with flat
zero-volume carry-forward bars in accept_bar / force_close_stale_bar /
process_tick, polluting incremental indicators after-hours and diverging from
the backtest (~30% of live 1Min SPY cache bars were these flats).

With BAR_GAP_FILL_ENABLED = False (the shipped default) these paths must NOT
emit flats — bar count, _bar_count, and the index must match the backtest's
dropna'd resample. The flag exists only as a one-flip rollback; the flag-on
case is also covered so the rollback path stays correct.

Run:  cd src && ../.venv/bin/python test_ralph_bar_no_gapfill.py
Or:   cd src && ../.venv/bin/pytest test_ralph_bar_no_gapfill.py -v
"""

from datetime import timedelta

import pandas as pd

import ralph_engine
from ralph_engine import BarBuilder


T0 = pd.Timestamp("2026-04-14T13:30:00+00:00")


def _minute_bars_with_gap():
    """1Min bars with a deliberate 28-minute wall-clock gap (e.g. the
    after-hours / no-trade hole that REST drops and live used to gap-fill).
    Returns (list_of_bar_dicts, equivalent_1min_dataframe)."""
    # Minutes present: 13:30, 13:31, 13:32, [gap], 14:00, 14:01
    offsets = [0, 1, 2, 30, 31]
    bars = []
    price = 500.0
    for k in offsets:
        ts = T0 + timedelta(minutes=k)
        price += 0.5
        bars.append({
            'timestamp': ts.isoformat(),
            'open': price, 'high': price + 0.3,
            'low': price - 0.2, 'close': price + 0.1,
            'volume': 1000.0 + k,
        })
    df = pd.DataFrame([
        {'open': b['open'], 'high': b['high'], 'low': b['low'],
         'close': b['close'], 'volume': b['volume'],
         'timestamp': pd.Timestamp(b['timestamp'])}
        for b in bars
    ]).set_index('timestamp')
    return bars, df


def _backtest_reference(df_1min):
    """Backtest reference for a 1Min strategy: REST returns a 1-min bar only
    for minutes that had trades, so the reference is exactly the traded-minute
    rows (the empty gap minutes never become bars — same as resample().dropna()
    for coarser TFs). df_1min already contains only traded minutes."""
    return df_1min


def test_accept_bar_no_gapfill_matches_backtest():
    bars, df = _minute_bars_with_gap()
    b = BarBuilder(tf_seconds=60)
    for bar in bars:
        b.accept_bar(bar)

    backtest = _backtest_reference(df)

    # No synthetic gap bars: exactly the fed bars survive.
    assert len(b.history) == len(bars) == len(backtest), (
        f"history={len(b.history)} fed={len(bars)} backtest={len(backtest)}")
    assert b._bar_count == len(bars), f"_bar_count={b._bar_count}"
    # No flat zero-volume carry-forwards anywhere.
    assert (b.history['volume'] > 0).all(), "found a zero-volume gap-fill bar"
    # Index matches the fed timestamps (no invented gap timestamps).
    fed_idx = pd.DatetimeIndex([pd.Timestamp(x['timestamp']) for x in bars])
    assert list(b.history.index) == list(fed_idx), "index has synthetic gaps"
    # And matches the backtest index bar-for-bar.
    assert list(b.history.index) == list(backtest.index)


def test_force_close_emits_one_real_bar_no_flats():
    b = BarBuilder(tf_seconds=60)
    # Seed a single forming minute (14:00:00..14:00:04) via per-second bars.
    minute = pd.Timestamp("2026-04-14T14:00:00+00:00")
    for s in range(5):
        ts = minute + timedelta(seconds=s)
        b.accept_second_bar({
            'timestamp': ts.isoformat(),
            'open': 501.0, 'high': 501.4, 'low': 500.8,
            'close': 501.2, 'volume': 300.0,
        })
    assert len(b.history) == 0, "no boundary crossed yet; only a partial"
    assert b._partial is not None

    # Force-close far in the future (30 min later) — must close exactly ONE
    # real bar and emit NO flat carry-forwards through the empty minutes.
    completed = b.force_close_stale_bar(minute + timedelta(minutes=30))
    assert completed is not None
    assert len(b.history) == 1, f"expected 1 real bar, got {len(b.history)}"
    assert b._partial is None, "partial must be cleared (no synthetic carry)"
    assert (b.history['volume'] > 0).all(), "force-close emitted a flat bar"


def test_bar_count_parity_with_backtest():
    """bar_count must count only traded bars (like the backtest's dropna),
    so bar_count_exit / max_hold_bars stay aligned across the gap."""
    bars, df = _minute_bars_with_gap()
    b = BarBuilder(tf_seconds=60)
    for bar in bars:
        b.accept_bar(bar)
    assert b._bar_count == len(_backtest_reference(df))


def test_flag_on_restores_gapfill_rollback_path():
    """The rollback flag still works: with gap-fill enabled, accept_bar
    DOES fill the empty minutes (so we can flip back if needed)."""
    bars, _ = _minute_bars_with_gap()
    orig = ralph_engine.BAR_GAP_FILL_ENABLED
    try:
        ralph_engine.BAR_GAP_FILL_ENABLED = True
        b = BarBuilder(tf_seconds=60)
        for bar in bars:
            b.accept_bar(bar)
        # 5 real bars + 27 gap bars between 13:32 and 14:00 (13:33..13:59).
        assert len(b.history) > len(bars), "flag-on should gap-fill"
        flats = (b.history['volume'] == 0).sum()
        assert flats == 27, f"expected 27 gap-fill flats, got {flats}"
    finally:
        ralph_engine.BAR_GAP_FILL_ENABLED = orig


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    p = f = 0
    for fn in tests:
        try:
            fn()
            print(f'  PASS  {fn.__name__}')
            p += 1
        except Exception as e:
            print(f'  FAIL  {fn.__name__}: {e}')
            f += 1
    print(f'\n{p} passed, {f} failed')
