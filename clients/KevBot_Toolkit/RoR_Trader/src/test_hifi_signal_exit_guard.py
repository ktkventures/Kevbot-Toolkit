"""Lock the Hi-Fi Pass 2 signal-exit guard (2026-06-11, iter 0611b).

THE BUG (confirmed on sid 302): the Pass 2 stop/target walker processed
SIGNAL-exit trades (its guards only filtered non-L stop/target exits).
Its 1s window (exit bar + padding) extends past the signal exit moment,
so it "found" the stop crossing AFTER the position was already closed
and rewrote a correct signal exit (utv4_bear_flip @14:48:00, matching
the live engine) into stop_loss @14:48:15 at a worse price — reason,
price, exit_fill_ts, and r_multiple all overwritten. Every backtest
write since the exit walker shipped converted signal exits this way
whenever price kept running after the close.

THE FIX: signal/time/bar-count exits never enter the stop/target
walker; the engine's fill is authoritative. Genuine stop/target exits
still get their sub-second refinement.

Run:  cd src && ../.venv/bin/python -m pytest test_hifi_signal_exit_guard.py -v
"""
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, '.')

import api.services.backtest_service as bts


def _bars_1s(start_iso: str, seconds: int, lows):
    idx = pd.date_range(start_iso, periods=seconds, freq='1s', tz='UTC')
    lows = list(lows) + [lows[-1]] * (seconds - len(lows))
    return pd.DataFrame({
        'open': [l + 0.10 for l in lows],
        'high': [l + 0.20 for l in lows],
        'low': lows,
        'close': [l + 0.05 for l in lows],
        'volume': [100.0] * seconds,
    }, index=idx)


def _trade(exit_reason: str, exit_time: str) -> pd.DataFrame:
    return pd.DataFrame([{
        'entry_time': '2026-06-11T14:47:10+00:00',
        'entry_fill_ts': '2026-06-11T14:47:10+00:00',
        'exit_time': exit_time,
        'exit_fill_ts': exit_time,
        'exit_reason': exit_reason,
        'exit_price': 728.38,
        'entry_price': 728.61,
        'stop_price': 728.05,
        'initial_stop_price': 728.05,
        'target_price': None,
        'direction': 'LONG',
        'exec_type': 'C',
        'entry_exec_type': 'C',
        'stop_exec_type': 'L',
        'target_exec_type': 'L',
        'exit_trigger': 'utv4_bear_flip',
        'r_multiple': -0.41,
        'win': False,
        'pnl': -0.23,
        'hifi_resolved': False,
    }])


def test_signal_exit_never_rewritten_by_stop_walker(monkeypatch):
    """The sid-302 reproduction: stop level crosses AFTER the signal
    exit (inside the walker's padded window) — the trade must stay a
    signal exit with its original price and timestamp."""
    # 1s lows: above the 728.05 stop until 15s in, then cross — i.e.
    # the cross happens after the 14:48:00 signal exit.
    lows = [728.30] * 14 + [728.00] * 6
    monkeypatch.setattr(
        bts, 'fetch_1s_bars_for_window',
        lambda *a, **k: _bars_1s('2026-06-11T14:48:00+00:00', 20, lows),
        raising=False)
    import data_loader
    monkeypatch.setattr(
        data_loader, 'fetch_1s_bars_for_window',
        lambda *a, **k: _bars_1s('2026-06-11T14:48:00+00:00', 20, lows))

    out = bts._hifi_resolve_trades(
        _trade('utv4_bear_flip', '2026-06-11T14:48:00+00:00'),
        'SPY', '10Sec')
    row = out.iloc[0]
    assert row['exit_reason'] == 'utv4_bear_flip', (
        "signal exit was rewritten by the stop walker")
    assert row['exit_fill_ts'] == '2026-06-11T14:48:00+00:00'
    assert abs(float(row['exit_price']) - 728.38) < 1e-9
    assert abs(float(row['r_multiple']) - (-0.41)) < 1e-9


def test_genuine_stop_exit_still_refined(monkeypatch):
    """Real stop exits keep their sub-second Hi-Fi refinement."""
    lows = [728.30] * 5 + [728.00] * 15  # crosses 5s into the exit bar
    import data_loader
    monkeypatch.setattr(
        data_loader, 'fetch_1s_bars_for_window',
        lambda *a, **k: _bars_1s('2026-06-11T14:48:10+00:00', 20, lows))

    out = bts._hifi_resolve_trades(
        _trade('stop_loss', '2026-06-11T14:48:10+00:00'),
        'SPY', '10Sec')
    row = out.iloc[0]
    assert row['exit_reason'] == 'stop_loss'
    assert bool(row['hifi_resolved']) is True
    # refined to the per-second cross moment (5s into the bar)
    assert row['exit_fill_ts'].startswith('2026-06-11T14:48:15')


def test_time_and_barcount_exits_untouched(monkeypatch):
    import data_loader
    lows = [728.00] * 20  # stop "crossed" the whole window
    monkeypatch.setattr(
        data_loader, 'fetch_1s_bars_for_window',
        lambda *a, **k: _bars_1s('2026-06-11T14:48:00+00:00', 20, lows))
    for reason in ('bar_count_exit', 'time_exit_eod'):
        out = bts._hifi_resolve_trades(
            _trade(reason, '2026-06-11T14:48:00+00:00'), 'SPY', '10Sec')
        assert out.iloc[0]['exit_reason'] == reason
