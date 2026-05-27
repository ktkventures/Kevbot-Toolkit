#!/usr/bin/env python3
"""Regression test for the Hi-Fi Pass 2 exit-timestamp bug
(discovered 2026-05-26 on sid 246).

Bug: `_hifi_resolve_trades` exit walker (backtest_service.py:811-867)
computed the sub-second `new_time` from `_walk_1s_for_exit` and used it
only for `hold_time_seconds` — it never persisted the timestamp back
to `exit_fill_ts` / `exit_time`. The entry walker correctly persists
(line 685), and the signal-exit walker correctly persists (line 774);
only the stop/target walker was missing the write.

This test:
  - Builds a synthetic trade that closes via stop_loss
  - Patches `fetch_1s_bars_for_window` to return 1-sec bars where the
    stop level is hit at a specific sub-second moment
  - Calls `_hifi_resolve_trades`
  - Asserts the refined trade's exit_fill_ts is sub-second (NOT
    bar-aligned)

Run:
    .venv/bin/python src/test_hifi_exit_timestamp.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

os.environ["USE_DB"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd


def _mock_1s_bars(start: pd.Timestamp, stop_level: float, hit_at_second: int):
    """Build a 1-second OHLCV df where the stop level gets touched at
    `hit_at_second` seconds after `start`. Before that second: price
    above the stop (LONG trade). At/after that second: price below.
    """
    rows = []
    for s in range(60):
        ts = start + pd.Timedelta(seconds=s)
        if s < hit_at_second:
            # Above stop — no hit
            rows.append({
                "open": stop_level + 0.5, "high": stop_level + 0.6,
                "low":  stop_level + 0.4, "close": stop_level + 0.5,
                "volume": 100,
            })
        else:
            # Stop hit (low dips below stop_level)
            rows.append({
                "open": stop_level - 0.1, "high": stop_level + 0.1,
                "low":  stop_level - 0.5, "close": stop_level - 0.1,
                "volume": 100,
            })
    return pd.DataFrame(rows, index=pd.DatetimeIndex(
        [start + pd.Timedelta(seconds=s) for s in range(60)],
        tz="UTC",
    ))


def test_exit_fill_ts_persisted_sub_second():
    """Regression: exit_fill_ts must be sub-second after Hi-Fi Pass 2
    on a stop_loss exit."""
    from api.services.backtest_service import _hifi_resolve_trades

    entry_ts = pd.Timestamp("2026-05-04T15:35:00+00:00")
    exit_bar_start = pd.Timestamp("2026-05-04T15:40:00+00:00")
    stop_level = 432.50
    target_level = 440.00
    hit_at_second = 23  # stop hit at :23 of the exit bar

    trades_df = pd.DataFrame([{
        "entry_fill_ts": entry_ts.isoformat(),
        "exit_fill_ts": exit_bar_start.isoformat(),  # bar-aligned input
        "entry_time": entry_ts.isoformat(),
        "exit_time": exit_bar_start.isoformat(),
        "direction": "LONG",
        "entry_price": 433.50,
        "exit_price": exit_bar_start.value % 100,  # placeholder
        "stop_price": stop_level,
        "target_price": target_level,
        "exit_reason": "stop_loss",
        "exec_type": "C",
        "stop_exec_type": "L",  # L-type stops are eligible for refinement
        "target_exec_type": "L",
        "initial_stop_price": stop_level,
    }])

    mock_bars = _mock_1s_bars(exit_bar_start, stop_level, hit_at_second)

    # The fetch is imported inside _hifi_resolve_trades from data_loader,
    # so patch at the source module.
    with patch(
        "data_loader.fetch_1s_bars_for_window",
        return_value=mock_bars,
    ):
        refined = _hifi_resolve_trades(
            trades_df, symbol="TEST", timeframe="5Min", bar_df=None,
        )

    out = refined.iloc[0]
    new_exit = str(out.get("exit_fill_ts") or "")
    print(f"refined exit_fill_ts = {new_exit}")
    print(f"refined exit_time    = {out.get('exit_time')}")
    print(f"refined exit_price   = {out.get('exit_price')}")
    print(f"refined exit_reason  = {out.get('exit_reason')}")
    print(f"refined hold_seconds = {out.get('hold_time_seconds')}")

    # The seconds portion of the persisted exit_fill_ts must NOT be :00.
    # On the bug, exit_fill_ts stays as exit_bar_start (:00 seconds).
    # After the fix, exit_fill_ts should reflect the sub-second moment
    # the stop was hit (:23).
    assert new_exit, "exit_fill_ts missing entirely"
    # Parse and check seconds; should match hit_at_second.
    parsed = pd.Timestamp(new_exit)
    assert parsed.second == hit_at_second, (
        f"FAIL: expected exit_fill_ts.second == {hit_at_second}, "
        f"got {parsed.second}. exit_fill_ts={new_exit!r}. "
        f"Bug: exit walker didn't persist new_time."
    )
    print(f"PASS: exit_fill_ts persisted with sub-second precision "
          f"(:{parsed.second:02d})")


def test_mass_builder_hifi_refine_result_shape():
    """Mass Builder _hifi_refine_result helper integration: given a
    result dict in the Mass Builder shape, it should re-run Pass 2,
    refresh kpis/equity_curve/stored_trades, and stamp a `hifi`
    summary. Smoke-tests the wire-up at the top-K refinement step.

    Verifies:
      - Refined result has the same shape (config, kpis, equity_curve,
        stored_trades) it had on input
      - New field `hifi` present with int counts
      - stored_trades' exit_fill_ts is now sub-second
    """
    from mass_builder import _hifi_refine_result

    entry_ts = pd.Timestamp("2026-05-04T15:35:00+00:00")
    exit_bar_start = pd.Timestamp("2026-05-04T15:40:00+00:00")
    stop_level = 432.50
    hit_at_second = 31

    # Synthesize a Mass Builder result dict
    result = {
        "config": {
            "symbol": "TEST", "timeframe": "5Min", "direction": "LONG",
            "backtest_model": "rest_hifi", "starting_balance": 10000,
            "risk_per_trade": 100,
        },
        "kpis": {"total_trades": 1, "win_rate": 0.0, "total_r": -1.0,
                  "profit_factor": 0.0, "daily_r": -0.01},
        "equity_curve": [-1.0],
        "stored_trades": [{
            # mass_builder._serialize_trades shape
            "data_source": "backtest_rest_hifi",
            "entry_time": entry_ts.isoformat(),
            "exit_time": exit_bar_start.isoformat(),
            "entry_fill_ts": entry_ts.isoformat(),
            "exit_fill_ts": exit_bar_start.isoformat(),
            "direction": "LONG",
            "entry_price": 433.50,
            "exit_price": stop_level,
            "stop_price": stop_level,
            "target_price": 440.00,
            "exit_reason": "stop_loss",
            "exec_type": "C",
            "r_multiple": -1.0,
            "win": False,
        }],
        "status": "active",
        "confluence_str": "None",
    }

    mock_bars = _mock_1s_bars(exit_bar_start, stop_level, hit_at_second)

    with patch(
        "data_loader.fetch_1s_bars_for_window",
        return_value=mock_bars,
    ):
        refined = _hifi_refine_result(
            result,
            bar_df=None,  # no signal-exit refinement needed for stop_loss
            visible_start=entry_ts - pd.Timedelta(days=90),
            trading_days=90,
        )

    print("--- mass-builder integration ---")
    print(f"refined['hifi']: {refined.get('hifi')}")
    print(f"refined['kpis']: {refined.get('kpis')}")
    print(f"refined['stored_trades'][0]['exit_fill_ts']: "
          f"{refined['stored_trades'][0].get('exit_fill_ts')}")

    # Assertions
    assert "hifi" in refined, "Expected 'hifi' field on refined result"
    hifi = refined["hifi"]
    assert isinstance(hifi.get("entries_refined"), int)
    assert isinstance(hifi.get("exits_refined"), int)
    assert isinstance(hifi.get("signal_exits_refined"), int)
    assert isinstance(hifi.get("total_refined"), int)
    assert hifi["exits_refined"] >= 1, (
        f"Expected >=1 exits_refined (stop should have moved), "
        f"got {hifi}")
    assert hifi["total_refined"] >= 1

    new_exit_ts = refined["stored_trades"][0].get("exit_fill_ts")
    parsed = pd.Timestamp(new_exit_ts)
    assert parsed.second == hit_at_second, (
        f"Expected stored_trades[0]['exit_fill_ts'] seconds to be "
        f"{hit_at_second}, got {parsed.second} ({new_exit_ts})")

    # KPIs should be present and reasonable (calculate_kpis output)
    new_kpis = refined.get("kpis", {})
    assert "total_trades" in new_kpis
    assert "total_r" in new_kpis
    assert "win_rate" in new_kpis
    print(f"PASS: mass-builder integration — exits_refined={hifi['exits_refined']} "
          f"exit_fill_ts=:{parsed.second:02d}")


if __name__ == "__main__":
    try:
        test_exit_fill_ts_persisted_sub_second()
        test_mass_builder_hifi_refine_result_shape()
    except AssertionError as e:
        print(f"\nREGRESSION TEST FAILED:\n  {e}")
        sys.exit(1)
    print("\nAll tests passed.")
