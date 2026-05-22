"""Phase 2 data-worker streaming — verification.

Deterministic unit tests (no network) for the circuit breaker, the
new-trade filter, store→frame resampling, and the metrics counters —
plus `parity_check()`, the documented store-fed vs REST-fed parity
procedure (the cutover gate) which needs Polygon access + a real TSLA
strategy and is run manually during the staging window.

Run:  python src/_test_data_worker_streaming_parity.py
      python src/_test_data_worker_streaming_parity.py <strategy_id> <user_id>
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

_PASS = 0
_FAIL = 0


def _check(name: str, ok: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if ok:
        _PASS += 1
        print(f"  PASS  {name}")
    else:
        _FAIL += 1
        print(f"  FAIL  {name}  {detail}")


# ─────────────────────────────────────────────────────────────────────
# 1. Circuit breaker state machine
# ─────────────────────────────────────────────────────────────────────

def test_circuit_breaker():
    print("test_circuit_breaker")
    from data_worker_circuit import DBCircuitBreaker, CLOSED, OPEN, HALF_OPEN

    cb = DBCircuitBreaker(failure_threshold=3, base_cooldown_s=0.2,
                          max_cooldown_s=1.0)
    _check("starts CLOSED", cb.state == CLOSED)
    _check("CLOSED allows", cb.allow() is True)

    cb.record_failure()
    cb.record_failure()
    _check("2 failures still CLOSED", cb.state == CLOSED)
    cb.record_failure()
    _check("3rd failure trips OPEN", cb.state == OPEN)
    _check("OPEN blocks calls", cb.allow() is False)
    _check("open_count incremented", cb.open_count == 1)

    time.sleep(0.25)  # past the cooldown
    _check("post-cooldown allow → HALF_OPEN probe", cb.allow() is True)
    _check("state is HALF_OPEN", cb.state == HALF_OPEN)

    cb.record_failure()  # probe fails
    _check("failed probe re-OPENs", cb.state == OPEN)

    time.sleep(1.1)  # cooldown doubled, then capped
    cb.allow()  # → HALF_OPEN
    cb.record_success()  # probe succeeds
    _check("successful probe → CLOSED", cb.state == CLOSED)
    _check("success resets failure count", cb.allow() is True)


# ─────────────────────────────────────────────────────────────────────
# 2. New-trade filter
# ─────────────────────────────────────────────────────────────────────

def test_new_trade_filter():
    print("test_new_trade_filter")
    from data_worker_engine import _filter_new_closed_trades

    trades = pd.DataFrame([
        {'entry_fill_ts': '2026-05-22T14:00:00Z',
         'exit_fill_ts': '2026-05-22T14:05:00Z'},   # old — before anchor
        {'entry_fill_ts': '2026-05-22T15:00:00Z',
         'exit_fill_ts': '2026-05-22T15:05:00Z'},   # new — closed
        {'entry_fill_ts': '2026-05-22T15:10:00Z',
         'exit_fill_ts': None},                     # new — but OPEN
    ])
    out = _filter_new_closed_trades(trades, '2026-05-22T14:30:00Z')
    _check("keeps only new closed trades", len(out) == 1,
           f"got {len(out)}")
    if len(out) == 1:
        _check("kept the right row",
               str(out.iloc[0]['entry_fill_ts']).startswith('2026-05-22T15:00'))

    out_all = _filter_new_closed_trades(trades, None)
    _check("no anchor → all closed (open dropped)", len(out_all) == 2,
           f"got {len(out_all)}")

    _check("empty frame → empty",
           len(_filter_new_closed_trades(pd.DataFrame(), None)) == 0)


# ─────────────────────────────────────────────────────────────────────
# 3. Store → injected frames
# ─────────────────────────────────────────────────────────────────────

def _synthetic_1s(minutes: int) -> pd.DataFrame:
    """A deterministic 1-second OHLCV frame, `minutes` long, UTC-indexed."""
    n = minutes * 60
    end = datetime(2026, 5, 22, 15, 0, 0, tzinfo=timezone.utc)
    idx = pd.date_range(end=end, periods=n, freq='1s', tz='UTC',
                        name='timestamp')
    price = pd.Series(range(n), dtype=float) + 100.0
    return pd.DataFrame({
        'open': price.values, 'high': price.values + 0.5,
        'low': price.values - 0.5, 'close': price.values + 0.1,
        'volume': [10.0] * n, 'vwap': price.values,
        'trade_count': [3] * n,
    }, index=idx)


def test_build_injected_frames():
    print("test_build_injected_frames")
    from bar_store import SymbolBarStore
    from data_worker_engine import build_injected_frames

    store = SymbolBarStore('TSLA', window_minutes=90)
    store.upsert_bars(_synthetic_1s(20))   # 20 min of 1s bars
    until = datetime(2026, 5, 22, 15, 0, 0, tzinfo=timezone.utc)

    primary, sec = build_injected_frames(store, '1Min', ('5Min',), until)
    _check("primary frame returned", primary is not None and len(primary) > 0)
    if primary is not None:
        _check("primary ~20 1-min bars", 18 <= len(primary) <= 21,
               f"got {len(primary)}")
        _check("primary has OHLCV",
               all(c in primary.columns for c in
                   ('open', 'high', 'low', 'close', 'volume')))
        _check("primary bars at/before until",
               bool((primary.index <= pd.Timestamp(until)).all()))
    _check("secondary 5Min present", '5Min' in sec and len(sec['5Min']) > 0)
    if '5Min' in sec:
        _check("5Min ~4 bars", 3 <= len(sec['5Min']) <= 5,
               f"got {len(sec['5Min'])}")

    # coverage() — the cheap accessor the tick uses each cycle.
    cov = store.coverage()
    _check("coverage count matches", cov['count'] == 20 * 60)
    _check("coverage first_ts/last_ts set",
           cov['first_ts'] is not None and cov['last_ts'] is not None)


# ─────────────────────────────────────────────────────────────────────
# 4. Streaming metrics
# ─────────────────────────────────────────────────────────────────────

def test_streaming_metrics():
    print("test_streaming_metrics")
    from data_worker_engine import StreamingMetrics

    m = StreamingMetrics()
    m.record_tick(errored=False, latency=0.1, trades=2)
    m.record_tick(errored=True)
    m.record_catchup()
    m.record_catchup(no_baseline=True)
    m.record_flush(ok=True)
    m.record_flush(ok=False)
    snap = m.snapshot()
    _check("ticks_total", snap['ticks_total'] == 2)
    _check("ticks_errored", snap['ticks_errored'] == 1)
    _check("trades_written_total", snap['trades_written_total'] == 2)
    _check("catchups_run", snap['catchups_run'] == 2)
    _check("catchups_skipped_no_baseline",
           snap['catchups_skipped_no_baseline'] == 1)
    _check("snapshot_flushes_total", snap['snapshot_flushes_total'] == 2)
    _check("snapshot_flush_errors", snap['snapshot_flush_errors'] == 1)


# ─────────────────────────────────────────────────────────────────────
# 5. Parity check — the cutover gate (manual, needs Polygon + a strategy)
# ─────────────────────────────────────────────────────────────────────

def parity_check(strategy_id: int, user_id: str):
    """Store-fed vs REST-fed parity for one strategy — the cutover gate.

    Pulls a fresh 90-min Polygon 1s window into a SymbolBarStore, runs
    the data-worker's store-fed window, and compares against the REST
    path (`services.get_strategy_trades_for_window`) over the same
    window from the same snapshot. Must produce identical trades +
    byte-identical snapshot before the data-worker is cut over to the
    real backtest lane.
    """
    print(f"parity_check  strategy_id={strategy_id}")
    import os
    if not os.getenv("POLYGON_API_KEY"):
        print("  SKIP  POLYGON_API_KEY not set")
        return
    from db import get_strategy_by_id_admin, get_admin_client
    from bar_store import SymbolBarStore
    from data_worker_ingest import prime_store
    from data_worker_engine import StrategyEngineState, classify_strategy, \
        run_store_fed_window
    import services as svc

    strat = get_strategy_by_id_admin(strategy_id, user_id)
    if strat is None:
        print(f"  SKIP  strategy {strategy_id} not found")
        return

    state = StrategyEngineState(strategy_id, user_id, strat)
    classify_strategy(state)
    if not state.streaming_eligible:
        print(f"  SKIP  ineligible: {state.ineligible_reason}")
        return

    # Pull the snapshot the cron persisted.
    client = get_admin_client()
    resp = client.table('strategies').select('config') \
        .eq('id', strategy_id).eq('user_id', user_id).single().execute()
    cfg = dict((resp.data or {}).get('config') or {})
    state.snapshot_b64 = cfg.get('engine_snapshot_b64')
    if not state.snapshot_b64:
        print("  SKIP  no engine_snapshot_b64 — run catch-up first")
        return

    store = SymbolBarStore(state.symbol, window_minutes=90)
    prime_store(store, 90)
    until = datetime.now(timezone.utc) - timedelta(minutes=16)

    trades_store, b64_store, last_bar, status = run_store_fed_window(
        state, store, until)
    print(f"  store-fed: status={status} trades={len(trades_store)} "
          f"last_bar={last_bar}")

    # REST path over the same window.
    from unified_engine import compute_backtest_fingerprint
    fp = compute_backtest_fingerprint(strat)
    since = pd.Timestamp(state.last_bar_ts) if state.last_bar_ts else \
        (until - timedelta(hours=2))
    trades_rest, b64_rest = svc.get_strategy_trades_for_window(
        strat, since_dt=since.to_pydatetime(), until_dt=until,
        resume_snapshot_b64=state.snapshot_b64,
        expected_fingerprint=fp, expected_model_id=state.data_source,
        return_snapshot=True)

    _check("parity: same trade count",
           len(trades_store) == len(trades_rest),
           f"store={len(trades_store)} rest={len(trades_rest)}")
    _check("parity: byte-identical snapshot", b64_store == b64_rest,
           "snapshot blobs differ — store bars diverge from REST bars")


def main():
    if len(sys.argv) >= 3:
        parity_check(int(sys.argv[1]), sys.argv[2])
        return
    print("=" * 60)
    print("Phase 2 data-worker streaming — deterministic tests")
    print("=" * 60)
    test_circuit_breaker()
    test_new_trade_filter()
    test_build_injected_frames()
    test_streaming_metrics()
    print("-" * 60)
    print(f"  {_PASS} passed, {_FAIL} failed")
    print("  (run with `<strategy_id> <user_id>` for the parity gate)")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
