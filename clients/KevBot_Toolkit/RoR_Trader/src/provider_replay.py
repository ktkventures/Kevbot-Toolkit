"""PROVIDER-EVENT REPLAY v1 (Brandon audit rec #6 — synthetic-event tier).

Drives the REAL live pipeline — `SymbolHub.on_second_bar` (per-second A events)
→ real BarBuilders/aggregation → `on_polygon_bar` (AM 1Min, arriving ~2s after
the minute) → `flush_stale_bars` on a simulated 5s wall-clock → grace fires →
`_fire_alert` dispatch — instead of calling `on_bar_close` directly like
`replay_harness`. This exercises the CONSTRUCTION layer (builder aggregation,
close_on_boundary, flush-vs-fan-out ordering, grace, >=60s dispatch, canonical
fine-TF state) that the bar-replay is structurally blind to (the P0/P1/grace
bug class).

TIER NOTE: events are SYNTHESIZED from REST 1Sec/1Min data — real WS artifacts
(reconnects, duplicates, late frames) need a raw-event recorder on the Worker
(follow-up in Workplan_Remaining_Fixes_2026-07-16). Mirror the ARMED flag stack
in the environment when running (same rule as every offline engine tool).

Usage (from src/, ARMED flags in env):
    ../.venv/bin/python provider_replay.py --sids 327,328,329 \
        --since 2026-07-16T14:00:00Z --until 2026-07-16T15:30:00Z
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.argv, _argv = sys.argv[:1], sys.argv  # keep RH's import-time argparse quiet
import replay_harness as RH  # noqa: E402  (env SOP: USE_DB, admin ctx, packs, flags)
sys.argv = _argv

from ralph_engine import SymbolHub, StrategyMonitor  # noqa: E402
from replay_harness import (  # noqa: E402
    get_strategy_by_id_admin, ADMIN, warmup_asof, OHLC)
from data_loader import load_market_data  # noqa: E402

AM_LATENCY_S = 2.0     # AM 1Min bar arrives ~2s after the minute closes
A_LATENCY_S = 1.0      # per-second A event arrives ~1s after its second
FLUSH_EVERY_S = 5.0    # periodic-loop flush cadence


def _bar_dict(ts, row):
    return {'timestamp': ts.isoformat(),
            **{c: float(row[c]) for c in OHLC}}


def _neuter_prod_writes():
    """The REAL pipeline writes to prod live_bars (on_polygon_bar/flush write
    paths; seed reconcile). write_bar is existence-checked+ignore_duplicates so
    replaying existing windows is a no-op — but a SIM must NEVER write (a gap
    window would get sim-flavored inserts). No-op them for this process."""
    import live_bars_writer as LBW
    LBW.write_bar = lambda *a, **k: None
    if hasattr(LBW, 'reconcile_seeded_history'):
        LBW.reconcile_seeded_history = lambda *a, **k: None


def run(sids, since, until, sb):
    _neuter_prod_writes()
    strats = [get_strategy_by_id_admin(s, ADMIN) for s in sids]
    syms = {s['symbol'] for s in strats}
    assert len(syms) == 1, f"v1 is single-symbol per run, got {syms}"
    sym = syms.pop()

    hub = SymbolHub(sym)
    monitors = []
    for strat in strats:
        m = StrategyMonitor(strat, None, general_packs=[])
        hub.add_monitor(m)
        monitors.append(m)
    hub.finalize_shadow_engines()

    # Warmup exactly like boot: monitors + shadows + seed builders/history.
    m1_warm = warmup_asof(sym, 60, monitors[0].session, since)
    if len(m1_warm):
        hub.seed_history(60, m1_warm)          # 1Min builder = 5b's canonical source
    for m in monitors:
        wdf = warmup_asof(sym, m.tf_seconds, m.session, since)
        if len(wdf) == 0:
            print(f"  !! sid {m.strat_id}: empty warmup — skipping run")
            return None
        hub.seed_history(m.tf_seconds, wdf)
        m.warmup(wdf)
    for (sec_tf, sec_sess), shadow in list(hub._shadow_engines.items()):
        sdf = warmup_asof(sym, sec_tf, sec_sess, since,
                          days=7 if sec_tf < 3600 else 60)
        if len(sdf):
            shadow.warmup(sdf)
            hub.seed_mtf_from_shadow(sec_tf, sec_sess)

    # ---- synthesize the provider event stream ----
    sec1 = load_market_data(sym, start_date=since, end_date=until,
                            timeframe='1Sec', feed='sip',
                            session=monitors[0].session)
    # AM events = the RECORDED decision-time 1Min bars (live_bars first_* —
    # what AM actually delivered), NOT a REST reconstruction: a pivot-level
    # REST-vs-decision difference flips path-dependent interps (SWING counts)
    # and fakes a construction divergence that live never had.
    m1 = RH.decision_time_bars(RH.get_admin_client(), sym, 60, since, until)
    events = []   # (arrival_dt, order, kind, payload)
    for ts, row in (sec1[OHLC] if sec1 is not None and len(sec1)
                    else pd.DataFrame()).iterrows():
        events.append((ts.to_pydatetime() + timedelta(seconds=A_LATENCY_S),
                       0, 'A', _bar_dict(ts, row)))
    for ts, row in (m1[OHLC] if m1 is not None and len(m1)
                    else pd.DataFrame()).iterrows():
        arrive = ts.to_pydatetime() + timedelta(seconds=60 + AM_LATENCY_S)
        events.append((arrive, 1, 'AM', _bar_dict(ts, row)))
    t = since
    while t <= until + timedelta(seconds=90):
        events.append((t, 2, 'FLUSH', None))
        t += timedelta(seconds=FLUSH_EVERY_S)
    events.sort(key=lambda e: (e[0], e[1]))

    # ---- alert capture through the REAL dispatch path ----
    fired = []

    def cb(sig, strategy, config):
        fired.append({'sid': strategy.get('id'),
                      'side': (sig.get('type') or sig.get('side') or '').lower(),
                      'fill_ts': sig.get('fill_ts') or sig.get('timestamp')})

    cfg: dict = {}
    for arrive, _o, kind, payload in events:
        if kind == 'A':
            hub.on_second_bar(payload, alert_callback=cb, config=cfg)
        elif kind == 'AM':
            hub.on_polygon_bar(payload, 60, alert_callback=cb, config=cfg)
        else:
            hub.flush_stale_bars(arrive, alert_callback=cb, config=cfg)

    # ---- compare per sid: provider entries vs backtest ----
    P = lambda s: datetime.fromisoformat(str(s).replace('Z', '+00:00'))  # noqa
    out = {}
    for sid in sids:
        ent = sorted(P(f['fill_ts']) for f in fired
                     if f['sid'] == sid and 'entry' in f['side'] and f['fill_ts'])
        bt_e, *_ = RH.lanes(sb, sid, since, until)
        paired = RH.pair(ent, sorted(bt_e), 10.0)
        out[sid] = (ent, sorted(bt_e), paired)
        hh = lambda t: t.strftime('%H:%M:%S')  # noqa: E731
        print(f"\n=== sid {sid} (provider-event replay) ===")
        print(f"  backtest ({len(bt_e)}): {[hh(t) for t in sorted(bt_e)]}")
        print(f"  provider ({len(ent)}): {[hh(t) for t in ent]}")
        print(f"  paired@10s: {paired}/{max(len(bt_e), 1)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sids', required=True)
    ap.add_argument('--since', required=True)
    ap.add_argument('--until', required=True)
    a = ap.parse_args()
    sb = RH.get_admin_client()
    run([int(s) for s in a.sids.split(',')],
        datetime.fromisoformat(a.since.replace('Z', '+00:00')),
        datetime.fromisoformat(a.until.replace('Z', '+00:00')), sb)
    return 0


if __name__ == '__main__':
    sys.exit(main())
