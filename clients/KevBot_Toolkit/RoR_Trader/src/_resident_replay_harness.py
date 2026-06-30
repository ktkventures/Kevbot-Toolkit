"""M-RS4 Phase 3 — Step A POSITIVE byte-identity proof (offline, no live market).

Sibling of `_shadow_replay_harness.py`. That harness proved the NEGATIVE result:
snapshot-CHAINED resume (PATH B) is NOT byte-identical to a from-cold run (~4% of
trades drop/change at chunk boundaries) — which is the "append breaks things" root.

This harness proves the POSITIVE premise Phase 3 is built on: a SINGLE warmed engine
fed each bar ONCE, incrementally, never re-windowed / never resumed (the
continuous-resident design) IS byte-identical to a from-cold full recompute.

  PATH A  from-cold one-shot : run_unified_backtest(full_df)            — today's recompute
  PATH C  continuous-resident: ONE UnifiedStrategy, warmed on the same df, fed bar-by-bar
                               via process_bar() — exactly what the shadow-worker will do.

Both run over the SAME prepared df, so the only variable is "one-shot internal loop"
vs "resident external feed." If A == C byte-identical (the trade_snapshot KEY/VAL gate)
across the canaries, the resident design is faithful BY CONSTRUCTION and Step B (build
the service) is unlocked. If they differ, this pinpoints the hidden cross-bar / df-level
state the resident engine must reproduce — i.e. exactly what would otherwise leak.

Scope: this proves the CONTINUOUS equivalence (warm + run as one engine). The cold
bootstrap-boundary heal (warm to T, then feed the tail) is Step D, validated separately.

Usage:  PYTHONPATH=. ../.venv/bin/python _resident_replay_harness.py [SIDS] [DAYS]
  SIDS   comma-separated strategy ids (default a single canary). The Step A GATE is
         >=3 canaries green INCLUDING a sub-minute primary and a secondary-TF-gated one
         (e.g. 338 = sub-min primary + 1Day gate covers both; pair with 267 + a 1Min).
  DAYS   window length anchored on each strategy's latest real backtest trade (default 1).
"""
import os, sys, logging
from datetime import datetime, timedelta, timezone

# Determinism: full warmup load, no secondary-TF snapshot fast path. Both paths
# share ONE df, so this only affects df-prep cost, never the A-vs-C comparison.
os.environ["RORT_SECONDARY_TF_SNAPSHOT"] = "0"
os.environ["USE_DB"] = "true"
from dotenv import load_dotenv
load_dotenv(".env", override=True)
logging.basicConfig(level=logging.WARNING)
for n in ("httpx", "httpcore", "urllib3"):
    logging.getLogger(n).setLevel(logging.ERROR)

import math
SIDS = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [263]
DAYS = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
WARMUP = int(os.environ.get("HARNESS_WARMUP", "300"))
ADMIN_USER = "19d47e46-f718-49a6-af32-5f5407f5b170"

import db
db.set_admin_user_context(ADMIN_USER)
import pack_registry
_np = pack_registry.scan_and_load_all()
print(f"registry: {len(_np) if isinstance(_np, dict) else _np} packs", flush=True)

import pandas as pd
from db import get_admin_client, get_strategy_by_id_admin
from services import (
    prepare_data_with_indicators, get_secondary_tf_map,
)
from data_loader import (
    BARS_PER_DAY, get_required_tfs_from_confluence, get_tf_from_label,
)
import general_packs as gp_module
from unified_engine import (
    run_unified_backtest, UnifiedStrategy, INTRABAR_LEVEL_MAP,
)
from trade_snapshot import KEY_FIELDS, VAL_FIELDS

_BUILTIN_INTERPS = {
    'EMA_STACK', 'EMA_PRICE_POSITION', 'EMA_PRICE_POSITION_V2',
    'MACD_LINE', 'MACD_HISTOGRAM', 'VWAP', 'RVOL', 'UTBOT', 'UTBOT_V2',
}


def keyed(df):
    """trades_df -> {entry|exit : {val fields}} using the trade_snapshot gate fields."""
    out = {}
    if df is None or len(df) == 0:
        return out
    for t in df.to_dict("records"):
        k = "|".join(str(t.get(f)) for f in KEY_FIELDS)
        out[k] = {f: (None if t.get(f) is None else t.get(f)) for f in VAL_FIELDS}
    return out


def prepare_window_df(strat, model, since_dt, until_dt):
    """Mirror get_strategy_trades_for_window's cold (no-resume) df-prep so PATH A
    and PATH C share an identical, fully-warmed df."""
    timeframe = strat.get('timeframe', '1Min')
    req_labels = get_required_tfs_from_confluence(strat.get('confluence', []))
    sec_tfs = tuple(sorted(get_tf_from_label(lbl) for lbl in req_labels))
    all_tfs = [timeframe] + list(sec_tfs)
    bpds = [BARS_PER_DAY.get(tf, 390) for tf in all_tfs]
    binding_bpd = min(bpd for bpd in bpds if bpd > 0) if bpds else 390
    warmup_days = max(1, math.ceil(WARMUP / max(binding_bpd, 0.001) * 365 / 252))

    since_naive = since_dt.replace(tzinfo=None) if since_dt.tzinfo else since_dt
    end_date = until_dt.replace(tzinfo=None) if until_dt.tzinfo else until_dt
    start_date = since_naive - timedelta(days=warmup_days)

    df = prepare_data_with_indicators(
        strat['symbol'], seed=strat.get('data_seed', 42),
        start_date=start_date, end_date=end_date, timeframe=timeframe,
        data_feed="sip", session=strat.get('trading_session', 'RTH'),
        secondary_tfs=sec_tfs, secondary_tf_dfs=None, strat=strat,
        model_override=model,
    )
    if len(df) == 0:
        return df, sec_tfs
    end_clip = pd.Timestamp(end_date)
    if df.index.tz is not None and end_clip.tz is None:
        end_clip = end_clip.tz_localize('UTC')
    elif df.index.tz is None and end_clip.tz is not None:
        end_clip = end_clip.tz_localize(None)
    df = df[df.index <= end_clip]
    return df, sec_tfs


class ResidentEngine:
    """Prototype of the Phase 3 shadow-worker engine: ONE warmed UnifiedStrategy,
    fed bars one at a time. Replicates run_unified_backtest's per-bar input
    construction VERBATIM so the only difference from PATH A is the drive loop
    (external feed vs internal loop). Never re-windows, never resumes."""

    def __init__(self, strategy, general_packs, df, secondary_tf_map):
        self.strat = UnifiedStrategy(strategy, general_packs)
        self.secondary_tf_map = secondary_tf_map
        # Static, per-df column detection — copied from run_unified_backtest.
        self._user_interp_cols = [
            ik for ik in self.strat.trigger_eval.required_interpreters
            if ik not in _BUILTIN_INTERPS and ik in df.columns
        ]
        _required_trig_set = {f'trig_{t}' for t in self.strat.trigger_eval.required_triggers}
        self._user_trig_cols = [
            col for col in df.columns
            if col in _required_trig_set and col.startswith('trig_')
        ]
        self._user_indicator_cols = set()
        for base_trigger in self.strat.trigger_eval.required_triggers:
            for suffix in ('_ib', '_lc', '_cc', '_hm', '_hl'):
                if base_trigger.endswith(suffix):
                    base = base_trigger[:-len(suffix)]
                    if base in INTRABAR_LEVEL_MAP:
                        level_col = INTRABAR_LEVEL_MAP[base].get('column', '')
                        if level_col and level_col in df.columns:
                            self._user_indicator_cols.add(level_col)
                    break

    def feed_bar(self, row, ts):
        """Apply ONE bar incrementally; return the 0-2 trade dicts it closed/opened."""
        bar = {
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row.get('volume', 0)),
            'timestamp': ts,
        }
        mtf_records = None
        if self.secondary_tf_map:
            mtf_set = set()
            for tf_label, cols in self.secondary_tf_map.items():
                for col in cols:
                    val = row.get(col)
                    if val is not None and pd.notna(val):
                        base_interp = col.rsplit('__', 1)[0]
                        mtf_set.add(f"{tf_label}-{base_interp}-{val}")
            if mtf_set:
                mtf_records = mtf_set

        user_pack_data = None
        if self._user_interp_cols or self._user_trig_cols or self._user_indicator_cols:
            up_interps, up_triggers, up_indicators = {}, {}, {}
            for col in self._user_interp_cols:
                val = row.get(col)
                if val is not None and pd.notna(val):
                    up_interps[col] = str(val)
            for col in self._user_trig_cols:
                val = row.get(col)
                up_triggers[col[5:]] = bool(val) if pd.notna(val) else False
            for col in self._user_indicator_cols:
                val = row.get(col)
                if val is not None and pd.notna(val):
                    up_indicators[col] = float(val)
            if up_interps or up_triggers or up_indicators:
                user_pack_data = {'interps': up_interps, 'triggers': up_triggers,
                                  'indicators': up_indicators}

        bar_trades, _ind, _interp, _trig = self.strat.process_bar(
            bar, mtf_records=mtf_records, partial=False, user_pack_data=user_pack_data)
        return bar_trades


def run_sid(sid):
    strat = get_strategy_by_id_admin(sid, ADMIN_USER)
    if not strat:
        print(f"sid {sid}: NOT FOUND for admin user — skipping", flush=True)
        return None
    tf = strat.get("timeframe")
    model = strat.get("backtest_model")

    # Anchor on the strategy's latest real backtest trade (settled RTH activity).
    c = get_admin_client()
    mx = (c.table("trades").select("entry_fill_ts").eq("strategy_id", sid)
          .like("data_source", "backtest_%").order("entry_fill_ts", desc=True)
          .limit(1).execute().data or [])
    if not mx:
        print(f"sid {sid}: no backtest trades to anchor on — skipping", flush=True)
        return None
    max_entry = datetime.fromisoformat(str(mx[0]["entry_fill_ts"]).replace("Z", "+00:00"))
    if max_entry.tzinfo is None:
        max_entry = max_entry.replace(tzinfo=timezone.utc)
    T_end = max_entry + timedelta(hours=1)
    T0 = T_end - timedelta(days=DAYS)
    print(f"\n--- sid={sid} tf={tf} model={model}  window {T0.date()}->{T_end.date()} "
          f"({DAYS}d, warmup={WARMUP}) ---", flush=True)

    df, sec_tfs = prepare_window_df(strat, model, T0, T_end)
    if len(df) < 2:
        print(f"sid {sid}: only {len(df)} bars in window — skipping", flush=True)
        return None
    sec_tf_map = get_secondary_tf_map(df)
    sec_tf_map = sec_tf_map if sec_tf_map else None
    enabled_gen = gp_module.get_enabled_general_packs(gp_module.load_general_packs())
    print(f"    df: {len(df)} bars  secondary_tfs={sec_tfs or '()'}", flush=True)

    import time
    # PATH A — from-cold one-shot
    t = time.time()
    df_A, _ = run_unified_backtest(
        df, strat, general_packs=enabled_gen,
        secondary_tf_map=sec_tf_map, include_open_position=False,
        last_bar_partial=False)
    A = keyed(df_A)
    ta = time.time() - t

    # PATH C — continuous-resident, bar-by-bar over the SAME df
    t = time.time()
    eng = ResidentEngine(strat, enabled_gen, df, sec_tf_map)
    trades_C = []
    for i in range(len(df)):
        trades_C.extend(eng.feed_bar(df.iloc[i], df.index[i]))
    C = keyed(pd.DataFrame(trades_C) if trades_C else pd.DataFrame())
    tc = time.time() - t

    ka, kc = set(A), set(C)
    added = kc - ka
    removed = ka - kc
    changed = [k for k in (ka & kc) if A[k] != C[k]]
    ok = not added and not removed and not changed
    print(f"    PATH A (one-shot)   : {len(A):4d} trades  {ta:5.1f}s", flush=True)
    print(f"    PATH C (resident)   : {len(C):4d} trades  {tc:5.1f}s", flush=True)
    print(f"    A={len(A)} C={len(C)} added(C-only)={len(added)} "
          f"removed(A-only)={len(removed)} changed={len(changed)}", flush=True)
    if ok:
        print(f"    ✅ sid {sid}: BYTE-IDENTICAL — resident feed == from-cold.", flush=True)
    else:
        print(f"    ❌ sid {sid}: DIVERGENCE — resident feed != from-cold:", flush=True)
        for k in list(removed)[:3]:
            print(f"        A-only: {k}\n            {A[k]}", flush=True)
        for k in list(added)[:3]:
            print(f"        C-only: {k}\n            {C[k]}", flush=True)
        for k in changed[:5]:
            diffs = {f: (A[k][f], C[k][f]) for f in VAL_FIELDS if A[k][f] != C[k][f]}
            print(f"        changed {k}: {diffs}", flush=True)
    return ok


results = {}
for sid in SIDS:
    try:
        results[sid] = run_sid(sid)
    except Exception as e:
        import traceback
        print(f"sid {sid}: ERROR {e}", flush=True)
        traceback.print_exc()
        results[sid] = None

print("\n" + "=" * 64)
print("STEP A GATE — continuous-resident byte-identity")
green = [s for s, r in results.items() if r is True]
red = [s for s, r in results.items() if r is False]
skipped = [s for s, r in results.items() if r is None]
print(f"  green={green}  red={red}  skipped={skipped}")
if red:
    print("  ❌ GATE RED — do NOT build the shadow-worker service yet.")
elif len(green) >= 3:
    print("  ✅ GATE GREEN (>=3 canaries) — Step B (scaffold service) unlocked.")
elif green:
    print(f"  🟡 {len(green)} green, none red — add canaries to reach the >=3 gate "
          "(incl. a sub-minute + a secondary-TF-gated strategy).")
else:
    print("  ⚠️  no canaries evaluated.")
print("=" * 64)
