"""M1c — CONSUMER #1 cutover validation (function-level, infra-robust).

Consumer #1 = strategy_data._build_coarse_secondary_from_1min. The
RORT_RESAMPLED_STORE_READ flag changes ONLY this function's output (its coarse
>=1Hour secondary OHLCV); everything downstream (indicators, engine, trades) is a
pure function of that output. So proving the function returns BYTE-IDENTICAL output
with the flag ON vs OFF proves the backtest trades are byte-identical ON vs OFF —
the main-session acceptance bar — without running the full engine (which loads user
config over PostgREST, currently flaky under the parallel bug-hunt's DB load).

It also proves the store ACTUALLY SERVED (ON output is the store read, not a silent
fallback) — otherwise ON==OFF would pass trivially.

Uses only direct-PG (store) + load_market_data (cache/Polygon) — no PostgREST config
loads. Canary 338: TSLA / Extended Hours / 1Day coarse gate.

SEQUENCING: populates the store for 338's window first (warm 1Min cache → store_window).
WRITES: resampled_bar_cache rows + normal bar_cache delta-fetch. No live/Worker touch.
"""
from __future__ import annotations
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_SRC, ".env"))
except Exception:
    pass
os.environ.setdefault("USE_DB", "true")

ADMIN = "19d47e46-f718-49a6-af32-5f5407f5b170"
CANARY = 338


def _get_strat_retry(sid, attempts=4):
    """Single PostgREST read with retry — dodges the transient 204s the shared
    Supabase throws under concurrent load (infra, not our change)."""
    from db import get_strategy_by_id_admin
    last = None
    for i in range(attempts):
        try:
            s = get_strategy_by_id_admin(sid, ADMIN)
            if s:
                return s
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"could not load strat {sid} after {attempts} tries: {last}")


def main() -> int:
    import importlib
    import pandas as pd
    os.environ["RORT_RIGHTSIZE_WARMUP"] = "1"
    os.environ["BAR_CACHE_ENABLED"] = "1"
    os.environ["RORT_COARSE_SECONDARY_FROM_1MIN"] = "1"

    import bar_cache, resampled_bar_store as rbs, data_loader, strategy_data as sd
    from data_loader import (load_market_data, get_required_tfs_from_confluence,
                             get_tf_from_label)

    print("=" * 78)
    print(f"M1c CONSUMER #1 validation (function-level) — canary {CANARY}")
    print("=" * 78)
    s = _get_strat_retry(CANARY)
    sym = s["symbol"]
    session = s.get("trading_session", "RTH")
    sectfs = sorted(get_tf_from_label(l)
                    for l in get_required_tfs_from_confluence(s.get("confluence", [])))
    coarse = tuple(t for t in sectfs if sd._tf_seconds_safe(t) >= 3600)
    vs, ve, _ = sd.resolve_visible_window(s)
    secw = max(sd._tf_warmup_days(t, sd.SECONDARY_WARMUP_BARS) for t in coarse)
    start = (vs - pd.Timedelta(days=secw)).to_pydatetime()
    end = ve.to_pydatetime()
    feed = s.get("data_feed", "sip")
    print(f"{sym} coarse={list(coarse)} session={session!r} [{start.date()}..{end.date()}]")

    # 1. warm the 1Min cache so read_bars (store source) == load_market_data (resample
    #    source), then populate the store WINDOW-REPLACE.
    os.environ["RORT_RESAMPLED_STORE_WRITE"] = "1"
    load_market_data(sym, start_date=start, end_date=end, timeframe="1Min",
                     session=session)
    for tf in coarse:
        w = rbs.store_window(sym, tf, session, start, end)
        print(f"  populate {sym}/{tf}/{session}: {w}")

    # 2. OFF: resample path (store-read disabled).
    os.environ.pop("RORT_RESAMPLED_STORE_READ", None)
    importlib.reload(rbs); importlib.reload(sd)
    off = sd._build_coarse_secondary_from_1min(sym, coarse, start, end, session, feed)

    # 3. ON: store-read path (byte-identity-gated).
    os.environ["RORT_RESAMPLED_STORE_READ"] = "1"
    importlib.reload(rbs); importlib.reload(sd)
    on = sd._coarse_secondary_from_store(sym, coarse, start, end, session)
    on_full = sd._build_coarse_secondary_from_1min(sym, coarse, start, end, session, feed)

    # --- assertions ---
    print("\n" + "-" * 60)
    served = on is not None and all(tf in on for tf in coarse)
    print(f"store SERVED (ON came from store, not fallback): {'✓' if served else '✗'}")

    all_ok = served
    for tf in coarse:
        a = off.get(tf) if off else None
        b = on_full.get(tf) if on_full else None
        if a is None or b is None:
            print(f"  {tf}: OFF={None if a is None else len(a)} ON={None if b is None else len(b)} ✗ missing")
            all_ok = False
            continue
        cols = ["open", "high", "low", "close", "volume"]
        idx_ok = a.index.equals(b.index)
        cell_diffs = 0
        if idx_ok:
            for c in cols:
                cell_diffs += int((a[c].values != b[c].values).sum())
        ok = idx_ok and cell_diffs == 0
        all_ok = all_ok and ok
        print(f"  {tf}: OFF n={len(a)} ON n={len(b)} index_equal={idx_ok} "
              f"ohlcv_cell_diffs={cell_diffs if idx_ok else 'N/A'} -> {'✓' if ok else '✗'}")

    print("\n" + "#" * 78)
    if all_ok:
        print("✅ CONSUMER #1 GREEN — _build_coarse_secondary_from_1min output BYTE-IDENTICAL "
              "with store-read ON vs OFF, and the store SERVED (byte-identity-gated).")
        print("   => downstream trades are byte-identical ON==OFF by construction.")
    else:
        print("❌ CONSUMER #1 not green — see above.")
    print("#" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
