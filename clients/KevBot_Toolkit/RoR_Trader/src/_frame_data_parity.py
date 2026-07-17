"""M-RS5a Phase-1 data-level gate (CORRECTED reference): prove the resident-frame DELTA
read feeds the SAME new bars each poll as the full-prep path — the faithful comparison,
since BOTH paths resample source up to the poll's `until` and both keep the partial
right-edge bucket at that boundary (the nightly from-cold recompute is the completeness
backstop). Compares, per step (poll):
    delta_new = load_market_data(sym, start=cursor, end=bound)[index > cursor]   # resident
    full_new  = load_market_data(sym, start=A,      end=bound)[index > cursor]   # full-prep
must be byte-identical. Read-only (no_backfill), no DB writes. (An earlier version wrongly
compared the accumulated delta walk to a single all-complete full read — that froze
intermediate partial boundary buckets and produced false mismatches on 5Min/10Sec.)
"""
import os, sys
os.environ["USE_DB"] = "true"
from dotenv import load_dotenv
load_dotenv(".env", override=True)
import logging
logging.basicConfig(level=logging.ERROR)
sys.path.insert(0, ".")
from datetime import datetime, timedelta, timezone
import pandas as pd
import db
db.set_admin_user_context("19d47e46-f718-49a6-af32-5f5407f5b170")
import pack_registry
pack_registry.scan_and_load_all()
from data_loader import load_market_data

OHLCV = ["open", "high", "low", "close", "volume"]


def _read(sym, tf, A, B, session="RTH"):
    d = load_market_data(sym, start_date=A, end_date=B, timeframe=tf, feed="sip",
                         session=session, no_backfill=True)
    if d is None or len(d) == 0 or not isinstance(d.index, pd.DatetimeIndex):
        return None
    if d.index.tz is None:
        d.index = d.index.tz_localize("UTC")
    return d


def poll_walk(sym, tf, A, B, step, session="RTH"):
    """Lockstep the resident delta read vs the full-prep read across poll bounds; compare
    the new bars each would feed. Returns (mismatches[], n_bars_compared)."""
    full0 = _read(sym, tf, A, B, session)
    if full0 is None or len(full0) < 4:
        return None
    cursor = full0.index[0]
    mism = []
    nbars = 0
    bound = pd.Timestamp(cursor).to_pydatetime() + step
    while bound <= B + step:
        b = min(bound, B)
        cur_dt = pd.Timestamp(cursor).to_pydatetime()
        delta = _read(sym, tf, cur_dt, b, session)
        full = _read(sym, tf, A, b, session)
        dn = delta[delta.index > pd.Timestamp(cursor)][OHLCV] if delta is not None else None
        fn = full[full.index > pd.Timestamp(cursor)][OHLCV] if full is not None else None
        if fn is not None and len(fn):
            if dn is None or not fn.index.equals(dn.index) or not fn.equals(dn):
                # locate the divergence
                if dn is None or not fn.index.equals(dn.index):
                    mism.append((str(cursor), str(b), "INDEX",
                                 len(fn), 0 if dn is None else len(dn)))
                else:
                    diff = (fn != dn) & ~(fn.isna() & dn.isna())
                    r = diff.any(axis=1)
                    t0 = r[r].index[0]
                    mism.append((str(cursor), str(b), f"VALUE@{t0}",
                                 len(fn), len(dn)))
            nbars += len(fn)
            cursor = fn.index[-1]
        bound += step
    return (mism, nbars)


D15 = datetime(2026, 7, 15, tzinfo=timezone.utc)
D17 = datetime(2026, 7, 17, tzinfo=timezone.utc)
cases = [
    ("1Min", D15, D17, timedelta(minutes=20)),
    ("5Min", D15, D17, timedelta(minutes=40)),
    ("10Sec", datetime(2026, 7, 16, 14, 30, tzinfo=timezone.utc),
              datetime(2026, 7, 16, 15, 30, tzinfo=timezone.utc), timedelta(minutes=5)),
]
print("=== M-RS5a Phase-1 per-poll DELTA vs FULL-PREP new-bar byte-identity ===")
allok = True
for sym in ("SPY", "TSLA"):
    print(f"\n[{sym}]")
    for tf, A, B, step in cases:
        res = poll_walk(sym, tf, A, B, step)
        if res is None:
            print(f"  {sym} {tf:<6} SKIP (insufficient data)")
            continue
        mism, nbars = res
        if not mism:
            print(f"  {sym} {tf:<6} ✅ byte-identical new bars across all polls ({nbars} bars)")
        else:
            allok = False
            print(f"  {sym} {tf:<6} ❌ {len(mism)} poll(s) diverged ({nbars} bars):")
            for m in mism[:4]:
                print(f"      cursor={m[0]} bound={m[1]} {m[2]} full_n={m[3]} delta_n={m[4]}")

print("\n" + ("✅ DELTA-READ PARITY GREEN — resident delta feeds == full-prep feeds"
             if allok else "❌ DELTA-READ PARITY has real failures"))
