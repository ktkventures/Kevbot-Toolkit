"""M1b shadow harness for the Resampled Bar Store — persist (WINDOW-REPLACE) +
comparator round-trip proof.

M1a proved build_window == canonical (in memory). M1b's NEW obligations:
  1. PERSISTENCE ROUND-TRIP: the store read back from Postgres (tz/float) still ==
     canonical, byte-for-byte. That's what a consumer would actually read.
  2. WINDOW-REPLACE idempotency: re-persisting the same window does NOT grow rows
     (delete-then-insert replaces, never appends).
  3. WINDOW-REPLACE retraction cleanup: a stale/phantom coarse row (we inject the
     literal 313 signature — a phantom 20:00-UTC 4H RTH bar) is DELETED on the next
     window-REPLACE. This is the property that makes 313 structurally impossible in
     the store: it cannot retain a bar the canonical resample never forms.

WRITES the resampled_bar_cache table (a new, isolated table referenced by no
consumer). Sets RORT_RESAMPLED_STORE_WRITE=1 explicitly (this IS the write-path
test). Does NOT touch bar_cache, live consumers, or the Worker.

USAGE:
    .venv/bin/python src/_resampled_store_shadow.py
    .venv/bin/python src/_resampled_store_shadow.py --symbols TSLA,SPY
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")
_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_SRC, ".env"))
except Exception:
    pass
os.environ.setdefault("USE_DB", "true")
os.environ["RORT_RESAMPLED_STORE_WRITE"] = "1"   # this harness exercises the write path

import psycopg  # noqa: E402
import bar_cache  # noqa: E402
import resampled_bar_store as rbs  # noqa: E402

# Representative window/subset. M1a already proved build_window == canonical across
# the FULL fleet (7 sym x 10 tf x 5 session x 2 windows). M1b's NEW obligation is the
# PERSISTENCE round-trip + window-REPLACE mechanics, which a representative subset
# proves — and stays light on the shared prod DB (a parallel bug-hunt runs on it).
# Window spans the weekend + July-3-observed-holiday close into Mon-after-holiday
# 07-06 (session boundary), 2 trading days (Mon+Tue).
START = datetime(2026, 7, 4, tzinfo=timezone.utc)
END = datetime(2026, 7, 7, 23, 59, tzinfo=timezone.utc)
SYMBOLS = ["TSLA", "SPY"]                          # representative (override via --symbols)
SESSIONS = ["RTH", "Extended Hours", "24/7"]        # no-filter + filtered + dominant


def _default_symbols() -> list[str]:
    return SYMBOLS


def _table_rows() -> int:
    with psycopg.connect(os.environ["SUPABASE_CONNECTION_STRING"], connect_timeout=15,
                         prepare_threshold=None, autocommit=True) as c:
        with c.cursor() as cur:
            cur.execute("select count(*) from resampled_bar_cache")
            return cur.fetchone()[0]


def _persist_and_check(symbols, verbose) -> dict:
    print("\n" + "=" * 78)
    print("PHASE 1 — persist (WINDOW-REPLACE) + read-back byte-identity vs canonical")
    print("=" * 78)
    totals = {"compared": 0, "match": 0, "fail": 0, "empty": 0, "rows": 0}
    fails = []
    for sym in symbols:
        one_min = bar_cache.read_bars(sym, "1Min", START, END)
        if one_min is None or len(one_min) == 0:
            print(f"  {sym}: no cached 1Min — SKIP")
            continue
        srow = 0
        for tf in rbs.COARSE_TFS:
            for sess in SESSIONS:
                w = rbs.store_window(sym, tf, sess, START, END, one_min_df=one_min)
                totals["rows"] += w.get("rows", 0)
                srow += w.get("rows", 0)
                res = rbs.shadow_check(sym, tf, sess, START, END,
                                       one_min_df=one_min, persist=False)
                totals["compared"] += 1
                if res.get("note") == "both empty":
                    totals["empty"] += 1
                elif res["match"]:
                    totals["match"] += 1
                else:
                    totals["fail"] += 1
                    fails.append(res)
        print(f"  {sym}: persisted {srow} coarse rows across {len(rbs.COARSE_TFS)}tf "
              f"x {len(SESSIONS)}session")
    print(f"\n  totals: compared={totals['compared']} match={totals['match']} "
          f"EMPTY={totals['empty']} FAIL={totals['fail']} rows_written={totals['rows']}")
    if fails and verbose:
        for f in fails[:20]:
            print(f"    FAIL {f['symbol']}/{f['tf']}/{f['session']}: {f.get('note','')} "
                  f"{f['cell_diffs'][:3]}")
    return {"totals": totals, "fails": fails}


def _idempotency(symbols) -> bool:
    print("\n" + "=" * 78)
    print("PHASE 2 — WINDOW-REPLACE idempotency (re-persist must NOT grow rows)")
    print("=" * 78)
    before = _table_rows()
    sym = symbols[0]
    one_min = bar_cache.read_bars(sym, "1Min", START, END)
    for tf in rbs.COARSE_TFS:
        for sess in SESSIONS:
            rbs.store_window(sym, tf, sess, START, END, one_min_df=one_min)
    after = _table_rows()
    ok = (before == after)
    print(f"  table rows before re-persist={before}  after={after}  "
          f"-> {'STABLE ✓' if ok else 'GREW ✗ (window-REPLACE is appending!)'}")
    return ok


def _retraction(symbols) -> bool:
    print("\n" + "=" * 78)
    print("PHASE 3 — retraction/phantom cleanup (inject the 313 signature, replace, "
          "confirm gone)")
    print("=" * 78)
    sym = symbols[0]
    tf, sess = "4Hour", "RTH"
    tf_s = rbs.tf_seconds(tf)
    # The 313 phantom: a 20:00-UTC 4H RTH bar (an after-hours bucket RTH resample
    # never forms). Inject it directly, then window-REPLACE that day.
    phantom_ts = datetime(2026, 7, 6, 20, 0, tzinfo=timezone.utc)
    dsn = os.environ["SUPABASE_CONNECTION_STRING"]
    with psycopg.connect(dsn, connect_timeout=15, prepare_threshold=None,
                         autocommit=True) as c:
        with c.cursor() as cur:
            cur.execute(
                "INSERT INTO resampled_bar_cache (symbol,timeframe_seconds,session,ts,"
                "open,high,low,close,volume,source_1min_span_hash,settled,revised_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (symbol,timeframe_seconds,session,ts) DO UPDATE SET close=EXCLUDED.close",
                (sym, tf_s, sess, phantom_ts.isoformat(),
                 999.0, 999.0, 999.0, 999.0, 1.0, "PHANTOM_313", True,
                 datetime.now(timezone.utc).isoformat()))
            cur.execute("select count(*) from resampled_bar_cache where symbol=%s and "
                        "timeframe_seconds=%s and session=%s and ts=%s",
                        (sym, tf_s, sess, phantom_ts.isoformat()))
            injected = cur.fetchone()[0]
    print(f"  injected phantom {sym} {tf} {sess} @ {phantom_ts} -> present={injected==1}")
    # window-REPLACE the phantom's day
    one_min = bar_cache.read_bars(sym, "1Min", START, END)
    rbs.store_window(sym, tf, sess, START, END, one_min_df=one_min)
    with psycopg.connect(dsn, connect_timeout=15, prepare_threshold=None,
                         autocommit=True) as c:
        with c.cursor() as cur:
            cur.execute("select count(*) from resampled_bar_cache where symbol=%s and "
                        "timeframe_seconds=%s and session=%s and ts=%s",
                        (sym, tf_s, sess, phantom_ts.isoformat()))
            still = cur.fetchone()[0]
    ok = (still == 0)
    print(f"  after WINDOW-REPLACE: phantom present={still==1} -> "
          f"{'DELETED ✓ (313 cannot persist)' if ok else 'STILL THERE ✗'}")
    # confirm the day's real bars are intact + match canonical
    res = rbs.shadow_check(sym, tf, sess, START, END, one_min_df=one_min, persist=False)
    print(f"  real {sym} {tf} {sess} bars still byte-identical to canonical -> "
          f"{res['match']} (n={res['n_built']})")
    return ok and res["match"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if not bar_cache.direct_pg_available():
        print("FATAL: no SUPABASE_CONNECTION_STRING")
        return 2
    symbols = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
               or _default_symbols())
    print(f"M1b shadow harness — WRITE flag ON; symbols={symbols}")
    print(f"window: {START.date()}..{END.date()} (incl Mon-after-holiday 07-06)")

    r1 = _persist_and_check(symbols, args.verbose)
    ok2 = _idempotency(symbols)
    ok3 = _retraction(symbols)

    t = r1["totals"]
    roundtrip_ok = (t["fail"] == 0)
    print("\n" + "#" * 78)
    print(f"PERSISTENCE ROUND-TRIP: {t['match']} match / {t['fail']} fail "
          f"({'✓' if roundtrip_ok else '✗'})")
    print(f"WINDOW-REPLACE idempotency: {'✓' if ok2 else '✗'}")
    print(f"RETRACTION/phantom cleanup: {'✓' if ok3 else '✗'}")
    total_rows = _table_rows()
    print(f"resampled_bar_cache now holds {total_rows} coarse rows (shadow-populated)")
    all_ok = roundtrip_ok and ok2 and ok3
    print("✅ M1b SHADOW GREEN — store round-trips byte-identical; WINDOW-REPLACE honest."
          if all_ok else "❌ M1b shadow has failures — see above.")
    print("#" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
