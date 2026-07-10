"""Offline byte-identity harness for the Resampled Bar Store (M-RS2 Phase 2, M1a).

THE PROMOTION GATE #1 (the spec's §7): store-built coarse bars must be
BYTE-IDENTICAL to `resample_to_timeframe(_filter_session(1Min, session), tf)` across
ALL tf × session × active symbols, over a multi-day + Monday-after-holiday +
session-boundary window. Without this the store ADDS divergence instead of removing
it.

What it proves (two layers, both byte-exact):
  1. The builder's WINDOW-REPLACE unit is honest: per-UTC-day resample
     (build_window) == whole-window resample (canonical_resampled). If a resample
     bin ever straddled a UTC day this would fail — it doesn't for any store TF,
     and this asserts it empirically instead of trusting the argument.
  2. OHLCV exact per bar, and the bar SET (index) matches — so no phantom/missing
     coarse bar (the empty-bucket / dropna case), per session.

READ-ONLY: reads cached 1Min via bar_cache.read_bars (a plain SELECT — no Polygon,
no upsert). Writes NOTHING. Safe to run against prod.

USAGE:
    .venv/bin/python src/_resampled_store_byteident.py
    .venv/bin/python src/_resampled_store_byteident.py --symbols TSLA,SPY
    .venv/bin/python src/_resampled_store_byteident.py --verbose   # dump every diff

Exit code non-zero on ANY byte mismatch (gate-able).
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

# ── env bootstrap (self-contained: load src/.env + USE_DB) ───────────────────
_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_SRC, ".env"))
except Exception:
    pass
os.environ.setdefault("USE_DB", "true")

import bar_cache  # noqa: E402
import resampled_bar_store as rbs  # noqa: E402


# ── windows: multi-day + Monday-after-holiday + session boundaries ───────────
# PRIMARY (EDT/summer): 2026-06-29..07-08 spans two full weeks incl the
#   July-3-observed-holiday close and Mon 2026-07-06 (Monday-after-holiday — the
#   exact session-boundary case the divergence log flags for 313).
# WINTER (EST) is added opportunistically if 1Min coverage reaches it: in EST the
#   Extended/After-Hours sessions cross UTC midnight, so it exercises the
#   day-boundary edge that July (EDT) can't. MLK Mon 2026-01-20 = its
#   Monday-after-holiday. Skipped (noted) if the cache doesn't go back that far.
_PRIMARY = (datetime(2026, 6, 29, tzinfo=timezone.utc),
            datetime(2026, 7, 8, 23, 59, tzinfo=timezone.utc),
            "PRIMARY (EDT) 06-29..07-08  [Mon-after-holiday 07-06]")
_WINTER = (datetime(2026, 1, 16, tzinfo=timezone.utc),
           datetime(2026, 1, 23, 23, 59, tzinfo=timezone.utc),
           "WINTER (EST) 01-16..01-23  [Mon-after-holiday 01-20, sessions cross UTC midnight]")

_MON_HOLIDAY_HILITE = datetime(2026, 7, 6, tzinfo=timezone.utc).date()

_CELL_OK = "  ✓ "
_CELL_EMPTY = "  · "  # both sides empty (no bars this session) — a match, but flagged for coverage


def _default_symbols() -> list[str]:
    """1Min capture-target symbols (the active set with cached 1Min)."""
    syms = sorted({t.get("symbol") for t in bar_cache.get_capture_targets(enabled_only=False)
                   if t.get("timeframe") == "1Min" and t.get("symbol")})
    return syms or ["TSLA", "SPY"]


def _run_window(symbols: list[str], start, end, label: str, verbose: bool) -> dict:
    print("\n" + "=" * 78)
    print(f"WINDOW: {label}")
    print("=" * 78)
    sessions = rbs.STORE_SESSIONS
    tfs = rbs.COARSE_TFS

    totals = {"compared": 0, "match": 0, "fail": 0, "empty": 0}
    fails: list[dict] = []
    covered_syms = 0

    for sym in symbols:
        one_min = bar_cache.read_bars(sym, "1Min", start, end)
        if one_min is None or len(one_min) == 0:
            print(f"\n  {sym}: no cached 1Min in window — SKIP (coverage gap)")
            continue
        covered_syms += 1
        span = f"{one_min.index[0]}..{one_min.index[-1]}  ({len(one_min)} 1Min bars)"
        print(f"\n  {sym}:  {span}")
        # header
        print("    " + "tf".ljust(8) + "".join(s[:14].ljust(15) for s in sessions))
        for tf in tfs:
            row = "    " + tf.ljust(8)
            for sess in sessions:
                res = rbs.build_and_compare(sym, tf, sess, start, end, one_min_df=one_min)
                totals["compared"] += 1
                if res.get("note") == "both empty":
                    totals["empty"] += 1
                    row += _CELL_EMPTY.ljust(15)
                    continue
                if res["match"]:
                    totals["match"] += 1
                    row += (_CELL_OK + f"({res['n_built']})").ljust(15)
                else:
                    totals["fail"] += 1
                    fails.append(res)
                    ncell = (len(res["cell_diffs"]) or "idx")
                    row += (f"  ✗ {ncell}").ljust(15)
            print(row)

    # Monday-after-holiday spotlight: show the 4Hour RTH bars (the 313-class bar)
    if start.date() <= _MON_HOLIDAY_HILITE <= end.date() and symbols:
        _spotlight_313(symbols[0], start, end)

    print("\n  " + "-" * 60)
    print(f"  window totals: compared={totals['compared']}  "
          f"match={totals['match']}  EMPTY(both)={totals['empty']}  "
          f"FAIL={totals['fail']}  (symbols covered={covered_syms}/{len(symbols)})")
    if fails and verbose:
        print("\n  FAIL DETAIL:")
        for f in fails:
            print(f"    {f['symbol']}/{f['tf']}/{f['session']}: "
                  f"n_built={f['n_built']} n_canon={f['n_canonical']} "
                  f"note={f.get('note','')}")
            if f["index_mismatch"]:
                print(f"      index_mismatch={f['index_mismatch']}")
            for d in f["cell_diffs"][:8]:
                print(f"      {d['ts']} {d['col']}: built={d['built']} canon={d['canonical']}")
    return {"totals": totals, "fails": fails, "covered": covered_syms}


def _spotlight_313(sym: str, start, end):
    """Print the RTH 4Hour bars around the Monday-after-holiday for eyeball —
    this is the exact bar class 313 got wrong (a phantom AH 4H bucket carried
    across the session boundary). Store must produce clean RTH-only 4H bars."""
    try:
        one_min = bar_cache.read_bars(sym, "1Min", start, end)
        built = rbs.build_window(sym, "4Hour", "RTH", start, end, one_min_df=one_min)
        print(f"\n  ── 313 spotlight: {sym} RTH 4Hour bars (should be RTH-only, "
              f"no AH 20:00-UTC phantom) ──")
        if built is None or len(built) == 0:
            print("     (no RTH 4H bars)")
            return
        for ts, r in built.iterrows():
            print(f"     {ts}  O={r['open']:.2f} H={r['high']:.2f} "
                  f"L={r['low']:.2f} C={r['close']:.2f} V={r['volume']:.0f}")
    except Exception as e:  # noqa: BLE001
        print(f"     spotlight failed: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="", help="comma-list; default = 1Min capture targets")
    ap.add_argument("--verbose", action="store_true", help="dump every diff")
    ap.add_argument("--no-winter", action="store_true", help="skip the winter (EST) window")
    args = ap.parse_args()

    if not bar_cache.direct_pg_available():
        print("FATAL: SUPABASE_CONNECTION_STRING not set — need direct-PG to read 1Min.")
        return 2

    symbols = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
               or _default_symbols())
    print(f"Resampled Bar Store — offline byte-identity harness")
    print(f"symbols: {symbols}")
    print(f"TFs:     {rbs.COARSE_TFS}")
    print(f"sessions:{rbs.STORE_SESSIONS}")

    windows = [_PRIMARY]
    # Opportunistic winter window: only if 1Min coverage reaches it for symbol[0].
    if not args.no_winter:
        earliest = bar_cache._min_cached_ts(symbols[0], "1Min")
        if earliest is not None and earliest <= _WINTER[0]:
            windows.append(_WINTER)
        else:
            print(f"\nNOTE: winter (EST) window SKIPPED — {symbols[0]} 1Min cache "
                  f"starts {earliest} (> {_WINTER[0].date()}). The DST/midnight-cross "
                  f"edge is unverified for this run (coverage gap, not a failure).")

    grand = {"compared": 0, "match": 0, "fail": 0, "empty": 0}
    all_fails = []
    for (s, e, lbl) in windows:
        r = _run_window(symbols, s, e, lbl, args.verbose)
        for k in grand:
            grand[k] += r["totals"][k]
        all_fails += r["fails"]

    print("\n" + "#" * 78)
    print(f"GRAND TOTAL: compared={grand['compared']}  match={grand['match']}  "
          f"EMPTY(both)={grand['empty']}  FAIL={grand['fail']}")
    if grand["fail"] == 0:
        print("✅ BYTE-IDENTICAL: store-built == canonical resample across the full "
              "matrix (window-REPLACE per-day == whole-window; OHLCV + bar-set exact).")
        print("#" * 78)
        return 0
    print(f"❌ {grand['fail']} MISMATCH(es) — the store is NOT yet byte-identical. "
          f"Run with --verbose for per-cell diffs.")
    print("#" * 78)
    return 1


if __name__ == "__main__":
    sys.exit(main())
