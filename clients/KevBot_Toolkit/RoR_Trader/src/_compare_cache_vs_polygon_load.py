"""M-RS2 Phase 2 — byte-identical + speed A/B of the PRIMARY bar load.

Compares load_from_polygon(native TF) vs bar_cache.read_bars(direct PG) for a
(symbol, timeframe, window). Proves the cache serves the SAME data the backtest's
primary load (services.prepare_data_with_indicators:314) would fetch from Polygon
— and how much faster. Since secondaries + engine derive deterministically from
the primary, a byte-identical primary ⇒ a byte-identical full backtest.

Usage:
  .venv/bin/python src/_compare_cache_vs_polygon_load.py --symbol TSLA --tf 30Sec --days 30
"""
from __future__ import annotations
import argparse, os, sys, time, warnings
warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="TSLA")
    ap.add_argument("--tf", default="30Sec")
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    os.environ["USE_DB"] = "true"
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
                override=True)
    os.environ["USE_DB"] = "true"
    from db import set_admin_user_context
    set_admin_user_context(os.environ.get(
        "DEV_USER_ID", "19d47e46-f718-49a6-af32-5f5407f5b170"))
    import bar_cache as bc
    import datetime as dt
    from data_loader import load_from_polygon

    lo = bc._min_cached_ts(args.symbol, args.tf)
    hi = bc._max_cached_ts(args.symbol, args.tf)
    if not lo:
        sys.exit(f"no cached rows for {args.symbol}/{args.tf} — backfill first")
    s = lo
    e = min(hi, lo + dt.timedelta(days=args.days))
    print(f"A/B {args.symbol}/{args.tf}  window {str(s)[:10]}..{str(e)[:10]}")

    t = time.time()
    poly = load_from_polygon(symbol=args.symbol, timeframe=args.tf,
                             start_date=s, end_date=e, session="24/7")
    tp = time.time() - t
    t = time.time()
    cache = bc.read_bars(args.symbol, args.tf, s, e)
    tc = time.time() - t
    np_ = 0 if poly is None else len(poly)
    nc = 0 if cache is None else len(cache)
    print(f"  Polygon native : {np_:>8,} rows  {tp:6.1f}s")
    print(f"  Cache read_bars: {nc:>8,} rows  {tc:6.2f}s  ({tp/tc:.1f}x faster)")
    if poly is not None and cache is not None:
        j = poly[["open", "high", "low", "close", "volume"]].join(
            cache, lsuffix="_p", rsuffix="_c", how="inner")
        diffs = sum(int((j[c + "_p"] - j[c + "_c"]).abs().gt(1e-9).sum())
                    for c in ("open", "high", "low", "close", "volume"))
        ok = diffs == 0 and np_ == nc
        print(f"  shared={len(j):,}  rowcount_match={np_==nc}  value_diffs={diffs}"
              f"  ->  {'BYTE-IDENTICAL' if ok else 'DIFF'}")


if __name__ == "__main__":
    main()
