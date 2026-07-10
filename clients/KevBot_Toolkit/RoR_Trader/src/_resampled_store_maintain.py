"""Resampled Bar Store — maintenance / backfill / shadow / coverage CLI (M-RS2 P2 §5/§7/§8).

The operator entrypoint for the store's WRITE loop. Hardened per the prime1s DOS
incident: chunked writes + circuit breaker + conservative cadence. All writes gated
RORT_RESAMPLED_STORE_WRITE (the store is truncatable/reversible). NOT wired to cron/
worker — run manually; the heavy historical SEED runs OFF-PEAK (after the 20:00Z close)
so it never contends with the trading path.

SUBCOMMANDS:
  shadow    [--days 5]                 READ-ONLY §7 comparator over the bounded targets
                                       (store vs on-the-fly canonical). Must be GREEN
                                       for N trading days before flipping a consumer
                                       READ flag on in prod.
  coverage  [--sample-days 5]          READ-ONLY §8 admin view: coverage/freshness/rows
                                       + live comparator diff-rate per (sym,tf,session).
  maintain  [--recent-days 3] --yes    Incremental WINDOW-REPLACE of the recent window
                                       for every target (settle/revise catch-up). WRITES.
  seed      [--days 400] [--chunk-days 30] [--no-compare] --yes
                                       Day-chunked historical SEED (WINDOW-REPLACE +
                                       per-chunk comparator). WRITES. RUN AFTER 20:00Z.

Bounded to resampled_bar_store.default_coarse_targets() (worked-on TSLA coarse TFs) —
override with --symbols/--tfs/--sessions. --yes is REQUIRED to actually write (else a
dry-run plan is printed).

  .venv/bin/python src/_resampled_store_maintain.py shadow
  .venv/bin/python src/_resampled_store_maintain.py seed --days 400 --yes   # AFTER CLOSE
"""
from __future__ import annotations
import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")
_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_SRC, ".env"))
except Exception:
    pass
os.environ.setdefault("USE_DB", "true")
os.environ["BAR_CACHE_ENABLED"] = "1"

import resampled_bar_store as rbs  # noqa: E402


def _targets(args) -> list:
    if args.symbols or args.tfs or args.sessions:
        syms = [s.strip().upper() for s in (args.symbols or "TSLA").split(",") if s.strip()]
        tfs = [t.strip() for t in (args.tfs or "4Hour,1Day").split(",") if t.strip()]
        sess = [s.strip() for s in (args.sessions or "RTH,Extended Hours").split(",") if s.strip()]
        return [(s, tf, se) for s in syms for tf in tfs for se in sess]
    return rbs.default_coarse_targets()


def _warm_1min(symbol, start, end, session):
    """Warm the 1Min cache for the window so the store is built from the same 1Min
    the consumers' canonical read sees (read-only-ish: bar_cache delta-fetch)."""
    try:
        from data_loader import load_market_data
        load_market_data(symbol, start_date=start, end_date=end, timeframe="1Min",
                         session=session)
    except Exception as e:  # noqa: BLE001
        print(f"  warn: 1Min warm failed {symbol} [{start.date()}..{end.date()}]: {e}")


def cmd_shadow(args):
    print(f"§7 SHADOW COMPARATOR (read-only) — last {args.days}d")
    res = rbs.shadow_compare_targets(_targets(args), days=args.days)
    allgreen = True
    for r in res:
        note = r.get("note") or (f"{len(r['cell_diffs'])} cell diffs" if r.get("cell_diffs") else "")
        mark = "✓" if r["match"] else "✗"
        if not r["match"]:
            allgreen = False
        print(f"  {mark} {r['symbol']}/{r['tf']}/{r['session']}: match={r['match']} "
              f"n_store={r['n_built']} n_canon={r['n_canonical']} {note}")
    print("✅ SHADOW GREEN" if allgreen else "❌ SHADOW has diffs — do NOT flip consumer READ on")
    return 0 if allgreen else 1


def cmd_coverage(args):
    print(f"§8 COVERAGE + comparator diff-rate (sample {args.sample_days}d)")
    rows = rbs.resampled_supply_coverage(_targets(args), sample_days=args.sample_days)
    print(f"  {'symbol':6} {'tf':6} {'session':16} {'rows':>6} {'min_ts':25} {'max_ts':25} {'match':6} {'diff_rate'}")
    for r in rows:
        print(f"  {r['symbol']:6} {r['tf']:6} {r['session']:16} {r['rows']:>6} "
              f"{str(r['min_ts'])[:25]:25} {str(r['max_ts'])[:25]:25} "
              f"{str(r['comparator_match']):6} {r['comparator_diff_rate']}")
    return 0


def _write_targets(args, is_seed):
    targets = _targets(args)
    if not args.yes:
        print("DRY-RUN (no --yes) — would WRITE these targets:")
        for t in targets:
            print("   ", t)
        print(f"(rerun with --yes to write; sets RORT_RESAMPLED_STORE_WRITE=1)")
        return None
    os.environ["RORT_RESAMPLED_STORE_WRITE"] = "1"
    return targets


def cmd_maintain(args):
    targets = _write_targets(args, is_seed=False)
    if targets is None:
        return 0
    print(f"MAINTAIN (WRITE) — recent {args.recent_days}d, {len(targets)} targets")
    for (s, tf, sess) in targets:
        now = datetime.now(timezone.utc)
        _warm_1min(s, now - timedelta(days=args.recent_days + 1), now, sess)
    r = rbs.maintain_all_coarse(targets, recent_days=args.recent_days)
    print(f"  result: {r}")
    return 1 if r.get("aborted") else 0


def cmd_seed(args):
    now_utc = datetime.now(timezone.utc)
    hh = now_utc.hour + now_utc.minute / 60.0
    if args.yes and 13.5 <= hh < 20.0 and not args.force_peak:
        print(f"REFUSING to seed during RTH ({now_utc:%H:%M}Z, close=20:00Z) — the "
              f"heavy seed must run OFF-PEAK so it can't contend with the trading path "
              f"/ today's clean divergence read. Re-run after 20:00Z, or pass "
              f"--force-peak to override (not recommended today).")
        return 2
    targets = _write_targets(args, is_seed=True)
    if targets is None:
        return 0
    print(f"SEED (WRITE, chunked+breaker) — {args.days}d back, chunk={args.chunk_days}d, "
          f"compare={not args.no_compare}, {len(targets)} targets")
    end = now_utc
    total = {"chunks": 0, "rows": 0, "diff_chunks": 0, "aborted": 0}
    for (s, tf, sess) in targets:
        start = end - timedelta(days=args.days)
        # warm the 1Min for the whole span first (chunked delta-fetch by bar_cache)
        _warm_1min(s, start, end, sess)
        r = rbs.backfill_coarse(s, tf, sess, start, end, chunk_days=args.chunk_days,
                                compare=not args.no_compare)
        print(f"  {s}/{tf}/{sess}: {r}")
        for k in ("chunks", "rows", "diff_chunks"):
            total[k] += r.get(k, 0)
        total["aborted"] += 1 if r.get("aborted") else 0
    print(f"\nSEED TOTAL: {total}")
    if total["aborted"]:
        print("❌ one or more targets tripped the breaker — investigate before retry")
        return 1
    if total["diff_chunks"]:
        print(f"⚠️  {total['diff_chunks']} chunk(s) had store!=canonical diffs — investigate "
              f"(do NOT flip consumer READ until shadow is green)")
        return 1
    print("✅ SEED complete, no breaker trips, comparator clean per chunk")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("shadow", "coverage", "maintain", "seed"):
        p = sub.add_parser(name)
        p.add_argument("--symbols", default="")
        p.add_argument("--tfs", default="")
        p.add_argument("--sessions", default="")
        if name == "shadow":
            p.add_argument("--days", type=int, default=5)
        if name == "coverage":
            p.add_argument("--sample-days", type=int, default=5)
        if name in ("maintain", "seed"):
            p.add_argument("--yes", action="store_true", help="actually WRITE (else dry-run)")
        if name == "maintain":
            p.add_argument("--recent-days", type=int, default=3)
        if name == "seed":
            p.add_argument("--days", type=int, default=400)
            p.add_argument("--chunk-days", type=int, default=30)
            p.add_argument("--no-compare", action="store_true")
            p.add_argument("--force-peak", action="store_true",
                           help="override the RTH refusal (not recommended)")
    args = ap.parse_args()
    if not rbs._dsn():
        print("FATAL: no SUPABASE_CONNECTION_STRING")
        return 2
    return {"shadow": cmd_shadow, "coverage": cmd_coverage,
            "maintain": cmd_maintain, "seed": cmd_seed}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
