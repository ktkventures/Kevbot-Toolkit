"""Probe bar_diagnostics: live vs backtest vs algo per-bar indicator state.

Usage:
    cd src && ../.venv/bin/python _probe_bar_diagnostics.py --sid 302
    cd src && ../.venv/bin/python _probe_bar_diagnostics.py --sid 302 --since 2026-06-05T14:30
    cd src && ../.venv/bin/python _probe_bar_diagnostics.py --sid 302 --column utv4_trailing_stop --tol 0.001

Prints a wide-format table aligned by bar_ts, with one column per
(source, indicator) pair. Highlights the first row where live and
backtest disagree by more than --tol (default 0.0001 = 1bp on a $750
price level).

This is the empirical confirmation/refutation tool for #53 (UT Bot
trailing-stop drift hypothesis). If live and backtest diverge after a
position close, the bar where they first split is the smoking gun.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone


def _load_env():
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", type=int, required=True,
                    help="strategy_id to probe")
    ap.add_argument("--since", type=str, default=None,
                    help="ISO timestamp; default: 2h ago")
    ap.add_argument("--until", type=str, default=None,
                    help="ISO timestamp; default: now")
    ap.add_argument("--column", type=str, default=None,
                    help="Indicator column to focus on (default: all diagnostic columns the pack declares)")
    ap.add_argument("--tol", type=float, default=0.0001,
                    help="Absolute tolerance for live-vs-backtest divergence flag")
    ap.add_argument("--source", action="append", choices=("live", "backtest", "algo"),
                    help="Filter to specific source(s); repeatable. Default: all")
    ap.add_argument("--limit", type=int, default=200,
                    help="Max bars to print (default 200)")
    args = ap.parse_args()

    _load_env()
    from db import get_admin_client, load_bar_diagnostics

    now = datetime.now(timezone.utc)
    until = _parse_iso(args.until) or now
    since = _parse_iso(args.since) or (until - timedelta(hours=2))

    # Pull all sources unless filtered.
    rows = load_bar_diagnostics(
        args.sid,
        start_ts=since.isoformat(),
        end_ts=until.isoformat(),
        source=None,
        limit=10000,
    )
    if not rows:
        print(f"No bar_diagnostics rows for sid={args.sid} in "
              f"[{since.isoformat()[:19]} → {until.isoformat()[:19]}].")
        print("Check: (a) migration applied, (b) Railway redeploy of api+worker, "
              "(c) UT Bot V4 pack has diagnostic_columns set.")
        return

    if args.source:
        wanted = set(args.source)
        rows = [r for r in rows if r.get("source") in wanted]
        if not rows:
            print(f"After --source filter ({args.source}): 0 rows.")
            return

    # Pivot: by bar_ts → {source → values dict}
    pivot: dict = defaultdict(dict)
    cols_seen: set = set()
    for r in rows:
        bts = r["bar_ts"]
        src = r["source"]
        vals = r.get("values") or {}
        pivot[bts][src] = vals
        cols_seen.update(vals.keys())

    cols_to_render = [args.column] if args.column else sorted(cols_seen)
    sources_order = ["live", "backtest", "algo"]
    sources_present = [s for s in sources_order
                       if any(s in v for v in pivot.values())]

    print(f"=== bar_diagnostics probe — sid={args.sid} ===")
    print(f"window: {since.isoformat()[:19]} → {until.isoformat()[:19]}")
    print(f"sources present: {sources_present}")
    print(f"columns probed: {cols_to_render}")
    print(f"total bars with at least one source: {len(pivot)}")
    print()

    # Per-source row counts
    counts = {s: 0 for s in sources_order}
    for v in pivot.values():
        for s in v:
            counts[s] = counts.get(s, 0) + 1
    print("per-source bar counts:")
    for s, n in counts.items():
        print(f"  {s:10s} {n}")
    print()

    # Table: bar_ts, then for each (source, column) pair show value.
    sorted_ts = sorted(pivot.keys())[-args.limit:]

    headers = ["bar_ts"]
    for s in sources_present:
        for c in cols_to_render:
            headers.append(f"{s[0]}_{c.replace('utv4_','')[:18]}")
    print("  " + "  ".join(f"{h:>22s}" for h in headers))
    print("  " + "-" * (22 * len(headers) + 2 * (len(headers) - 1)))

    first_divergence_ts: str | None = None
    for bts in sorted_ts:
        bar_vals = pivot[bts]
        cells = [bts[11:19]]
        for s in sources_present:
            vs = bar_vals.get(s, {})
            for c in cols_to_render:
                v = vs.get(c)
                cells.append("—" if v is None else f"{v:.4f}")

        # Divergence check: only for cols that have BOTH live + backtest
        marker = ""
        for c in cols_to_render:
            lv = bar_vals.get("live", {}).get(c)
            bv = bar_vals.get("backtest", {}).get(c)
            if lv is not None and bv is not None:
                if abs(lv - bv) > args.tol:
                    marker = f"  ⚠ Δ={abs(lv-bv):.4f} ({c})"
                    if first_divergence_ts is None:
                        first_divergence_ts = bts
                    break
        cells_str = "  ".join(f"{c:>22s}" for c in cells)
        print(f"  {cells_str}{marker}")

    print()
    if first_divergence_ts:
        print(f"FIRST DIVERGENCE: {first_divergence_ts} "
              f"(tolerance: {args.tol})")
    else:
        # Did we even have BOTH live + backtest for any bar?
        had_both = any("live" in v and "backtest" in v for v in pivot.values())
        if had_both:
            print(f"No divergences > {args.tol} found. Engines in agreement on "
                  f"the bars where both wrote.")
        else:
            print("No bars had BOTH 'live' AND 'backtest' rows — divergence "
                  "can't be checked yet. Wait for both engines to accumulate "
                  "rows on overlapping bars, then re-run.")


if __name__ == "__main__":
    main()
