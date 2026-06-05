"""Gap-detector CLI probe — find BT-lane gaps for a strategy and report.

Usage:
    cd src && ../.venv/bin/python _gap_detector_probe.py --sid 290
    cd src && ../.venv/bin/python _gap_detector_probe.py --sid 290 --hours 4
    cd src && ../.venv/bin/python _gap_detector_probe.py --cohort

The --cohort option runs detection across the 9-canary cohort and reports
per-strategy gap summaries. Useful for catching fleet-wide deploy-churn
gaps in one pass.

NOTE: This script currently only DETECTS gaps. It does not auto-backfill.
Verification approach (Kevin's, 2026-06-05): run this script → note the
detected gaps → click Update All Data on the strategy → confirm that UAD
fills the same windows the detector flagged. That validates the detection
logic before we wire automatic backfill.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone


def _load_env():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


def _print_gaps(sid: int, gaps: list, hours: float):
    if not gaps:
        print(f"  sid={sid}: ✓ no gaps detected ({hours}h lookback)")
        return
    total_alerts = sum(g.total_alerts for g in gaps)
    total_trades = sum(g.total_bt_trades for g in gaps)
    total_missing = total_alerts - total_trades
    print(f"  sid={sid}: ⚠ {len(gaps)} gap(s), "
          f"~{total_missing} alerts uncovered out of {total_alerts}")
    for i, g in enumerate(gaps, 1):
        print(f"    gap {i}: {g.start.isoformat()[11:19]} → "
              f"{g.end.isoformat()[11:19]} "
              f"({g.duration_minutes:.0f}min)  "
              f"alerts={g.total_alerts} bt_trades={g.total_bt_trades} "
              f"coverage={g.coverage_pct:.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", type=int, default=None,
                    help="single strategy_id to probe")
    ap.add_argument("--cohort", action="store_true",
                    help="probe the 9-canary cohort instead")
    ap.add_argument("--hours", type=float, default=4.0,
                    help="lookback hours (default 4)")
    ap.add_argument("--bucket", type=int, default=5,
                    help="bucket size minutes (default 5)")
    ap.add_argument("--threshold", type=float, default=0.33,
                    help="coverage threshold below which bucket is gap "
                         "(default 0.33 — flag if BT trades < 33%% of alerts)")
    ap.add_argument("--skip", type=int, default=10,
                    help="skip the most-recent N minutes (default 10) — "
                         "BT lane normally lags this much")
    args = ap.parse_args()

    if not args.sid and not args.cohort:
        print("Must specify --sid SID or --cohort")
        sys.exit(1)

    _load_env()
    from gap_detector import detect_gaps_for_strategy

    cohort = [284, 290, 296, 302, 303, 300, 299, 288, 293]
    sids = [args.sid] if args.sid else cohort

    now = datetime.now(timezone.utc)
    print(f"=== Gap detection — {now.isoformat()[:19]} ===")
    print(f"  Window: last {args.hours}h  |  bucket: {args.bucket}min  |  "
          f"threshold: <{int(100*args.threshold)}% coverage  |  "
          f"skip recent: {args.skip}min")
    print()

    sids_with_gaps = 0
    total_gap_count = 0
    for sid in sids:
        try:
            gaps = detect_gaps_for_strategy(
                sid,
                lookback_hours=args.hours,
                bucket_minutes=args.bucket,
                coverage_threshold=args.threshold,
                recent_skip_minutes=args.skip,
            )
        except Exception as e:
            print(f"  sid={sid}: ERROR {e}")
            continue
        if gaps:
            sids_with_gaps += 1
            total_gap_count += len(gaps)
        _print_gaps(sid, gaps, args.hours)

    if len(sids) > 1:
        print()
        print(f"Summary: {sids_with_gaps}/{len(sids)} strategies have gaps "
              f"({total_gap_count} total gap windows)")
        print()
        print("Verification next step:")
        print(f"  Click Update All Data on a flagged strategy → confirm BT "
              f"trades land in the detected gap windows.")


if __name__ == "__main__":
    main()
