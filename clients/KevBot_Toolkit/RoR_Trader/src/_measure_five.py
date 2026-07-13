"""Rolling intraday paired-% measure for the Gated Five (+ any sids passed).

Pairs live fill alerts vs backtest-lane trades at ±10s (greedy nearest-first,
entries and exits separately) over [--since, --until] (defaults: today's RTH
open → 20 minutes ago). Direct-PG, read-only.

Usage: measure_five.py [--sids 327,328,...] [--since ISO] [--until ISO]
NOTE: backtest lane must be fresh over the window; each sid's newest backtest
entry is printed so staleness is visible instead of silently deflating rates.
"""
import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv

load_dotenv(".env", override=True)
import psycopg

TOL = 10.0
FIVE = [327, 328, 329, 333, 340]


def pair(a_ts, b_ts, tol=TOL):
    a = sorted(a_ts)
    b = sorted(b_ts)
    pairs = 0
    for x in a:
        best, bd = None, None
        for y in b:
            d = abs((x - y).total_seconds())
            if bd is None or d < bd:
                best, bd = y, d
        if best is not None and bd <= tol:
            pairs += 1
            b.remove(best)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sids", default=",".join(map(str, FIVE)))
    ap.add_argument("--since", default=None)
    ap.add_argument("--until", default=None)
    args = ap.parse_args()
    sids = [int(s) for s in args.sids.split(",")]
    now = datetime.now(timezone.utc)
    since = (datetime.fromisoformat(args.since.replace("Z", "+00:00"))
             if args.since else now.replace(hour=13, minute=30, second=0,
                                            microsecond=0))
    until = (datetime.fromisoformat(args.until.replace("Z", "+00:00"))
             if args.until else now - timedelta(minutes=20))
    print(f"window: {since:%Y-%m-%d %H:%M}Z -> {until:%H:%M}Z  tol=±{TOL:.0f}s")

    dsn = os.environ["SUPABASE_CONNECTION_STRING"]
    with psycopg.connect(dsn, connect_timeout=15,
                         prepare_threshold=None) as conn:
        cur = conn.cursor()
        print(f"{'sid':>4} {'live e/x':>9} {'bt e/x':>9} {'pairedE':>8} "
              f"{'pairedX':>8} {'comb%':>6}  bt-fresh-to")
        for sid in sids:
            cur.execute("""
                select side,
                       case when side='entry' then data->>'entry_fill_ts'
                            else data->>'exit_fill_ts' end
                from alerts
                where strategy_id=%s and event_type='fill'
                and timestamp between %s and %s""", (sid, since, until))
            live_e, live_x = [], []
            dropped = 0
            for side, fts in cur.fetchall():
                try:
                    t = datetime.fromisoformat(str(fts).replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    dropped += 1
                    continue
                (live_e if (side or "").lower() == "entry"
                 else live_x).append(t)
            if dropped:
                print(f"  !! sid {sid}: {dropped} alerts unparsable fill_ts")
            cur.execute("""
                select entry_fill_ts, exit_fill_ts from trades
                where strategy_id=%s and data_source like 'backtest%%'
                and entry_fill_ts between %s and %s""", (sid, since, until))
            bt_e, bt_x = [], []
            for e, x in cur.fetchall():
                if e:
                    bt_e.append(e)
                if x and since <= x <= until:
                    bt_x.append(x)
            cur.execute("""select max(entry_fill_ts) from trades
                where strategy_id=%s and data_source like 'backtest%%'""",
                        (sid,))
            fresh = cur.fetchone()[0]
            pe = pair(live_e, bt_e)
            px = pair(live_x, bt_x)
            denom = max(len(live_e), len(bt_e)) + max(len(live_x), len(bt_x))
            comb = (100.0 * (pe + px) / denom) if denom else float("nan")
            print(f"{sid:>4} {len(live_e):>4}/{len(live_x):<4} "
                  f"{len(bt_e):>4}/{len(bt_x):<4} {pe:>8} {px:>8} "
                  f"{comb:>5.0f}%  {fresh}")


if __name__ == "__main__":
    main()
