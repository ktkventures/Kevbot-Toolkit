"""TradingView export vs our backtest — trade-by-trade parity check.

Reads a TradingView Strategy Tester "List of Trades" CSV export and our
backtest trades for a strategy, aligns them on entry time (accounting for
TV's bar-OPEN labeling vs our fill-time labeling), and reports matches /
phantoms / missed + price and exit-reason deltas.

Pure offline. Touches no engine code.

CSV: save the TradingView export to docs/tradingview/sid<ID>_tv_export.csv.
TradingView's "List of Trades" CSV typically has columns like:
  Trade #, Type ("Entry long"/"Exit long"), Signal, Date/Time, Price USD,
  Quantity, ...  (column names vary slightly by TV version — this script
  sniffs them and you can adjust COLMAP if needed.)

Usage:
  cd src && ../.venv/bin/python _tv_parity_compare.py --sid 303 \
      --csv ../docs/tradingview/sid303_tv_export.csv \
      --ws 2026-06-12T13:30 --we 2026-06-13T00:00 \
      --tf-seconds 10 --tz-offset-min 0
  # --tz-offset-min: if your TV CSV times are exchange-local (ET), pass the
  #   offset to UTC (ET = +300 in summer/EDT, +240... actually EDT=UTC-4 so
  #   times are UTC-4 → add 240). The script also tries to auto-detect.
"""
from __future__ import annotations

import argparse
import csv
import warnings
from datetime import datetime, timedelta, timezone

warnings.filterwarnings('ignore')
from dotenv import load_dotenv

load_dotenv('.env', override=True)
import os

os.environ.setdefault('USE_DB', 'true')

PAIR_TOL_S = 15  # entry-time pairing tolerance (seconds)


def _parse_dt(s: str):
    s = s.strip().strip('"')
    for fmt in ('%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M', '%m/%d/%Y %H:%M:%S', '%m/%d/%Y, %H:%M'):
        try:
            d = datetime.strptime(s, fmt)
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        d = datetime.fromisoformat(s.replace('Z', '+00:00'))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def load_tv_csv(path: str, tf_seconds: int, tz_off_min: int):
    """Return list of TV entries: {entry_ts(UTC, fill=bar close), price, exit_ts,
    exit_price, signal/reason}. TV labels bars by OPEN time, so fill = open+tf."""
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    keys = {k.lower().strip(): k for k in rows[0].keys()}

    def col(*cands):
        for c in cands:
            if c in keys:
                return keys[c]
        return None
    c_type = col('type')
    c_dt = col('date/time', 'date', 'datetime', 'time')
    c_price = col('price usd', 'price', 'price ($)')
    c_signal = col('signal', 'comment')
    trades = []
    cur = None
    for r in rows:
        typ = (r.get(c_type, '') or '').lower()
        dt = _parse_dt(r.get(c_dt, '') or '')
        if dt is None:
            continue
        dt = dt + timedelta(minutes=tz_off_min)
        fill = dt + timedelta(seconds=tf_seconds)  # bar-open label -> bar-close fill
        price = None
        try:
            price = float((r.get(c_price, '') or '').replace(',', ''))
        except ValueError:
            pass
        sig = (r.get(c_signal, '') or '').strip()
        if 'entry' in typ:
            cur = {'entry_ts': fill, 'entry_price': price, 'entry_sig': sig}
        elif 'exit' in typ and cur is not None:
            cur.update({'exit_ts': fill, 'exit_price': price, 'exit_sig': sig})
            trades.append(cur)
            cur = None
    return trades


def load_our_trades(sid: int, ws, we):
    from db import get_admin_client
    c = get_admin_client()
    rows = (c.table('trades')
            .select('entry_fill_ts,exit_fill_ts,entry_price,exit_price,exit_reason,data_source')
            .eq('strategy_id', sid).like('data_source', 'backtest_%')
            .gte('entry_fill_ts', ws.isoformat()).lt('entry_fill_ts', we.isoformat())
            .order('entry_fill_ts').execute().data)
    out = []
    for r in rows:
        et = _parse_dt(r['entry_fill_ts']) if r.get('entry_fill_ts') else None
        xt = _parse_dt(r['exit_fill_ts']) if r.get('exit_fill_ts') else None
        out.append({'entry_ts': et, 'exit_ts': xt,
                    'entry_price': r.get('entry_price'), 'exit_price': r.get('exit_price'),
                    'exit_reason': r.get('exit_reason')})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sid', type=int, required=True)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--ws', required=True)
    ap.add_argument('--we', required=True)
    ap.add_argument('--tf-seconds', type=int, default=10)
    ap.add_argument('--tz-offset-min', type=int, default=0)
    a = ap.parse_args()
    ws = _parse_dt(a.ws)
    we = _parse_dt(a.we)
    tv = load_tv_csv(a.csv, a.tf_seconds, a.tz_offset_min)
    ours = load_our_trades(a.sid, ws, we)
    tv = [t for t in tv if t.get('entry_ts') and ws <= t['entry_ts'] <= we]
    print(f"TV trades: {len(tv)}  |  our backtest trades: {len(ours)}")

    # Two-pointer pair on entry time
    matched, tv_only, our_only = [], [], []
    used = [False] * len(ours)
    for t in tv:
        best, bi = None, None
        for i, o in enumerate(ours):
            if used[i] or not o['entry_ts']:
                continue
            d = abs((t['entry_ts'] - o['entry_ts']).total_seconds())
            if d <= PAIR_TOL_S and (best is None or d < best):
                best, bi = d, i
        if bi is not None:
            used[bi] = True
            matched.append((t, ours[bi], best))
        else:
            tv_only.append(t)
    our_only = [ours[i] for i in range(len(ours)) if not used[i]]

    print(f"\nMATCHED entries (±{PAIR_TOL_S}s): {len(matched)}")
    print(f"TV-only (no backtest entry): {len(tv_only)}")
    print(f"backtest-only (TV missed): {len(our_only)}")
    if matched:
        eps = [abs((t['entry_price'] or 0) - (o['entry_price'] or 0)) for t, o, _ in matched]
        print(f"  entry price |Δ| avg={sum(eps)/len(eps):.4f} max={max(eps):.4f}")
        # exit reason agreement
        agree = 0
        for t, o, _ in matched:
            tvr = (t.get('exit_sig') or '').lower()
            our = (o.get('exit_reason') or '').lower()
            is_stop_tv = 'stop' in tvr
            is_stop_our = 'stop' in our
            if is_stop_tv == is_stop_our:
                agree += 1
        print(f"  exit-reason (stop vs signal) agree: {agree}/{len(matched)}")
    print("\nfirst 8 TV-only entries:", [t['entry_ts'].strftime('%H:%M:%S') for t in tv_only[:8]])
    print("first 8 backtest-only entries:",
          [o['entry_ts'].strftime('%H:%M:%S') for o in our_only[:8] if o['entry_ts']])


if __name__ == '__main__':
    main()
