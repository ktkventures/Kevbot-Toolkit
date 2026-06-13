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
    """Return list of TV trades: {entry_ts(UTC, fill=bar close), entry_price,
    exit_ts, exit_price, exit_sig}. TV labels bars by OPEN time → fill=open+tf.

    Handles the TradingView "List of Trades" export: rows grouped by trade
    number, with the Exit row appearing BEFORE the Entry row within a trade.
    """
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    keys = {k.lower().strip(): k for k in rows[0].keys()}

    def col(*cands):
        for c in cands:
            if c in keys:
                return keys[c]
        return None
    c_num = col('trade number', 'trade #', 'trade')
    c_type = col('type')
    c_dt = col('date and time', 'date/time', 'datetime', 'date', 'time')
    c_price = col('price usd', 'price', 'price ($)')
    c_signal = col('signal', 'comment')

    def price_of(r):
        try:
            return float((r.get(c_price, '') or '').replace(',', ''))
        except ValueError:
            return None

    def fill_of(r):
        dt = _parse_dt(r.get(c_dt, '') or '')
        if dt is None:
            return None
        return dt + timedelta(minutes=tz_off_min) + timedelta(seconds=tf_seconds)

    # Group rows by trade number; classify by Type.
    by_num: dict = {}
    for r in rows:
        num = (r.get(c_num, '') or '').strip()
        by_num.setdefault(num, {'entry': None, 'exit': None})
        typ = (r.get(c_type, '') or '').lower()
        if 'entry' in typ:
            by_num[num]['entry'] = r
        elif 'exit' in typ:
            by_num[num]['exit'] = r
    trades = []
    for num, pair in by_num.items():
        en, ex = pair['entry'], pair['exit']
        if en is None:
            continue
        trades.append({
            'entry_ts': fill_of(en), 'entry_price': price_of(en),
            'entry_sig': (en.get(c_signal, '') or '').strip(),
            'exit_ts': fill_of(ex) if ex is not None else None,
            'exit_price': price_of(ex) if ex is not None else None,
            'exit_sig': (ex.get(c_signal, '') or '').strip() if ex is not None else '',
        })
    trades = [t for t in trades if t['entry_ts'] is not None]
    trades.sort(key=lambda t: t['entry_ts'])
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
