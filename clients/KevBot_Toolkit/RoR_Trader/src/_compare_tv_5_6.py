"""Compare TV 1Min + 10Sec CSV against our cache for 2026-05-06.
Skip the last bar in each CSV (forming) per Kevin's instruction.
"""
from dotenv import load_dotenv
load_dotenv(override=True)

import pandas as pd
from datetime import datetime, timezone, timedelta
from db import get_admin_client


def fetch_cache(symbol, tf_seconds, since_iso, until_iso):
    c = get_admin_client()
    rows = []
    offset = 0
    while True:
        r = c.table('live_bars').select(
            'symbol,bar_start,open,high,low,close,volume,source'
        ).eq('symbol', symbol).eq('timeframe_seconds', tf_seconds) \
         .gte('bar_start', since_iso).lte('bar_start', until_iso) \
         .order('bar_start').range(offset, offset + 999).execute()
        page = r.data or []
        rows.extend(page)
        if len(page) < 1000:
            break
        offset += 1000
    return rows


def compare(label, csv_path, tf_seconds):
    print(f"\n{'='*90}\n{label}: {csv_path.split('/')[-1]}\n{'='*90}")
    tv = pd.read_csv(csv_path)
    tv = tv.iloc[:-1].copy()  # skip forming
    tv['ts_utc'] = tv.time.apply(lambda t: datetime.fromtimestamp(t, tz=timezone.utc))
    since = tv.ts_utc.iloc[0].isoformat()
    until = tv.ts_utc.iloc[-1].isoformat()
    print(f"TV bars: {len(tv)}  range: {since} -> {until}")

    cache_rows = fetch_cache('SPY', tf_seconds, since, until)
    print(f"Cache rows: {len(cache_rows)}")
    # Index cache by unix-second of bar_start
    by_ts = {}
    for r in cache_rows:
        ts = int(datetime.fromisoformat(r['bar_start'].replace('Z','+00:00')).timestamp())
        by_ts[ts] = r

    matched = []
    unmatched_tv = []
    for _, row in tv.iterrows():
        ts = int(row.time)
        cb = by_ts.get(ts)
        if cb is None:
            unmatched_tv.append(int(row.time))
            continue
        matched.append({
            'ts': ts,
            'tv_o': row.open, 'tv_h': row.high, 'tv_l': row.low, 'tv_c': row.close, 'tv_v': row.Volume,
            'c_o':  float(cb['open']), 'c_h': float(cb['high']), 'c_l': float(cb['low']),
            'c_c':  float(cb['close']), 'c_v': float(cb['volume']), 'src': cb['source'],
        })

    if not matched:
        print("No matches.")
        return

    m = pd.DataFrame(matched)
    print(f"Matched: {len(m)} / {len(tv)} (unmatched TV bars: {len(unmatched_tv)})")
    if unmatched_tv[:5]:
        for ts in unmatched_tv[:5]:
            print(f"  unmatched TV bar at {datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()}")
        if len(unmatched_tv) > 5:
            print(f"  ... +{len(unmatched_tv)-5} more")

    for fld in ('o','h','l','c'):
        d = (m[f'c_{fld}'] - m[f'tv_{fld}']).abs()
        identical = int((d == 0).sum())
        print(f"  {fld}: |Δ| p50={d.median():.4f}  p95={d.quantile(0.95):.4f}  max={d.max():.4f}  "
              f"identical={identical}/{len(m)} ({100*identical/len(m):.1f}%)")
    dv = (m.c_v - m.tv_v)
    print(f"  v: cache-tv signed median={dv.median():.0f}  mean={dv.mean():.1f}  "
          f"|d|.max={dv.abs().max():.0f}")

    # Source breakdown
    print(f"  cache source mix: {m.src.value_counts().to_dict()}")

    # Top divergences on close
    m['close_diff'] = (m.c_c - m.tv_c).abs()
    worst = m.nlargest(5, 'close_diff')
    if worst.iloc[0].close_diff > 0:
        print(f"  worst close diffs:")
        for _, w in worst.iterrows():
            ts = datetime.fromtimestamp(w.ts, tz=timezone.utc).strftime('%H:%M:%S')
            print(f"    {ts}  tv_c={w.tv_c}  c_c={w.c_c}  Δ={w.close_diff:.4f}  src={w.src}")


if __name__ == '__main__':
    compare('1Min', '/home/kevin/projects/Kevbot-Toolkit/clients/KevBot_Toolkit/RoR_Trader/docs/tv_data/5-6 test/AMEX_SPY, 1.csv', 60)
    compare('10Sec', '/home/kevin/projects/Kevbot-Toolkit/clients/KevBot_Toolkit/RoR_Trader/docs/tv_data/5-6 test/AMEX_SPY, 10S (34).csv', 10)
