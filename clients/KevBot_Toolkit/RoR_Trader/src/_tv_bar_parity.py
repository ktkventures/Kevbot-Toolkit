"""TradingView chart-export vs our engine — bar-by-bar parity.

Two independent checks:
  (A) MATH parity: run our ema_pp_v3 incremental class on TV's OWN close
      series, compare to TV's exported EMA short/mid columns. Isolates
      "does our EMA math == Pine's" independent of data differences.
  (B) DATA parity: our cache view (first_*, what live saw) and our REST
      view (Polygon 1s rolled) vs TV's bars — bar existence + close/EMA
      drift on shared bars + entry-signal alignment.

Offline; no engine code touched.
Usage: cd src && ../.venv/bin/python _tv_bar_parity.py
"""
from __future__ import annotations
import csv
import sys
import warnings
from datetime import datetime, timezone

warnings.filterwarnings('ignore')
from dotenv import load_dotenv
load_dotenv('.env', override=True)
import os
os.environ.setdefault('USE_DB', 'true')

CSV = '../docs/reference_images/AMEX_SPY, 10S (35).csv'
sys.path.insert(0, '../user_packs/ema_pp_v3')
from indicator_incremental import EmaPpV3Incremental  # noqa: E402


def load_tv():
    rows = list(csv.DictReader(open(CSV)))
    out = {}
    for r in rows:
        ep = int(r['time'])
        out[ep] = {
            'open': float(r['open']), 'high': float(r['high']),
            'low': float(r['low']), 'close': float(r['close']),
            'ema_s': float(r['EMA short']), 'ema_m': float(r['EMA mid']),
            'entry': float(r['entry']) != 0, 'exit': float(r['exit']) != 0,
        }
    return out


def run_ema(bars_by_ep):
    """Run our eppv3 over an ordered {epoch: bar} dict -> {epoch: out}."""
    eng = EmaPpV3Incremental()
    res = {}
    for ep in sorted(bars_by_ep):
        b = bars_by_ep[ep]
        res[ep] = eng.update_bar(b)
    return res


def stats(deltas):
    if not deltas:
        return 'n=0'
    return f'n={len(deltas)} avg={sum(deltas)/len(deltas):.5f} max={max(deltas):.5f}'


def main():
    tv = load_tv()
    eps = sorted(tv)
    t0 = datetime.fromtimestamp(eps[0], timezone.utc)
    t1 = datetime.fromtimestamp(eps[-1], timezone.utc)
    print(f"TV bars: {len(tv)}  window {t0.isoformat()} -> {t1.isoformat()}")

    # ---- (A) MATH parity: our EMA on TV's own closes vs TV's EMA cols ----
    ours_on_tv = run_ema({ep: tv[ep] for ep in eps})
    ds, dm = [], []
    # skip first ~50 bars (warmup convergence) for a fair steady-state read
    for ep in eps[50:]:
        ds.append(abs(ours_on_tv[ep]['eppv3_short'] - tv[ep]['ema_s']))
        dm.append(abs(ours_on_tv[ep]['eppv3_mid'] - tv[ep]['ema_m']))
    print("\n(A) MATH parity — our ema_pp_v3 on TV's own closes vs TV's EMA cols:")
    print(f"    EMA short |Δ|: {stats(ds)}")
    print(f"    EMA mid   |Δ|: {stats(dm)}")
    # entry-signal agreement on TV's own bars
    our_entry = {ep for ep in eps if ours_on_tv[ep]['eppv3_cross_short_up']}
    tv_entry = {ep for ep in eps if tv[ep]['entry']}
    print(f"    entry signals: ours={len(our_entry)} tv={len(tv_entry)} "
          f"agree={len(our_entry & tv_entry)} ours-only={len(our_entry - tv_entry)} "
          f"tv-only={len(tv_entry - our_entry)}")

    # ---- (B) DATA parity: our cache + REST bars vs TV bars ----
    from datetime import timedelta
    from data_loader import (ENGINE_CONSUMED_SOURCES, fetch_1s_bars_for_window,
                             fetch_cache_as_df, resample_to_timeframe)
    s = t0 - timedelta(seconds=5)
    e = t1 + timedelta(seconds=5)
    cache = fetch_cache_as_df('SPY', 10, s, e, value_type='first',
                              sources=ENGINE_CONSUMED_SOURCES)
    rest1s = fetch_1s_bars_for_window('SPY', s, e)
    rest = resample_to_timeframe(rest1s, '10Sec')
    rest = rest[(rest.index >= s) & (rest.index <= e)]

    def to_ep(df):
        d = {}
        for ts, r in df.iterrows():
            ep = int(ts.timestamp())
            d[ep] = {'open': float(r['open']), 'high': float(r['high']),
                     'low': float(r['low']), 'close': float(r['close'])}
        return d
    cache_b = to_ep(cache)
    rest_b = to_ep(rest)
    tvset = set(eps)
    for name, ours in (('CACHE (live saw)', cache_b), ('REST (backtest)', rest_b)):
        oset = set(ours)
        shared = tvset & oset
        cds = [abs(tv[ep]['close'] - ours[ep]['close']) for ep in shared]
        print(f"\n(B) DATA parity — TV vs {name}:")
        print(f"    bars: TV={len(tvset)} {name.split()[0]}={len(oset)} shared={len(shared)}")
        print(f"    TV-only bars (not in ours): {len(tvset - oset)}")
        print(f"    ours-only bars (not in TV): {len(oset - tvset)}")
        print(f"    close |Δ| on shared bars: {stats(cds)}")


if __name__ == '__main__':
    main()
