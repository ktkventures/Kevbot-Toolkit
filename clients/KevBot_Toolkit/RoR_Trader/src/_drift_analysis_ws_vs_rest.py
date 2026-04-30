"""WS-vs-REST drift analysis.

Pulls live entry alerts from the alerts table (worker's view via alert.price)
and compares against Polygon REST close prices for the same bars. Computes
drift statistics by symbol, TF, and per-strategy.

Goal: validate (or refute) the hypothesis that WS-aggregated bars differ
from REST bars enough to affect indicator-driven triggers. Informs whether
the live-bar cache (Milestone 8.7) is the right architectural fix.

Output: drift summary table.
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import statistics

import pandas as pd

from db import get_admin_client, set_admin_user_context
from data_loader import load_market_data

USER = "19d47e46-f718-49a6-af32-5f5407f5b170"
HOURS_BACK = 96  # widen to 4 days to catch 5Min strategies + more samples
EXEC_TYPE_FILTER = ('C',)  # only C-type (bar close) — others have legitimate slippage
SAMPLE_SIZE_PER_GROUP = 8  # how many sample trades to print per (sym, tf) for verification

set_admin_user_context(USER)
c = get_admin_client()

since = (datetime.now(timezone.utc) - timedelta(hours=HOURS_BACK)).isoformat()
print(f"\nPulling alerts from last {HOURS_BACK}h (since {since[:19]} UTC)\n")

# Pull entry alerts with bar context
r = c.table('alerts').select(
    'id,strategy_id,strategy_name,symbol,timeframe,trigger_id,'
    'fill_ts,bar_time,price,exec_type,side'
).gte('timestamp', since).execute()
alerts = r.data or []
entry_alerts = [a for a in alerts
                if a.get('side') == 'entry'
                and a.get('exec_type') in EXEC_TYPE_FILTER]

print(f"Total entry+C alerts: {len(entry_alerts)}\n")

# Group by (symbol, timeframe) for batched REST loads
by_sym_tf: dict[tuple[str, str], list] = defaultdict(list)
for a in entry_alerts:
    sym = a.get('symbol')
    tf = a.get('timeframe')
    if sym and tf:
        by_sym_tf[(sym, tf)].append(a)

print(f"Groups (symbol, tf): {len(by_sym_tf)}")
for (sym, tf), group in by_sym_tf.items():
    print(f"  {sym}/{tf}: {len(group)} alerts")
print()

# For each group, load REST bars once and compare each alert
results = []  # per-alert {symbol, tf, bar_time, alert_price, rest_close, diff, diff_bps}
for (sym, tf), group in by_sym_tf.items():
    print(f"Loading REST {sym}/{tf} for {HOURS_BACK}h window...")
    try:
        df = load_market_data(symbol=sym, days=HOURS_BACK // 24 + 2,
                               timeframe=tf, session='RTH')
    except Exception as e:
        print(f"  failed: {e}")
        continue

    for a in group:
        bar_time_str = a.get('bar_time')
        if not bar_time_str:
            continue
        try:
            bar_ts = pd.Timestamp(bar_time_str)
            if bar_ts.tz is None:
                bar_ts = bar_ts.tz_localize('UTC')
            else:
                bar_ts = bar_ts.tz_convert('UTC')
        except Exception:
            continue

        if bar_ts not in df.index:
            results.append({
                'symbol': sym, 'tf': tf, 'sid': a.get('strategy_id'),
                'bar_time': bar_time_str,
                'alert_price': float(a.get('price', 0) or 0),
                'rest_close': None, 'diff': None, 'diff_bps': None,
                'in_rest': False,
            })
            continue

        rest_close = float(df.loc[bar_ts, 'close'])
        alert_price = float(a.get('price', 0) or 0)
        diff = alert_price - rest_close
        diff_bps = (diff / rest_close * 10000) if rest_close else None
        results.append({
            'symbol': sym, 'tf': tf, 'sid': a.get('strategy_id'),
            'bar_time': bar_time_str,
            'alert_price': alert_price,
            'rest_close': rest_close,
            'diff': diff,
            'diff_bps': diff_bps,
            'in_rest': True,
        })

print(f"\nResults: {len(results)} alerts compared\n")

# Per-(symbol, tf) summary
print("=" * 100)
print(f"{'symbol/tf':<14}  {'n':>5}  {'in_rest':>7}  {'matched':>7}  {'differ':>6}  "
      f"{'mean|diff|':>10}  {'max|diff|':>10}  {'p95|diff|':>10}")
print("-" * 100)

by_group_summary = defaultdict(list)
for r in results:
    if r['in_rest']:
        by_group_summary[(r['symbol'], r['tf'])].append(r)

overall_matched = overall_differ = overall_total = 0
for (sym, tf), rows in sorted(by_group_summary.items()):
    n = len(rows)
    matched = sum(1 for r in rows if abs(r['diff']) < 0.0001)
    differ = n - matched
    abs_diffs = [abs(r['diff']) for r in rows if abs(r['diff']) >= 0.0001]
    mean_d = statistics.mean(abs_diffs) if abs_diffs else 0.0
    max_d = max(abs_diffs) if abs_diffs else 0.0
    p95_d = (sorted(abs_diffs)[int(0.95 * len(abs_diffs))]
             if abs_diffs else 0.0)
    print(f"{sym + '/' + tf:<14}  {n:>5}  {n:>7}  "
          f"{matched:>7}  {differ:>6}  "
          f"{mean_d:>10.4f}  {max_d:>10.4f}  {p95_d:>10.4f}")
    overall_matched += matched
    overall_differ += differ
    overall_total += n

print("-" * 100)
print(f"{'TOTAL':<14}  {overall_total:>5}  {overall_total:>7}  "
      f"{overall_matched:>7}  {overall_differ:>6}  "
      f"({overall_matched/overall_total*100:.1f}% matched, "
      f"{overall_differ/overall_total*100:.1f}% differ)")

# Distribution of differences
print("\n=== Distribution of |diff| across all differing bars ===")
all_abs_diffs = sorted([abs(r['diff']) for r in results
                         if r['in_rest'] and abs(r['diff']) >= 0.0001])
if all_abs_diffs:
    n = len(all_abs_diffs)
    print(f"  N differing: {n}")
    print(f"  min: {all_abs_diffs[0]:.4f}")
    print(f"  p25: {all_abs_diffs[n // 4]:.4f}")
    print(f"  p50: {all_abs_diffs[n // 2]:.4f}")
    print(f"  p75: {all_abs_diffs[3 * n // 4]:.4f}")
    print(f"  p95: {all_abs_diffs[int(0.95 * n)]:.4f}")
    print(f"  max: {all_abs_diffs[-1]:.4f}")

# Asymmetry — is REST systematically higher or lower?
signed_diffs = [r['diff'] for r in results
                if r['in_rest'] and abs(r['diff']) >= 0.0001]
if signed_diffs:
    pos = sum(1 for d in signed_diffs if d > 0)
    neg = sum(1 for d in signed_diffs if d < 0)
    print(f"\n=== Signed direction (alert_price - rest_close) ===")
    print(f"  alert > REST (worker bar > REST bar): {pos}  ({pos/len(signed_diffs)*100:.1f}%)")
    print(f"  alert < REST (worker bar < REST bar): {neg}  ({neg/len(signed_diffs)*100:.1f}%)")

# How many alerts differed by > 1 cent (meaningful for indicators)
meaningful = sum(1 for r in results
                 if r['in_rest'] and abs(r['diff'] or 0) >= 0.01)
print(f"\n=== Drift severity ===")
print(f"  alerts differing by ≥$0.01: {meaningful} of {overall_total} ({meaningful/overall_total*100:.1f}%)")
print(f"  alerts differing by ≥$0.05: "
      f"{sum(1 for r in results if r['in_rest'] and abs(r['diff'] or 0) >= 0.05)} "
      f"({sum(1 for r in results if r['in_rest'] and abs(r['diff'] or 0) >= 0.05)/overall_total*100:.1f}%)")
print(f"  alerts differing by ≥$0.10: "
      f"{sum(1 for r in results if r['in_rest'] and abs(r['diff'] or 0) >= 0.10)} "
      f"({sum(1 for r in results if r['in_rest'] and abs(r['diff'] or 0) >= 0.10)/overall_total*100:.1f}%)")

# Sample trades for visual verification — mix of matched and differing per group
print("\n" + "=" * 100)
print("SAMPLE TRADES FOR VISUAL VERIFICATION")
print("=" * 100)
print("Use these to cross-check against your charts. Pull up the strategy in")
print("the UI, navigate to the bar_time, compare the chart's bar close to the")
print("'rest_close' here, and the algo-history alert price to the 'alert_price' here.\n")

import random
random.seed(42)
for (sym, tf), rows in sorted(by_group_summary.items()):
    matched_rows = [r for r in rows if abs(r['diff']) < 0.0001]
    differ_rows = [r for r in rows if abs(r['diff']) >= 0.0001]
    n_match_show = min(SAMPLE_SIZE_PER_GROUP // 2, len(matched_rows))
    n_differ_show = min(SAMPLE_SIZE_PER_GROUP - n_match_show, len(differ_rows))
    sample_matched = random.sample(matched_rows, n_match_show) if matched_rows else []
    # Sort differing by absolute diff to show range
    differ_sorted = sorted(differ_rows, key=lambda r: abs(r['diff']))
    if n_differ_show <= 0 or not differ_sorted:
        sample_differ = []
    elif len(differ_sorted) <= n_differ_show:
        sample_differ = differ_sorted
    else:
        # Take evenly-spaced samples across the drift range
        step = max(1, len(differ_sorted) // n_differ_show)
        sample_differ = differ_sorted[::step][:n_differ_show]

    print(f"\n--- {sym}/{tf} ---")
    print(f"  MATCHED ({len(matched_rows)} total, sample of {len(sample_matched)}):")
    for r in sample_matched:
        print(f"    sid={r['sid']}  bar_time={r['bar_time']}  "
              f"alert={r['alert_price']:.4f}  rest={r['rest_close']:.4f}  diff=$0.0000")
    print(f"  DIFFER ({len(differ_rows)} total, sample of {len(sample_differ)} across drift range):")
    for r in sample_differ:
        sign = '+' if r['diff'] > 0 else '-'
        print(f"    sid={r['sid']}  bar_time={r['bar_time']}  "
              f"alert={r['alert_price']:.4f}  rest={r['rest_close']:.4f}  "
              f"diff={sign}${abs(r['diff']):.4f}")

# Per-strategy summary too
print("\n" + "=" * 100)
print("PER-STRATEGY DRIFT SUMMARY")
print("=" * 100)
print(f"{'sid':>5}  {'symbol/tf':<14}  {'n':>5}  {'match':>6}  {'differ':>6}  "
      f"{'mean|diff|':>10}  {'max|diff|':>10}")
print("-" * 80)
by_sid: dict = defaultdict(list)
for r in results:
    if r['in_rest']:
        by_sid[(r['sid'], r['symbol'], r['tf'])].append(r)
for (sid, sym, tf), rows in sorted(by_sid.items()):
    n = len(rows)
    matched = sum(1 for r in rows if abs(r['diff']) < 0.0001)
    differ = n - matched
    abs_diffs = [abs(r['diff']) for r in rows if abs(r['diff']) >= 0.0001]
    mean_d = statistics.mean(abs_diffs) if abs_diffs else 0.0
    max_d = max(abs_diffs) if abs_diffs else 0.0
    print(f"{sid:>5}  {sym + '/' + tf:<14}  {n:>5}  {matched:>6}  {differ:>6}  "
          f"{mean_d:>10.4f}  {max_d:>10.4f}")
