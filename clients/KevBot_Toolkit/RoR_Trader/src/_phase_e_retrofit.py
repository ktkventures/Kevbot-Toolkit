"""Phase E retrofit — run the synthetic-strategy fire test on each
existing v2 user pack to validate that yesterday's parity fixes
propagate correctly. Each pack should produce a sensible verdict
(PASS or PARTIAL with documented residuals); a FAIL on a previously-
trustworthy pack is a regression to investigate.

Packs covered (per docs/User_Pack_Parity_Baseline_2026-04-29.md):
  - macd_line_v2, macd_histogram_v2, vwap_v2, rvol_v2
  - ema_stack_v2, ema_pp_v3, ema_pp_v4
  - ut_bot_v4, swing_123, swing_123_test

Each probe spawns a synthetic strategy on SPY/1Min/7d, runs backtest
+ parity, returns a verdict, and tears down the synthetic strategy.
Total expected wall-clock: ~3-5 minutes (semaphore=1).
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

import json
import time

import pack_registry
pack_registry.scan_and_load_all()

from db import set_admin_user_context
from api.services.synthetic_probe import run_pack_probe

USER = "19d47e46-f718-49a6-af32-5f5407f5b170"
PACKS = [
    'macd_line_v2',
    'macd_histogram_v2',
    'vwap_v2',
    'rvol_v2',
    'ema_stack_v2',
    'ema_pp_v3',
    'ema_pp_v4',
    'ut_bot_v4',
    'swing_123',
    'swing_123_test',
]

set_admin_user_context(USER)

print(f"\n{'='*100}")
print(f"Phase E retrofit — synthetic probe on {len(PACKS)} v2 packs")
print(f"{'='*100}\n")

results = []
for pack_slug in PACKS:
    print(f"\n[{pack_slug}] running...")
    t0 = time.time()
    try:
        result = run_pack_probe(
            pack_slug, USER,
            symbol='SPY', timeframe='1Min', days=7,
            add_cross_tf_gate=False, cleanup=True,
        )
        elapsed = time.time() - t0
        results.append({'pack': pack_slug, 'elapsed_s': elapsed, **result})
        v = result.get('verdict')
        s = result.get('score')
        st = result.get('status')
        m = result.get('matched_count')
        sc = result.get('stored_count')
        score_str = f'{s:.2f}' if isinstance(s, (int, float)) else '-'
        print(f"  {st:<10} verdict={v:<22} score={score_str:>6} "
              f"matched={m}/{sc}  elapsed={elapsed:.0f}s")
    except Exception as e:
        elapsed = time.time() - t0
        results.append({'pack': pack_slug, 'elapsed_s': elapsed,
                         'status': 'error', 'error': str(e)})
        print(f"  ERROR  {type(e).__name__}: {e}  elapsed={elapsed:.0f}s")

print(f"\n{'='*100}")
print(f"SUMMARY")
print(f"{'='*100}")
print(f"{'pack':<22} {'status':<10} {'verdict':<22} {'score':>6}  matched/stored")
print('-' * 100)
for r in results:
    pack = r['pack']
    st = r.get('status', '?')
    v = r.get('verdict') or '-'
    s = r.get('score')
    score_str = f'{s:.2f}' if isinstance(s, (int, float)) else '-'
    m = r.get('matched_count', '-')
    sc = r.get('stored_count', '-')
    print(f"{pack:<22} {st:<10} {v:<22} {score_str:>6}  {m}/{sc}")

# Roll-up counts
pass_n = sum(1 for r in results if r.get('status') == 'pass')
partial_n = sum(1 for r in results if r.get('status') == 'partial')
fail_n = sum(1 for r in results if r.get('status') == 'fail')
err_n = sum(1 for r in results if r.get('status') == 'error')
skip_n = sum(1 for r in results if r.get('status') == 'skipped')

print(f"\nVERDICTS: {pass_n} pass · {partial_n} partial · {fail_n} fail · "
      f"{skip_n} skipped · {err_n} error")
