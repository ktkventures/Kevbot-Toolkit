"""Phase 1: run 4-quadrant parity on the 8 v2 packs that the Tier-A
mirror strategies actually use, persist results to
user_pack_parity_status, and print a verdict matrix.

Per the migration plan (/home/kevin/.claude/plans/eager-wiggling-twilight.md):
'Strategy-level parity scores are meaningless if the underlying pack
has batch↔live drift — we'd waste time chasing strategy-level fixes
for a pack-level bug.'

Run from src/:
    ../.venv/bin/python _run_pack_parity_baseline.py [--apply]

Without --apply: dry-run; runs the tests but doesn't persist.
With --apply: also writes to user_pack_parity_status table.
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

import json
import sys
import time

import pack_registry
pack_registry.scan_and_load_all()

from db import set_admin_user_context, save_pack_parity_status
from parity_simulator import run_pack_parity_test_4q

USER = "19d47e46-f718-49a6-af32-5f5407f5b170"
APPLY = '--apply' in sys.argv

set_admin_user_context(USER)

# The 8 packs the Tier-A mirror strategies use. ut_bot_v4 + swing_123
# already have records (per memory), but we re-run for fresh reads.
PACKS = [
    'macd_line_v2',
    'macd_histogram_v2',
    'vwap_v2',
    'rvol_v2',
    'ema_stack_v2',
    'ema_pp_v3',
    'ema_pp_v4',
    'swing_123_test',
    # Re-test these too to confirm prior FAIL status hasn't changed
    'swing_123',
    'ut_bot_v4',
]

TEST_CONFIG = {
    'symbol': 'SPY',
    'primary_tf': '1Min',
    'secondary_tf': '15Min',
    'days': 7,
    'session': 'RTH',
    'feed': 'sip',
    'warmup_bars': 200,
}

print(f"\n=== Pack 4Q parity baseline ({'APPLY' if APPLY else 'DRY RUN'}) ===")
print(f"Config: {TEST_CONFIG}")
print(f"Packs: {len(PACKS)}\n")

results = []
for slug in PACKS:
    print(f"  Testing {slug}...", flush=True)
    t0 = time.time()
    try:
        report = run_pack_parity_test_4q(
            slug,
            symbol=TEST_CONFIG['symbol'],
            primary_tf=TEST_CONFIG['primary_tf'],
            secondary_tf=TEST_CONFIG['secondary_tf'],
            days=TEST_CONFIG['days'],
            session=TEST_CONFIG['session'],
            feed=TEST_CONFIG['feed'],
            warmup_bars=TEST_CONFIG['warmup_bars'],
        )
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"    ERROR after {elapsed:.0f}s — {type(exc).__name__}: {exc}",
              flush=True)
        results.append({
            'pack_slug': slug, 'overall_verdict': 'ERROR',
            'summary': f"{type(exc).__name__}: {exc}",
            'duration_s': elapsed, 'quadrants': {},
        })
        continue

    elapsed = time.time() - t0
    overall = report.get('overall_verdict', 'UNKNOWN')
    summary = report.get('summary', '')
    qs = report.get('quadrants', {}) or {}
    print(f"    {overall:<5} in {elapsed:.0f}s  |  {summary}", flush=True)
    results.append({
        'pack_slug': slug,
        'overall_verdict': overall,
        'summary': summary,
        'duration_s': elapsed,
        'quadrants': qs,
    })

    if APPLY and overall in ('PASS', 'WARN', 'FAIL'):
        try:
            save_pack_parity_status(
                pack_slug=slug, user_id=USER,
                overall_verdict=overall, summary=summary,
                quadrants=qs, test_config=TEST_CONFIG)
            print(f"    ✓ persisted")
        except Exception as exc:
            print(f"    persist failed: {exc}")

# Summary table
print("\n" + "=" * 110)
print("BASELINE VERDICT MATRIX")
print("=" * 110)
print(f"{'pack':<22} {'overall':<8} {'Q1':<6} {'Q2':<6} {'Q3':<6} {'Q4':<6} {'time':>6}  summary")
print("-" * 110)
for r in results:
    qs = r.get('quadrants', {}) or {}
    q1 = (qs.get('Q1') or {}).get('verdict', '-')
    q2 = (qs.get('Q2') or {}).get('verdict', '-')
    q3 = (qs.get('Q3') or {}).get('verdict', '-')
    q4 = (qs.get('Q4') or {}).get('verdict', '-')
    print(f"{r['pack_slug']:<22} {r['overall_verdict']:<8} "
          f"{q1:<6} {q2:<6} {q3:<6} {q4:<6} "
          f"{r['duration_s']:>5.0f}s  {r['summary'][:50]}")

print("\nDone.")
