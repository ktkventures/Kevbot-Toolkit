"""Smoke test: run synthetic probe on a known-good pack to verify
scaffold works end-to-end. Tries ema_pp_v4 (continuous-state, simplest)
with no cross-TF gate first, then with one.
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

import json

import pack_registry
pack_registry.scan_and_load_all()

from db import set_admin_user_context
from api.services.synthetic_probe import run_pack_probe, build_synthetic_strategy

USER = "19d47e46-f718-49a6-af32-5f5407f5b170"
set_admin_user_context(USER)

# Step 1: introspect what the probe builds
print("=== Probe configuration introspection ===")
for slug in ['ema_pp_v4', 'ut_bot_v4', 'macd_line_v2']:
    strat = build_synthetic_strategy(slug, USER, days=7, add_cross_tf_gate=True)
    if strat is None:
        print(f"  {slug}: SKIPPED (not eligible)")
        continue
    print(f"  {slug}:")
    print(f"    entry: {strat.get('entry_trigger_confluence_id')}")
    print(f"    exit:  {strat.get('exit_trigger_confluence_ids')}")
    print(f"    conf:  {strat.get('confluence')}")

# Step 2: actually run probe on ema_pp_v4 with no cross-TF (simplest case)
print("\n=== Running probe on ema_pp_v4 (no cross-TF gate) ===")
verdict = run_pack_probe('ema_pp_v4', USER, add_cross_tf_gate=False, cleanup=True)
print(json.dumps(verdict, indent=2, default=str))
