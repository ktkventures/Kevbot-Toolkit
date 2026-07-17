"""Classify the monitored fleet for M-RS5a resident-frame eligibility (Phase 2):
which slots are (A) no-secondary user-pack TRIGGER-only (OHLCV-only proven byte-identical),
(B) no-secondary but GATE on a user-pack interp (needs df interp state → not yet), or
(C) secondary-gated (Phase 2 secondary columns). Config-only, no bars/DB-heavy work.
"""
import os, sys
os.environ["USE_DB"] = "true"
from dotenv import load_dotenv
load_dotenv(".env", override=True)
import logging
logging.basicConfig(level=logging.ERROR)
sys.path.insert(0, ".")
import db
ADMIN = "19d47e46-f718-49a6-af32-5f5407f5b170"
db.set_admin_user_context(ADMIN)
import pack_registry
pack_registry.scan_and_load_all()
import general_packs as gp
from unified_engine import UnifiedStrategy
from shadow_engine import _BUILTIN_INTERPS
from shadow_manager import EngineSlot
from db import load_all_desired_states, load_strategies_monitoring_admin

gen = gp.get_enabled_general_packs(gp.load_general_packs())
desired = load_all_desired_states() or []
uids = sorted({d.get('user_id') for d in desired if d.get('user_id')})

A_trigger, B_gate, C_secondary, other = [], [], [], []
for uid in uids:
    try:
        strats = load_strategies_monitoring_admin(uid) or []
    except Exception:
        continue
    for strat in strats:
        if not strat.get('symbol') or 'entry_trigger_confluence_id' not in strat:
            continue
        if strat.get('strategy_origin') == 'webhook_inbound':
            continue
        sid = strat.get('id')
        if sid is None:
            continue
        try:
            slot = EngineSlot(sid, uid, strat)
        except Exception:
            continue
        if not slot.eligible:
            continue
        conf = strat.get('confluence') or []
        if slot.sec_tfs:
            C_secondary.append((sid, slot.symbol, slot.timeframe))
            continue
        # Does any confluence GATE leg reference a non-builtin (user-pack) interpreter?
        # Leg format: '{TF}-{INTERP}-{STATE}'. Extract the interp token robustly by
        # checking whether any non-builtin required interpreter name appears in a leg.
        try:
            us = UnifiedStrategy(strat, gen)
            ri = getattr(getattr(us, 'trigger_eval', None), 'required_interpreters', ()) or ()
            nonbuiltin = [k for k in ri if k not in _BUILTIN_INTERPS]
        except Exception:
            other.append((sid, slot.symbol, slot.timeframe, 'classify-fail'))
            continue
        gate_userpack = []
        for leg in conf:
            if not isinstance(leg, str) or leg.startswith('GEN-'):
                continue
            for nb in nonbuiltin:
                if nb in leg:
                    gate_userpack.append(leg)
        if gate_userpack:
            B_gate.append((sid, slot.symbol, slot.timeframe, gate_userpack[:2]))
        else:
            A_trigger.append((sid, slot.symbol, slot.timeframe, len(conf)))

print(f"\n=== A: no-secondary, TRIGGER-only (OHLCV-only proven) : {len(A_trigger)} ===")
for p in sorted(A_trigger, key=lambda x: (x[2], x[0])):
    print(f"  sid={p[0]:>4} {p[1]:<6} {p[2]:<7} conf_legs={p[3]}")
print(f"\n=== B: no-secondary but GATES on a user-pack interp (needs df state) : {len(B_gate)} ===")
for p in sorted(B_gate, key=lambda x: (x[2], x[0])):
    print(f"  sid={p[0]:>4} {p[1]:<6} {p[2]:<7} legs={p[3]}")
print(f"\n=== C: secondary-gated (Phase 2 secondary cols) : {len(C_secondary)} ===")
for p in sorted(C_secondary, key=lambda x: (x[2], x[0]))[:15]:
    print(f"  sid={p[0]:>4} {p[1]:<6} {p[2]:<7}")
if len(C_secondary) > 15:
    print(f"  ... +{len(C_secondary)-15} more")
print(f"\n=== other/unclassified : {len(other)} ===")
for p in other[:8]:
    print(f"  {p}")
print(f"\nTOTALS: A(trigger)={len(A_trigger)} B(userpack-gate)={len(B_gate)} "
      f"C(secondary)={len(C_secondary)} other={len(other)}")
