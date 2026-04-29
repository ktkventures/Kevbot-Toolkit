"""Time a single strategy refresh end-to-end so we can see whether
recompute vs auto-parity is the bottleneck. Run from src/:

    ../.venv/bin/python _time_one_refresh.py [SID]

Prints:
    [SID] recompute=X.XXs trades=N parity_verdict=... parity_score=...
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

import sys
import time
import logging

# Surface app-logger warnings so we see [PARITY:] / [FT-RECOMPUTE:] lines
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

import pack_registry
pack_registry.scan_and_load_all()

from db import set_admin_user_context
USER = "19d47e46-f718-49a6-af32-5f5407f5b170"
set_admin_user_context(USER)

SID = int(sys.argv[1]) if len(sys.argv) > 1 else 136

from api.services.forward_test_service import recompute_and_persist_stored_trades

print(f"\n>>> Starting refresh for sid {SID}", flush=True)
t0 = time.time()
result = recompute_and_persist_stored_trades(SID, USER, compute_parity=True)
elapsed = time.time() - t0
print(f"\n>>> sid {SID}: status={result.get('status')} trades={result.get('trades')} "
      f"elapsed={elapsed:.1f}s", flush=True)
parity = result.get('parity')
if parity:
    print(f"    parity verdict={parity['verdict']} score={parity['score']} "
          f"matched={parity['matched_count']}/{parity['stored_count']} "
          f"duration={parity['duration_s']}s", flush=True)
