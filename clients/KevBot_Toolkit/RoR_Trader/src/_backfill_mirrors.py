"""Sequentially backfill the remaining mirror strategies. Single Python
process so the in-process bar cache shares work across same-symbol mirrors:

- SPY 1Min: sid 136, 140 (cold first, then warm)
- TSLA 5Min: sid 137, 138 (cold first, then warm)
- TSLA 10Sec: sid 139, 142, 143 (cold first, then 2 warm)
- TSLL 1Min: sid 141 (cold)
- META 1Min: 135 already done, skip

(sid 144 AMD 1Min already done as canary.)
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

import json
import sys
import time

import pack_registry
pack_registry.scan_and_load_all()

from db import set_admin_user_context, clear_current_user, get_admin_client

USER = "19d47e46-f718-49a6-af32-5f5407f5b170"
set_admin_user_context(USER)

# Order matters: cluster same symbol/tf together so the in-process bar
# cache is warm for the second+ in each group. 135 + 144 already have
# trades from earlier sessions but lack parity_status (auto-parity is
# new as of 2026-04-28 dbe0cb3); refreshing them is idempotent and
# stamps them with the score.
ORDER = [
    136,  # SPY 1Min cold
    140,  # SPY 1Min warm
    135,  # META 1Min cold (separate symbol, no SPY cache benefit)
    137,  # TSLA 5Min cold
    138,  # TSLA 5Min warm
    139,  # TSLA 10Sec cold
    142,  # TSLA 10Sec warm
    143,  # TSLA 10Sec warm
    141,  # TSLL 1Min cold
    144,  # AMD 1Min cold
]

from api.services.forward_test_service import recompute_and_persist_stored_trades

results = []
for sid in ORDER:
    t0 = time.time()
    try:
        result = recompute_and_persist_stored_trades(sid, USER)
        elapsed = time.time() - t0
        trades = result.get('trades', 0)
        status = result.get('status', '?')
        print(f"  sid {sid}: {status} {trades} trades in {elapsed:.0f}s",
              flush=True)
        results.append((sid, status, trades, elapsed))
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  sid {sid}: ERROR after {elapsed:.0f}s — {type(e).__name__}: {e}",
              flush=True)
        results.append((sid, 'ERROR', 0, elapsed))

print("\n" + "=" * 60)
print("BACKFILL SUMMARY")
print("=" * 60)
total = sum(e for _, _, _, e in results)
print(f"  Total wall time: {total:.0f}s ({total/60:.1f} min)")
ok = sum(1 for _, s, _, _ in results if s == 'refreshed')
print(f"  Refreshed: {ok}/{len(results)}")
for sid, status, trades, elapsed in results:
    print(f"  sid {sid}: {status:<12} {trades:>4} trades  {elapsed:>5.0f}s")
