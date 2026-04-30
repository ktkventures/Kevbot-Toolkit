"""One-off: backfill backtests for sids 147, 148 which were created
during the migration outage but never got their stored_trades populated.
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

import time
import pack_registry
pack_registry.scan_and_load_all()

from db import set_admin_user_context
from api.services.forward_test_service import recompute_and_persist_stored_trades

USER = "19d47e46-f718-49a6-af32-5f5407f5b170"
SIDS = [147, 148]

set_admin_user_context(USER)

for sid in SIDS:
    print(f"\nBacktesting sid {sid}...")
    t0 = time.time()
    try:
        result = recompute_and_persist_stored_trades(sid, USER, compute_parity=False)
        print(f"  ✓ status={result.get('status')} trades={result.get('trades')} elapsed={time.time()-t0:.0f}s")
    except Exception as e:
        print(f"  ✗ {type(e).__name__}: {e} (after {time.time()-t0:.0f}s)")

print("\nDone.")
