"""One-shot snapshot of strategies' config.live_model with retry/backoff.

Saves to /tmp/strategies_live_model_snapshot_<utc>.json.  Used as a
revert point before bulk-flipping all strategies to ws_agg_locked.
"""
from dotenv import load_dotenv
load_dotenv(override=True)

import json
import time
from datetime import datetime, timezone
from collections import Counter

from db import get_admin_client


def main():
    c = get_admin_client()
    rows = None
    last_err = None
    for attempt in range(8):
        try:
            r = c.table('strategies').select('id,name,config').execute()
            rows = r.data or []
            break
        except Exception as e:
            last_err = e
            wait_s = min(60, 5 * (attempt + 1))
            print(f"[attempt {attempt+1}] failed: {str(e)[:120]} — retrying in {wait_s}s")
            time.sleep(wait_s)
    if rows is None:
        raise SystemExit(f"Snapshot failed after 8 attempts: {last_err}")

    snapshot = []
    for row in rows:
        cfg = row.get('config') or {}
        snapshot.append({
            'id': row['id'],
            'name': row.get('name', ''),
            'live_model_before': cfg.get('live_model'),
        })

    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_path = f'/tmp/strategies_live_model_snapshot_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(snapshot, f, indent=2)
    print(f"Snapshot saved: {out_path}")
    print(f"Total strategies: {len(snapshot)}")
    by = Counter(s['live_model_before'] for s in snapshot)
    print(f"Distribution: {dict(by)}")


if __name__ == "__main__":
    main()
