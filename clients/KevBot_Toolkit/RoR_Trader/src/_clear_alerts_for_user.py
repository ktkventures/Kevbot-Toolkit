"""Clear all alerts for Kevin's user_id (admin-context).

Run from src/:
  ../.venv/bin/python _clear_alerts_for_user.py
"""
from dotenv import load_dotenv
load_dotenv(override=True)

import sys
import time
from db import get_admin_client


def main():
    c = get_admin_client()
    # Resolve user_id from any strategy row (Kevin is the only user)
    r = c.table('strategies').select('user_id').limit(1).execute()
    if not r.data:
        sys.exit('No strategies found - cannot resolve user_id')
    uid = r.data[0]['user_id']

    # Pre-count
    r0 = c.table('alerts').select('id', count='exact').eq('user_id', uid).execute()
    pre = r0.count or 0
    print(f'User {uid[:8]}...: {pre} alerts before delete')

    if pre == 0:
        print('Nothing to delete.')
        return

    # Delete in chunks to avoid timeout on 50k+ rows
    deleted = 0
    while True:
        # Page 1000 ids at a time
        r = c.table('alerts').select('id').eq('user_id', uid).limit(1000).execute()
        ids = [row['id'] for row in (r.data or [])]
        if not ids:
            break
        c.table('alerts').delete().in_('id', ids).execute()
        deleted += len(ids)
        print(f'  deleted {deleted}/{pre}...')
        time.sleep(0.1)  # pace

    # Post-count
    r2 = c.table('alerts').select('id', count='exact').eq('user_id', uid).execute()
    post = r2.count or 0
    print(f'After: {post} alerts remaining (deleted {deleted})')


if __name__ == '__main__':
    main()
