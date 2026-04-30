"""Cross-check the 8 MIRRORABLE originals still alive: do they have
recent alerts/trades that justify keeping them as a side-by-side reference?
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

from datetime import datetime, timedelta, timezone
from db import get_admin_client, set_admin_user_context

USER = "19d47e46-f718-49a6-af32-5f5407f5b170"
ORIGS_AND_MIRRORS = [
    (50,  136, 'SPY LONG - Mass #11',     'SPY/1Min'),
    (51,  135, 'META LONG - Mass #13',    'META/1Min'),
    (88,  149, '10s test',                'SPY/10Sec'),
    (114, 150, 'ext 4 16 test 10 second', 'SPY/10Sec'),
    (117, 151, '10s Close',               'SPY/10Sec'),
    (124, 152, 'yearp1',                  'SPY/1Min'),
    (131, 153, 'SPY LONG - 1',            'SPY/10Sec'),
    (134, 154, 'SPY LONG 10Sec Mass #1',  'SPY/10Sec'),
]

set_admin_user_context(USER)
c = get_admin_client()

# Pull last 14 days of alerts + forward trades grouped by strategy_id
since = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()

print(f"\n{'sid':>4}  {'mirror':>6}  {'name':<28}  {'sym/tf':<12}  {'alerts_14d':>10}  {'fwd_trades_14d':>14}  monitored?")
print('-' * 110)

for orig_sid, mirror_sid, name, symtf in ORIGS_AND_MIRRORS:
    # Alerts
    a = c.table('alerts').select('id', count='exact').eq('strategy_id', orig_sid).gte('timestamp', since).execute()
    alerts_n = a.count or 0

    # Live trades (data_source='live')
    t = c.table('trades').select('id', count='exact').eq('strategy_id', orig_sid).eq('data_source', 'live').gte('created_at', since).execute()
    trades_n = t.count or 0

    # Monitor state
    s = c.table('strategies').select('forward_testing,alert_tracking_enabled').eq('id', orig_sid).single().execute()
    fwd = (s.data or {}).get('forward_testing')
    alt = (s.data or {}).get('alert_tracking_enabled')
    mon = f"fwd={fwd!s:<5}  alert={alt!s}"

    print(f"{orig_sid:>4}  {mirror_sid:>6}  {name[:28]:<28}  {symtf:<12}  {alerts_n:>10}  {trades_n:>14}  {mon}")

# Also: sanity check 132/126/129/111 — pure user-pack originals (no mirror in name)
print("\n\nPure user-pack strategies (no [mirror] tag — verify they're intentional, not stragglers):")
for sid in [111, 126, 129, 132]:
    s = c.table('strategies').select('id,name,symbol,timeframe,forward_testing,alert_tracking_enabled').eq('id', sid).single().execute()
    d = s.data or {}
    a = c.table('alerts').select('id', count='exact').eq('strategy_id', sid).gte('timestamp', since).execute()
    print(f"  sid {sid:>3}: {d.get('name', '?'):<35}  {d.get('symbol', '?')}/{d.get('timeframe', '?'):<6}  alerts_14d={a.count or 0:>4}  fwd={d.get('forward_testing')}")
