"""iter 0609a — prove the rebroadcast cascade now refreshes _mtf_confluence
for 2m+ secondary gates instead of freezing it. Drives the REAL
SymbolHub._fanout_to_secondary_builders duplicate path on sid 277."""
from __future__ import annotations
from dotenv import load_dotenv; load_dotenv(override=True)
from datetime import datetime, timezone, timedelta
import pandas as pd
import pack_registry; pack_registry.scan_and_load_all()

from db import get_admin_client, _row_to_strategy
from ralph_engine import StrategyMonitor, SymbolHub
from data_loader import load_market_data
from data_loader import resample_to_timeframe

c = get_admin_client()
row = c.table('strategies').select('*').eq('id', 277).execute().data[0]
strat = _row_to_strategy(row)
print('strat:', strat['id'], strat['symbol'], strat['timeframe'],
      'confluence=', strat.get('confluence'))

mon = StrategyMonitor(strat, None, general_packs=[])
print('required_secondary_tf:', mon._required_secondary_tf)
print('confluence_set (normalized):', mon.position.confluence_set)

hub = SymbolHub('SPY')
hub.add_monitor(mon)
hub.finalize_shadow_engines()
print('shadow engines:', list(hub._shadow_engines.keys()))
sh = hub._shadow_engines.get((120, 'RTH'))
assert sh is not None, '120s shadow missing!'
print('120s shadow req_interp:', sh.trigger_eval.req_interp if hasattr(sh.trigger_eval,'req_interp') else '(n/a)')

# Warm up the 120s shadow + builder with real recent SPY 2m history
df1 = load_market_data('SPY', days=3, timeframe='1Min')
df2 = resample_to_timeframe(df1, '2Min').dropna(subset=['open'])
warm = df2.iloc[:-3]
sh.warmup(warm)
hub.builders[120].seed_history(warm)
print(f'warmed shadow with {len(warm)} 2m bars; last warm close={warm["close"].iloc[-1]:.2f}')

def minute_bar(ts, o,h,l,cl,v=1000):
    return {'open':o,'high':h,'low':l,'close':cl,'volume':v,'timestamp':ts.isoformat()}

# Build 3 consecutive minute bars straddling a 2m boundary so a 2m bar closes,
# then replay one as a rebroadcast (duplicate) to hit the cascade branch.
base = pd.Timestamp(warm.index[-1]).tz_convert('UTC') + pd.Timedelta(minutes=2)
m0 = base                        # 2m bucket A start
m1 = base + pd.Timedelta(minutes=1)
m2 = base + pd.Timedelta(minutes=2)   # closes bucket A, opens bucket B
px = float(warm['close'].iloc[-1])

hub._fanout_to_secondary_builders(minute_bar(m0, px, px+1, px-1, px+0.5), 'ws_agg')
hub._fanout_to_secondary_builders(minute_bar(m1, px+0.5, px+2, px, px+1.5), 'ws_agg')
hub._fanout_to_secondary_builders(minute_bar(m2, px+1.5, px+2, px+1, px+1.8), 'ws_agg')
rec_after_close = set(hub._mtf_confluence.get((120, 'RTH'), set()))
print('\n_mtf_confluence[120] after normal closes:', rec_after_close)

# Now a rebroadcast/correction of m0 (period_start aligns to closed bucket A)
# with a DIFFERENT close — exercises the duplicate branch.
hub._fanout_to_secondary_builders(minute_bar(m0, px, px+5, px-1, px+4.0), 'ws')
rec_after_dup = set(hub._mtf_confluence.get((120, 'RTH'), set()))
print('_mtf_confluence[120] after rebroadcast dup :', rec_after_dup)

# Assertions
assert (120, 'RTH') in hub._mtf_confluence, 'FAIL: _mtf_confluence[120] missing after dup'
assert rec_after_dup == set(sh._current_confluence), \
    'FAIL: buffer not synced to shadow current records'
print('\nPASS: rebroadcast cascade refreshed _mtf_confluence[120] '
      '(was frozen before the fix).')
print('records changed by correction:', rec_after_dup != rec_after_close)
