from __future__ import annotations
from dotenv import load_dotenv; load_dotenv(override=True)
from collections import Counter
import pandas as pd
import pack_registry; pack_registry.scan_and_load_all()
from db import get_admin_client, _row_to_strategy
from ralph_engine import StrategyMonitor, SymbolHub
from data_loader import load_market_data, resample_to_timeframe

c = get_admin_client()
strat = _row_to_strategy(c.table('strategies').select('*').eq('id',277).execute().data[0])
mon = StrategyMonitor(strat, None, general_packs=[])
hub = SymbolHub('SPY'); hub.add_monitor(mon); hub.finalize_shadow_engines()
sh = hub._shadow_engines[120]

df2 = resample_to_timeframe(load_market_data('SPY', days=3, timeframe='1Min'), '2Min').dropna(subset=['open'])
warm, replay = df2.iloc[:60], df2.iloc[60:]
sh.warmup(warm)

gate = '2M-BOLLINGER_BANDS-SQUEEZE_UPPER'
states = Counter(); passes = 0
for ts, r in replay.iterrows():
    recs = sh.on_bar_close({'open':float(r.open),'high':float(r.high),'low':float(r.low),
                            'close':float(r.close),'volume':float(r.volume),'timestamp':ts.isoformat()})
    for rec in recs:
        if rec.startswith('2M-BOLLINGER_BANDS-'):
            states[rec.split('-',2)[2]] += 1
    if gate in recs: passes += 1
n = len(replay)
print(f'Replayed {n} closed 2m bars (last 3 trading days)')
print('BOLLINGER_BANDS state distribution:')
for s,k in states.most_common(): print(f'   {s:<18} {k:>4}  ({100*k/n:.1f}%)')
print(f'\nGate "{gate}" would PASS on {passes}/{n} bars = {100*passes/n:.1f}%')
print(f'=> a FRESH gate blocks ~{100*(n-passes)/n:.0f}% of entries; '
      f'frozen-on-squeeze passes 100%. That gap IS the flood.')
