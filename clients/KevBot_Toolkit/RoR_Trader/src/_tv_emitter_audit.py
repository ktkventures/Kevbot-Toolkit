"""Offline audit: run generate_pine() over every strategy and report which
ones export cleanly vs. fail, grouped by failure reason / missing pack.

This is the definitive work-list for TradingView-export emitter coverage:
it mirrors the GET /strategies/{id}/pine-export endpoint exactly (same
generate_pine call, same ValueError surface) but sweeps all strategies at once.

Read-only against prod (admin client, bypasses RLS). Run from src/:
    ../.venv/bin/python _tv_emitter_audit.py
"""
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv(override=True)

from db import get_admin_client
from pine_generator import generate_pine


def _trigger_pack(trig):
    """Best-effort pack label from a trigger string's prefix."""
    if not trig:
        return None
    return trig.split('_')[0] + '_*'


def main():
    client = get_admin_client()
    res = (client.table('strategies')
           .select('id,name,symbol,timeframe,config,user_id')
           .execute())
    strats = res.data or []
    print(f"Auditing {len(strats)} strategies\n")

    ok, fail = [], []
    fail_by_reason = defaultdict(list)
    pack_usage = defaultdict(lambda: {'ok': 0, 'fail': 0})

    for s in strats:
        cfg = s.get('config') or {}
        entry = cfg.get('entry_trigger')
        gates = [g.get('interpreter') or g.get('pack') or g.get('name')
                 for g in (cfg.get('confluence') or []) if isinstance(g, dict)]
        label = f"sid {s['id']:>4} {s.get('symbol') or '?':<6} {(s.get('name') or '')[:34]:<34}"
        try:
            generate_pine(s)
            ok.append(s['id'])
            for p in [_trigger_pack(entry)] + gates:
                if p:
                    pack_usage[p]['ok'] += 1
        except ValueError as e:
            reason = str(e)
            fail.append(s['id'])
            fail_by_reason[reason].append(label.strip())
            for p in [_trigger_pack(entry)] + gates:
                if p:
                    pack_usage[p]['fail'] += 1
        except Exception as e:
            reason = f"[NON-ValueError {type(e).__name__}] {e}"
            fail.append(s['id'])
            fail_by_reason[reason].append(label.strip())

    print(f"=== RESULT: {len(ok)} OK / {len(fail)} FAIL ===\n")
    print("--- FAILURES grouped by reason ---")
    for reason, items in sorted(fail_by_reason.items(),
                                key=lambda kv: -len(kv[1])):
        print(f"\n[{len(items)}] {reason}")
        for it in items[:25]:
            print(f"      {it}")
        if len(items) > 25:
            print(f"      ... +{len(items) - 25} more")

    print("\n--- PACK USAGE (entry-prefix + gate interpreters) ---")
    for p, c in sorted(pack_usage.items(), key=lambda kv: -(kv[1]['ok'] + kv[1]['fail'])):
        print(f"   {p:<22} ok={c['ok']:>3}  fail={c['fail']:>3}")


if __name__ == '__main__':
    main()
