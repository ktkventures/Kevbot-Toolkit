"""Offline TV-export readiness audit: run tv_export_readiness() over every
strategy and report which export cleanly vs. what's missing, grouped by gap.

This is the definitive offline work-list for emitter coverage. It shares the
SAME code path as the GET /strategies/{id}/pine-readiness endpoint
(pine_generator.tv_export_readiness) — so the audit and the live badge can never
disagree. See docs/_active/Design_TV_Export_Readiness.md.

Read-only against prod (admin client, bypasses RLS). Run from src/:
    ../.venv/bin/python _tv_emitter_audit.py
"""
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv(override=True)

from db import get_admin_client
from pine_generator import tv_export_readiness
from tv_export_canaries import coverage_summary


def main():
    client = get_admin_client()
    res = (client.table('strategies')
           .select('id,name,symbol,timeframe,config,user_id')
           .execute())
    strats = res.data or []
    print(f"Auditing {len(strats)} strategies\n")

    ready, not_ready = [], []
    by_gap = defaultdict(list)

    for s in strats:
        r = tv_export_readiness(s)
        label = (f"sid {s['id']:>4} {s.get('symbol') or '?':<6} "
                 f"{(s.get('name') or '')[:34]:<34}").strip()
        if r['ready']:
            ready.append(s['id'])
        else:
            not_ready.append(s['id'])
            for gap in r['missing']:
                by_gap[gap['detail']].append(label)

    print(f"=== RESULT: {len(ready)} READY / {len(not_ready)} NOT-READY ===\n")
    print("--- GAPS grouped (every blocking issue, not just the first) ---")
    for detail, items in sorted(by_gap.items(), key=lambda kv: -len(kv[1])):
        print(f"\n[{len(items)}] {detail}")
        for it in items[:25]:
            print(f"      {it}")
        if len(items) > 25:
            print(f"      ... +{len(items) - 25} more")

    cov = coverage_summary()
    print(f"\n--- PARITY CANARIES (validated vs merely-emitting) ---")
    print(f"   {cov['validated']} validated, {cov['pending']} pending "
          f"of {cov['total']} tracked components")


if __name__ == '__main__':
    main()
