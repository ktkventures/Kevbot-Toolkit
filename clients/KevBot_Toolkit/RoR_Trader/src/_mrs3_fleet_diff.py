"""Diff the post-fleet-recompute persisted state vs the pre-run benchmark.

Loads /tmp/mrs3_benchmark_LATEST.json (the snapshot taken before the parallel
fleet run), takes a fresh snapshot, and reports per-strategy deltas in backtest
trade count, date range, and KPIs. Read-only.
"""
import json
import os

os.environ["USE_DB"] = "true"
from dotenv import load_dotenv  # noqa: E402
load_dotenv(".env", override=True)
import db  # noqa: E402

USER = "19d47e46-f718-49a6-af32-5f5407f5b170"
KPI_KEYS = ["num_trades", "total_trades", "total_r", "total_return",
            "win_rate", "sharpe", "sharpe_ratio", "max_drawdown",
            "profit_factor", "avg_r", "daily_r"]


def snapshot():
    c = db.get_admin_client()
    strats = (c.table('strategies').select('id,name,kpis')
              .eq('user_id', USER).execute().data)
    by = {s['id']: s for s in strats}
    out = {}
    for sid in sorted(by):
        cnt = (c.table('trades').select('id', count='exact')
               .eq('strategy_id', sid).like('data_source', 'backtest_%')
               .execute().count)
        lo = (c.table('trades').select('entry_fill_ts').eq('strategy_id', sid)
              .like('data_source', 'backtest_%').order('entry_fill_ts')
              .limit(1).execute().data)
        hi = (c.table('trades').select('entry_fill_ts').eq('strategy_id', sid)
              .like('data_source', 'backtest_%')
              .order('entry_fill_ts', desc=True).limit(1).execute().data)
        out[str(sid)] = {
            'name': by[sid].get('name'),
            'bt_count': cnt,
            'earliest': lo[0]['entry_fill_ts'] if lo else None,
            'latest': hi[0]['entry_fill_ts'] if hi else None,
            'kpis': by[sid].get('kpis') or {},
        }
    return out


def _num(v):
    try:
        return round(float(v), 4)
    except Exception:  # noqa: BLE001
        return v


def main():
    bench = json.load(open("/tmp/mrs3_benchmark_LATEST.json"))['strategies']
    after = snapshot()
    json.dump({'strategies': after}, open("/tmp/mrs3_after.json", "w"), default=str)

    identical = []
    count_changed = []
    kpi_only = []
    missing = []
    tot_b = tot_a = 0
    for sid in sorted(bench, key=int):
        b, a = bench[sid], after.get(sid)
        if a is None:
            missing.append(sid)
            continue
        bc, ac = b['bt_count'], a['bt_count']
        tot_b += bc
        tot_a += ac
        cnt_same = bc == ac
        date_same = (b['earliest'] == a['earliest'] and b['latest'] == a['latest'])
        kdiff = {}
        for k in KPI_KEYS:
            bv, av = _num((b['kpis'] or {}).get(k)), _num((a['kpis'] or {}).get(k))
            if bv != av and not (bv is None and av is None):
                kdiff[k] = (bv, av)
        if cnt_same and date_same and not kdiff:
            identical.append(sid)
        elif not cnt_same or not date_same:
            count_changed.append((sid, b['name'], bc, ac, b['earliest'], a['earliest'], kdiff))
        else:
            kpi_only.append((sid, b['name'], kdiff))

    print(f"=== FLEET RECOMPUTE DIFF (parallel run vs benchmark) ===")
    print(f"strategies: {len(bench)} | fleet backtest trades: before={tot_b} after={tot_a} (delta={tot_a-tot_b})")
    print(f"\nIDENTICAL (count+dates+kpis): {len(identical)}/{len(bench)}")
    print(f"COUNT/DATE CHANGED: {len(count_changed)}")
    for sid, name, bc, ac, be, ae, kd in count_changed:
        de = "" if be == ae else f" earliest {str(be)[:10]}->{str(ae)[:10]}"
        print(f"  sid {sid} [{str(name)[:28]}]: bt {bc}->{ac} (delta {ac-bc}){de}")
    print(f"\nKPI-ONLY CHANGED (count+dates same): {len(kpi_only)}")
    for sid, name, kd in kpi_only[:20]:
        print(f"  sid {sid} [{str(name)[:28]}]: {kd}")
    if missing:
        print(f"\nMISSING after: {missing}")


if __name__ == '__main__':
    main()
