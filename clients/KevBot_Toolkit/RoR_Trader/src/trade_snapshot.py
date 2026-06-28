"""Trade-level snapshot + diff — exact fidelity audits across recompute runs.

The per-strategy count/date/KPI snapshot can tell you a strategy *changed*, but
not *which* trades changed. This captures every backtest-lane trade (keyed by
entry|exit fill timestamp, with r/prices/reason), so two snapshots diff to the
exact added / removed / changed trades — pinpointing data-revision vs an engine
change, and giving the bit-perfect-vs-before confirmation a count diff can't.

Usage (from src/):
  # before a fleet run:
  ../.venv/bin/python trade_snapshot.py snapshot --all --label before
  # after:
  ../.venv/bin/python trade_snapshot.py snapshot --all --label after
  # compare:
  ../.venv/bin/python trade_snapshot.py diff --before <f1.json> --after <f2.json>
  # narrow set:
  ../.venv/bin/python trade_snapshot.py snapshot --sids 293,304 --label x
"""
import argparse
import json
import os
from datetime import datetime, timezone

os.environ["USE_DB"] = "true"
from dotenv import load_dotenv  # noqa: E402
load_dotenv(".env", override=True)
import db  # noqa: E402

USER = "19d47e46-f718-49a6-af32-5f5407f5b170"
# Fields that define a trade's identity + comparable value. entry|exit fill ts
# is the key; the rest detect a "same trade, different fill/r" change.
KEY_FIELDS = ("entry_fill_ts", "exit_fill_ts")
VAL_FIELDS = ("r_multiple", "entry_price", "exit_price", "exit_reason",
              "dollar_pnl", "direction", "exec_type",
              "entry_trigger_ts", "exit_trigger_ts")
PAGE = 1000


def _all_sids(c):
    rows = (c.table('strategies').select('id').eq('user_id', USER)
            .execute().data)
    return sorted(r['id'] for r in rows)


def _fetch_trades(c, sid):
    out, off = [], 0
    cols = ",".join(KEY_FIELDS + VAL_FIELDS)
    while True:
        page = (c.table('trades').select(cols)
                .eq('strategy_id', sid).like('data_source', 'backtest_%')
                .order('entry_fill_ts').range(off, off + PAGE - 1)
                .execute().data)
        if not page:
            break
        out.extend(page)
        if len(page) < PAGE:
            break
        off += PAGE
    return out


def snapshot(sids, label):
    c = db.get_admin_client()
    snap = {}
    for i, sid in enumerate(sids):
        trades = _fetch_trades(c, sid)
        keyed = {}
        for t in trades:
            k = "|".join(str(t.get(f)) for f in KEY_FIELDS)
            keyed[k] = {f: t.get(f) for f in VAL_FIELDS}
        snap[str(sid)] = {"n": len(trades), "trades": keyed}
        if i % 10 == 0:
            print(f"  [{i}/{len(sids)}] sid {sid}: {len(trades)} trades",
                  flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = f"/tmp/trade_snapshot_{label}_{stamp}.json"
    json.dump({"label": label, "taken_at": stamp, "user": USER,
               "strategies": snap}, open(path, "w"), default=str)
    total = sum(v["n"] for v in snap.values())
    print(f"\nSAVED {path}\n  {len(sids)} strategies, {total} trades", flush=True)
    return path


def diff(before_path, after_path):
    b = json.load(open(before_path))["strategies"]
    a = json.load(open(after_path))["strategies"]
    sids = sorted(set(b) | set(a), key=int)
    tot_add = tot_rem = tot_chg = 0
    clean = 0
    print(f"=== TRADE-LEVEL DIFF  before={before_path}  after={after_path} ===")
    for sid in sids:
        bt = (b.get(sid) or {}).get("trades", {})
        at = (a.get(sid) or {}).get("trades", {})
        bk, ak = set(bt), set(at)
        added = ak - bk
        removed = bk - ak
        changed = [k for k in (bk & ak) if bt[k] != at[k]]
        if not (added or removed or changed):
            clean += 1
            continue
        tot_add += len(added)
        tot_rem += len(removed)
        tot_chg += len(changed)
        print(f"\nsid {sid}: +{len(added)} added  -{len(removed)} removed  "
              f"~{len(changed)} changed  (n {len(bt)}->{len(at)})")
        for k in sorted(added)[:3]:
            print(f"    + {k}  r={at[k].get('r_multiple')}")
        for k in sorted(removed)[:3]:
            print(f"    - {k}  r={bt[k].get('r_multiple')}")
        for k in sorted(changed)[:3]:
            d = {f: (bt[k].get(f), at[k].get(f)) for f in VAL_FIELDS
                 if bt[k].get(f) != at[k].get(f)}
            print(f"    ~ {k}  {d}")
    print(f"\n=== SUMMARY: {clean}/{len(sids)} trade-for-trade IDENTICAL | "
          f"added={tot_add} removed={tot_rem} changed={tot_chg} ===")
    if tot_add or tot_rem:
        print("  (added/removed at stable date edges = data revision; "
              "spread across history = trigger-threshold flips from revised bars)")
    if tot_chg and not (tot_add or tot_rem):
        print("  (changed-only = same trades, Hi-Fi refined fills)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--sids")
    s.add_argument("--all", action="store_true")
    s.add_argument("--label", required=True)
    d = sub.add_parser("diff")
    d.add_argument("--before", required=True)
    d.add_argument("--after", required=True)
    args = ap.parse_args()
    if args.cmd == "snapshot":
        c = db.get_admin_client()
        sids = _all_sids(c) if args.all else \
            [int(x) for x in args.sids.split(",") if x.strip()]
        snapshot(sids, args.label)
    else:
        diff(args.before, args.after)


if __name__ == "__main__":
    main()
