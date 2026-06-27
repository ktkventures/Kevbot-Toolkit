"""M-RS3 Phase 1 — FAST machinery proof (read-only, windowed).

The pool risk is entirely in _pool_init (pack registry + Supabase admin client +
admin user-context in a fresh spawn worker). Hi-Fi / persistence / algo-lane are
deterministic, pool-insensitive code. So we don't need a full production
recompute to prove the worker ENVIRONMENT is correct — we need a deterministic
engine replay run identically in the parent and in a spawn worker.

This runs `services.get_strategy_trades` on a SMALL (~N-day) window — read-only,
no DB writes, no Hi-Fi cold-seed of the full year — in the parent and in a spawn
ProcessPoolExecutor worker (initialized by recompute_jobs._pool_init), then
asserts the serialized trades are byte-identical. A worker whose registry didn't
load (the silent-0 trap) or whose admin client/context is wrong would produce
different (or zero) trades — which this catches in minutes.

Usage (from src/):
    ../.venv/bin/python _validate_mrs3_fast.py --sids 263,267 --days 12
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

os.environ["USE_DB"] = "true"
os.environ.setdefault("RORT_SECONDARY_TF_SNAPSHOT", "1")  # match Railway api
from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env", override=True)

import db  # noqa: E402

USER = "19d47e46-f718-49a6-af32-5f5407f5b170"


def _windowed_replay(sid, user_id, start_iso):
    """Deterministic engine replay from a FIXED window start, returned as the
    canonical serialized trade list. Top-level + picklable so it runs in a spawn
    worker. Read-only (no persist). The window start must be passed in (frozen
    once by the caller) — computing `now()` independently in parent vs worker
    would drift the leading edge and shift a boundary trade (a validator bug,
    not a pool bug)."""
    from db import get_strategy_by_id_admin
    import services
    from api.services.forward_test_service import _serialize_trades
    strat = get_strategy_by_id_admin(sid, user_id)
    if strat is None:
        return {'sid': sid, 'error': 'not found'}
    # Shrink the visible window so the replay is seconds, not ~1yr. The pool
    # machinery is window-size-independent; a small window proves it just as
    # well and far faster.
    wstrat = {**strat, 'backtest_start_date': start_iso}
    wstrat.pop('lookback_mode', None)  # don't let a lookback override the anchor
    t0 = time.time()
    trades = services.get_strategy_trades(wstrat)
    stored = _serialize_trades(trades) if trades is not None and len(trades) else []
    return {'sid': sid, 'n': len(stored), 'stored': stored,
            'elapsed': round(time.time() - t0, 1)}


def _pool_replay(sids, user_id, start_iso, n):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import recompute_jobs as rj
    out = {}
    with ProcessPoolExecutor(max_workers=n,
                             mp_context=mp.get_context('spawn'),
                             initializer=rj._pool_init,
                             initargs=(user_id,)) as ex:
        futs = {ex.submit(_windowed_replay, s, user_id, start_iso): s for s in sids}
        for f in as_completed(futs):
            r = f.result()
            out[r['sid']] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sids', required=True)
    ap.add_argument('--days', type=int, default=12)
    ap.add_argument('--n', type=int, default=2)
    args = ap.parse_args()
    sids = [int(x) for x in args.sids.split(',') if x.strip()]

    import pack_registry
    npacks = len(pack_registry.scan_and_load_all())
    db.set_admin_user_context(USER)
    # Freeze the window start ONCE so parent and worker replay the EXACT same
    # window (else now()-drift between the two runs shifts a boundary trade).
    start_iso = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
    print(f"[setup] parent registry packs={npacks} sids={sids} "
          f"days={args.days} n={args.n} window_start={start_iso}", flush=True)

    print("=== PARENT replays (in-process) ===", flush=True)
    parent = {}
    for s in sids:
        r = _windowed_replay(s, USER, start_iso)
        parent[s] = r
        print(f"  parent sid {s}: n={r.get('n')} elapsed={r.get('elapsed')}s",
              flush=True)

    print(f"=== WORKER replays (spawn pool N={args.n}) ===", flush=True)
    tw = time.time()
    worker = _pool_replay(sids, USER, start_iso, args.n)
    for s in sids:
        r = worker.get(s, {})
        print(f"  worker sid {s}: n={r.get('n')} elapsed={r.get('elapsed')}s",
              flush=True)
    print(f"  (worker wall incl spawn startup: {time.time()-tw:.1f}s)",
          flush=True)

    print("=== COMPARISON (parent vs spawn-worker) ===", flush=True)
    import json as _json

    # `confluence_records` is a diagnostic list built from set/dict iteration,
    # so its ELEMENT ORDER is per-process PYTHONHASHSEED-dependent — it already
    # varies between ANY two Python processes (cron recompute vs manual refresh),
    # independent of parallelism. Normalize its order before comparing so the
    # gate measures FIDELITY (entry/exit/price/r/...), not hash-seed noise.
    ORDER_UNSTABLE = {'confluence_records'}

    def _norm(rec):
        r = dict(rec)
        for k in ORDER_UNSTABLE:
            v = r.get(k)
            if isinstance(v, list):
                r[k] = sorted(v, key=lambda x: str(x))
        return r

    all_ok = True
    real_diff_keys = set()
    for s in sids:
        ps, ws = parent[s].get('stored'), worker.get(s, {}).get('stored')
        try:
            _json.dump(ps, open(f"/tmp/mrs3_{s}_parent.json", "w"))
            _json.dump(ws, open(f"/tmp/mrs3_{s}_worker.json", "w"))
        except Exception:
            pass
        nz = (parent[s].get('n') or 0) > 0
        same_len = len(ps or []) == len(ws or [])
        order_only = 0
        real = 0
        for a, b in zip(ps or [], ws or []):
            if a == b:
                continue
            if _norm(a) == _norm(b):
                order_only += 1
            else:
                real += 1
                real_diff_keys.update(k for k in (set(a) | set(b))
                                      if a.get(k) != b.get(k))
        ok = same_len and nz and real == 0
        all_ok &= ok
        print(f"  [{'OK ' if ok else 'FAIL'}] sid {s}: parent_n={len(ps or [])} "
              f"worker_n={len(ws or [])} real_diffs={real} "
              f"confluence_order_only_diffs={order_only}")
        if not nz:
            print(f"        WARN sid {s}: 0 trades in {args.days}d window")
    if real_diff_keys:
        print(f"\n  REAL DIFFERING KEYS: {sorted(real_diff_keys)}")
    print("\n" + ("✅ BYTE-IDENTICAL on fidelity fields (confluence_records "
                  "order normalized — pre-existing hash-seed artifact)"
                  if all_ok else "❌ MISMATCH — investigate _pool_init"))
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
