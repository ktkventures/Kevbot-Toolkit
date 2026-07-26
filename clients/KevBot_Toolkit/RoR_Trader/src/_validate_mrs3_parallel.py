"""M-RS3 Phase 1 validation: seq-vs-parallel byte-identity.

Runs a recompute job over a small strategy subset twice — once with
RORT_RECOMPUTE_PARALLELISM=1 (sequential, runs in-parent) and once with =N
(ProcessPool, spawn workers) — then asserts the PERSISTED backtest-lane trade
signatures + KPIs are identical per strategy. Any diff means an environment
landmine in the pool worker (registry not loaded / DB client) — NOT a logic
change, since the per-strategy compute function is the same. (See
Design_M-RS3_Parallel_Recompute.md.)

Writes the production DB (real DELETE+INSERT recompute). Run only when markets
are closed. Idempotent — re-running converges to the same trades.

Usage (from src/):
    ../.venv/bin/python _validate_mrs3_parallel.py --sids 263,338 \
        --job full_recompute --n 4
"""
import argparse
import os
import sys
import time

os.environ["USE_DB"] = "true"
os.environ.setdefault("RORT_SECONDARY_TF_SNAPSHOT", "1")  # match Railway api
from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env", override=True)

import db  # noqa: E402

USER = "19d47e46-f718-49a6-af32-5f5407f5b170"


def _sig(sid):
    """Persisted backtest-lane signature: ordered (entry, exit, r) tuples."""
    rows = (db.get_admin_client().table('trades')
            .select('entry_fill_ts,exit_fill_ts,r_multiple')
            .eq('strategy_id', sid).like('data_source', 'backtest_%')
            .order('entry_fill_ts').execute().data)
    return [(r.get('entry_fill_ts'), r.get('exit_fill_ts'),
             r.get('r_multiple')) for r in rows]


def _kpis(sid):
    row = (db.get_admin_client().table('strategies').select('kpis')
           .eq('id', sid).eq('user_id', USER).single().execute().data)
    return (row or {}).get('kpis') or {}


def _run_job(sids, job_type, parallelism):
    os.environ['RORT_RECOMPUTE_PARALLELISM'] = str(parallelism)
    # Import AFTER setting the env so there's no doubt about read timing
    # (the worker reads os.getenv at runtime, but be explicit).
    import recompute_jobs
    t0 = time.time()
    job_id = recompute_jobs.submit_recompute_job(USER, sids, job_type)
    while True:
        st = recompute_jobs.get_job_status(job_id)
        if st and st.get('status') in ('completed', 'failed', 'cancelled'):
            break
        time.sleep(2)
    elapsed = time.time() - t0
    return st, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sids', required=True,
                    help='comma-separated strategy ids')
    ap.add_argument('--job', default='full_recompute',
                    choices=['full_recompute', 'append_recent'])
    ap.add_argument('--n', type=int, default=4, help='parallel degree')
    args = ap.parse_args()
    sids = [int(x) for x in args.sids.split(',') if x.strip()]

    import pack_registry
    npacks = pack_registry.scan_and_load_all()
    print(f"[setup] registry packs={npacks}  sids={sids}  job={args.job}")
    if not npacks:
        sys.exit("ABORT: parent registry empty")

    print(f"\n=== PASS A: sequential (N=1) {args.job} ===")
    st1, t1 = _run_job(sids, args.job, 1)
    print(f"  status={st1.get('status')} summary={st1.get('summary')} "
          f"elapsed={t1:.1f}s")
    sig_seq = {s: _sig(s) for s in sids}
    kpi_seq = {s: _kpis(s) for s in sids}

    print(f"\n=== PASS B: parallel (N={args.n}) {args.job} ===")
    st2, t2 = _run_job(sids, args.job, args.n)
    print(f"  status={st2.get('status')} summary={st2.get('summary')} "
          f"elapsed={t2:.1f}s")
    sig_par = {s: _sig(s) for s in sids}
    kpi_par = {s: _kpis(s) for s in sids}

    # The parallel pass must not have failed any strategy.
    failed = (st2.get('summary') or {}).get('total_failed', 0)
    par_statuses = [(r.get('strategy_id'), r.get('status'))
                    for r in (st2.get('per_strategy_results') or [])]

    print("\n=== COMPARISON (seq vs parallel) ===")
    all_ok = True
    for s in sids:
        n_seq, n_par = len(sig_seq[s]), len(sig_par[s])
        trades_match = sig_seq[s] == sig_par[s]
        kpis_match = kpi_seq[s] == kpi_par[s]
        ok = trades_match and kpis_match and n_seq > 0
        all_ok &= ok
        flag = "OK " if ok else "FAIL"
        print(f"  [{flag}] sid {s}: trades seq={n_seq} par={n_par} "
              f"match={trades_match}  kpis_match={kpis_match}")
        if not trades_match:
            # First differing index for debugging
            for i, (a, b) in enumerate(zip(sig_seq[s], sig_par[s])):
                if a != b:
                    print(f"        first diff @{i}: seq={a} par={b}")
                    break
        if n_seq == 0:
            print(f"        WARN sid {s} produced 0 trades — can't prove "
                  f"the pipeline ran (registry/window?)")

    print(f"\n  parallel total_failed={failed}  statuses={par_statuses}")
    if failed:
        all_ok = False
        print("  FAIL: parallel pass reported failed strategies")

    speedup = (t1 / t2) if t2 else 0
    print(f"\n  wall: seq={t1:.1f}s  parallel={t2:.1f}s  speedup={speedup:.2f}×")
    print("\n" + ("✅ BYTE-IDENTICAL — pool machinery sound"
                  if all_ok else "❌ MISMATCH — investigate _pool_init"))
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
