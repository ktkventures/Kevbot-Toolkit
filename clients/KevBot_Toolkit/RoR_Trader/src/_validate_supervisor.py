"""Validate the supervised pool's reap paths (RORT_RECOMPUTE_SUPERVISOR).

Injects a wedged (or crashed) worker via RORT_TEST_HANG_SID and confirms the
supervisor reaps it after the stall window while the OTHER strategies complete
normally — i.e. one bad strategy no longer wedges the batch, and there's no
wall-clock cap (the reap is progress-based).

Usage (from src/):
    RORT_RECOMPUTE_SUPERVISOR=1 RORT_RECOMPUTE_PARALLELISM=3 \
    RORT_RECOMPUTE_STALL_SECONDS=25 RORT_TEST_HANG_SID=311 \
    RORT_TEST_HANG_MODE=wedge \
      ../.venv/bin/python _validate_supervisor.py
"""
import os
import time

os.environ["USE_DB"] = "true"
os.environ["RORT_COMPUTE_REMOTE"] = "0"  # in-process job
from dotenv import load_dotenv  # noqa: E402
load_dotenv(".env", override=True)

USER = "19d47e46-f718-49a6-af32-5f5407f5b170"


def main():
    hang = os.getenv("RORT_TEST_HANG_SID", "311")
    stall = os.getenv("RORT_RECOMPUTE_STALL_SECONDS", "25")
    mode = os.getenv("RORT_TEST_HANG_MODE", "wedge")
    import pack_registry
    pack_registry.scan_and_load_all()
    import recompute_jobs as rj
    sids = [263, int(hang), 267]
    t0 = time.time()
    jid = rj.submit_recompute_job(USER, sids, "append_recent")
    print(f"submitted {jid[:8]} | hang_sid={hang} mode={mode} stall={stall}s "
          f"sids={sids}", flush=True)
    for _ in range(120):
        st = rj.get_job_status(jid)
        if st and st.get("status") in ("completed", "failed", "cancelled"):
            wall = time.time() - t0
            print(f"\nFINISHED in {wall:.0f}s status={st.get('status')}",
                  flush=True)
            rows = {x['strategy_id']: x for x in
                    (st.get('per_strategy_results') or [])}
            for sid in sids:
                r = rows.get(sid, {})
                err = (r.get('backtest_lane') or {}).get('reason') or r.get('error')
                print(f"  sid {sid}: status={r.get('status')} "
                      f"{('['+str(err)+']') if r.get('status')=='error' else ''}",
                      flush=True)
            # Verdict
            hang_row = rows.get(int(hang), {})
            others_ok = all(rows.get(s, {}).get('status') not in (None, 'error')
                            for s in sids if s != int(hang))
            reaped = hang_row.get('status') == 'error'
            no_wedge = wall < 90
            ok = reaped and others_ok and no_wedge
            print("\n" + ("✅ REAP VALIDATED — wedged sid reaped, others "
                          "completed, batch did NOT hang"
                          if ok else
                          f"❌ FAIL reaped={reaped} others_ok={others_ok} "
                          f"no_wedge={no_wedge}"), flush=True)
            return 0 if ok else 1
        time.sleep(2)
    print("❌ job did NOT finish in 240s — supervisor itself may be stuck",
          flush=True)
    return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
