"""Local smoke test for the supervised pool (RORT_RECOMPUTE_SUPERVISOR).
Run from a REAL file (spawn Manager re-imports __main__, which can't be <stdin>).
    RORT_RECOMPUTE_SUPERVISOR=1 RORT_RECOMPUTE_PARALLELISM=2 \
      ../.venv/bin/python _smoke_supervisor.py
"""
import os
import time

os.environ["USE_DB"] = "true"
os.environ["RORT_COMPUTE_REMOTE"] = "0"  # in-process job for the smoke
from dotenv import load_dotenv  # noqa: E402
load_dotenv(".env", override=True)

USER = "19d47e46-f718-49a6-af32-5f5407f5b170"


def main():
    import pack_registry
    pack_registry.scan_and_load_all()
    import recompute_jobs as rj
    jid = rj.submit_recompute_job(USER, [263, 311], "append_recent")
    print(f"submitted {jid[:8]} supervisor={os.getenv('RORT_RECOMPUTE_SUPERVISOR')}",
          flush=True)
    for _ in range(90):
        st = rj.get_job_status(jid)
        if st and st.get("status") in ("completed", "failed", "cancelled"):
            print("STATUS:", st.get("status"), "summary:", st.get("summary"),
                  flush=True)
            print("per_strategy:",
                  [(x['strategy_id'], x['status'])
                   for x in (st.get('per_strategy_results') or [])], flush=True)
            return
        time.sleep(2)
    print("did NOT finish in 180s", flush=True)


if __name__ == '__main__':
    main()
