#!/usr/bin/env python3
"""Per-poll RUNTIME budget gate for the shadow-worker resident lane.

WHY (2026-07-07 incident, docs/_active/Bug_Hunt_Wave1_2026-07-06.md LIVE INCIDENT):
PR #39 was gated byte-identical on OUTPUT but never on RUNTIME, and separately a
wrong-tree `railway up` shipped an image whose shadow-worker feature set silently
regressed (lost #33 pass telemetry/watchdog and #34 heartbeats). 90+ minutes of
production debugging chased a code regression that did not exist. This gate makes
both failure classes visible BEFORE a deploy:

  1. It prints the SAME code fingerprint the service logs at boot
     (`shadow_worker._code_fingerprint`) for the tree under test — after deploying,
     the boot log's `code fingerprint=` line must match this value, or the image
     was not built from this tree.
  2. It asserts every single-slot poll — cold bootstrap AND warm steady-state —
     completes within a wall-time budget with the #39 warmup flag OFF. Baseline
     measured at 7386781 (2026-07-07, sids 174/268/296/325/330/336): worst
     bootstrap 12.4s, worst warm 3.7s. Default budget 40s ≈ 3× worst bootstrap.

USAGE (from clients/KevBot_Toolkit/RoR_Trader):
    PYTHONPATH=src .venv/bin/python src/_validate_poll_runtime.py [SIDS] [BUDGET_S]
      SIDS      comma-separated strategy ids, default "174,268,296"
      BUDGET_S  per-poll wall budget in seconds, default RORT_POLL_BUDGET_S or 40
    RORT_VALIDATE_WARMUP_FLAG=1  additionally re-times the polls with
      RORT_PREP_BAR_COUNT_WARMUP=1 and reports (informational; the hard budget
      applies to the flag-OFF legs, whose cost must stay a pure passthrough).

READ-ONLY by construction: ResidentEngineManager(dry_run=True) (trade/KPI/heartbeat
writes all no-op), RORT_SHADOW_PERSIST_SNAPSHOT forced 0 (prep never writes the
secondary snapshot), RORT_SHADOW_READ_ONLY_BARS default 1 (no Polygon backfill).

EXIT: 0 = fingerprint printed + every flag-OFF poll within budget.
      1 = budget breach (or no eligible slots). A hard hang dumps all-thread
          stacks at 3× budget via faulthandler and exits nonzero.
"""
from __future__ import annotations

import faulthandler
import os
import sys
import time

# Force the #39 mechanism flag OFF for the gated legs — the contract under test is
# "flag-off is a zero-cost passthrough". (An optional flag-ON informational leg
# re-enables it explicitly below.)
os.environ["RORT_PREP_BAR_COUNT_WARMUP"] = "0"
os.environ["RORT_SHADOW_PERSIST_SNAPSHOT"] = "0"
os.environ.setdefault("USE_DB", "true")

try:
    from dotenv import load_dotenv
    if os.path.exists("src/.env"):
        load_dotenv("src/.env", override=False)
    elif os.path.exists(".env"):
        load_dotenv(".env", override=False)
except Exception:  # noqa: BLE001
    pass


def main() -> int:
    sids = [int(x) for x in
            (sys.argv[1] if len(sys.argv) > 1 else "174,268,296").split(",")]
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else float(
        os.environ.get("RORT_POLL_BUDGET_S", "40"))

    from shadow_worker import _code_fingerprint, _FINGERPRINT_FILES
    print(f"[gate] code fingerprint={_code_fingerprint()} "
          f"(sha256/12 over {'+'.join(_FINGERPRINT_FILES)}) — the deployed boot "
          f"log's fingerprint line must match this")

    import pack_registry
    npacks = len(pack_registry.scan_and_load_all())
    if not npacks:
        print("[gate] FAIL: pack registry empty")
        return 1

    from shadow_manager import ResidentEngineManager
    mgr = ResidentEngineManager(dry_run=True, shard_sids=set(sids))
    mgr.discover()
    eligible = [s for s in mgr.slots.values() if s.eligible]
    print(f"[gate] budget={budget:.0f}s/poll flag_off "
          f"slots={sorted(mgr.slots)} eligible={[s.sid for s in eligible]}")
    if not eligible:
        print("[gate] FAIL: no eligible slots (bad sids for this DB?)")
        return 1

    failures = []

    def timed_poll(slot, leg: str) -> float:
        # Soft stack dump at budget (diagnose while continuing); hard exit at 3×
        # budget (a wedged poll must fail the gate WITH stacks, not hang CI).
        faulthandler.dump_traceback_later(int(budget), exit=False)
        faulthandler.dump_traceback_later(int(budget * 3), exit=True)
        t0 = time.perf_counter()
        status = None
        try:
            status = mgr.poll(slot).get("status")
        except Exception as e:  # noqa: BLE001
            status = f"EXC:{type(e).__name__}"
        finally:
            faulthandler.cancel_dump_traceback_later()
        wall = time.perf_counter() - t0
        verdict = "PASS" if wall <= budget else "FAIL"
        if wall > budget:
            failures.append((slot.sid, leg, wall))
        print(f"[gate] {verdict} sid={slot.sid} {leg} wall={wall:.2f}s "
              f"status={status} ({slot.symbol} {slot.timeframe})")
        return wall

    for slot in sorted(eligible, key=lambda s: s.sid):
        timed_poll(slot, "poll1_bootstrap")   # cold: warm-window prep + engine build
        timed_poll(slot, "poll2_warm")        # steady-state: probe/no_new_bar/incremental

    if os.environ.get("RORT_VALIDATE_WARMUP_FLAG", "0") == "1":
        os.environ["RORT_PREP_BAR_COUNT_WARMUP"] = "1"
        print("[gate] informational flag-ON re-time (no budget applied):")
        for slot in sorted(eligible, key=lambda s: s.sid):
            slot.engine = None                # force a fresh bootstrap prep
            slot.last_processed_ts = None
            timed_poll(slot, "poll_flag_on")
        os.environ["RORT_PREP_BAR_COUNT_WARMUP"] = "0"
        failures[:] = [f for f in failures if f[1] != "poll_flag_on"]

    if failures:
        print(f"[gate] FAIL: {len(failures)} poll(s) over {budget:.0f}s budget: "
              f"{[(s, leg, round(w, 1)) for s, leg, w in failures]}")
        return 1
    print(f"[gate] PASS: all flag-off polls within {budget:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
