#!/usr/bin/env python3
"""Per-poll RUNTIME budget gate for the shadow-worker resident lane.

WHY (2026-07-07 incident, Bug_Hunt_Wave1_2026-07-06.md LIVE INCIDENT):
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
  3. FLAG-ON leg (2026-07-07 'Re-arm attempt FAILED' close-out): the #39 widen
     was only ever timed on 1Min/10Sec archetypes; armed fleet-wide it starved
     the pass on SUB-MINUTE TSLA Hi-Fi slots (multi-day 1Sec loads x doubling
     rounds x ~20 slots vs the 600s watchdog — zero polls completed). The
     default sid set now includes 325 (TSLA 30Sec rest_hifi = 1Sec-source
     Hi-Fi; its 780 bars/day mean the widen fires on EVERY same-day bootstrap),
     and every poll is re-timed with RORT_PREP_BAR_COUNT_WARMUP=1 under its own
     HARD budget (default 45s) — a breach fails the gate, exactly the number
     the #39 harness never measured.

USAGE (from clients/KevBot_Toolkit/RoR_Trader):
    PYTHONPATH=src .venv/bin/python src/_validate_poll_runtime.py [SIDS] [BUDGET_S]
      SIDS      comma-separated strategy ids, default "174,268,296,325"
      BUDGET_S  per-poll wall budget in seconds, default RORT_POLL_BUDGET_S or 40
    RORT_POLL_BUDGET_FLAGON_S   flag-ON per-poll budget, default 45
    RORT_VALIDATE_WARMUP_FLAG=0 skips the flag-ON leg (default 1 = run + gate;
      the flag-OFF budget still applies to the flag-OFF legs, whose cost must
      stay a pure passthrough).

READ-ONLY by construction: ResidentEngineManager(dry_run=True) (trade/KPI/heartbeat
writes all no-op), RORT_SHADOW_PERSIST_SNAPSHOT forced 0 (prep never writes the
secondary snapshot), RORT_SHADOW_READ_ONLY_BARS default 1 (no Polygon backfill).

EXIT: 0 = fingerprint printed + every flag-OFF poll within budget + every
          flag-ON poll within the flag-ON budget (unless the leg is skipped).
      1 = budget breach (or no eligible slots). A hard hang dumps all-thread
          stacks at 3× the leg's budget via faulthandler and exits nonzero.
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
            (sys.argv[1] if len(sys.argv) > 1 else "174,268,296,325").split(",")]
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else float(
        os.environ.get("RORT_POLL_BUDGET_S", "40"))
    budget_flag_on = float(os.environ.get("RORT_POLL_BUDGET_FLAGON_S", "45"))

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

    def timed_poll(slot, leg: str, leg_budget: float) -> float:
        # Soft stack dump at budget (diagnose while continuing); hard exit at 3×
        # budget (a wedged poll must fail the gate WITH stacks, not hang CI).
        faulthandler.dump_traceback_later(int(leg_budget), exit=False)
        faulthandler.dump_traceback_later(int(leg_budget * 3), exit=True)
        t0 = time.perf_counter()
        status = None
        try:
            status = mgr.poll(slot).get("status")
        except Exception as e:  # noqa: BLE001
            status = f"EXC:{type(e).__name__}"
        finally:
            faulthandler.cancel_dump_traceback_later()
        wall = time.perf_counter() - t0
        verdict = "PASS" if wall <= leg_budget else "FAIL"
        if wall > leg_budget:
            failures.append((slot.sid, leg, wall))
        print(f"[gate] {verdict} sid={slot.sid} {leg} wall={wall:.2f}s "
              f"budget={leg_budget:.0f}s status={status} "
              f"({slot.symbol} {slot.timeframe})")
        return wall

    for slot in sorted(eligible, key=lambda s: s.sid):
        timed_poll(slot, "poll1_bootstrap", budget)  # cold: warm prep + engine build
        timed_poll(slot, "poll2_warm", budget)       # steady: probe/no_new_bar/incr

    # Flag-ON leg (default ON, RORT_VALIDATE_WARMUP_FLAG=0 to skip): re-bootstrap
    # every slot with the #39 widen armed, under its own HARD budget. This is the
    # 2026-07-07 incident's missing archetype measurement — sub-minute Hi-Fi
    # slots (325-class) MUST bootstrap within budget or arming the flag will
    # starve the shadow pass again. Two polls per slot: cold (pays the widen)
    # then warm (must ride the per-slot widen cache).
    if os.environ.get("RORT_VALIDATE_WARMUP_FLAG", "1") == "1":
        os.environ["RORT_PREP_BAR_COUNT_WARMUP"] = "1"
        print(f"[gate] flag-ON leg (RORT_PREP_BAR_COUNT_WARMUP=1, "
              f"budget={budget_flag_on:.0f}s/poll):")
        for slot in sorted(eligible, key=lambda s: s.sid):
            slot.engine = None                # force a fresh bootstrap prep
            slot.last_processed_ts = None
            timed_poll(slot, "poll1_flag_on_bootstrap", budget_flag_on)
            timed_poll(slot, "poll2_flag_on_warm", budget_flag_on)
        os.environ["RORT_PREP_BAR_COUNT_WARMUP"] = "0"

    if failures:
        print(f"[gate] FAIL: {len(failures)} poll(s) over budget: "
              f"{[(s, leg, round(w, 1)) for s, leg, w in failures]}")
        return 1
    print(f"[gate] PASS: all flag-off polls within {budget:.0f}s + all flag-on "
          f"polls within {budget_flag_on:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
