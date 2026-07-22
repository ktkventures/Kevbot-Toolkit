# Design note — Provisional Backtest Tail (near-real-time backtest lane)

**Status: DIRECTION AGREED 2026-07-22 (Kevin), NOT scheduled. Build AFTER M-RS5a promotes
fleet-wide (the resident frame is the enabler that makes this affordable).**

## What Kevin wants (the end state)
The **backtest-model lane** updates near-real-time on **unsettled REST bars** — the same
freshness rhythm as the live-bar path (REST re-fetch/revision for up to ~15–45 min until
settled) — with trades in the unsettled window **tagged provisional**, converging to
settled truth as bars pass the horizon. NOT WebSocket data; REST only, backtest
construction, just without waiting the flat 15 minutes.

Scope note (Kevin, 2026-07-22): this is explicitly about the BACKTEST model. The algo
lane (`cache_cache_locked`, the decision-time replica) is a separate concern — likely
already near-real-time-able since it records decision-time data; VERIFY its actual write
cadence before touching it (open question #1).

## Why the 15-minute settle lag exists today (and why that reason expired)
`shadow_manager` advances only on bars settled ≥ LAG_MINUTES=15. That was an ECONOMICS
decision under the old full-prep engine (re-truing an unsettled tail per poll for the
fleet was exactly the starvation class of 2026-07-14), plus a lane-stability preference
(no trades appearing/vanishing as bars revise). It was never a fidelity judgment.
**M-RS5a's resident frame changes the economics**: re-truing a short tail per poll is
cheap once the window is resident.

## What it buys
- **Append-New disparity class retired by obsolescence**: a lane continuously current to
  ~now never needs intraday append (the historical under-production class stops
  mattering rather than getting patched again).
- **Intraday parity reads get honest**: alerts pair within ~a poll instead of sitting TBD
  15+ min; phantom/missed intraday becomes "is logic diverging?" not "has data settled?".
- **Final accuracy unchanged**: settled recompute always overwrites the provisional tail;
  the settled lane remains the spec.

## Semantics to lock at build time
1. **Tagging**: provisional trades carry an explicit marker (column or lane variant);
   EXCLUDED from KPIs until settled; Strategy Health pairing prefers settled rows on
   overlap and may pair provisionally with a distinct visual state.
2. **Replacement unit**: the whole unsettled window is delete-and-replaced per re-true
   (extend the existing delete-provisional-rows-then-advance machinery in
   `shadow_manager.poll` — this is an extension, not an invention). Never upsert
   (retracted-bar lesson).
3. **Tail depth**: settle threshold (15m) vs bar_cache revision horizon (45m) — decide
   which bound defines "provisional"; deeper = more churn, safer convergence.
4. **Trades-table safety**: one careful pass over the unique-index/lane rules so
   provisional and settled rows cannot collide (see the resolved
   trades_unique_index_lane_collision incident).
5. **Never touches the live/alert path.**

## Open questions
1. Algo lane (`cache_cache_locked`) write cadence today — measure, don't assume.
2. Write volume bound per poll (fleet × tail depth) vs DB health (EGRESS-sensitive).
3. UI: how provisional trades render (greyed? dotted?) and whether the paired-% shows a
   provisional variant alongside the settled number.

## Sequencing
M-RS5a fleet-wide promotion (SIDS all @POLL_S=60 → then 5s) ➜ THEN this design, built on
the resident frame. Cross-refs: `Impl_MRS5a_Resident_Window.md`,
`Roadmap_Trading_At_Scale.md`, Divergence_Hunt_Log 07-22 brief (append/TBD context).
