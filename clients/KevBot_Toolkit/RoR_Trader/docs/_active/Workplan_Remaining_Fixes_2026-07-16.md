# Workplan — Remaining Fixes (2026-07-16)

**Agreed Kevin ↔ Claude 07-16 ~16:15Z.** Implement everything below TODAY, flag-gated +
test-gated, batch-push live, verify nothing breaks, get early divergence reads. **PAUSE
before M-RS5** (that's the next big rock, not today).

Already DONE + ARMED (context): P0+P1-routing (`RORT_CANONICAL_PRIMARY_CLOSE=1`),
P1-grace (`RORT_GRACE_FINAL_CLOSE_ELIGIBLE=1`) — both validated live at the 07-16 open
(canary-silence check passed; 269=89%/265=82%/333=100% on the settled morning window).
Harness scoring fixes done. Today's RTH window is deploy-gap-free (all redeploys were
last night; only blemish = single 20.8s stall at 15:16Z — analysis flags clustering there).

---

## 1. P1-hot-reload (Brandon P1) — hot-added/edited strategies must match a clean boot

**Purpose:** `worker.py`'s `db_hot_reload` rebuilds an added/edited strategy with a FLAT
7-day warmup and skips `finalize_shadow_engines`. Result: a strategy behaves differently
depending on WHEN it was loaded (boot vs hot-add vs UI-edit) — e.g. a newly-added 1Day
gate can have a ~5-bar shadow (or none) until restart. A hidden per-strategy divergence
source unrelated to the strategy itself.

**Design (flag `RORT_HOTRELOAD_BOOT_PARITY`, default OFF):**
- Hot-reload path uses the SAME TF-scaled warmup as boot (`ralph_engine` prod warmup
  ~1872-2034 region) instead of the flat 7-day load.
- After (re)instantiating monitors, call `finalize_shadow_engines()` + seed the mtf
  buffer the same way boot does, so a newly-introduced secondary gate gets a real shadow.

**Acceptance (no Brandon test exists — write ours):** hot-add a strategy with a coarse
(1Day) gate via the reload path; its shadow topology + warmup depth must equal a clean
boot of the identical config. Flag-OFF: legacy path byte-identical.

**Risk:** touches the reload path only (boot untouched); flag-OFF = current behavior.
Medium care: reload runs while the engine is live — keep the added work bounded (single
strategy, not fleet-wide; do NOT reintroduce a resync-style fleet replay).

## 2. P2-healthcheck (Brandon P2) — bounded boot grace + (later) feed freshness

**Purpose:** the current Docker healthcheck catches a STALLED engine (heartbeat ages out)
but not (a) an engine that NEVER starts — the continuously-refreshed MANAGER heartbeat
grants unlimited "boot grace" — or (b) a silently-dead data feed (engine pulse ticks on a
timer regardless of tick arrival).

**Design (flag `RORT_HEALTHCHECK_BOUNDED_BOOT`, default OFF):**
- Worker startup writes a **write-once BOOT MARKER** file (unconditional — inert by itself).
- `engine_health.check` flag-ON: engine heartbeat absent → healthy ONLY while
  `now - boot_marker_mtime < RORT_ENGINE_BOOT_DEADLINE_S` (default 600s, well beyond any
  real warmup). Marker absent or deadline exceeded → UNHEALTHY. Flag-OFF: legacy
  (manager-fallback) behavior byte-identical.
- Feed-freshness (part b) is a FOLLOW-UP (needs a last-tick timestamp plumbed into the
  heartbeat; session-aware to avoid weekend false-positives) — not in today's scope.

**Acceptance:** Brandon's `test_engine_never_started_boot_grace_is_bounded` passes
flag-ON, still fails flag-OFF. Companion tests: real boot sequence (marker written →
grace within deadline → healthy; engine appears → normal path), marker-absent → unhealthy.

**⚠️ Risk — the reason this waited for daylight:** the healthcheck GATES container
restarts; a bad deadline/marker = restart LOOP. Mitigations: flag default OFF; deadline
600s (≫ warmup) ; Docker start_period+retries absorb the first moments; before arming,
verify in logs the marker is written at boot; arm with active monitoring and flip-off on
any unexpected restart.

## 3. Smoke-canary fix (tooling — unblocks fresh recomputes)

**Purpose:** `local_update.py`'s safety gate proves the environment via canary 321 —
but 321 is itself live-trading, so its recent trades never bit-match a clean recompute
during/near sessions (aborted the 07-15 forward-check recompute; correct abort, wrong
canary). Blocks `--mode all` when we most want it.

**Design:** teach the smoke gate a **fixed historical verification window** (e.g. compare
lane-vs-recompute only for trades older than 24h / before today's session) so live-append
freshness can't race it — the gate still proves registry/packs/flags correctness
(the 0-trades trap + the 99.8%-overlap class both still caught). No prod-write risk:
the gate change only makes the CHECK window historical; writes unchanged.

**Acceptance:** run `--mode all --strategies <canary>` during RTH → smoke PASSES on the
historical window; deliberately break the env (unset RORT flags) → still ABORTS.

## 4. 329 under-firing (investigation, no code yet)

0% on the settled morning window (0 live vs 2 bt entries) — NOT in the P0/P1 blast radius
(329 isn't 1Min+ default-model), so pre-existing. 329 has a history (1M-gate flag-parity
repair 07-13). Step 1: pull its 2 missed bt entries + check gate state / trigger telemetry
at those timestamps; classify (gate divergence vs dispatch vs data). Feeds the fix list if
it's systematic.

## 5. Later / explicitly NOT today
- **Harness → provider-event replay** (Brandon rec #6): make `/replay-check` consume raw
  WS events through the real builders so it can SEE construction-class bugs. Bigger build.
- **M-RS5 (PAUSE POINT):** structural cross-TF unification ("one bar store, one engine";
  `Design_MRS5_Resident_Window.md`). The durable fix for the 120s-refresher band-aid.
  Everything above lands first; M-RS5 gets its own planning pass.

---

## Today's execution plan
1. ✅ EOD **replay check** running now (Five + canaries, 13:30→16:30Z, corrected splitter);
   analysis will flag divergence clustering at the 15:16Z stall (no deploy gaps in window).
2. Implement **#1 hot-reload → #2 P2 → #3 smoke-canary** (risk-ascending pushes batched):
   each flag-gated, flag-OFF==baseline (stash-proof), acceptance tests green, existing
   suites at baseline (3 documented pre-existing fails only).
3. **ONE batched push** (single Worker redeploy — note the deploy time for analysis), verify
   deploy SUCCESS + healthy, then **ONE combined arm** (`--set` both flags together = one
   more redeploy). Monitor: alert-lag, stall watchdogs, canary firing.
4. **#4 329 investigation** while monitoring settles.
5. When the replay check lands: read it with the stall-window caveat → whatever it surfaces
   joins this list.
6. **Early reads** late-RTH/EH: canaries still firing, dashboard-live on the post-arm
   window, no new stalls → report.
