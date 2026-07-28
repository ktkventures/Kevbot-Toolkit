# Divergence Hunt Log (bug-hunt aggressive, started 2026-07-09)

Standing record. Metric = Strategy Health `combined_pct` @5s (screen-faithful via
`api.routers.strategy_health.get_strategy_health` offline). Goal: 21 W2-6 strategies each
≥95% @5s (or ≥95%@10s w/ note), OR root-caused + replay-validated fix. Ledger cross-ref:
`Bug_Hunt_Wave1_2026-07-06.md`.

## 🌙 MORNING BRIEF — nightly bug-hunt 2026-07-27→28 (Mode 3)

**Headline: MON 07-27 was NOT a regression — the ~10pt "drop" that cost 07-27 was
100% the settle-confound. Re-read post-settle, the fleet is 92.3%@10s vs a
92.7/94.1/93.0 pre-flag band (0.4pt below band min = flat within noise). #125's
machinery ran end-to-end and the suite is 18/18 (the recurring SPY/10Sec false-red
is gone) — BUT tonight was a no-revision night, so #125 is NOT yet adversarially
proven. #121 is NO-SIGNAL for a 2nd day (NVDA silent in BOTH lanes). ARMED: NOTHING
(no provable candidate — rail 2).**

### Phase 0 — nightly recompute: CLEAN ✅
Fleet `data_refreshed_at` advanced 00:25:00Z → 00:54:14Z (23/23 active strategies),
matching the 00:20Z `RORT_NIGHTLY_RECOMPUTE` fire. `bt_current_through` = 07-27T19:59:50Z,
heartbeat 04:28Z. No partial/aborted signature. (Nightly runs via batch-worker, not
`update_jobs` — that table's newest row is 07-21, which is expected, not a miss.)

### ⭐ THE 07-27 "REGRESSION" WAS THE SETTLE-CONFOUND — closed
All four days are now settled, so the comparison is finally apples-to-apples:

| combined_pct @10s | WED 07-22 | THU 07-23 | FRI 07-24 | **MON 07-27** |
|---|---|---|---|---|
| FLEET | 92.7 | 94.1 | 93.0 | **92.3** |

0.4pt below the band minimum — flat. The ~10pt gap seen on 07-27 was the intraday
APPEND lane being compared against prior days' nightly SETTLE, exactly as the skill's
settle-confound section predicts. **No fleet regression happened.** Goal-1 (gated
≥90%@10s) = **6/12** with data, unchanged from the 07-23 read; @5s also 6/12.

Per-sid @10s (gated): 136=100.0 · 314=93.8 · 325=100.0 · 330=100.0 · 338=99.1 ·
342=100.0 ✅ | 267=77.2 · 271=81.8 · 331=77.4 · 310=54.5 · 194=50.0 · 309=0.0 ✗.
Non-gated (deprioritized): 263=98.1 · 321=98.1 · 269=97.4. NVDA 341/343/344/345 = no
data (both lanes silent — see #121).

Two sids tripped the below-band check; **neither is a real regression**:
- **314 (93.8 vs 97.5)** — its "band" was a single day (07-22, N=81); 07-27 ran N=130.
  Larger sample, still passes Goal-1. Variance, not regression.
- **267 (77.2 vs 80.5)** — real but modest, and the known class (below).

### Bugs found + CLASSIFIED
| sid | @10s | three-lane verdict | class | evidence |
|-----|------|--------------------|-------|----------|
| 267 | 77.2 | **algo≈bt, live≠both** (bt↔algo 220/222) | **PLUMBING / live-process-state** | missed spread across EVERY hour (13:2 14:6 15:2 16:4 17:8 18:6 19:4) — diffuse, not deploy- or stall-clustered. Same signature as 07-23. Missed 13→32 as volume grew 191→222 edges. |
| 331 | 77.4 | **algo≈bt, live≠both** (all 8 phantoms algo-absent) | **PLUMBING / live-only** *(new)* | 8 live-only edges; 2 of 3 missed were taken by algo ⇒ live path. ⚠️ **duplicate-dispatch smell:** phantom PAIRS 8–11s apart (14:01:00+14:01:08, 15:32:30+15:32:41) and off-boundary fills (:08/:41) on a **30Sec** strategy, where every edge should land on :00/:30. |
| 314 | 93.8 | algo≈bt (124/124), live +6/−2 | PLUMBING (mild) | passes Goal-1; no action |

**Common thread (unchanged):** every laggard is `algo ≈ backtest, live ≠ both` — the
offline engine is faithful, the residual is accumulated live-process state. Structural
cure = canonical live serving / state re-true (M-RS5), not per-sid patches.

### 🔎 Kevin's two questions — answered
**Q1: did SPY revision-drift recur? → NO.** 64 independent cache-vs-fresh-Polygon
comparisons (SPY 10Sec/1Min + TSLA 1Sec/1Min/10Sec, both sessions) across 07-17, 07-20,
07-21, 07-22, 07-23, 07-24, 07-27: **OHLCVdiffs=0, rowMM=0 everywhere.** The parity suite
independently returned **18/18**, with both SPY/10Sec items — last night's 2 FAILs —
clean at first compare.

**Q2: did self-heal fire? → NO, and it never needed to.** No item hit the drift signature,
so layer 3 never engaged (no `self-healed revision drift` annotation anywhere). Self-heal
remains **unexercised in production**; it is proven only by unit test
(`_test_parity_self_heal.py`, 8/8 — including the critical property that a diff which
SURVIVES the retrue still FAILs, so it cannot paper over a real bug).

**What DID demonstrably run — #125 layers 1 and 2 (DB-effect proof, since logs had rolled off):**
- **Layer 1 (pointer, `MIN_TDAYS=4`)** — suite tested settled day **07-21** (T+4 trading
  days). Under legacy it would have tested **07-24** (T+1) — deep in the churn zone.
- **Layer 2 (window, `RETRUE_WINDOW=5`)** — `bar_cache.revised_at` shows 07-21/22/23/24/27
  all re-trued at **00:21:33–00:24:00Z tonight**, across NVDA/SPY/TSLA, sequenced exactly
  as `run_nightly_settled_retrue` loops. **07-20 was NOT touched** (still 07-27T22:20Z) —
  the window boundary is precisely correct, not over-broad.

**⚠️ But the honest caveat — tonight did not TEST #125.** Counterfactual: 07-17 and 07-20
sit *outside* the retrue window and hadn't been trued in days; if revisions were still
arriving they should have drifted. **Both are perfectly clean too.** So no revision landed
anywhere — the green proves the machinery executed correctly, not that it prevented
anything. **#125 = EXECUTED CORRECTLY, NO-SIGNAL on efficacy.** Keep armed; it needs a
night where a revision actually arrives (prior rate: 3 in 4 days, so likely soon).

### #121 (NVDA MTF PB-prev-epoch) — NO-SIGNAL, 2nd consecutive session
341/343/344/345 each: **0 backtest trades AND 0 alerts** on 07-27 (vs 5/5/5/1 trades and
8/8/8/2 alerts on 07-24). Both lanes silent ⇒ a genuinely quiet day, not a miss. Still
needs an NVDA RVOL-spike day. Correctly held in Review.

### 🆕 Engine-snapshot freeze — now FLEET-WIDE, and the write path is identified
Board has this as "'0 healthy' is a dead alarm … 22 of 24 since ~07-21". Tonight it is
**23/23 active strategies stale** (339 has *no* snapshot at all). New precision: every
snapshot is frozen at **07-21 22:21:55Z → 22:40:09Z, sequential by sid** — a single ~19-min
sweep, after which nothing has written one for **6.3 days**.
- **Write path = `data_worker_engine.flush_snapshot()` (line 776), the STREAMING loop** —
  *not* the recompute path. **This exonerates the nightly recompute and
  `RORT_UPDATE_ALL_SKIP_ALGO`** (that flag gates `recompute_and_persist_algo_trades`, a
  different lane) — I checked that hypothesis and it is wrong.
- Two silent-freeze candidates to check next, both fail without an error:
  `if not state.snapshot_dirty or state.snapshot_b64 is None: return True` (returns
  **success** when nothing is dirty), and `if not circuit.allow()` (open DB circuit breaker
  → `record_circuit_skip`, no raise).
- Not chased further tonight (E-lane board item, and no provable fix available).

### 🆕 Local parity runs test the WRONG day off-UTC (measurement hazard, not a prod bug)
`fidelity_parity_suite._settled_test_day()` walks from `dt.date.today()` (**local**) while
`settle_sweeper._settled_pointer_day()` / `run_nightly_settled_retrue()` use
`datetime.now(timezone.utc).date()` (**UTC**). On this box (America/Denver) a suite run
between 18:00–24:00 MDT picks a day **one earlier** than the retrue covered — tonight that
was 07-20 vs prod's 07-21. Railway containers are UTC so prod is unaffected, but #125's own
docstring claims the shared walk means the two "can never target different days" — that
claim is false off-UTC. **All measurements in this brief were taken with `TZ=UTC` forced**
so they mirror prod. Cheap fix: have the suite pass a UTC date.

### ARMED last night: **NOTHING** (0/3 flag budget — live behavior untouched)
Correct per rail 2: no fix was built tonight because no laggard is in a provable class.
267/331/310 are live-process-state and coarse-gate (4h) classes the replay harness cannot
reconstruct → diagnose + hold, never auto-arm. No code was pushed; no flags flipped.

### Held for Kevin
1. **#125 — keep armed, needs one revision-bearing night.** Machinery verified; efficacy
   untested. Revert if ever needed: `railway variables --set "RORT_NIGHTLY_SETTLE_RETRUE_WINDOW=1" --service batch-worker`
   (and `RORT_PARITY_SETTLED_MIN_TDAYS=0`, `RORT_PARITY_SELF_HEAL=0`).
2. **#121 — hold in Review** until an NVDA spike day.
3. **331's duplicate-dispatch smell** — off-boundary + paired phantoms on a 30Sec strategy
   is the sharpest *new* lead in the fleet; worth a targeted look even though the aggregate
   class is process-state. `RORT_BAR_DUP_GUARD` is already armed, so this is either a
   different mechanism or the guard not covering the 30Sec path.
4. **Snapshot freeze** — fleet-wide for 6.3d; the alarm is dead. Pointer above.

### Outliers
No individual >30s-late fills inside a passing strategy. 331's paired phantoms (8–11s
apart) are the only anomalous timing shape.

### Open / next
- Re-run the #125 check on the next night a revision lands (watch for a
  `self-healed revision drift` annotation — that's layer 3's first real firing).
- 331: walk the 8 live-only edges against the 10m/4h SWING_123 gate ribbons.
- 267: unchanged structural class — no offline fix; M-RS5 direction.

## 🌙 MORNING BRIEF — nightly bug-hunt 2026-07-23 (Mode 3; E successor abe9f5c3, first act)

**Headline: a NEW live-engine bug was found, repro'd, and fixed tonight — `BarBuilder`
DUPLICATE-PERIOD rows + `_bar_count` inflation (the mechanism behind 136's one-bar-early
max-hold exits, and a candidate feeder of the fleet-wide live-process-state divergence
class). Fix is on dev flag-OFF (`RORT_BAR_DUP_GUARD`, PR #73, dev @`ef447bd`, deployed
03:12:50Z); the ARM is HELD per rail 2, with an always-on tripwire already measuring the
class live. Everything else tonight classified PLUMBING with proven ceilings 99–100 — no
new logic bugs in the gated laggards.**

### Phase 0 — nightly recompute: CLEAN ✅
Job `d0a3d991` fired 00:20:01Z → completed 00:52:25Z: **23/23 strategies, 0 failed,
0 skipped, 33,310 rows.** Fleet `data_refreshed_at` 00:20–00:50Z; shadow heartbeats
streaming (fleet 23/23 @POLL_S=60 — E2's V1.2 flip held overnight). Batch container is
new (`ad8b9078…` vs `cfb4f2bb…`) — expected after the 21:39Z V4.5 rebuild, not an anomaly.

### Fleet SCAN (07-22 RTH 13:30–20:00Z, post-recompute; combined-% identical @5s and @10s)
**Goal-1 (gated) PASS:** 338=100 · 344=100 · 314=97.5 · 341/343/345=100 (NVDA note below).
**Goal-1 FAIL:** 267=81.0 · 271=83.3 · 310=55.0 · 136=33.3 · 194=50.0 · 309=0.
**Non-gated (deprioritized):** 263=100 · 321=100 · 269=98.1.
**Goal-1 read: 6/12 active gated ≥90%@10s.**
- **NVDA trio dashboard 85.7/85.7/80 = window-boundary ARTIFACT; true read 100:** every
  edge pairs exactly, incl. a 20:00:00Z entry BOTH lanes took — the "missed" edge is that
  open trade, whose live alert dispatches seconds past the window end bound. Not a bug.
- Quiet both lanes (legitimately inactive): 312/313/325/330/331/342/340/339. 339's
  live-vs-NEW-settled pairing still awaits live alerts in its 10:00–11:30 ET GEN window
  (board V1.6) — none tonight, as expected.

### 🔥 NEW BUG — BarBuilder duplicate-period rows / `_bar_count` inflation (found → repro'd → fixed)
- **Exposing symptom:** 136 (SPY 1Min, `max_hold_bars: 4`): live max-hold exits fire
  exactly ONE bar early — 16:59/17:32 live vs 17:00/17:33 backtest+algo, entries paired to
  the second. 10-day exit-delta histogram: **+0s ×5, −60s ×7, −120s ×2, −180s, −356s** ⇒
  variable count DRIFT, not a fencepost (retires yesterday's dispatch-lag theory for 136).
- **Mechanism (repro'd against the real class — `src/test_ralph_bar_dup_guard.py`):** the
  AM authority bar closes period P and clears the partial → a late out-of-order tick
  stamped inside P re-opens a partial (`process_tick` has no history check) → next
  rollover closes it: a SECOND P row appends and `_bar_count` increments again.
  `bars_held = bar_count − entry_bar_count` over-counts ⇒ early max-hold exit. Beyond 136:
  the dup row (stale partial OHLC) **feeds the incremental indicator stream**, and
  monitors get a **spurious extra bar-close dispatch** — exactly the shape of the
  live-process-state class in tonight's table. `accept_second_bar` has the same hole in a
  narrower window (post-force-close late sub-bar older than the last row).
- **Class:** LOGIC (live real-time path). Blast radius: 136 is the only `max_hold_bars`
  strategy (fleet audit tonight), but dup-row/extra-dispatch side effects touch every builder.
- **Fix (dev @`ef447bd`, flag-OFF):** `RORT_BAR_DUP_GUARD` — drop late ticks/sub-bars
  targeting a period ≤ the last closed row; `_close_bar` refuses stale-partial closes (no
  append, no increment, no dispatch). Matches backtest semantics (each period exists once;
  the provider aggregate already contains late prints). **Tripwire `BAR_DUP_GUARD` WARN
  logs in BOTH modes** (rate-limited 1/period/builder) — today's RTH quantifies the class
  fleet-wide BEFORE any arm.
- **Validation:** repro tests 4/4 (OFF locks legacy dup; ON drops at both choke points);
  ralph suites (no-gapfill 4/4, subminute parity 6/6, gap-heal 23/23); fidelity parity
  suite **18/18 AND full `--coarse --writethrough` 29/29 — each run flag OFF and flag ON**
  (canary 267 byte-identical).
- **ARMED: NO — held per rail 2.** The dup never lands in recorded lanes (`live_bars`
  upserts per period), so replay cannot predict the paired-% gain ⇒ harness can't prove it
  ⇒ no auto-arm. Proof plan: today's tripwire counts (grep Worker logs for `BAR_DUP_GUARD`)
  correlated with any 136-style early exit.
  **Arm:** `railway variables --set "RORT_BAR_DUP_GUARD=1" --service Worker` · revert: set `0`.

### Bugs found + CLASSIFIED (replay-harness + three-lane arbitrated)
| sid | live@10s | ceiling dtime/corr | self r≈l | class | evidence |
|-----|---------|--------------------|-----------|-------|----------|
| 267 | 81.0 | 99.0 / 99.0 | 89% | **PLUMBING/process-state** | 19 phantom entries spread 13:42–19:50 (not deploy-clustered), all dispatch-lag ~3.3s; **algo↔bt = 100% (97/97)**; clean replay of recorded decision-time bars ≈ backtest ⇒ live's accumulated in-process state, not bars, not logic |
| 310 | 55.0 | 100 / 100 | 71% | **PLUMBING** | live quiet 17:46–17:57 while BOTH offline lanes fired (5 missed, 4 algo-agreed) + 2 tip phantoms |
| 271 | 83.3 | 100 / 100 | 91% | **PLUMBING** (+algo note) | all 6 phantoms shared with algo, 0 missed; separate: the ALGO lane itself over-fires settled bt by 25 entries today (10Sec SPY, decision-time vs settled) — doesn't touch the health metric but matters when the algo lane arbitrates |
| 194 | 50.0 | 100 / 100 | 67% | **PLUMBING/state** | tiny N: live missed a 1-bar bt round-trip 14:01→14:02 (utv4 flip); algo lane empty for the day |
| 309 | 0 | (known 42.9, corr==dtime) | — | **LOGIC — KNOWN structural** | fine-TF VWAP/RVOL volume-gate trap on sub-min primary; 2 phantoms 13:39:30/13:58:45 fit the class; not a whack-a-mole target |
| 136 | 33.3 | 100 / 100 | 50% | **LOGIC → FIXED (above)** | note: its ALGO lane also took 3 extra trades (14:39–17:41) vs bt — small separate algo-lane divergence, same family as 271's note |

**Common thread:** every laggard's ceiling is 99–100 at BOTH decision-time and corrected
bars — the offline engine is faithful; the residual lives in accumulated live-process
state. Structural cure = the canonical-serving / state re-true direction (M-RS5 resident
window family), not per-sid patches. The dup-row bug fixed tonight is a plausible
contributor — the tripwire will show how much.

### ARMED last night: **NOTHING** (0/3 flag budget; live behavior untouched)
The fix shipped **flag-OFF (inert)** — dev merge @03:12:50Z rebuilt api/Worker/batch/
frontend (Worker back up 03:14:36Z after one transient "Server disconnected" during
switchover; warmups clean). **Expected overnight rebuilds: (1) mine @03:12Z ✓, (2) this
docs push, (3) V2.5 tasks-page merge (F lane, after the hunt per Kevin). Anything beyond
those is off-script — flag it.**

### Held for Kevin
1. **Arm decision — `RORT_BAR_DUP_GUARD=1` on Worker** after today's tripwire evidence.
   Rec: if `BAR_DUP_GUARD` warns appear during RTH (esp. SPY 1Min / sub-minute builders),
   arm tomorrow evening. Zero-risk revert (flag to 0).
2. **267/310/271/194 process-state class** — no offline-armable fix exists by construction
   (replay cannot reproduce live's accumulated state). Structural options: (a) canonical
   live serving (M-RS5 resident-window direction), (b) periodic live-state re-true from
   canonical bars (refresher family). Worth a design slot after the fleet promotion.
3. **Pre-existing test failure (NOT from tonight's change):**
   `test_ralph_fidelity.py::test_strategy_monitor_warmup_and_bar_close` "Missing ema_8" —
   fails identically on clean `origin/dev`; stale test vs current indicator set.

### Outliers
- 136's −60s exits (the bug above) were the only systematic >30s-class offsets tonight;
  no >30s-late fills inside passing strategies.

### Open / next
- **Tripwire watch (today RTH):** count `BAR_DUP_GUARD` warns per symbol/TF in Worker
  logs; correlate with any 136 early exit → that's the arm evidence.
- V1.6: 339 live-vs-settled pairing in the 10:00–11:30 ET GEN window.
- 310's 17:46–17:57 live-quiet window: if it recurs, check WS feed continuity /
  monitor state at those minutes (engine-stall telemetry is aboard).
- Fuller fix option (later): rebroadcast-style MERGE for late ticks on the == last-row
  period in `process_tick` (guard drops them; AM authority already corrects closed rows,
  so drop ≈ merge in practice for 1Min+).
- Board: V4.6 folded into PR #73 (session-handoff skill now tracked) → done.

## 🌙 MORNING BRIEF — nightly bug-hunt 2026-07-21→22 (Mode 3)

**Context first (read before the numbers):** 07-21 was operationally MESSY by design —
two mid-RTH Worker restarts (14:52Z label-fix arm, 17:42Z submin-canonical deploy), the
sub-minute canonical source armed post-close (its full record:
`Impl_SubMinute_Canonical_Source.md`), and the shadow-worker's continuous lane FROZEN all
day (below). Today's paired-% reads carry that noise; tomorrow (07-22) is the first clean
scorecard. Nightly recompute: fired 00:20Z, completed 00:59Z, 22/23 refreshed (345 crashed
in the ×8 pool — 3rd time today; its lane is fresh from a 23:50Z solo retry that passed).

**Fleet metric (clean window 17:44–20:00Z, @10s):** gated strategies with activity:
3/9 ≥90% — 338 (100), 314 (98.6), 271 (97.2→100 by window). Below: 341/343/344 (40),
310 (55.6), 136 (60, n=5), 267 (70.8). Gated but silent today: 340 (bt entered 15:11,
live silent), 312/313/325/330/331 (no activity either lane), 339 (canonical validation =
today's 10:00–11:30 ET window). Non-gated (263/269/321) 100% — ignored per SOP.

**Bugs found + CLASSIFIED (three-lane + harness with corrected leg, all verdicts tool-backed):**
- **267 (LooseConf canary) — PLUMBING.** 70.8% live vs CEILING 97.1/97.1 (dtime==corr ⇒ not
  floor, not logic). 13 phantoms spread evenly 18:10→19:33 then trade-for-trade clean
  19:50→20:00. Dispatch healthy (lag p50 3.3s / max 6.2s ⇒ no stall — live *decided* the
  extras). Live=47 vs algo=33 ≈ bt=35 entries. Mechanism hypothesis: in-memory state
  diverged from canonical after the 17:42Z mid-RTH boot (path-dependent utv4 primary state
  / 2m-SWING gate seeded differently at boot than a clean warmup) — drip starts ~26 min
  post-boot. NOT auto-armable; prevention = avoid mid-RTH restarts; cure direction =
  primary-state canonical heal (the deleted PRIMARY-RESYNC's safe successor).
- **341/343/344 (NVDA 1Min trio) — PLUMBING.** 40% vs CEILING 100/100. All three missed
  ONLY the 19:58:00 entry (fired 19:56 ✓ and 20:00 ✓, FLAT at the time — live exited 19:57
  like bt). Bars byte-identical WS==healed==REST 19:55–19:59; algo lane fired 19:58; coarse
  gates (1h/2h/4h) constant in that span ⇒ a one-bar live close-dispatch/evaluation gap at
  19:57→19:58. Logs rolled; mechanism unconfirmed.
- **310 (5m-RVOL + 1d-SUPERTREND) — PLUMBING.** 63.6% vs CEILING 100/100.
- **340 (5m-BB + 3m-VWAP, 1Min primary) — PLUMBING.** Live fired NOTHING all day; bt + replay
  both take the 15:11:00 entry (CEILING 100). 15:11 = 19 min after the 14:52Z restart —
  same boot-state-divergence signature as 267.
- **309 (2m-BB + 3m-VWAP on a 15Sec primary) — LOGIC (known class) + ops on top.** LIVE 25%
  but CEILING itself only 42.9/42.9 (corr==dtime ⇒ engine reproduces the miss offline).
  This is the DOCUMENTED volume-gate trap: fine-TF VWAP/RVOL gates on a SUB-MINUTE primary
  are structurally divergent (offline 3m VWAP sums Σsub-minute volume; live 3m shadow
  builds from 1Min — 1Min ≠ Σsub-minute). See `project_fine_tf_construction_falsified` +
  consumer-#2's volume nuance. Structural cure = volume-consistent canonical fine-TF
  construction (same family as the sub-minute store, one octave up). Held — design item.
- **345 (NVDA 1Min) — INFRA.** Crashes the ×8 recompute pool (3× today: fleet run, nightly,
  reproducible), passes solo (298 trades, 700s). Mitigation available:
  `railway variables --set "RORT_RECOMPUTE_PARALLELISM=6" -s batch-worker` (revert: same
  command with 8). HELD — unproven until a night runs it; costs ~33% nightly duration.
- **Shadow-worker continuous lane FROZEN — INFRA (pre-existing).** `shadow_heartbeats`
  cursors all stopped 07-20 ~19:19Z (the M-RS5a arming evening); polls run ("tracking 4"),
  writes don't. This is why the Strategy Health board showed only-TBDs all day 07-21 until
  the manual Update-All. First item of the mrs5a-merge + shadow-worker `railway up` task.

**ARMED by the hunt: NOTHING.** All findings are live-runtime/ops or known-structural
classes — none clear rail 2 (provable by replay tonight). (`RORT_CANONICAL_SUBMIN_STATE`
was armed pre-hunt at 20:26Z by Kevin's direction — separate record, all gates green,
339's lane rebuilt canonical: 07-16 = {14:11:00, 14:19:30}, replay pairs 14:19:30 exact.)

**Outliers:** none beyond the classified items (267 max dispatch lag 6.2s — fine).

**Open / next:**
1. 339 gate-4: XTF_BLOCK_DIAG present-sets + first canonical live trades, 10:00–11:30 ET
   today, scored vs the NEW canonical lane.
2. mrs5a-branch merge → shadow-worker `railway up` + fingerprint + flag (also unfreezes the
   continuous lane = intraday board freshness returns).
3. 309 volume-gate class: RH_DEBUG drill + design the volume-consistent fine-TF construction.
4. Decide 345 pool width (one-liner above) before tonight's nightly.
5. Watch 267/340 on a clean no-restart day before deeper surgery on the boot-state class.

## SCAN 2026-07-09 ~13:20Z (post-nightly 69/69 clean) — @5s, 24h window
THE STRUCTURAL FINDING: the 21 split by **backtest coverage cutoff** (`fair_cutoff_ts`), not by
strategy quality. Nightly recompute RAN for all (last_recompute=07-09 01:1x) but coverage frozen
for a subset → their backtest lane produces no recent trades.

### GROUP A — coverage current (07-08/09), REAL divergence to work (ranked worst):
| sid | tf | comb% | paired/phan/miss (fair) | theory |
|----|----|-------|--------------------------|--------|
| 329 | 30Sec | 17.6 | 6/0/28 (fair 6/0/18) | under-fires live; backtest MISSES (28) — live gate too tight vs backtest |
| 266 | 5Min | 29.4 | 5/5/7 | two-sided; 5Min-primary class + AH thin |
| 333 | 30Sec | 52.9 | 18/12/4 | 12 phantom — live over-fires vs backtest |
| 327 | 30Sec | 71.4 | 20/2/6 | healing post-#38; missed=6 |
| 328 | 30Sec | 77.8 | 28/2/6 | healing post-#38 |
| 308 | 15Sec | 87.9 | 225/15/16 | close; 15 phantom/16 miss |
| 336 | 15Sec | 89.7 | 52/0/6 | miss=6 only |
| 321/334 | 15Sec | 94.7 | 180/2/8 | ~there |
| 335 | 15Sec | 97.3 | PASS | ✓ |
| 311 | 15Sec | 100 | PASS | ✓ |

### GROUP B — backtest coverage FROZEN despite nightly running:
- **313** (cov 07-02, maxBT 07-02T15:53) — but **51 live alerts since 07-08**. LIVE-vs-BACKTEST
  GATE DISAGREEMENT: gates=1d-UT_BOT_V4-BULL_TREND + 4h-STRAT_ASSISTANT. Hypothesis: live warms
  the 1Day gate from ~70d cache (floor-clamped) → BULL_TREND → fires; backtest warms from full
  history → different state → gate closed → 0 trades. THE daily-gate divergence class. [P1]
- **309** (cov 07-07, maxBT 07-07T13:32) — live near-silent now (1 alert). gates=2m-BB + 3m-VWAP.
  Regressed 96%→0. Backtest stopped 07-07. [P2]
- **310/312/314/325/330/331/339/340** — QUIET both lanes (0 live since 07-08, backtest frozen
  June–07-06), comb%=None. Restrictive gates (1d-SUPERTREND-BEAR, 5m-RVOL-HIGH, dual-daily).
  Likely legitimately inactive (no divergence signal). VERIFY 1-2 aren't silently broken; else
  NOTE as inactive-not-divergent.

## Work order: 313 → 309 → 329 → 266 → 333 → 327/328 sweep. Quiet-8 = verify+note.

## Trade-by-trade findings (13:35Z)
- **309 → FALSE ALARM / NOTE**: over 07-06..09 LIVE(25)≈BT(26) trade-for-trade identical; pairs
  fine. Went inactive after 07-06 (gate closed both lanes); 24h window caught 1 stray → fake 0%.
  NOT a regression. Drop from active list.
- **329 → 1M-SWING live gate UNDER-fires**: BT fires 17 (07-08 RTH), LIVE only 3. gates=
  1M-SWING_123-BULL_C2 + 5m-SWING_123-BULL_C2. 5m works (327/328 post-#38); culprit = 1M gate
  live-vs-backtest disagreement (1M key served "by luck" by 174's monitor per W2-5). Live 1M-SWING
  state ≠ backtest → live gate closed when backtest open.
- **333 → mild two-way jitter** (52.9%): several exact pairs + a few disagreements each direction.
  Lower priority; likely timing + occasional gate flip.

## ⭐ EMERGING SYSTEMIC CLASS: live secondary-TF gate state ≠ backtest secondary-TF gate state
- 313: 1Day gate too OPEN live → over-fire 51/0.  329: 1M gate too CLOSED live → under-fire 3/17.
- Same root (live vs backtest compute a secondary-TF gate state differently — warmup depth /
  computation path). Direction varies. 313 agent diagnosing the 1Day instance = keystone; fix
  likely generalizes to 329 (1M) and others. This is THE class to crack for the batch.

## Second class identified — 266 (primary utv4-flip timing, NOT a gate issue)
266 = TSLA 5Min, UNGATED (no confluence), trigger ut_bot_v4_bull_flip. 07-08 LIVE
[13:45,15:35,16:40,17:30,18:40] vs BT [14:25,15:55,16:45,17:30,18:40,19:10] — 2 exact pairs,
rest off 5-40min. Pure PRIMARY-TF utv4 recursive-trailing-stop epsilon → flip fires on adjacent
5Min bars live vs backtest. Same utv4-epsilon family as the resident-vs-canonical jitter work.
Harder (recursive state convergence); 5Min amplifies (1-bar = 5min). Class 2.

## TWO CLASSES for the batch:
- **CLASS 1 (secondary-TF gate state divergence)** — 313 (1Day), 329 (1M), likely quiet-8 w/
  coarse gates. Live vs backtest compute a secondary gate state differently. BIGGEST lever.
  313 agent diagnosing the keystone mechanism now.
- **CLASS 2 (primary utv4-flip epsilon)** — 266 (5Min ungated), contributes to 308/321/333 jitter.
  Recursive trailing-stop convergence; harder; likely "accept + document" or a warmup-convergence
  fix behind fidelity gate.

## 313 KEYSTONE VERDICT (agent, 13:55Z) — hypothesis REFUTED, real cause found
NOT the daily gate / warmup (deep=shallow=ultra identical; daily+trigger align 51/51). REAL cause:
**4h-STRAT_ASSISTANT gate — live OVER-FIRES on a non-canonical 4H bar.** Live holds 4H=INSIDE
(fires 51x) while backtest correctly = TWO_DOWN (0 trades). Live's INSIDE reproduced by NO clean
resample of bar_cache OR live_bars (both TWO_DOWN); INSIDE appears at 07-06 20:00 AH → live carries
a stale/forming 4H bar across the overnight/session boundary (coarse-gate session-mismatch class,
cf 325/338). BACKTEST IS CORRECT; live is the bug. Recompute genuinely emits 0 (reproduced).
Replay-predict: pin 4H to completed bar → 313 0%->~100% (single-variable). Sub-mechanism
(forming CB vs stale-carry) = MEDIUM confidence — needs one more probe.
BONUS (Kevin's blank Gate Parity panel EXPLAINED): gate_parity_harness.build_gate_parity_view
reads strat.get('entry_trigger') but modern strategies store entry_trigger_confluence_id ->
trigger=None -> 0/0 blank. ALSO only analyzes conf[0]. CLEAN reversible observability fix.
## REFRAME CLASS 1: "live gates on a NON-CANONICAL coarse bar" (forming/stale carried across
session boundary). Session-shadow #31/#32 has a gap for 313's 4H STRAT_ASSISTANT. 329's 1M likely
a different sub-case (174-monitor-served). Diagnosing both.

## 313 SUB-MECHANISM PINNED (agent B, high conf, 3 confirmations) — 14:10Z
CLASS 1a = COARSE-GATE SESSION CONTAMINATION. Live coarse (>=3600s) shadow close-feed is
session-UNFILTERED → ingests AH 4H buckets (16:00-20:00 ET / 20:00 UTC) that RTH resample never
forms; carries stale INSIDE across overnight into RTH. Signature: 51 over-fires 13:36-16:03Z then
STOP at the 16:00Z 4H boundary (live closes RTH bucket→TWO_DOWN, gate shuts). NOT forming-bar.
Set at ralph_engine `_close_shadow_with_bar` RTH branch (eats raw session-unfiltered fan-out bar).
#31 fixed the OPPOSITE archetype (338 non-RTH on RTH shadow); 313 = RTH-on-contaminated-shadow,
untouched by #31. #32 refresher would heal but not effectively running/covering coarse key.
FIX (option 1, surgical): for coarse TFs take the session-correct reload branch for ALL sessions
incl RTH (drop `if session=='RTH': eat completed` for tf>=3600) → _load_warmup_df + _closed_bars_only
+ recompute_confluence = RTH last-closed 4H = backtest. Flag RORT_MTF_COARSE_RTH_RELOAD default OFF,
byte-identical-off. Replay: 313 0%->~100%. Generalizes to any coarse-gated strategy (4H/1Day).
329 = SEPARATE (fine-grain 1M/5m real-monitor own_records, W2-5 topology gap) — NOT batched.

## Option 2 RULED OUT: RORT_MTF_STATE_REFRESH_S=600 IS armed on Worker but 4H gate still
INSIDE at 07-09T12:07 → #32 refresher does NOT cover coarse shadows. Option 1 (close-path
reload) is REQUIRED. Building it.

## SHIPPED — PR #48 gate-parity harness (merged 14:21:44Z, deployed) [reversible/observability]
Fixed: trigger read (entry_trigger→base id via alerts._get_base_trigger_id) + all-gates iteration +
ungated/1M-unresolvable graceful. 313 now shows live_actual:38 + divergent_gate=4h-STRAT_ASSISTANT
(CB 0% blocks all phantoms); 329/266 no crash; 291 17/17 regression-clean. = Kevin's panel + fleet
divergence tool. Deferred (noted in PR): coarse PB ribbon NaN-shallow (uses CB), 1M-from-sub-minute
resample gap (329) — need engine per-TF windowed load.

## Quiet-8 coarse-gate map (for post-coarse-fix re-scan):
- COARSE-gated (4h/1Day, could be affected by RORT_MTF_COARSE_RTH_RELOAD): 310(1d), 312(1d×2),
  314(4h), 325(4h), 330(4h), 331(4h). All 0 live AND 0 backtest → backtest gate genuinely closed →
  likely LEGIT-INACTIVE (not divergence). Re-scan after coarse fix arms to confirm no surprise.
- NON-coarse: 339(30m/10s), 340(5m/3m) — unaffected by coarse fix; inactive.

## SEPARATE DATA ISSUE (noted, not blocking): SPY 10Sec bar_cache drift for 07-06
Parity suite FAILs 2 checks: cache-parity SPY/10Sec/24-7 (12 OHLCVdiffs) + RTH (4 diffs).
CONFIRMED PRE-EXISTING on clean origin/dev (identical fails, canary 267 passes). NOT caused by
#49 (which only touches ralph_engine.py). = bar_cache revision drift for SPY 10Sec on 07-06
(settle-sweeper/T+2 class). DATA-REPAIR task: re-true SPY 10Sec 07-06 via settle sweeper. Queued.

## SHIPPED — PR #49 coarse-gate RTH session reload (merged 14:48:36Z; Worker deployed; ARMED 14:49:52Z)
Flag RORT_MTF_COARSE_RTH_RELOAD=1. For coarse TFs (>=3600s) live shadow takes session-correct
reload for ALL sessions incl RTH → live coarse gate = backtest RTH-resampled last-closed bar.
Fixes the close chokepoint + 3 correction/revision paths (rebroadcast cascade, apply_rest_correction,
apply_rest_insert) that also re-contaminated = CLASS FIX. Validated: flag-on 4H=TWO_DOWN (was stale
INSIDE); flag-off byte-identical (golden-fixture 308/309/313 + routing test); parity suite IDENTICAL
to baseline (2 SPY fails pre-existing/orthogonal, isolated on clean dev). Replay: 313 0%->~100%.
Verifying live: [COARSE-RTH-RELOAD] log + 313 4H flips TWO_DOWN + over-fire stops.
Backup: backup/dev-pre-coarserth-2026-07-09.

## 313 fix ARMED+DEPLOYED confirmed (Worker c071ebb7 @14:49:53Z, 1s post-arm, flag=1).
313 firing 0 entries today post-arm (no over-fire). Reload log fires at next coarse close 16:00Z
(and overnight-crossing test tomorrow) = full live confirmation. Newest 4H telemetry already shows
STRAT_ASSISTANT=TWO_DOWN (correct). Proceeding to 329 in aggressive mode; 16:05Z check queued.

## 329 VERDICT (agent, high conf) — #38 coverage gap (own_records < shadow fidelity)
1M-SWING gate served by 174's OWN_RECORDS (174 incidentally computes SWING_123 from its 2m conf →
#38 _real_monitor_covers counts it as covered → suppresses the 1M shadow). own_records matches
offline only ~60-78% → misses BULL_C2 transients → 329 under-fires 3/17. CLINCHER: 329 offline/algo
lane = backtest (16/17); only LIVE path collapses; sole diff vs 327/328 (85-88%) = shadow-served vs
own_records-served. NOT absent (W2-5), NOT frozen (#2). New sub-class: interp-coverage ≠ fidelity.
FIX: _real_monitor_covers (ralph_engine.py:2775) — incidental interp (resolved via DIFFERENT-TF
confluence, not the monitor's own gate/trigger at THIS tf) must NOT suppress the shadow → 1M gets a
faithful shadow (like its 5m already). Flag RORT_OWNRECORDS_INCIDENTAL_NO_COVER default OFF.
Replay: 329 18%->~85-100% (15-17 live vs 17 bt). GENERALIZES: fleet sweep for sub-coarse gates
whose interp is neighbor-own_records-served (signature = key carries neighbor's full own interp set).
BUILDING; hold merge for batch after 313 16:00Z confirm.

## 313 fix WORKING LIVE (16:01Z gate diag): 4H-STRAT_ASSISTANT=TWO_UP (canonical RTH), NOT phantom
INSIDE. Gate closed → 313 silent (0 over-fire). Definitive overnight-crossing proof tomorrow.

## 329 fix STOP — diagnosis DISPROVEN by build-agent validation (discipline win, NO ship)
Shadow-coverage fix is INERT: sid 340 (1Min monitor) has sw123_bull_c2 as its PRIMARY trigger →
GENUINELY computes 1M-SWING_123 → TF-aware coverage still credits it → 0 shadows added → cannot
change 329. REAL mechanism (proven): cross-TF gate ALIGNMENT/HOLD timing — SWING C2 is a
sub-telemetry transient; live gate holds last-closed secondary bar, backtest uses shifted/aligned
bar (secondary_tf_shift semantics). Proof: 329's 5m leg is ALREADY a shadow, still 3/17 — topology
isn't it. Real fix = live secondary gate alignment/hold vs backtest shift — DELICATE (high fidelity
risk, touches secondary_tf_shift). DEFER; not a quick win. See feedback_parity_secondary_tf_shift.
RECLASSIFY 329 → cross-TF-alignment class (with 266? — 266 is primary utv4, different). 
NEXT: easier wins first (333 jitter, 308 88%, 327/328 healing tail); 329 alignment fix needs its
own careful diagnosis + parity gate.

## ⭐ COARSE-GATE FIX (#49) = MAJOR WIN — heals BOTH directions (regression check → improvement)
POST-arm (14:50-16:20) vs PRE-arm (13:30-14:49):
- 325: 0%(0live/7bt) -> 100% (8/8)   | 330: 0%(0/7) -> 100% (9/9)   | 331: 4pair -> 100% (7/7)
- 313: over-fire -> silent (gate correctly closed)
Same contamination caused 313 OVER-fire (4h wrongly INSIDE/open) AND 325/330/331 UNDER-fire (4h-SWING
wrongly closed). Pinning live 4h to RTH-canonical fixed BOTH. 24h scan hid it (averages pre-fix).
1 flag, 4 strategies corrected, 3 to ~100%. NO regression. This is the class-fix leverage.

## 308 drill → PRIMARY-TRIGGER WS-vs-REST TIMING FLOOR (irreducible, accept+document)
308 ungated 15Sec swing_123. [15:30-16:00Z] LIVE 9 / BT 8: 6 EXACT pairs, 1 off-by-one-15Sec-bar
(15:36:15 vs :30), 2 phantom, 1 missed. NOT a logic bug — swing_123 C2 is a 2-bar candle pattern;
live fires on real-time WS decision bar, backtest on settled REST bar → occasional 1-bar early/flip.
~85-88%@5s, higher at wider tolerance. INHERENT real-time-vs-fidelity cost; not fixable without
changing the real-time firing model (wait-for-settle defeats sub-second latency). Bar-store
unification (Kevin idea#1) does NOT fix this (it fixes GATE construction, not the real-time trigger
decision). CLASS: primary-trigger timing floor = 308, 321, 334 (swing_123), 266 (utv4). ACCEPT+DOC.

## CONSOLIDATED STATUS (2026-07-09 bug-hunt, ~16:30Z)
FIXABLE GATE CLASSES — SHIPPED/HEALED:
- Coarse-gate session contamination (#49): 313 over-fire→silent, 325 0%→~90-100%, 330 →~90-100%,
  331 →~100% (post-arm). ONE flag, 4 strategies, both directions. + gate-parity panel (#48).
DEFERRED-FIXABLE (delicate, own diagnosis + parity gate):
- Cross-TF transient alignment (327/328/329): 5m/1M-SWING C2 hold vs backtest shift (secondary_tf_shift).
IRREDUCIBLE FLOOR (accept+document, NOT bugs): 308/321/334 (swing_123), 266 (utv4) primary timing.
INACTIVE (legit, gates rarely open): 310/312/314/339.
PASS: 311, 335. AH/small-N: 337, 340, 309.

## CORRECTION (Kevin): "308 floor" was UNVALIDATED — 308 has NO stored algo lane (only
backtest_rest_hifi); the floor claim needs the ALGO-LANE cross-check (compute offline).
Three-lane test: algo≈live (both fire off-by-one) → WS-vs-REST floor CONFIRMED; algo≈backtest
(live diverges) → LIVE-PATH BUG (fixable), NOT a floor. Do NOT throw hands up without it.
NOTE: 329's algo check WAS done (agent): algo≈backtest 16/17, live 3/17 → 329 is a LIVE-PATH bug
(cross-TF alignment), NOT a data floor. Same test now required for 308/321/334/266 before calling floor.

## 329 ROOT CAUSE (agent, PROVEN) — '1M' TF-LABEL NAMESPACE COLLISION → BACKTEST OVER-COUNTS
329 gate 1M-SWING_123-BULL_C2 uses '1M' (UPPERCASE) for 1-MINUTE. LIVE (_LABEL_TO_TF_SECONDS):
'1M'=60s → enforces a real 1-min gate → live=3. OFFLINE (backtest+algo): '1M'=RESERVED PRIMARY
SENTINEL → data_loader.get_required_tfs_from_confluence:891 DROPS it → no-op → backtest=17.
PROOF: 328(5m) & 329(5m+1M) backtests BYTE-IDENTICAL (17≡17). Only LIVE reacts. ARBITRATION FLIPS:
BACKTEST+ALGO OVER-COUNT; LIVE CORRECT. 329 stored bt KPIs INVALID. Algo shares unified_engine →
can't arbitrate (algo≈backtest = shared bug). = Kevin's "don't assume backtest right" strongest form.
The earlier "make live fire more" fix would've made LIVE WRONG to match a broken backtest.
308/266 FLOOR CONFIRMED via algo (algo≈live≠bt on live data; algo≡bt settled). 327/328 minor floor.
329 FIX NEEDS KEVIN INTENT: (A) enforce 1M offline (fixes over-count, HIGH namespace risk+parity gate);
(B) drop 1M live (isolated/kill-switch, 329→328-dup, discards intent); (C) config hygiene regardless.

## 329 '1M' FIX — Kevin-aligned plan (build): treat gate as a gate, primary-AWARE, native 1Min
Root: '1M' overloaded — interpreters.py:863 uses '1M' as PRIMARY-TF sentinel; MB (confluence_groups
:1323) emits '1M' meaning 1-minute → offline get_required_tfs_from_confluence:891 blanket-drops '1M'
→ gate silently ignored. Only 2 sids: 329 (30Sec primary, 1min=SECONDARY→enforce) + 136 (1Min
primary, 1min=primary MACD condition). Lowercase '1m' IS enforced offline; resample rules have NO
1Min (it's native). FIX (Kevin's principle): exclusion should skip a gate only when its TF == the
strategy's PRIMARY TF (self-ref), not blanket-drop '1M'. Build 1-min secondary from NATIVE 1Min bars
(consistent w/ 1Min-primary; no resample drift; Kevin's steer). Flag-gated default-OFF byte-identical
+ parity suite 18/18 + 329 bt 17→~3 matching live proof. + MB emits '1m' lowercase (source fix) +
silent-drop TRIPWIRE (no silent fails). Replay: 329 →~100% paired, KPIs become valid.

## ✅ 329 FIX VERIFIED LIVE (PR #50, armed 18:24Z): backtest 07-08 17→3 = live 3 (recompute
completed 0-fail, +121 rows). Backtest over-count ELIMINATED; 329 KPIs now valid; pairs ~100%.
shadow-worker fingerprint c202b6b210c8 OK (verified tree). Flag RORT_ENFORCE_1MIN_GATE=1 on
api+batch-worker+shadow-worker. Primary-aware class fix + MB '1m' source fix + silent-drop tripwire.

## SESSION CLOSE 2026-07-09 — bug-hunt aggressive, 3 fixes shipped+verified, 2 correctly NOT shipped
SHIPPED+VERIFIED: #48 gate panel · #49 coarse-gate (313 silent + 325/330/331 ~0→90-100% BOTH dirs)
· #50 1-min-gate (329 bt 17→3=live; backtest was the bug). DISCIPLINE: 2 inert 329 fixes caught
pre-ship; 308/266 floor CONFIRMED via algo lane (not hand-wave); backtest-is-the-bug inversion
exposed by Kevin's algo-lane insistence. ARCH captured: Design_Gate_Fidelity_Hardening.md.
TOMORROW: 313 overnight-crossing proof; clean-day rescan (325/330/331/329); SPY 10Sec re-true;
own_records fleet sweep; data_worker/app.py 1Min wiring + frontend '1M' canonicalization follow-ups.

## PROACTIVE AUDITS (follow-up hours, 18:45Z) — fleet is CLEAN of the class
- SILENT-DROP AUDIT (all 70 configs): ONLY 329 (fixed #50) + 136 (primary-self-ref, handled #50).
  ZERO other silent-drops. '1M' class fully contained; tripwire guards future. ✓
- COARSE-GATE FLEET: 9 coarse-gated (310/311/312/313/314/325/330/331/338). #49 covers all; 4 healed
  (313/325/330/331), 311 pass, 338 (#31 archetype), 310/312/314 inactive. Generalizes, no regression. ✓
- SPY 10Sec re-true: sweeper is on dev (not this checkout) → bundle w/ tomorrow dev tooling.
- #50 resident-path gap: LOW severity (329 07-09 resident bt=1 vs live=2, NOT over-count; nightly
  masks). Building completeness PR now, DEPLOY TOMORROW (no end-of-day Worker deploy).

## PR #51 READY (resident-path 1-min-gate completeness) — DEPLOY TOMORROW, not today
Gap DEEPER: 3 unwired sites (services.prepare_strategy_window_df + ResidentStrategyEngine +
data_worker_engine). KEY: normalize engine confluence_set '1M-'→'1m-' too (sub-minute primary
own-records are '1M-' → raw '1M-' gate matches primary's own state). Validated: 329 resident OFF
11==11 / ON 2==2; 136 self-ref byte-identical; flag-off byte-identical; canary 267 symdiff=0.
TOMORROW: arm on Worker+shadow-worker (fingerprint SOP); _shadow_manager_validate RED under flag-ON
= HARNESS LIMITATION not regression (recompute is oracle). Held unmerged (avoid orphaning 20:20Z recompute).
