# Gated-Five Divergence Autopsy (sids 327 / 328 / 329 / 333 / 340)

> **CORRECTION 2 (2026-07-13, 340 exits reclassified D→C):** the MACD-cross exit divergence is NOT
> the WS≠REST floor — every canonical reconstruction (REST settled, live_bars first-write, revised)
> crosses on the SAME bars as backtest. The live monitor's IN-MEMORY incremental MACD state drifts:
> injected at boot (native days=7 warmup reads an unsettled bar_cache tail) then wanders (262
> unconsumed rest_insert gap bars; fire-and-forget writer). Cross margins ~0.005-0.03 vs state error
> 0.01-0.16 → ±1-4 bar shifts, both directions. Fixes: Phase-4 live-serve warmup (kills boot
> injection); interim settled-only warmup tail; extending the refresher to PRIMARY-TF pack engines
> needs a Kevin ruling (touches the live position machine mid-session). LATENT (separate ticket):
> grace-fire sets _fired_bucket even with no signals (ralph 2838) → close-commit signals for that
> bucket suppressed (3846) — a one-shot cross can be silently dropped on spliced models.

> **CORRECTION (2026-07-13, validation-gated):** cause **A2's mechanism was FALSIFIED** during
> implementation — sid 333's 15Min BEAR lock is NOT warmup hysteresis (UT Bot's ratchet resets per
> flip; 0/728 depth-sensitive closes; the '7 reboots' were pre-market when BEAR was correct). The
> REAL mechanism is a **15Min close-feed FREEZE** (0609a family): live_gate telemetry shows ZERO
> 900s-key evaluations 13:30-17:30Z on 07-10 while the 180s key updated every ~3min and truth
> flipped 6×. Operative fix = the existing state refresher, ARMED 17:07Z
> (RORT_MTF_STATE_REFRESH_S=600). Deep-anchor shipped default-OFF as insurance (PR #57). OPEN
> ROOT-CAUSE: why the 900s fan-out starves while 180s flows — dedicated hunt queued.


**Date:** 2026-07-11 (Saturday, markets closed — all data settled)
**Weekend Sprint Item 4** — `Plan_MRS2_P2_Rollout.md` Phase 2.4
**Window:** last 5 trading days, 2026-07-06 → 2026-07-10, RTH
**Question:** Sprint Item 2 (`Fine_TF_Divergence_Quantification.md`) proved fine-TF bar
*construction* imposes at most a ~1% gate-flip ceiling — so WHY do these five gated
strategies pair at only 22–63% vs live alerts?

**Answer (headline):** Four distinct causes, none of them bar construction, and the
biggest one is **already fixed**:

| rank | cause | share of unpaired entries | status |
|---|---|---|---|
| 1 (tie) | **A. Interp-blind secondary shadow** (pre-PR#38): the 5Min shadow carried NO SWING_123 / BOLLINGER_BANDS record at all → gate could never pass → live never fired | 33/109 (30%) | **FIXED 07-07 ~17:30Z** (W2-5, PR#38). Window straddles the fix; headline paired-% is stale |
| 1 (tie) | **C. Live shadow interpreter-state divergence** (stale / wrong / one-bin-behind records on path-dependent packs: SWING_123 5m/1m, UT_BOT_V4 15m, BB 5m) | 33/109 (30%) | OPEN — the 272-class family; store cutover + shadow re-true |
| 3 | **B. Cross-TF boundary race (shift semantics)**: when a secondary close coincides with the primary close, live gates on the JUST-closed secondary bar; backtest (PB, shift+ffill) sees it one primary bar later | 29/109 (27%) | OPEN — deterministic semantics gap, needs an alignment decision |
| 4 | **D. Primary trigger-lane timing** (WS vs REST on the 30Sec/1Min primary) | 7/109 (6%) + sid 340's exits | the known WS≠REST floor |
| 5 | **E. Gate-pass anomaly**: live fired while BOTH settled truth AND its own 2s-cadence gate telemetry said the gate was closed | 7/109 (6%) | OPEN — needs Railway `GATE_DIAG` logs (see §8) |

Entry vs exit split: **for 327/328/329/333 the exits are pure knock-on** — on paired
entries, exits pair 18/18, 26/26, 2/2, 42/43. **Only sid 340 has genuine exit-side
divergence** (MACD-cross exits 60–240s apart on the 1Min primary; stops pair exactly).

---

## 1. Method

1. **Event ledger.** `alerts` (event_type='fill') vs `trades` (data_source
   'backtest_rest_hifi'), 07-06 00:00Z → 07-11 00:00Z. Greedy nearest-first pairing at
   **±10s**, entries and exits separately. Entry triggers: `sw123_bull_c2`
   (327/328/329/340), `utv4_bull_flip` (333); exits: `sw123_bear_c2`/`utv4_bear_flip`/
   `mlv2_cross_bear` + `stop_loss`.
2. **Theoretical truth = the real engine.** Per sid:
   `services.prepare_data_with_indicators` (REST/hifi lane, `no_backfill=True`,
   `RORT_COARSE_SECONDARY_FROM_1MIN=1`, `RORT_ENFORCE_1MIN_GATE=1`, session RTH,
   native-1Min injection for 329's 1M gate) over warmup+window (327/328/329 from
   06-24; 333 from its true backtest start 06-15; 340 from 06-22), then
   `unified_trades`. Gate columns extracted per gate: PB = `<INTERP>__<tf>`
   (previous completed secondary bar — what backtest gates on), CB =
   `_spec_<INTERP>__<tf>` (the containing bar at its close).
   **Validation: the reconstruction reproduced the recorded backtest ledger EXACTLY —
   38/38, 59/59, 7/7, 50/50, 10/10 entries (164/164, ±10s).** The theoretical gate
   columns therefore ARE the backtest lane's truth; every unpaired event is a
   live-side divergence.
3. **Live's recorded belief.** `bar_diagnostics` source=`live_gate` (change-only
   snapshots of the hub's `_mtf_confluence[(tf, session)]`, emitted from the periodic
   loop every `PICKLE_WRITE_INTERVAL` = **2s** — so any record state lasting >2s is
   captured). For each unpaired entry, the latest row ≤ decision time per gate TF
   (RTH-keyed), compared to the theoretical PB state at the decision bar, at the
   belief row's own write time, and at one bin earlier (lag detection).
4. **Sibling cross-check.** 327/328/329 share `sw123_bull_c2` on the same TSLA hub —
   a sibling's live fire at ±3s proves the live primary trigger fired, isolating
   gate-level blocks from trigger-lane misses.
5. Every unpaired ENTRY (both directions) got a verdict; ambiguous cases are labeled,
   not forced. Read-only throughout (config loads via direct PG; all user-config save
   paths monkeypatched to no-ops — the empty-load auto-save trap is real:
   `db._load_user_config` → `maybe_single()` 204 → `create_default_groups()` → save).

Scratch scripts (session scratchpad, not committed): `s1_configs_ledger.py`,
`s2_pairing.py`, `s3_theoretical.py`, `s4b_verdicts.py`, `s5_final.py`.

## 2. Pairing results (the denominator)

Full window 07-06 → 07-10, ±10s:

| sid | entries live/bt → paired | exits live/bt → paired | combined | exits on paired entries |
|---|---|---|---|---|
| 327 | 23 / 38 → 18 (47%) | 23 / 38 → 18 (47%) | 47% | **18/18 @10s** |
| 328 | 33 / 59 → 26 (44%) | 33 / 59 → 26 (44%) | 44% | **26/26 @10s** |
| 329 | 7 / 7 → 2 (29%) | 7 / 7 → 2 (29%) | 29% | **2/2 @10s** |
| 333 | 59 / 50 → 43 (73%) | 59 / 50 → 43 (73%) | 73% | **42/43 @10s** |
| 340 | 11 / 10 → 5 (45%) | 10 / 10 → 2 (20%) | 33% | **2/5 @10s** |

Clean window (07-08 → 07-10, i.e. post-#38 only):

| sid | entries | exits | combined |
|---|---|---|---|
| 327 | 18/27 (67%) | 67% | **67%** |
| 328 | 25/39 (64%) | 64% | **64%** |
| 329 | 2/7 (29%) | 29% | **29%** |
| 333 | 19/31 (61%) | 61% | **61%** |
| 340 | 5/11 (45%) | 2/10 (20%) | **33%** |

Two structural facts:
- **Cascades are ~zero.** Almost every unpaired event happened while the OTHER lane
  was flat (bt_in_pos / live_in_pos ≈ 0 across the board) — these are genuine
  gate/trigger divergences, not position-state knock-on at the entry level.
- **Divergence is correlated across sids, not independent:** the 5Min RTH shadow is
  shared, so one bad 5m SWING_123 record blocks (or fires) 327+328+329
  simultaneously (e.g. all three blocked 07-09 19:06–19:09; 327+328 both fired the
  07-09 18:04:30 anomaly).

## 3. sid 327 — TSLA 30Sec, gates `2m-SWING_123-NEUTRAL` + `5m-SWING_123-BULL_C2`

25 unpaired entries (5 live-only, 20 bt-only):
**A pre-#38 blind 11 · B boundary race 7 · C shadow-state 5 · E anomaly 2**

- **A (11):** all of 07-06 and 07-07 morning. The 5Min shadow emitted records for
  other interpreters but **no `5M-SWING_123-*` record existed until 07-07 17:30Z**
  (first-ever alert for 327: 07-07 18:40). Gate could never pass → live missed every
  backtest entry.
- **B examples:** live-only 07-10 15:10:00 (5m boundary): th_pb=`NEUTRAL`,
  th_cb=`BULL_C2`, live belief=`BULL_C2` (age 0s, right-at-write) — live's 5m record
  flipped at the 15:10:00 close and the 30Sec trigger on that same close fired;
  backtest gates on the previous 5m bar → phantom. Siblings 328+329 fired too.
  Mirror-image block: 07-10 14:15:00 (5m boundary): th_pb=`BULL_C2` (backtest open,
  entered), th_cb=`NEUTRAL` — live saw the just-closed NEUTRAL bar → blocked.
- **C examples:** 07-09 19:06:30 / 19:08:00 / 19:09:00 bt-only: truth th_pb=`BULL_C2`
  for all three, live 5m record `5M-SWING_123-NEUTRAL` (age 90–240s, wrong at write) —
  blocked 327, 328 AND 329. 07-08 14:57:30: belief `NEUTRAL` written 14:55:02 = the
  state of the bar that closed at 14:50 (one full 5m bin behind truth `BULL_C2`).
- **E:** 07-07 19:33:00 and 07-09 18:04:30 — see §8.

**Dominant cause:** pre-#38 blind shadow (now fixed); residual = 5m SWING shadow-state
divergence + boundary race, ~50/50. Exits: 18/18 — pure knock-on.

## 4. sid 328 — TSLA 30Sec, gate `5m-SWING_123-BULL_C2`

40 unpaired entries (7 live-only, 33 bt-only):
**A 19 · C 11 · B 6 · E 4**

- **A (19):** 07-06 ×12 + 07-07 ×7 — same blind-shadow window (first-ever alert
  07-07 18:40:08, same second as 327).
- **C (11, the biggest residual):** the live 5m SWING_123 record repeatedly reads
  `NEUTRAL` while settled truth is `BULL_C2` at decision time — 07-08 14:57:30 +
  14:59:00 (one-bin-behind), 07-09 19:35:30/19:36:30/19:39:00 + 07-10 13:45:30
  (belief `NEUTRAL` at age 30–240s vs truth `BULL_C2`), 07-09 16:18:30 (belief right
  when written 23 min earlier, 4 state changes since — classic stale). SWING_123 is a
  pattern interpreter with internal swing-anchor state: the live shadow's state
  LINEAGE (boot warmup + WS feed + reconcile recomputes) drifts from the settled
  computation and stays drifted — this is NOT bar-byte construction (Item 2 measured
  that at 0.0% for 5m SWING with identical warmup).
- **B (6):** same boundary-race signatures as 327 (07-08 19:55:00, 07-09 16:20:00,
  07-10 14:15:00, 16:35:00 blocks; 07-07 18:40:00, 07-10 15:10:00 fires).

**Dominant cause:** pre-#38 blind shadow; residual = live 5m shadow state
stale/wrong/lagged (11) > boundary race (6) > anomaly (4). Exits 26/26 — knock-on.

## 5. sid 329 — TSLA 30Sec, gates `1M-SWING_123-BULL_C2` + `5m-SWING_123-BULL_C2`

10 unpaired entries (5 live-only, 5 bt-only):
**B 6 · C 2 · A 1 · E 1** — and still only 29% in the clean window (small n: 7 live alerts).

- 329 stacks TWO gates and the 1M gate is boundary-aligned with EVERY other primary
  bar → double exposure to both the boundary race and shadow-state divergence, which
  is why it scores worst. It also fired live only from 07-08 16:30 (its 5m gate was
  blind pre-#38 like its siblings; its bt-only 07-06 16:37:30 is class A).
- **B examples:** bt-only 07-09 19:09:00 — 1m gate: th_pb=`BULL_C2` (backtest open)
  but th_cb=`BEAR_C2` on the bar closing that instant; live belief `NEUTRAL`/fresh →
  blocked while both siblings were also blocked by the 5m record. Live-only 07-08
  18:06:00 (both 1m and 5m aligned): live fired off just-closed states
  (1m belief `BULL_C2`, age 120s ≈ CB), th_pb says `NEUTRAL`.
- **GATE_FORMING oddity:** live-only 07-08 17:22:30 — belief `1M-SWING_123-BULL_C2`
  recorded at 17:22:00 equals the state of the 1m bar that only CLOSES at 17:23:00
  per REST semantics (th_pb=`NEUTRAL`, th_cb=`BULL_C2`) — the live 1m record was one
  bin AHEAD (forming-bar state or bin-labeling skew in the native-1Min feed).

**Dominant cause:** cross-TF evaluation timing (boundary race) amplified by the 1M
gate; the multiplicative two-gate structure makes every per-gate divergence bite
twice. Exits 2/2 — knock-on.

## 6. sid 333 — TSLA 30Sec, gates `3m-STRAT_ASSISTANT-TWO_DOWN` + `15m-UT_BOT_V4-BEAR_TREND`

23 unpaired entries (16 live-only, 7 bt-only) — never interp-blind (both records
since 06-22): **C 12 · B 8 · D 3**

- **C (12) — the 15m UT_BOT wrong-side lock, the sid's signature failure.** The live
  15m UT_BOT_V4 record sits on the WRONG side of the trend for HOURS:
  - 07-10: telemetry rows at 11:15/12:00/12:15/12:45/13:00/13:15/13:30 (pre-market
    re-boots) all `15M-UT_BOT_V4-BEAR_TREND`, unchanged through 17:30 — while settled
    truth flipped `BULL_FLIP`→`BULL_TREND` around 15:44 (6 theoretical state changes
    13:30→16:30). Result: 6 phantom live entries 15:44:30→16:29:30 gated by a
    BEAR_TREND belief that was ~3h wrong. Crucially the wrong state SURVIVED REBOOTS
    — re-warmup re-derives BEAR — so this is NOT a stopped feed; it is **hysteretic
    interpreter state divergence** (UT BOT's ATR trail is path-dependent; a shallow/
    different warmup latches the opposite trend and stays latched).
  - 07-08: live's 15m UT_BOT flipped BULL at 19:45 (its only RTH record change that
    day) — ~3.5h after truth flipped (~16:15) → 3 phantom fires 16:27–16:50 + stale
    blocks at 17:00/18:19 the day before.
- **B (8):** 3m boundary race, textbook: live-only 07-07 16:27:00 (3m boundary) —
  th_pb=`TWO_UP` (backtest gate closed), th_cb=`TWO_DOWN`, live belief updated to
  `TWO_DOWN` at 16:27:00 (age 0s) → live fired on the just-closed 3m bar. Same
  pattern 07-06 13:57, 07-07 14:06, 07-08 14:06/15:06; and the block mirror 07-09
  17:00:00 (th_cb=`TWO_UP` closing that instant blocked live while backtest entered
  on th_pb=`TWO_DOWN`).
- **D (3):** e.g. bt-only 07-10 14:46:00 — both gates open in truth AND in live's
  belief, theoretical trigger fired, live's 30Sec `utv4_bull_flip` didn't → primary
  trigger-lane (WS vs REST) miss.
- Observation: live fired on 07-07 14:06/16:12 while its recorded 15m state was
  `BEAR_FLIP` (gate wants `BEAR_TREND`) — consistent with the live FLIP→TREND gate
  semantics (cf. feedback_userpack_interpreter_live_dispatch); not the divergent
  gate on those events (the 3m was), but worth remembering when reading telemetry.
- Exits: 42/43 knock-on-clean; one real exit divergence 07-08 13:36:30 entry (live
  exited 13:38:30, backtest 13:48:30 — utv4_bear_flip 10 min apart) and one cosmetic
  same-second reason swap (07-07 19:46-47 stop_loss vs utv4_bear_flip).

**Dominant cause:** live shadow state divergence — specifically the hysteretic 15m
UT_BOT lock — with the 3m boundary race second.

## 7. sid 340 — TSLA 1Min, gates `5m-BOLLINGER_BANDS-SQUEEZE_MID` + `3m-VWAP_V2->+1σ`

11 unpaired entries (6 live-only, 5 bt-only): **D 4 · C 3 · B 2 · A 2** — plus the
only genuine EXIT divergence of the five.

- **A (2):** 07-06 16:50 + 17:19 — the 5Min shadow had NO `5M-BOLLINGER_BANDS`
  record until 07-07 17:30Z (same interp-blind class; its `3M-VWAP_V2` record existed
  all along). 340's first-ever alert: 07-09 15:01.
- **D (4 entries):** live-only 07-09 15:38 / 18:08, 07-10 17:07 — both gates open in
  truth and belief, but the theoretical 1Min `sw123_bull_c2` did NOT fire → live's
  WS 1Min bar shaped a swing pattern REST doesn't show. Mirror: bt-only 07-09
  19:33:00 (all open, live trigger silent).
- **Exit-side (the real story):** entries pair 5/11 but exits only 2/10. On the 5
  paired entries: both `stop_loss` exits match to the second; all 3
  `mlv2_cross_bear` (MACD-cross) exits differ by **60s / 240s / 60s** — the MACD
  EMA path on live's spliced 1Min lane crosses on different bars than REST. Same
  class as D but on the exit trigger, and it pushes every affected pair outside the
  10s bar.
- **C (3):** e.g. 07-09 17:36:00 live-only — BB gate truth `SQUEEZE_UPPER`, live
  belief `SQUEEZE_MID` (wrong at write); 07-10 16:26:00 — belief `SQUEEZE_MID`
  written 11 min earlier, truth moved to `SQUEEZE_UPPER` (1 change since) → stale fire.

**Dominant cause:** primary-lane (1Min WS vs REST) trigger/exit timing — entries AND
especially MACD-cross exits — with gate shadow-state divergence second. The only sid
where "fix the gates" won't be enough.

## 8. The gate-pass anomaly (E) — 7 events, unexplained, actionable

327×2, 328×4, 329×1 (07-07 19:33:00; 07-09 18:04:30 [327+328 simultaneously],
19:21:00, 19:23:30, 17:00:30; 07-10 17:54:00; 07-08 14:29:00 classed WRONG_FIRE is
adjacent). At 07-09 18:04:30:

- settled truth: `SWING_123__5m` th_pb=`NEUTRAL`, th_cb=`NEUTRAL` (not a boundary);
- live telemetry (2s cadence, change-only): `5M-SWING_123-NEUTRAL` at 17:55:02 and
  still `NEUTRAL` at 18:05:01 — no change row in between;
- yet BOTH 327 and 328 fired `sw123_bull_c2` entries gated on `5M-SWING_123-BULL_C2`
  (alert `indicator_snapshot` confirms the primary trigger really fired:
  `sw123_bull_c2: true`).

Since the 5m record set is shared, whatever passed the gate passed it for both
monitors at once. Candidate mechanisms: (a) a sub-2s transient in
`_mtf_confluence[(300,'RTH')]` (e.g. `shadow.recompute_confluence` after a REST
reconcile briefly emitting `BULL_C2`, reverted at the next close); (b) a gate-check
bypass — note `unified_engine.check_entry` line ~2527 skips the subset check entirely
when `confluence_records` is empty (`if self.confluence_set and confluence_records:`),
though the monitor's own-TF records should make the set non-empty at every close.
**Action:** pull Railway worker logs for `GATE_DIAG strat=327|328` at those
timestamps — ralph_engine.py:1357 logs `subset_ok` and the exact record set at every
gated fire. `subset_ok=0` ⇒ fail-open path; `subset_ok=1` ⇒ transient record flip.

## 9. Cross-sid synthesis and fixes

Ranked causes over all 109 unpaired entries (07-06→07-10):

1. **A. Interp-blind shadow, 33 (30%) — already fixed** (W2-5 / PR#38, live
   07-07 ~17:30Z). Implication: the 22–63% headline is measured across a dead bug.
   **Re-baseline gated paired-% from 07-08** (or from each sid's
   most-recent-activation per the 90%@10s bar): 327→67%, 328→64%, 333→61%,
   340→33%, 329→29%.
2. **C. Live shadow interpreter-state divergence, 33 (30%).** Stale / wrong /
   one-bin-behind gate records on path-dependent packs (SWING_123 pattern anchors,
   UT_BOT ATR-trail hysteresis, BB). Construction is exonerated (Item 2); the
   shadow's STATE LINEAGE is the problem — and it survives reboots because re-warmup
   re-derives it (sid 333's 15m BEAR lock across 7 pre-market boots on 07-10).
   Fixes, in increasing strength: (i) extend the 272-class refresher (#32) to fine
   TFs — periodic `recompute_confluence` from settled canonical bars; (ii) M-RS2
   Phase 4 live serve cutover so warmup/reload reads the store (kills feed-lineage
   drift); (iii) for hysteretic packs, deep-anchor warmup (M-RS1 right-sizing —
   UT_BOT trail needs history to converge, cf. feedback_indicator_warmup).
3. **B. Cross-TF boundary race, 29 (27%).** Deterministic semantics gap: at primary
   closes that coincide with a secondary close, live's gate already reflects the
   just-closed secondary bar (telemetry age 0s, "right at write"), while backtest's
   shift+ffill applies it from the NEXT primary bar. Fires phantoms when the fresh
   state opens the gate, blocks real entries when it closes it (~symmetric: 15 fires
   / 14 blocks). The store cutover does NOT fix this — it is ordering, not bytes.
   Fix = pick a semantic and align: defer secondary-close application by one primary
   bar in live (parity-preserving, matches PB), or move backtest to
   close≤decision-time semantics (changes every backtest). Exposure scales with
   boundary alignment frequency — 329 (1M gate = every other bar) is the worst hit.
4. **D. Primary trigger-lane WS≠REST, 7 (6%) + sid 340's exits.** The accepted floor
   class per the 90%@10s bar for 30Sec sids — but 340's MACD-cross exits (60–240s
   apart) blow through ±10s and cap its combined score near ~50% even with perfect
   gates; if 340-style 1Min MACD exits matter, the spliced-lane cross timing needs
   its own look.
5. **E. Gate-pass anomaly, 7 (6%).** §8 — one log-pull from resolving.

**What this predicts:** with A already gone, fixing C+B should lift 327/328/333 from
~61–67% to ~90%+ (their D+E residue is small); 329 additionally needs the 1M-gate
boundary exposure addressed (it's in B); 340 needs D (primary + exit lane) more than
it needs gates.

## 10. Honest caveats

- Live "belief" comes from change-only telemetry at 2s cadence: sub-2s transients are
  invisible (relevant only to class E), and for the 1m gate the telemetry granularity
  is close to the gate period — 329's per-event belief reads carry more uncertainty.
- The one-bin-lag test (belief == previous bin's theoretical state) can coincide with
  legitimate state repetition (NEUTRAL is common); LAG vs WRONG sub-labels within
  class C are indicative, the class total is solid.
- Verdicts attribute each event to the first theoretically-closed gate; multi-gate
  sids (329, 333, 340) can have compound causes per event.
- Theoretical truth is the CURRENT engine on today's settled hifi cache — validated
  byte-equivalent to the recorded backtest ledger (164/164), so this is exactly the
  lane live is being paired against.
- Pre-#38 events were bulk-attributed to class A by timestamp (< 07-07 17:30Z);
  a few of those would otherwise have landed in B/C — A is, if anything, slightly
  overstated and B/C understated for 07-06/07-07.
