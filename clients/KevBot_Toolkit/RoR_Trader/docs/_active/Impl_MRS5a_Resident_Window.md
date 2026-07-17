# Impl — M-RS5a Resident Data Window (build tracker)

**Branch:** `feat/mrs5a-resident-window` (off `origin/dev`).
**Runs on:** SHADOW-WORKER only (separate Railway service; deploy via `railway up` from repo
root — NOT git auto-deploy). Main-Worker flags are OFF-LIMITS (parallel live-validation session).
**Design SSOT:** `Design_MRS5_Resident_Window.md` + `Plan_MRS5_2026-07-16.md` (5b DONE/armed; this is 5a).
**Discipline:** offline byte-identity gate BEFORE any live arm; everything flag-gated + reversible;
per-slot kill switch; nightly from-cold recompute stays the truth backstop. PAUSE for Kevin at every
arming decision.

---

## The load-bearing architectural insight (verified in code, 2026-07-17)

The waste is entirely in `shadow_manager.advance()` (src/shadow_manager.py:288): every poll it calls
`prepare_window()` → `prepare_strategy_window_df()` which re-preps `[cursor − 300 warmup bars, until]`
from the DB, then `engine.feed(df)` **skips every bar ≤ its cursor** (shadow_engine.py:147) and applies
only the 1–5 new rows. The whole warmup portion of that fresh prep is recomputed and thrown away.

**What the resident engine actually consumes from the df** (shadow_engine.py `_detect_cols`/`_build_inputs`,
lines 70–133):
- **OHLCV** (open/high/low/close/volume) — always.
- **user-pack columns** — interp cols (`required_interpreters` ∉ `_BUILTIN_INTERPS`), `trig_*` cols,
  intrabar-level indicator cols.
- **secondary-TF columns** — the `__<tf_label>` suffixed cols, via `secondary_tf_map`.

**What it does NOT consume from the df:** the built-in indicators
(`EMA*`, `MACD*`, `VWAP`, `RVOL`, `UTBOT*` — `_BUILTIN_INTERPS`, shadow_engine.py:37). The resident
engine computes those itself, incrementally, from the OHLCV it is fed — it already holds full indicator
state from bootstrap. So the resident frame **never needs to reproduce built-in indicator columns.**

⟹ The resident frame's only fidelity job: deliver, for each NEW bar, the byte-identical
**OHLCV + user-pack + secondary** columns a fresh `prepare_window` would have produced. This is what
splits the archetypes cleanly across phases:
- **Plain** (no user-packs, `sec_tfs == ()`): new rows need only OHLCV → **Phase 1** can be byte-identical.
- **User-pack / secondary-gated:** need incremental derived columns → **Phase 2**.

---

## Phase 1 — frame lifecycle (THIS phase)

New object `ResidentFrame` (per `EngineSlot`), gated by **`RORT_SHADOW_RESIDENT_FRAME`** (default OFF).

- **Capability gate (Phase 1 scope):** the resident-frame path activates only when the flag is ON AND
  the slot is *plain* (no enabled user-pack columns for its triggers AND `slot.sec_tfs == ()`). Every
  other slot falls through to today's full-prep `advance()` even with the flag ON. Phase 2 lifts this.
- **Bootstrap (cold):** unchanged — first `advance()` runs `prepare_window()` and we retain the returned
  df as the resident frame (trimmed to warmup depth). Engine warms as today → byte-identical left edge.
  *(Resampled-store settled-prefix bootstrap — M-RS2 P2, `RORT_RESAMPLED_STORE_SERVE_LIVE` already armed
  on Worker — is an optimization layered on after the plain lifecycle is green; see Open Qs.)*
- **Per-poll (warm):**
  1. one indexed delta read of SOURCE bars `(frame_tail, advance_bound]` (`bar_cache`, layer = 1Sec if
     `tf_seconds < 60` else 1Min — mirror `_source_bars_exist`, shadow_manager.py:133).
  2. resample the delta to the primary TF using the SAME primitive `prepare_data_with_indicators` uses
     (session filter + label/closed conventions must match exactly — see Open Q1); append complete
     (settled) buckets to the frame.
  3. `engine.feed(new_rows_only)` — the resident engine advances; built-ins from its own state.
  4. trim frame head to warmup depth.
- **INVARIANT (assert every poll):** `frame_tail == engine.last_bar_ts` after feed. On mismatch →
  drop the frame → cold re-prep for that slot (fail-safe to today's behavior). This is the
  restart/gap-skip/holiday-walk guard (Design risk row 4).
- **Kill switch:** `RORT_SHADOW_RESIDENT_FRAME=0` reverts a slot to full-prep instantly. Kept ≥ a month
  past fleet rollout.
- **Fallbacks that MUST drop-frame → cold re-prep (never silently diverge):** invariant mismatch;
  a revision to an already-consumed bar (Phase 2 REVISION RE-FEED — for Phase 1, any delta-read row
  whose ts ≤ frame_tail with changed values); fingerprint/classify change (already nulls the engine);
  empty/failed delta read is fail-open to full-prep, not a silent skip.

### Phase 1 offline gate
Extend `_shadow_manager_validate.py`: add a PATH gate that drives `advance()` with
`RORT_SHADOW_RESIDENT_FRAME=1` and asserts the resident-frame trades are byte-identical to the
from-cold DEEP reference over a multi-day window, for the **plain** archetype (e.g. 1Min plain).
Reuse the existing `keyed()`/DEEP-REF harness — add a MODE that flips the frame flag for the MANAGER
pass and compares to the same from-cold reference. GREEN = zero added/removed/changed.

---

## Phase 2 — incremental derived columns (NEXT)
- user-pack cols via each pack's `incremental_class` (same machinery the live shadow engines use).
- secondary-TF cols via the snapshot machinery, honoring the **1-period cross-TF shift** + coarse
  bucket boundaries.
- **REVISION RE-FEED:** settle-sweeper corrects an already-consumed bar → re-apply from that bar forward
  (or drop-frame → cold re-prep).
- Then the offline gate widens to ALL archetypes (10Sec RVOL, 30Sec sw123+4h gate, 1Min plain, coarse-gated TSLA).

## Phase 3 — comparator (both paths per poll, byte-diff new-row cols; `RORT_SHADOW_FRAME_COMPARE=1`).
## Phase 4 — rollout (canary → fleet; PAUSE for Kevin at each arm).

---

## Resolved seams (architecture map, 2026-07-17)
1. **Resample primitive parity — RESOLVED (this is the load-bearing byte-identity fact):**
   the full prep's primary OHLCV comes from `load_market_data → bar_cache.cached_load_market_data`
   (bar_cache.py:426). Order: read source layer (1Sec if `tf_seconds<60` else 1Min) for `[start,end]`
   → **`_filter_session`** (per-bar ET time-of-day mask, data_loader.py:48; window-independent) →
   **`resample_to_timeframe`** (data_loader.py:1142; agg first/max/min/last/sum, `dropna(open)`,
   pandas midnight-origin buckets → window-independent). ⟹ a delta read
   `load_market_data(sym, start=frame_tail, end=bound, tf, feed='sip', session, no_backfill=True)`
   then `df[df.index > frame_tail]` yields primary bars **byte-identical** to the full-window prep for
   every complete bucket past the tail, reading only the delta. **Phase 1 reuses `load_market_data`
   directly — no resample reimplementation.**
2. **incremental_class interface (Phase 2):** `pack.incremental_class(**params).update_bar(bar) -> dict`
   (instantiate unified_engine.py:952, call :1611); state codec in pack_spec.py:1090.
3. **Secondary-TF (Phase 2):** `_secondary_snapshot_load_extend` (services.py:979) supplies extended
   coarse `sec_df`; the 1-period cross-TF shift is services.py:437-444 (`index += period_offset`,
   `reindex(df.index, method='ffill')`). Shift stays at GATE-EVAL time (resampled_bar_store.py:40).
4. **Resampled-store read API (bootstrap opt):** `resampled_bar_store.read_store(sym, tf, session,
   start, end)` (resampled_bar_store.py:478); settled cutoff `_settled_cutoff_ts` (:458); serve wiring
   `ralph_engine.py:2062` + `services.py:423`. Layer this onto the settled-prefix bootstrap after the
   plain lifecycle is green.

## 🚨 MAJOR FINDING (2026-07-17) — the fleet has ZERO plain strategies
Config-level classification of all 28 eligible no-secondary slots (`_find_plain2.py`): **every one
drives off a USER PACK** (`ut_bot_v4`, `bollinger_bands`, `ema_pp_v3/v4`, `ema_stack_v2`,
`macd_histogram_v2`, `macd_line_v2`, `rsi_zones_2`, `rvol_v2`, …). There is **no "1Min plain"
strategy** anywhere in the monitored fleet — the design's plain archetype is hypothetical.

**Implications for the plan:**
- **Phase 1 alone delivers no real-slot cost win** — every real slot is user-pack, so nothing runs the
  resident-frame fast path until Phase 2 (incremental user-pack columns) lands. Phase 1's standalone
  deliverable is therefore the **frame-lifecycle machinery + the delta-read OHLCV byte-identity
  primitive** that Phase 2 is built directly on — NOT a shippable optimization by itself.
- **The trade-level end-to-end lifecycle validation moves to Phase 2** (on real user-pack archetypes),
  because there is no plain strategy to run it on. `_advance_resident_frame` + the frame + the invariant
  are unchanged between phases; Phase 2 only adds per-new-bar derived-column computation.
- **Revised archetype list** (real, all user-pack): 10Sec `ut_bot_v4` (263/268/270/273), 10Sec
  `ema_pp_v4`/`ema_stack_v2`/`macd_line_v2`/`rvol_v2`/`rsi_zones_2`/`bollinger_bands` (278–290),
  15Sec (308/321/334–337), 1Min `ut_bot_v4` (269 SPY / 265 TSLA), 5Min (266). Coarse-gated/secondary
  archetypes (sw123 + 4h gate) are the Phase-2 secondary-column path.

**Lesson (plain-detection):** `shadow_engine._detect_cols` puts every required trigger's `trig_*` column
into `_user_trig_cols`, including BUILT-IN triggers — but `process_bar` recomputes built-ins from OHLCV
and ignores those df values ("Don't override built-in", unified_engine.py:3540-3542). So the correct
frame-eligibility signal is **no `_user_pack_*` markers in `engine.engine.indicators.required` and no
secondary TFs** — NOT the presence of `trig_*` columns. (Fixed in `advance()`'s eligibility block.)

## Phase 1 completion criteria (revised)
1. ✅ Code implemented; **flag-OFF byte-identical** to today (every new branch guarded by `RESIDENT_FRAME`
   default OFF + `slot.frame is not None`). Compiles.
2. ✅ Eligibility detector correct (`_user_pack_*` markers + secondary), proven by the gate correctly
   reporting RESIDENT-INVALID on user-pack sid 269 (no vacuous pass).
3. ✅ **Delta-read OHLCV byte-identity GREEN** across all three resample regimes (1Min no-resample,
   5Min 1Min→5Min, 10Sec 1Sec→10Sec), SPY + TSLA — `_frame_data_parity.py`. The resident delta read
   feeds byte-identical new bars to the full-prep path each poll. NOTE: the correct reference is
   full-prep-PER-POLL, not a single all-complete read — both paths keep the partial right-edge bucket
   at each poll's `until` (the nightly from-cold recompute is the completeness backstop); comparing the
   accumulated delta walk to a single full read gives FALSE mismatches on resample TFs (test v1 bug).
4. Offline `_shadow_manager_validate.py` gained a `VALIDATE_RESIDENT_FRAME=1` PATH mode with a
   RESIDENT-INVALID guard — ready to certify trade-level byte-identity once a slot is frame-eligible
   (Phase 2 makes real slots eligible).

## 🎯 PHASE 2 REFRAMED (2026-07-17) — the engine is already self-sufficient for user-pack TRIGGERS
**Trace result:** the resident engine is NOT OHLCV-limited for user packs. `IncrementalIndicatorEngine.
_update_indicators` **drives each pack's `incremental_class.update_bar(bar)` internally** (unified_engine.py:
1609-1613) and merges the returned indicator values + trigger booleans into `state.current`, which the
`TriggerEvaluator` then reads. **All 10 fleet packs have `incremental_class=True`.** (`process_bar`:3523
also overrides user-pack *indicator levels* with the df batch values when present, and :3536-3542 merges
user-pack interp/trigger states from the df only `if not already computed` — so with OHLCV-only the
internal incremental values are used.)

**Probe (`_frame_userpack_probe.py`): force an OHLCV-only frame on a user-pack slot, compare vs full-prep
(both bootstrap identically):**
- sid 263 (ut_bot_v4, 10Sec, trigger-only, no secondary) on a **SETTLED** window (2026-07-15):
  **FULL=137 FRAME=137, byte-identical (added/removed/changed = 0)**, full-window AND settled-only.
- ⚠️ **Live-edge runs are non-deterministic** — anchoring on the latest trade puts the window on the
  unsettled edge where `bar_cache` is revised BETWEEN the two sequential manager runs → phantom 1-trade
  diffs. **Always probe on a settled pinned window (`PIN_T0`/`PIN_TEND`).** (This is why the first probe
  runs showed a spurious +1 trade.)

**⟹ OHLCV-only (existing Phase-1 code) already covers user-pack TRIGGER strategies.** Phase 2 is mostly
about **broadening the eligibility detector**, not building a derived-column pipeline. Reframed scope:
1. **Broaden eligibility** to slots that are (no secondary) AND (every required user pack has an
   `incremental_class`) AND (no user-pack interp used as a GATE/confluence leg — see #2). Verify
   byte-identity per PACK archetype first (ema_stack_v2/macd_line_v2/rvol_v2/bollinger/rsi_zones_2
   probe RUNNING — rvol_v2 is the memory-flagged forming-bar risk to watch).
2. **Interp-as-GATE case:** if a strategy gates on a user-pack interp (confluence leg), the interp STATE
   must land in `interps` for the confluence record. Trace whether the resident engine builds user-pack
   interp states internally (evaluate_bar_for_backtest) or needs the df `__`/interp column. If needed →
   supply it (or exclude these from eligibility for now).
3. **Secondary-TF strategies:** the `__<tf>` columns still come from the df → need
   `_secondary_snapshot_load_extend` + 1-period shift (services.py:437-444). This is the only genuine
   "derived column" build left, and it's scoped to secondary-gated slots.
4. **REVISION RE-FEED** (settle-sweeper corrections to consumed bars) — needed regardless.

Gate everything with `_shadow_manager_validate.py VALIDATE_RESIDENT_FRAME=1` on the real archetypes,
ALWAYS on settled windows (`VALIDATE_T0`/`VALIDATE_TEND`), never the live edge.

## ✅ PHASE 2b VALIDATED (2026-07-17) — user-pack TRIGGER strategies covered (28/69 slots)
Broadened the `advance()` eligibility detector: a slot is frame-eligible when
`(not slot.sec_tfs)` AND (every `_user_pack_*` marker's pack has an `incremental_class`) AND
(no confluence gate leg references a non-builtin interpreter). Fleet split: **28 no-secondary
user-pack trigger slots** (now eligible), **0** user-pack-interp-gate slots, **41** secondary-gated
(Phase 2c). Two bugs fixed along the way:
- **`sec_tf_map` vs `slot.sec_tfs`:** `get_secondary_tf_map(df)` matches ANY `__` and misreads
  `__`-PREFIXED trigger columns (`__utv4_bull_flip`) as bogus secondaries → wrongly disqualified
  every trigger slot. Fixed to use the authoritative `slot.sec_tfs` (from `classify`).
- The plain-detector counted built-in trigger cols (fixed earlier in Phase 1).

**Gate result — the correct comparison is frame-manager vs FULL-PREP manager** (the production path),
not vs a shallow from-cold reference:
- `_frame_userpack_probe.py` on SETTLED window 2026-07-15, 6 pack archetypes (ut_bot_v4, ema_stack_v2,
  macd_line_v2, **rvol_v2**, bollinger_bands, rsi_zones_2): **frame == full-prep byte-identical**
  (added/removed/changed = 0), full-window AND settled-only, on BAR-ALIGNED **and** PARTIAL-BAR
  (`EDGE_OFFSET_S=7`, the live cadence) edges. 269 (1Min) also frame==from-cold directly.
- **`DEEP_REF=1` from-cold confirmation (post-close 2026-07-17, settled window):** 269 (1Min)
  **byte-identical** to deep from-cold (A=21 M=21, 0/0/0). 263/290 (10Sec) STILL diverge from deep
  from-cold (263: 1/1/1; 290: 3/3/**119** float diffs) — BUT this is a **pre-existing sub-minute
  MANAGER-vs-from-cold property, NOT the frame**: the frame reproduces the full-prep manager's EXACT
  divergence (same 1/1/1 and 3/3/119 as the full-prep DEEP_REF run), and the probe proves frame ==
  full-prep byte-identical (6 archetypes). So DEEP_REF=1 does NOT make 10Sec green because the
  RESIDENT MANAGER approach itself (bootstrap + poll re-feed) has a small sub-minute divergence from a
  single from-cold pass (the known append/forming-bar sub-minute class) — inherited from M-RS4, present
  in production today, unchanged by M-RS5a.
- **⟹ The correct M-RS5a acceptance = frame == FULL-PREP manager (probe, byte-identical), NOT frame ==
  from-cold.** The frame adds ZERO divergence over the production path. **CONFIRMED: full-prep manager
  DEEP_REF=1 shows the IDENTICAL divergence (263: 1/1/1, 290: 3/3/119 — exactly the frame's numbers)** ⟹
  frame == full-prep proven via the validator too, not just the probe. For the arming decision: the frame
  is as-correct-as the current shadow-worker; the sub-minute from-cold gap is a separate, pre-existing
  concern (nightly from-cold recompute remains the backstop).

**Remaining Phase 2c:** secondary-TF columns (41 slots) via `_secondary_snapshot_load_extend` + 1-period
shift; REVISION RE-FEED. Everything still flag OFF; nothing armed.

## PHASE 2c DESIGN (2026-07-17) — secondary-TF columns for the 41 secondary-gated slots
**Confirmed:** the engine does NOT compute secondary-TF state internally (no coarse-TF resampling) —
probe of secondary slot 267 OHLCV-only = **0 trades vs 91** (the secondary gate blocks everything).
So secondary `__<tf>` columns genuinely come from the df. Unlike user packs, these MUST be supplied.

**Machinery** (services.py:408-449, prepare_data_with_indicators secondary block): per secondary TF —
coarse OHLCV (from `_secondary_snapshot_load_extend` cached+short-extend, else resample) → `run_all_
indicators` + `run_indicators_for_group` + `run_all_interpreters` → **1-period shift** (services.py:438-444:
`shifted.index = sec_df[interp].index + period_offset; df[f"{interp}__{tflabel}"] = shifted.reindex(
df.index, method='ffill')`) + a `_spec_` unshifted copy (heatmap only). `_secondary_snapshot_load_extend`
(services.py:979) is the cheap coarse path: loads `secondary_snapshot_b64` + a short `load_market_data`
from the last cached coarse boundary → `until`; byte-identical to full resample by construction.

**Design — reuse the cheap secondary pipeline per poll (byte-identical, window-independent for a fixed
`until`, exactly like the primary delta-read):**
1. Factor the inline secondary block (services.py:408-449) into a reusable helper
   `compute_secondary_columns(strat, sec_tfs, coarse_inject, target_index, scoped_groups) -> {col: Series}`;
   call it from BOTH prepare_data_with_indicators (refactor = byte-identical, guarded by parity gates) AND
   the resident frame. DRY ⟹ byte-identity by construction.
2. In `_advance_resident_frame`, for a secondary slot: delta-read new primary OHLCV (as today) →
   `coarse = _secondary_snapshot_load_extend(strat, sec_tfs, until, primary_tf, session, feed,
   no_backfill=True)` → `sec_cols = compute_secondary_columns(..., new_rows.index)` → attach `__<tf>` cols
   to `new_rows` → `engine.feed(new_rows, sec_tf_map)` with the real sec_tf_map (the `__<tf>` names).
   The frame stores OHLCV + `__<tf>` columns now (extend ResidentFrame.OHLCV → a per-slot column set).
3. Eligibility: broaden to secondary slots whose `_secondary_snapshot_load_extend` returns non-None
   (valid `secondary_snapshot_b64`); on None → stay full-prep (fail-safe). No user-pack-interp-gate
   (fleet has 0). Bootstrap (full-prep) persists the snapshot, so it exists by the first warm poll.

**Cost/refresh caveat:** the snapshot `last_ts` is from bootstrap; the extend's recent load grows as
`until` advances over a session (from last cached coarse bar → until). Still FAR cheaper than the full
PRIMARY warmup re-prep (the ~1000:1 waste, eliminated by the resident primary). v1 = accept the growing
extend + periodic drop-frame→re-bootstrap (re-persists the snapshot, bounding the gap). v2 (optional) =
maintain a RESIDENT coarse snapshot that advances per poll (append closed coarse bars in-frame). Ship v1.

**Verify:** probe secondary slots (267 + a coarse-gated TSLA) frame-vs-full-prep on SETTLED windows,
aligned + partial-bar edges; must be byte-identical before broadening eligibility. Dependency to check
at impl: do the 41 secondary slots actually carry a valid `secondary_snapshot_b64`? (if not, they stay
full-prep — safe, but no win until the append/bootstrap lane warms the snapshot.)

## PHASE 2c BUILD — Step 1 DONE (2026-07-17): compute_secondary_columns helper
Factored the inline secondary block (services.py:408-449) into `services.compute_secondary_columns(
strat, secondary_tfs, primary_df, session, scoped_groups, secondary_tf_dfs=None, target_index=None)`
returning `{col: Series}`; `prepare_data_with_indicators` now calls it. **Byte-identical verified**
(`_verify_refactor.py`, git-stash before/after): sid 267 (2m sec) + 271 (5m sec) deep-window trade
hashes IDENTICAL pre/post refactor. Compiles.

**Step 2 approach — RESOLVED = C (snapshot per poll). Approach B (resample frame primary→coarse) is
WRONG:** verified all secondary slots (267/271/272/275…) run with `RORT_SECONDARY_TF_SNAPSHOT=1` and have
a VALID `secondary_snapshot_b64`, so the full-prep bootstraps with a SHALLOW primary (~WARMUP_BARS) +
snapshot-served coarse. The frame's shallow primary has no coarse warmup ⟹ can't resample it. Use the
snapshot. Good news: **ResidentFrame needs NO change (stays OHLCV, shallow);** secondary cols are computed
per poll and attached to `new_rows`, not stored in the frame.

Per poll in `_advance_resident_frame`, for a secondary slot (after the OHLCV delta read):
1. `coarse = _secondary_snapshot_load_extend(_eng_strat, sec_tfs, until_dt, slot.timeframe, session, "sip",
   slot.bt_model, no_backfill=READ_ONLY_BARS)` — cheap cached coarse + short extend. None → drop-frame/full-prep.
2. `sec_cols = compute_secondary_columns(_eng_strat, sec_tfs, new_rows, session, scoped_groups,
   secondary_tf_dfs=coarse, target_index=new_rows.index)`; attach each to `new_rows`.
3. `engine.feed(new_rows, get_secondary_tf_map(new_rows) or None)`.
Win: eliminates the shallow-primary DB re-read + primary user-pack recompute; keeps only the short
snapshot extend. Optimization (later): resident coarse snapshot advanced in-frame (no per-poll extend).

**Delicate byte-identity pieces that MUST match the batch path (services.prepare_strategy_window_df) —
this is why step 2 is a focused build, not a quick add:**
- **scoped_groups**: replicate prepare's `RORT_SCOPE_CONFLUENCE_GROUPS` scoping
  (`resolve_required_confluence_groups(strat, get_enabled_groups(load_confluence_groups()))`, services.py:
  322-337/385-387) — else the frame produces a different `__<tf>` column set than full-prep.
- **1-minute-gate normalization**: full-prep applies `normalize_1min_secondary_gate` inside
  prepare_strategy_window_df; the frame must use the same `_eng_strat` for BOTH load_extend and
  compute_secondary_columns (advance() already builds `_eng_strat` for the engine — pass it through).
- **snapshot blob**: confirmed present in the slot's loaded strat (has_blob=True), so load_extend finds it.
- Eligibility: broaden to secondary slots WITH a valid snapshot (`_has_valid_secondary_snapshot` /
  load_extend non-None); else stay full-prep (fail-safe). ⚠️ **The `_gate_userpack` guard must be
  REFINED for secondary slots:** a secondary leg (`2m-UT_BOT_V4-BULL`) references a non-builtin interp
  but is HANDLED (compute_secondary_columns emits `UT_BOT_V4__2m` — confirmed in sec_cols output), so it
  must NOT disqualify. Only a PRIMARY-TF gate on a user-pack interp is the open question (needs the
  primary interp STATE in confluence_records — untested whether the engine computes user-pack interp
  states internally; fleet B-class = 0 for no-secondary, but a secondary slot COULD also carry a primary
  user-pack-interp leg). NEXT-ITER investigation: (a) scan the 41 secondary slots for primary-TF
  user-pack-interp legs; (b) if any, test whether the engine emits the primary user-pack interp state
  internally; refine the guard to flag only primary-TF non-builtin-interp legs. The probe is the backstop.
- **PROBE `_frame_userpack_probe.py` on 267/271 (2m/5m) SETTLED, aligned + partial-bar edges, byte-identical
  BEFORE broadening.** (Probe already forces a frame regardless of eligibility — extend it to attach
  secondary cols on the frame path, mirroring _advance_resident_frame.)

## PHASE 2c BUILD — Step 2 DONE + REVISION RE-FEED not needed (2026-07-17)
**Secondary-column frame extension IMPLEMENTED** (shadow_manager.py): `_advance_resident_frame` branches
on `slot.sec_tfs` → `_attach_secondary_columns` (normalized `_eng_strat`, sec_tfs derived like prepare,
coarse via `_secondary_snapshot_load_extend` read-only, RORT_SCOPE_CONFLUENCE_GROUPS scoping,
`compute_secondary_columns(secondary_tf_dfs=coarse, target_index=new_rows.index)`, attach, feed). Eligibility
broadened: secondary slots with a valid snapshot are eligible; `_gate_userpack` refined to flag only
PRIMARY-`1M-` user-pack-interp legs (sids 136/329 stay full-prep). ResidentFrame UNCHANGED (OHLCV-only;
secondary cols live on new_rows for the feed, not stored). Probe fix: default `RORT_SECONDARY_TF_SNAPSHOT=1`
(match shadow-worker) — with snapshot OFF, full-prep deep-resamples while the frame uses the snapshot →
mismatch + frame drop. **Probe GREEN (snapshot ON, settled window): 267(2m) 91=91, 271(5m) 55=55,
frame_survived=True, byte-identical full+settled.** Broader probe (267/272/327/340/312/325, offset edges)
RUNNING.

**REVISION RE-FEED — NOT NEEDED (and would be WRONG to add).** M-RS5a's acceptance = frame == FULL-PREP
manager (the production path), NOT frame == canonical. Both paths feed only bars `> cursor` (read at the
same poll time → identical) and both IGNORE revisions to already-consumed bars (full-prep re-reads them but
the engine skips `index <= last_bar_ts`; the frame doesn't re-read them) — so the frame keeps the exact same
consumed/partial values full-prep keeps. Adding re-feed would make the frame BETTER than full-prep = CHANGE
outputs, violating M-RS5's "changes cost, not outputs" mandate. The sub-minute revision jitter is the
pre-existing MANAGER-vs-canonical gap (affects full-prep equally; nightly from-cold recompute is the
backstop). ⟹ Phase 2c is COMPLETE once the broader secondary probe is green.

## Progress log
- 2026-07-17: Orientation complete; branch cut. **Phase 1 code SHIPPED (uncommitted, flag OFF):**
  `ResidentFrame` class, `EngineSlot.frame`/`frame_eligible` (+ fingerprint invalidation),
  `advance()` fast-path branch + bootstrap frame-build + eligibility detector, `_advance_resident_frame`
  (delta read via `load_market_data`, feed, append, trim, frame↔engine invariant, drop-frame fallback),
  kill switch `RORT_SHADOW_RESIDENT_FRAME`. Offline gate extended (`VALIDATE_RESIDENT_FRAME` + RESIDENT-
  INVALID guard). Verified flag-OFF byte-identical. Fixed the plain-detector bug (built-in trig cols).
  **Found: fleet is 100% user-pack (0 plain)** → Phase 1 = lifecycle+primitive only; Phase 2 is the real
  win. Delta-read byte-identity test running. Scratch: `_frame_data_parity.py` (keep), `_dbg_frame.py`/
  `_find_plain2.py`/scratchpad `find_plain.py` (throwaway).
