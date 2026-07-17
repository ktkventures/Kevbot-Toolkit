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

## Phase 2 shape (next — where all real value is)
Extend the resident frame to carry user-pack (+ secondary) columns for each new bar:
- Per new bar, produce the user-pack interp/trigger/indicator columns the batch prep would have
  (`pack.incremental_class.update_bar(bar)` — note the resident engine's `IncrementalIndicatorEngine`
  may already drive user-pack indicator engines internally; must trace whether the resident path today
  gets user-pack INTERP/TRIGGER states from the df vs computes them, unified_engine.py ~1604-1616 vs
  process_bar 3536-3542 — resolve before coding).
- Secondary `__<tf>` columns via `_secondary_snapshot_load_extend` honoring the 1-period shift.
- REVISION RE-FEED. Then run `_shadow_manager_validate.py VALIDATE_RESIDENT_FRAME=1` on the real
  archetypes — that is the trade-level byte-identity gate, and it becomes non-vacuous once slots are
  frame-eligible.

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
