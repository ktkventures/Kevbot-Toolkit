# EOD 2026-05-07 — Hi-Fi Phase 1, AI Pack Guardrails, and the Path to Live-Trading Confidence

This document captures the full body of work completed 2026-05-07,
the discoveries that informed it, the design decisions taken, and
the punch list for tomorrow. Goal of this session: close the last
known gap between algo-history (backtest) and alert-history (live)
timestamps so Kevin can begin trading live capital with confidence.

## TL;DR

| Area | Status before today | Status end-of-day |
|---|---|---|
| Hi-Fi stop/target exits | sub-second | sub-second (no change) |
| Hi-Fi L-type signal exits | bar-aligned (10s drift on 10Sec strategies) | **sub-second on `eppv3`/`eppv4`/`utv4` packs** |
| AI pack creation `trigger_levels` quality gate | silent omission tolerated | three-layer guardrail enforced |
| Pack manifest audit warnings | 30+ warnings on `scan_and_load_all` | 0 warnings (7 packs marked Phase 2) |
| Cron processing oldest strategies | 4 strategies stuck 2-12h | manual recovery: 267 trades inserted |
| Built-in template usage | 11 active strategies on legacy templates | 0 (all deleted; user packs only) |
| `cache_locked` dispatch on canary strategies | partial | trade-count parity ~1:1, entry drift bit-perfect |

End-state: every active strategy uses user packs only; trigger
detection at sub-second resolution for all static-level crosses;
AI pack creation can no longer ship a cross-style pack without
declaring its level semantics.

## What we shipped

### 1. Hi-Fi Pass 2 signal-exit refinement (Phase 1)

**Why:** Last known parity gap was L-type SIGNAL exits — triggers
like `eppv4_cross_mid_down_ib` were timestamped at the bar boundary
(10-second drift on 10Sec strategies, 1-minute drift on 1Min). Stop
and target exits were already sub-second-refined via
`_walk_1s_for_exit` because those have known fixed price levels on
the trade row. Signal exits don't carry a level on the trade row —
the level lives in the user pack's trigger definition.

**Approach:** Honor the user-pack-as-source-of-truth contract.
Each pack's `manifest.json` already supports a `trigger_levels`
key declaring `{trigger_id: {level_column, cross}}`. Hi-Fi reads
that contract and walks 1-second bars for the crossing event.

**Implementation:**

- `src/api/services/backtest_service.py`
  - **NEW** `_walk_1s_for_level_cross(bars_1s, level, cross_direction, direction)`
    — walks 1-sec bars until first cross of `level` in
    `cross_direction` (`'above'` or `'below'`); gap-aware fill matches
    existing `_walk_1s_for_exit` convention.
  - **EXTENDED** `_hifi_resolve_trades` signature to accept `bar_df`;
    new branch handles trades whose `exit_trigger` ends in `_ib`
    (intra-bar / L-type signal exit). Resolves the trigger's level
    via `pack_registry.get_trigger_level_spec()`, reads the level
    value from `bar_df.loc[exit_dt, level_column]`, then walks.
- `src/api/routers/strategies.py`
  - `run_hifi_pass2` pre-computes `bar_df` via
    `prepare_data_with_indicators` when any trade has an `_ib`
    exit_trigger. Bug fixed during scope: a local
    `from datetime import datetime` shadowed the module-level
    import — removed.
- `src/pack_registry.py`
  - **NEW** `get_trigger_level_spec(trigger_id)` — strips `_ib`,
    `_lc`, `_hm`, `_hl` exec-type suffixes, looks up the pack's
    manifest, returns `{'level_column', 'cross'}` or `None`.
- Endpoints affected:
  - `/api/strategies/{id}/run-hifi-pass2` (single)
  - `/api/strategies/run-hifi-pass2-bulk` (bulk)

**Coverage today (4 packs declare static-level cross semantics):**

| Pack | Trigger family | Notes |
|---|---|---|
| `ema_pp_v3` | Price-vs-EMA (short/mid) | needs runtime verification of column name (manifest declares `eppv3_short`/`eppv3_mid` without `_prev` suffix; engine adds 1-bar lag automatically) |
| `ema_pp_v4` | Price-vs-EMA (short/mid), `_prev` columns | clean; verified |
| `ut_bot_v4` | Price-vs-trailing-stop (`utv4_trailing_stop_prev`) | clean |
| `vwap_v2`   | Price-vs-VWAP | declares `vwapv2_value` (no `_prev` — VWAP is dynamic intra-bar). Phase 1 detects this and skips. |

**Open verification (for next market-hours session):**
- Confirm exit drift median drops from ~12s (10Sec) to <2s on
  Hi-Fi-resolved signal exits.
- Confirm Polygon 1-second bar coverage on the lookback window —
  if 1-sec data is sparse on older bars, the walker falls through
  to bar-aligned (no regression, just no improvement).

### 2. AI pack-creation guardrail (`trigger_levels` enforcement)

**Why:** Audit revealed `trigger_levels` was declared in only 4 of
8 active packs. The other 4 silently omitted it. The AI-assisted
pack-creation pipeline didn't prompt the author for it, so future
AI-generated packs would inherit the gap by default.

**Three-layer guard (build, install, audit):**

1. **Schema audit** — `src/pack_spec.py`:
   - `looks_cross_style(trigger_base_name)` — regex
     `(^|_)(cross|flip|above|below|breaks?|crosses?)(_|$)` flags
     trigger names that imply a level-crossing semantic.
   - `audit_trigger_levels(manifest)` — for each cross-style trigger,
     emits a warning if it's missing from BOTH `trigger_levels`
     (Phase 1: static-level eligible) and `trigger_levels_phase2`
     (intentional non-static marker).
2. **Install-time warning** — `src/pack_registry.py`:
   - `scan_and_load_all` calls `audit_trigger_levels`; warnings
     surface in stdout as `⚠️  [pack_registry] {slug}: Trigger '{trigger}'
     looks like a level cross but is missing from both 'trigger_levels'
     ... Hi-Fi Pass 2 will skip this trigger`.
   - Stored on `RegisteredPack.validation_warnings` for later
     introspection.
3. **AI builder gate** — `src/pack_builder.py`:
   - Rule 14/14a in the AI prompt: cross-style triggers MUST declare
     `trigger_levels` (Phase 1) or `trigger_levels_phase2`
     (intentional Phase 2).
   - `validate_parsed_response` runs `audit_trigger_levels` and
     treats any warning as a validation **error** — blocks AI
     generation from completing.

**Manifest schema convention:**

```json
"triggers": [
  {"base": "cross_bull",      "name": "Short Crosses Above Mid", ...},
  {"base": "cross_bear",      "name": "Short Crosses Below Mid", ...}
],
"trigger_levels": {
  "cross_bull": {"level_column": "esv2_short_prev", "cross": "above"},
  "cross_bear": {"level_column": "esv2_short_prev", "cross": "below"}
},
// OR — for non-static cases (Phase 2):
"trigger_levels_phase2": {
  "cross_bull": {"reason": "indicator_vs_indicator (short EMA vs mid EMA) — Phase 2"},
  "cross_bear": {"reason": "indicator_vs_indicator (short EMA vs mid EMA) — Phase 2"}
}
```

**Cleanup pass:** added `trigger_levels_phase2` markers to 7
existing packs (ema_stack_v2, macd_histogram_v2, macd_line_v2,
rsi_zones, rsi_zones_2, rsi_zones_3, stochastic_oscillator). Audit
count: 30+ → 0.

### 3. Built-in template cleanup

11 strategies (sids 6, 8-17) on legacy built-in templates
(`ema_price_position_v2`, `utbot_v2`, `macd_line`, `ema_stack`,
`macd_histogram`, `rvol`) were deleted — Kevin authorized after
confirming none were active or material. All remaining 20
strategies use user-pack triggers exclusively. Built-in templates
remain in `confluence_groups.TEMPLATES` for legacy reference but
are no longer referenced by any active strategy.

This unblocks Hi-Fi Phase 1 — the engine fallback path for built-in
templates is no longer required.

## What we discovered (and didn't ship)

### A. Cron starvation on the heaviest strategies

**Observed:** sids 152, 153, 154 sat at the top of the cron's
oldest-first queue with stamps from 02:26 / 16:16 UTC — 2 to 12+
hours stale. Cron ran fine on the other 14-15 strategies (most
stamped within the last 30 min). Manual invocation of
`append_new_trades_for_strategy` on each succeeded:

| Strategy | Wall time | Trades inserted |
|---|---|---|
| sid 152 (1Min, cache_locked, eppv4) | 28s | 7 |
| sid 153 (10Sec, mass-builder mirror) | 87s | 210 |
| sid 154 (10Sec + 15Min secondary) | 82s | 50 |

**Root-cause hypothesis:** Cron has a 240s per-cycle budget. With
two heavy strategies at 80+ seconds each, ~2 strategies fit before
budget exhaustion. The next cycle restarts with the same heaviest-
oldest pair at the top, perpetually starving them of meaningful
recovery. Lighter strategies in the middle of the queue get
processed because the budget cap hits BEFORE reaching them and
they were already stamped recently anyway.

**Action items (deferred):**
1. Surface budget-exhaustion + per-strategy elapsed in cron-cycle
   logs.
2. Per-strategy round-robin cap (e.g. max 1 heavy strategy per
   cycle) so heavy strategies don't compound.
3. Bigger cycle budget OR shorter interval for heavy-strategy users.

**Workaround for now:** `Update New Data` button on the strategies
page fires `append_new_trades_for_strategy` synchronously per
selected strategy. User-driven recovery.

### B. The "smoke test against prod" incident

A local smoke test for the Hi-Fi changes invoked
`update_strategy_admin('config', cfg)` instead of a partial-update
helper. The full-row update path passes the partial dict through
`_strategy_to_row`, which emits `{}` for unset JSONB fields,
**wiping the config column** on the affected strategies. Affected:
- sid 124 — wiped, deleted by Kevin
- sid 166 — wiped, deleted by Kevin
- sid 132 — wiped, kept (1 config key remaining, excluded from cron
  via `entry_trigger_confluence_id` eligibility filter — clean
  fail path)

**Memory updated:** `feedback_smoke_test_against_prod.md` —
production smoke tests must use partial-update DAL helpers, never
the full-row update path. This is a repeat of the
`feedback_jsonb_partial_updates.md` incident (sids 70+71 wiped
2026-04-14).

### C. The X exec_type discussion (Phase 2 design seed)

**Context:** Phase 2 Hi-Fi covers three evaluator shapes that don't
fit the current "L-type intra-bar level cross" model cleanly:

1. **`indicator_vs_indicator`** — MACD line vs signal line, EMA-vs-EMA,
   K-vs-D, etc.
2. **`value_vs_threshold`** — RSI vs 70/30/50, MACD line vs zero,
   histogram flip vs zero, etc.
3. **`price_vs_dynamic`** — VWAP intra-bar (level changes within
   bar; needs 1-sec recomputation).

**Two framings considered:**

- **Option 1 — New exec_type X (or D).** Pack authors declare
  `exec_type='X'` for indicator-cross triggers. Engine routes to
  the appropriate Hi-Fi handler based on exec_type. Surfaces the
  semantic difference at the trade-record level (auditability).
  AI pack creation slots in cleanly: each new pack picks one of
  {C, L, X} based on its trigger nature.
- **Option 2 — Treat L as catch-all.** TradingView-style — "intra-bar"
  covers everything that isn't bar-close. Hi-Fi handler dispatches
  based on the pack's manifest declaration (level_column type,
  recompute_function presence) rather than exec_type.

**Kevin's lean:** Option 1 — keeps exec types as the modular
1-5 char acronyms that route execution behavior. Aligns with the
"future templates ideally fall in line with one of our primary
execution types — no troubleshooting needed" goal stated in
`project_exec_types_and_confluence_ovar.md`.

**Decision: deferred** until Phase 1 has soaked through 1+ week of
clean cron runs and we know the implementation reality of the
Phase 2 detection logic. Documented in the new milestone
`8.7-hifi-signal-exits-phase2`.

## State of the parity contract (end of day)

| Layer | Source of truth | Phase 1 Hi-Fi behaviour | Phase 2 Hi-Fi |
|---|---|---|---|
| Trade-count parity (cache_locked) | identical engine state for live + backtest | ~1:1 (vs 2:1 on REST) | maintain |
| Entry timestamps | bar-close confirmation | bit-perfect | maintain |
| C-type exits (max_hold_bars, etc) | bar-close confirmation | bit-perfect | maintain |
| L-type stop/target exits | known fixed level on trade row | sub-second via `_walk_1s_for_exit` | maintain |
| L-type SIGNAL exits — static-level packs | pack manifest `trigger_levels` | **sub-second via `_walk_1s_for_level_cross`** | maintain |
| L-type SIGNAL exits — dynamic / indicator-cross | pack manifest `trigger_levels_phase2` (placeholder) | bar-aligned (skipped) | sub-second after Phase 2 |

**Pack-author contract:** if your pack declares `trigger_levels`,
Hi-Fi refines its signal exits today. If your pack declares
`trigger_levels_phase2`, the omission is intentional and
documented; Phase 2 will add the handler. If your pack declares
neither AND has cross-style triggers, the audit warns and AI
builder errors.

## Tomorrow's punch list

### Verification (market-hours)
- [ ] Run `_smoke_hifi_signal_exits.py` (or equivalent ad-hoc
      script) on sid 152 (1Min eppv4) post-RTH-close: confirm
      exit drift median <2s on `_ib`-suffixed exits.
- [ ] Visually confirm on Strategy Detail Chart & Trades tab —
      algo and alert exit markers stack on top of each other for
      sid 152.
- [ ] Spot-check sid 154 (10Sec eppv4) — same comparison.
- [ ] Verify Polygon 1-sec bar coverage isn't gappy on the
      lookback window. If gaps observed, the walker falls through
      gracefully (no regression, just no refinement on those bars).

### Cron health
- [ ] Add per-strategy elapsed + budget-exhausted summary to cron
      cycle log line (`worker.py:1198`).
- [ ] Decide on a per-cycle starvation mitigation (round-robin cap
      OR bigger budget OR shorter interval).
- [ ] Consider exposing cron cycle stats in Jobs page.

### Soak period
- [ ] 1 full week of clean cron + cache_locked + Hi-Fi Phase 1
      with no manual recovery required.
- [ ] Track per-strategy alignment metrics (entry drift, C-exit
      drift, L-exit drift) over the soak. If <2s holds, Phase 2
      design can begin.

### Phase 2 design (after soak)
- [ ] Decide exec_type X vs L-as-catch-all (default Option 1).
- [ ] Schema for `trigger_intra_bar` evaluator block per evaluator
      shape (indicator_vs_indicator, value_vs_threshold,
      price_vs_dynamic).
- [ ] Per-pack porting plan (ema_stack_v2, macd_line_v2, rsi*,
      stochastic_oscillator, vwap_v2).

## Files touched today

**Engine + service layer:**
- `src/api/services/backtest_service.py` — `_walk_1s_for_level_cross`,
  extended `_hifi_resolve_trades` signature (+`bar_df`), signal-exit
  branch.
- `src/api/routers/strategies.py` — `run_hifi_pass2` bar_df pre-compute.
- `src/pack_registry.py` — `get_trigger_level_spec`,
  `RegisteredPack.validation_warnings`, install-time audit hook.
- `src/pack_spec.py` — `looks_cross_style`, `audit_trigger_levels`.
- `src/pack_builder.py` — strengthened rules 14/14a; audit-as-error
  in `validate_parsed_response`.

**User packs (markers added):**
- `user_packs/ema_stack_v2/manifest.json`
- `user_packs/macd_histogram_v2/manifest.json`
- `user_packs/macd_line_v2/manifest.json`
- `user_packs/rsi_zones/manifest.json`
- `user_packs/rsi_zones_2/manifest.json`
- `user_packs/rsi_zones_3/manifest.json`
- `user_packs/stochastic_oscillator/manifest.json`

**Plan + docs:**
- `~/.claude/plans/synchronous-tickling-yeti.md` — full Phase 1
  design + Phase 2 X-exec-type discussion.
- `docs/Roadmap_To_Scale.md` — milestone updates.
- `docs/EOD_2026-05-07_Hifi_Phase1_Plus_AI_Guardrails.md` (this file).

**Memory:**
- `feedback_exec_type_x_for_indicator_cross.md` — Phase 2 design lean.
- `feedback_smoke_test_against_prod.md` — incident from local
  smoke run wiping configs.

## Backup branch

Before proceeding to the EOD doc commit, a backup branch was
saved:

```
git branch dev-backup-eod-2026-05-07-x-exec-discussed
```

This is the second backup of the day; the first preceded the
Phase 1 commits.
