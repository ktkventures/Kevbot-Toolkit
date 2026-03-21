# Phase 31F: HiFi Backtest & Fidelity System — Implementation Spec

**Status:** PLANNING — Playbook finalized, ready for implementation
**Date:** 2026-03-20 (Updated)
**Depends on:** Phase 31A-31E (Polygon data provider — COMPLETE)
**Source of truth:** `docs/Execution_and_Fidelity_Playbook.md`
**Branch:** `dev`

---

## 1. What We're Building

A fidelity system that makes backtests more reliable by using Polygon 1-second bars to resolve intra-bar ambiguity on entry and exit bars.

### Core Architecture

- **Pass 1** = Current unified engine (UNCHANGED)
- **Pass 2** = HiFi resolution on entry/exit bars (NEW, additive)
- **Default:** HiFi on all exit bars + all non-`[C]` entry bars
- **Performance:** 3-20 seconds overhead on a 90-day backtest (acceptable)

### Execution Types

Four tags — philosophy of how you enter:
- `[C]` Close — enter on bar close (existing, renamed from `[C]`)
- `[L]` Level — enter on intra-bar level cross (existing, renamed from `[L1]`)
- `[LC]` Level-Close — two-stage: level cross entry, bar close confirmation (existing, renamed from `[HM]`/`[HL]`)
- `[CC]` Close-Close — two-stage: bar close entry, next bar close confirmation (new)

Parameters live on the **confluence pack** (not the strategy), in four self-contained sections per execution type. Pack creator sets defaults.

### Confluence Fidelity

- `[PB]` Previous Bar — standard, always reliable
- `[CB]` Current Bar — uses 1-second sub-data, requires HiFi in backtest

---

## 2. Implementation Phases

### Phase 31F-1: Tag Rename + Execution Parameter Structure

**Goal:** Rename existing tags, add execution parameter sections to confluence pack templates. No logic changes.

**Changes:**
- Rename display tags: `[L1]` → `[L]`, `[HM]` → `[LC]`, `[HL]` → `[LC]`
- Add execution parameter sections to TEMPLATES dict in `confluence_groups.py`
- Update tag display in `app.py` (everywhere tags are shown)
- Update `_execution_tag()` function

**Risk:** LOW — display-only changes. Underlying trigger logic untouched.

**Checkpoints:**
- [ ] Tags show new names in UI (Strategy Builder, trade history, strategy cards)
- [ ] Existing strategies still work (logic unchanged)
- [ ] +/x symbols still plot correctly on charts

### Phase 31F-2: Pass 2 Engine — Stop/Target Resolution

**Goal:** On all exit bars, fetch 1-second data and determine which was hit first (stop or target).

**Changes:**
- Add `run_hifi_pass()` function in `unified_engine.py`
- After Pass 1 completes, iterate exit bars, fetch 1-second data, walk through
- First level hit wins (replaces hardcoded "stop wins")
- Add `hifi_resolved` flag to trade records
- Add `hifi_exit_second` timestamp to trade records

**Risk:** MEDIUM — modifies trade outcomes on ambiguous bars. Pass 1 results preserved for comparison.

**Checkpoints:**
- [ ] Ambiguous exit bars (both stop+target reachable) get 1-second resolution
- [ ] Trade outcomes change only on ambiguous bars (non-ambiguous identical)
- [ ] `hifi_resolved` flag present on resolved trades
- [ ] Performance: <20 seconds for 90-day backtest

### Phase 31F-3: Pass 2 Engine — Entry Fill Timing for [L] and [LC]

**Goal:** On entry bars for `[L]` and `[LC]` triggers, find exact fill second and price.

**Changes:**
- Extend `run_hifi_pass()` to process entry bars
- For `[L]`: walk 1-second bars to find exact cross second
- For `[LC]`: simulate entry on cross, then check bar close for confirmation
- Record precise fill price and timestamp

**Risk:** MEDIUM — more accurate entry prices may shift R-multiples slightly.

**Checkpoints:**
- [ ] L-type entries get precise fill second
- [ ] LC entries simulate cross + confirmation correctly
- [ ] R-multiples adjust based on precise fill (vs level-price fill)

### Phase 31F-4: Confluence Pack Execution Parameters UI

**Goal:** Add the four execution parameter sections to the confluence pack editor.

**Changes:**
- Update confluence pack template structure in `confluence_groups.py`
- Add execution parameter sections to the pack editor UI in `app.py`
- Each of the four tags (`[C]`, `[L]`, `[LC]`, `[CC]`) gets its own collapsible section
- Parameters conditionally shown based on relevance (limit params only for limit order types)

**Risk:** LOW — UI addition, no engine changes.

**Checkpoints:**
- [ ] Four execution sections visible in confluence pack editor
- [ ] Parameters save and load correctly
- [ ] Strategy Builder shows distinct trigger options per execution type

### Phase 31F-5: Current Bar Confluence [CB]

**Goal:** Add `[CB]` as a fidelity option for cross-TF confluence conditions.

**Changes:**
- In backtest Pass 2: recompute higher-TF indicators from 1-second data at trigger moment
- In live: shadow engines already update per-second (Phase 31C) — wire to condition check
- Strategy Builder: show `[PB]` and `[CB]` as distinct condition options

**Risk:** MEDIUM — new confluence evaluation path. Must verify no look-ahead bias.

**Checkpoints:**
- [ ] `[CB]` conditions available in Strategy Builder
- [ ] Backtest results differ from `[PB]` only where the forming bar's state differs from the closed bar's state
- [ ] Live evaluation matches backtest (same 1-second data source)
- [ ] No look-ahead bias (verified by comparing live alerts vs backtest)

### Phase 31F-6: QA Harness

**Goal:** Verification system for batch QA of HiFi-resolved trades.

**Changes:**
- Log every HiFi-resolved trade: strategy, bar time, Pass 1 result, Pass 2 result
- UI page or modal: "HiFi Verification Queue"
- Per-trade "View 1-Second Chart" button showing zoomed-in execution
- Enables batch verification: run strategies, collect data, verify all at once

**Risk:** LOW — read-only diagnostic tool.

**Checkpoints:**
- [ ] HiFi trades logged with Pass 1 vs Pass 2 comparison
- [ ] 1-second chart modal shows entry/exit on sub-bar candles
- [ ] User can mark trades as "verified" or "flagged"

---

## 3. Safety Guardrails

- Pass 1 code is READ-ONLY — zero modifications to existing unified_engine.py bar-by-bar logic
- Pass 2 is a NEW function that runs AFTER Pass 1, on specific bars only
- Backup branches available: `main-backup-pre-31`, `main-backup-pre-37`
- Tag renames are display-only — underlying trigger IDs unchanged
- Each sub-phase explicitly states whether QA of +/x plotting is needed
- Confluence pack parameter sections are additive — existing packs get default values auto-populated

---

## 4. Build Order

```
31F-1 (tag rename + param structure) — no logic changes, safe
  └→ 31F-2 (stop/target HiFi) — first real engine addition
      └→ 31F-3 (entry fill HiFi) — extends Pass 2
  └→ 31F-4 (pack editor UI) — can parallel with 31F-2/3
      └→ 31F-5 (CB confluence) — depends on UI + engine
          └→ 31F-6 (QA harness) — after all features in place
```

31F-1 and 31F-4 can run in parallel (display + UI changes).
31F-2 and 31F-3 are sequential (both modify Pass 2).
31F-6 should be available as early as possible for verification.

### Cascading Drill-Down for Larger Timeframes

For HiFi resolution, the drill-down depth depends on the primary timeframe:

| Primary TF | Drill-Down Strategy | Bars to Walk |
|-----------|--------------------|----|
| 1Min–15Min | Direct to 1-second | 60–900 bars |
| 30Min | Direct to 1-second | 1,800 bars (borderline, still OK) |
| 1Hour | Cascade: first to 1-minute (60 bars), then ambiguous minute to 1-second (60 bars) | 60 + 60 = 120 bars |
| 4Hour | Cascade: to 1-minute, then to 1-second | Same pattern |
| 1Day | Cascade: to 1-minute (390 bars RTH), then to 1-second | 390 + 60 = 450 bars |

Rule: if primary TF > 15Min, use two-step cascade. Two API calls instead of one giant one.

---

## 5. What Success Looks Like

1. Existing strategies produce identical results (Pass 1 unchanged)
2. HiFi resolution corrects ambiguous bars with 1-second evidence
3. All four execution types available as distinct trigger options per confluence pack
4. Pack creators can set execution parameter defaults per type
5. `[PB]` and `[CB]` confluence available as distinct conditions
6. User can visually verify HiFi trades via 1-second chart modal
7. Backtest matches live for all execution types (Golden Rule)
8. +/x symbols on charts continue plotting correctly
