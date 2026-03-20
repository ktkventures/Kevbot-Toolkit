# Phase 31F: HiFi Backtest & Fidelity System — Implementation Spec

**Status:** PLANNING — Playbook under review, spec is tentative
**Date:** 2026-03-20
**Depends on:** Phase 31A-31E (Polygon data provider — COMPLETE)
**Source of truth for definitions:** `docs/Execution_and_Fidelity_Playbook.md`
**Branch:** `dev`
**Backup:** `main-backup-pre-31`

---

## 1. What We're Building (Big Picture)

A fidelity system that uses Polygon's 1-second bar data to make backtests more reliable and to unlock new execution/confluence capabilities that weren't possible with bar-level-only data.

### Core Concepts

1. **Two-Pass Backtest** — Pass 1 is the existing engine (unchanged). Pass 2 "zooms in" on ambiguous bars using 1-second data to resolve what actually happened intra-bar. This means the current engine stays safe — Pass 2 is an additive layer.

2. **Fidelity Tags** — Per-trigger and per-condition tags that declare whether something evaluates on the previous closed bar `[PB]` or the current forming bar `[CB*]`. These are treated as distinct conditions in the Strategy Builder and mass search, so backtesting naturally discovers which fidelity mode works better for each setup.

3. **Execution Type Tags** — Existing `[C]`, `[L0]`, `[L1]`, `[HM]`, `[HL]` stay as-is. New HiFi variants (`[L0*]`, `[HM*]`, `[HL*]`, `[HC]`) are added alongside — not replacing. Redundant types removed later after validation.

4. **QA Harness** — A verification system that logs HiFi-resolved trades so the user can visually confirm execution in batches, rather than hunting for individual trades during live testing.

### The Golden Rule

**Does the backtest match live behavior?** Every feature must produce consistent results between backtest and live. If it can't, it gets a disclaimer. The fidelity system exists specifically to IMPROVE this consistency by giving backtests access to the same intra-bar resolution that live trading has via per-second bars.

---

## 2. Key Decisions Made

| Decision | What We Agreed |
|----------|---------------|
| Keep existing engine untouched | Pass 1 = current unified_engine.py, zero changes |
| New types alongside old | [L0*] exists next to [L0], not replacing it |
| Fidelity tags on conditions | [PB] and [CB*] are distinct conditions in Strategy Builder |
| Naming convention | [Square brackets] = execution/fidelity tag, (Parentheses) = parameter variant, * = HiFi required |
| No global toggle | Fidelity is per-trigger and per-condition, not a strategy-wide setting |
| Playbook is gospel | Plain-English spec that code is held accountable to |
| QA in batches | Collect multiple HiFi scenarios, verify together, not one at a time |

---

## 3. What This Enables (Use Cases)

### Use Case 1: Stop vs Target Resolution
**Problem:** Bar hits both stop and target. Currently hardcoded "stop wins."
**HiFi:** Walk 1-second bars. First level hit wins. Actual win rate and P&L become accurate.
**Impact:** Directly changes trade outcomes on ambiguous bars.

### Use Case 2: L-Type Fill Precision
**Problem:** L0 entry fills at indicator level (e.g., VWAP), but we don't know when in the bar it crossed.
**HiFi:** 1-second walk finds exact cross second and precise fill price.
**Impact:** Better slippage estimation, more accurate R-multiples.

### Use Case 3: Current Bar Confluence [CB*]
**Problem:** Cross-TF confluence waits for higher-TF bar to close (up to 5 minutes delayed).
**HiFi:** Check what the forming higher-TF bar currently shows. Catch "hot crosses" immediately.
**Impact:** Earlier entries on momentum shifts. Backtestable via 1-second indicator recomputation.

### Use Case 4: HM/HL Backtestability
**Problem:** Hybrid market/limit orders can't be fully simulated with bar-level data.
**HiFi:** 1-second simulation of entry detection → hold → confirmation/reversal.
**Impact:** Makes HM/HL strategies properly backtestable for the first time.

### Use Case 5: Limit Order Verification
**Problem:** Was a limit price actually reached intra-bar?
**HiFi:** 1-second walk confirms whether price came back to the limit level.
**Impact:** Validates limit vs market execution strategy.

### Use Case 6: Hold Confirmation [HC]
**Problem:** Want to enter only if a condition holds for N seconds (e.g., VWAP cross sustained).
**HiFi:** Per-second evaluation counts consecutive seconds where condition is true.
**Impact:** New execution type that filters out false/flickering signals.

### Use Case 7: Volume-Based Current-Bar Triggers
**Problem:** RVOL at 3X should trigger immediately, not wait for bar close.
**HiFi:** Per-second volume accumulation checked against threshold.
**Impact:** Faster entries on volume spikes.

### Use Case 8: Swing Stop on Current Bar
**Problem:** Swing stop uses lowest low of last N closed bars, ignoring the forming bar.
**HiFi:** Current bar's developing low (via 1-second data) can inform stop placement.
**Impact:** Tighter, more responsive stop placement.

---

## 4. Implementation Phases (Tentative)

### Phase 31F-1: Stop/Target HiFi Resolution
**Easiest to verify.** Flag ambiguous bars (both levels reachable), fetch 1-second data, walk through, first hit wins. Add `hifi_resolved` flag to trade records.

### Phase 31F-2: L-Type HiFi Fill Timing [L0*]
Flag L0/L1 entry bars. 1-second walk for exact cross second. Precise fill price recorded. New `[L0*]` execution type tag.

### Phase 31F-3: Current Bar Confluence [CB*]
Biggest conceptual change. Add `[CB*]` fidelity option to confluence conditions. In backtest: recompute higher-TF indicators from 1-second data at trigger moment. In live: shadow engines already update per-second (Phase 31C). Strategy Builder shows `[PB]` and `[CB*]` as distinct condition choices.

### Phase 31F-4: HM*/HL* Backtestability
Full 1-second simulation of hybrid order flow. Makes these execution types properly backtestable.

### Phase 31F-5: Hold Confirm [HC]
New trigger execution type. Condition must hold for N seconds before entry. Backtest: 1-second walk counts consecutive true. Live: per-second evaluation.

### Phase 31F-6: QA Harness
Verification log/UI that flags HiFi-resolved trades. User can open 1-second chart modal per trade to visually confirm. Enables batch QA rather than hunting individual scenarios.

---

## 5. Safety Guardrails

- **Backup main branch** before any unified_engine.py changes
- **Pass 1 is read-only** — the existing bar-by-bar engine code is not modified
- **Pass 2 is additive** — new code that runs AFTER Pass 1, on flagged bars only
- **New types alongside old** — [L0*] coexists with [L0] until validated
- **QA harness built early** (31F-6 could be pulled earlier) so verification is available from the start
- **Each sub-phase gets explicit "does this need QA?" communication** — if engine changes could affect alert plotting, user is told explicitly

---

## 6. Files Affected (Expected)

| File | What | Risk Level |
|------|------|------------|
| `unified_engine.py` | Pass 2 logic, ambiguous bar flagging | HIGH — core engine. Pass 1 untouched, Pass 2 is new code. |
| `data_loader.py` | 1-second bar fetching for specific time windows | LOW — already works (Phase 31A) |
| `ralph_engine.py` | Shadow engine [CB*] live evaluation | MEDIUM — builds on Phase 31C per-second bars |
| `app.py` | Strategy Builder fidelity tag UI, QA harness, trade history badges | LOW — UI only |
| `triggers.py` | [HC] hold-confirm trigger type | MEDIUM — new trigger type |
| `confluence_groups.py` | [PB]/[CB*] fidelity option per condition | MEDIUM — new option on existing system |
| `Execution_and_Fidelity_Playbook.md` | Source of truth — updated as we refine | N/A |

---

## 7. What Success Looks Like

1. **Stop/target trades on ambiguous bars show the actual winner** (verified via 1-second chart modal)
2. **[CB*] confluence catches "hot crosses"** on higher TFs without waiting for bar close
3. **HM/HL strategies have backtests** that were previously impossible
4. **Backtest results match live execution** for all HiFi-resolved trades (verified via QA harness)
5. **User can visually verify** any HiFi trade via a 1-second chart modal in the trade history
6. **Existing [C], [L0], [L1] strategies continue working identically** — zero regression
7. **Mass builder can iterate** over [PB] vs [CB*] variations to find optimal fidelity per setup

---

## 8. Open Questions (To Resolve During Planning)

- [ ] Should QA harness (31F-6) be pulled earlier in the build order?
- [ ] What's the performance budget for Pass 2? (target: <5 seconds for a 90-day backtest)
- [ ] How do we handle [CB*] for very high timeframes (daily, weekly)? Always [PB]?
- [ ] Should the Strategy Builder auto-suggest fidelity type based on trigger type?
- [ ] How do we display the [PB]/[CB*] distinction in the confluence heatmap on charts?
- [ ] Webhook payload: does the fidelity tag need to be in the payload?
