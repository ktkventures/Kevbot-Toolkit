# User Pack Roadmap

A living per-pack health register. Distinct from `Known_Bugs.md` — that doc is for active fixable bugs across the whole system; this one is the **ongoing per-pack catalog** that survives bug-fix cycles and tracks each pack's trigger-mode and gate-mode health over time.

**Last update:** 2026-06-02 — after the Phase 2 codec ship + PACKTEST canary creation + worker restart
**Authoritative truth source:** PACKTEST canaries (sids 276-305, 2 per pack: `· trigger` and `· gate`)

---

## Top-of-doc summary

Two large systemic findings dominate the per-pack health picture today. Read these first — they reframe most of the per-pack data below.

### 1. Gate-mode is broken across nearly every user pack
**Status:** [Known_Bugs.md → User-pack confluence gates silently fail on secondary timeframes]

14 of 15 PACKTEST gate canaries fired **0 alerts** over a full trading day. The only exception (Swing 1-2-3) appears to pass because `NEUTRAL` is the engine's default-when-uncomputed value, so the gate trivially passes. Other packs require an actively-computed state name that's never produced because the secondary-TF shadow engine likely doesn't instantiate user-pack incremental engines.

**Implication for this doc:** *Every* "gate-mode state" cell below reads as BROKEN. Don't treat that as 15 separate per-pack bugs — it's one systemic bug. Per-pack gate-mode investigation is blocked until the secondary-TF user-pack-engine gap is fixed.

### 2. Trigger-mode divergences are mostly catchup lag, not pack bugs
The streaming-tick path is currently 10-30 minutes behind the live alert stream (data_worker is still chewing through post-restart catchup as of 2026-06-02 21:15 UTC). The "missed pairs" / "off >5min" data points in the trigger-mode analyses below are mostly explained by this lag, not by pack defects. Re-measure tomorrow morning once worker is steady-state to get a clean trigger-mode read.

### 3. Two packs have a distinct backtest-path bug (live works, batch doesn't)
EMA PP v3 (sid 278) and v4 (sid 280) — trigger canaries fire alerts (5 each today) but **zero backtest trades**. Inverse of gate-mode bug. Trigger-time alert dispatch works; batch indicator produces no entries.

---

## Per-pack catalog (sorted by trigger-mode health — best first)

| Pack | Trigger sid | Trigger alerts today | Trigger pair rate ±5s | Gate sid | Gate alerts | Status |
|---|---|---|---|---|---|---|
| MACD Histogram v2 | 284 | 347 | 92% (109/119) | 285 | **0** | trigger ✓, gate BROKEN |
| UT Bot V4 | 302 | 202 | 94% (61/65) | 303 | **0** | trigger ✓, gate BROKEN |
| EMA Stack v2 | 282 | 56 | 91% (20/22) | 283 | **0** | trigger ✓, gate BROKEN |
| RSI Zones 2 | 288 | 152 | 89% (51/57) | 289 | **0** | trigger ✓, gate BROKEN |
| Swing 1-2-3 | 300 | 106 | 87% (97/111) | 301 | 204 ⚠️ | trigger ✓, gate fires-by-default |
| Bollinger Bands | 276 | 50 | 83% (15/18) | 277 | **0** | trigger ✓, gate BROKEN |
| SuperTrend | 298 | 45 | 82% (9/11) | 299 | **0** | trigger ✓ (small sample), gate BROKEN |
| Strat Assistant | 296 | 367 | 77% (103/133) | 297 | **0** | trigger mid, gate BROKEN |
| Relative Volume v2 | 290 | 306 | 72% (75/104) | 291 | **0** | trigger mid, gate BROKEN |
| Stochastic Oscillator | 294 | 70 | 64% (18/28) | 295 | **0** | trigger mid, gate BROKEN |
| MACD Line v2 | 286 | 112 | 52% (21/40) | 287 | **0** | trigger mid (sample noise?), gate BROKEN |
| Support Resistance Channels | 292 | 125 | 21% (11/52) | 293 | **0** | trigger BROKEN + chart broken, gate BROKEN |
| VWAP v2 | 304 | 3 | 100% (2/2) | 305 | **0** | trigger insufficient sample, gate BROKEN |
| EMA Price Position v3 | 278 | 5 | n/a (no backtest trades) | 279 | 0 | **TRIGGER backtest-path BROKEN**, gate BROKEN |
| EMA Price Position v4 | 280 | 5 | n/a (no backtest trades) | 281 | 0 | **TRIGGER backtest-path BROKEN**, gate BROKEN |

(Pair rate measurements: ±5s, manual alert↔trade pairing on today's 13:30-20:00 UTC window.)

---

## Per-pack deep dive — by priority

### Tier A — trigger-mode healthy, ready for production once gate-mode lands

Working as designed; backtest matches live for the entry trigger to within 5 seconds for 80-94% of alerts. The remaining 6-20% divergence is dominated by:
1. **Worker catchup lag** at the right edge of the window (most divergent alerts from the last 15-20 min)
2. The known **streaming-tick under-fire ~7%** structural residual (Known_Bugs.md)

When gate-mode is fixed and worker catchup is back to steady-state, these should all sit at ≥95% pair rate.

**Packs in this tier:** MACD Histogram v2, UT Bot V4, EMA Stack v2, RSI Zones 2, Swing 1-2-3, Bollinger Bands, SuperTrend

No per-pack bugs to file. Re-evaluate after gate-mode fix lands.

### Tier B — trigger-mode mid-range, needs Layer 4 walkthrough

Pair rates 50-77% — possible real per-pack divergences mixed with catchup-lag noise. Don't try to fix yet; first re-measure tomorrow with worker at steady state, then drill in on whatever residual remains.

**Strat Assistant (sid 296) — 77%**
- 367 alerts today. Largest sample of any pack except MACD Hist. Real divergences likely substantial in absolute count even at 77%.
- Hypothesis: candlestick-pattern packs are timing-sensitive (the bar's close is the discriminator). If live engine evaluates on a slightly different close value than the batch indicator (e.g., WS-aggregated vs REST-reconciled close), bar-classification flips and entry doesn't pair.
- Action: after gate-mode fix + worker catchup steady-state, run Layer 4 on 5 divergent pairs.

**Relative Volume v2 (sid 290) — 72%**
- 306 alerts. RVOL needs N-bar volume history to compute the multiplier. If the live engine's volume buffer differs from the batch indicator's (different bar-close volumes, different session-reset semantics, etc.), the multiplier threshold crosses at different times.
- Action: confirm rvol_v2's `_buffer` survives the Phase 2 codec round-trip (we have a per-pack unit test for that — `_test_userpack_state_codec.py` showed PASS on 2026-06-01). Then Layer 4.

**Stochastic Oscillator (sid 294) — 64%**
- 70 alerts. Uses 4 rolling lists (highs/lows/raw_k/stoch_k) — the most state-dependent indicator in the pack set. State drift between live and batch is plausible.
- Action: round-trip test already passed; re-measure tomorrow; if still mid then Layer 4 on K/D values at divergent fire moments.

**MACD Line v2 (sid 286) — 52%**
- 112 alerts. This is suspicious because MACD HISTOGRAM v2 (which uses the same EMAs internally) is at 92%. If MACD Line is genuinely diverging while MACD Histogram isn't, that suggests the divergence is in the LINE-trigger evaluation (M>S, M<S crosses) rather than the underlying EMA state.
- Hypothesis: line-crossover timing offset between live and batch. Maybe the live engine fires the cross on intra-bar tick, batch fires on bar close.
- Action: HIGH priority for Layer 4 — biggest gap between expected and actual.

### Tier C — trigger-mode broken

**Support Resistance Channels (sid 292) — 21% trigger pair rate + chart visibly broken**
- 125 alerts but only 21% pair within 5s. That's the worst of any pack with a meaningful sample.
- Chart screenshot (`image copy 39.png`) shows the SR levels flat-lining / not updating on the price chart — indicator output appears stuck.
- This pack has 6 parallel ring buffers (highs/lows/closes/opens/pivot_highs/pivot_lows) — the most complex state of any pack. High chance of state corruption or stale-buffer behavior.
- User memory: "this might have been one of those kinds of stress tests where we wanted to build a pretty complicated indicator and use it"
- Status: **Treat as not production-ready.** Don't gate other work on it. Re-architect or rebuild before relying on it.
- Action: deprioritize unless someone has time to rebuild it from the reference. Tag as `experimental` in pack manifest.

**EMA Price Position v3 (sid 278) — L-type triggers don't fire in batch path**
- 5 alerts today (live engine works). **Zero backtest trades** despite UAD running.
- **Root cause identified 2026-06-02 evening:** eppv3 declares its triggers as L-type via `trigger_levels` (with `level_column: eppv3_short` + `cross: above`). The live engine handles L-type fine — alerts fire. But the batch backtest path apparently doesn't wire user-pack L-type triggers into `INTRABAR_LEVEL_MAP`/intrabar reachability the same way it handles C-type. The `__eppv3_cross_*` columns the batch indicator writes never become entry signals.
- **Compare to working ema_stack_v2**: declares triggers as C-type via `trigger_levels_phase2` (which is just a doc marker, not real L-type spec). Batch path reads `__esv2_cross_bull` etc. as bar-close booleans directly. Works.
- **Proposed fixes (pick one):**
  1. **Quick:** rewrite eppv3 manifest's `trigger_levels` → `trigger_levels_phase2` (just a doc move) and verify `__eppv3_cross_*` columns are picked up by the batch C-type path the same way esv2's are. May regress live L-type intra-bar fill semantics if any strategy relies on them.
  2. **Proper:** fix the batch path's user-pack L-type wiring to actually evaluate intrabar level crosses against the user-pack indicator columns. Touches `unified_engine.py` trigger evaluation core. Higher risk; needs careful test coverage.
- **Action:** investigate further tomorrow. Read `unified_engine.py:INTRABAR_LEVEL_MAP` resolution + the L-type entry decision logic to determine if quick fix is safe.

**EMA Price Position v4 (sid 280) — same as v3**
- 5 alerts, 0 backtest trades. **Same L-type batch-path bug** — v4's manifest has the same `trigger_levels` declaration.
- Action: same as v3; fix as a pair.

---

## Investigation order (highest impact first)

1. **Fix gate-mode systemic bug** — single root cause, fixes 14 packs' gate-mode at once. Largest leverage. Tracked in Known_Bugs.md.
2. **Wait for worker catchup steady-state**, re-measure trigger-mode pair rates tomorrow morning. Many "mid" packs may climb to healthy on their own.
3. **EMA PP v3/v4 backtest-path bug** — small in scope (2 packs), localized hypothesis (column contract). Probably an hour of debugging.
4. **MACD Line v2 52% trigger** — highest residual divergence after worker settles. Layer 4 on 5 pairs.
5. **Other mid-range packs** (Strat Assistant, RVOL, Stochastic) — Layer 4 per pack as time permits.
6. **SR Channels** — defer / consider rebuild. Don't block on it.

---

## Open per-pack questions

- Is there a `validate_column_contract` validator that ran on every pack at registration time? If yes, EMA PP v3/v4 should have been flagged. If they passed validation but still produce no trades, the validator has a gap.
- Is there a way to programmatically test "does this pack's gate produce any state changes on the secondary TF?" — would catch the gate-mode bug at pack-creation time instead of only via runtime canaries.
- Are the PACKTEST canaries created by Recreate-all the right shape for ongoing health monitoring, or do we need additional canary patterns (e.g., one per (pack × symbol × TF) combo)?

---

## Revision log

- **2026-06-02** — Initial creation. Top-7 trigger-mode analysis + gate-mode systemic bug + EMA PP v3/v4 backtest-path bug + SR Channels deprioritization. Top finding: 14 of 15 gate canaries fired 0 alerts → gate-mode is universally broken.
