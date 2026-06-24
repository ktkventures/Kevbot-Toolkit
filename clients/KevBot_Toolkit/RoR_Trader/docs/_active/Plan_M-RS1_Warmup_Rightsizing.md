# Plan — M-RS1: Right-size the recompute warmup window — 2026-06-24

**Status:** READY TO BUILD (awaiting green-light). Part of the Recompute Scalability track —
see `Recompute_Scalability_Findings.md`. **Do this FIRST** (biggest bang, least risk, fraction
of M-RS2's effort). It's what actually makes a single recompute fast.

## The bug (measured)
The full-recompute path sizes warmup with a flat multiplier that ignores what the indicators
actually need:

```python
# src/strategy_data.py:41
WARMUP_MULTIPLIER = 2
LEGACY_FALLBACK_DAYS = 90

# src/strategy_data.py:131
def compute_warmup_days(strat, visible_days):
    """Warmup buffer in days. v1: visible_days * WARMUP_MULTIPLIER."""
    _ = strat  # unused in v1; reserved for v1+1   <-- the TODO admits it
    return max(1.0, visible_days * WARMUP_MULTIPLIER)
```

So total load = `visible + 2×visible = 3× the visible window`. For an old coarse-gate strategy
whose `backtest_start_date` is ~1.5yr back, visible ≈ 540 days → loads ~1,620 days ≈ **~555k
1-min bars**, and the engine runs `process_bar` over all of it — **~2/3 is warmup that gets
trimmed away** (`trim_trades_to_visible`). This is the 353s load + much of the 153k-bar engine
cost in the cProfile.

It scales with `visible_days`, which is exactly wrong: a 4h indicator needs a fixed ~250 bars
(~6 months calendar) to warm up, regardless of whether the visible window is 90 days or 540.

## The standard it should use (already exists — two places, full path is the odd one out)
1. **`ralph_engine._secondary_warmup_days()`** (`ralph_engine.py:1748`) with
   `_SHADOW_WARMUP_TARGET_BARS = 250` — the LIVE engine warms each secondary TF with ~250 of
   its own bars. Derived, not flat.
2. **`get_strategy_trades_for_window`** (the append path, `services.py:653-683`) already sizes
   warmup off the **binding (coarsest) TF**:
   ```python
   all_tfs = [timeframe] + list(sec_tfs)
   binding_bpd = min(bpd for bpd in bpds if bpd > 0)        # coarsest TF
   warmup_days = max(1, ceil(warmup_bars / binding_bpd * 365/252))
   ```
   (`BARS_PER_DAY`: `data_loader.py:861`; `4Hour`=2 → 250 bars ≈ 125 trading days ≈ ~181
   calendar days.)

The full path (`get_strategy_trades` → `load_strategy_data` → `compute_warmup_days`) uses
neither. **The fix makes the full path consistent with the live engine + the append path** —
which is also a parity win (backtest warms like live warms).

## The fix
Replace the flat `visible_days × 2` with a **TF/indicator-derived warmup**, extracted into ONE
canonical helper that the full path, the append path, and (ideally) the live warmup all call —
so warmup sizing stops being duplicated/ad-hoc (the audit `Audit_Warmup_Window_Alignment.md`
flagged 4 divergent magnitudes: ×2, 100-bar, 365-day, 150-day).

### Canonical helper (PER-TF — a single global bar target backfires)
**Key design correction:** a global bar target high enough to converge an EMA-200, applied to
the COARSE TF, gives ~2.4yr of warmup (1200 bars ÷ 2 bars/day) — WORSE than today. Warmup must
be sized **per-TF**, then take the max calendar span:
- **Primary TF** (fastest): generous bar count (covers long-EMA convergence) — nearly free in
  calendar terms (1200 bars of 30Sec ≈ 1.5 trading days).
- **Secondary TFs** (coarse): **250 bars each** = `ralph_engine._SHADOW_WARMUP_TARGET_BARS`, so
  backtest warms its secondaries exactly like LIVE does (parity bonus).

```python
PRIMARY_WARMUP_BARS   = 1200  # generous; covers EMA-200 convergence. Cheap — primary is fastest TF.
SECONDARY_WARMUP_BARS = 250   # == ralph_engine._SHADOW_WARMUP_TARGET_BARS (live secondary standard)
_TRADING_TO_CALENDAR  = 365.0 / 252.0  # inflate trading days → calendar days (weekends/holidays)

def compute_warmup_days(strat, visible_days=None):
    """Calendar days of warmup, sized PER-TF to the bars each TF needs to
    converge — INDEPENDENT of visible_days. Kill-switch RORT_RIGHTSIZE_WARMUP."""
    if not _rightsize_warmup_enabled():
        return max(1.0, float(visible_days) * float(WARMUP_MULTIPLIER))  # legacy ×2
    primary = strat.get('timeframe', '1Min')
    sec_tfs = [get_tf_from_label(l) for l in get_required_tfs_from_confluence(strat.get('confluence', []))]
    def _days(bars, tf):
        bpd = BARS_PER_DAY.get(tf, 390) or 390
        return math.ceil(bars / bpd * _TRADING_TO_CALENDAR)
    days = _days(PRIMARY_WARMUP_BARS, primary)
    for tf in sec_tfs:
        days = max(days, _days(SECONDARY_WARMUP_BARS, tf))
    return max(1.0, float(days))
```
- **Refinement (only if the gate fails on a coarse-secondary EMA):** bump `SECONDARY_WARMUP_BARS`
  or pull the secondary's longest indicator period from `resolve_strategy_requirements`
  (`unified_engine.py:460`, returns indicators+params). For v1, 250 matches live + the gate
  catches any shortfall. Note: a sub-byte coarse-secondary diff that ALIGNS backtest with live
  (both 250 bars) is arguably more-correct than matching the over-warmed current backtest.

### Wiring
- `strategy_data.compute_warmup_days` becomes the canonical helper (or delegates to a new
  `unified_engine`/shared module fn so live + append + full all import the SAME function).
- `load_strategy_data` (`strategy_data.py:147`) calls it (already does — just gets the smarter
  result).
- Optionally refactor `get_strategy_trades_for_window`'s inline calc to call the shared helper
  too (de-dup; not strictly required for the speedup).
- **Kill-switch `RORT_RIGHTSIZE_WARMUP`** (default OFF) so we A/B old-vs-new and roll back instantly.

## Validation gate (MUST pass before flipping default ON — fidelity is paramount)
Byte-identical **right-sized == current (×2)** for `unified_trades` output (entry/exit ts,
price, exec_type) AND the prepared indicator columns **in the visible window**, on:
- an ungated 15Sec (primary cohort),
- a sub-minute + secondary (sid 309),
- a coarse 1d/4h gate (sid 313 / 325 — the worst case).

**Interpretation:** warmup only affects indicator *convergence*. If right-sized == full, 250
bars was enough → safe + much faster. If trades differ, warmup was too short for that
indicator → bump `WARMUP_TARGET_BARS` or wire `_indicator_lookback_bars`. This empirical gate
is exactly what makes the change safe.

## Expected effect
- **Load:** ~1,620 days → ~181 days (4h) ≈ ~9× less 1-min data fetched → 353s → tens of s.
- **Engine/indicators/interpreters:** ~153k bars → ~the visible window (~47k) + small warmup
  → ~2-3× less CPU on the dominant buckets.
- **Net (325):** ~20 min → plausibly ~5-7 min, before M-RS2/M-RS3 even land.
- **Parity bonus:** backtest now warms like the live engine (250-bar standard) → fewer
  warmup-induced backtest↔live divergences.

## Risks / watch-items
- Indicators needing > 250 bars (long EMAs, slow oscillators) — caught by the byte-identical
  gate; mitigate via `_indicator_lookback_bars`.
- Strategies with NO secondaries: binding_bpd = primary bpd; warmup shrinks too (good) — verify
  the ungated 15Sec stays byte-identical.
- Legacy strategies without `backtest_start_date` still use `LEGACY_FALLBACK_DAYS=90` for the
  *visible* window — unaffected by this change (this only touches warmup sizing).

## Build sequence
1. Write the byte-identical harness (right-sized vs current) reusing `_compare_cache_replay.py`
   discipline — on the 3 strategy shapes.
2. Implement the canonical helper + kill-switch (default OFF).
3. Run the harness locally; confirm byte-identical on all 3 (bump target if not).
4. Time 325/309 right-sized vs current; record vs the cProfile baseline.
5. Flip default ON after parity proven; keep kill-switch for one cycle.
