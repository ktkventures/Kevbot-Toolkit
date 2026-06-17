# Fidelity Gate — regression harness against backtest↔live drift

_Created 2026-06-17. The standardized test suite we run before/after any
**fidelity-critical** change, so we never re-introduce the invisible
backtest↔live divergences it took weeks to eliminate._

## The two change classes (the [FC] convention)

- **Fidelity-critical (`[FC]`)** — anything that *could* change how
  indicators/interpreters/triggers/gates/trades resolve, or how backtest-vs-live
  data is loaded/warmed/trimmed. **Must pass the Fidelity Gate before merge.**
  Touches: `services.py` (prep), `unified_engine.py`, `interpreters.py`,
  `indicators.py`, `triggers.py`, `confluence_groups.py`, the engines
  (`data_worker_engine.py`, `ralph_engine.py`), `forward_test_service.py`.
- **Surface** — UI shape/copy, dashboards, non-engine routers, docs. Normal
  review, no gate. (e.g. the 2026-06-17 divergence-tab disable was surface — it
  touched no math.)

Tag fidelity-critical commits/PRs `[FC]`.

## Why an engine-only test is NOT enough (the crux)

`test_unified_parity.py` tests the **engine** (`run_unified_backtest` on synthetic
bars). But the changes we fear most — #21 (scope confluence-group computation),
#29 (warmup/trim) — live in **`prepare_data_with_indicators`**, the *prep* path.
An engine-only golden would pass while prep silently shifted a gate. So the gate
MUST exercise the **full prep pipeline on frozen real bars**.

## Architecture — two tiers

### Tier 1: Hermetic golden suite (GitHub Actions, every push)
No network/secrets — committed fixtures only.
1. **Prep-path golden (the headline guard):** for each frozen strategy, a fixture
   of **raw bars** (primary + secondary TFs). The test **patches
   `data_loader.load_market_data`** to return the frozen bars, runs the real
   `services.prepare_data_with_indicators(...)`, then `unified_trades(...)`, and
   asserts (a) the strategy's gate/trigger/interpreter columns and (b) the trades
   are **byte-identical** to the golden snapshot (`assert_frame_equal`,
   `assert_series_equal(check_exact=True)`).
2. **Engine golden:** `run_unified_backtest(frozen_enriched_df, strat)` → exact
   trades (catches engine-logic drift independent of prep).
3. **Trigger/gate emission:** every enabled template emits its declared trigger
   columns (catches rsi_zones "template not found", bull_c3/bull_flip
   "not in cache" #22).
4. **Wire in existing hermetic suites:** `test_unified_parity.py` (31),
   `test_mass_builder_fidelity_parity.py`, `test_parity_regression.py`.

Representative frozen strategies (cover the failure modes): ungated single-trigger;
2m-gated; **1h/4h/1d-gated (sid 309/313 class)**; sub-minute 15Sec; one per
in-use pack template.

### Tier 2: Live before/after command (local, manual, for [FC] changes)
`fidelity_gate.py` against live data — catches data-path drift fixtures can't:
- `--snapshot` : capture current backtest+algo trades for the live strategies
  (308–314 + canaries) from the DB lanes → `before.json` (instant; no UND needed).
- `--check` : after the change, regenerate via `get_strategy_trades(strat)` and
  diff vs `before.json`; assert no entry/exit/price/timestamp shifts; report any
  strategy whose trade set changed.
- Optionally re-run Update-New-Data one-at-a-time before+after to prove the UND
  path itself doesn't relapse.

## The before/after protocol for an [FC] change
1. Tier 2 `--snapshot` (freeze current trusted trades).
2. Prepare the change locally; run Tier 1 (hermetic golden) — must be byte-identical
   unless the change is *intended* to alter trades (then re-bless fixtures deliberately).
3. Tier 2 `--check` (live diff); confirm alerts still fire where they currently fire
   (cross-check Strategy Health combined% doesn't regress).
4. Only then merge + deploy.

## Files
- `src/fidelity/` — harness package: `capture.py` (freeze fixtures), `golden.py`
  (Tier-1 tests), `fidelity_gate.py` (Tier-2 live command).
- `src/fidelity/fixtures/` — committed raw-bar + enriched + golden-trade snapshots.
- `.github/workflows/fidelity-gate.yml` — Tier-1 runner on push/PR.

## Status
- 2026-06-17: design locked; harness build starting. Existing hermetic tests
  (`test_unified_parity` etc.) to be wired into Tier 1.

## Additional tests under consideration (Kevin to weigh in)
- Strategy-Health combined% as a post-deploy smoke threshold (live monitor, not CI).
- JSONB partial-update guard (the strategies-70/71 config-wipe class).
- Snapshot model_id lineage (the infinite re-catchup class).
- No-wipe guards on UAD/append (lane-wipe-on-fetch-failure, #16).
