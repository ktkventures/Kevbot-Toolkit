# Enhancement — Snapshot the secondary-TF engine state (UND speed) — 2026-06-22

**Want:** Make `update_new_data` (mode='new') much faster on cross-TF / coarse-gated
strategies.

**Why it's slow today:** UND already resumes the **primary** TF from a serialized engine
snapshot (`engine_snapshot_b64` in config, ~3–5 KB, validated by fingerprint + model_id) and
only computes trades over `[last_snapshot_ts → now]`. So the *primary* lane is incremental and
the trade-compute is bounded to the new gap — confirmed in
`api/services/forward_test_service.py::append_new_backtest_trades_for_strategy` (calls
`svc.get_strategy_trades_for_window(..., resume_snapshot_b64=...)`).

BUT the **secondary-TF shadow engines** (e.g. `1d`, `4h`, `30m` confluence gates) are **not**
captured in that snapshot. To evaluate a coarse gate on the new bars, the engine re-warms the
secondary indicator series every run — and because coarse bars are **resampled from 1-minute
bars** (never native — split-adjustment rule, see CLAUDE.md), warming a daily/4h indicator
forces loading a long 1-minute history each time. That warmup load — proportional to how
*coarse* the secondary TF is, NOT to the size of the new gap — is the dominant UND cost on
strategies like 308/309 (15Sec, 95d) and 312–314 (1d/4h gates). Ungated / fine-secondary
strategies (canaries, 7-day windows) are fast because they have little/no secondary warmup.

**Proposed fix:** Extend the snapshot scheme to also serialize the **secondary-TF shadow
engine state** (per-secondary indicator state + last-bar position), keyed alongside the primary
`engine_snapshot_b64`. On resume, restore each secondary shadow instead of re-warming it from a
fresh 1-min resample. Gate it behind the same fingerprint + model_id validation (any mismatch →
fall back to today's warmup-windowed path). Net effect: coarse-gated UND cost drops from
"reload weeks of 1-min + resample" to "restore state + process the new gap," same as the
primary lane already enjoys.

**Risks / care:** must preserve backtest↔live fidelity — the restored secondary state has to be
byte-equivalent to a fresh warmup at the resume boundary (parity guard, same discipline as the
primary snapshot). Secondary shadows currently have a regression history (Phase 31 broke cross-TF
live feed; see [[feedback_polygon_xtf_live_regression]]) — add a parity check.

**Priority:** medium-high — Kevin: "we need this update new data process to be a lot faster."
Pairs with the long-window backtest design (`Design_Long_Window_Backtests.md`).

---

## Audit: did anything we shipped since Thursday (06-18) slow UND/backtest? (2026-06-22)

Reviewed every commit since Thu 06-18 touching the backtest/UND compute path:
- `api/services/forward_test_service.py` (the UND append): **0 commits** — append logic unchanged.
- `services.py` / `strategy_data.py` / `unified_engine.py`: only **confluence scoping**
  (`#21` default OFF globally; mass-builder-only `force_scope`) + the gen-pack tz fix
  (negligible compute). Scoping *reduces* work / is off for UND → not a slowdown.
- **Bug-5 (TF-scaled warmup, 0e19838 + 8c2fdcb + 3159d96): touches `ralph_engine.py` ONLY**
  = the LIVE engine. Does **not** change UND/backtest warmup. **Exonerated.**
- **`data_loader.py` — 6c7de2c (Thu 06-18): the one real contributor.**

**6c7de2c — "never silently return partial Polygon data (retry + raise)":** two timing effects,
both side-effects of a *correctness* fix (NOT a logic regression):
  1. **Stopped silently truncating.** Before, a mid-pagination error returned partial data,
     dropping the most-recent bars (TSLA 15Sec 90d = 70,858 bars, missing ~3 weeks → caused the
     all-phantom bug, sid 308). After, it returns the COMPLETE set (97,303 bars, +37%). So
     backtests now grind the *correct, larger* bar count — the old "speed" was an illusion from
     dropping data. Concentrated on long sub-minute loads + their warmup (the same coarse-gate
     strategies already slow).
  2. **Retry/backoff adds wall-clock under Polygon stress.** Per-page timeout 30→60s; up to 6
     retries with exponential backoff (2→30s) on 429/5xx. UND hammering Polygon concurrently
     across 65 strategies makes 429 rate-limits likely → backoff sleeps; a hung page now waits
     up to 60s before retry vs an instant (wrong) partial return before.

**Verdict:** the since-Thursday slowdown is real and is 6c7de2c — but it's the cost of correct,
complete backtest lanes. **Do NOT revert** (re-introduces the truncation/all-phantom bug). Speed
levers instead: (a) the secondary-TF snapshot above (skip redundant warmup re-loads); (b) avoid
concurrent Polygon hammering (smaller UND batches; the retry backoff is worst under concurrency).
