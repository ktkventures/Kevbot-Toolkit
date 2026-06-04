# Roadmap — Divergence Hunting (Live ↔ Backtest Pair-Rate to 95%+)

**Last updated:** 2026-06-04 EOD
**Goal:** Drive fleet-wide live↔backtest pair rate from ~57% → 95%+ across the
canary cohort. The path is "structural drift first, then artifacts."

This doc is the single source of truth for what's done, what's open, and where
the related artifacts live. Update at each session's end.

---

## Current state — most recent reliable measurement

Window: 2026-06-04 16:18–18:18 ET (last clean read before streaming was toggled
off for the on-demand UAD test). Cohort: 9 active SPY 10Sec canaries.

### Layer 1 — Bar fidelity: GREEN
| status | % | target |
|---|---|---|
| verified | 87.0% | — |
| corrected | 10.7% | — |
| **drift_uncorrected** | **0.7%** | ≤2% on 10Sec ✓ |
| rest_unavailable | 0.0% | ~0% ✓ |
| NULL | 1.2% | (was 90% before [b2387ff](#shipped-today)) |

`live_model = ws_rest_spliced` on 100% of alerts (was ~10% yesterday).

### Layer 2A — Pair rate (per-strategy)
| sid | Pack | 06-03 baseline | 06-04 today | Δ |
|---|---|---|---|---|
| 302 | UT Bot V4 | 57.3% | **75.0%** | +17.7pp |
| 296 | Strat Assistant | 65.4% | **83.3%** | +17.9pp |
| 294 | Stochastic | 32.4% | **60.0%** | +27.6pp |
| 290 | RVOL v2 | 69.8% | 71.4% | +1.6 |
| 284 | MACD Hist v2 | 66.6% | 67.9% | ≈ |
| 286 | MACD Line v2 | 51.6% | 42.9% | −8.7 (small N) |

Fleet macro pair rate: **~57%** (243 paired / 425 events). Phantom-heavy:
126 phantoms vs 56 missed (2.25:1).

### Layer 2B — Per-event drill
**184 distinct clusters in 6h**. Most `needs_investigation`. Now broken down:
- **290 phantom entries** (unpaired live alerts) — the dominant bucket
- 56 missed (backtest emits, no live alert)
- **185 cross_exec_type_mismatch pairs surfaced by [#56](#shipped-today)** — previously hidden behind "paired = OK"

### Layer 3 — Fill-time delta on paired
- **Entry**: median 0s, avg 2.5s, **91.2% ≤5s** (target ≥95%)
- **Exit**: median 0s, avg 6.9s, **69.3% ≤5s** (target ≥90%)
- Pair-delta distribution (6h, 880 entry / 867 exit pairs):
  - Entry pairs: 94.3% within ≤1s (essentially exact)
  - Exit pairs: 66.3% within ≤1s, 28.1% are 5-60s loose tail = cross_exec_type_mismatch

### Cross-layer correlation
- Layer 1 clean + Layer 3 wide → drift is NOT the cause. The wide exit deltas trace to cross_exec_type_mismatch (live C-exit paired with backtest L-stop). [#47](#shipped-today) confirmed walker works correctly.
- Layer 2 phantom-heavy + Layer 3 tight entries → most phantoms are exit-side timing artifacts, not entirely false alerts. But the **290 unpaired entries** are real divergence, traced to UT Bot trailing-stop state drift between engines. See [#53 partial finding](#open-deep-investigation).

**Next session's first task:** trigger Update New Data on all 46 strategies (will run fast post [`3322298`](#shipped-today)), then re-run all four layers for a clean read.

---

## Shipped today

| # | Commit | What | Impact |
|---|---|---|---|
| 50 | abb6745 | `strategy_factory.py` — unified strategy creation across all 4 paths | All paths now emit identical 42-key config shape |
| 49 | 8c20122 | Pack canary recreate routed through factory | Was 12-key, now 42-key |
| — | f061076 | `_backfill_strategy_configs.py` migration | 558 fields filled across 42 strategies |
| 48 | 4be824a | Phase 2 backend: streaming toggle + async UAD jobs | DB-persisted; survives worker restart via orphan cleanup |
| 51 | f7eec9b | Streamlit Update Jobs subpage | UI parity with Mass Builder pattern |
| 51 | 768e353 | Next.js `/admin/update-jobs` page + bulk button swap | Production UI now uses DB-backed system |
| — | 2fd0c10 | Streaming-mode AUTO/MANUAL toggle UI | No-redeploy flag, 30s cache in Data Worker |
| — | 8fed13c | `_verify_streaming_toggle.py` observability script | Confirmed flip → DB → Data Worker propagation working |
| — | **3322298** | **ALGO-APPEND Hi-Fi: pass incremental=True (was missing)** | **179s → 27s per strategy (6.4×)** |
| 47 | (analysis) | Pass 2 walker investigation — closed false alarm | Walker working correctly; 86% per-second on stops |
| 56 | 7dba9ea | cross_exec_type_mismatch promoted to auto-classifier | 185 hidden mismatches now visible as `event_type=mismatch` |
| 54 | (analysis) | Tight pair window experiment | Entry pairs are 94% truly tight; exit pairs have 28% loose tail = mismatch |
| 57 | b87f059 | bar_diagnostics schema + manifest contract + UT Bot V4 seed | Foundation for per-bar state capture (engine hooks tomorrow) |

---

## Open — deep investigation

### #53 Phantom-entry root cause (partial finding)

**State:** Backtest's UT Bot trailing stop drifts from live's after a trade exit, then the two engines make different decisions on every subsequent bar.

**Evidence (sid 302, 20:40–20:58 UTC):**
- 20:40:10 → backtest entered (utv4_bull_flip)
- 20:41:18 → backtest exited (stop_loss)
- 20:41:18 onwards → backtest never re-emitted bull_flip
- Live engine fired **12 entries** in same 18-min window
- Layer 1 confirms bar prices byte-identical (close_delta=0.0)
- Indicator code is shared class (UtBotV4Incremental) → math is identical

**Hypothesis:** snapshot-resume state diverges between BT recompute and live engine after a position closes. The trailing stop's path-dependence amplifies tiny float-level differences into persistent state divergence.

**Unblocked by:** [#57C](#open-engineering-work) — engine hooks for per-bar `utv4_trailing_stop` logging. Once data is flowing we can see exactly where the levels diverge.

### #56 cross_exec_type_mismatch root cause

**State:** Auto-classifier ships ([7dba9ea](#shipped-today)) and surfaces the 185-mismatch pattern. But why does live fire C-type signal exit at bar close while backtest fires L-type stop intra-bar (or vice versa)?

**Hypothesis A:** WS-REST drift (Layer 1 shows 11% `corrected` rate) — live WS missed the intra-bar stop touch, exited only at signal C later.
**Hypothesis B:** Backtest reads slightly different Polygon 1-sec low than live WS sees at the same second.

**Investigation:** pick 5 mismatch trades, compare alert-side bar values vs backtest-side bar values at the trade exit second.

### #44 15 silent swing-stop strategies (sid 270 cluster)

**State:** Cluster of 15 strategies fired 0 alerts for ~24h during a known incident. Recovered overnight without restart.
**Question:** What heals these? Worker reload? Data Worker snapshot refresh? Need a clean repro before fixing.

---

## Open — engineering work

### #57C Engine write hooks for bar_diagnostics (tomorrow AM)

**Foundation shipped:** [b87f059](#shipped-today). Schema, manifest contract, UT Bot V4 seed in place.
**Next step:**
1. Apply `src/migrations/bar_diagnostics_table.sql` via Supabase SQL editor.
2. Add buffered writes in `unified_engine.run_backtest` (source='backtest' or 'algo' per `model_override`) and `ralph_engine` tick path (source='live').
3. Flush every 60 bars OR end-of-run. Non-fatal failures — never block hot path.

**Once running:** answers #53 in one trading session.

### #42 RTH session filter on injected primary_df (services.py:264)

**Possible contributor** to phantom-entry count. Live applies RTH filter; injected primary_df in services.py may not. If live skips a bar that backtest processes (or vice versa), state drifts from there.

---

## Quality-of-life UI

### #55 Split BT/Algo UAD buttons

**State:** "Update New Data" runs both lanes (~27s post-fix). Splitting into `Update New Backtest` + `Update New Algo` lets users pick the cheaper lane under scale strain. Defer until needed.

### #28/46 Admin observability dashboards

- **Strategy Health page** — 4-layer rollup with the cross-layer correlation table
- **Visual integrity dashboard** — fleet-wide worker/data health

### #58 #57C as a task (tomorrow's pickup)

---

## Linked artifacts

| Doc | Contents |
|---|---|
| `docs/SOP_Strategy_Health_Check.md` | The 4-layer methodology + active classifier buckets |
| `docs/Known_Bugs.md` | Active bug log |
| `src/_remeasure_pair_rates_5s.py` | Macro pair-rate measurement |
| `src/_divergence_walkthrough.py` | Layer 1 + 2B walkthrough |
| `src/_fill_delta_analysis.py` | Layer 3 fill_ts delta |
| `src/_verify_streaming_toggle.py` | Streaming-toggle observability |
| `src/migrations/system_settings_table.sql` | Applied 2026-06-04 |
| `src/migrations/update_jobs_table.sql` | Applied 2026-06-04 |
| `src/migrations/bar_diagnostics_table.sql` | **PENDING — apply before 57C** |
| `dev-backup-pre-streaming-toggle-test` | Pre-toggle-test snapshot |

---

## Run-order for next session

1. **Apply** `bar_diagnostics_table.sql` migration (Supabase SQL editor)
2. **Catchup**: trigger Update New Data on all 46 strategies (will be ~25 min total post-fix); let cron resume
3. **Re-baseline** 4-layer analysis with fresh BT data
4. **Ship #57C** engine write hooks (buffered, non-blocking)
5. **Probe #53**: pull bar_diagnostics for sid 302 over a known phantom cluster, plot live vs backtest `utv4_trailing_stop` side by side
6. If state drift confirmed → design the fix (snapshot serialization, periodic resync, etc.)
