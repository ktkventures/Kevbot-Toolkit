# Baseline Metrics — 2026-05-28 EOD (post-M8 ws_rest_spliced rollout)

> **Purpose:** Capture the bar-verification + divergence-backlog signal levels under
> `ws_rest_spliced` *as currently implemented* so that when we ship subsequent
> improvements (per-second splice, intra-bar stop fix, etc.) we can re-run the
> same analyses and measure delta.
>
> **Snapshot taken:** 2026-05-28 ~20:11Z (1h post-M8 rollout cb7eeaf)
>
> **Live model:** all 41 strategies on `ws_rest_spliced`. Latest deployed commit
> at snapshot time: `1bc4150` (SOP + tool + log). Live engine code as of `f95aa64`
> (exec_type stamp fix) + `cb7eeaf` (M8 registry promotion).
>
> **Filter conditions for snapshot:**
> - `apples_to_apples = TRUE` (drops events newer than min(last_alert_at, last_backtest_created_at))
> - `only_needs_investigation = TRUE` (excludes auto-classified known-cause events)
> - `live_model == 'ws_rest_spliced'` (excludes any pre-rollout alerts that snuck into the window)
> - Window: last 1 hour

## Bar verification fidelity (the headline analysis Kevin cares about)

Source: `alerts.verification_status` populated by `rest_verifier`. Each row =
one verification check.

### Fleet aggregate (since 18:30Z, 382 alerts on ws_rest_spliced)

| Status | Count | % |
|---|---|---|
| `verified` (REST close = WS close ±$0.01) | 334 | **87.4%** |
| `corrected` (drift detected, engine spliced REST) | 16 | 4.2% |
| `drift_uncorrected` (drift detected, bar stale, splice rejected) | 30 | 7.9% |
| `rest_unavailable` (REST never settled in window) | 0 | **0.0%** |
| `NULL` (in flight at snapshot time) | 2 | 0.5% |

### Per-strategy breakdown (≥5 alerts in window)

| sid | sym | TF | total | verified | drift_uncorrected | corrected | drift%+rest_unavail% |
|---|---|---|---|---|---|---|---|
| 149 | SPY | 10Sec | 18 | 16 | 2 | 0 | 11% |
| 150 | SPY | 10Sec | 55 | 48 | 7 | 0 | 13% |
| 151 | SPY | 10Sec | 78 | 73 | 5 | 0 | 6% |
| 152 | SPY | 1Min | 8 | 6 | 1 | 1 | 12% |
| 153 | SPY | 10Sec | 64 | 60 | 4 | 0 | 6% |
| 170 | SPY | 10Sec | 51 | 47 | 4 | 0 | 8% |
| 171 | SPY | 10Sec | 51 | 47 | 4 | 0 | 8% |
| 172 | SPY | 10Sec | 53 | 50 | 3 | 0 | 6% |

### Coverage gap (non-SPY strategies)

**No non-SPY strategies fired alerts in the snapshot window.** 30 strategies on
TSLA, AAPL, META, AMD, TSLL across 1Min/5Min haven't yet produced data on
`ws_rest_spliced`. Those baselines remain to be captured when those strategies
naturally trigger (or via a future market session).

## Divergence backlog (trade-level pairing)

Source: `/api/admin/strategy-health/backlog` endpoint, last 1h, apples-to-apples
ON, only needs_investigation.

### Per-strategy phantom/missed rates

| sid | sym | TF | paired | phantom | missed | **phantom% of total alerts** |
|---|---|---|---|---|---|---|
| 150 | SPY | 10Sec | 17 | 23 | 1 | **56%** |
| 151 | SPY | 10Sec | 30 | 26 | 2 | **45%** |
| 153 | SPY | 10Sec | 30 | 25 | 2 | **44%** |
| 170 | SPY | 10Sec | 13 | 20 | 10 | **47%** |
| 171 | SPY | 10Sec | 16 | 16 | 11 | **37%** |
| 172 | SPY | 10Sec | 17 | 15 | 12 | **34%** |

**Fleet total in 1h window:** 76 needs_investigation events (42 phantom + 29 missed,
plus a few that shifted between sub-counts). After cluster-deduplication by
(timestamp, edge, type, exit_reason), this collapses to **43 distinct clusters**.

## Why the gap matters

The bar-verification numbers look healthy (87% verified, 0% rest_unavailable).
But the divergence backlog says ~45% of alerts don't pair with backtest. The
explanation per the 2026-05-28 investigation log (see
`Divergence_Investigation_Log_2026-05-28.md`):

1. **`drift_uncorrected_cascade`** — When a sub-minute bar can't be REST-spliced
   in time (engine rejects because the bar is no longer latest), the
   8% of bars affected create state divergence that propagates through subsequent
   bars. Even bars that themselves verify Δ=0.0 can produce phantom/missed
   alerts because the underlying indicator state was offset by the prior
   uncorrected drift.
2. **`live_intrabar_stop_missed`** — Live engine isn't firing intra-bar stops
   for some strategies (170/171/172 missing them; 151/153 producing them).
   stop_config comparison pending (Supabase restart 2026-05-28 ~21:26Z
   blocked the query).
3. **`cluster_duplicate`** — Shared-trigger strategies multiply the count.

## What we expect each future fix to move

| Fix | Bar verification expected impact | Divergence backlog expected impact |
|---|---|---|
| Per-second splice (`project_per_second_splice_idea`) | `drift_uncorrected` → near 0% on sub-minute TFs | `drift_uncorrected_cascade` phantoms eliminated |
| Intra-bar stop fix (pending stop_config investigation) | No direct effect (verification is bar-close) | `live_intrabar_stop_missed` missed events eliminated |
| Cluster dedup at endpoint | No effect | Count of `needs_investigation` rows drops by ~3x on shared-trigger clusters |

## How to re-snapshot after a fix

1. `python _divergence_walkthrough.py --window-hours 1 --denom`
2. Append a new section to this doc with the post-fix table + delta vs this baseline
3. Reference the commit hash of the fix in the section header

## Linked artifacts

- [SOP_Divergence_Investigation.md](SOP_Divergence_Investigation.md) — methodology
- [Divergence_Investigation_Log_2026-05-28.md](Divergence_Investigation_Log_2026-05-28.md) — cluster-level findings
- Memory: `project_ws_rest_spliced_canary` — rollout context
- Memory: `project_per_second_splice_idea` — the forward-looking architectural option
