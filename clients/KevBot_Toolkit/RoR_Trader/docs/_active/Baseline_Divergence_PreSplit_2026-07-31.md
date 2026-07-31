# Baseline — divergence / fidelity BEFORE the environment split

**Captured:** 2026-07-31T20:20:00Z → 2026-07-31T20:20:24Z  
**Board:** #264 (E) · pre-split ruler for V5 / #162 environment split  
**`dev` commit at capture:** `80ae7db0`  
**Instrument:** `api.routers.strategy_health.get_strategy_health` — the dashboard's own code path, called in-process. No new metric was invented.  
**Generators (re-run these):** `src/_divergence_baseline_264.py` (§0–§11, pairing) · `src/_lag_baseline_264.py` (§12, latency dispersion — appended 2026-07-31T20:48Z)

> **This artifact has TWO halves and they are one document on purpose.** §0–§11 answer *did the alert pair?*; §12 answers *when did the data arrive?* A split baseline invites re-running half of it, and the half most likely to be skipped is the one that catches the failure mode the other cannot see.

> Measurement SSOT: `docs/_active/Plan_Measurement_Trust.md`.
> This artifact is READ-ONLY output. Nothing was armed, recomputed or deployed to produce it.

## 0. The one number, and what it is not

**Fleet pooled paired% @10s over the settled primary window (`W1_settled_2d`) = 78.1%** (2815 paired / 276 phantom / 512 missed edges across 14 strategies carrying signal, out of 24 in the fleet).

That is the BEFORE. If the same invocation over the same window after the split returns materially different numbers **and the stored lane has not been re-trued in between**, the split changed something.

**It is a fidelity ruler, not a health certificate.** It measures ONE thing: do recorded alerts pair with stored backtest trades. §9 lists — at length and on purpose — what it does not measure. Read §9 before using this to clear anything. In particular §0–§11 would **not** have caught the 2026-07-27 signal (a latency dispersion change, p90 6.9s → 13.9s), which is close to the single most likely way an environment split goes wrong — **that is what §12 was added for.**

**The second number, from §12: `ws_agg` 1Min RTH write-lag over the SAME window — p50 3.41s / p90 4.72s, worst symbol p50 3.85s against the #118 6.0s tripwire (CLEAR).** Cite both numbers or neither: pairing intact with delivery degraded is a real state, and it is the state the 07-27 scare was in.

## 1. Capture conditions

| condition | value |
|---|---|
| Market session | **CLOSED** — capture began 2026-07-31T20:20:00Z (RTH close is 20:00Z). Post-close per the light-load rail. |
| `dev` commit | `80ae7db0` |
| Capture wall-clock | 24.5s |
| Strategies returned | 24 (fleet-wide, all users) |

### Straddle guard (#156)

A capture that straddles an intraday recompute yields a subset>superset artifact. In-flight recompute/mass-search work was sampled either side of the capture:

| probe | at | update_jobs in-flight | mass_searches in-flight |
|---|---|---|---|
| BEFORE capture | 2026-07-31T20:20:00Z | 0 | 0 |
| AFTER capture | 2026-07-31T20:20:24Z | 0 | 0 |

**Quiescence check — primary window re-captured and compared: ✅ IDENTICAL.** The primary window @10s is scored twice in one run and the per-strategy paired/phantom/missed triples are compared. Identical ⇒ the stored lane did not mutate under the capture, so these numbers are not a #156 straddle artifact. **If this ever reads DIVERGED, discard the run and re-capture on a quiet DB** — do not reconcile the two readings.

## 2. Windows measured

Absolute start/end — a rolling `window_hours` read is anchored to `now` and is **not** reproducible later. Do not change these when re-running for comparison.

| id | start (UTC) | end (UTC) | basis |
|---|---|---|---|
| `W1_settled_2d` | 2026-07-29T13:30:00Z | 2026-07-31T00:00:00Z | PRIMARY. Wed 07-29 + Thu 07-30 RTH, both through the 00:20Z nightly Update-All => settled, current-logic stored lane. This is the ruler. |
| `W2_today_unsettled` | 2026-07-31T13:30:00Z | 2026-07-31T20:00:00Z | Fri 07-31 RTH (capture day). Stored lane NOT yet through the nightly => UNSETTLED. Recorded for completeness; NOT comparable head-to-head. |
| `W3_settled_5d` | 2026-07-24T13:30:00Z | 2026-07-31T00:00:00Z | Wider N for the same settled basis (5 sessions). Smooths single-day regime noise in the fleet aggregate. |

## 3. Fleet aggregate (the headline BEFORE number)

`combined% = paired / (paired + phantom + missed)` over coverage-classified counts (`*_cov`), TBD-excluded — the dashboard's own definition (`strategy_health.py` combined_pct). Two ways to aggregate, both reported:
- **pooled** = sum all edges fleet-wide, then divide (high-N strategies dominate).
- **mean-of-strategies** = per-strategy combined%, averaged over strategies that had any edges (every strategy counts once).

| window | tol | strategies w/ edges | paired | phantom | missed | TBD | pooled combined% | mean-of-strategies% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `W1_settled_2d` | 10s | 14 | 2815 | 276 | 512 | 0 | **78.1** | 69.7 |
| `W1_settled_2d` | 60s | 14 | 2917 | 174 | 410 | 0 | **83.3** | 75.1 |
| `W2_today_unsettled` | 10s | 13 | 1219 | 195 | 205 | 0 | **75.3** | 71.8 |
| `W2_today_unsettled` | 60s | 13 | 1295 | 119 | 129 | 0 | **83.9** | 80.6 |
| `W3_settled_5d` | 10s | 21 | 8164 | 433 | 862 | 0 | **86.3** | 77.8 |
| `W3_settled_5d` | 60s | 21 | 8291 | 306 | 735 | 0 | **88.8** | 80.8 |

## 4. Per-strategy paired% — PRIMARY window (`W1_settled_2d`, 2026-07-29T13:30:00Z → 2026-07-31T00:00:00Z)

Sorted by combined%@10s ascending (worst first). Strategies with zero edges in the window are listed separately in §5 — they carry no signal and must not be read as 0%.

| sid | name | symbol | TF | lane (`data_source`) | fwd | paired | phantom | missed | TBD | **combined%@10s** | combined%@60s | red flags |
|---:|---|---|---|---|:--:|---:|---:|---:|---:|---:|---:|---|
| 340 | TSLA LONG 1Min Mass #31 | TSLA | 1Min | hifi | Y | 2 | 5 | 4 | 0 | **18.2** | 44.4 | snapshot_stale, phantom_alerts, missed_alerts |
| 309 | TSLA LONG 15Sec Mass #2 | TSLA | 15Sec | hifi | Y | 19 | 15 | 23 | 0 | **33.3** | 46.2 | snapshot_stale, phantom_alerts, missed_alerts |
| 194 | TSLL LONG 1Min Mass #30 | TSLL | 1Min | hifi | Y | 3 | 0 | 5 | 0 | **37.5** | 37.5 | snapshot_stale, missed_alerts |
| 310 | TSLA LONG 15Sec Mass #3 | TSLA | 15Sec | hifi | Y | 36 | 24 | 34 | 0 | **38.3** | 46.1 | snapshot_stale, phantom_alerts, missed_alerts |
| 267 | TSLA-CANARY-10s-LooseConf | TSLA | 10Sec | hifi | Y | 331 | 63 | 73 | 0 | **70.9** | 75.0 | snapshot_stale, phantom_alerts, missed_alerts |
| 321 | TSLA LONG 15Sec Mass #6 | TSLA | 15Sec | hifi | Y | 313 | 44 | 83 | 0 | **71.1** | 83.2 | snapshot_stale, phantom_alerts, missed_alerts |
| 271 | TEST-P2-multipack-10s-0601 | SPY | 10Sec | hifi | Y | 96 | 15 | 22 | 0 | **72.2** | 76.2 | snapshot_stale, phantom_alerts, missed_alerts |
| 269 | SPY-CANARY-1m-Control | SPY | 1Min | hifi | Y | 78 | 1 | 17 | 0 | **81.2** | 81.2 | snapshot_stale, phantom_alerts, missed_alerts |
| 263 | TSLA-CANARY-10s-NoConf | TSLA | 10Sec | hifi | Y | 545 | 33 | 79 | 0 | **83.0** | 86.4 | snapshot_stale, phantom_alerts, missed_alerts |
| 338 | TEST-COARSEGATE-10s-1d-TSL | TSLA | 10Sec | hifi | Y | 1245 | 71 | 166 | 0 | **84.0** | 88.3 | snapshot_stale, phantom_alerts, missed_alerts |
| 314 | TSLA LONG 15Sec Mass #22 | TSLA | 15Sec | hifi | Y | 111 | 5 | 5 | 0 | **91.7** | 93.3 | snapshot_stale, phantom_alerts, missed_alerts |
| 342 | SPY LONG 1Min Mass #392 | SPY | 1Min | hifi | Y | 15 | 0 | 1 | 0 | **93.8** | 93.8 | snapshot_stale, kpis_stale, data_refresh_stale, missed_alerts |
| 339 | TSLA LONG 30Sec Mass #1 | TSLA | 30Sec | hifi | Y | 14 | 0 | 0 | 0 | **100.0** | 100.0 | snapshot_missing, kpis_stale, data_refresh_stale |
| 136 | SPY LONG - Mass #11 [mirro | SPY | 1Min | hifi | Y | 7 | 0 | 0 | 0 | **100.0** | 100.0 | snapshot_stale |

## 5. Zero-edge strategies in the primary window (NO SIGNAL — not 0%)

10 strategies produced no paired/phantom/missed edges in `W1_settled_2d`. A post-split re-run must treat these as **unmeasured**, not as healthy and not as broken. If one of these starts producing edges after the split that is a *coverage* change, not a fidelity regression.

| sid | name | symbol | TF | lane | fwd | red flags |
|---:|---|---|---|---|:--:|---|
| 7 | AAPL LONG - Webhook Inboun | AAPL | 5Min | full | Y | no_baseline |
| 312 | TSLA LONG 15Sec Mass #21 | TSLA | 15Sec | hifi | Y | snapshot_stale, kpis_stale, data_refresh_stale, no_recent_trades |
| 313 | TSLA LONG 15Sec Mass #9 | TSLA | 15Sec | hifi | Y | snapshot_stale, kpis_stale, data_refresh_stale, no_recent_trades |
| 325 | TSLA LONG 30Sec Mass #4 | TSLA | 30Sec | hifi | Y | snapshot_stale, kpis_stale, data_refresh_stale |
| 330 | TSLA LONG 30Sec Mass #9 | TSLA | 30Sec | hifi | Y | snapshot_stale, kpis_stale, data_refresh_stale |
| 331 | TSLA LONG 30Sec Mass #19 | TSLA | 30Sec | hifi | Y | snapshot_stale, kpis_stale, data_refresh_stale |
| 341 | NVDA LONG 1Min Mass #152 | NVDA | 1Min | hifi | Y | snapshot_stale, kpis_stale, data_refresh_stale, no_recent_trades |
| 343 | NVDA LONG 1Min Mass #148 | NVDA | 1Min | hifi | Y | snapshot_stale, kpis_stale, data_refresh_stale, no_recent_trades |
| 344 | NVDA LONG 1Min Mass #157 | NVDA | 1Min | hifi | Y | snapshot_stale, kpis_stale, data_refresh_stale, no_recent_trades |
| 345 | NVDA LONG 1Min Mass #134 | NVDA | 1Min | hifi | Y | snapshot_stale |

## 6. Health lanes and red-flag inventory (primary window)

`data_source` is the lane tag. Red flags are the dashboard's own `red_flags` list — the health-lane signal, captured verbatim so a post-split run can diff the *distribution*, not just the paired%.

| lane (`data_source`) | strategies |
|---|---:|
| hifi | 23 |
| full | 1 |

| red flag | strategies |
|---|---:|
| `snapshot_stale` | 22 |
| `missed_alerts` | 12 |
| `kpis_stale` | 10 |
| `data_refresh_stale` | 10 |
| `phantom_alerts` | 10 |
| `no_recent_trades` | 5 |
| `snapshot_missing` | 1 |
| `no_baseline` | 1 |
| **strategies with ≥1 red flag** | **24 / 24** |

**⚠️ `snapshot_stale` is a KNOWN DEAD ALARM — do not read it as fleet ill-health.** It is computed from the deprecated `engine_snapshot_at`, whose legacy data-worker flush stopped ~2026-07-21 (board #157). The live signal is `shadow_heartbeats`; on 07-28 the proof was 22/22 heartbeats FRESH against 20 frozen snapshots. The fix (`RORT_HEALTH_HB_SUPERSEDES_SNAPSHOT`) is BUILT but NOT ARMED as of this capture, so the flag fires here. **Its near-fleet-wide presence in this baseline is expected and is not a pre-split defect.** If that flag gets armed before the post-split re-run, `snapshot_stale` will legitimately collapse toward zero — that is the fix landing, not the split fixing anything.

## 7. Capture-day window (UNSETTLED — context only)

`W2_today_unsettled` has not been through the 00:20Z nightly Update-All, so its stored backtest lane is the incrementally-built one. Per `Plan_Measurement_Trust` Phase B this is **not** a valid head-to-head basis; it is recorded so the capture day is not a hole in the record.

| sid | name | symbol | TF | paired | phantom | missed | TBD | combined%@10s |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 309 | TSLA LONG 15Sec Mass #2 | TSLA | 15Sec | 19 | 11 | 19 | 0 | 38.8 |
| 310 | TSLA LONG 15Sec Mass #3 | TSLA | 15Sec | 18 | 10 | 6 | 0 | 52.9 |
| 340 | TSLA LONG 1Min Mass #31 | TSLA | 1Min | 5 | 3 | 1 | 0 | 55.6 |
| 314 | TSLA LONG 15Sec Mass #22 | TSLA | 15Sec | 155 | 47 | 37 | 0 | 64.9 |
| 136 | SPY LONG - Mass #11 [mirro | SPY | 1Min | 2 | 0 | 1 | 0 | 66.7 |
| 321 | TSLA LONG 15Sec Mass #6 | TSLA | 15Sec | 156 | 36 | 40 | 0 | 67.2 |
| 345 | NVDA LONG 1Min Mass #134 | NVDA | 1Min | 5 | 1 | 1 | 0 | 71.4 |
| 267 | TSLA-CANARY-10s-LooseConf | TSLA | 10Sec | 173 | 33 | 19 | 0 | 76.9 |
| 271 | TEST-P2-multipack-10s-0601 | SPY | 10Sec | 103 | 12 | 15 | 0 | 79.2 |
| 338 | TEST-COARSEGATE-10s-1d-TSL | TSLA | 10Sec | 259 | 27 | 33 | 0 | 81.2 |
| 263 | TSLA-CANARY-10s-NoConf | TSLA | 10Sec | 269 | 15 | 29 | 0 | 85.9 |
| 269 | SPY-CANARY-1m-Control | SPY | 1Min | 51 | 0 | 4 | 0 | 92.7 |
| 194 | TSLL LONG 1Min Mass #30 | TSLL | 1Min | 4 | 0 | 0 | 0 | 100.0 |

## 8. Flag surface at this commit

**135** distinct `RORT_*` flag names are referenced in `src/` at `80ae7db0`; **58** have a literal in-code default.

> **🚨 THIS IS NOT AN ARMED-FLAG INVENTORY.** These are the flag names the code knows and their CODE defaults. The ARMED values live in Railway service variables, and reading them (`railway variables`) is outside what this headless capture is permitted to do. **Board #265 owns the real armed inventory and this baseline is INCOMPLETE without it.** Two fleets can share this identical flag surface and still behave differently because one has a flag armed. Treat the list below as a vocabulary fingerprint only.

<details><summary>In-code defaults (58)</summary>

| flag | code default |
|---|---|
| `RORT_ALERT_LAG_WARN_S` | `30` |
| `RORT_BAR_DUP_GUARD` | `0` |
| `RORT_CANONICAL_FINE_TF_STATE` | `0` |
| `RORT_CANONICAL_PRIMARY_CLOSE` | `0` |
| `RORT_CANONICAL_SUBMIN_STATE` | `0` |
| `RORT_COARSE_SECONDARY_FROM_1MIN` | `1` |
| `RORT_COMPUTE_REMOTE` | `1` |
| `RORT_ENFORCE_1MIN_GATE` | `0` |
| `RORT_ENGINE_BOOT_DEADLINE_S` | `1200` |
| `RORT_ENGINE_FEED_MARKER_FILE` | `/tmp/engine_last_tick` |
| `RORT_ENGINE_HEARTBEAT_FILE` | `/tmp/engine_alive` |
| `RORT_ENGINE_HEARTBEAT_MAX_AGE_S` | `300` |
| `RORT_ENGINE_STALL_WARN_S` | `15` |
| `RORT_FEED_STALE_CAP_S` | `7200` |
| `RORT_FEED_STALE_S` | `900` |
| `RORT_GATE_FAIL_CLOSED` | `0` |
| `RORT_GRACE_FINAL_CLOSE_ELIGIBLE` | `0` |
| `RORT_HEALTHCHECK_BOUNDED_BOOT` | `0` |
| `RORT_HEALTHCHECK_FEED_FRESHNESS` | `0` |
| `RORT_HIFI_INCREMENTAL_LOAD` | `1` |
| `RORT_INTERP_AWARE_SHADOWS` | `0` |
| `RORT_MTF_COARSE_RTH_RELOAD` | `0` |
| `RORT_MTF_FINE_INCREMENTAL_AUTHORITY` | `0` |
| `RORT_MTF_PB_DEFER` | `0` |
| `RORT_MTF_PB_PREV_EPOCH` | `0` |
| `RORT_MTF_SESSION_SHADOWS` | `0` |
| `RORT_MTF_STATE_REFRESH_S` | `0` |
| `RORT_PRIMARY_STATE_RESYNC_S` | `0` |
| `RORT_RECOMPUTE_STALL_SECONDS` | `25` |
| `RORT_RESAMPLED_STORE_SERVE` | `0` |
| `RORT_RIGHTSIZE_WARMUP` | `1` |
| `RORT_SESSION_LABEL_GATE` | `0` |
| `RORT_SHADOW_BOOTSTRAP_DAYS` | `1` |
| `RORT_SHADOW_CONFIG_CACHE_TTL_S` | `0` |
| `RORT_SHADOW_DEEP_ANCHOR` | `0` |
| `RORT_SHADOW_DEEP_ANCHOR_MAX_DAYS` | `90` |
| `RORT_SHADOW_DEEP_ANCHOR_MULT` | `4` |
| `RORT_SHADOW_EMPTY_PROBE` | `1` |
| `RORT_SHADOW_GAP_SKIP` | `1` |
| `RORT_SHADOW_HEARTBEAT` | `1` |
| `RORT_SHADOW_KPI_ASYNC` | `1` |
| `RORT_SHADOW_KPI_DEBOUNCE_S` | `300` |
| `RORT_SHADOW_MAX_ADVANCE_S` | `0` |
| `RORT_SHADOW_PERSIST_SNAPSHOT` | `1` |
| `RORT_SHADOW_PROVISIONAL` | `0` |
| `RORT_SHADOW_PROVISIONAL_S` | `60` |
| `RORT_SHADOW_READ_ONLY_BARS` | `1` |
| `RORT_SHADOW_RECOMPUTE_KPIS` | `1` |
| `RORT_SHADOW_RESIDENT_FRAME` | `0` |
| `RORT_SHADOW_RETRUE_FORCE_FULL` | `0` |
| `RORT_SHADOW_WARMUP_BARS` | `300` |
| `RORT_SUBMIN_DERIVE_BARS` | `3000` |
| `RORT_SUPPRESS_EOD_REENTRY` | `0` |
| `RORT_TEST_HANG_MODE` | `wedge` |
| `RORT_TEST_HANG_SID` | `311` |
| `RORT_TF_LABEL_SEC_FIX` | `0` |
| `RORT_WARMUP_PREV_CHAIN` | `0` |
| `RORT_WORKER_HEARTBEAT_FILE` | `/tmp/worker_alive` |

</details>

<details><summary>Full flag-name surface (135)</summary>

```
RORT_ALERT_LAG_WARN_S  RORT_APPEND_SUPERSEDE_OPEN  RORT_BACKTEST_LANE_MODE
RORT_BARCACHE_WRITETHROUGH  RORT_BAR_DUP_GUARD  RORT_BOARD_API_URL
RORT_BOARD_SERVICE_KEY  RORT_CANONICAL_FINE_TF_STATE  RORT_CANONICAL_PRIMARY_CLOSE
RORT_CANONICAL_SUBMIN_STATE  RORT_COARSE_LAYER_READ  RORT_COARSE_SECONDARY_FROM_1MIN
RORT_COMPUTE_REMOTE  RORT_ENFORCE_1MIN_GATE  RORT_ENGINE_BOOT_DEADLINE_S
RORT_ENGINE_BOOT_MARKER_FILE  RORT_ENGINE_FEED_MARKER_FILE  RORT_ENGINE_HEARTBEAT_FILE
RORT_ENGINE_HEARTBEAT_MAX_AGE_S  RORT_ENGINE_STALL_WARN_S  RORT_FEED_STALE_CAP_S
RORT_FEED_STALE_S  RORT_FETCH1S_FROM_CACHE  RORT_GATE_FAIL_CLOSED
RORT_GRACE_FINAL_CLOSE_ELIGIBLE  RORT_HEALTHCHECK_BOUNDED_BOOT  RORT_HEALTHCHECK_FEED_FRESHNESS
RORT_HEALTH_HB_SUPERSEDES_SNAPSHOT  RORT_HIFI_INCREMENTAL_LOAD  RORT_HIFI_NO_PREBAR_FILL
RORT_HOTRELOAD_BOOT_PARITY  RORT_INTERP_AWARE_SHADOWS  RORT_MASS_PARTIAL_FLUSH_SEC
RORT_MASS_RECOVER_QUEUED  RORT_MASS_SCOPE_CONFLUENCE  RORT_MTF_COARSE_RTH_RELOAD
RORT_MTF_FINE_INCREMENTAL_AUTHORITY  RORT_MTF_PB_DEFER  RORT_MTF_PB_PREV_EPOCH
RORT_MTF_SESSION_SHADOWS  RORT_MTF_STATE_REFRESH  RORT_MTF_STATE_REFRESH_S
RORT_NIGHTLY_RECOMPUTE  RORT_NIGHTLY_RECOMPUTE_AT  RORT_NIGHTLY_SETTLE_RETRUE
RORT_NIGHTLY_SETTLE_RETRUE_WINDOW  RORT_NIGHTLY_SIDS  RORT_PARITY_SELF_HEAL
RORT_PARITY_SETTLED_MIN_TDAYS  RORT_POLL_BUDGET_FLAGON_S  RORT_POLL_BUDGET_S
RORT_PREP_BAR_COUNT_WARMUP  RORT_PREP_WARMUP_SUBMIN_CAP_DAYS  RORT_PRIMARY_STATE_RESYNC_APPLY
RORT_PRIMARY_STATE_RESYNC_S  RORT_PRIME1S_BREAKER_COOLDOWN_S  RORT_PRIME1S_CACHE_MAX_ROWS
RORT_PRIME1S_CHUNKED  RORT_PRIME1S_CHUNK_DAYS  RORT_RECOMPUTE_PARALLELISM
RORT_RECOMPUTE_STALL_SECONDS  RORT_RECOMPUTE_SUPERVISOR  RORT_REPO_ROOT
RORT_RESAMPLED_STORE_COMPARE  RORT_RESAMPLED_STORE_MAINTAIN_OFFHOURS_S  RORT_RESAMPLED_STORE_MAINTAIN_S
RORT_RESAMPLED_STORE_READ  RORT_RESAMPLED_STORE_SERVE  RORT_RESAMPLED_STORE_SERVE_LIVE
RORT_RESAMPLED_STORE_WRITE  RORT_RESUME_INHERIT_POSITION  RORT_RIGHTSIZE_WARMUP
RORT_RULES_ROOT  RORT_SCOPE_CONFLUENCE_GROUPS  RORT_SECONDARY_TF_CACHE
RORT_SECONDARY_TF_SNAPSHOT  RORT_SEED_MTF_FROM_WARMUP  RORT_SESSION_LABEL_GATE
RORT_SETTLE_SWEEPER  RORT_SETTLE_SWEEP_INTERVAL_S  RORT_SETTLE_SWEEP_LAG_MIN
RORT_SETTLE_SWEEP_LOOKBACK_MIN  RORT_SHADOW_BOOTSTRAP_DAYS  RORT_SHADOW_CONFIG_CACHE_TTL_S
RORT_SHADOW_DEEP_ANCHOR  RORT_SHADOW_DEEP_ANCHOR_MAX_DAYS  RORT_SHADOW_DEEP_ANCHOR_MULT
RORT_SHADOW_DRY_RUN  RORT_SHADOW_EMPTY_PROBE  RORT_SHADOW_FAIR_ORDER
RORT_SHADOW_GAP_SKIP  RORT_SHADOW_HEARTBEAT  RORT_SHADOW_KPI_ASYNC
RORT_SHADOW_KPI_DEBOUNCE_S  RORT_SHADOW_MAX_ADVANCE_S  RORT_SHADOW_PASS_TIMEOUT_S
RORT_SHADOW_PERSIST_SNAPSHOT  RORT_SHADOW_POLL_WORKERS  RORT_SHADOW_PROVISIONAL
RORT_SHADOW_PROVISIONAL_S  RORT_SHADOW_READ_ONLY_BARS  RORT_SHADOW_RECOMPUTE_KPIS
RORT_SHADOW_RELOAD_S  RORT_SHADOW_RESIDENT_FRAME  RORT_SHADOW_RETRUE_FORCE_FULL
RORT_SHADOW_SETTLE_MIN  RORT_SHADOW_SHARD  RORT_SHADOW_SIDS
RORT_SHADOW_WARMUP_BARS  RORT_SKIP_ALGO_COLD_SEED  RORT_SOME_NEW_PROD_FLAG
RORT_SR_CHANNELS_BOGUS  RORT_SR_CHANNELS_PINE  RORT_SUBMIN_DERIVE_BARS
RORT_SUBMIN_MAINTAIN_RECENT_DAYS  RORT_SUBMIN_SEED_DAYS  RORT_SUBMIN_STORE_SYMBOLS
RORT_SUPPRESS_EOD_REENTRY  RORT_T0_HEALTH_SNAPSHOT  RORT_T0_HEALTH_SNAPSHOT_AT_UTC
RORT_T0_HEALTH_SNAPSHOT_LATE_MIN  RORT_T0_HEALTH_SNAPSHOT_POLL_S  RORT_T0_HEALTH_SNAPSHOT_TOLS
RORT_T0_HEALTH_SNAPSHOT_WINDOW_H  RORT_TEST_HANG_MODE  RORT_TEST_HANG_SID
RORT_TF_LABEL_SEC_FIX  RORT_TF_SCALED_WARMUP  RORT_UAD_GUARD_EMPTY_LANE
RORT_UAD_PRESERVE_RANGE  RORT_UPDATE_ALL_SKIP_ALGO  RORT_USE_DOUGH_CACHE
RORT_VALIDATE_WARMUP_FLAG  RORT_WARMUP_PREV_CHAIN  RORT_WORKER_HEARTBEAT_FILE
```

</details>

## 9. What this baseline does NOT cover

**A baseline that overclaims is worse than none**, because the after-comparison reads as a regression that was never actually measured. Every item below is a hole. If the post-split question falls in one of these, this artifact cannot answer it and must not be cited as if it could.

**1. Latency and its dispersion — the biggest hole. → NARROWED 2026-07-31 by §12.** This measures *whether* an alert paired, not *when* it arrived. The 07-27 divergence scare was a p90 write-lag change (6.9s → 13.9s) with pairing largely intact. The gateway split's own design doc names symmetric sub-second delivery as the load-bearing property. **§0–§11 do not measure delivery latency at all** — still true, and still the reason to read §12 beside them. The `live_bars` `written_at`-lag distribution capture called for here **is now part of this artifact as §12** (generator `src/_lag_baseline_264.py`; tripwire definition board #118). It is *narrowed*, not closed: §12 instruments the `live_bars` write hop only — not end-to-end alert latency — and establishes a new series with no continuity to the 07-27 figures. See §12.9 for §12's own does-NOT-cover list.

**2. The replay ceiling — no ops-vs-logic splitter.** `/replay-check` (`src/replay_harness.py`) is the instrument that separates operational loss from a real logic residual from the WS/REST floor. It is a multi-minute *per-strategy* offline job and was not run fleet-wide here. Consequence: a post-split drop in these numbers **cannot be attributed** to logic vs operations from this artifact alone. To close this hole, per sid:

```
cd clients/KevBot_Toolkit/RoR_Trader/src
../.venv/bin/python replay_harness.py --sids 263,267,338 \
    --since 2026-07-29T13:30:00Z --until 2026-07-31T00:00:00Z
```

**3. The stored backtest lane is MUTABLE — this is the deepest caveat.** The 'backtest' half of every number here is the *stored* lane, and a later recompute / nightly re-true rewrites it. So re-running the identical window months from now can return different numbers **with no split and no code change**. The method is reproducible; the referent is not frozen. Mitigation for the post-split run: re-run this capture immediately BEFORE the split cutover as well, so the before/after pair is close in time and lane-mutation has less room to confound. (Related: #156 capture-mutation artifact.)

**4. Alert lane only — NOT the algo lane.** `get_strategy_health` pairs recorded ALERTS against stored backtest trades. The algo/execution lane is a different lane and is not scored here. 'Health = backtest↔ALERTS' is the standing definition; do not read these as execution fidelity.

**5. Bar construction and gates are not verified.** No `/bar-parity` read, no gate-parity view. Even the replay ceiling (item 2) is a bar/engine replay, not a provider-event replay: it consumes already-aggregated `live_bars.first_*` and bypasses WS parsing, BarBuilder construction, reconnect/dup/correction ordering, and grace firing. **Nothing in this baseline certifies that bars were CONSTRUCTED or ROUTED correctly** — precisely the layer a market-data gateway relocates.

**6. Thin and narrow coverage.** 14 strategies carry signal in the primary window out of 24 in the fleet; 10 are silent (§5) and are therefore UNMEASURED, not healthy. Symbols reduce to roughly TSLA/SPY/NVDA/TSLL. Several signal-carrying strategies have single-digit edge counts (e.g. sid 340 at 11 edges, sid 194 at 8) where a one-trade difference swings combined% by tens of points — **do not read those rows as trend**.

**7. No armed-flag inventory (§8).** Same-code, different-flags is a real and likely post-split failure mode, and it is invisible to this artifact.

**8. No P&L, KPI, or portfolio comparison.** Fidelity only. A split that preserved pairing but changed fills would pass this baseline.

**9. One capture, no run-to-run variance estimate.** There is no repeat-capture here establishing how much these numbers move day-to-day with nothing changed. Without that, a small post-split delta is **not** interpretable as signal. The three windows (2d / 5d / capture-day) give a weak sense of spread — note the pooled @10s figure ranges 75.3–86.3 across them — but that is window-composition variance, not run-to-run noise. **Establishing a real noise floor needs the capture re-run on several settled days before the split.** Until then, treat anything under roughly 5 points as inconclusive.

## 10. How to re-run this after the split

The entire value of this artifact is that the identical method runs again and the two are compared. If you cannot reproduce the invocation, this is decoration.

```
cd clients/KevBot_Toolkit/RoR_Trader/src
../.venv/bin/python _divergence_baseline_264.py \
    --dev-sha $(git rev-parse --short HEAD) \
    --out ../docs/_active/Baseline_Divergence_PostSplit_<YYYY-MM-DD>.md
```

Rules for the re-run, in priority order:

1. **Do NOT edit `WINDOWS` in the script.** The absolute windows ARE the comparison. Changing them silently invalidates the diff. If a newer window is wanted, ADD one; never modify or remove `W1_settled_2d`.
2. **Run post-close on a quiet DB** and check the §1 straddle-guard probe reads 0 in-flight on BOTH sides. A straddled capture is not comparable (#156).
3. **Run it against BOTH fleets** once the split exists — one capture per environment — and diff those against each other as well as against this file. Same-session, same-bars is the whole point of the split; a cross-day comparison reintroduces exactly the confound the split was built to remove.
4. **Record the armed flags for both fleets** (§8 hole, board #265). A paired% difference with different armed flags is uninterpretable.
5. **Compare @10s first.** 60s hides real timing divergence and will make a latency regression look like a non-event.
6. Read §9 before declaring anything proved.

## 11. Timings (capture cost, for planning the re-run)

| window | tol | seconds |
|---|---:|---:|
| `W1_settled_2d` | 10s | 3.5 |
| `W1_settled_2d` | 60s | 3.2 |
| `W2_today_unsettled` | 10s | 3.2 |
| `W2_today_unsettled` | 60s | 3.3 |
| `W3_settled_5d` | 10s | 3.6 |
| `W3_settled_5d` | 60s | 3.5 |
| **total** | | **24.5** |

Cheap enough to re-run daily. Given §9 item 9, **it should be** — several pre-split settled-day captures are what turns a single number into a noise floor.


## 12. Latency-dispersion baseline — `live_bars` `written_at` lag

**Captured:** 2026-07-31T20:45:21Z · **`dev` commit:** `759af1d7` · **Generator (re-run this):** `src/_lag_baseline_264.py`

> **Why this section exists.** §9 item 1 named latency dispersion as this baseline's biggest hole, and M inserted a step to close it. §0–§11 measure *whether* an alert paired; this section measures *when the data arrived*. They are deliberately in ONE artifact — a split baseline invites re-running half of it.

### 12.0 The numbers

| lane | n | **p50** | **p90** | p99 | max | mean |
|---|---:|---:|---:|---:|---:|---:|
| `ws_agg` 1Min (the #118 lane) | 2900 | **3.41s** | **4.72s** | 77.84s | 317.05s | 6.38s |
| `ws` 10Sec (the direct-WS control) | 8557 | **3.22s** | **3.72s** | 49.20s | 240.29s | 4.84s |

**#118 tripwire (ws_agg 1Min p50 > 6s RTH): worst symbol = 3.85s — 🟢 CLEAR.** That is the BEFORE. If the same invocation after the split returns materially different numbers, the split changed delivery.

**Read p90, not p50.** The centre of this distribution barely moves — five settled sessions (§12.5) put the `ws_agg` 1Min p50 inside a ~0.2s band. The 07-27 scare was a **p90** move (6.9s → 13.9s) that left p50 and pairing largely intact, which is precisely why §0–§11 would have read that day as healthy. p90 is where this section earns its keep.

### 12.1 What is measured (exact definition)

```
lag_s = written_at - (bar_start + timeframe_seconds)
      = seconds after the bar CLOSED that the row first landed in the DB
```

- **`written_at`, not `last_updated_at`.** `written_at` is `NOT NULL DEFAULT now()`; `live_bars_writer._write_bar_sync` never includes it in the upsert payload and no trigger touches it on UPDATE, so it holds **first arrival** and survives Polygon rebroadcasts. `last_updated_at` is bumped by `live_bars_preserve_first_trg` on every rebroadcast — that pair is reported separately in §12.6 as *settle duration*.
- **This is the #118 tripwire's own definition**, reused verbatim: *watch `live_bars` ws_agg 1Min `written_at - (bar_start+60s)` per symbol; **med >6s RTH = saturation returning***.
- **`ws` and `ws_agg` are different producers, not two views of one thing.** `ws` = bars Polygon delivers directly (the 10/15/30Sec fleet TFs, plus a sparse scatter of coarse rows); `ws_agg` = bars the worker aggregates client-side and closes on the wall-clock boundary (1Min and coarser). #118 was a `ws_agg`-only pathology: the boundary burst is what saturates, and the direct `ws` path stayed clean throughout. **Any post-split comparison must keep them separate** or a `ws_agg` regression will be diluted by clean `ws` rows.
- **RTH = `bar_start` time-of-day in `[13:30Z, 20:00Z)` on Mon–Fri.** Correct for July (ET = UTC-4). ⚠️ A re-run between November and March must widen to `14:30Z–21:00Z` or it will silently measure the wrong hours.

**The referent is IMMUTABLE — this is a stronger baseline than §0–§11.** §9 item 3 warns that the stored backtest lane is mutable, so re-reading the same window later can return different pairing numbers with no split and no code change. Nothing rewrites `written_at`. The only way this population moves is a *late INSERT* of a bar that was previously absent (`rest_insert` / `rest_correction` / `rest_backfill`), which is exactly why every table below is scoped **by `source`** and never pooled across sources.

### 12.2 Capture conditions

| condition | value |
|---|---|
| Market session | **CLOSED** — captured after 20:00Z on 2026-07-31, same post-close / quiet-DB rail as the §1 pairing capture |
| Instrument | one aggregate `SELECT` per section over `live_bars`, percentiles computed **server-side** (`percentile_cont`) — no bulk row transfer, negligible DB load |
| Mutations | **none.** Read-only; nothing armed, recomputed or deployed |
| `dev` commit | `759af1d7` |
| Window | `W1_settled_2d` = 2026-07-29T13:30:00Z → 2026-07-31T00:00:00Z — **the same absolute window as §3–§6** |

No straddle guard is needed here (contrast §1): a recompute cannot mutate `written_at`, so there is no subset>superset failure mode for this metric.

### 12.3 PRIMARY — `W1_settled_2d` RTH, by source × timeframe

All figures in **seconds after bar close**. Sorted by source then TF.

| source | TF | n | min | p50 | p90 | p99 | max | mean | note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `rest_backfill` | 1Min | 9 | 272.04 | **284.41** | **1000.52** | 1100.89 | 1112.04 | 460.15 | low-n |
| `rest_backfill` | 3Min | 3 | 276.94 | **280.42** | **286.30** | 287.62 | 287.77 | 281.71 | low-n |
| `rest_backfill` | 2Hour | 1 | 1081.71 | **1081.71** | **1081.71** | 1081.71 | 1081.71 | 1081.71 | low-n |
| `rest_correction` | 10Sec | 69 | 3.11 | **3.21** | **3.47** | 16.95 | 44.12 | 3.86 | low-n |
| `rest_correction` | 15Sec | 63 | 3.12 | **3.28** | **3.55** | 3.91 | 4.06 | 3.33 | low-n |
| `rest_correction` | 30Sec | 1 | 3.26 | **3.26** | **3.26** | 3.26 | 3.26 | 3.26 | low-n |
| `rest_correction` | 1Min | 1 | 3.11 | **3.11** | **3.11** | 3.11 | 3.11 | 3.11 | low-n |
| `rest_insert` | 10Sec | 392 | 3.02 | **35.04** | **92.89** | 229.02 | 258.81 | 46.91 |  |
| `rest_insert` | 15Sec | 87 | 3.14 | **33.58** | **71.45** | 239.65 | 252.02 | 45.42 | low-n |
| `rest_insert` | 30Sec | 14 | 3.27 | **33.63** | **185.19** | 275.46 | 279.27 | 55.93 | low-n |
| `rest_insert` | 1Min | 70 | 3.12 | **61.73** | **121.46** | 127.80 | 139.43 | 67.27 | low-n |
| `rest_insert` | 2Min | 1 | 122.49 | **122.49** | **122.49** | 122.49 | 122.49 | 122.49 | low-n |
| `warmup_seed` | 10Sec | 341 | 2.07 | **70.78** | **506.95** | 694.01 | 6899.20 | 195.76 |  |
| `warmup_seed` | 15Sec | 148 | 1.60 | **94.26** | **474.71** | 666.21 | 1328.75 | 173.25 |  |
| `warmup_seed` | 30Sec | 91 | 3.33 | **102.58** | **461.09** | 674.09 | 701.09 | 168.57 | low-n |
| `warmup_seed` | 1Min | 140 | 3.45 | **88.64** | **373.13** | 603.59 | 632.79 | 139.35 |  |
| `warmup_seed` | 2Min | 29 | 5.02 | **75.75** | **439.30** | 568.53 | 573.50 | 158.81 | low-n |
| `warmup_seed` | 3Min | 18 | 4.49 | **68.70** | **330.79** | 461.76 | 468.26 | 130.25 | low-n |
| `warmup_seed` | 5Min | 9 | 3.63 | **123.57** | **492.07** | 530.70 | 534.99 | 188.33 | low-n |
| `warmup_seed` | 10Min | 9 | 17.84 | **78.02** | **211.18** | 253.89 | 258.63 | 99.84 | low-n |
| `warmup_seed` | 15Min | 1 | 45.40 | **45.40** | **45.40** | 45.40 | 45.40 | 45.40 | low-n |
| `warmup_seed` | 30Min | 1 | 149.87 | **149.87** | **149.87** | 149.87 | 149.87 | 149.87 | low-n |
| `warmup_seed` | 1Hour | 3 | 63.20 | **133.97** | **163.86** | 170.58 | 171.33 | 122.83 | low-n |
| `ws` | 10Sec | 8557 | 1.69 | **3.22** | **3.72** | 49.20 | 240.29 | 4.84 |  |
| `ws` | 15Sec | 2821 | 3.09 | **3.25** | **3.91** | 56.72 | 245.16 | 5.04 |  |
| `ws` | 30Sec | 1454 | 3.23 | **3.46** | **4.33** | 65.75 | 255.20 | 5.70 |  |
| `ws` | 2Min | 48 | 0.46 | **6.00** | **99.00** | 205.62 | 209.57 | 36.01 | low-n |
| `ws` | 3Min | 25 | 0.09 | **40.14** | **102.43** | 208.32 | 210.61 | 53.55 | low-n |
| `ws` | 5Min | 13 | 10.62 | **44.92** | **186.20** | 208.64 | 210.65 | 78.76 | low-n |
| `ws` | 10Min | 5 | 41.52 | **74.61** | **211.00** | 216.91 | 217.56 | 116.00 | low-n |
| `ws` | 15Min | 1 | 188.49 | **188.49** | **188.49** | 188.49 | 188.49 | 188.49 | low-n |
| `ws` | 30Min | 1 | 217.45 | **217.45** | **217.45** | 217.45 | 217.45 | 217.45 | low-n |
| `ws` | 1Hour | 3 | 190.62 | **210.76** | **254.71** | 264.59 | 265.69 | 222.36 | low-n |
| `ws_agg` | 1Min | 2900 | 2.46 | **3.41** | **4.72** | 77.84 | 317.05 | 6.38 |  |
| `ws_agg` | 2Min | 702 | 0.09 | **1.17** | **2.08** | 13.16 | 314.38 | 2.78 |  |
| `ws_agg` | 3Min | 474 | 0.08 | **1.22** | **2.11** | 12.89 | 262.67 | 2.83 |  |
| `ws_agg` | 5Min | 290 | 0.10 | **1.29** | **2.15** | 7.67 | 62.56 | 1.90 |  |
| `ws_agg` | 10Min | 142 | 0.33 | **1.76** | **8.21** | 45.87 | 62.69 | 3.66 |  |
| `ws_agg` | 15Min | 50 | 0.09 | **0.88** | **1.65** | 2.77 | 3.61 | 0.95 | low-n |
| `ws_agg` | 30Min | 24 | 0.93 | **5.66** | **15.64** | 20.94 | 21.98 | 6.91 | low-n |
| `ws_agg` | 1Hour | 30 | 0.67 | **6.24** | **18.16** | 28.09 | 30.54 | 8.31 | low-n |
| `ws_agg` | 2Hour | 5 | 9.32 | **16.60** | **20.34** | 21.93 | 22.10 | 15.92 | low-n |
| `ws_agg` | 4Hour | 4 | 6.97 | **19.98** | **36.09** | 39.78 | 40.19 | 21.78 | low-n |

**Reading that table:**

- **Only `ws` and `ws_agg` are delivery latency.** `warmup_seed`, `rest_insert`, `rest_correction` and `rest_backfill` are gap-fill/seed writes whose lag records *when a job ran*, not how fast data arrived. Their large numbers here are correct and expected. Do not put them in a verdict (§12.9 hole 5).
- **The coarse `ws_agg` TFs (2Min+) read LOWER than `ws_agg` 1Min, and that is not a paradox.** §12.6 shows those rows are ~100% rebroadcast: the coarse bar is written promptly when its boundary passes and then updated as it settles, so `written_at` is early by construction. **`ws_agg` 1Min is the lane the fleet actually trades on and the lane #118 named — it is the ruler; the coarse rows are context.**
- **`ws` sub-minute has a hard floor around 3.1s.** min, p50 and p90 all sit just above 3s across every sub-minute row. That floor is a property of the current pipeline, not of the market, and it is the single most useful thing to re-measure after the split: **a gateway that preserves pairing but moves this floor has changed the system in a way §0–§11 cannot see.**

### 12.4 #118 tripwire read — `ws_agg` 1Min per symbol, RTH, `W1_settled_2d`

**Threshold: p50 > 6.0s RTH = minute-boundary saturation returning** (board #118). June-2026 healthy baseline was 4.0–4.8s; the 07-08→07-21 degraded era ran 12.6–12.7s; post-cull sessions have run 3.2–3.5s.

| symbol | n | min | p50 | p90 | p99 | max | mean | p50 vs 6s |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| NVDA | 727 | 3.08 | **3.35** | **3.83** | 67.33 | 315.72 | 6.11 | 🟢 under |
| SPY | 722 | 2.95 | **3.29** | **3.85** | 73.74 | 314.76 | 6.10 | 🟢 under |
| TSLA | 727 | 2.46 | **3.31** | **4.12** | 86.70 | 317.05 | 6.19 | 🟢 under |
| TSLL | 724 | 3.09 | **3.85** | **7.16** | 75.64 | 312.73 | 7.11 | 🟢 under |

**Where today sits: worst per-symbol p50 = 3.85s against the 6.0s tripwire — 🟢 CLEAR.** This is the pre-split reading. A post-split run that moves a symbol from under to over has reproduced #118 on the new topology.

**Two pre-split facts in that table, stated so a post-split reader does not rediscover them as regressions:**

1. **`mean` sits far above `p50` for every symbol** NVDA 6.11 vs 3.35, SPY 6.10 vs 3.29, TSLA 6.19 vs 3.31, TSLL 7.11 vs 3.85. The distribution is **already heavy-tailed pre-split** — a small minority of minutes land tens of seconds late (see the p99/max columns). That is the BEFORE state, not a defect introduced later.
2. **Symbol dispersion is not uniform: TSLL p90 = 7.16s against NVDA at 3.83s**, a 3.33s spread across symbols on identical infrastructure. A per-fleet comparison after the split must compare **like symbol to like symbol**; pooling symbols would let a single-symbol regression hide inside this existing spread.

### 12.5 Day-to-day spread of the same metric (the poor man's noise floor)

§9 item 9 says a single capture gives no run-to-run variance estimate, so a small post-split delta is uninterpretable. For THIS metric the sessions inside `W3_settled_5d` give a usable spread for free. **A post-split move smaller than the spread below is not signal.**

**`ws_agg` / 1Min — one row per RTH session**

| session (UTC date) | n | min | p50 | p90 | p99 | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-07-24 | 1552 | 2.86 | **3.53** | 4.24 | 33.61 | 79.84 | 4.43 |
| 2026-07-27 | 1556 | 3.09 | **3.57** | 5.23 | 41.72 | 108.16 | 5.10 |
| 2026-07-28 | 1484 | 2.67 | **3.42** | 4.45 | 26.76 | 69.12 | 4.38 |
| 2026-07-29 | 1519 | 2.95 | **3.44** | 4.66 | 35.63 | 63.76 | 4.44 |
| 2026-07-30 | 1381 | 2.46 | **3.39** | 5.11 | 190.13 | 317.05 | 8.50 |

→ **p50 spans 3.39–3.57s** (range 0.18s) · **p90 spans 4.24–5.23s** (range 0.99s) · p99 spans 26.76–190.13s (range 163.37s).

**Read the ranges, not just the medians: the centre of this distribution is stable to well under a second day-to-day, while the p99 tail moves by 163s across the same five sessions with nothing changed.** For `ws_agg`/1Min that makes ~0.18s a **lower bound** on p50 noise and ~0.99s a lower bound on p90 noise — five sessions of one regime cannot establish an upper bound, so treat these as 'at least this much movement means nothing', not as significance thresholds. A p99 move is not interpretable from five sessions at all. **The p90 is the usable middle and the one to compare first.**

**`ws` / 10Sec — one row per RTH session**

| session (UTC date) | n | min | p50 | p90 | p99 | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-07-24 | 4640 | 3.11 | **3.24** | 3.76 | 21.64 | 79.63 | 3.83 |
| 2026-07-27 | 4610 | 3.12 | **3.27** | 4.19 | 41.38 | 86.90 | 4.53 |
| 2026-07-28 | 4411 | 3.10 | **3.22** | 3.81 | 15.09 | 66.29 | 3.69 |
| 2026-07-29 | 4515 | 3.09 | **3.22** | 3.61 | 20.16 | 63.93 | 3.73 |
| 2026-07-30 | 4042 | 1.69 | **3.23** | 4.09 | 116.37 | 240.29 | 6.09 |

→ **p50 spans 3.22–3.27s** (range 0.05s) · **p90 spans 3.61–4.19s** (range 0.58s) · p99 spans 15.09–116.37s (range 101.28s).

**Read the ranges, not just the medians: the centre of this distribution is stable to well under a second day-to-day, while the p99 tail moves by 101s across the same five sessions with nothing changed.** For `ws`/10Sec that makes ~0.05s a **lower bound** on p50 noise and ~0.58s a lower bound on p90 noise — five sessions of one regime cannot establish an upper bound, so treat these as 'at least this much movement means nothing', not as significance thresholds. A p99 move is not interpretable from five sessions at all. **The p90 is the usable middle and the one to compare first.**

### 12.5b `ws` sub-minute per symbol, RTH, `W1_settled_2d`

The direct-WS path, which #118 found IMMUNE to the boundary burst. It is the control: if a post-split run moves `ws_agg` and NOT this, the cause is boundary-chain saturation; if it moves BOTH, the cause is upstream of the worker (feed, gateway, network).

| TF | symbol | n | min | p50 | p90 | p99 | max | mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 10Sec | SPY | 4305 | 3.09 | **3.25** | **3.73** | 49.06 | 240.29 | 4.87 |
| 10Sec | TSLA | 4252 | 1.69 | **3.21** | **3.69** | 48.31 | 240.17 | 4.81 |
| 15Sec | TSLA | 2821 | 3.09 | **3.25** | **3.91** | 56.72 | 245.16 | 5.04 |
| 30Sec | TSLA | 1454 | 3.23 | **3.46** | **4.33** | 65.75 | 255.20 | 5.70 |

### 12.5c Reconciliation with the 2026-07-27 reference figure — **it does not fully reconcile, and that matters**

The step that commissioned this section cites the 07-27 scare as **p90 write-lag 6.9s → 13.9s**. Anyone comparing a post-split run against §12 will reach for those numbers, so the relationship has to be stated rather than assumed.

**Under this section's primary scoping (`ws_agg` 1Min, RTH, per session), 07-27 reads p90 = 5.23s — it is NOT elevated** (§12.5). So the 07-27 figures were not produced by the recipe §12.3/§12.4 use.

Widening the scope to **all engine-consumed WS sources, all TFs, full 24h** does land on the reference number:

| UTC date | n | p50 | p90 | p99 |
|---|---:|---:|---:|---:|
| 2026-07-22 | 18763 | 3.38 | **6.92** | 49.06 |
| 2026-07-23 | 20036 | 3.36 | **6.64** | 35.11 |
| 2026-07-24 | 19447 | 3.55 | **6.99** | 49.10 |
| 2026-07-27 | 18628 | 3.68 | **7.13** | 48.90 |
| 2026-07-28 | 18561 | 3.50 | **6.99** | 46.10 |
| 2026-07-29 | 19495 | 3.46 | **6.93** | 41.09 |
| 2026-07-30 | 17821 | 3.57 | **7.18** | 117.13 |
| 2026-07-31 | 16304 | 3.38 | **6.82** | 39.96 |

⚠️ The final row is the capture day and is **still accumulating** (post-close and overnight bars land after this capture), so its `n` is not comparable to the complete sessions above it. Every other table in §12 ends at or before 2026-07-31T00:00:00Z and is frozen.

**That pooled p90 sits at 6.64–7.18s on every one of these 8 sessions — the 6.9s half of the reference figure, essentially exactly.** That is strong circumstantial evidence the 07-27 recipe was this pooled/unscoped one.

**But it is circumstantial, and this artifact does not claim more.** The 07-27 recipe is not recorded anywhere this capture can read: the elevated 13.9s half is not reproducible here (07-27 pooled p90 reads 7.13s), which is consistent with the incident having been measured on an intraday slice rather than a whole session, but is equally consistent with a different metric altogether. **Two consequences, both binding:**

1. **Compare §12 against §12's own re-run, never against the 07-27 numbers.** Cross-recipe comparison is exactly the overclaiming §9 forbids.
2. The pooled table above is included **only** as a legacy bridge. It mixes producers with different latency profiles and TFs whose bar-close semantics differ, so a move in it is not attributable. **`ws_agg` 1Min RTH (§12.4) remains the ruler.**

### 12.6 Settle duration — `last_updated_at − written_at`, RTH, `W1_settled_2d`

How long a bar kept being rebroadcast/corrected after first arrival. Distinct from delivery lag and reported so a post-split change in *correction behaviour* is not mistaken for a change in *delivery*. Groups with n < 50 omitted.

| source | TF | n | rows rebroadcast | p50 | p90 | max |
|---|---|---:|---:|---:|---:|---:|
| `ws` | 10Sec | 8557 | 28 (0.3%) | 0.00 | 0.00 | 117.14 |
| `ws` | 15Sec | 2821 | 14 (0.5%) | 0.00 | 0.00 | 92.71 |
| `ws` | 30Sec | 1454 | 10 (0.7%) | 0.00 | 0.00 | 34.85 |
| `ws_agg` | 1Min | 2900 | 26 (0.9%) | 0.00 | 0.00 | 89.00 |
| `ws_agg` | 2Min | 702 | 699 (99.6%) | 2.50 | 4.72 | 80.21 |
| `ws_agg` | 3Min | 474 | 472 (99.6%) | 2.32 | 3.44 | 80.21 |
| `ws_agg` | 5Min | 290 | 290 (100.0%) | 2.37 | 4.41 | 80.08 |
| `ws_agg` | 10Min | 142 | 142 (100.0%) | 2.53 | 19.86 | 80.14 |
| `ws_agg` | 15Min | 50 | 50 (100.0%) | 3.32 | 28.27 | 71.51 |

### 12.7 The other two pairing windows (same metric, for one-to-one comparability)

§3 reports pairing over `W2_today_unsettled` and `W3_settled_5d` as well. The same lag metric over those exact windows, so every pairing row in this artifact has a latency row beside it. **`W2` is the capture day and is UNSETTLED for pairing — but lag is unaffected by settlement**, so W2's lag figures ARE directly comparable (a rare case where W2 is usable head-to-head).

**`W2_today_unsettled`** (2026-07-31T13:30:00Z → 2026-07-31T20:00:00Z) — RTH, `ws_agg`/1Min and `ws` sub-minute only

| source | TF | n | min | p50 | p90 | p99 | max | mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `ws` | 10Sec | 4584 | 3.10 | **3.23** | **3.69** | 8.48 | 88.34 | 3.60 |
| `ws` | 15Sec | 1487 | 3.09 | **3.25** | **4.13** | 13.18 | 64.19 | 3.71 |
| `ws` | 30Sec | 770 | 3.23 | **3.44** | **4.36** | 20.89 | 64.32 | 4.01 |
| `ws_agg` | 1Min | 1532 | 2.49 | **3.44** | **4.56** | 27.59 | 68.43 | 4.25 |

**`W3_settled_5d`** (2026-07-24T13:30:00Z → 2026-07-31T00:00:00Z) — RTH, `ws_agg`/1Min and `ws` sub-minute only

| source | TF | n | min | p50 | p90 | p99 | max | mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `ws` | 10Sec | 22218 | 1.69 | **3.23** | **3.86** | 34.70 | 240.29 | 4.34 |
| `ws` | 15Sec | 7314 | 3.09 | **3.27** | **3.92** | 36.55 | 245.16 | 4.45 |
| `ws` | 30Sec | 3730 | 3.23 | **3.50** | **4.32** | 41.69 | 255.20 | 4.94 |
| `ws_agg` | 1Min | 7492 | 2.46 | **3.47** | **4.55** | 46.72 | 317.05 | 5.32 |

### 12.8 `W1_settled_2d` WITHOUT RTH scoping (the literal pairing window)

§3–§6 score the whole `W1` span, extended hours included. This is the same literal span for lag, so nobody has to wonder whether the RTH scoping above changed the answer. Extended-hours tape is thin and `ws_agg` closes on the boundary regardless of prints, so these run higher — **use §12.3, not this, as the ruler**; this is here to keep the two halves honestly aligned.

| source | TF | n | min | p50 | p90 | p99 | max | mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `ws` | 10Sec | 14291 | 1.69 | **3.31** | **6.36** | 47.49 | 240.29 | 5.48 |
| `ws` | 15Sec | 5311 | 3.09 | **3.50** | **6.50** | 56.70 | 245.16 | 5.83 |
| `ws` | 30Sec | 2888 | 3.10 | **3.88** | **6.69** | 64.03 | 255.20 | 6.41 |
| `ws_agg` | 1Min | 5751 | 2.46 | **3.97** | **18.13** | 94.54 | 317.05 | 9.70 |

### 12.9 What the LATENCY baseline does NOT cover

Same rule as §9: a baseline that overclaims is worse than none. These are holes in §12 specifically, on top of every hole in §9.

1. **This is DB-write lag, not end-to-end alert latency.** The chain is `Polygon emits → worker receives → bar closes → live_bars write → shadow/monitor pipeline → alert insert`. `written_at` instruments ONE hop. #118 happened to be visible there because the write sat inside the saturated synchronous chain — but a regression that lands entirely AFTER the write (monitor pipeline, alert insert) is invisible to this section. `alerts.fired_at` vs bar close would cover that hop and is NOT captured here.
2. **No provider-side latency.** `bar_start` is Polygon's timestamp; the lag from real-world trade to Polygon emitting is upstream of everything we can see and is folded into these numbers as an unknown constant. If the gateway changes the *provider connection* (not just the routing), part of any move will be provider-side and this metric cannot separate it.
3. **Distribution shape only at 4 points.** p50/p90/p99/max. A bimodal distribution — say, most bars fast and a periodic slow cohort — can move materially with all four points roughly intact. If a post-split delta is ambiguous, pull the raw lag column for the window and compare full distributions rather than adding percentiles to this table after the fact.
4. **Absent bars are invisible.** A bar the worker never wrote contributes no row and therefore no lag. **Infinite latency reads as no data, not as a big number** — so §12 must always be read next to a coverage/row-count check (the `n` columns are there for exactly this; a post-split `n` that drops materially is a coverage regression masquerading as a clean latency table).
5. **`rest_insert` / `rest_correction` / `rest_backfill` rows are gap-fill, not delivery.** Their lag measures when a backfill ran, which is a scheduling fact, not a latency fact. They are reported in §12.3 for completeness and excluded from every verdict.
6. **Two RTH sessions in the primary window.** §12.5 widens to five for the spread, but that is still five days of one market regime. A quiet-tape week and a volatile week are not interchangeable here.
7. **No per-strategy attribution.** This is per (source, TF, symbol). It cannot say which strategy ate the lag, and it does not connect a lag figure to a specific missed/phantom edge in §4.
8. **No continuity with the 07-27 reference figure.** §12.5c shows the 07-27 numbers are not reproducible under this section's scoping and only half-reproducible under a pooled one. **§12 therefore establishes a NEW series starting today, not a continuation of an existing one** — it cannot be used to re-litigate 07-27, and pre-07-31 latency history remains un-baselined.
9. **The day-to-day spread in §12.5 is a LOWER bound on noise, not a significance threshold.** Five sessions of one market regime, all on the post-cull fleet. It can prove a small move is meaningless; it cannot prove a moderate move is meaningful. Closing this properly needs the capture run on more pre-split sessions — the same ask §9 item 9 makes of the pairing half, and the generator is cheap enough to run daily.

### 12.10 How to re-run §12 after the split

```
cd clients/KevBot_Toolkit/RoR_Trader/src
../.venv/bin/python _lag_baseline_264.py \
    --dev-sha $(git rev-parse --short HEAD) \
    --out ../docs/_active/_lag_section_<YYYY-MM-DD>.md
```

1. **Do NOT edit `WINDOWS`.** They are the same absolute windows as `_divergence_baseline_264.py`. Changing either copy desynchronises the two halves of this artifact.
2. **Run it against BOTH fleets** and diff them against each other as well as against this file. Symmetric delivery between fleets is the #164 gateway's load-bearing claim; a cross-fleet diff is the only thing that tests it.
3. **For the #163 pilot, expect a constant offset by construction** (~3.5s). Subtract the *designed* offset before reading a regression — and if the offset is not constant across symbols and TFs, that itself is the finding.
4. **Compare p90 before p50.** The 07-27 scare moved p90 6.9→13.9s. A p50 that holds while p90 doubles is the exact shape of the failure this section exists to catch.
5. **Check `n` first** (hole 4). A clean latency table over half the rows is worse than a dirty one over all of them.
6. Read §12.9 and §9 before declaring anything proved.

*§12 capture cost: 3.4s of DB time.*
