# Baseline — divergence / fidelity BEFORE the environment split

**Captured:** 2026-07-31T20:20:00Z → 2026-07-31T20:20:24Z  
**Board:** #264 (E) · pre-split ruler for V5 / #162 environment split  
**`dev` commit at capture:** `80ae7db0`  
**Instrument:** `api.routers.strategy_health.get_strategy_health` — the dashboard's own code path, called in-process. No new metric was invented.  
**Generator (re-run this):** `src/_divergence_baseline_264.py`

> Measurement SSOT: `docs/_active/Plan_Measurement_Trust.md`.
> This artifact is READ-ONLY output. Nothing was armed, recomputed or deployed to produce it.

## 0. The one number, and what it is not

**Fleet pooled paired% @10s over the settled primary window (`W1_settled_2d`) = 78.1%** (2815 paired / 276 phantom / 512 missed edges across 14 strategies carrying signal, out of 24 in the fleet).

That is the BEFORE. If the same invocation over the same window after the split returns materially different numbers **and the stored lane has not been re-trued in between**, the split changed something.

**It is a fidelity ruler, not a health certificate.** It measures ONE thing: do recorded alerts pair with stored backtest trades. §9 lists — at length and on purpose — what it does not measure. Read §9 before using this to clear anything. In particular it would **not** have caught the 2026-07-27 signal (an alert-latency dispersion change, p90 6.9s → 13.9s), which is close to the single most likely way an environment split goes wrong.

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

**1. Latency and its dispersion — the biggest hole.** This measures *whether* an alert paired, not *when* it arrived. The 07-27 divergence scare was a p90 write-lag change (6.9s → 13.9s) with pairing largely intact. The gateway split's own design doc names symmetric sub-second delivery as the load-bearing property. **This baseline does not measure delivery latency at all.** A separate `live_bars` `written_at`-lag distribution capture is required and is NOT in here (tripwire definition: board #118).

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

