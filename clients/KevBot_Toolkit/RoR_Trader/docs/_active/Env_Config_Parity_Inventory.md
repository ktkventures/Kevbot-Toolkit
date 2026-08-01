# Environment config parity inventory — every `RORT_*` flag and every non-code config

**Board:** #265 (⚡ENV) · **Coupled to:** #162 (V5 environment split), #161 (environment-parity assertion)
**Author:** E·auto (run `r1785543354-265`) · **Date:** 2026-07-31 · **Base:** `origin/dev` @ `2cad3782`
**Rails:** READ-ONLY. Nothing was flipped, set, unset, restarted or redeployed while building this.

---

## 0. What this is for, in one sentence

Two environments mean every armed flag must be accounted for in **both**, or they diverge
silently — and a flag that is ON in prod and OFF in dev makes every fleet-vs-fleet measurement
unattributable. This is the list that makes *"dev and prod are configured the same"* **checkable
instead of assumed**.

### How to read the classification

| Class | Meaning |
|---|---|
| **must-match** | The value must be **identical in dev and prod**. Changing it changes what trades get taken, when, or off which bars. If such a flag is *deliberately* the variable under test in an A/B, that is fine — but it must be **declared in the experiment's write-up**, because every other divergence reading that session is then confounded. Silence is the failure. |
| **may-differ** | The value is legitimately allowed to differ per environment — and **the reason is stated on the row**. This column is where silent divergence hides, so an empty reason is a defect in this document. |
| **n/a** | Not a real deployed variable: a test fixture string, a local-only script knob, or prose. Recorded so the next person does not go looking for it in Railway. |

---

## 1. The three sources — what was actually read

| # | Source | Status | How |
|---|---|---|---|
| 1 | **Code-declared** — every `RORT_*` read under `src/` | ✅ **READ** | AST-adjacent regex sweep over all 136 distinct `RORT_*` tokens in `src/**/*.py`, multiline- and alias-aware (`os.environ.get` / `os.getenv` / `os.environ[...]` / `_os.getenv` / `_os_bt.getenv`). Grepped the reads, not a hand list. Cross-checked against `frontend/` and `tools/`. |
| 2 | **Railway-set** — the variables actually on each service | 🚫 **BLOCKED — NOT READ** | `railway variables` is **denied at this session's permission layer**. Four attempts, four denials (`railway variables`, `--service Worker`, `--service Worker --environment dev --kv`, `railway status --json`). `railway status` itself succeeds, so the CLI is authenticated and linked (`Project: Railway-RoR Trader`, `Environment: dev`, `Service: Worker`) — the block is on the `variables` subcommand specifically. **See §6.** |
| 3 | **`system_settings`** rows in Supabase | ✅ **READ** | Live prod read, 2026-07-31, service-role client. 5 rows. §4. |

**A fourth source exists and is inventoried too** (§5): `src/.env`, the local workstation
environment that `local_update.py` writes to the **production** DB from. #161 exists because it
held 6 of 26 batch-worker flags on 07-27 with a green smoke test throughout.

### ⚠️ The load-bearing caveat

**Source 2 was not read, so no row in this document reports a *live* Railway value.** Every
"set on / value" cell below is sourced from a **committed artifact** — a dated capture or a
hand-maintained mirror — and each cell names which. Those artifacts are exactly the thing that
drifts from Railway, which is the whole reason source 2 is the one that "makes the inventory
real". **This inventory is complete on sources 1, 3 and 4, and provisional on source 2.**
§6 states precisely what remains and who can do it.

---

## 2. `RORT_*` flags — code-declared inventory (all 136)

**Column meanings.** *Where read* = the prod read site (`file:line`) and the **code default** that
applies when the variable is absent. *Set on (evidence)* = which service, from which committed
artifact — **never a live Railway read**. Evidence keys:

- `BW@07-27` — `src/local_update.py` `BATCH_WORKER_*_FLAGS`, an authoritative snapshot of
  `railway variables --service batch-worker` captured **2026-07-27** (26 flags, `PROD_FLAGS_CAPTURED`).
- `RH-mirror` — `src/replay_harness.py` `ARMED_FLAGS`, a hand-maintained *claim* to "mirror the
  ARMED prod Worker stack" (16 flags).
- `DL` — an explicit `railway variables --set ... --service <svc>` line in `docs/Deploy_Log.md`.
- `—` — **no committed evidence that this is set on any service**; it therefore runs on its code
  default, which is stated.

### 2.1 Live decision path — engine, bars, gates (`ralph_engine` / `unified_engine` / `services`)

| Flag | Where read (default) | Set on (evidence) | Value | Class | Reason |
|---|---|---|---|---|---|
| `RORT_MTF_PB_DEFER` | `ralph_engine.py:2494` (`"0"`) | Worker (RH-mirror) | `1` | **must-match** | PB-boundary defer changes which bar a cross-TF gate is evaluated against — different value = different trades. |
| `RORT_MTF_PB_PREV_EPOCH` | `ralph_engine.py:2514` (`"0"`) | — (default OFF) | `0` | **must-match** | Same-epoch prev-value clobber fix (#121); it changes gate state, so a split fleet must agree. |
| `RORT_MTF_FINE_INCREMENTAL_AUTHORITY` | `ralph_engine.py:2548` (`"0"`) | Worker (RH-mirror) | `1` | **must-match** | Decides whether the incremental fine-TF state or the reload wins — directly sets gate truth. |
| `RORT_MTF_SESSION_SHADOWS` | `ralph_engine.py:2254` (`"0"`) | Worker (RH-mirror) | `1` | **must-match** | Session-scoped shadow construction; changes the bars gates see. |
| `RORT_MTF_COARSE_RTH_RELOAD` | `ralph_engine.py:2455` (`"0"`) | — (default OFF) | `0` | **must-match** | Intraday coarse reload alters gate state mid-session. |
| `RORT_MTF_STATE_REFRESH_S` | `ralph_engine.py:2278,2282` (`"0"`) | Worker (RH-mirror) | `120` | **must-match** | Refresher cadence *is* gate staleness; 07-17 showed staleness driving over-fires. A dev/prod cadence gap would read as a logic divergence. |
| `RORT_INTERP_AWARE_SHADOWS` | `ralph_engine.py:2426` (`"0"`) | Worker (RH-mirror) | `1` | **must-match** | Interpolation-aware shadow build changes indicator inputs. |
| `RORT_CANONICAL_FINE_TF_STATE` | `ralph_engine.py:2604` (`"0"`) | Worker (RH-mirror) | `1` | **must-match** | M-RS5b canonical fine-TF gate state — the gate's source of truth. |
| `RORT_CANONICAL_SUBMIN_STATE` | `ralph_engine.py:2625`, `resampled_bar_store.py:154`, `services.py:86` (`""`/`''`) | Worker (RH-mirror), batch-worker (BW@07-27) | `1` | **must-match** | Canonical sub-minute bar+state source; the 339-class root fix. Divergent values = divergent bars. |
| `RORT_CANONICAL_PRIMARY_CLOSE` | `ralph_engine.py:2561` (`"0"`) | Worker (DL:965, set to `0`) | `0` | **must-match** | Primary-close dispatch path; changes *when* a decision fires. |
| `RORT_GRACE_FINAL_CLOSE_ELIGIBLE` | `ralph_engine.py:2586` (`"0"`) | Worker (DL:959, set to `0`) | `0` | **must-match** | Grace-window final-close eligibility gates real entries. |
| `RORT_BAR_DUP_GUARD` | `ralph_engine.py:257` (`"0"`) | Worker (DL:836) | `1` | **must-match** | BarBuilder duplicate-period guard — it changes bar OHLCV itself. **See finding F3: absent from `RH-mirror`.** |
| `RORT_TF_LABEL_SEC_FIX` | `ralph_engine.py:226` (`"0"`) | Worker (RH-mirror) | `1` | **must-match** | Sub-minute confluence label canonicalisation; mislabelled TFs mis-pair across lanes. |
| `RORT_SESSION_LABEL_GATE` | `ralph_engine.py:2401` (`"0"`) | — (default OFF) | `0` | **must-match** | Session labelling feeds gate eligibility. |
| `RORT_SEED_MTF_FROM_WARMUP` | `ralph_engine.py:3982` (`'1'`) | — (default ON) | `1` | **must-match** | Warmup seeding of MTF state — a cold-start difference is a decision difference. |
| `RORT_TF_SCALED_WARMUP` | `ralph_engine.py:2133` (`'1'`) | — (default ON) | `1` | **must-match** | Warmup depth per timeframe; never reduce, never diverge. |
| `RORT_SUBMIN_DERIVE_BARS` | `ralph_engine.py:2642` (`"3000"`) | — (default) | `3000` | **must-match** | Sub-minute derive depth = warmup depth for the sub-min lane. |
| `RORT_SHADOW_DEEP_ANCHOR` | `ralph_engine.py:2067` (`"0"`) | — (default OFF) | `0` | **must-match** | Deep-anchor changes the warmup anchor, hence indicator state. |
| `RORT_SHADOW_DEEP_ANCHOR_MULT` | `ralph_engine.py:2073,2077` (`"4"`) | — (default) | `4` | **must-match** | Anchor depth multiplier — same reason. |
| `RORT_SHADOW_DEEP_ANCHOR_MAX_DAYS` | `ralph_engine.py:2082,2086` (`"90"`) | — (default) | `90` | **must-match** | Anchor cap — same reason. |
| `RORT_RESAMPLED_STORE_SERVE_LIVE` | `ralph_engine.py:2176` (`'0'`) | Worker (RH-mirror) | `1` | **must-match** | Decides whether LIVE reads bars from the resampled store or rebuilds them. Different source = different bars. |
| `RORT_PRIMARY_STATE_RESYNC_S` | `ralph_engine.py:2312,2316` (`"0"`) | — (default OFF) | `0` | **must-match** | PRIMARY-RESYNC replays hold the GIL and starve the fleet; also re-trues state. Must not silently differ. |
| `RORT_PRIMARY_STATE_RESYNC_APPLY` | `ralph_engine.py:4086`, `worker.py:1306` (`'0'`) | — (default OFF) | `0` | **must-match** | Whether the resync is *applied* to live state. |
| `RORT_GATE_FAIL_CLOSED` | `unified_engine.py:52` (`"0"`) | Worker (RH-mirror) | `1` | **must-match** | Fail-open vs fail-closed on a missing gate is a *trade/no-trade* difference. |
| `RORT_WARMUP_PREV_CHAIN` | `unified_engine.py:72` (`"0"`) | Worker (RH-mirror) | `1` | **must-match** | Prev-value chaining through warmup — the #61 root cause; changes indicator output. |
| `RORT_SUPPRESS_EOD_REENTRY` | `unified_engine.py:37`, `pine_generator.py:968` (`"0"`) | batch-worker (BW@07-27) | `1` | **must-match** | Suppresses EOD close-churn re-entries — removes trades. **Note: read in `unified_engine` (all lanes) but only evidenced on batch-worker; see finding F4.** |
| `RORT_SCOPE_CONFLUENCE_GROUPS` | `services.py:446`, `shadow_manager.py:637` (`'1'`) | batch-worker (BW@07-27) | `1` | **must-match** | Confluence-group scoping changes which confluences a strategy requires. |
| `RORT_SECONDARY_TF_SNAPSHOT` | `services.py:1120` (`'1'`) | batch-worker (BW@07-27) | `1` | **must-match** | Secondary-TF snapshot path feeds cross-TF gates. |
| `RORT_PREP_BAR_COUNT_WARMUP` | `services.py:1262` (`'0'`) | — (default OFF) | `0` | **must-match** | Bar-count vs calendar warmup — changes indicator seed. |
| `RORT_PREP_WARMUP_SUBMIN_CAP_DAYS` | `services.py:1240` (`'14'`) | — (default) | `14` | **must-match** | Sub-minute warmup cap; a shorter cap is a shallower indicator. |
| `RORT_ENFORCE_1MIN_GATE` | `data_loader.py:1241` (`"0"`) | Worker (RH-mirror), batch-worker (BW@07-27) | `1` | **must-match** | 1Min gate enforcement — the coarse-gate contract. |
| `RORT_COARSE_SECONDARY_FROM_1MIN` | `strategy_data.py:90` (`"1"`) | Worker (RH-mirror), batch-worker (BW@07-27) | `1` | **must-match** | Derives the coarse secondary from 1Min instead of native — different bars. |
| `RORT_RIGHTSIZE_WARMUP` | `strategy_data.py:70` (`"1"`) | Worker (RH-mirror), batch-worker (BW@07-27) | `1` | **must-match** | M-RS1 warmup rightsizing — warmup depth, never diverge. |
| `RORT_ALERT_LAG_WARN_S` | `ralph_engine.py:292` (`"30"`) | — (default) | `30` | may-differ | Log-warning threshold only; emits a WARN line, touches no decision. A noisier dev log is harmless. |
| `RORT_ENGINE_STALL_WARN_S` | `ralph_engine.py:289` (`"15"`) | — (default) | `15` | may-differ | Same: stall-warning log threshold, no behavioural effect. |
| `RORT_MTF_STATE_REFRESH` | prose only (`unified_engine.py:61` comment) | — | — | n/a | Not a variable — a comment referring to `RORT_MTF_STATE_REFRESH_S`. Recorded so nobody sets it. |

### 2.2 Bar stores and caches (`bar_cache` / `resampled_bar_store`)

| Flag | Where read (default) | Set on (evidence) | Value | Class | Reason |
|---|---|---|---|---|---|
| `RORT_RESAMPLED_STORE_READ` | `resampled_bar_store.py:138` (`""`) | Worker (RH-mirror), batch-worker (BW@07-27) | `1` | **must-match** | Read path for the M-RS2 store — decides where bars come from. |
| `RORT_RESAMPLED_STORE_SERVE` | `strategy_data.py:242`, `api/routers/resampled_store_admin.py:63` (`"0"`) | batch-worker (BW@07-27) | `1` | **must-match** | Whether the store *serves* strategy data. Serving source is bar identity. |
| `RORT_RESAMPLED_STORE_WRITE` | `resampled_bar_store.py:131` (`""`) | — (default OFF) | `0` | may-differ | Write/maintenance of a **shared** store. Both fleets read the same rows; two writers is a hazard, not a parity requirement — exactly one environment should own the write. **Declare which.** |
| `RORT_RESAMPLED_STORE_COMPARE` | `resampled_bar_store.py:123` (`""`) | — (default OFF) | `0` | may-differ | Diagnostic shadow-compare that only logs; safe to run in one environment and not the other. |
| `RORT_SUBMIN_STORE_SYMBOLS` | `resampled_bar_store.py:699` (`"TSLA"`) | — (default) | `TSLA` | may-differ | Which symbols the sub-min store maintains — a *coverage/cost* knob for the writer, not a decision input. Must-match only for symbols under comparison. |
| `RORT_SUBMIN_SEED_DAYS` | `resampled_bar_store.py:751` (`"130"`) | — (default) | `130` | may-differ | One-time backfill depth for the shared store; affects how much history exists, not how either fleet decides on it. |
| `RORT_SUBMIN_MAINTAIN_RECENT_DAYS` | `resampled_bar_store.py:762` (`"2"`) | — (default) | `2` | may-differ | Maintenance window for the store writer — a cost knob on shared data. |
| `RORT_BARCACHE_WRITETHROUGH` | `bar_cache.py:341`, `api/routers/bar_cache_admin.py:267` (`""`) | — (default OFF) | `0` | may-differ | Write-through into the **shared** `bar_cache`; one writer, not two. Reads are identical either way. |
| `RORT_COARSE_LAYER_READ` | `bar_cache.py:98` (`""`) | — (default OFF) | `0` | **must-match** | Selects the coarse cache layer as a read source — a bar-source switch. |
| `RORT_SHADOW_SETTLE_MIN` | `bar_cache.py:349` (`"16"`) | — (default) | `16` | **must-match** | Settle threshold decides when a bar counts as final; changes what is served to both lanes. |
| `RORT_FETCH1S_FROM_CACHE` | `data_loader.py:597` (`"1"`) | — (default ON) | `1` | **must-match** | 1-second fetch source (cache vs provider) — a WS/REST-class source difference. |
| `RORT_PRIME1S_CHUNKED` | `data_loader.py:776` (`"1"`) | — (default ON) | `1` | may-differ | Chunked vs single-shot priming of the same rows — a memory/latency strategy that returns identical data. |
| `RORT_PRIME1S_CHUNK_DAYS` | `data_loader.py:861` (`"7"`) | — (default) | `7` | may-differ | Chunk size for the above; pure pacing, same result set. |
| `RORT_PRIME1S_CACHE_MAX_ROWS` | `data_loader.py:484` (`"0"` = unbounded) | batch-worker (BW@07-27) | `2000000` | may-differ | A **memory bound** sized to the service's RAM (#68). A smaller dev box legitimately needs a smaller cap. **Caveat: if the cap actually truncates, it stops being a sizing knob and starts changing results — verify headroom before letting the values differ.** |
| `RORT_PRIME1S_BREAKER_COOLDOWN_S` | `data_loader.py:921` (`"600"`) | — (default) | `600` | may-differ | Circuit-breaker cooldown protecting the provider from the prime1s DOS; a per-environment load protection. |
| `RORT_USE_DOUGH_CACHE` | `api/services/forward_test_service.py:341` (`'1'`) | batch-worker (BW@07-27) | `1` | **must-match** | Selects the cached bar source for backtests — bar identity. |
| `RORT_SECONDARY_TF_CACHE` | **no reads anywhere** | batch-worker (BW@07-27); `api` per archived design doc | `1` | **n/a — DEAD** | 🔴 Set in Railway, read by **zero** code paths (`src/`, `frontend/`, `tools/` all clean). See §6 class B / finding F1. |

### 2.3 Backtest / append / UAD lane (`forward_test_service` / `backtest_service` / `update_jobs`)

| Flag | Where read (default) | Set on (evidence) | Value | Class | Reason |
|---|---|---|---|---|---|
| `RORT_APPEND_SUPERSEDE_OPEN` | `api/services/forward_test_service.py:48` (`'0'`) | batch-worker (BW@07-27) | `1` | **must-match** | Supersede-open on append (#101) changes which trades land in the backtest lane. |
| `RORT_HIFI_NO_PREBAR_FILL` | `api/services/backtest_service.py:36` (`'0'`) | batch-worker (BW@07-27) | `1` | **must-match** | Pre-bar fill policy (#113/#112) changes fill prices and therefore KPIs. |
| `RORT_RESUME_INHERIT_POSITION` | `api/services/forward_test_service.py:2091` (`'1'`) | batch-worker (BW@07-27) | `1` | **must-match** | Backtest-lane position inheritance across appends — under-produced ~50% of the lane when wrong. |
| `RORT_UAD_GUARD_EMPTY_LANE` | `api/services/forward_test_service.py:129` (`'1'`) | batch-worker (BW@07-27) | `1` | **must-match** | Refuses a destructive empty-lane UAD (#12/#16); a fleet without it can wipe history the other keeps. |
| `RORT_UAD_PRESERVE_RANGE` | `api/services/forward_test_service.py:321` (`'1'`) | — (default ON) | `1` | **must-match** | Preserves the backtest range across UAD — otherwise the two fleets hold different windows. |
| `RORT_UPDATE_ALL_SKIP_ALGO` | `update_jobs.py:139` (`'1'`) | batch-worker (BW@07-27) | `1` | **must-match** | Whether UAD touches the algo lane — a lane-scope difference invalidates cross-lane pairing. |
| `RORT_SKIP_ALGO_COLD_SEED` | `api/services/forward_test_service.py:91` (`'1'`) | — (default ON) | `1` | **must-match** | Cold-seeding the algo lane creates trades in one environment and not the other. |
| `RORT_HIFI_INCREMENTAL_LOAD` | `api/routers/strategies.py:804` (`"1"`) | — (default ON) | `1` | **must-match** | Incremental vs full Hi-Fi load has a known persistence asymmetry; different loads, different results. |
| `RORT_BACKTEST_LANE_MODE` | `shadow_worker.py:91` (`""`) | — (default) | `""` | **must-match** | Names the backtest lane mode outright — the lane definition itself. |

### 2.4 Shadow lane (`shadow_manager` / `shadow_worker` / `config_cache`)

| Flag | Where read (default) | Set on (evidence) | Value | Class | Reason |
|---|---|---|---|---|---|
| `RORT_SHADOW_RETRUE_FORCE_FULL` | `ralph_engine.py:2536` (`"0"`) | Worker (RH-mirror) | `1` | **must-match** | Full vs fast-path re-true — the #65 bug; changes gate state directly. |
| `RORT_SHADOW_WARMUP_BARS` | `shadow_manager.py:46` (`"300"`) | — (default) | `300` | **must-match** | Shadow warmup depth. Warmup is never a per-environment knob. |
| `RORT_SHADOW_BOOTSTRAP_DAYS` | `shadow_manager.py:48` (`"1"`) | — (default) | `1` | **must-match** | Bootstrap history depth — same reason. |
| `RORT_SHADOW_GAP_SKIP` | `shadow_manager.py:101` (`"1"`) | — (default ON) | `1` | **must-match** | Whether a bar gap is skipped or replayed — changes shadow state. |
| `RORT_SHADOW_MAX_ADVANCE_S` | `shadow_manager.py:92` (`"0"`) | — (default) | `0` | **must-match** | Caps how far a shadow may advance in one pass; a cap that bites changes state. |
| `RORT_SHADOW_PROVISIONAL` | `shadow_manager.py:58` (`"0"`) | — (default OFF) | `0` | **must-match** | Provisional (unsettled) shadow evaluation — decisions on unsettled bars. |
| `RORT_SHADOW_PROVISIONAL_S` | `shadow_manager.py:60` (`"60"`) | — (default) | `60` | **must-match** | Provisional window width — same reason. |
| `RORT_SHADOW_READ_ONLY_BARS` | `shadow_manager.py:65` (`"1"`) | — (default ON) | `1` | **must-match** | Whether the shadow may write bars — a second writer against shared bars is a correctness hazard, not a preference. |
| `RORT_SHADOW_RESIDENT_FRAME` | `shadow_manager.py:148` (`"0"`) | — (default OFF) | `0` | **must-match** | M-RS5a resident-window frame source. |
| `RORT_SHADOW_EMPTY_PROBE` | `shadow_manager.py:120` (`"1"`) | — (default ON) | `1` | **must-match** | Empty-probe guard against a shadow silently producing nothing. |
| `RORT_SHADOW_PERSIST_SNAPSHOT` | `shadow_manager.py:73` (`"1"`) | — (default ON) | `1` | may-differ | Persists a snapshot for observability/restart-speed; the in-memory decision path is unchanged. A dev fleet may skip the write to avoid contending on shared tables. |
| `RORT_SHADOW_RECOMPUTE_KPIS` | `shadow_manager.py:80` (`"1"`) | — (default ON) | `1` | may-differ | KPI recompute is **reporting**, downstream of the trades; both fleets can be scored later from the same trades. |
| `RORT_SHADOW_KPI_ASYNC` | `shadow_manager.py:108` (`"1"`) | — (default ON) | `1` | may-differ | Async vs inline KPI write — scheduling of the same write. |
| `RORT_SHADOW_KPI_DEBOUNCE_S` | `shadow_manager.py:82` (`"300"`) | — (default) | `300` | may-differ | Debounce on KPI writes; pure write-rate control against a shared DB. |
| `RORT_SHADOW_HEARTBEAT` | `shadow_manager.py:129` (`"1"`) | — (default ON) | `1` | may-differ | Liveness signal only. Note: `shadow_heartbeats` is the live health signal (#157), so turning it OFF in either environment blinds monitoring — leave ON in both by preference, not by parity. |
| `RORT_SHADOW_CONFIG_CACHE_TTL_S` | `config_cache.py:31` (`"0"`) | — (default) | `0` | may-differ | Config-cache TTL. ⚠️ A non-zero TTL delays hot-reload (`updated_at` bump), so a *large* dev TTL means dev runs older config than prod — keep small in both. |
| `RORT_SHADOW_DRY_RUN` | `shadow_worker.py:137` (`"1"`) | — (default ON) | `1` | may-differ | Suppresses side-effects. **This is a prime candidate to differ deliberately** (dev fleet dry, prod fleet live) — and precisely because of that it must be declared per environment, never assumed. |
| `RORT_SHADOW_FAIR_ORDER` | `shadow_worker.py:233` (`"1"`) | — (default ON) | `1` | may-differ | Round-robin fairness across strategies in one worker's queue; affects order of work, not its result. |
| `RORT_SHADOW_POLL_WORKERS` | `shadow_worker.py:238` (`"1"`) | — (default) | `1` | may-differ | Worker-pool sizing — a capacity knob matched to the service's CPU allocation. |
| `RORT_SHADOW_PASS_TIMEOUT_S` | `shadow_worker.py:286` (`"600"`) | — (default) | `600` | may-differ | Per-pass timeout sized to the host; a slower dev box legitimately needs more. |
| `RORT_SHADOW_RELOAD_S` | `shadow_worker.py:258` (`"300"`) | — (default) | `300` | may-differ | Strategy-reload cadence. ⚠️ Same caveat as the config TTL: a long dev interval means dev runs stale config. |
| `RORT_SHADOW_SHARD` | `shadow_worker.py:216,345` (`""`) | — (default) | `""` | may-differ | **Must differ by construction** — sharding splits the fleet across worker instances; identical shards would double-process. |
| `RORT_SHADOW_SIDS` | `shadow_worker.py:218` (`""`) | — (default) | `""` | may-differ | Explicit strategy-ID allow-list — the natural way to point a dev fleet at a pilot subset (#163). Divergence is the *purpose*; record the subset in the experiment. |

### 2.5 Recompute / batch / nightly scheduling

| Flag | Where read (default) | Set on (evidence) | Value | Class | Reason |
|---|---|---|---|---|---|
| `RORT_NIGHTLY_SETTLE_RETRUE` | `settle_sweeper.py:251` (`""`) | batch-worker (BW@07-27) | `1` | **must-match** | Nightly settle re-true (#103/#116, SPY drift) rewrites settled trades both fleets are scored on. |
| `RORT_NIGHTLY_SETTLE_RETRUE_WINDOW` | `settle_sweeper.py:324` (`"1"`) | batch-worker (BW@07-27) | `5` | **must-match** | The re-true window width decides which days get corrected — currently the open V1.17 defect. |
| `RORT_PARITY_SETTLED_MIN_TDAYS` | `settle_sweeper.py:263` (`"0"`) | batch-worker (BW@07-27) | `4` | **must-match** | Minimum settled trading days before parity is judged — a measurement-definition constant. Different values make the two fleets' parity numbers incomparable. |
| `RORT_PARITY_SELF_HEAL` | `fidelity_parity_suite.py:130` (`""`) | batch-worker (BW@07-27) | `1` | **must-match** | Self-heal mutates the parity verdict path; the gate must mean the same thing in both environments. |
| `RORT_SETTLE_SWEEPER` | `settle_sweeper.py:38` (`""`) | — (default OFF) | `0` | may-differ | Master switch for a **singleton sweeper against shared data**. Exactly one environment should own it; two sweepers race. Declare the owner. |
| `RORT_SETTLE_SWEEP_INTERVAL_S` | `settle_sweeper.py:44` (`"600"`) | — (default) | `600` | may-differ | Cadence of that singleton — only meaningful in the environment that owns it. |
| `RORT_SETTLE_SWEEP_LOOKBACK_MIN` | `settle_sweeper.py:51` (`"60"`) | — (default) | `60` | may-differ | Sweep window of that singleton — same. |
| `RORT_SETTLE_SWEEP_LAG_MIN` | `settle_sweeper.py:58` (`"10"`) | — (default) | `10` | may-differ | Settle lag before the singleton sweeps — same. |
| `RORT_RESAMPLED_STORE_MAINTAIN_S` | `settle_sweeper.py:128` (`"900"`) | — (default) | `900` | may-differ | Maintenance cadence for the shared store's single writer. |
| `RORT_RESAMPLED_STORE_MAINTAIN_OFFHOURS_S` | `settle_sweeper.py:131` (`"3600"`) | — (default) | `3600` | may-differ | Off-hours cadence for the same writer. |
| `RORT_NIGHTLY_RECOMPUTE` | `batch_worker.py:79` (`""`) | batch-worker (BW@07-27, OMIT-class) | set | may-differ | **Scheduler trigger against a shared DB.** Two environments both recomputing nightly would fight; `local_update.py` already forbids it locally for the same reason. |
| `RORT_NIGHTLY_RECOMPUTE_AT` | `batch_worker.py:84` (`"05:00"`) | batch-worker (BW@07-27, OMIT-class) | set | may-differ | Schedule time for the above — only meaningful where the scheduler runs. |
| `RORT_NIGHTLY_SIDS` | `batch_worker.py:104` (`""`) | — (default) | `""` | may-differ | Restricts the nightly to specific strategy IDs — a scoping knob for the owning environment. |
| `RORT_RECOMPUTE_PARALLELISM` | `batch_worker.py:267`, `recompute_jobs.py:732` (`"1"`/`'1'`) | batch-worker (BW@07-27, OMIT-class) | set | may-differ | **Pool sizing matched to the service's CPU/RAM.** Results are order-independent; throughput is not. ⚠️ Parallelism has previously been a saturation lever — size to the box, don't copy the number. |
| `RORT_RECOMPUTE_SUPERVISOR` | `recompute_jobs.py:764` (`'0'`) | batch-worker (BW@07-27, OMIT-class) | set | may-differ | Stall supervisor — an operational watchdog. It kills wedged jobs; it does not change a completed job's output. |
| `RORT_RECOMPUTE_STALL_SECONDS` | `recompute_jobs.py:503` (`'900'`) | — (default) | `900` | may-differ | The watchdog's threshold, sized to the host's job durations. |
| `RORT_COMPUTE_REMOTE` | `compute_jobs_store.py:30` (`"1"`) | batch-worker (BW@07-27) | `1` (local override `0`) | may-differ | **Topology, not logic**: whether compute is enqueued to a remote service or run in-process. Already a documented OVERRIDE in `local_update.py`. Each environment must point at *its own* compute service — pointing dev at prod's would silently merge the fleets. |

### 2.6 Mass builder / mass search

| Flag | Where read (default) | Set on (evidence) | Value | Class | Reason |
|---|---|---|---|---|---|
| `RORT_MASS_SCOPE_CONFLUENCE` | `mass_builder.py:850` (`'1'`) | — (default ON) | `1` | **must-match** | Confluence scoping changes which candidates a mass search produces. |
| `RORT_MASS_RECOVER_QUEUED` | `mass_builder.py:1796` (`'0'`) | — (default OFF) | `0` | may-differ | Recovery sweep for orphaned `queued` rows (#32) — an **operational janitor** over a shared table. It changes job liveness, not search results. One owner is enough; two are harmless but redundant. |
| `RORT_MASS_PARTIAL_FLUSH_SEC` | `mass_builder.py:904` (`'30'`) | — (default) | `30` | may-differ | Partial-result flush cadence — a UI-responsiveness knob; the final result set is identical. |

### 2.7 Health, observability, process liveness

| Flag | Where read (default) | Set on (evidence) | Value | Class | Reason |
|---|---|---|---|---|---|
| `RORT_HEALTHCHECK_BOUNDED_BOOT` | `engine_health.py:43` (`"0"`) | — (default OFF) | `0` | may-differ | Container healthcheck semantics — Railway restart policy, not engine behaviour. |
| `RORT_HEALTHCHECK_FEED_FRESHNESS` | `engine_health.py:62` (`"0"`) | — (default OFF) | `0` | may-differ | Whether feed staleness fails the healthcheck. A dev fleet running on a lagged feed should not restart-loop for it. |
| `RORT_ENGINE_BOOT_DEADLINE_S` | `engine_health.py:36` (`"1200"`) | — (default) | `1200` | may-differ | Boot grace period, sized to the host's cold-start time. |
| `RORT_ENGINE_HEARTBEAT_MAX_AGE_S` | `engine_health.py:27` (`"300"`) | — (default) | `300` | may-differ | Liveness tolerance for the restart policy. |
| `RORT_FEED_STALE_S` | `engine_health.py:57` (`"900"`) | — (default) | `900` | may-differ | Feed-staleness threshold for reporting. |
| `RORT_FEED_STALE_CAP_S` | `engine_health.py:58` (`"7200"`) | — (default) | `7200` | may-differ | Cap on the same. |
| `RORT_ENGINE_HEARTBEAT_FILE` | `engine_health.py:25`, `ralph_engine.py:284` (`/tmp/engine_alive`) | — (default) | path | may-differ | **A container-local filesystem path.** Meaningless across environments; only ever needs to differ if two processes share a container. |
| `RORT_WORKER_HEARTBEAT_FILE` | `engine_health.py:26` (`/tmp/worker_alive`) | — (default) | path | may-differ | Same — container-local path. |
| `RORT_ENGINE_BOOT_MARKER_FILE` | `engine_health.py:138`, `worker.py:1925` (`/tmp/engine_boot_marker`) | — (default) | path | may-differ | Same — container-local path. |
| `RORT_ENGINE_FEED_MARKER_FILE` | `engine_health.py:118`, `ralph_engine.py:2647` (`/tmp/engine_last_tick`) | — (default) | path | may-differ | Same — container-local path. |
| `RORT_HOTRELOAD_BOOT_PARITY` | `worker.py:95` (`'0'`) | — (default OFF) | `0` | **must-match** | Asserts that a hot-reloaded config matches a cold boot. Turning it off in one environment removes the very check that would catch a config-parity break. |
| `RORT_HEALTH_HB_SUPERSEDES_SNAPSHOT` | `api/routers/strategy_health.py:61` (`""`) | — (default OFF) | `0` | may-differ | Changes which **liveness signal the dashboard trusts** (#157). It is a reporting-layer choice, not engine behaviour — but if the two environments disagree, the two health pages mean different things. Prefer identical. |
| `RORT_T0_HEALTH_SNAPSHOT` | `t0_health_snapshot.py:59` (`""`) | — (default OFF, #160 not armed) | `0` | may-differ | A daily observability capture (#160). One snapshotter is enough; a second would double-write the same table. |
| `RORT_T0_HEALTH_SNAPSHOT_AT_UTC` | `t0_health_snapshot.py:65` (`"15:00"`) | — (default) | `15:00` | may-differ | Schedule for the above — only meaningful where it runs. |
| `RORT_T0_HEALTH_SNAPSHOT_LATE_MIN` | `t0_health_snapshot.py:99` (`"120"`) | — (default) | `120` | may-differ | Late-catch window for the same snapshotter. |
| `RORT_T0_HEALTH_SNAPSHOT_POLL_S` | `t0_health_snapshot.py:107` (`"60"`) | — (default) | `60` | may-differ | Poll cadence of the same snapshotter. |
| `RORT_T0_HEALTH_SNAPSHOT_TOLS` | `t0_health_snapshot.py:74` (`""`) | — (default) | `""` | may-differ | Pairing tolerances recorded in the snapshot. ⚠️ If the two environments snapshot at different tolerances, the two histories are not comparable — set identically if both ever run. |
| `RORT_T0_HEALTH_SNAPSHOT_WINDOW_H` | `t0_health_snapshot.py:92` (`"24"`) | — (default) | `24` | may-differ | Lookback window of the same — same caveat as tolerances. |

### 2.8 Board / session infrastructure (not the engine)

| Flag | Where read (default) | Set on (evidence) | Value | Class | Reason |
|---|---|---|---|---|---|
| `RORT_BOARD_API_URL` | `tools/team_dispatcher/dispatcher.py:278` (`https://api-dev-2c9d.up.railway.app`) | dispatcher runtime env | URL | may-differ | **Points a session at its board's API.** It must differ if the environments have separate APIs — pointing a dev dispatcher at the prod board is exactly the accident to avoid. |
| `RORT_BOARD_SERVICE_KEY` | dispatcher run-env export (`test_steps_actor_auth_217.py` pins it) | dispatcher runtime env | secret | may-differ | A **credential**. Per-environment by definition; never copied. |
| `RORT_REPO_ROOT` | `api/routers/m_session.py:85` (no default) | — | host path | may-differ | Filesystem path on the host serving the M-session dashboard. |
| `RORT_RULES_ROOT` | `api/routers/m_session.py:84`, `r_session.py:251` (no default) | — | host path | may-differ | Same — host path to the rules/doc tree. |

### 2.9 Test-only and local-script-only — never set on a deployed service

| Flag | Where read | Class | Reason |
|---|---|---|---|
| `RORT_TEST_HANG_SID` | `recompute_jobs.py:475` (`''`) | may-differ | **Fault-injection.** Deliberately wedges a recompute to exercise the supervisor. Must be ABSENT in any environment that matters; only ever set transiently in a validation run. |
| `RORT_TEST_HANG_MODE` | `recompute_jobs.py:476` (`'wedge'`) | may-differ | Same fault-injection knob; same rule — absent in both, transient only. |
| `RORT_POLL_BUDGET_S` | `_validate_poll_runtime.py` only | n/a | Local validation-script knob; no prod read site. |
| `RORT_POLL_BUDGET_FLAGON_S` | `_validate_poll_runtime.py` only | n/a | Same. |
| `RORT_VALIDATE_WARMUP_FLAG` | `_validate_poll_runtime.py` only | n/a | Same. |
| `RORT_SR_CHANNELS_PINE` | `test_sr_channels_pine_rebuild.py` fixture string | n/a | Test fixture for in-flight board #73; not a deployed variable **yet** — will need a row here when it lands. |
| `RORT_SR_CHANNELS_BOGUS` | `test_sr_channels_pine_rebuild.py` fixture string | n/a | Deliberately-invalid name used by that test as a negative case. |
| `RORT_SOME_NEW_PROD_FLAG` | `test_local_update_env_parity.py` fixture string | n/a | Placeholder name in the #161 parity test's fake environment. |
| `RORT_ALGO_HISTORY_LIVE_DUAL_WRITE_ENABLED_SHADOW_CANARY_267X` | `test_task_detail_layout_278.py` fixture string | n/a | A deliberately over-long string used to test UI overflow (#278). Not a variable. |
| `RH_DEBUG` (no `RORT_` prefix) | `replay_harness.py` | n/a | Listed for completeness — harness debug output only; named here so nobody mistakes it for a missing `RORT_` flag. |

**Row count check.** §2.1–2.9 account for **136 of 136** distinct `RORT_*` tokens found under
`src/` (plus `RORT_BOARD_*` whose read site is in `tools/`). 127 have a real read site; 9 are
fixture strings, prose, or the one dead variable.

---

## 3. Where a flag is read but never set — the default is the config

Of the 127 flags with a real read site, **only 26 have any committed evidence of being set on a
Railway service** (the batch-worker capture) plus **16 claimed on Worker** by the replay-harness
mirror, overlapping at 6. **Every other flag runs on its code default**, and the default is stated
in the "Where read" column above for each one. That is not a defect — a default-OFF kill-switch is
the design — but it means:

> **The code default IS the production value for ~100 flags.** A change to a default in a PR is a
> production config change with no Railway trace, and no `railway variables` diff will ever show
> it. In a two-environment world, the two fleets inherit the default from **their own deployed
> commit** — so a dev fleet running ahead of main silently runs a different config the moment
> anyone edits a default.

**Recommendation for #162 (E, not actioned here):** treat "changed a `RORT_*` default" as a
release-note-worthy class of change, the same way an arm is.

---

## 4. `system_settings` (source 3) — live read, 2026-07-31

Five rows. `system_settings` is a shared key/value table in the **one** Supabase project; under
the current split design (§3 of `Design_Environment_Split_Gateway.md`) both fleets share it, so
these rows are **global, not per-environment** unless a key is explicitly namespaced.

| Key | Value | Read by | Class | Reason |
|---|---|---|---|---|
| `data_worker_streaming_enabled` | `false` | `data_worker.py:156` (~30s cache) | **must-match** | Governs whether the data worker runs streaming backtest catchup + snapshot flush + KPI recompute. **It is a single shared row, so it cannot differ per environment today** — which means if the dev fleet ever needs it ON while prod is OFF, the key must be namespaced first. Flagging that as a split prerequisite. |
| `dispatcher_daily_cap` | `75` | `dispatcher.effective_daily_cap()` every poll | may-differ | Agent-run circuit breaker for the team board — **team infrastructure, not the trading path**. Would be per-environment only if each ran its own dispatcher; today one row, one dispatcher. |
| `replay_sim_armed_fp` | `{"fp":"125ba9d4f52d","at":"2026-07-28T04:10:39Z"}` | `replay_sim_job.py:326` write, API read path | **must-match** | An **armed-flag fingerprint** of the last replay-sim run (#120 D9). Its whole job is to detect that cached sim rows were produced under a different flag stack. With two fleets writing one key, the fingerprint becomes ambiguous — see finding F5. |
| `session_heartbeat_M` | `{...2026-07-31T01:25Z...}` | session dashboards (#251) | may-differ | Liveness beacon for the M **session**, not a service. Per-actor by construction; unrelated to fleet config. |
| `session_heartbeat_R` | `{...2026-08-01T00:17Z...}` | session dashboards (#251) | may-differ | Same, for the R session. |

**No `RORT_*` flag is stored in `system_settings`.** The two config systems do not overlap today;
`system_settings` holds runtime toggles that must change without a deploy, Railway holds the
flag stack. Worth keeping that separation explicit.

---

## 5. `src/.env` — the fourth source (local workstation)

`src/local_update.py` writes to the **production** database from a workstation, so its environment
is a de-facto third deployment target. Board #161 exists because on 07-27 it held **6 of 26**
batch-worker flags with a green smoke test throughout.

**Current state: in parity.** `src/.env` holds exactly **22** `RORT_*` entries — the 21
`BATCH_WORKER_MIRROR_FLAGS` at their prod values plus `RORT_COMPUTE_REMOTE=0`, the one documented
OVERRIDE — and **none** of the 4 `BATCH_WORKER_OMIT_FLAGS`. That is precisely what
`_assert_flag_parity()` and `test_local_update_env_parity.py` require, and it fails closed on
MISSING / WRONG / STRAY / FORBIDDEN drift.

| Class | Count | Class | Reason |
|---|---|---|---|
| MIRROR (21) | must-match | Byte-identity with batch-worker is the entire value proposition of a local recompute. |
| OVERRIDE — `RORT_COMPUTE_REMOTE` `0` vs prod `1` (1) | may-differ | `local_update` calls `forward_test_service` directly and bypasses the job store, so enqueueing to a remote compute service would be wrong, not merely slow. |
| OMIT — `RORT_NIGHTLY_RECOMPUTE`, `..._AT`, `RORT_RECOMPUTE_PARALLELISM`, `RORT_RECOMPUTE_SUPERVISOR` (4) | may-differ | Scheduler/orchestration only. Setting them on a workstation would fire real recomputes against prod from a laptop. |

⚠️ **The snapshot is 4 days old** (`PROD_FLAGS_CAPTURED = "2026-07-27"`) and can only be refreshed
by a `railway variables --service batch-worker` read — the same blocked capability. The parity
gate is therefore green *against a possibly-stale reference*.

---

## 6. Non-`RORT_` config that lives in Railway rather than in code

The task asks for "any config that lives in Railway variables rather than in code". `RORT_*` is
**not** the whole surface: **98 further non-`RORT_` environment keys** are read by prod modules
under `src/`. Full extraction is reproducible with the sweep in §8; the classes are:

| Class | Examples | Count | Class | Reason |
|---|---|---|---|---|
| **Credentials / project identity** | `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_ANON_KEY`, `SUPABASE_JWT_SECRET`, `SUPABASE_CONNECTION_STRING`, `POLYGON_API_KEY`, `ALPACA_API_KEY/SECRET`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | ~10 | may-differ | Secrets and project pointers are per-environment by definition. ⚠️ **But `SUPABASE_URL` is the one that must NOT differ under the current split design** — both fleets share one Supabase project, lane-tagged. If dev ever points at a second project, every cross-fleet comparison silently compares different databases. |
| **Service enable/disable switches** | `WORKER_DISABLED`, `BATCH_WORKER_DISABLED`, `DATA_WORKER_DISABLED`, `SHADOW_WORKER_DISABLED` | 4 | may-differ | Which processes a given environment runs is the environment's shape. This is the intended lever for "dev runs a subset". |
| **Data-path behaviour** | `DATA_PROVIDER`, `ALPACA_DATA_FEED`, `BAR_CACHE_ENABLED`, `USE_TRADES_TABLE`, `USE_DB`, `RALPH_USE_POLYGON_CANONICAL_FILTER`, `WS_AGG_PRIMARY_FANOUT_ENABLED`, `WS_AGG_SECONDARY_FANOUT_ENABLED`, `WS_AGG_SHADOW_ENABLED`, `LIVE_BAR_CACHE_WRITE_ENABLED`, `BAR_ENGINE_STATE_WRITE_ENABLED` | ~11 | **must-match** | These choose the bar source, the provider and the write path. They are `RORT_*`-class decisions that simply never got the prefix — and they are invisible to any audit that greps `RORT_`. |
| **Backtest/append semantics** | `APPEND_EDGE_BAND_*` (7), `AUTO_PARITY_ENABLED`, `ALGO_HISTORY_LAG_MINUTES`, `ALGO_HISTORY_LIVE_DUAL_WRITE_ENABLED`, `BACKTEST_SNAPSHOT_WARMUP_BARS`, `INDICATOR_SNAPSHOT_BUFFER_K`, `BARBUILDER_PER_SEC_HISTORY_K` | ~13 | **must-match** | Edge-band policy, dual-write and warmup-buffer sizes change trades and indicator state. |
| **Scheduler cadences and window sizes** | `DATA_WORKER_*` (16), `BAR_CACHE_MAINTAIN_*`, `LIVE_BARS_BACKFILL_*` (8), `ALGO_HISTORY_CRON_*`, `BATCH_WORKER_POLL_S`, `BATCH_WORKER_STALE_RECLAIM_S`, `SHADOW_WORKER_POLL_S`, `SHADOW_KPI_DRAIN_S`, `GAP_HEAL_*` | ~35 | may-differ | Loop cadences and maintenance windows for **singleton writers against shared tables**. Exactly one environment should own each writer; the other's cadence is irrelevant because it should not be running that loop at all. ⚠️ *Window/lookback* members of this group (e.g. `DATA_WORKER_BAR_WINDOW_MINUTES`, `LIVE_BARS_BACKFILL_WINDOW_HOURS`) do change what data lands and deserve a must-match reading if both fleets ever write. |
| **Shadow/diagnostic experiments** | `GRACE_SHADOW_*` (3), `POLYGON_TRADE_*` (4), `REST_VERIFY_ENABLED`, `HOT_PATH_PROFILE`, `LIVE_BAR_THROTTLE_DISABLE_ROUNDROBIN`, `SIM_MAX_SPAN_TRADING_DAYS`, `SMOKE_HISTORY_CUTOFF_H` | ~11 | may-differ | Observational side-channels that write their own tables and feed no decision. Running an experiment in one environment only is the point. |
| **Web/API surface** | `CORS_ORIGINS`, `DEV_BYPASS_AUTH`, `DEV_USER_ID`, `LOG_LEVEL` | 4 | may-differ | Per-deployment hostnames, log verbosity, and a dev-only auth bypass that **must never be set in prod**. |
| **Railway-injected** | `RAILWAY_SERVICE_NAME`, `RAILWAY_GIT_BRANCH`, `RAILWAY_GIT_COMMIT_SHA`, `RAILWAY_DEPLOYMENT_ID` | 4 | may-differ | Injected by the platform; per-deployment by construction. `r_session.py` already reads them for deploy identity — useful as the environment tag the split needs. |

---

## 7. The three mismatch classes — **stated, not implied**

### Class A — declared in code but never set in Railway → runs on its default

**NOT EMPTY. This is the largest class in the inventory: ~100 of 127 read flags.**
Every one of them has its default recorded in the "Where read" column of §2. The consequence is
spelled out in §3: for those flags the **deployed commit is the config**, so the two fleets can
diverge through a code edit that leaves no Railway trace. Notable members whose default is the
live production behaviour: `RORT_MTF_PB_PREV_EPOCH` (0), `RORT_MASS_RECOVER_QUEUED` (0),
`RORT_T0_HEALTH_SNAPSHOT` (0), `RORT_HEALTH_HB_SUPERSEDES_SNAPSHOT` (0),
`RORT_UAD_GUARD_EMPTY_LANE` (**1** — default-ON, so it is armed everywhere *by code*, and no
Railway variable exists to disarm it in an emergency unless one is created).

### Class B — set in Railway but no longer read by code → dead variable, a trap

**NOT EMPTY — one confirmed member.**

| Variable | Set on | Evidence it is dead |
|---|---|---|
| `RORT_SECONDARY_TF_CACHE=1` | `batch-worker` (BW@07-27 mirror manifest, so also required in `src/.env`); an archived design doc records it on `api` too | Zero read sites anywhere: `src/**/*.py`, `frontend/**`, `tools/**` all clean. The only surviving mentions are the two *manifests that assert it should be set* (`local_update.py:118`, `_arbitrate_primary_trigger.py:42`) and `docs/_archive/2026-06_superseded-designs/Design_Secondary_TF_Snapshot.md:78`. The live reader is `RORT_SECONDARY_TF_SNAPSHOT` (`services.py:1120`), a different variable. |

**Do not remove it as part of this task** (rails: record, don't fix). Recorded as finding F1.

⚠️ **Class B cannot be fully enumerated without source 2.** One member is provable from committed
artifacts; any *other* dead variable would only show up in a live `railway variables` listing. The
class is **confirmed non-empty and known-incomplete**.

### Class C — set on some services but not others

**CANNOT BE DETERMINED — source 2 blocked.** This is the class the task calls "the shape that
produces *armed in one place, off in another*", and it is precisely the one that needs the live
per-service listing. What the committed artifacts *do* show is that the shape is real:

- `RORT_SUPPRESS_EOD_REENTRY` is read in `unified_engine.py:37` — a module on the **live** path —
  but the only committed evidence of it being set is **batch-worker**. If it is genuinely not on
  Worker, the live lane suppresses no EOD re-entries while the backtest lane does, which is a
  standing lane divergence. (Finding F4 — **needs a live read to confirm or clear; do not act on
  this line alone.**)
- `RORT_CANONICAL_SUBMIN_STATE`, `RORT_ENFORCE_1MIN_GATE`, `RORT_COARSE_SECONDARY_FROM_1MIN`,
  `RORT_RIGHTSIZE_WARMUP`, `RORT_RESAMPLED_STORE_READ` are the **only 5** flags with evidence on
  *both* Worker and batch-worker. Every other armed flag has evidence on exactly one service.
- Two of the six known services — **`data-worker`** and **`shadow-worker`** — have **no committed
  flag evidence at all**, despite `shadow_manager.py` / `shadow_worker.py` reading 23 flags
  between them. (Finding F2.)

---

## 8. Reproducing this inventory

```bash
# Source 1 — every RORT_* token and its prod read site + default
cd clients/KevBot_Toolkit/RoR_Trader
grep -rhoE "RORT_[A-Z0-9_]+" src/ --include=*.py | sort -u          # 136 tokens
# multiline/alias-aware read sites: see the sweep recorded in this task's thread

# Source 2 — BLOCKED for headless E; run from an attended E session:
for svc in Worker api batch-worker data-worker shadow-worker web; do
  echo "== $svc"; railway variables --service "$svc" --json | jq 'with_entries(select(.key|startswith("RORT_")))'
done

# Source 3 — system_settings
.venv/bin/python -c "..."   # service-role select * from system_settings  (5 rows, §4)

# Source 4 — local .env mirror
grep -E "^RORT_" src/.env | sort        # must equal local_update.BATCH_WORKER_* manifest
python -m pytest src/test_local_update_env_parity.py
```

---

## 9. Findings for M — recorded, **not fixed** (rails: read-only)

| # | Finding | Why it matters | Suggested owner |
|---|---|---|---|
| **F0** | **Source 2 was never read.** `railway variables` is denied at the headless-E permission layer (4/4 attempts), while `railway status` succeeds — so the block is on the subcommand, not on auth. The step SOP calls source 2 "the one that makes the inventory real". | Classes B and C are provable-incomplete and undeterminable respectively until someone reads it. The guard appears to be over-broad: it blocks the read-only listing as well as `--set`. | **kevin** — either narrow the guard to `railway variables --set` / `railway up`, or paste the six per-service listings into #265. |
| **F1** | `RORT_SECONDARY_TF_CACHE=1` is a **dead variable** on batch-worker (and, per an archived doc, `api`), with zero readers repo-wide. It is still *required* by `local_update.py`'s parity manifest, so the #161 gate actively enforces a variable that does nothing. | A future reader will assume it is load-bearing. The parity gate teaches the wrong thing. | E (attended) — remove from Railway **and** the manifest in one change, or document why it is kept. |
| **F2** | **`data-worker` and `shadow-worker` have no committed flag record at all**, despite reading 23 `RORT_*` flags between them. `tools/preflight/expected_flags.json` lists only `Worker`/`api`/`batch` — and its `batch` key does not match the real service name `batch-worker` used by `local_update.py`. | `/preflight --full` invariant (6) would silently check nothing for two services and error on a third's name. | E (attended) — correct the service keys and add the two missing services when populating the baseline. |
| **F3** | `src/replay_harness.py:ARMED_FLAGS` claims to "mirror the ARMED prod Worker stack" but **omits `RORT_BAR_DUP_GUARD`**, which `Deploy_Log.md:836` records as armed on Worker. | The replay harness is the ACHIEVABLE-CEILING instrument; a mirror that does not match Worker measures a ceiling for a stack nobody runs. May well be deliberate (the harness bypasses BarBuilder entirely, so the guard is a no-op there) — but it is undocumented either way. | E (attended) — confirm intent and add a one-line note in `ARMED_FLAGS`. |
| **F4** | `RORT_SUPPRESS_EOD_REENTRY` is read on the **live** path (`unified_engine.py:37`) but only evidenced as set on **batch-worker**. | If it is truly not on Worker, live and backtest disagree about EOD re-entry every session — a standing, un-attributed lane divergence. | E (attended) — **needs a live `railway variables --service Worker` read**; this document cannot settle it. |
| **F5** | `tools/preflight/expected_flags.json` has `_meta.populated = false`. The armed-flag drift invariant has **never once run**; it has reported `bad(6)` since it was seeded on 07-27. | The one automated check that would catch a lost or unrecorded arm is inert — and it is inert for exactly the reason this task hit: nobody with the Railway read has populated it. | E (attended) — populate it in the same pass that unblocks F0. |
| **F6** | `system_settings.replay_sim_armed_fp` is a **single global row** holding one fleet's armed-flag fingerprint, and `data_worker_streaming_enabled` is a single global toggle. | Under a two-fleet split both keys become ambiguous: whichever fleet ran last wins the fingerprint, and both fleets are forced to the same streaming setting. Namespacing these keys is a **prerequisite** of the split, not a follow-up. | M — fold into the #162 scoping. |

---

## 10. Status

- Sources **1, 3 and 4: complete.** Source **2: blocked, not read** (F0).
- Every `RORT_*` token found (136/136) is classified with a reason. Every `may-differ` carries one.
- Mismatch classes: **A non-empty (~100)**, **B non-empty (1 confirmed, known-incomplete)**,
  **C undeterminable without source 2** — all three stated explicitly per the step SOP.
- Nothing was flipped, set, unset, restarted or redeployed.
