# Impl — Sub-Minute Canonical Bar + State Source (Phase 2b)

**Branch:** `feat/submin-canonical-source` (off dev bb36204) · **Flag:** `RORT_CANONICAL_SUBMIN_STATE` (default OFF, one kill-switch for all three legs) · **Plan:** `Plan_SubMinute_Canonical_Source.md` · **Session:** 2026-07-21

## 🚨 The plan's premise was WRONG — and the fix is bigger than scoped

The plan assumed "the backtest derives sub-minute secondaries from 1Sec Hi-Fi." It does not.
`services.prepare_data_with_indicators` builds EVERY non-injected secondary by
`resample_to_timeframe(primary_df, sec_tf)` — for a sub-minute gate on a coarser-or-equal
primary (339: 10s gate, 30Sec primary) that is an **UPSAMPLE**: pandas relabels each 30s bar
into its 10s bin → a pseudo-"10s" series at **30s cadence**. The backtest's 10s UT_BOT ribbon
is computed on ~2 bars/min of 30s data; live's ribbon on 6/min true 10s WS bars. The lanes
never computed on the same bar set — that (plus incremental-vs-replayed state, plus live's
raw-stream/premarket contamination) is the 339-class root. The 1m-gate class hit the same
wall and was fixed the same way (`RORT_ENFORCE_1MIN_GATE` native-secondary injection).

**Consequence: arming this flag CHANGES 339's backtest** (next recompute rebuilds the 10s
ribbon on true bars). That is the point — both lanes on ONE canonical bar — but it means
"ceiling matches settled backtest" is only measurable against a **flag-ON recompute**, and
the flag must arm on EVERY offline lane (api, batch-worker, data-worker, shadow-worker) at
the same time as the Worker, or lanes diverge from each other.

## What was built (all behind the one flag; OFF = byte-identical, tested)

**B — store layers** (`resampled_bar_store.py`): `SUBMIN_STORE_TFS = [10Sec, 30Sec]`
resampled from stored 1Sec (`base_layer_for_tf` seam), provenance span-hash over the 1Sec
members (same `source_1min_span_hash` column — name is historical), WINDOW-REPLACE per
UTC-day (10/30 divide the day; property holds). Targets = `RORT_SUBMIN_STORE_SYMBOLS`
(default TSLA) ∩ 1Sec-capture authority × {RTH, Extended Hours}. Maintenance rides the
existing settle-sweeper hook via `maintain_all_coarse` → `maintain_submin_symbol` (ONE
shared 1Sec read per symbol per cycle, whole-UTC-day rebuilds, breaker-guarded). Seed depth
`RORT_SUBMIN_SEED_DAYS=130` (covers 339's 108d visible window), CLI `--submin`, 7d chunks,
off-peak refusal unchanged. `is_store_tf` deliberately untouched — coarse consumers
(#1/#2/#3, `_load_warmup_df` serve) can never admit a sub-minute TF.

**C — offline lane** (`services._submin_secondary_canonical`, hooked before the legacy
resample): sub-minute secondary OHLCV = store whole-days + canonical-oracle 1Sec spans for
the final day / store gaps (identical construction — a gap changes cost, never bytes; deep
gaps log ERROR "seed the store"). If the whole canonical build fails: ERROR + legacy pseudo
fallback (keeps the strategy evaluable; that line firing = investigate). Prep cache key
carries the flag. Proven on 339's real config (3-day window): `UT_BOT_V4__10s` OFF≠ON on
1103/2340 primary bars, flips 542→989 (~3× bar density), **30m leg byte-identical OFF==ON**.

**A — live engine** (`ralph_engine`): sub-minute shadow closes (per-second path — they never
ride `_close_shadow_with_bar`) extracted to `SymbolHub._close_subminute_shadows` (testable
seam, byte-identical refactor). Flag ON: records rebuilt per close by full clean replay over
the builder's closed history, **session-filtered to the shadow's session** (sub-minute
builders eat the RAW stream incl. premarket; offline never sees those bars — the filter
removes that divergence class too), tail-bounded `RORT_SUBMIN_DERIVE_BARS=3000` (~1.3 RTH
days ≥ the measured ~1-day utv4 convergence bound). Fail-loud `[CANONICAL-SUBMIN-STATE]`
ERROR + HOLD previous records; correction-rebroadcasts skip the redundant legacy recompute.
**Bench (real UT_BOT_V4 shadow pipeline): 1200 bars ≈ 54ms · 2340 ≈ 105ms · 4700 ≈ 206ms**
per close (~130ms at the 3000 default) — budgeted for 10s cadence.

**Harness** (`replay_harness.py`): ported the secondary-TF-gap fixes from the mrs5a branch
(GEN packs, coarse-depth re-true, sub-minute re-true, fail-loud UNRES/LABEL-DRIFT) onto dev
+ added the flag-ON mirror at sub-minute secondary closes (same session-filter + tail +
hold semantics as live).

**Tests:** `test_canonical_submin_state.py` — 13 checks (OFF legacy, ON canonical, session
filter drops premarket, tail bound, loud hold on empty derive, correction skip, store flag
plumbing, synthetic-1Sec byte-identity 10Sec/30Sec × RTH/Ext, offline swap None-when-off).
All pass + the 6 M-RS5b fine-state tests unchanged. `test_current_dev_candle_parity_audit`
4 fails are PRE-EXISTING flag-dependent (pass with the armed prod flags set; identical on
the mrs5a checkout).

## Live topology notes (the label-drift lesson, pre-verified)
- Publish-filter collision: real 10Sec monitor (267) + 10s shadow on the same TSLA hub —
  safe: the own-records publish defers when a session-shadow owns the key
  (`ralph_engine` "session-shadow owns this key").
- Records emit in the canonical `10S-` namespace (bench confirmed
  `10S-UT_BOT_V4-BULL_TREND`) — requires `RORT_TF_LABEL_SEC_FIX=1` (armed in prod 07-21).
- Gate 4 (XTF_BLOCK_DIAG present-sets, multi-leg) still REQUIRES live observation after
  arming — 339's GEN-TIME_OF_DAY window is 10:00–11:30 ET.

## Gate status (2026-07-21 evening)
| Gate | Status |
|---|---|
| 1. parity suite 18/18 OFF and ON | ✅ **PASS both legs** (267: 301==301 trades, symdiff 0, each leg) |
| 2a. harness: 314/340/267 unchanged ON vs OFF | ✅ **PASS** (92.3/100/100 identical both legs; no UNRES/LABEL-DRIFT) |
| 2b. 339 → flag-ON backtest pairing | vs STORED lane stays 14.3% (expected: that lane is the pseudo ribbon); flag-ON pairing + post-recompute live-vs-settled = the real measure |
| 3. pass time / cost | bench 54–206ms/close; shadow-worker healthy (no PASS TIMEOUT, ~5.5min cadence, 4 slots, 339 not in set); seed clean (837,237 rows, 0 comparator diffs) |
| 4. live XTF_BLOCK_DIAG multi-leg | **caught a real topology gap on first look** (below); re-verify canonical publishes post-fix + present-sets in the 10:00–11:30 ET window 07-22 |

## ⚠️ Post-arm live findings (2026-07-21 evening) — the topology trap, TWICE
Armed 20:26Z; logs showed NO `[canonical-submin]` publishes. TWO live-only causes, found
and fixed the same evening (a single-monitor harness hub can never see either — the
close/publish topology only exists live):

**Finding #1 — wrong close site (`6ae7117`).** On the LIVE TSLA hub, 10Sec is ALSO a
PRIMARY TF (real monitors 267/338), and `on_second_bar`'s sub-minute secondary loop SKIPS
primary TFs — the (10,'RTH') shadow closes through `_close_shadow_with_bar` (primary
pipeline), whose canonical branch only covered ≥60s. FIX: derive factored into
`SymbolHub._canonical_submin_close`, owned by BOTH close sites.

**Finding #2 — the shadow never existed (`22c6b1f`).** Still zero publishes after #1; the
boot warmup list had NO tf=10s shadow. 267/338 are utv4-TRIGGERED, so interp-aware
suppression concluded "real monitor covers UT_BOT_V4" and never created it — **339's 10s
gate had been consuming 267's INCREMENTAL own-records all along** (own-records publish
only defers when a shadow owns the key). FIX: under the flag, `_real_monitor_covers`
returns False for sub-minute keys — the dedicated shadow is always created and owns the
key canonically. Flag OFF = legacy suppression, byte-identical.

**VERIFIED LIVE 21:12Z:** `Shadow warmup TSLA: tf=10s session=RTH bars=11700
initialized=True` → `shadow_close TSLA/10s records={'10S-UT_BOT_V4-BEAR_TREND'}
[canonical-submin n=3000 sess=RTH]` per close; 0 fail-loud errors. Gate-2b pairing:
flag-OFF pinned-window backtest == stored settled lane EXACTLY (14:11/14:18/14:30);
flag-ON backtest = {14:11:00, 14:19:30}; flag-ON decision-time replay entered 14:19:30
SECOND-EXACT (14:11 residual = bar-data tip class). Tests 23/23.

## Arming SOP (when gates 1–3 green)
1. Seed: `_resampled_store_maintain.py seed --submin --yes` (post-20:00Z; ~130d TSLA
   10Sec/30Sec ≈ 1.2M rows, chunked+breaker).
2. Merge PR → dev deploys Worker/api/batch-worker; data-worker picks up sweeper maintenance.
3. Flags: `RORT_CANONICAL_SUBMIN_STATE=1` on **Worker + api + batch-worker + data-worker**
   (data-worker also needs it for maintenance; it already has RORT_RESAMPLED_STORE_WRITE=1).
   Shadow-worker: `railway up` + fingerprint SOP only (NEVER var-set), same flag.
4. Recompute 339 (flag-ON lanes) → live-vs-settled + XTF_BLOCK_DIAG present-sets next RTH.
5. Kill switch: flip the one flag OFF everywhere → byte-identical legacy (store rows are
   inert data; no cleanup needed).
