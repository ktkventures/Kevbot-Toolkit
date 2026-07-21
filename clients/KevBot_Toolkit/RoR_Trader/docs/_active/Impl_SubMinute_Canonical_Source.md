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

## Gate status
| Gate | Status |
|---|---|
| 1. parity suite 18/18 OFF and ON | pending (post-close — RTH edge drift causes false diffs) |
| 2. harness: 339 → flag-ON backtest; 314/340/267 unchanged | pending (post-close) |
| 3. pass time / cost | bench done (54–206ms); shadow-worker currently tracks ONLY sids 263/267/269/271 (339 not in set → zero added load until SIDS widens); grab `pass done in` lines post-close |
| 4. live XTF_BLOCK_DIAG multi-leg | after arming (next RTH window 10:00–11:30 ET) |

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
