# Mass-Builder Forward-Test display + live-fire bugs (309–314)

**Created 2026-06-18.** Root-cause audit of why new MB strategies show
inconsistent Alerts / Forward-trade counts / equity lines. Goal: fix these
**in the pipeline** so adding thousands of strategies "just works."

## The data stores (what each UI element reads)
For each strategy there are now (post Phase-40 trades-table migration):
- `trades` table, `data_source='backtest_rest_hifi'` — the **backtest** lane.
- `trades` table, `data_source='cache_None'` — the **forward/algo (cache)** lane.
  Seeded as a COPY of backtest (#26 cold-seed); diverges only once live runs.
- `alerts` table — **live signals** (the "Alerts (N)" pill).
- `stored_trades` JSONB on the strategy — **EMPTY (0)** for all. Legacy store;
  no longer the source of truth. (So MB `save` DID write to the trades table —
  the JSONB-vs-table fear is mostly resolved; the lanes are in the table.)

Card elements:
- **"Fwd 1d (N)"** = `_count_forward_trades` (strategies.py:354): counts ALL
  trades with `entry_fill_ts >= forward_test_start`, **computed live per page
  load**, **no data_source filter**.
- **Yellow "Forward" equity line** = the **cache lane** (`cache_%`) equity.
- **Blue "Backtest" line** = backtest lane.

## Confirmed audit (since forward_test_start = created_at ≈ Jun-17 05:3x)
| sid | gate | backtest | cache | alerts | Fwd-count (code, NOW) | UI showed |
|----|------|---------|-------|--------|----------------------|-----------|
|309|2m Bollinger|682|681 (diverged −1)|34|**37**|0 (stale)|
|310|5m RVOL|341|341 (identical copy)|0|**38**|38 ✅|
|311|1h VWAP+10mEMA|342|342|22|0|0 ✅|
|312|1d coarse|325|325|0|0|0 ✅ (healthy/quiet)|
|313|1d UT_BOT|455|455 (identical copy)|0|**112**|0 (stale)|
|314|trigger-only|262|262|0|0|0 ✅|

## The bugs, separated from the artifacts

### Artifact A — stale card counts (NOT a bug, just a refresh)
"Fwd" is computed live per load; 309/313 will show 37/112 on reload, not 0.
Resolves Kevin's "yellow line but Fwd=0" on 309 & 313.

### Bug 1 — "Forward trades" count is semantically wrong (double-counts)
`_count_forward_trades` has NO data_source filter, so it counts
**backtest + cache** rows after fwd_start. Per Kevin: forward-test should be the
**BACKTEST-MODEL lane only**, split at the divider (forward test = the same
backtest model, just the post-divider slice). So the count should filter
`data_source='backtest_%'` and `entry >= divider` (310→19, 313→56) — the cache
lane belongs to the live/divergence view, NOT the forward-test number. This is
why 310 reads "38 forward trades, 0 alerts": 19 backtest + 19 backtest-SEEDED
cache rows, none of them real live fills.

### Bug 2 — cache lane is a backtest seed-copy → yellow line lies
The `cache_None` lane starts identical to backtest (#26 cold-seed). So the
yellow "Forward" equity appears even when **zero** live trades occurred (310,
313). "Forward test" currently means "backtest projection until live diverges,"
not "live results." Decide: seed should be visually distinguished (e.g., dashed)
or the forward line should only plot post-first-live-fill rows.

### Bug 3 — `forward_test_start` dual-storage
Set on the **top-level column** (= created_at) but **None in `config`**. Code
reads the top-level value, but the split is a landmine (an earlier path already
wiped configs via the forward_test_start endpoint — see
`feedback_jsonb_partial_updates`). One source of truth.

### Bug 4 — cache lane model stamped `None` → `data_source='cache_None'`
Should be `cache_<algo_model>`. A None model id risks the snapshot-mismatch
re-catchup loop (`feedback_dataworker_model_id_default`). Verify the algo/live
model is stamped when the cache lane is written.

### Bug 5 (THE REAL TRADING BUG) — gated strategies don't fire LIVE
310 (33 BT) & 313 (103 BT) since Jun-16 → **0 live alerts**; ungated 308 fires
fine (482). PB vs CB gate ribbons identical → NOT timing. The live engine is
almost certainly **not producing the higher-TF gate state** (cross-TF secondary
feed), same class as the Phase-31 regression (ca57cac) + user-pack secondary-TF
FLIP gap. THIS is what blocks "execute these." Needs market-hours capture of the
live gate state for 313 (daily) to confirm, then fix the secondary-TF feed in
ralph_engine. (309 fired 34 alerts → its 2m gate partially works; 5m/1d worse.)

## Stage 1 (get 309–314 executable) vs Stage 2 (pipeline, scales to thousands)
- **Stage 1:** confirm + fix Bug 5 (live gate feed) so gated strategies actually
  enter live. Everything else is cosmetic to "can it trade."
- **Stage 2 (systematic):** Bug 1 (forward count = cache-lane only, exclude
  seed), Bug 2 (seed visualization), Bug 3 (one fwd_start source), Bug 4 (stamp
  cache model). Bake into MB save + the cold-seed path so new strategies are
  clean by construction.

## UPDATE 2026-06-18b — forward_test_start anchor + 311 (Kevin's OOS question)

**Confirmed the anchor bug.** MB config carries `in_sample_end: 2026-05-15` +
`backtest_start_date: 2026-03-19` (in-sample Mar19→May15, out-of-sample
May15→now). But `forward_test_start` = `created_at` (**Jun-17**), NOT the OOS
start (**May-15**). So the "forward test" divider = "when live monitoring began,"
not "when out-of-sample began." Traders expect forward-test = the OOS held-out
period (May15→now, backtest-model, NO alerts expected since not live-monitored).
DESIGN FIX (Stage 2): forward-test divider should be `in_sample_end`, and
"live-monitoring start" (alerts overlay) is a SEPARATE concept = created_at.

**311 is the PHANTOM direction (opposite of 310/313).** 0 backtest trades after
Jun-17 but **22 live alerts** Jun-18 (11 sw123_bull_c2 entries + 7 stop_loss + 4
utv4_bear_flip). Live fired where backtest (recomputed) shows nothing → phantom.
Either real live-only fires OR the creation-anchored window hides OOS backtest
trades that exist May15–Jun17. Re-check 311 with the OOS anchor before judging.

Net concept: there are THREE timelines, currently collapsed into one divider:
  in-sample backtest | out-of-sample (forward) backtest | live monitoring (alerts)
       Mar19–May15   |          May15–now (no alerts)    |   Jun17–now (alerts)

## UPDATE 2026-06-18c — Bug 5 ROOT CAUSE CONFIRMED (live gate warmup = flat days=7)

`ralph_engine.py:4689` warms the secondary-TF shadow engine (and primary monitor)
with `load_market_data(sym, days=7, timeframe=tf_str)` — a **flat 7 days
regardless of the gate's timeframe**. Bars produced:
- 2m gate (309): ~2,700 bars → warms → fires live (34 alerts) ✅
- 1h gate (311): ~49 bars → marginal → fires (22) ✅ (its 0-BT-post-Jun17 is the
  separate phantom/anchor question)
- 5m gate (310): ~546 bars (enough), BUT the documented **RVOL_V2 live-gate
  volume caveat** (line 691, "live RVOL read EXTREME near-constantly") → 0 live
- **4h/1d gate (312, 313): ~5–12 bars** → UT_BOT/coarse indicators NEVER warm →
  gate state invalid → **0 live entries**. ← systematic; hits every coarse-gated
  strategy.

**Fix direction (Stage 1, fidelity-IMPROVING — makes live match backtest):**
scale warmup `days` by the secondary TF so the gate indicator gets ~250+ bars
(daily → ~365d, 4h → ~60d, etc.), NOT a flat 7. Parity nuance: backtest builds
its daily/4h gate by **resampling from 1-min** (CLAUDE.md), while this warmup
loads NATIVE `tf_str` bars (split-adjustment risk). To truly match backtest, warm
the shadow from resampled-1min history, not native coarse bars. This REDUCES
divergence (backtest already uses full-history gate; live should too), so it's a
parity fix, not a risk — but it IS a live-engine change → greenlight before edit.

Separately: 310's RVOL live-gate reliability is its own sub-bug (volume feed),
partially addressed 2026-06-12; verify independently.

## UPDATE 2026-06-18d — Bug 5 FIX SHIPPED + validated

`ralph_engine.py`: new `_load_warmup_df()` (single source of truth for all 3
warmup paths — startup `_warmup_all` + both hot-reload seeds). Sub-minute TFs
load native (days=7); **≥60s TFs warm from resampled-1min over a TF-scaled
window** (`_secondary_warmup_days`: 1Day=355d, 4h=180d, 1h=55d, 5m=10d) —
matching backtest's resample-from-1min construction. Backup branch:
`dev-backup-2026-06-18-pre-shadow-warmup`.

Validation (parity-#1):
- 313 daily UT_BOT_V4: fix-window(355d/245 bars) vs backtest-window(135d/95) →
  **0/12 recent-date mismatches**; Jun 16/17/18 = BULL_TREND (the gate IS bullish
  on the days backtest fired 103× → live will now pass it).
- Regression on currently-firing strategies: 309 (2m Bollinger) old-native-7d vs
  new-resampled → **0/12**; 311 (1h VWAP) 36→273 bars → **0/12**. No regression.

Post-deploy live measure: 313/312 (coarse gates) should begin producing live
alerts once a primary entry fires while the gate is satisfied. 310's RVOL gate is
the SEPARATE volume-double-count sub-bug (still open).

## UPDATE 2026-06-18e — Tier 1 shipped (MB save auto-populates lane) + Tier 2 plan

**Confirmed MB pathway == UAD:** MB runs `prepare_data_with_indicators` +
`run_unified_backtest` (mass_builder.py:535/1195) — same engine as UAD's
`recompute_and_persist_stored_trades`. MB even pre-bakes a `precompute_bar_cache`
(the "dough": bars+indicators+superset triggers) and re-applies each candidate's
confluence via `run_trades_from_cache` (10-50× replay) — exactly Kevin's
pizza-dough model.

**Root cause of the save→UAD→UND→UND dance:** MB strips `stored_trades` from
search results (too large for JSONB, mass_builder.py:1914). So a saved MB
candidate lands with an EMPTY trades-table lane (`save_strategy_db` CAN persist
trades — `replace_trades_for_strategy` — but is handed none). UND then has
nothing to append from → manual UAD required. Candidate trades pop ~60s after a
search (`_active_searches` cleanup, 1967); API is `--workers 1` so the per-
(symbol,tf) dough cache (LRU 20) IS reachable from a later save but is lost on
deploy/eviction.

**Tier 1 SHIPPED (`8e270c7`, backup `dev-backup-2026-06-18-pre-tier1-autopopulate`):**
`create_strategy` auto-enqueues a single-strategy `mode='all'` recompute (async
update_jobs) when the save carries no trades → lane populates → UND works, no
manual UAD. Gated + try/except (a save never fails). Verify with a real MB save
(an "Auto-populate backtest lane" job should appear).

**Tier 2 (planned, scale + durability):** persist the dough (`CachedBarState`)
as a **Parquet blob in object storage** (Supabase Storage), indexed by a small
table (symbol, tf, window → blob_url) — NOT row-per-bar Postgres (1.4M
bars/15Sec-90d is too many rows; Parquet = a few MB, ~1s load). Then the save's
cold path becomes "load dough + `run_trades_from_cache` re-bake" (seconds,
deploy-durable) instead of a full recompute. Store one dough per symbol/TF, not
trade-detail per candidate (Kevin's storage point).

**Skip:** pre-baking stop *prices* — the cache already stores stop *inputs*
(ATR/swing levels in `current_values`); the final price must vary per candidate
(it's an optimization variable), so there's no win. Hi-Fi analog (cache the
1-second data) is a later phase.

## UPDATE 2026-06-18f — Tier 2 implemented END-TO-END (gated off) + validated

Behind kill-switch `RORT_USE_DOUGH_CACHE` (default **OFF** → production unchanged):
- **Storage** (`bar_cache_store`, `bddbf7c`): `dough-cache` Supabase Storage bucket
  (existing service-role creds — no new access). gzip(pickle) envelope w/
  schema_version + freshness (`_built_at`, 36h default). Window-only key
  (symbol/tf/session/data_days/backtest_start_date — NO user_id; doughs are
  market-data+superset, shareable, trigger-not-in-superset falls back).
- **Producer** (`mass_builder.run_mass_search`, `074518a`): persists each
  group's freshly-built dough (best-effort try/except, trading_days stamped).
- **Consumer** (`forward_test_service._do_recompute`, `2b4aeb5`): when flag on +
  a fresh dough exists, re-bakes the lane via `run_trades_from_cache` instead of
  a full recompute; carries trading_days onto `.attrs` (else KPI daily_r inflates
  ~180×); any miss/stale/error → full recompute.

Validated (all gated tests, throwaways cleaned): serialization round-trip
byte-identical (503==503); storage put→get→re-bake identical + freshness guard;
producer-key == consumer-key (MATCH); full flow engine==dough-rebake (503==503)
with trading_days propagated (4→4).

**Flip gate:** turn `RORT_USE_DOUGH_CACHE=1` ONLY after a dry-run confirms a
dough re-bake == a full UAD on the REAL window (the dough is built over the
search window; recompute uses resolve_visible_window+warmup — same resolver, so
expected identical, but confirm empirically). Until then: Tier 1's full recompute
populates saved-MB lanes (slow but correct); Tier 2 makes it seconds once flipped.
