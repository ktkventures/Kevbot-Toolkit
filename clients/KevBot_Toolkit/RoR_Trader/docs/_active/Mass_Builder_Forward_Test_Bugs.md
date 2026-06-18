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
**backtest + cache** rows after fwd_start. A *forward* count should be the
**cache/live lane only** (and should exclude backtest-seeded copies). This is
why 310 reads "38 forward trades, 0 alerts" — those are 19 backtest + 19
backtest-SEEDED cache rows, **not real live fills**.

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
