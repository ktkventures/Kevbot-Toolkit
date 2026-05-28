# Divergence Investigation Log — 2026-05-28

> SOP: `docs/SOP_Divergence_Investigation.md`
> Tool: `src/_divergence_walkthrough.py`
> Window: last 1 hour (apples-to-apples ON, needs_investigation ONLY)

## Headline numbers (T=20:11Z, post-ws_rest_spliced rollout +1.5h)

| sid | sym | TF | paired | phantom | missed | **phantom%** | **missed%** |
|---|---|---|---|---|---|---|---|
| 150 | SPY | 10Sec | 17 | 23 | 1 | **56%** | 2% |
| 151 | SPY | 10Sec | 30 | 26 | 2 | **45%** | 3% |
| 153 | SPY | 10Sec | 30 | 25 | 2 | **44%** | 4% |
| 170 | SPY | 10Sec | 13 | 20 | 10 | **47%** | 23% |
| 171 | SPY | 10Sec | 16 | 16 | 11 | **37%** | 26% |
| 172 | SPY | 10Sec | 17 | 15 | 12 | **34%** | 27% |

Phantom rate of 34-56% is **much higher than acceptable**. Before today's M8 rollout most of these strategies were on `ws_agg_reconciled` — we don't have a clean before/after but Kevin's recall is that the backlog had a meaningful needs_investigation count yesterday already.

## Clusters investigated (10 of 43)

### Cluster 1 — phantom max_hold_bars at 20:11:10, sid 150 (alone)
- alert: exit `max_hold_bars` exec=C px=755.30 verified Δ=0.0
- preceded by: entry `utv4_bull_flip` at 20:09:30 verified Δ=0.0
- both alert bars verified clean
- backtest has no matching exit edge within ±60s
- **Hypothesis:** backtest's intra-bar stop fired earlier within the position, so backtest exited via stop_loss not max_hold. Live engine didn't fire that intra-bar stop. Same pattern as clusters 3-6.
- **Proposed classification:** `live_intrabar_stop_missed`

### Cluster 2 — phantom mlv2_cross_bear at 20:13:10, sids 170/171/172
- 3 strategies sharing trigger → 1 root cause
- exit `mlv2_cross_bear` verified Δ=0.0
- backtest has no matching exit at this time for any of the 3
- **Hypothesis:** all 3 strategies share the `mlv2_cross_bear` exit signal. Backtest may have already exited these positions earlier via intra-bar stops (same cascade as cluster 1). Live engine rode to mlv2_cross_bear.
- **Proposed classification:** `live_intrabar_stop_missed` + `cluster_duplicate` (3x dedup)

### Clusters 3-6 — missed entries + exits at 20:15:20-20:16:40, sids 170/171/172
- backtest produced trades 418230 (entry 20:15:20, exit 20:15:30 via stop_loss) and 418231 (entry 20:16:10, exit 20:16:40 via stop_loss)
- live fired ONE entry at 20:15:00 (paired) — backtest sees THREE entries in same window
- live has zero stop_loss alerts in this window
- **Hypothesis:** live engine isn't firing intra-bar stop_loss alerts for these strategies, OR backtest is allowing concurrent positions while live single-positions. Need to investigate.
- **Proposed classification:** `live_intrabar_stop_missed` (likely) — but cluster needs deeper look because there's also a re-entry question

### Cluster 8 — phantom exit at 20:18:30, sid 153 (post-drift_uncorrected cascade)
- alert: exit `sw123_bear_c2` exec=C px=755.27 verified Δ=0.0
- **Preceded by** a `drift_uncorrected` stop_loss alert at 20:16:41 with Δ=$0.05 (exec=L, intra-bar)
- The drift event poisoned indicator state — subsequent alerts fire on diverged state vs backtest
- Even though THIS alert's bar is verified Δ=0.0, the underlying trailing-stop/indicator state is offset because the prior bar's REST correction couldn't land
- **Proposed classification:** `drift_uncorrected_cascade` — **a real consequence** of sub-minute correction structural limit

### Cluster 9 — phantom exit at 20:18:40, sid 151 (same cascade pattern)
- alert: exit `max_hold_bars` verified Δ=0.0
- Preceded by drift_uncorrected stop_loss at 20:16:41 Δ=$0.05 (same shared event as cluster 8 — sid 151 and 153 both saw it)
- Then a rest_unavailable exit at 20:19:40 (Δ=None)
- **Proposed classification:** `drift_uncorrected_cascade`

### Cluster 10 — missed entry at 20:19:00, sids 170/171
- backtest entered via `mlv2_cross_bear` exit_reason trade
- live had no nearby alerts in the ±2 min window for sid 170 — but ±2 min window might have been too tight (the earlier alerts in the window had been displayed in prior clusters)
- needs re-check with wider context window
- **Proposed classification:** TBD pending re-investigation

## Patterns confirmed so far

1. **`live_intrabar_stop_missed`** — biggest category. Live engine doesn't appear to fire intra-bar stop_loss alerts on sids 170/171/172 (confirmed by missed entries that exit via stop_loss). On sids 151/153 it DOES fire (cluster 8/9 shows intra-bar stop_loss firing). Per-strategy difference — possibly stop_config difference. Need to compare configs.
2. **`drift_uncorrected_cascade`** — when sub-minute drift can't be corrected in time, indicator state diverges from backtest and the divergence persists across multiple subsequent trades until equilibrium re-establishes. This is the structural cost of bar-level (not per-second) REST splicing on 10s TF.
3. **`cluster_duplicate`** — strategies sharing a trigger get N copies of the same divergence. The "real" issue count is significantly lower than the raw event count.

## Recommended new buckets to add to the code

- `ws_drift_phantom` — phantom where the alert had `drift_uncorrected` (known cause, structural)
- `drift_uncorrected_cascade` — phantom/missed that's a downstream effect of a prior `drift_uncorrected` on the same strategy within last N bars
- `live_intrabar_stop_missed` — missed where backtest exits via intra-bar stop_loss but no corresponding live stop_loss alert was fired
- `cluster_duplicate` — same (timestamp, trigger) shared across N strategies → count as 1 with N-multiplier

## Open questions

1. **Why don't sids 170/171/172 fire intra-bar stop_loss alerts?** sid 151/153 do (we saw it in clusters 8/9). Check stop_config differences between these strategies.
2. **The 45-56% phantom rate** — is this normal background (i.e., this rate has been here all along) or did the M8 rollout introduce more? Without a clean baseline we can't tell, but if it's been "normal" then `ws_rest_spliced` isn't *worse* — it's just exposing what was always there.
3. **Should we accelerate the per-second splice idea?** Cluster 8/9 are the strongest argument: structural drift_uncorrected cascades are a real cost that bar-level correction can't fix on sub-minute TFs.
