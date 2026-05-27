# Plan — `rest_polled` Live Model (B1)

**Companion:** `docs/Spec_Live_Bar_Source_Options.md` (decision doc)
**Status:** PLAN — written 2026-05-27, pending Kevin's go-ahead
**Target:** SPY 10Sec phantom/missed counts materially drop after
rollout; 1Min behavior unchanged or improved.

## Why this plan exists

Phase A surfaced that Ralph's WS bar pipeline is structurally bad on
sub-minute SPY (27% bar loss, 7-8 sec write latency). The decided fix
is a new `live_model = rest_polled` where Ralph polls Polygon REST
directly for bars. Same source as backtest → no structural drift.

## Architecture summary

```
                    Today (ws_agg_reconciled)
                    ─────────────────────────────────
  Polygon WS  ────▶  Ralph BarBuilder  ────▶  Ralph engine state
                    └─▶ live_bars table (Supabase)

                    NEW (rest_polled)
                    ─────────────────────────────────
  Polygon REST ───▶  Ralph REST poller ────▶  Ralph engine state
       (1s aggs)    └─▶ live_bars table (still written for charts)

                    Backtest (unchanged)
                    ─────────────────────────────────
  Polygon REST ───▶  Data-worker REST poller  ─▶  Data-worker engine state
       (1s aggs)    └─▶ in-memory bar stores
                    └─▶ trades table (backtest_rest_hifi)
```

Two services still poll Polygon REST (12 calls/sec total). Acceptable
on $199/mo plan. Same bar source means engines converge by construction.

## Milestones

### M1 — Register the new live model (~30 min)
- Add `'rest_polled'` to `LIVE_MODELS` registry with `available: True,
  default: False`
- Description copy + UI rendering picks it up automatically (existing
  pattern from `ws_agg_reconciled` and `ws_agg_locked`)
- **Verify:** New model shows up in the strategy detail page model
  picker but nothing's using it yet

### M2 — REST poller for Ralph (~3-4 hr)
- New module `src/ralph_rest_poller.py` (or extend existing
  `data_loader.py`)
- Per-symbol polling loop: every 1 sec, fetch the latest 1-sec
  aggregate from Polygon REST
- Maintain a per-symbol in-memory rolling buffer (last N seconds of
  bars)
- Per-strategy bar resampler: aggregate 1-sec bars into the strategy's
  TF
- When a bar's close time has passed by `grace_seconds`, treat it as
  finalized
- **Verify:** Stand up the poller against ONE symbol (e.g., SPY) and
  log bar arrivals. Confirm 1Min bar lands within 2-3 sec of close,
  10Sec within 4-5 sec.

### M3 — Wire the poller into the live engine (~2-3 hr)
- In `worker.py`, when a strategy's `live_model == 'rest_polled'`,
  route bar ingestion through the REST poller instead of the WS
  BarBuilder
- Reuse the existing incremental indicator engine + trigger
  evaluator + position state machine — no changes there
- Alerts get `live_model='rest_polled'` (existing column auto-fills
  from strategy config)
- **Verify:** Run on the dev branch locally with a known sid, watch
  alerts fire on REST bars, confirm timing matches expectation

### M4 — Cutover sequence (~30 min after M3 ships)

Phased rollout:
1. **Pick ONE canary** — sid 151 (highest phantom count last hour,
   simple utv4 + max_hold strategy). Direct DB update:
   `config.live_model = 'rest_polled'` for sid 151 only. Watch
   Railway logs for ~30 min.
2. **If canary green:** expand to all SPY 10Sec strategies in the
   "needs investigation" cluster: 151, 153, 171, 170, 172, 150.
3. **Tomorrow (RTH):** observe behavior under high-liquidity
   conditions. Should be even cleaner than extended hours.
4. **Once trusted:** flip to default for new strategies of that
   TF, after Kevin's sign-off.

### M5 — Validation (Thursday-Friday)

Success criteria:
- `needs_investigation` events drop materially on rolled-out
  strategies (target: from ~7 per strategy/hour → < 2 per
  strategy/hour during RTH)
- No alerts on `rest_polled` strategies have `live_model != 'rest_polled'`
- No new error patterns in Railway logs related to REST polling

If criteria not met:
- Roll back: set `live_model` back to `ws_agg_reconciled` on the
  affected strategies via DB update
- Investigate which assumption was wrong; update spec doc

## Code change summary

| File | Change | Estimated LOC |
|---|---|---|
| `strategy_models.py` (or registry source) | Add `rest_polled` entry | ~15 |
| `src/ralph_rest_poller.py` (NEW) | REST polling + bar resampler | ~200-300 |
| `src/worker.py` | Route strategies on new model through poller | ~50-80 |
| `src/ralph_engine.py` | Minimal: ensure engine accepts bars from new source | ~20 |
| `docs/Known_Bugs.md` | Add the WS 27%-bar-loss as documented + WONTFIX | ~20 |

Total: ~300-450 LOC. Mostly the new poller module.

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| Polygon REST rate-limits us | Start with 6 symbols, monitor; back off poll interval if hit |
| REST polling adds latency on tight TFs | Acceptable for MVP. 10Sec is the floor. Add B2 (rest_bars table) if we ever want sub-sec latency. |
| Ralph engine state diverges between WS and REST models on the same strategy during cutover | Cutover does a clean state reset per strategy on model change. Existing `_hot_reload_strategies` path handles this. |
| Engine snapshot pickle issue (user packs not picklable per existing memory) | Pre-existing — handled by warming up from market data when snapshot can't restore. Not made worse. |
| Polygon outage during RTH | Existing failure mode for backtest too. Both engines would degrade together. Alert on it. |

## Backup branches

- Before M1: `dev-backup-pre-rest-polled-model-2026-05-27`
- Before M3 (the risky wire-up): `dev-backup-pre-rest-polled-wireup-2026-05-27`

## What I need from Kevin

- **Go-ahead on this plan as written** (or edits to it)
- **Approval for the M1 commit when ready** (low risk, just registry)
- **Confirmation when ready for M4 cutover** — that's the moment that
  affects live trading

I won't ship M4 (cutover) without explicit "yes go" from you. M1-M3
land as commits on dev that Railway auto-deploys, but they don't
affect existing strategies because nothing's on the new model yet
until M4.

## Open punch list for follow-ups

These are NOT in this plan's scope but should not be forgotten:

1. **WS aggregator bug** — Ralph's WS pipeline drops 27% of bars on
   SPY 10Sec extended hours. Document in `Known_Bugs.md`. WONTFIX
   directly (we're migrating away) but worth knowing for ws_agg_locked
   strategies that remain.
2. **Two-pass grace reconciliation** — first pass at grace_seconds,
   second pass at ~15 sec. Logged in spec doc as future enhancement.
3. **B2 — `rest_bars` table** — single source of truth across services.
   Build only if MVP succeeds AND we want to scale to more symbols.
4. **L2 — Data-worker fires alerts directly** — Kevin's mental model
   of "as soon as backtest says trade, fire alert." Major refactor.
   Defer until L1 outcomes are clear.
5. **mlv2_cross_bear walker reclassification** — already in
   `Known_Bugs.md`. Independent of this plan.

---

2026-05-27 — ready to execute on your go-ahead.
