# Live Bar Source Options — Reducing Phantom/Missed Drift on Sub-Minute TFs

**Status:** DECISION = B1 (Ralph polls Polygon REST directly) for MVP, 2026-05-27.
B2 and B3 retained as future options if MVP outcomes warrant them.

## The problem this addresses

Phase A survey (last hour, apples-to-apples filter ON) showed 48 real
divergence events across 7 SPY 10Sec strategies. Live engine fires
entry triggers (`utv4_bull_flip`, `eppv3_cross_short_up`) on bars
where backtest fires nothing — and vice-versa for ~8 missed edges.

Per-bar parity (Bars Comparison page):

| Source | TF | Close match | Volume match |
|---|---|---|---|
| Cache+backfill vs REST 1s aggs | SPY 1Min | 100% | 78.9% |
| Cache+backfill vs REST 1s aggs | TSLA 1Min | 100% | 43.6% |
| Cache+backfill vs REST 1s aggs | SPY 10Sec | 93% | 89.7% |
| ws_agg_reconciled grace 5s | SPY 10Sec | 90% | 87.2% |
| ws_agg_reconciled grace 7s | SPY 10Sec | 92.9% | 89.6% |

**Smoking gun:** SPY 10Sec close match drops to 90-93% (vs 100% on
1Min). The 7-10% of bars where close diverges is where knife-edge
triggers produce phantoms.

## Architecture as of 2026-05-27

- **`live_bars` table** is written by **Ralph (the live worker)** —
  each time `BarBuilder.finalize` runs, Ralph upserts the bar to
  Supabase via `live_bars_writer.py`. Source is WS-aggregated.
- **Data-worker** keeps its REST-derived bars **in process memory
  only** (Tier-1 1s store + Tier-2 1Min coarse store). It does NOT
  write to `live_bars` or any shared persisted table.
- **There is no shared REST-backed bar layer today.**

Empirical measurements (2026-05-27, 18:00-20:00 UTC):

- `live_bars` (Ralph's WS pipeline): **27% of SPY 10Sec bars MISSING**
  during low-liquidity extended hours. Plus a small number of bars
  with ~20-256× volume-undercount and 0.01-0.025 close offset.
- Bar-close → row-written latency on `live_bars`: **median 7-8 sec**
  for both 1Min and 10Sec. p90 ~45 sec for 1Min. Ralph's BarBuilder
  finalize doesn't fire until the next bar's first tick arrives, plus
  Supabase upsert latency.

These two findings together: Ralph's WS bar pipeline is **structurally
bad on sub-minute TFs**.

## Three options considered

### Option B1 — Ralph polls Polygon REST directly (CHOSEN)

New live model: `rest_polled`

How it works:
- Ralph subscribes to a set of symbols (same as today)
- Instead of opening a WS connection, Ralph polls Polygon REST
  per-second aggregates every ~1 sec for each symbol
- Aggregates REST 1s bars to the strategy's TF locally
- After `grace_seconds` past bar close, processes the bar and fires
  alerts if triggers hit

Pros:
- Simplest implementation (~1-2 sessions of work)
- Same bar source as backtest (REST 1s aggregates) — zero structural
  drift expected
- No new infrastructure
- Easy rollback: just change `live_model` config back

Cons:
- Duplicates Polygon API calls (data-worker + Ralph both polling). At
  ~6 symbols × 1 call/sec/symbol × 2 services = 12 calls/sec. The
  Polygon $199/mo plan has plenty of headroom but it's wasteful.
- Each new model on Ralph means more code in Ralph
- Doesn't position us for future "single source of truth" architecture

### Option B2 — Build new `rest_bars` table (future)

How it would work:
- Data-worker writes its REST-derived 1-sec aggregates to a new
  Supabase table (e.g., `rest_bars`) per second
- Ralph reads from this table instead of polling Polygon
- One service hits Polygon; the other reads the shared table

Pros:
- Single source of truth for bars
- Doesn't duplicate Polygon API calls
- Other services could read from the same table later (chart UI, etc.)

Cons:
- Genuinely new infrastructure (~2-3 days of work to do right)
- Ralph becomes dependent on data-worker uptime (need a fallback)
- More DB write pressure (6 symbols × 1 row/sec = 6 writes/sec from
  data-worker; modest but adds up)

Worth doing if B1 succeeds AND we either want to scale to more symbols
OR add more downstream bar consumers.

### Option B3 — Speed up Ralph's existing live_bars pipeline (future)

How it would work:
- Keep current WS-based architecture
- Fix the 27% bar-loss issue in Ralph's WS aggregator
- Reduce the 7-8 sec write latency
- Make ws_agg_reconciled actually work as advertised

Pros:
- No live-model migration needed
- Keeps WS speed advantage for tip-of-bar decisions

Cons:
- Doesn't fix the WS-vs-REST structural divergence (different sources)
- Repairing WS aggregator is its own deep investigation
- Even if perfect, knife-edge triggers will still flip on any 1-tick
  bar-OHLC difference between WS and REST

Worth doing IF we ever want sub-second decision latency in the future
AND B1's ~2-3 sec latency proves to be limiting.

## Why B1 was chosen

1. **Simplest path to validation.** Whether the WS-vs-REST hypothesis
   actually explains the phantom/missed drift can be confirmed in 1-2
   sessions of work with B1. B2 requires building infrastructure first.

2. **Risk-symmetric rollback.** If B1 doesn't materially drop phantom/
   missed counts, we revert by flipping `live_model` config back. If
   B2 doesn't, we've spent days on infrastructure for nothing.

3. **MVP fitness.** Kevin's target TFs are 1Min and above; sub-minute
   is exploratory. B1's ~2-3 sec latency on 1Min is 3-5% of bar — not
   user-perceivable for swing-style strategies.

## Non-goals

- **Not** modifying `ws_agg_locked` or `ws_agg_reconciled`. Both stay
  in the registry as legacy/current. New model added alongside.
- **Not** changing the backtest pipeline. Backtest stays REST-fed.
- **Not** changing the alert dispatcher, webhook layer, or position
  state tracking. Only the bar source changes.
- **Not** removing Ralph's WS path — strategies on existing live
  models continue using it.

## Grace value design

- **1Min+ strategies:** 2 sec default grace. Polygon per-sec
  aggregates settle ~2 sec; 1Min bars are stable by then. (Subject to
  empirical confirmation during RTH.)
- **30Sec strategies:** 3 sec default grace.
- **10Sec strategies:** 4 sec default grace.
- **5Sec or below:** 5 sec default grace.

Per-strategy override via `config.grace_seconds` (already in the
codebase from earlier work).

**Future enhancement (deferred):** two-pass reconciliation. First
pass at grace_seconds; second pass at ~15 sec to catch late
corrections. Not in MVP scope. Document this as a follow-up.

## Open empirical questions for during MVP rollout

1. **Polygon REST poll rate ceiling.** Confirm per-second-aggs
   endpoint can be hit 1×/sec for 6+ symbols without rate limits.
2. **Actual bar-close-to-Polygon-availability latency for 1Min bars.**
   If consistently under 1 sec, we can tighten the 1Min grace.
3. **Phantom/missed drop after rollout.** Compare 24h pre-rollout
   (current `ws_agg_reconciled` baseline) vs 24h post-rollout
   (`rest_polled` baseline) on the same SPY 10Sec strategies.

---

2026-05-27 — decision and architecture frozen for MVP. Implementation
plan in `docs/Plan_REST_Polled_Live_Model.md`.
