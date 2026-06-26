# The Two Bar Caches — Definitions & Red Lines

**Status:** canonical reference. Written 2026-06-25 after the 2nd–3rd time context
loss put the two-cache distinction at risk. If you are an AI assistant resuming
work and touch bar storage, **read this first.**

We keep **two** separate bar caches. They are *related* (both store OHLCV bars
for a symbol/timeframe) but they are **not the same thing** and must never be
conflated, merged, or have one's update logic point at the other.

---

## 1. Live Bars — `live_bars` table

- **What it is:** the bars our system **saw live**, at decision time, primarily
  from the **WebSocket** (`source='ws_agg'`).
- **Source of truth for:** "what did the engine actually see in the moment?"
- **Used for:** replays, and **phantom/mistrade forensics** — i.e. diagnosing
  divergences caused by **WebSocket visibility** (a bar the WS missed or saw
  late). This forensic value depends on the data being *what we saw*, warts and
  all.
- **Mutability — THE RED LINE:** a bar that the live engine saw
  (`source='ws_agg'`) is **immutable**. It is **never** post-corrected with
  later/revised REST data. Overwriting a seen bar with "what Polygon says now"
  would destroy the exact thing this cache exists to preserve.
- **Allowed exception — gap-heal only:** `live_bars_rest_backfill.py` (env-gated
  `LIVE_BARS_BACKFILL_ENABLED`) fills **genuine gaps** — minutes the WS *missed
  entirely* — from REST, written `source='rest_backfill'`, **INSERT-only
  (`ON CONFLICT DO NOTHING`)** so it can **never overwrite** a `ws_agg` row. The
  live engine treats `rest_backfill` rows as consume-only (never fires alerts on
  them). Filling a *missing* bar is allowed; *overwriting a seen* bar is not.
- **Written by:** `live_bars_writer.py` (WS), `live_bars_rest_backfill.py` /
  `live_bars_rest_backfill_subminute.py` (gap-heal). Related: `live_bars_trades`.

## 2. REST Bars — `bar_cache` table

- **What it is:** Polygon **REST** canonical bars, cached locally for speed.
- **Source of truth for:** **backtests** (our backtests are REST-based) and any
  recompute/warmup load path.
- **Used for:** making backtest/recompute **fast** (serve from Postgres instead
  of re-pulling Polygon) while staying **byte-identical to a fresh Polygon REST
  pull**.
- **Mutability:** **revisable on purpose.** Polygon revises its own historical
  1-second/1-minute bars for *days* after the fact (late prints). Keeping
  `bar_cache` matching current Polygon REST is the *goal* — `maintain_symbol` +
  the revision-horizon refresh exist precisely to re-pull and overwrite so the
  store tracks canonical REST. (`BAR_CACHE_REVISION_HORIZON_MIN`, default 45,
  governs the live-tip refresh window; see open issue below re: the 1-second
  multi-day revision window.)
- **Written by:** `bar_cache.py` (`backfill_symbol`, `maintain_symbol`,
  `maintain_all_enabled`). Read by `bar_cache.read_bars()` (direct Postgres).
- **Feature flags:** `BAR_CACHE_ENABLED` (read path), `BAR_CACHE_MAINTAIN_ENABLED`
  (worker maintain cron).

---

## The one rule that ties it together

> **`bar_cache` (REST Bars) is kept matching current Polygon REST.
> `live_bars` (Live Bars) is kept as what we saw and is never post-corrected.**

Concretely, the maintain / revision-refresh logic that overwrites bars to match
Polygon operates **only on `bar_cache`**. It must **never** be pointed at
`live_bars`. When in doubt about which table a piece of update logic should
touch: revision/refresh → `bar_cache`; never `live_bars`.

| | **Live Bars** | **REST Bars** |
|---|---|---|
| Table | `live_bars` | `bar_cache` |
| Source | WebSocket (decision-time) | Polygon REST |
| Seen-bar mutability | **immutable** (never post-corrected) | **revisable** (tracks Polygon) |
| REST fill | gap-heal only, INSERT-only, marked `rest_backfill` | the whole table is REST |
| Purpose | replay + WS-visibility forensics | backtest fidelity + speed |
| Written by | `live_bars_writer.py`, `live_bars_rest_backfill*.py` | `bar_cache.py` |
| Read by | live/replay/forensics paths | `bar_cache.read_bars()` |

---

## Verified facts (2026-06-25) — earlier alarms were measurement artifacts

Two scary findings were **debunked** after careful, row-count-verified testing
(both were bugs in the *test harness*, not the cache):

- **REST Bars 1-second is byte-identical to Polygon REST, and complete.** Clean
  single-day comparison (TSLA): exact row-count parity every trading day
  (06-16 25710=25710, 06-22 25095=25095; 06-19=0 both = Juneteenth holiday),
  `only_in_cache=0`. A fully-settled day (06-16) is **literally zero-diff**
  (`max|Δ|=0.0`, volume exact). Columns are `float8`.
  - *Debunked "50% incomplete":* my baseline `load_from_polygon(end_date=d+1)`
    pulled **two days** (Polygon's date range is inclusive) vs the cache's one.
    Per-hour counts matched once corrected.
- **Polygon does NOT revise settled intraday bars** (confirmed by Polygon docs:
  minute aggregates recalc for 15 min then fold late trades only at EOD; second
  aggregates settle after a 2s + 15min window; daily bars update continuously).
  - *Debunked "revises over days":* the "~197 diffs" were a `1e-9` threshold
    catching **sub-penny** values (compounded by the 2-day join). The only real
    residual: a day captured *very soon* after formation (06-22) has ~0.2% of
    bars differing by **≤5e-5** (half a thousandth of a cent; **0** above a
    hundredth of a cent), from documented late-trade EOD folding. A maintain
    re-fetch once the day is EOD-settled zeroes it; negligible for trades regardless.

**Implication:** Hi-Fi (1-second) from REST Bars is viable and byte-identical.
The speed win comes from a **single load-once read** (one `read_bars` over the
whole period ≈5s for ~530k rows = ~5.7× vs Hi-Fi's current per-day Polygon
fetches), NOT per-day reads (those are a wash). See
`docs/_active/Design_M-RS2_Shared_Bar_Store.md`.

## Remaining nice-to-have (not a blocker)

- Maintain cron could re-fetch a just-captured day once it's EOD-settled to
  drive the sub-penny residual to exactly zero. Low priority (≤5e-5 is
  trade-irrelevant).
