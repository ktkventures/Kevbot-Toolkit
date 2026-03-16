# Phase 33: Mass Strategy Builder — Implementation Spec

## 1. Overview

The Mass Strategy Builder allows users to test thousands of strategy combinations across multiple tickers, timeframes, directions, triggers, confluences, and risk profiles in a single run. It reuses the exact same unified engine pipeline as the Strategy Builder, ensuring backtest results are identical. Results are surfaced as strategy cards that can be saved directly to My Strategies.

**Core principle:** Same engine, same data pipeline, same KPI calculations. The Mass Builder is a parallelized, multi-parameter wrapper around the Strategy Builder's backtest flow.

---

## 2. Data Model

### 2.1 Mass Search Record

Stored in Supabase `mass_searches` table (cloud) or `config/mass_searches.json` (desktop).

```python
{
    "id": "ms_abc123",          # Unique search ID
    "user_id": "uuid",          # Supabase user (cloud only)
    "name": "SPY + QQQ Sweep",  # User-defined name
    "created_at": "2026-03-17T10:30:00Z",
    "updated_at": "2026-03-17T11:45:00Z",
    "status": "completed",      # pending | running | completed | failed | cancelled
    "progress": {
        "current_step": 450,
        "total_steps": 450,
        "current_label": "QQQ 5Min SHORT",  # Human-readable progress label
        "started_at": "2026-03-17T10:30:15Z",
        "elapsed_seconds": 4500,
    },
    "config": {
        "tickers": ["SPY", "QQQ", "AAPL"],
        "date_range": {"mode": "days", "days": 90},  # or {"mode": "range", "start": "...", "end": "..."}
        "timeframes": ["1Min", "5Min", "15Min"],
        "directions": ["LONG", "SHORT"],
        "session": "RTH",
        "entry_triggers": ["cid_ema_cross_bull", "cid_utbot_buy", ...],
        "exit_triggers": ["cid_ema_cross_bear", "cid_utbot_sell", ...],
        "exit_depth": 1,        # Best 1 / 2 / 3 / 4 exits combined
        "tf_confluences": ["cid_ema_stack_bull", "cid_macd_line_pos", ...],
        "tf_confluence_depth": 2,  # Auto-search max factors
        "general_confluences": ["cid_session_rth", ...],
        "general_confluence_depth": 1,
        "rm_packs": ["rm_atr_default", "rm_atr_tight"],  # Risk Management pack IDs
        "required_performance": {
            "min_trades": 10,
            "min_win_rate": null,
            "min_profit_factor": null,
            "min_daily_r": null,
            "min_r_squared": null,
        },
        "max_results": 500,     # Cap on stored results
    },
    "results": [ ... ],         # List of MassSearchResult objects (see below)
    "summary": {
        "total_combinations_tested": 12400,
        "results_stored": 347,
        "best_daily_r": 0.45,
        "best_win_rate": 72.3,
    }
}
```

### 2.2 Mass Search Result (individual strategy)

Each result is a fully specified strategy that can be saved to My Strategies.

```python
{
    "result_id": "msr_001",
    "search_id": "ms_abc123",
    "rank": 1,                  # Sorted by primary KPI (default: daily_r)
    "status": "active",         # active | saved | passed
    "saved_strategy_id": null,  # Set when saved to My Strategies
    "config": {
        # Full strategy config — identical format to strategies.json
        "symbol": "SPY",
        "timeframe": "5Min",
        "direction": "LONG",
        "trading_session": "RTH",
        "entry_trigger": "trig_utbot_buy",
        "entry_trigger_confluence_id": "cid_utbot_buy",
        "entry_trigger_name": "UT Bot Buy",
        "exit_triggers": ["trig_ema_cross_bear"],
        "exit_trigger_confluence_ids": ["cid_ema_cross_bear"],
        "exit_trigger_names": ["EMA Cross Bear"],
        "bar_count_exit": null,
        "stop_config": {"method": "atr", "atr_mult": 1.5},
        "target_config": {"method": "risk_reward", "rr_ratio": 2.0},
        "confluence": ["5m-EMA_STACK-FULL_BULL_STACK", "5m-MACD_LINE-M>S+"],
        "general_confluences": [],
        "data_days": 90,
        "asset_type": "equity",
    },
    "kpis": {
        "total_trades": 47,
        "win_rate": 63.8,
        "profit_factor": 2.14,
        "avg_r": 0.32,
        "total_r": 15.04,
        "daily_r": 0.24,
        "r_squared": 0.87,
        "max_r_drawdown": -3.2,
    },
    "equity_curve": [0.0, 1.2, 0.8, 2.1, 1.5, 3.4, ...],  # Cumulative R series
}
```

### 2.3 Database Schema (Supabase)

```sql
CREATE TABLE mass_searches (
    id TEXT PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id),
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    status TEXT DEFAULT 'pending',
    config JSONB NOT NULL,
    progress JSONB DEFAULT '{}',
    results JSONB DEFAULT '[]',
    summary JSONB DEFAULT '{}',
    CONSTRAINT mass_searches_user_fk FOREIGN KEY (user_id)
        REFERENCES auth.users(id) ON DELETE CASCADE
);

CREATE INDEX idx_mass_searches_user ON mass_searches(user_id);
CREATE INDEX idx_mass_searches_status ON mass_searches(user_id, status);

-- RLS: users can only see their own searches
ALTER TABLE mass_searches ENABLE ROW LEVEL SECURITY;

CREATE POLICY mass_searches_select ON mass_searches
    FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY mass_searches_insert ON mass_searches
    FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY mass_searches_update ON mass_searches
    FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY mass_searches_delete ON mass_searches
    FOR DELETE USING (auth.uid() = user_id);
```

---

## 3. Combination Engine

### 3.1 Execution Strategy

The engine groups work to minimize expensive operations:

```
Level 1: (symbol, timeframe, session)     → prepare_data_with_indicators() [EXPENSIVE, cached]
Level 2: (direction, entry, exits, RM)    → run_trades_from_cache() [FAST, bar_cache replay]
Level 3: (confluence combos)              → filter existing trades by confluence_records [FASTEST]
```

### 3.2 Combination Count Formula

```
Total base configs = |tickers| × |timeframes| × |directions| × |entry_triggers| × C(|exits|, exit_depth) × |rm_packs|

Per base config (if meets min_trades):
  TF confluence combos = Σ(d=1..tf_depth) C(|tf_confluences|, d)
  General confluence combos = Σ(d=1..gen_depth) C(|general_confluences|, d)
  Confluence combos = TF_combos × General_combos  (or TF_combos + General_combos if independent)

Total evaluations = Σ(base configs) × confluence_combos_per_base
```

**Example:** 3 tickers × 3 TFs × 2 directions × 4 entries × 3 exits (depth 1) × 2 RM packs = 432 base configs. With 8 TF confluences at depth 2 = 36 combos each → 15,552 total evaluations. At ~5ms per confluence filter = ~78 seconds.

### 3.3 Core Algorithm

```python
def run_mass_search(search_config, progress_callback):
    results = []

    # Pre-compute data groups
    groups = []
    for symbol in search_config['tickers']:
        for tf in search_config['timeframes']:
            session = search_config.get('session', 'RTH')
            groups.append((symbol, tf, session))

    total_steps = estimate_total_steps(search_config)
    step = 0

    for symbol, tf, session in groups:
        # Level 1: Load data (cached)
        df = prepare_data_with_indicators(symbol, days, seed,
                                          timeframe=tf, session=session,
                                          secondary_tfs=get_secondary_tfs(tf))
        if len(df) == 0:
            continue

        sec_tf_map = get_secondary_tf_map(df)
        enabled_gen = get_enabled_general_packs(load_general_packs())

        for direction in search_config['directions']:
            for entry_cid in search_config['entry_triggers']:
                entry_base = get_base_trigger_id(entry_cid)

                for exit_combo in generate_exit_combos(search_config['exit_triggers'],
                                                       search_config['exit_depth']):
                    for rm_pack in search_config['rm_packs']:
                        stop_config, target_config = get_rm_pack_configs(rm_pack)

                        # Build strategy config
                        config = build_strategy_config(
                            symbol, tf, direction, session,
                            entry_base, entry_cid,
                            exit_combo, stop_config, target_config,
                            search_config['date_range'])

                        # Level 2: Run trades (bar_cache replay)
                        bar_cache, cache_meta = precompute_bar_cache(
                            df, config, general_packs=enabled_gen,
                            secondary_tf_map=sec_tf_map)

                        trades = run_trades_from_cache(
                            bar_cache, cache_meta, config)

                        if len(trades) < search_config['required_performance']['min_trades']:
                            step += 1
                            progress_callback(step, total_steps, f"{symbol} {tf} {direction}")
                            continue

                        # Level 3: Auto-search confluences
                        best_combos = auto_search_confluences(
                            trades, bar_cache, cache_meta, config,
                            tf_confluences=search_config['tf_confluences'],
                            gen_confluences=search_config['general_confluences'],
                            tf_depth=search_config['tf_confluence_depth'],
                            gen_depth=search_config['general_confluence_depth'],
                            required_perf=search_config['required_performance'],
                        )

                        for combo in best_combos:
                            kpis = combo['kpis']
                            if meets_required_performance(kpis, search_config['required_performance']):
                                results.append(build_result(config, combo, kpis))

                        step += 1
                        progress_callback(step, total_steps, f"{symbol} {tf} {direction}")

    # Sort and trim to max_results
    results.sort(key=lambda r: r['kpis']['daily_r'], reverse=True)
    return results[:search_config.get('max_results', 500)]
```

### 3.4 Bar Cache Reuse Optimization

Within a (symbol, TF, session) group, `prepare_data_with_indicators()` is called once. However, `precompute_bar_cache()` depends on the full strategy config (entry trigger, exit triggers, RM pack settings). Different entry/exit triggers require different bar caches because the cache stores trigger booleans specific to those triggers.

**Optimization opportunity:** If we precompute bar caches for ALL triggers at once (not just the strategy's triggers), we can reuse a single cache across all entry/exit combinations within a (symbol, TF) group. This requires a modification to `precompute_bar_cache()` to accept a broader trigger set.

**Decision:** Implement in 33B. For 33A, use per-config bar caches (correct but slower). Optimize in 33B if profiling shows bar_cache computation is the bottleneck.

### 3.5 Auto-Search Confluence Integration

Reuse the existing `find_best_combinations()` and `analyze_confluences()` functions from the Strategy Builder. These already support:
- Configurable depth (1–4 factors)
- KPI-based filtering and sorting
- bar_cache replay for fast iteration

The mass builder calls these functions per base config, collecting the top N results.

---

## 4. Background Execution

### 4.1 Threading Model

```python
import threading

def start_mass_search(search_id, search_config):
    """Launch mass search in a background thread."""
    def _worker():
        try:
            update_search_status(search_id, 'running')

            def _progress(step, total, label):
                update_search_progress(search_id, step, total, label)

            results = run_mass_search(search_config, _progress)

            save_search_results(search_id, results)
            update_search_status(search_id, 'completed')
        except Exception as e:
            update_search_status(search_id, 'failed', error=str(e))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
```

### 4.2 Progress Tracking

- Background thread writes progress to DB/JSON every N steps (e.g., every 10 steps or every 5 seconds)
- UI polls progress via `@st.fragment(run_every=3)` — lightweight, doesn't rerun the full page
- Progress display: `"Analyzing QQQ 5Min LONG — 234 / 450 combinations (52%)"` + progress bar

### 4.3 Railway Constraints

- **Memory:** Cap concurrent bar_cache size. Process one (symbol, TF) group at a time, release cache before next group.
- **CPU:** Single background thread (no multiprocessing on Railway). Yield periodically so the web service stays responsive.
- **Time:** Long searches may exceed Railway's request timeout. Background thread is independent of HTTP requests — results are polled via DB.
- **Throttle setting:** Optional "Processing Speed" selector: Fast (no throttle) / Balanced (10ms sleep between combos) / Conservative (50ms sleep). Default: Balanced.

### 4.4 Cancellation

- User clicks "Cancel" on the Mass Strategy Builder page
- Sets search status to `cancelled` in DB
- Background thread checks status every N steps, exits cleanly if cancelled
- Partial results are preserved (status = `cancelled`, results contain what was found so far)

---

## 5. UI Specification

### 5.1 Mass Strategy Builder Page

#### Section 1: Header Row

```
[Search Name: _______________] [💾 Save]
```

- Text input for search name (auto-generated default: "Search {date}")
- Save button: saves current config + results to Mass Strategy Results

#### Section 2: Config Row

```
[📋 Select Tickers] [⚙ Select Variables] [📊 Required Performance] [🔴 Analyze]
```

- **Select Tickers** — Modal with:
  - Text area for comma/newline-separated tickers
  - Quick-add buttons: "S&P Top 10", "Tech", "Crypto" (predefined lists)
  - Badge showing count: "5 tickers selected"

- **Select Variables** — Modal with tabs:
  - **Date Range:** Days slider or date range picker
  - **Timeframes:** Checkboxes for each available TF
  - **Direction:** Checkboxes for LONG / SHORT
  - **Entry Triggers:** Checkboxes from enabled confluence packs, grouped by pack
  - **Exit Triggers:** Checkboxes + depth selector (1/2/3/4)
  - **TF Confluences:** Checkboxes + depth selector (1/2/3/4)
  - **General Confluences:** Checkboxes + depth selector (1/2/3/4)
  - **Risk Management:** Checkboxes from enabled RM packs
  - Each tab shows count badge: "4 of 12 selected"

- **Required Performance** — Modal with:
  - Min trades (default 10)
  - Min win rate % (optional)
  - Min profit factor (optional)
  - Min daily R (optional)
  - Min R² (optional)

- **Analyze** — Red primary button. Disabled until at least 1 ticker + 1 entry trigger + 1 exit trigger selected. Click starts background analysis.

#### Section 3: Preview + Progress

**Before analysis:**
```
Preview: 3 tickers × 3 TFs × 2 dirs × 4 entries × 3 exits × 2 RM packs = 432 base configs
         + confluence auto-search (depth 2, 8 TF factors) ≈ 15,552 evaluations
         Estimated time: ~2 minutes
```

**During analysis:**
```
[████████████░░░░░░░░] 52% — Analyzing QQQ 5Min LONG (234 / 450)
Elapsed: 1m 12s · Results found: 89
```

**After analysis:**
```
✅ Complete — 347 strategies found from 12,400 combinations tested (4m 32s)
```

#### Section 4: Post-Analysis Filters

Single row of filter controls (only visible after results exist):

```
Sort by: [Daily R ▼]  Min Win Rate: [___]  Min PF: [___]  Min Trades: [___]  Min R²: [___]  Search: [___]
```

- Filters apply instantly to the result cards below (client-side filtering, no rerun needed)

#### Section 5: Result Cards

Card layout (2–3 columns), each card contains:

```
┌─────────────────────────────────────────────┐
│ SPY · 5Min · LONG                    Rank #1 │
│ Entry: UT Bot Buy → Exit: EMA Cross Bear     │
│ Stop: ATR 1.5x · Target: 2.0 RR             │
│ Confluence: EMA_STACK (BULL) + MACD (M>S+)   │
│                                               │
│ [equity curve sparkline ~~~~~~~~~~~]          │
│                                               │
│ Trades  WR     PF    Avg R  Daily R  R²      │
│ 47      63.8%  2.14  +0.32  +0.24   0.87    │
│                                               │
│ [Save to My Strategies]  [Pass]              │
└─────────────────────────────────────────────┘
```

- **Save to My Strategies:** Creates a full strategy in My Strategies (same format as Strategy Builder save). Card shows "✓ Saved" badge after.
- **Pass:** Toggles gray overlay + strikethrough. Reversible (click again to un-pass). Visual-only, doesn't delete.
- **Equity curve:** Small Plotly sparkline or lightweight-charts mini chart. Data source: cumulative R series stored in result.

### 5.2 Mass Strategy Results Page

```
┌──────────────────────────────────────────────────────────────┐
│ Mass Strategy Results                                         │
│                                                               │
│ ┌─────────────────────────────────────────────────────────┐  │
│ │ 🟢 SPY + QQQ Sweep           347 results   Mar 17 4:32m │  │
│ │ 🟡 Tech Momentum Scan        Running... 52%              │  │
│ │ 🔴 Crypto Breakout Search    Failed                      │  │
│ │ ⚪ Broad Market Scan          289 results   Mar 15 8:15m │  │
│ └─────────────────────────────────────────────────────────┘  │
│                                                               │
│ Click a search to view results...                             │
└──────────────────────────────────────────────────────────────┘
```

- List of past searches with status indicators
- Click to expand → shows Section 4 (filters) + Section 5 (cards) inline
- Running searches show live progress bar
- Delete button per search (with confirmation)

---

## 6. File Structure

```
src/
  mass_builder.py         # New — combination engine, background execution
  app.py                  # Modified — new pages: Mass Strategy Builder, Mass Strategy Results
  db.py                   # Modified — mass_searches CRUD (Supabase + local JSON fallback)
```

### 6.1 mass_builder.py — Module API

```python
# ── Combination Engine ──
def estimate_combinations(config: dict) -> dict:
    """Return {base_configs: int, confluence_combos: int, total: int, est_seconds: float}"""

def run_mass_search(search_config: dict, progress_callback) -> list:
    """Execute the full search. Returns list of MassSearchResult dicts."""

def auto_search_confluences(trades, bar_cache, cache_meta, config,
                            tf_confluences, gen_confluences,
                            tf_depth, gen_depth, required_perf) -> list:
    """Run confluence auto-search for a single base config. Returns top combos."""

# ── Background Execution ──
def start_mass_search_async(search_id: str, search_config: dict):
    """Launch search in background thread. Updates DB/JSON with progress."""

def cancel_mass_search(search_id: str):
    """Signal cancellation. Background thread exits on next progress check."""

# ── Helpers ──
def generate_exit_combos(exit_triggers: list, depth: int) -> list:
    """Generate exit trigger combinations up to given depth."""

def build_strategy_config(...) -> dict:
    """Build a strategy config dict compatible with _unified_trades()."""

def build_result(config, combo, kpis) -> dict:
    """Build a MassSearchResult dict from a successful backtest."""

def meets_required_performance(kpis: dict, required: dict) -> bool:
    """Check if KPIs meet all required performance thresholds."""
```

### 6.2 db.py Additions

```python
# ── Mass Search CRUD ──
def save_mass_search(search: dict) -> str: ...
def load_mass_searches() -> list: ...
def get_mass_search(search_id: str) -> dict: ...
def update_mass_search(search_id: str, updates: dict): ...
def delete_mass_search(search_id: str): ...
```

---

## 7. Phase Breakdown

### Phase 33A: Data Model + UI Skeleton

**Scope:**
- [ ] `mass_builder.py` — module skeleton with `estimate_combinations()`, `build_strategy_config()`, `meets_required_performance()`
- [ ] `db.py` — mass search CRUD (Supabase table + local JSON fallback)
- [ ] Supabase migration script for `mass_searches` table + RLS policies
- [ ] Mass Strategy Builder page skeleton in `app.py`:
  - Section 1: Header row (name + save)
  - Section 2: Config row with 4 buttons (modals render but don't execute yet)
  - Select Tickers modal (text area + predefined lists)
  - Select Variables modal (all tabs with checkboxes, reads from enabled packs)
  - Required Performance modal (threshold inputs)
  - Section 3: Combination preview (calls `estimate_combinations()`)
- [ ] Mass Strategy Results page skeleton:
  - List of saved searches (name, date, status, result count)
  - Click to expand (placeholder for cards)
- [ ] Navigation: add both pages to sidebar nav

**Acceptance criteria:** User can configure a search (select tickers, variables, performance thresholds), see the combination preview, and save/load search configs. Analyze button is present but non-functional.

### Phase 33B: Combination Engine

**Scope:**
- [ ] `run_mass_search()` — full combination engine with 3-level optimization
- [ ] Group-by (symbol, TF, session) for data loading
- [ ] `precompute_bar_cache()` per base config, `run_trades_from_cache()` replay
- [ ] `auto_search_confluences()` — reuse Strategy Builder's `find_best_combinations()` / `analyze_confluences()`
- [ ] `generate_exit_combos()` — itertools.combinations for exit depth
- [ ] KPI calculation + equity curve extraction per result
- [ ] Required Performance pre-filter at each level
- [ ] Sort + trim to max_results
- [ ] Unit tests for combination count estimation, exit combo generation, performance filtering

**Acceptance criteria:** `run_mass_search()` can be called synchronously and returns correct results. Verified against manual Strategy Builder runs for a small set of tickers/triggers.

### Phase 33C: Background Execution + Progress

**Scope:**
- [ ] `start_mass_search_async()` — threading.Thread wrapper
- [ ] Progress callback writes to DB/JSON every 10 steps or 5 seconds
- [ ] UI: Analyze button starts async search, page shows progress bar via `@st.fragment(run_every=3)`
- [ ] Cancellation: UI cancel button → DB status update → thread exit
- [ ] Railway throttle: optional sleep between combinations (configurable)
- [ ] Memory management: release bar_cache after each (symbol, TF) group
- [ ] Error handling: catch exceptions per combination (skip, don't crash entire search)
- [ ] Partial results: save results to DB incrementally (every 50 results or every group)

**Acceptance criteria:** User can start a search, navigate away, come back, and see progress/results. Cancellation works. Railway deployment doesn't OOM on moderate searches (5 tickers × 3 TFs × 20 triggers).

### Phase 33D: Result Cards + Save Flow

**Scope:**
- [ ] Result card component: KPIs, config summary, equity curve sparkline
- [ ] Equity curve: small Plotly line chart from cumulative R series
- [ ] Save to My Strategies: creates strategy with same format as Strategy Builder save, sets `saved_strategy_id` on result, shows "✓ Saved" badge
- [ ] Pass toggle: visual gray-out, stored in result status field, reversible
- [ ] Post-analysis KPI filters/sort (Section 4): client-side filtering of result cards
- [ ] Mass Strategy Results page: click search → expand to show filter + cards
- [ ] Delete search (with confirmation dialog)
- [ ] Search status indicators: green (completed), yellow (running), red (failed), gray (cancelled)

**Acceptance criteria:** Full end-to-end flow: configure → analyze → view results → filter → save strategies → view in My Strategies. Saved strategies work identically to Strategy Builder strategies (backtests, live charts, alerts, portfolios).

---

## 8. Risk & Constraints

| Risk | Mitigation |
|------|------------|
| Combinatorial explosion (millions of combos) | Required Performance pre-filter, max_results cap, combination preview with time estimate |
| Railway OOM on large searches | Release bar_cache per group, throttle setting, MAX_BARS_CLOUD check per ticker |
| Long-running searches (30+ minutes) | Background thread, incremental result saving, cancellation support |
| Stale cached data (Alpaca @st.cache_data) | Use same TTL as Strategy Builder (1 hour). Searches within TTL reuse cached data. |
| Result storage size (500 results × equity curves) | Equity curve stored as compact float list (~100 trades × 8 bytes = 800 bytes per result). 500 results ≈ 400KB. |
| Bar cache memory per (symbol, TF) group | Process groups sequentially, release cache after each. Typical cache: ~10MB for 30 days of 1Min data. |

---

## 9. Future Enhancements (Post-Phase 33)

- **Scheduled searches:** Run mass searches on a cron (e.g., weekly market sweep)
- **Cross-search comparison:** Compare results across multiple mass searches
- **Template searches:** Save search configs as reusable templates
- **Portfolio auto-builder:** Auto-generate portfolios from top N results with diversification constraints
- **Parallel execution:** Multi-threaded processing within a (symbol, TF) group (requires thread-safe bar_cache)
