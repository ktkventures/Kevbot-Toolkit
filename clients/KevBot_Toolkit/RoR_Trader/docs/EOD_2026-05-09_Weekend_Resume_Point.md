# EOD 2026-05-09 — Weekend resume point

## State entering weekend

**Active branch:** `dev` (HEAD = `0ca6ee8` deploy log entry).
**Backup branches in safety net:**
- `dev-backup-eod-2026-05-09-pre-weekend` (today)
- `dev-backup-pre-builder-uis-2026-05-08`
- `dev-backup-pre-lane-mode-matrix-2026-05-08`
- `dev-backup-pre-algo-model-2026-05-07-evening`
- `dev-backup-pre-divergence-tab-2026-05-07`

**Migrations applied on Supabase (intermittent during outage):**
- `algo_history_cron_cycles.sql` (2026-05-07)
- `alerts_live_model_column.sql` (2026-05-07)

**Bulk migrations applied:**
- All 20 strategies on `backtest_model=rest_hifi`, `algo_model=cache_locked`,
  `live_model=ws_agg_locked` (sid 132 still has wiped config)

## What's shipped and verified

- ✅ **Hi-Fi Phase 1 (signal-exit refinement)** — eppv3/eppv4/utv4
- ✅ **AI pack-creation guardrail** — three-layer trigger_levels enforcement
- ✅ **Cron cycle stats panel** — DB+Worker+API+Jobs page
- ✅ **Divergence tab v1** — 3-lane comparison
- ✅ **algo_model split** — 3 model fields, bulk migration
- ✅ **Live model labeling** — verified 1600 alerts on 2026-05-08
  carry `ws_agg_locked`
- ✅ **Lane×mode matrix complete** — all 4 update combinations wired
- ✅ **JWT-safe /update endpoint** — admin context wrapper
- ✅ **Button uniformity** — Strategy Detail header + Divergence tab +
  bulk job worker all converge on lane×mode dispatch
- ✅ **Builder UIs** — Strategy Builder + Mass Builder 3-dropdown picker
- ✅ **Admin rename** — Backtest Models → Backtest / Algo Models
- ✅ **REST-vs-CACHE divergence v2 design doc** — Option B (JSONB map)

## 2026-05-08 Supabase outage — collected data

Despite all-day Supabase intermittent outages:
- **1,625 alerts captured during 2026-05-08** (08:00-16:21 UTC)
- 1,600 carry `live_model='ws_agg_locked'` (correctly stamped)
- 25 carry `live_model='ws_with_corrections'` — INVESTIGATE: likely
  one strategy on legacy live model
- Lost coverage: ~16:21-20:00 UTC late RTH (Worker presumably struggled)

## 2026-05-09 Bulk Update All Data observation

**3 strategies hit "read operation timeout" per user screenshot.**
Recovery: re-click Update All Data on those 3 once Supabase is reliably
healthy. JWT-safety wrapper from yesterday's deploy should let them
succeed on a clean attempt.

**Concerning side-finding (open for Monday):** all 19 strategies'
`stored_trades` show latest entry from April 9-30 despite
`last_recompute_until_ts` stamped today (20:11-20:15 UTC). Algo lane
refreshed correctly (trades table has May 1-8 data on sid 152). Backtest
lane appears to have silently failed-soft on the whole bulk run.
Hypothesis: rest_hifi engine timed out under Supabase intermittent
issues, OR `recompute_and_persist_algo_trades` is overwriting
stored_trades, OR `update_strategy_admin` is being filtered. Roadmap
item: `8.7-stored-trades-staleness`.

## Verification list — weekend (markets closed)

Doable Saturday/Sunday with whatever Supabase health is available:

1. **Strategy Builder** open → see Models card (3 dropdowns)
2. **Mass Builder** Config tab → see 3 model dropdowns
3. **Sidebar nav** → "Backtest / Algo Models"
4. **Admin page** at `/admin/backtest-models` → page title +
   description updated
5. **Strategy Detail Configuration tab** → 3 dropdowns visible
6. **Strategy Detail header** → 2 buttons (Update New + Update All)
   instead of single "Refresh"
7. **Divergence tab on sid 152/153/154** → review 3-way matched count
   from yesterday's 1,625 alerts; confirm Live model column shows
   `ws_agg_locked`
8. **Cron stats panel** at `/jobs` → see if cycles populated post-outage
9. **Click Update All Data on one of the 3 errored strategies** →
   confirm both lanes succeed (no JWT error)
10. **Click Update New Data** → confirm backtest now does forward
    append (not skipped)

If anything fails: leave for Monday. Don't fight Supabase.

## Verification list — Monday RTH

Needs fresh live alerts firing:

1. New Monday alerts carry `live_model='ws_agg_locked'`
2. Cron stats panel populates fresh cycles every 5 min
3. Real-time Divergence updates as alerts fire mid-RTH
4. **stored_trades-stuck-at-April investigation** (see above) — click
   Update All Data on sid 152 with healthy Supabase; verify
   stored_trades extends past April 30
5. **25 ws_with_corrections alerts investigation** — find which
   strategy(ies) and decide if intentional or migration miss
6. Bulk Update All Data on multiple strategies — verify no
   JWT/timeout errors when system is fully healthy
7. Re-test mode=new on backtest lane — verify it actually extends
   stored_trades JSONB

## Known issues entering weekend

- **stored_trades staleness** (above) — investigate Monday
- **25 alerts on ws_with_corrections live_model** — investigate Monday
- **3 strategies erred on bulk Update All Data** — recover by
  re-clicking once Supabase stable
- **Cron starvation** on heaviest strategies — roadmap item
  `8.7-cron-throughput-starvation`
- **Cron stats panel "0 buckets" anomaly** — roadmap item
- **sid 144 alerts-count UI bug** — roadmap item, cosmetic

## Active in-progress work

None. Punch list closed for the weekend. Resume Monday with
verification checklist + the stored_trades investigation.

## Roadmap reference

- `docs/Roadmap_To_Scale.md` — items `8.7-divergence-tab-v1`,
  `8.7-algo-model-split`, `8.7-lane-mode-matrix`, `8.7-builder-uis`
  marked complete this week. Open: `8.7-stored-trades-staleness`,
  `8.7-rest-vs-cache-divergence-v2`, `8.7-cron-throughput-starvation`.
- `docs/Design_REST_vs_CACHE_Divergence_v2.md` — v2 design doc
  written 2026-05-08, ready when execution begins.
- `~/.claude/plans/synchronous-tickling-yeti.md` — daily punch lists.
