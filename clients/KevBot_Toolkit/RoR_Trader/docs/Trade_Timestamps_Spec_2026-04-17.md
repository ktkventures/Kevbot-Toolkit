# Trade Timestamps Spec — 2026-04-17

## Design decisions (locked 2026-04-17)

Six decisions are locked before implementation begins. Any change to these
reopens the spec discussion.

1. **Timestamps per side: 2, not 3.** Each side of a trade (entry/exit) stores
   `trigger_ts` and `fill_ts` only. Confirmation is never a distinct
   user-facing moment — in Behavior A it collapses into `fill_ts`, in
   Behavior B passing validation is silent and failing validation materializes
   as an exit event with its own `exit_trigger_ts`. A separate `confirm_ts`
   field would add data-model noise without informing display.
2. **Column naming: `entry_*_ts` / `exit_*_ts`.** Descriptive and explicit —
   names carry side (entry/exit), role (trigger/fill), and type (timestamp).
   Uniform across every surface.
3. **Exec type schema: flexible manifest.** `exec_type` stored as a free-text
   identifier plus a JSON manifest describing its lifecycle. Supports
   unlimited user-defined and AI-generated exec types without schema changes.
4. **`alerts.timestamp` stays** as the audit field (wall-clock DB INSERT time).
   Application logic reads `fill_ts` / `trigger_ts` instead; `timestamp`
   survives only for forensic debugging.
5. **Webhook payload: clean slate.** No legacy aliases, no backward-compat
   `entry_time` field in new webhook payloads. Downstream integrations were
   never heavily invested in the old shape and starting fresh avoids
   perpetually maintaining two mental models.
6. **LC / CC default behavior: B (fill-then-validate).** Matches current
   implementation (`unified_engine.py:1807-1813` — `pending_confirm_bar` +
   `bail_action` machinery). Behavior A variants are deferred to a future
   spec; the data model supports both but no Behavior A pack ships in this
   phase.

Also locked: **soft clean-slate migration.** On migration, keep strategy
config rows, wipe `stored_trades = []`, truncate old `alerts`. Strategies
restart forward-testing clean under the new model; existing configs are
preserved so visual testing still works after a single `/refresh` per
strategy.

---

## Context

The trade record stored in `stored_trades` and the alert row stored in `alerts`
currently anchor on different moments of the same logical event. Concretely, on
a 1Min C-type trade:

- `alerts.timestamp` = wall-clock INSERT time (bar close + ~0.7s save latency)
- `stored_trades.entry_time` = firing bar's START time (bar close − 60s)

For a bar labeled `21:21:00` (covers `[21:21:00, 21:22:00)`), this produces:

- Alert history row displays `21:22:00.674`
- Algo history row displays `21:21:00`

They represent the same trade but display ~1 minute apart, because one is
anchored on when Supabase received the row and the other is anchored on the
bar's identifier. The effect is worse on higher timeframes — on 15Min bars,
trigger and fill can display 15 minutes apart.

This spec defines a two-timestamp model (`trigger_ts` / `fill_ts`) applied to
both entry and exit sides of every trade, picks `fill_ts` as the universal
display anchor across every surface, and catalogs every codebase location
that must change to adopt the model. The spec also handles execution types
with hold behavior (both pre-filter and post-check variants) and the
per-exec-type event lifecycle that drives webhook emission.

---

## Part 1 — The two-timestamp model

### Definitions

Each side of a trade (entry, exit) has two stored timestamps:

| Field | Meaning |
|---|---|
| `entry_trigger_ts` | The instant the entry condition first became true (bar close for C, level cross for L, first bar close for CC, etc.). |
| `entry_fill_ts` | The instant the position actually opened. Computed per exec type manifest. |
| `exit_trigger_ts` | The instant the exit condition first became true. |
| `exit_fill_ts` | The instant the position actually closed. |

Four timestamps total per closed trade. Plus supporting descriptive fields:

| Field | Meaning |
|---|---|
| `exec_type` | Free-text identifier (`'C'`, `'L'`, `'LC'`, `'CC'`, or user-defined). |
| `hold_duration_s` | Configured hold period in seconds (0 if none). Descriptive, not a measured moment. |
| `behavior` | `'A'` (wait-then-fill) or `'B'` (fill-then-validate). Informs display; doesn't add a timestamp. |

### Why 2, not 3

A third timestamp representing "confirmation" was considered and rejected —
it's never a distinct user-facing moment:

- **Behavior A** (wait-then-fill): confirmation moment equals fill moment.
- **Behavior B passing**: nothing happens at the would-be confirm moment; the
  trade continues silently.
- **Behavior B failing**: validation failure materializes as an **exit event**
  with its own `exit_trigger_ts` + `exit_fill_ts` + `exit_reason =
  'validation_failed'`. The confirm moment IS the exit event's trigger.

So the data model never needs a `confirm_ts` field. `hold_duration_s` is
enough to reconstruct the in-between moment if any downstream tool wants it,
and failed confirmations surface naturally as exits.

### Exec type lifecycle table

| Exec type | Behavior | `entry_trigger_ts` | `entry_fill_ts` | `fill_ts - trigger_ts` |
|---|---|---|---|---|
| **C** (candle close) | — | bar close | = trigger | 0 |
| **L** (level cross, immediate) | — | cross moment | = trigger | 0 |
| **LC** pre-filter (future variant) | A | cross moment | trigger + hold_s (if still valid) | `hold_s` |
| **LC** post-check (current default) | B | cross moment | = trigger | 0 |
| **CC** pre-filter (future variant) | A | bar N close | bar N+1 open (if still valid) | `bar_duration` |
| **CC** post-check (current default) | B | bar N close | bar N+1 open | `bar_duration` |

For **current** `LC` and `CC`: Behavior B is the default per locked decision
#6. Behavior A variants can be added later without schema changes — just new
exec type manifests.

### Behavior A vs Behavior B — how the model handles both

- **Behavior A** ("wait, then fill"). Trigger fires, wait `hold_duration_s`,
  if still valid then fill, else no entry. `entry_fill_ts > entry_trigger_ts`
  by `hold_duration_s`. If the hold fails, **no trade record exists**; only a
  `trigger` event and a `cancel` event land in the `alerts` table.
- **Behavior B** ("fill, then validate"). Trigger fires, fill immediately.
  `entry_fill_ts == entry_trigger_ts` (or one bar later for CC). Over the
  hold period, the engine watches for invalidation. If validation fails, it
  emits a normal exit with `exit_reason = 'validation_failed'`. The trade
  record always exists.

### Universal display anchor

Every surface displays `fill_ts` as the primary anchor. `trigger_ts` is
available in tooltips, expanded rows, and scenario replays — never in the
primary column of a table.

Clean parity:

- Alert history table — anchored on `entry_fill_ts` / `exit_fill_ts`
- Algo history table — anchored on `entry_fill_ts` / `exit_fill_ts`
- Chart entry/exit arrows — placed at `fill_ts`
- Chart trigger dots (optional) — placed at `trigger_ts` behind a user toggle

On a 15Min C-type strategy, both tables display `15:30` (the fill moment)
instead of one showing `15:15` (bar start) and the other showing `15:30:00.7`
(wall-clock save). Traders see the moment they could have executed.

### Storage strategy

Store `trigger_ts` and `fill_ts` explicitly on each trade record and alert
row — no compute-on-display. The fill formula is exec-type-dependent and
belongs in one place (the exec type manifest). Writing the computed result to
storage keeps every downstream consumer honest with a single source of truth.

---

## Part 2 — Event lifecycle and webhook policy

Each exec type declares which events it emits:

| Event | Fires when |
|---|---|
| `trigger_event` | At `entry_trigger_ts` or `exit_trigger_ts`. Optional per exec type — off by default, opt-in for early-warning packs. |
| `fill_event` | At `entry_fill_ts` or `exit_fill_ts`. On by default. This is the "normal" alert most consumers want. |
| `cancel_event` | When Behavior A's hold fails (no fill occurred). Paired with a prior `trigger_event`. |

**There is no separate `confirm_event`.** A successful Behavior B validation
is silent. A failed Behavior B validation becomes an `exit_trigger_event`
paired with an `exit_fill_event` where `exit_reason = 'validation_failed'`.
Standard exit machinery, no special cases.

Webhook groups subscribe to whichever events they want to route. Simple
platforms consume `fill_event` only. Sophisticated flows subscribe to
`trigger + cancel` for intent signaling.

### Records

- `alerts` table — **one row per event**, including triggers that never
  filled and cancels. Each row has `event_type`, the side (`entry` or
  `exit`), and the relevant `trigger_ts` / `fill_ts` (whichever apply to the
  event). Full lifecycle reconstructible by grouping rows on `(strategy_id,
  entry_trigger_ts)`.
- `stored_trades` — **one record per filled trade only**. A trigger that
  never filled (Behavior A hold fail) produces no trade record; only alert
  rows.

This keeps `stored_trades` representing "trades that executed" while `alerts`
retains full observability.

---

## Part 3 — Codebase touchpoints

### Backend — trade generation & alert dispatch

| File | Location | What changes |
|---|---|---|
| `src/unified_engine.py` | `PositionState` class (line ~1601, `@dataclass`) | Add `entry_trigger_ts`, `entry_fill_ts`, `hold_duration_s`, `behavior` fields. |
| `src/unified_engine.py` | `PositionStateMachine.check_entry()` (lines 1730-1814) | Set `entry_trigger_ts` and `entry_fill_ts`. Delegate `fill_ts` computation to the exec type manifest. |
| `src/unified_engine.py` | `PositionStateMachine.check_entry_intrabar()` (line 2044+) | Same as `check_entry()` for the live-tick path. |
| `src/unified_engine.py` | `PositionStateMachine.get_trade_record()` (lines 1921-1976) | Emit four timestamps (`entry_trigger_ts`, `entry_fill_ts`, `exit_trigger_ts`, `exit_fill_ts`) plus `hold_duration_s` and `behavior`. No legacy `entry_time` / `exit_time` aliases — clean break per locked decision #5. |
| `src/unified_engine.py` | `PositionStateMachine._exit()` (line 1998) | Set `exit_trigger_ts` and `exit_fill_ts` before `_reset_position()`. |
| `src/unified_engine.py` | `PositionStateMachine._signal_exit()` (lines 2004-2040) | Include the exit timestamps in the signal dict. |
| `src/execution_types/*.py` (C, L, LC, CC modules) | New `ExecTypeManifest` + `compute_fill_ts()` helper | Each module exposes a declarative manifest (see Part 5) and a pure function `fill_ts = f(trigger_ts, bar_duration, hold_duration_s, behavior)`. |
| `src/ralph_engine.py` | Bar-close pipeline where signals are dispatched | Signal dict carries `entry_trigger_ts` and `entry_fill_ts` (or exit counterparts) to alert callback. |
| `src/worker.py:84-225` | `DBAlertDispatcher.dispatch()` + `_persist_algo_trade()` | Include all four timestamps (entry + exit, trigger + fill) in the alert row and trade record. Map to new top-level columns once schema migrates. |
| `src/db.py` | `save_alert()` / `save_alert_admin()` | Write new timestamp columns. Remove JSONB-fallback for timestamp fields (keep fallback for genuine extensions like portfolio_context). |

### Backend — API + services

| File | What changes |
|---|---|
| `src/api/services/backtest_service.py` | Trade serializer emits `entry_trigger_ts` / `entry_fill_ts` / etc. `classify_chart_indicators()` unchanged. |
| `src/api/services/forward_test_service.py` | `_serialize_trades()` includes the six new timestamp fields. |
| `src/api/routers/strategies.py` | `/chart-data` + `/refresh` response shapes include the timestamps. No sanitization changes needed. |
| `src/api/routers/alerts.py` | `/api/alerts/strategy/{id}` response includes `event_type`, `side`, `trigger_ts`, `fill_ts`. |
| `src/api/routers/execution_types.py` | Each pack's API response declares its event lifecycle (`{trigger: bool, fill: bool, cancel: bool}`). |
| `src/api/schemas/backtest.py` | Pydantic models gain optional `entry_trigger_ts` / `entry_fill_ts` fields. |

### Backend — tests

| File | What changes |
|---|---|
| `src/test_unified_parity.py` | Add: each exec type produces correct timestamps on a known bar sequence. |
| `src/test_ralph_subminute_parity.py` | Same for live streaming path. |
| `src/test_c_type_real_backtest.py` / `src/test_c_type_swing_no_lookahead.py` | Update assertions referencing `entry_time` to use `entry_fill_ts`. |
| `src/test_overnight_gap_full_cycle.py` | Same. |
| New: `src/test_timestamp_semantics.py` | Comprehensive: C/L/LC/CC × Behavior A/B × with/without hold, producing the exact four timestamps (`entry_trigger_ts`, `entry_fill_ts`, `exit_trigger_ts`, `exit_fill_ts`) and verifying UI-facing display equivalents. |

### Frontend — display surfaces

| File | What changes |
|---|---|
| `frontend/src/views/StrategyDetailPage.tsx` | Alert history table displays `entry_fill_ts` / `exit_fill_ts`. Algo history table unchanged in anchor (already displays `entry_time` which becomes `entry_fill_ts`). |
| `frontend/src/charts/buildStrategyChartPanes.ts` | Marker placement on `fill_ts`. Optional trigger-dot markers behind a `showTriggers` toggle. |
| `frontend/src/components/EnhancedTradeTable.tsx` | Trade table rows display `entry_fill_ts` / `exit_fill_ts` + optional tooltip with trigger and confirm times. |
| `frontend/src/components/TradeZoomModal.tsx` | 1-second drill-down timeline labels the three moments. |
| `frontend/src/components/TradeWorkflowModal.tsx` | Show trigger → confirm → fill timeline for the selected trade. |
| `frontend/src/components/ScenarioReplayCard.tsx` + `hooks/useScenarioReplay.ts` | Replay steps through `trigger_ts` → `fill_ts` with accurate time deltas (and, for failed Behavior A holds, through the cancel moment). |
| `frontend/src/views/AlertsPage.tsx` | Row time column displays `fill_ts` when present, else `trigger_ts`. Event type column. |
| `frontend/src/views/DashboardPage.tsx` | P&L keyed on `fill_ts`. |
| `frontend/src/views/StrategyBuilderPage.tsx` | Backtest results table anchored on `fill_ts`. |
| `frontend/src/views/MassBuilderPage.tsx` | Mass backtest results anchored on `fill_ts`. |
| `frontend/src/views/PackBuilderPage.tsx` | Exec type pack wizard shows its lifecycle diagram; user can toggle which events the pack emits (`trigger`/`fill`/`cancel`). |
| `frontend/src/charts/PerformanceVsPlan.tsx` | Equity curve x-axis on `fill_ts`. |
| `frontend/src/components/SandboxPanel.tsx` | Sandbox display uses `fill_ts`. |
| `frontend/src/hooks/queries/useAlerts.ts` | Alert shape type extended with `event_type`, `side`, `trigger_ts`, `fill_ts`. |
| `frontend/src/hooks/queries/useStrategies.ts` | Strategy / trade types include new fields. |

### Archived frontend version files

These exist but appear unreferenced by the active app and are treated as
reference-only — not updated unless user mounts them again:

- `frontend/src/app/strategy-builder/versions/V2.tsx`, `V3.tsx`, `V4.tsx`, `V5.tsx`
- `frontend/src/app/mass-builder/versions/V2.tsx`, `V5.tsx`, `V6.tsx`
- `frontend/src/app/strategies/[id]/versions/V2.tsx`, `V3.tsx`, `V4.tsx`, `V5.tsx`
- `frontend/src/app/alerts/versions/V4.tsx`, `V5.tsx`
- `frontend/src/app/dashboard/versions/V4.tsx`, `V5.tsx`, `V6.tsx`, `V7.tsx`

---

## Part 4 — Database migration

### `alerts` table (current schema)

Current top-level columns (from live DB probe 2026-04-17):

```
['acknowledged', 'data', 'direction', 'id', 'source', 'strategy_id',
 'strategy_name', 'symbol', 'timeframe', 'timestamp', 'type', 'user_id',
 'webhook_sent']
```

Columns the code writes to but silently drops into `data` JSONB (pre-existing
schema drift):

```
bar_time, trigger, price, stop_price, target_price, atr,
entry_price, exit_price, created_at, webhook_deliveries
```

### Proposed new `alerts` schema

Add as top-level columns:

- `event_type` TEXT — one of `trigger`, `fill`, `cancel`, `stop`, `target`,
  `signal_exit`, `time_exit`, `validation_failed`, etc.
- `side` TEXT — `entry` or `exit`
- `trigger_ts` TIMESTAMPTZ — the trigger moment for this event's side
- `fill_ts` TIMESTAMPTZ (nullable — null for trigger-only events and cancels)
- `exec_type` TEXT
- `trigger_id` TEXT (previously `trigger`, dropped from JSONB drift)
- `price` NUMERIC
- `bar_time` TIMESTAMPTZ (raw bar label, kept for backward reference)

Promote from JSONB to top-level while we're in there (cleans up the
`webhook_deliveries` and `bar_time` drift identified during recovery). Keep
`data` JSONB for genuinely schema-less extensions (portfolio_context, webhook
delivery metadata).

**No `confirm_ts` column** — per locked decision #1, confirmation is never a
distinct user-facing moment.

Drop: nothing. `timestamp` column stays (per locked decision #4) as
server-side DB insert time for audit purposes; application logic reads
`fill_ts` instead.

### `stored_trades` JSONB shape

Add fields per entry:

```
entry_trigger_ts, entry_fill_ts,
exit_trigger_ts,  exit_fill_ts,
hold_duration_s,  behavior
```

No legacy `entry_time` / `exit_time` aliases — per locked decision #5 (clean
slate), these fields are dropped entirely from the new trade record shape.
Any reader referencing them must be updated.

### Migration strategy: soft clean-slate

Per the locked decision, existing trade data is wiped rather than backfilled:

- **Existing strategy rows:** kept intact. Name, symbol, timeframe, config,
  stop/target/confluence settings all preserved.
- **Existing `stored_trades`:** set to `[]` on every strategy during
  migration. Empty array, clean slate.
- **Existing `alerts` rows:** TRUNCATE the table (or filter to `created_at >=
  migration_date` if audit needs suggest keeping a window).
- **Forward testing restarts cleanly** under the new model. User clicks
  `Update All Data` (`/api/strategies/{id}/refresh`) on any strategy to
  regenerate `stored_trades` from live Polygon bars using `unified_engine`
  under the new timestamp model.

Migration script in `src/migrations/2026_04_XX_timestamp_model.sql` (or
equivalent Python) does the `stored_trades = []` wipe and `alerts`
truncation in one transaction. Reversible only via backup restore — take a
Supabase snapshot first.

### Why soft-clean-slate beats backfill

- Legacy `entry_time` was "firing bar's START" for old rows. For C-type on
  1Min, that's 1 minute before the bar's close. Backfilling to `fill_ts` =
  `entry_time + bar_duration` works for C-type but introduces approximation
  for L-type (no intra-bar cross timestamp stored).
- Mixed-schema state (some rows with `entry_fill_ts`, some with legacy
  `entry_time` semantic + approximation flag) adds conditional logic to
  every read path indefinitely.
- User isn't attached to existing trade data — strategies are treated as
  configs for visual testing, not as historical records of real money.
- `/refresh` regenerates trade data correctly from Polygon bars on demand.
  Recovery of "what trades would this strategy have made last week" is one
  click away and produces cleaner data than any backfill could.

---

## Part 5 — User-facing semantics & pack definitions

### Exec type pack manifest

Each exec type pack (`execution_types/c.py`, `l.py`, `lc.py`, `cc.py`, and
future user-created or AI-generated packs) declares:

```python
ExecTypeManifest(
    code='LC',
    display_name='Level Cross (Confirmed)',
    description=(
        'Cross above (LONG) or below (SHORT) the trigger line; enter '
        'immediately; exit early with reason=validation_failed if price '
        'drops back through the line during the hold window.'
    ),
    trigger_kind='level_cross',        # 'bar_close' | 'level_cross' | 'custom'
    behavior='B',                      # 'A' = wait-then-fill, 'B' = fill-then-validate
    hold_duration_bars=1,              # mutually exclusive with hold_duration_seconds
    hold_duration_seconds=0,
    fill_offset='immediate',           # 'immediate' | 'next_bar_open' | 'after_hold'
    events_emitted=['fill_event'],     # 'trigger_event' | 'fill_event' | 'cancel_event'
    early_exit_on_invalidation=True,   # Behavior B: emit exit_reason=validation_failed
    source='system',                   # 'system' | 'user' | 'ai_generated'
    created_by=None,                   # user_id if source != 'system'
    created_at=None,
)
```

Every existing exec type (C, L, LC, CC) migrates to this manifest format with
their current behavior preserved. No breaking changes — the manifest just
makes what was implicit in module code explicit in structured data, so
user-created and AI-generated exec types can plug into the same machinery.

**Behavior mapping for existing types:**

```
C  → trigger_kind=bar_close,    fill_offset=next_bar_open, hold_s=0, behavior=B (no hold)
L  → trigger_kind=level_cross,  fill_offset=immediate,     hold_s=0, behavior=B (no hold)
LC → trigger_kind=level_cross,  fill_offset=immediate,     hold_bars=1, behavior=B, early_exit=True
CC → trigger_kind=bar_close,    fill_offset=next_bar_open, hold_bars=1, behavior=B, early_exit=True
```

### Pack Builder — existing exec types

"Execution Types" page lists C / L / LC / CC cards plus any user-created /
AI-generated types. Each card shows:

- The manifest's `display_name` and `description`
- A **lifecycle timeline component** — two dots (`trigger`, `fill`) with a
  labeled bar between them showing `hold_duration`. For Behavior B with
  early-exit enabled, a dashed line continues past `fill` to a `?` marker
  showing the validation moment and "exits here if invalidated."
- Event toggles (`trigger_event`, `fill_event`, `cancel_event`) shown as
  badges
- Edit button → opens the manifest in a form view for tweaking parameters

### Pack Builder — AI Wizard for new exec types

Matches the pattern from the existing user-packs wizard. A
`+ Create Execution Type` button on the Execution Types page opens a
multi-step wizard:

1. **Free-form description** — "Describe what you want this execution type to
   do in plain English." Example user input: *"I want to enter when the price
   crosses above the line, but only if it stays above the line for at least
   5 seconds. If it drops back below during those 5 seconds, don't enter at
   all."*
2. **LLM parses the description** into structured answers:
   - `trigger_kind = 'level_cross'`
   - `behavior = 'A'` (wait-then-fill)
   - `hold_duration_seconds = 5`
   - `fill_offset = 'after_hold'`
   - `events_emitted = ['trigger_event', 'fill_event', 'cancel_event']` (A
     needs cancel for the hold-fail path)
3. **User confirms each parameter** in a step-by-step review, with the
   lifecycle diagram re-rendering as answers change. They can edit any
   field inline.
4. **Live preview** — timeline animates showing "trigger at 10:00:00 → hold
   5s (green bar) → fill at 10:00:05" with a "What if the hold fails?"
   toggle that replays the cancel path.
5. **Name + save** — user names the pack (`"Level Cross (5s Hold)"`), it
   lands on the Execution Types page as a new card, available immediately
   in the Strategy Builder's exec type dropdown alongside the system packs.

LLM prompt template lives in `src/pack_builder_exec_type_context.md`,
analogous to the existing `src/pack_builder_context.md` for indicator packs.
Gets updated alongside the manifest schema so prompts always match current
shape.

### Edit flow for existing packs

Clicking Edit on any exec type card (system or user) opens the same wizard,
pre-populated with the pack's current manifest. User can tweak fields or the
description and regenerate. System packs (C, L, LC, CC) can be edited but
display a warning: "Editing a system exec type affects all strategies using
it. Create a new exec type from this one instead?" with a "Clone to new"
shortcut.

### User-visible copy

Documentation, tooltips, and UI labels use:

- **Trigger Time** — "when the signal first appeared" (supplementary; in
  tooltips and expanded rows)
- **Fill Time** — "when the position actually opened/closed" (primary
  anchor; always visible on every table and chart)
- **Hold Duration** — "how long the exec type waited before filling
  (Behavior A) or validated after filling (Behavior B)"

Avoid "entry time" as a display label — ambiguous between trigger and fill.
Use "Fill Time" as the canonical table column header.

---

## Part 6 — Rollout plan

Suggested branching and ordering:

1. **Branch `feat/timestamp-spec`** — all work on this branch until merged.
2. **Phase 1 — backend data model.** Update `PositionState`, `get_trade_record()`,
   `check_entry()` / `_signal_exit()`, exec type modules. Keep `entry_time` as
   deprecated alias. Parity tests pass (including new timestamp assertions).
3. **Phase 2 — schema migration + soft clean-slate.** Add new columns to
   `alerts`. Wipe `stored_trades = []` on every strategy row. Truncate
   `alerts` rows older than migration timestamp. Take a Supabase snapshot
   immediately before this step.
4. **Phase 3 — backend serializers + API.** `backtest_service`,
   `forward_test_service`, `api/routers/*` emit the new fields. Clean break
   — no legacy field aliases (per locked decision #5).
5. **Phase 4 — frontend display layer.** Switch every table, chart, modal,
   replay UI to display `fill_ts`. Trigger-dot toggles on charts.
6. **Phase 5 — Pack Builder + AI Wizard.** Manifest format shipped for
   existing exec types. AI Wizard for user-created exec types implemented on
   the Execution Types page.
7. **Phase 6 — verification.** Each strategy clicked through: `Update All
   Data` regenerates `stored_trades` cleanly under the new model. Alert +
   algo history display same anchors. Live trades fire with correct
   timestamps end-to-end.

Total estimate: ~3-5 days of focused work, ~half in backend, half in
frontend. No code written until you've reviewed and signed off on this
spec.

---

## Part 7 — Risks and open questions

### Risks

- **Chart marker placement shifts visibly.** After migration, every
  regenerated trade's chart markers land at `fill_ts` instead of the legacy
  bar-start position. Since all `stored_trades` start empty post-migration
  and repopulate via `/refresh` under the new model, users never see a
  "half-migrated" chart — every displayed trade is already new-model. No
  progressive rollout needed.
- **KPI calculations keying off `entry_time`.** Any drawdown / equity /
  per-trade-P&L calculation that reads `entry_time` must be updated to key
  on `entry_fill_ts`. Audit every `entry_time`/`exit_time` reference during
  Phase 3.
- **Bar-row click handlers.** UI lookup logic for "which trade is on this
  bar" needs updating to match `fill_ts`. Audit during Phase 4.
- **Scenario replay timing.** Currently uses bar indices; after this
  change, replay steps through `trigger_ts` → `fill_ts` with accurate time
  deltas. Users with muscle memory may notice the extra intermediate step.
- **Webhook payload shape change.** Clean slate per locked decision #5: new
  payloads use `entry_fill_ts` / `entry_trigger_ts` with no legacy aliases.
  Any external integrations consuming the old payload shape need to update
  — coordinate with user if downstream systems exist.
- **Migration is irreversible without snapshot restore.** The `stored_trades
  = []` wipe + `alerts` truncation in Phase 2 cannot be undone by re-running
  a script. Take a full Supabase snapshot immediately before Phase 2 and
  retain it through verification (Phase 6).

### Settled decisions

All open questions from the initial draft have been locked — see the
**Design decisions** section at the top of this document. Summary:

1. Column naming: `entry_*_ts` / `exit_*_ts` with `_ts` suffix
2. Exec type schema: flexible manifest, not enum-constrained
3. `alerts.timestamp`: kept as audit field
4. Webhook payload: clean slate, no legacy aliases
5. LC / CC default: Behavior B (matches current implementation)
6. Timestamp count: 2 per side, not 3

---

## Part 8 — Relationship to other open work

- **`webhook_deliveries` schema drift** (identified 2026-04-17): same family of
  issue as `bar_time` missing as a top-level column. This spec's Phase 2
  schema migration can also add `webhook_deliveries` at the same time,
  resolving both. Alternatively, resolved in Phase 39 webhook redesign —
  coordinate.
- **Phase 39 Webhook Event System** (design approved 2026-03-24): the 11-event
  model aligns with this spec's event lifecycle. Any naming here should match
  Phase 39 naming for consistency. Cross-reference
  `docs/Webhook_Event_System.md`.
- **Phase 31F Hi-Fi Backtest** (planning): Hi-Fi's per-second fidelity naturally
  produces precise `trigger_ts` for L-type triggers. This spec and Phase 31F
  should ship in compatible order — Phase 31F's precise timestamps feed
  directly into this spec's `trigger_ts` field.
- **Alert Recovery Plan 2026-04-17**: the fixes shipped today (atomic
  stored_trades write, conditional subscription, executor dispatch, forming
  throttle, SESSIONS fix) are unrelated to this spec and complete
  independently. This spec builds on top of the now-working alert path.
- **Confluence Pack Builder audit — OUT OF SCOPE for this spec.** Confluence
  packs (indicator-condition packs in `confluence_groups.py` and
  `user_packs/`) produce booleans and don't need the trigger/fill model.
  They're orthogonal and stay untouched. A future milestone can separately
  audit and migrate them to the standardized AI-assisted flow — at that time
  a snapshot export (`docs/legacy_packs_snapshot_YYYY-MM-DD.json`) captures
  the current state before any deletion. This spec only updates the
  **execution type** Pack Builder (Part 5), not the confluence pack builder.
- **User packs in `user_packs/`** (`rsi_zones_2`, `stochastic_oscillator`,
  `swing_123_test`): currently referenced by tests but not fully integrated.
  Left as-is for this spec. Their audit is also future work alongside the
  confluence pack migration.

---

## Part 9 — Acceptance criteria

Spec is "done" when, after implementation:

1. Alert history table and algo history table display the **same timestamp**
   for every trade on every exec type, on every timeframe. Both anchor on
   `fill_ts`.
2. Chart entry/exit markers align with both tables' displayed timestamps.
   Trigger dots optional behind a user toggle.
3. For a 15Min C-type strategy, opening the strategy detail page and
   inspecting one trade shows the same 15-minute bar anchor everywhere;
   `trigger_ts` available via tooltip or expanded row.
4. For an LC **Behavior B** strategy (current default): fill fires
   immediately on cross, validation fails → `alerts` has one `fill` row on
   entry and one `validation_failed` exit row; `stored_trades` has one
   record with `exit_reason = 'validation_failed'`.
5. For a future Behavior A exec type with 5s hold: trigger fires but hold
   fails → `alerts` has one `trigger` row and one `cancel` row;
   `stored_trades` has **no** entry.
6. Parity tests lock all four timestamps (`entry_trigger_ts`,
   `entry_fill_ts`, `exit_trigger_ts`, `exit_fill_ts`) for every exec type
   × behavior combination currently shipped.
7. Execution Type Pack page shows lifecycle diagram + edit flow for each
   shipped exec type. "+ Create Execution Type" AI wizard creates a new
   user-defined exec type that appears in the Strategy Builder's exec type
   dropdown and behaves correctly end-to-end.
8. Soft clean-slate migration complete: all existing strategies have
   `stored_trades = []`, old `alerts` rows truncated, strategy configs
   preserved. Clicking `Update All Data` on any strategy regenerates
   `stored_trades` under the new timestamp model without error.

---

## Part 10 — Unified Trade Reconciliation

### Why

The equity curve and any "trade view" surface is currently rendering two
parallel series (forward-test / pink and alerts / green) that can legitimately
diverge in several ways:

- **Worker downtime** — algo history shows the trade (because `/refresh` runs
  the unified engine on historical bars and finds it), but no alert fired in
  real time because the worker wasn't up.
- **Phantom alert** — an alert fired live but a subsequent backtest recompute
  doesn't reproduce it (real live-vs-backtest divergence).
- **Matched** — both present, aligned to the same fill moment.

Today's alert/algo matching uses a "slippage tolerance" setting against
wall-clock `alerts.timestamp` vs `stored_trades.entry_time`. The slippage
tolerance has to accommodate both structural shift (bar duration) AND noise
(save latency), which yields either false positives on high-frequency
strategies or false negatives on slow saves.

Once the spec's universal `fill_ts` anchor lands (Parts 1-6), the join reduces
to "match by fill_ts within noise-only tolerance."

### Terminology

Adopting **"fill alert"** as the universal label for an alert anchored on
`fill_ts`. Unambiguous, matches the two-timestamp model, reads naturally
("strategy fired 14 fill alerts this week"). Legacy "alert" stays as an
umbrella term for any row in the `alerts` table (trigger / fill / cancel /
exit events).

### Unified Trade row states

Each displayed trade is categorized by its algo-vs-alert reconciliation
status:

| State | Meaning | Cause |
|---|---|---|
| `matched` | Algo trade + fill alert present, `fill_ts` agree within tolerance | Normal case |
| `algo_only` | Algo trade present, no matching fill alert | Worker down, or trade fell outside monitored symbols/session, or alert dispatch failed |
| `alert_only` | Fill alert present, no matching algo trade | Backtest recompute doesn't reproduce the signal — real divergence worth investigating |
| `partial_match` | Both present but `fill_ts` drifted beyond tolerance | Clock skew, unusual save latency, or a real timing bug |

### Display-time join (minimum implementation)

At render time (strategy detail, dashboard, mass-results), join the two
series:

```text
for each algo_trade in stored_trades:
    candidate = first fill_alert where
        |algo_trade.entry_fill_ts - alert.entry_fill_ts| <= slippage_tolerance
    if candidate:
        state = 'matched'; pair them; mark alert consumed
    else:
        state = 'algo_only'

for each unconsumed fill_alert:
    state = 'alert_only'
```

Slippage tolerance is the existing user setting (currently in display
settings). Default value is appropriate for the new millisecond-range noise
profile — likely 1-2 seconds.

### Unified Trades table UI

A new "Unified Trades" view (or extended "Trades" view) renders:

- One row per unique trade (by `entry_fill_ts` grouping)
- Columns: `entry_fill_ts`, `exit_fill_ts`, `state` (badge), `algo R`, `alert
  R`, `delta`, `fill price`, `algo fill price`, `alert fill price`
- Color coding: matched = neutral, algo_only = yellow, alert_only = orange,
  partial_match = yellow with warning icon
- Click-through: both modal views (the algo record + the alert record with
  its event lifecycle)

### Optional: reconciliation cache table

For performance at scale, materialize the joins nightly or on
`/strategies/{id}/refresh`:

```sql
CREATE TABLE trade_reconciliations (
    id BIGSERIAL PRIMARY KEY,
    strategy_id BIGINT REFERENCES strategies(id),
    user_id UUID REFERENCES auth.users(id),
    entry_fill_ts TIMESTAMPTZ,
    exit_fill_ts TIMESTAMPTZ,
    state TEXT,  -- matched | algo_only | alert_only | partial_match
    algo_trade_ref JSONB,  -- full trade record snapshot (or key into stored_trades)
    alert_ids BIGINT[],    -- fill + exit fill alert rows
    slippage_ms INT,       -- fill_ts delta observed
    computed_at TIMESTAMPTZ DEFAULT NOW(),
    INDEX (strategy_id, entry_fill_ts)
);
```

Populated by the existing `forward_test_service.recompute_and_persist_stored_trades`
call — already runs on `/refresh`, already touches both tables' truth, natural
place to emit reconciliation rows.

Benefit: unified view renders from this one table instead of re-joining
thousands of algo entries against tens of thousands of alert rows on every
page load. Enables indexes on `state` for "show me all the `alert_only` rows
across all strategies" queries — useful for divergence monitoring.

### Divergence monitoring (the original intent of `b83629b`)

The `alert_only` and `partial_match` states are what `b83629b` was trying to
surface when it split `stored_trades` from the alert dispatch. This spec
recovers that intent cleanly: the split isn't in WRITE ordering (which
caused the latency/lockstep problems), it's in READ reconciliation. You get
the observability benefit without breaking atomic writes.

---

## Part 11 — Storage Scaling Notes

### Current shape

`stored_trades` is a JSONB list column on the `strategies` table. One
strategy, one potentially very large array. Live probe (2026-04-17) shows
strategy 117 has **1,561 trade entries** already, roughly 300 KB of JSONB in
a single column.

Every `_persist_algo_trade` call:

1. Fetches the full strategy row (pulls the whole 300 KB array)
2. Appends one record
3. Writes the full array back (300 KB + ~1 KB)

At the scale you're aiming for ("thousands of strategies, forward-testing
perpetually"), this becomes:

- 2 MB blob per strategy by 10,000 trades
- 2 GB of mostly-cold data loaded and rewritten on every live exit alert
- `update_strategy_admin` round-trip time grows with blob size

### The refactor

Normalize `stored_trades` into its own table:

```sql
CREATE TABLE strategy_trades (
    id BIGSERIAL PRIMARY KEY,
    strategy_id BIGINT REFERENCES strategies(id),
    user_id UUID,
    entry_trigger_ts TIMESTAMPTZ,
    entry_fill_ts TIMESTAMPTZ NOT NULL,
    exit_trigger_ts TIMESTAMPTZ,
    exit_fill_ts TIMESTAMPTZ,
    hold_duration_s INT,
    behavior TEXT,
    entry_price NUMERIC,
    exit_price NUMERIC,
    r_multiple NUMERIC,
    exec_type TEXT,
    stop_exec_type TEXT,
    target_exec_type TEXT,
    entry_trigger TEXT,
    exit_trigger TEXT,
    exit_reason TEXT,
    confluence_records JSONB,  -- still flexible-shape
    bars_held INT,
    hold_time_seconds NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    INDEX (strategy_id, entry_fill_ts),
    INDEX (user_id, entry_fill_ts)  -- for cross-strategy queries
);
```

Benefits:

- Append is O(1) regardless of history size
- Unified queries across strategies (e.g. dashboard aggregates) become cheap
- RLS policies apply naturally at the row level
- Pagination on the history table is trivial (`LIMIT 100`) without frontend
  slicing a giant array
- Joins with `alerts` / `trade_reconciliations` are first-class SQL

### Why NOT do it in this spec

Splitting the storage touches every writer (live alert dispatch,
`forward_test_service.recompute`, `/refresh`, `/api/strategies/{id}` GET),
every reader (backtest results UI, chart endpoint, dashboard, mass-results),
and requires a full history backfill that must preserve trade_record
field-by-field including any custom keys users' packs may have added. That's
a larger migration than Part 1-7.

### Recommended sequence

1. **Ship this spec first** (Parts 1-10, timestamp model + reconciliation).
2. **Then, separately**, plan and execute the storage refactor as its own
   milestone. The reconciliation cache table (Part 10) can serve as a
   migration pattern — prove the normalized-storage approach on
   `trade_reconciliations` first, then extend to `strategy_trades`.
3. Keep `stored_trades` JSONB as a denormalized cache for a transition period
   (populated from `strategy_trades` on read) so the frontend isn't forced to
   migrate in lockstep.

### What to capture now

Within this spec, include `entry_fill_ts` / `entry_trigger_ts` / etc. as
first-class fields in the trade record shape. That way, when `strategy_trades`
is created later, the column mapping is trivial — every field already has a
home and a name. The spec's Part 3 migration makes Part 11's eventual
refactor cheap.

---

## Appendix — Reference live-DB probe (2026-04-17)

Strategy 70 most recent entry alert (id 9531):

```
alerts.timestamp        : 2026-04-17T21:22:00.674749+00:00  (wall-clock)
alerts.data.bar_time    : 2026-04-17T21:21:00+00:00         (firing bar START)
stored_trades entry_time: 2026-04-17T21:21:00+00:00         (= bar START)
stored_trades exec_type : 'C'
```

Under the proposed model:

```
entry_trigger_ts : 2026-04-17T21:22:00+00:00  (bar CLOSE for C-type — when
                                               the trigger condition was
                                               confirmed)
entry_fill_ts    : 2026-04-17T21:22:00+00:00  (next bar open = bar close for
                                               C-type; same instant)
hold_duration_s  : 0                          (C-type has no hold)
behavior         : B                          (not meaningful for C; no
                                               validation period)
exec_type        : C
```

Both Alert History and Algo History would then display `21:22:00` for this
trade, matching what actually happened in the market. On a 15Min C-type, the
fill moment would be the bar close (e.g. `15:30:00`) — both tables display
that moment, regardless of whether the bar label used by the chart is the
bar's start (`15:15`) or end (`15:30`).
