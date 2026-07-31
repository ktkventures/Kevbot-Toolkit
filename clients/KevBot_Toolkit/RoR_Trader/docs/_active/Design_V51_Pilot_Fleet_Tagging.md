# Design — V5.1 pilot fleet: environment tag + write isolation

**Status:** DESIGN ONLY. Nothing built, nothing armed, no migration applied.
**Board:** #163 (V5.1), step 1 of 6. Parent vision #162 / `Design_Environment_Split_Gateway.md` §6 Q5.
**Author:** E·auto (`r1785529539-163`) · **Date:** 2026-07-31
**Next:** M reviews *"can a pilot row ever read as a live row?"* (step 2), then Kevin authorises (step 3).

---

## 0. The one-line answer

**A pilot row is distinguishable from a live row because it is in a table the live fleet
never writes.** The pilot's entire write surface is one new table, `pilot_alerts`. It writes
nothing to `alerts`, `trades`, `strategies`, `monitor_status`, `live_bars`, `bar_cache`, or
any other existing table, and it makes no outbound HTTP call.

Everything below is the evidence for why that is the right shape rather than a
`data_source`-style discriminator on `alerts`.

---

## 1. The finding that drives the design

`Design_Environment_Split_Gateway.md` §6 Q5 asks: *extend `data_source`, or add an orthogonal
column?* The board SOP sharpens it: **follow the `data_source` axis, do not invent a second
one.** That instruction is correct for `trades`. It does not transfer to `alerts`, and the
reason is a fact about the live fleet, not a preference.

### 1.1 The live fleet's decision record is `alerts`, not `trades`

`src/worker.py:505-520`:

```
# Algo-history dual-write removed 2026-05-06 (M8.7). Live alert
# dispatch only writes the alerts table now; the trades table is
# populated by the algo-history cron in WorkerManager ...
# Clean separation: alerts = live, sub-second; algo-history =
# backtest, bar-aligned, 15-min lag for late-print stability.
if sig_type == 'exit_signal' and os.environ.get(
        'ALGO_HISTORY_LIVE_DUAL_WRITE_ENABLED', ...):
    self._persist_algo_trade(strategy, signal_data)
```

The live worker's `trades` write is **retained for hot-revert only** and gated OFF by
default. So in production:

| Lane | Table | Written by | Timestamp semantics |
|---|---|---|---|
| **live (money)** | `alerts` | `worker.py` `DBAlertDispatcher` | sub-second, at the fire moment |
| algo | `trades` `data_source='cache_<algo_model>'` | algo-history **cron** → `forward_test_service.append_new_trades_for_strategy` | bar-aligned, ~15-min lag |
| backtest | `trades` `data_source='backtest_<model>'` | data-worker / shadow-worker | bar-aligned, settled |

This matches `feedback_algo_vs_alert_history` and `feedback_health_is_backtest_vs_alerts`
("Health = backtest↔ALERTS — NOT algo lane").

**Consequence for the pilot.** The pilot fleet is a *live engine*. Its output is
alert-shaped: sub-second, per-decision, carrying `trigger_id` / `exec_type` / `side` /
`trigger_ts` / `fill_ts` / `price`. If we recorded it as `trades` rows we would be comparing
*pilot trades* against *live alerts* — two different record types with different timestamp
semantics produced by different code paths. That is the exact confound the 2026-05-06
dual-write removal existed to kill (near-duplicate rows differing by seconds). Re-importing
it into the one experiment whose job is to isolate a variable would make the verdict
uninterpretable before the ~3.5 s lag ever got a chance to.

**So the pilot's record must be alert-shaped.** The tagging question is therefore a question
about `alerts`, not about `trades`.

### 1.2 `alerts` has no lane axis, and adding one inverts the safety default

| | `trades` | `alerts` |
|---|---|---|
| lane column | `data_source TEXT` | **none** |
| how readers scope | `LIKE 'backtest_%'` / `LIKE 'cache_%'` — a filter is passed at ~30 call sites | **no reader filters at all** — every one of ~17 read sites reads unscoped |
| dedupe index | `trades_dedupe_idx` **includes** `COALESCE(data_source,'')` (`phase41_backtest_trades_relax_unique.sql`) | none |
| effect of a NEW lane value | **invisible** to every existing reader — `LIKE 'cache_%'` does not match `pilot_%` | **visible** to every existing reader by default |

That last row is the whole argument. The precedent's load-bearing property is not "the column
is called `data_source`" — it is **a new lane is invisible to existing readers unless they opt
in.** On `trades`, a new prefix delivers that property for free. On `alerts`, a discriminator
column delivers the *opposite*: pilot rows land in every existing query on day one, and
correctness becomes contingent on ~17 separate edits all landing and staying landed. That is
precisely the failure `feedback_lane_filter_symmetry` was written about — an unscoped read
silently mixes lanes — except here *every* read is the unscoped one.

A separate table delivers the precedent's actual property with **zero** edits to existing
readers, and makes the money lane unambiguous structurally rather than by discipline.

> **Same principle, different mechanism, because the two tables have opposite defaults.**
> This is the one design decision worth pushing on in step 2. If M disagrees and wants
> `alerts.fleet_tag`, the counter-cost is the ~17-site edit list in §4.1 plus a backfill of
> ~512k existing rows, and every future alert reader inheriting an obligation to remember.

---

## 2. The design

### 2.1 New table: `pilot_alerts`

Mirrors the `alerts` column set (so the existing `_alert_to_row` / `_row_to_alert` split and
the Strategy-Health pairing algorithm can be reused verbatim) plus fleet provenance. **Not
applied — migration text is illustrative, step 3 authorises, an attended E applies.**

```sql
CREATE TABLE IF NOT EXISTS pilot_alerts (
  id              BIGSERIAL PRIMARY KEY,
  -- fleet provenance (the tag)
  fleet_tag       TEXT        NOT NULL,   -- 'pilot_ws_agg_db' for V5.1
  code_fingerprint TEXT,                  -- sha256[:12] of engine sources, as shadow_worker.py does
  flag_set        JSONB       NOT NULL DEFAULT '{}'::jsonb,  -- the RORT_* delta vs the live fleet
  -- the asymmetry, MEASURED per decision rather than assumed (see §5)
  bar_source      TEXT        NOT NULL DEFAULT 'live_bars_ws_agg',
  bar_written_at  TIMESTAMPTZ,            -- live_bars.written_at of the deciding bar
  bar_read_at     TIMESTAMPTZ,            -- when the pilot pulled it
  -- alert shape, mirroring ALERT_COLUMN_FIELDS (db.py:331)
  user_id         UUID        NOT NULL,
  type            TEXT        NOT NULL,
  strategy_id     INTEGER     NOT NULL,
  strategy_name   TEXT,
  symbol          TEXT        NOT NULL,
  direction       TEXT,
  timeframe       TEXT,
  event_type      TEXT,  side TEXT,
  trigger_ts      TIMESTAMPTZ, fill_ts TIMESTAMPTZ,
  exec_type       TEXT,  trigger_id TEXT,
  price           DOUBLE PRECISION, actual_price DOUBLE PRECISION,
  bar_time        TIMESTAMPTZ, hold_duration_s DOUBLE PRECISION, behavior TEXT,
  live_model      TEXT,
  data            JSONB       NOT NULL DEFAULT '{}'::jsonb,
  timestamp       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS pilot_alerts_sid_ts_idx
  ON pilot_alerts (strategy_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS pilot_alerts_fleet_idx
  ON pilot_alerts (fleet_tag, timestamp DESC);

-- RLS: disabled. Admin-client writes, admin-client reads, no user surface.
-- Same posture as live_bars (global, non-user-scoped operational data).
```

Deliberately **absent**: `acknowledged`, `webhook_sent`, `webhook_deliveries`,
`verification_*`. Those columns only mean something for rows on the money path. Omitting them
means a pilot row cannot even be *shaped* like a delivered live alert.

### 2.2 The tag values

| Field | V5.1 value | Purpose |
|---|---|---|
| `fleet_tag` | `pilot_ws_agg_db` | names the fleet **and** its feed. Follows the existing `<lane>_<model>` shape (`backtest_rest_hifi`, `cache_cache_locked`) so it reads as a sibling, not a new vocabulary |
| `code_fingerprint` | `sha256(engine sources)[:12]` | the 2026-07-07 deploy-provenance lesson (`shadow_worker.py:52-70`): "which code is this container running" must be answerable from the row, not from Railway metadata |
| `flag_set` | `{"RORT_X": "1", ...}` | the *delta* vs the live fleet. Without it a divergence is unattributable to the flag that caused it, and the pilot answers nothing |

### 2.3 Optional add-on, separately approvable — `trades` under `pilot_%`

If R-multiple / KPI comparison is wanted on top of decision pairing, the pilot may **also**
write closed trades as `trades` rows with `data_source = 'pilot_ws_agg_db'`. This is safe by
prefix (§1.2) and the dedupe index already tolerates a same-timestamp row in a different lane.

**Recommendation: do not include it in V5.1.** The verdict question is *"do the two fleets
take the same trades?"*, which decision-pairing answers on its own using the existing
Strategy-Health algorithm. The add-on doubles the review surface and introduces the four
unscoped-read hazards in §4.2 for a nice-to-have. Ship it in 5.2 if the verdict earns it.

---

## 3. Write-path inventory — every path the pilot process touches

The pilot runs the same engine code as the live fleet, so **every write below is one the
pilot's process is capable of making by default.** Each must be positively disabled. "Pilot
row vs live row" is answered in the right-hand column.

| # | Table / effect | Writer | Pilot | Distinguisher / why |
|---|---|---|---|---|
| 1 | `pilot_alerts` INSERT | *new* pilot dispatcher | ✅ **the only write** | separate table |
| 2 | `alerts` INSERT | `worker.py:448` `save_alert_admin` | ⛔ OFF | no lane column exists — see §4.1 |
| 3 | **webhook HTTP POST** | `alerts.py:1104` `send_webhook` via `_deliver_alert_fn` (`worker.py:551`) | ⛔ OFF | **the only irreversible, external effect in the system.** No DB cleanup un-sends it. Must be killed at the dispatcher — a config-level "no active webhooks" is one hot-reload away from becoming untrue (`feedback_alert_config_hot_reload`) |
| 4 | `alerts` UPDATE | `ralph_engine.py:5457` (`would_fire_post_correction`); `rest_verifier.py:612` (`verification_*`) | ⛔ OFF | these **mutate existing live rows**. There is no lane to scope them to. Keep `REST_VERIFY_ENABLED` unset on the pilot |
| 5 | `trades` DELETE | `worker.py:603` `delete_trade_placeholder_admin(sid, entry_fill_ts)` — **called unscoped** | ⛔ OFF | 🚨 **the most dangerous line in the inventory.** A pilot running stock worker code **deletes the live fleet's open-position placeholder rows**. The function grew a `data_source_like` param for exactly this family (#101, `db.py:1305-1309`) and this call site does not pass it |
| 6 | `trades` INSERT | `worker.py:522` `_persist_algo_trade` → `trades_store.insert_trade` | ⛔ OFF (see §2.3) | gated by `ALGO_HISTORY_LIVE_DUAL_WRITE_ENABLED`, default OFF. **Leave unset.** ⚠️ verification needed — see §7 |
| 7 | `strategies` UPDATE | `worker.py:649` `update_strategy_admin({'stored_trades':…})`; `worker.py:810` `append_position_carryovers_admin` → `config.position_carryovers` | ⛔ OFF | one row per strategy, **no lane axis at all**. A pilot carryover write lands in the live strategy's config and the live fleet reads it back on hot-reload. Also a JSONB partial-update hazard (`feedback_jsonb_partial_updates`) |
| 8 | `monitor_status` UPSERT | `worker.py:938/1462/1481/1857` → `save_monitor_status_admin`, `on_conflict='user_id'` | ⛔ OFF | 🚨 **one row per user.** A pilot under the same `user_id` overwrites the live fleet's heartbeat, and the monitor/Health UI would then be reporting the pilot. Worse: the row carries `desired_state`, a **control channel** — a pilot write could stop or start the live fleet |
| 9 | `live_bars` UPSERT | `live_bars_writer.write_bar`, gated `LIVE_BAR_CACHE_WRITE_ENABLED` | ⛔ OFF (leave unset) | the pilot **reads** this stream. Writing back would upsert over the forensic WS record and create a feedback loop where the pilot's own bars become its input |
| 10 | `bar_engine_states` UPSERT | `bar_engine_state_writer.py:108`, gated `BAR_ENGINE_STATE_WRITE_ENABLED` | ⛔ OFF (leave unset) | keyed on engine identity, no fleet axis |
| 11 | `engine_audit_log` INSERT | `db.py:1656` `DBAuditor` | ⛔ OFF | no fleet column; would mix two engines' audit trails |
| 12 | `bar_diagnostics` UPSERT | `db.py:2565`, `on_conflict='strategy_id,bar_ts,source'` | ⛔ OFF | worse than mixing — the conflict key has no fleet component, so a pilot write **overwrites the live fleet's diagnostic row** for the same (strategy, bar, source) |
| 13 | `bar_cache` / `resampled_bar_cache` | maintenance loop, gated `BAR_CACHE_MAINTAIN_ENABLED` | ⛔ OFF (leave unset) | two writers racing on a shared cache the backtest lane depends on |
| 14 | `trades` (backtest lane), `shadow_heartbeats` | `shadow_manager` when `RORT_BACKTEST_LANE_MODE=shadow` | ⛔ leave `button` | pilot must not drive the backtest lane |
| 15 | broker orders | **none exists.** `grep -rn 'submit_order\|place_order\|TradingClient\|MarketOrderRequest' src/` returns nothing outside scratch `_*.py` scripts; Alpaca keys are env-passed for *data* only (`worker.py:764-766`) | ⛔ n/a | the only outbound money instruction is the webhook (row 3). Killing row 3 kills the money path entirely |

### 3.1 The rail that enforces it: positive assertion, fail-loud

Disabling 14 things by remembering to disable them is not a rail. Two mechanisms, both
default-safe:

1. **`RORT_PILOT_FLEET_TAG`** — the pilot dispatcher writes only when this is set to a
   non-empty value. The live services never set it, so **the same image deployed to the live
   worker cannot write pilot rows**, and a pilot process that lost its env writes nothing at
   all rather than writing live rows.
2. **Boot assertion, refuse to start** (`feedback_no_silent_defaults` — fail loud, never
   silently default). On boot the pilot asserts every entry in §3 is off and **exits non-zero**
   otherwise:
   `ALGO_HISTORY_LIVE_DUAL_WRITE_ENABLED`, `LIVE_BAR_CACHE_WRITE_ENABLED`,
   `BAR_ENGINE_STATE_WRITE_ENABLED`, `BAR_CACHE_MAINTAIN_ENABLED`,
   `ALGO_HISTORY_CRON_ENABLED`, `REST_VERIFY_ENABLED` all unset/false;
   `RORT_BACKTEST_LANE_MODE == 'button'`; `_deliver_alert_fn is None`;
   `RORT_PILOT_FLEET_TAG` non-empty.
   A pilot that cannot prove it is isolated must not run — refusing to boot is a cheap,
   loud, self-diagnosing failure, and it is the same posture `shadow_worker.py` takes.

---

## 4. Queries that break or mislead if the tag is wrong

### 4.1 If a pilot row ever landed in `alerts` (the case the design exists to prevent)

Every one of these reads `alerts` **unscoped**. None would need to change under the §2 design;
all of them break under a discriminator column until individually edited.

| Site | What it drives | Failure |
|---|---|---|
| `strategy_health.py:377` recent-alerts window | 🚨 **paired-% / phantom / missed** | pilot alerts near a live alert **pair against backtest edges and inflate paired-%**; pilot alerts with no edge inflate `phantom`. This is the project's headline metric *and* the `#264` pre-split baseline ruler — corrupting it mid-measurement destroys the BEFORE reading the whole V5 split is measured against |
| `strategy_health.py:290` last-alert-per-sid (30 d) | "is this strategy alive" | a strategy fed only by the pilot reads as live |
| `strategy_health.py:981 / 1020 / 1626 / 1705 / 2095 / 2175` | drill-down + fleet-wide health variants | same contamination, per-strategy |
| `health_last10.py:127` | Last-10 alerts panel | shows pilot decisions as live |
| `pack_canary_service.py:243/247/250` | canary liveness (last / today / week) | 🚨 **canary-267 is the parity gate** (`feedback_fidelity_parity_gate`). A polluted canary makes the gate unreadable |
| `portfolios.py:1471 / 1605 / 1759-1763` | Trade History `raw_alerts` / `live_executions` / `matched` | a pilot decision renders as a **live execution in a portfolio** |
| `forward_test_service.py:1388` | alerts-existence check gating a lane decision | wrong branch taken |
| `strategies.py:346`, `ai_builder.py:327`, `db.py:742/764` | Strategy Detail history, AI builder, user Alerts page | pilot rows surface to the user as their own alerts |
| `rest_verifier.py:836` sweeper | picks up unverified alerts | would **queue and then UPDATE pilot rows**, and burn REST budget doing it |

### 4.2 If the §2.3 `trades` add-on is approved

**Safe by prefix** (no change needed): every caller passing `'backtest_%'` or `'cache_%'` —
`forward_test_service`, `data_worker_engine`, `shadow_manager`, `admin_parity.py:86-88`,
`pack_canary_service.py:303-305`, `strategies.py:2097/2130/2280`, `db.py:315`.

**Would silently mix** — these pass no filter:

| Site | Failure |
|---|---|
| `parity_service.py:748` `load_trades_admin(sid, uid)` | 🚨 the **parity/fidelity gate** would score pilot trades as the strategy's trades |
| `db.load_trades_admin` / `load_trades_kpi_fields_admin` / `trades_store.load_trades_for_strategy` — all default `data_source_filter=None` | any caller that forgets; **KPIs** computed over a mixed lane |
| `db.delete_trades_for_strategy_admin(sid)` | deletes **all** rows for a strategy including pilot rows — harmless to live, but a silent, unlogged loss of the experiment's data |
| `db.replace_trades_admin(..., data_source_filter=None)` legacy default | DELETE-then-insert over **all** lanes |

This asymmetry is the real cost of §2.3 and the reason to defer it.

---

## 5. The limitation, recorded rather than remembered

Task #163 and `Design_Environment_Split_Gateway.md` §3 both state it: `live_bars` `ws_agg`
write-lag ran **~3.5 s median, p90 ~4–6 s** on 07-27. A fleet reading the DB is that far
behind one reading the socket. Fine for a decision-set comparison, fatal for a timing one.

The design's contribution is to stop this being a footnote someone has to remember:
`bar_written_at` and `bar_read_at` are on every `pilot_alerts` row, so **the asymmetry is
measured per decision**, not assumed at a session average. The step-4 verdict can then say
"the fleets' decision sets differ by N, and the observed feed lag on those N decisions was
X±Y" instead of leaving the reader to guess whether N is logic or lag.

Standing constraint, restated for the verdict: **V5.1 answers "same trades?" and must not be
quoted on "same time?".**

---

## 6. Stopping the pilot, and cleanup

### 6.1 Stop — three levers, cheapest first

| Lever | Effect | Touches live? |
|---|---|---|
| Railway **scale the pilot service to zero / `railway down`** | process gone | **no** — separate service. This is why the pilot must be its own Railway service, never a second process inside the worker service |
| Unset `RORT_PILOT_FLEET_TAG` on the pilot | engine keeps running, writes stop | no |
| `WORKER_DISABLED=1` on the pilot (`worker.py:1916`, `feedback_worker_disabled_switch`) | existing kill-switch | no |

All three are Railway var-set / deploy actions → **attended E or Kevin, never headless.**
Whoever runs step 4 must have one of these levers in hand before the session opens, not be
looking for it mid-session.

### 6.2 Abort criteria (the pilot shares prod Supabase with the money path)

The pilot is a second consumer of the same connection pool and the same hot tables — the
class of risk behind `project_worker_minute_boundary_saturation` and
`project_incident_prime1s_dos_2026-07-02`. Abort on any of:

- live-fleet `live_bars` `ws_agg` `written_at`-lag median **> 6 s** during RTH (the #118
  tripwire — the existing, already-agreed measure);
- Supabase pool exhaustion / statement timeouts on the live worker;
- any `pilot_alerts` write failure rate that would make the session's comparison partial
  (a partial comparison read as complete is worse than no comparison).

Also: the pilot's `live_bars` read must be a bounded poll, not a tight loop.

### 6.3 Cleanup

- **Under the §2 design: none.** No live table is touched, so there is nothing to unwind.
  Retiring the experiment is `DELETE FROM pilot_alerts WHERE fleet_tag = 'pilot_ws_agg_db'`
  or `DROP TABLE pilot_alerts` — and either is optional, because nothing reads it.
- **If §2.3 is approved:** `DELETE FROM trades WHERE data_source LIKE 'pilot_%'`, and it must
  run **before** any Update-All-Data, KPI recompute or parity run that uses one of the
  unscoped reads in §4.2. That ordering dependency is the third reason to defer §2.3.
- A mid-session crash leaves partial `pilot_alerts` rows. They are inert. The verdict must
  scope its read to a window both fleets covered, or it is comparing coverage, not logic.

---

## 7. Facts this design needs that only a probe can give — asking, not probing

Per the step rails, not probed:

1. **Is `ALGO_HISTORY_LIVE_DUAL_WRITE_ENABLED` actually unset on the prod worker service?**
   Row 6 of §3 and the §1.1 lane table both assume it is (that is the code default). If it is
   ON in Railway, the live fleet *is* writing `trades` and §1.1's conclusion changes. Needs a
   `railway variables` read → attended E. **Verify before build, not during step 4.**
2. **Is `USE_TRADES_TABLE` ON in prod?** It gates whether `worker.py:603`'s unscoped
   placeholder DELETE (§3 row 5) is reachable at all. Assumed ON; same read, same lane.

Neither blocks step 2 — both change build details, not the tagging decision.

---

## 8. Chain gap raised to M

The #163 chain runs `design → M review → Kevin authorises → run one full session → …`.
**There is no BUILD step.** Step 4 ("run one full session") presupposes: pilot dispatcher
written, `pilot_alerts` migration applied, a new Railway service created and deployed, a
`live_bars` feed adapter into the engine's bar path. The migration and the deploy are
attended-only, so step 4 as written cannot be dispatched headless.

Suggested: insert **(E) build the pilot fleet behind `RORT_PILOT_FLEET_TAG`, branch only** after
step 2, and **(kevin/E-attended) apply migration + deploy the pilot service** after step 3.

Also for step 3's stamp: this design changes what is being authorised. Not *"a second fleet
that writes to prod tables"* but *"a second fleet that writes to one new, pilot-only table in
the prod DB, and to nothing else."* Worth Kevin seeing the narrower version, because it is a
materially smaller thing to say yes to.
