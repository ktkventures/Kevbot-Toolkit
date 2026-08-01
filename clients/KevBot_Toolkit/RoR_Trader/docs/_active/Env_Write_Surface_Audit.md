# Env Write Surface Audit — which routes write, and which read across users

**Board #285** · E·auto, 2026-07-31 · read-only audit, no code changed
**Input to:** #165 Phase A step 9 — which services and routes may run in the dev environment at all

---

## 1. Why this exists

Under #165 option A, dev and production **share one Supabase**. The protection against
cross-contamination is **user scoping**: dev logs in as a different account, so it sees and
writes only its own rows.

**That protection holds exactly as far as the code is actually user-scoped, and no further.**
This document establishes where that line falls, route by route.

Kevin's own evidence that the line is already crossed somewhere:
> *"the strategy health page currently shows strategies that are not part of my current account."*

**That symptom is explained in §4.1.** It is not a display bug — it is the intended behaviour of
an admin-client query with no user filter, and it is a whole class, not one route.

---

## 2. Method

Every route in `src/api/routers/` (37 modules, **249 routes**) was enumerated by parsing the
route decorators, then each was classified by reading the handler and following its writes down
into `src/db.py` and the module layer. Counts below are from that parse, not an estimate.

Confirming M's prior verification (re-checked, not re-derived):

| surface | scoping | confirmed |
|---|---|---|
| `/strategies` list | `get_current_user` → `load_strategies_db*` → `get_current_user_id()` | ✅ |
| packs (confluence / general / risk) | `get_current_user` → per-user pack rows | ✅ |
| strategy health **badge** | rides on the user-scoped strategies list | ✅ |

Note the badge is scoped; the **Strategy Health page** (§4.1) is a different surface and is not.

---

## 3. The three boundary mechanisms — this is the whole finding in one table

Every route relies on exactly one of these. Which one it relies on is the audit.

| # | mechanism | how it scopes | survives `DEV_BYPASS_AUTH`? |
|---|---|---|---|
| **A** | `get_client()` + user JWT | Postgres **RLS** enforces `user_id = auth.uid()` server-side | ❌ **no** — bypass switches to service-role |
| **B** | `get_admin_client()` + an explicit `.eq("user_id", …)` in the handler | code-level filter | ✅ yes — the filter is in Python |
| **C** | `get_admin_client()` and **no user filter at all** | **nothing scopes it** | n/a — already unscoped |

**Mechanism C is the dangerous class, and it is dangerous *today*, not only under a shared DB.**
Every C route is reachable by any authenticated account. The tables it touches
(`dev_tasks`, `replay_sim_*`, `bar_cache_config`, `system_settings`, `agents`, `strategy_notes`,
`task_mentions`, `m_session_*`, `run_history`) mostly have **RLS enabled with zero policies** —
which means they are reachable *only* through the service-role client, so RLS was never the
boundary for them. Verified in `src/migrations/`:

```
dev_tasks            RLS enabled, 0 policies    ("service role bypasses it" — dev_tasks_table.sql:4)
replay_sim_requests  RLS enabled, 0 policies
replay_sim_optin     RLS enabled, 0 policies
system_settings      RLS enabled, read-only policy; writes are service-role
bar_cache_config     no RLS migration at all
strategy_notes       no RLS migration at all
strategies/alerts/trades/mass_searches/update_jobs   RLS enabled WITH policies  ← the scoped ones
```

---

## 4. Totals

| verdict | count | meaning |
|---|---:|---|
| ✅ user-scoped | 154 | safe under a shared DB |
| ✅ user-scoped *(via RLS only)* | 13 | safe **unless** `DEV_BYPASS_AUTH=true` — mechanism A |
| 🟡 fleet-wide read | 39 | dev sees production data; usually fine, sometimes confusing |
| 🔴 **fleet-wide WRITE** | **29** | **a dev action mutates production data regardless of account** |
| ⚪ no DB | 13 | compute / static registry / external auth |
| ⚠️ unauthenticated | 1 | `GET /api/strategies/models` — no `Depends()`; static registry, touches no DB |
| **total** | **249** | |

### 4.1 The reported symptom, explained

`GET /api/admin/strategy-health` (`strategy_health.py:250`) does:

```python
c = get_admin_client()
_strat_q = c.table("strategies").select("id,user_id,name,…").order("id")   # ← no user filter
```

It returns `user_id` in each row but **never filters on it**. It is mechanism C: every strategy in
the database, for every account. The same shape appears in `model_parity.py:133`,
`health_last10.py:246`, and `data_health.py:109`. This is a **fleet-wide read**, so under a shared
DB it is confusing rather than destructive — but it is the class Kevin spotted, and it confirms
the audit's premise.

---

## 5. 🔴 FLEET-WIDE WRITE — the dangerous class

**This section is NOT empty. There are 29 such routes, in 10 routers.**
Each row: what it touches, what triggers it, and whether it must be blocked in dev.

### 5.1 Highest blast radius — production engine behaviour

| route | table | trigger | blast radius | disable in dev? |
|---|---|---|---|---|
| `PATCH /api/admin/system-settings/{key}` | `system_settings` (upsert on `key`) | Admin → System Settings toggle | **Changes live engine behaviour.** The Data Worker re-reads within ~30s. A dev flip silently reconfigures production. | **YES — hard block** |
| `POST /api/r-session/heartbeat` | `system_settings` (`heartbeat_key(letter)`) | any R/M session tick | A dev session's heartbeat overwrites the production session key; the session dashboards then show a dead prod session as alive (or vice versa). | **YES** |

### 5.2 Shared market-data control plane — `bar_cache_admin` (5 routes)

Table `bar_cache_config` has **no user column and no RLS**. All writes are service-role.

| route | touches | trigger | blast radius | disable in dev? |
|---|---|---|---|---|
| `POST /api/bar-cache/add-ticker` | `bar_cache_config` + enqueues **2× 1-year backfill** on the batch-worker | Supply page "Add ticker" | Enrolls a symbol fleet-wide **and** queues ~5.8M rows of 1Sec backfill onto the shared batch-worker. | **YES** |
| `POST /api/bar-cache/targets` | `bar_cache_config` upsert | Supply page CRUD | Changes what the fleet captures. | **YES** |
| `DELETE /api/bar-cache/targets/{symbol}/{timeframe}` | `bar_cache_config` delete | Supply page | **Stops production capture for a symbol.** Bars already cached survive; new ones stop. | **YES** |
| `POST /api/bar-cache/backfill` | `bar_cache_config` + `compute_jobs` + writes `bar_cache` rows | Supply page "Backfill" | Writes production bar data and saturates the shared batch-worker. | **YES** |
| `POST /api/bar-cache/maintain` | `bar_cache` via `maintain_all_enabled()` | Supply page "Maintain" | Runs the keep-updated loop over **all** enabled targets. | **YES** |

### 5.3 Cross-user job sweeps — the two `cleanup-orphans` routes

Both use the admin client and sweep **every** user's rows, not the caller's.

| route | table | trigger | blast radius | disable in dev? |
|---|---|---|---|---|
| `POST /api/mass-builder/cleanup-orphans` | `mass_searches` — flips **all** `status='running'` → `orphaned` | button; also runs **on API boot** | A dev API boot marks production's genuinely-running mass searches as dead. | **YES — and see the boot path below** |
| `POST /api/update-jobs/cleanup-orphans` | `update_jobs` — same sweep | button; also runs **on API boot** | Same, for UAD jobs. | **YES** |

⚠️ **The boot path matters more than the button.** `cleanup_orphaned_update_jobs()` is documented
as *"Called once on api boot"*. Under option A, **starting the dev API is itself a fleet-wide
write** — no one has to click anything. This is the single most likely way dev silently corrupts
production state, because it happens on every dev deploy.

### 5.4 Per-strategy actions with no ownership check — `replay_sim` (2 routes)

Both look up `strategies` by id with **no user filter** — the only check is 404-if-absent.

| route | table | trigger | blast radius | disable in dev? |
|---|---|---|---|---|
| `POST /api/strategy-health-sim/{sid}/run` | `replay_sim_requests` insert | SIM button on any strategy | Enqueues a **heavy** replay against any production strategy. `requested_by` defaults to the literal string `"kevin"`, so the audit trail lies. | **YES** |
| `PUT /api/strategy-health-sim/{sid}/optin` | `replay_sim_optin` upsert | opt-in toggle | Arms/disarms **nightly** SIM for any production strategy — a recurring cost Kevin deliberately made selective (07-26). | **YES** |

### 5.5 The team board and session state (17 routes)

`dev_tasks` (9 writes) · `mentions` (2) · `m_session` (4) · `agents` (2) · `strategy_notes` (1).
All admin-client, none has a user column. Any authenticated account can create, edit, comment on,
stamp, tick, and delete **any** board task.

| router | tables | blast radius | disable in dev? |
|---|---|---|---|
| `dev_tasks` | `dev_tasks`, `dev_task_comments`, `run_history` | The board is the multi-session SSOT. A dev write here corrupts live coordination — and `/steps/complete` + `/steps/raise-issue` additionally accept the **service-role key** as bearer (`get_service_or_user`), which the dev environment would also hold. | **Probably NOT** — see below |
| `mentions`, `m_session`, `agents`, `strategy_notes` | `task_mentions`, `m_session_*`, `agents`, `strategy_notes` | Same shared-coordination surface. | Same |

**Recommendation differs here, deliberately.** The board is *meant* to be shared — one board
across all sessions is the design, and a dev environment that cannot reach it cannot be driven by
the dispatcher. The risk is not "dev sees the board", it is **"dev writes to the board believing it
is a dev board"**. Suggested posture: leave reachable, but make the environment visible on every
write (an `env` column or an actor suffix) so a dev-origin board write is identifiable rather than
silently indistinguishable. **Flagged, not fixed — this is its own task.**

### 5.6 Full 🔴 list (all 29, for checking against the table in §7)

```
system_settings.py   PATCH  /api/admin/system-settings/{key}
r_session.py         POST   /api/r-session/heartbeat
bar_cache_admin.py   POST   /api/bar-cache/add-ticker
bar_cache_admin.py   POST   /api/bar-cache/targets
bar_cache_admin.py   DELETE /api/bar-cache/targets/{symbol}/{timeframe}
bar_cache_admin.py   POST   /api/bar-cache/backfill
bar_cache_admin.py   POST   /api/bar-cache/maintain
mass_builder.py      POST   /api/mass-builder/cleanup-orphans
update_jobs.py       POST   /api/update-jobs/cleanup-orphans
replay_sim.py        POST   /api/strategy-health-sim/{sid}/run
replay_sim.py        PUT    /api/strategy-health-sim/{sid}/optin
agents.py            POST   /api/agents
agents.py            PATCH  /api/agents/{agent_id}
strategy_notes.py    POST   /api/strategy-notes/{sid}
mentions.py          POST   /api/mentions/seen-all
mentions.py          POST   /api/mentions/{mention_id}/seen
m_session.py         PUT    /api/m-session/canvas
m_session.py         PUT    /api/m-session/order
m_session.py         POST   /api/m-session/marks
m_session.py         DELETE /api/m-session/marks/{event_key:path}
dev_tasks.py         POST   /api/dev-tasks
dev_tasks.py         PATCH  /api/dev-tasks/{task_id}
dev_tasks.py         DELETE /api/dev-tasks/{task_id}
dev_tasks.py         POST   /api/dev-tasks/{task_id}/run-request
dev_tasks.py         POST   /api/dev-tasks/{task_id}/stamp
dev_tasks.py         POST   /api/dev-tasks/{task_id}/steps/complete
dev_tasks.py         POST   /api/dev-tasks/{task_id}/steps/raise-issue
dev_tasks.py         POST   /api/dev-tasks/{task_id}/steps/stamp
dev_tasks.py         POST   /api/dev-tasks/{task_id}/comments
```

---

## 6. `DEV_BYPASS_AUTH` — what setting it in dev would cost

`src/api/deps.py:26-30, 70-73`:

```python
_DEV_BYPASS = os.getenv("DEV_BYPASS_AUTH", "").lower() in ("true", "1", "yes")
_DEV_USER_ID = os.getenv("DEV_USER_ID", "00000000-0000-0000-0000-000000000000")
...
if _DEV_BYPASS:
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    set_current_user(_DEV_USER_ID, service_key)          # ← service-role, RLS bypassed
    return {"id": _DEV_USER_ID, "email": "dev@local"}
```

**Production status.** M measured it **unset in the live environment on 07-31**. This audit did not
independently re-measure it — reading Railway variables is outside headless E scope — so the
07-31 measurement stands as the current record. It is also **not** in any committed env file
(`src/.env.example`, `frontend/.env.local.example` do not define it); the only in-repo `=true`
assignments are inside local diagnostic scripts and tests, not deployment config.

**What it would cost if set in dev.** Three separate losses, and the third is the one people miss:

1. **RLS stops applying.** Mechanism A (13 routes, plus every `get_client()` call under it) loses
   its only boundary. `mass_searches`, `update_jobs`, `strategies`, `trades`, `alerts` become
   fully readable and writable across accounts.
2. **Auth stops applying.** Every route becomes reachable with no token at all — including all 29
   fleet-wide writes in §5. A dev URL left open to the internet is then an unauthenticated
   production write endpoint.
3. **⚠️ `DEV_USER_ID` may already point at production.** Two committed scripts default it to a
   real UUID:
   `src/_divergence_walkthrough.py:42` and `src/_compare_cache_vs_polygon_load.py:32` both use
   `os.getenv("DEV_USER_ID", "19d47e46-f718-49a6-af32-5f5407f5b170")`.
   If the dev environment inherits that default, bypass mode does not merely drop scoping — it
   **authenticates every dev request as that specific production user**, so even the correctly
   user-scoped 154 routes write production rows. **This is its own task: confirm whose UUID that
   is before the dev environment is stood up.**

**Recommendation: `DEV_BYPASS_AUTH` must never be set in the shared-database dev environment.**
Under option A it is not a convenience flag — it is the removal of the only boundary the option
depends on. If dev needs unattended API access, use a real dev-account JWT (the `src/.env` test
account pattern already used for board API auth), not the bypass.

---

## 7. Full inventory — every route in `src/api/routers/`

Legend: ✅ user-scoped · ✅ *(via RLS only)* = safe unless `DEV_BYPASS_AUTH` · 🟡 fleet-wide read ·
🔴 **fleet-wide WRITE** · ⚪ no DB · ⚠️ unauthenticated

#### `admin_parity.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/admin/parity/snapshot` | 40 | ✅ user-scoped | explicit owner check: 403 if strategies.user_id != caller (L75) |
| GET | `/api/admin/parity/observable` | 163 | 🟡 fleet-wide read | auth-only; market/bar data, not user rows |
| GET | `/api/admin/parity/rest-bars` | 285 | 🟡 fleet-wide read | auth-only; market/bar data, not user rows |
| GET | `/api/admin/parity/trade-bars` | 388 | 🟡 fleet-wide read | auth-only; market/bar data, not user rows |

#### `agents.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/agents` | 53 | 🟡 fleet-wide read | whole agent registry |
| POST | `/api/agents` | 64 | 🔴 **fleet-wide WRITE** | INSERT into `agents` — shared registry, no user column |
| PATCH | `/api/agents/{agent_id}` | 77 | 🔴 **fleet-wide WRITE** | UPDATE any `agents` row |

#### `ai_builder.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/packs/builder/user-packs` | 283 | ✅ user-scoped | user packs + user strategies via get_current_user |
| POST | `/api/packs/builder/user-packs/{slug}/canaries` | 394 | ✅ user-scoped | user packs + user strategies via get_current_user |
| GET | `/api/packs/builder/user-packs/{slug}/canaries` | 409 | ✅ user-scoped | user packs + user strategies via get_current_user |
| POST | `/api/packs/builder/user-packs/canaries/recreate-all` | 423 | ✅ user-scoped | user packs + user strategies via get_current_user |
| POST | `/api/packs/builder/user-packs/canaries/bulk-toggle` | 439 | ✅ user-scoped | user packs + user strategies via get_current_user |
| GET | `/api/packs/builder/user-packs/{slug}/code` | 455 | ✅ user-scoped | user packs + user strategies via get_current_user |
| POST | `/api/packs/builder/user-packs/{slug}/parity-test` | 497 | ✅ user-scoped | user packs + user strategies via get_current_user |
| GET | `/api/packs/builder/user-packs/{slug}/parity-status` | 582 | ✅ user-scoped | user packs + user strategies via get_current_user |
| POST | `/api/packs/builder/generate-structure` | 606 | ✅ user-scoped | user packs + user strategies via get_current_user |
| POST | `/api/packs/builder/generate-code` | 646 | ✅ user-scoped | user packs + user strategies via get_current_user |
| POST | `/api/packs/builder/fix` | 698 | ✅ user-scoped | user packs + user strategies via get_current_user |
| POST | `/api/packs/builder/validate` | 749 | ✅ user-scoped | user packs + user strategies via get_current_user |
| POST | `/api/packs/builder/install` | 768 | ✅ user-scoped | user packs + user strategies via get_current_user |
| POST | `/api/packs/builder/user-packs/{slug}/preview` | 996 | ✅ user-scoped | user packs + user strategies via get_current_user |

#### `alerts.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/alerts` | 19 | ✅ user-scoped | db.py alert helpers scope on get_current_user_id() |
| GET | `/api/alerts/strategy/{strategy_id}` | 29 | ✅ user-scoped | db.py alert helpers scope on get_current_user_id() |
| PUT | `/api/alerts/{alert_id}/acknowledge` | 40 | ✅ user-scoped | db.py alert helpers scope on get_current_user_id() |
| POST | `/api/alerts/clear` | 50 | ✅ user-scoped | db.py alert helpers scope on get_current_user_id() |
| POST | `/api/alerts/clear-by-strategies` | 58 | ✅ user-scoped | db.py alert helpers scope on get_current_user_id() |
| GET | `/api/alerts/config` | 123 | ✅ user-scoped | db.py alert helpers scope on get_current_user_id() |
| PUT | `/api/alerts/config` | 130 | ✅ user-scoped | db.py alert helpers scope on get_current_user_id() |
| GET | `/api/alerts/{alert_id}` | 147 | ✅ user-scoped | db.py alert helpers scope on get_current_user_id() |

#### `auth.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| POST | `/api/auth/login` | 17 | ⚪ no DB | Supabase auth endpoints |
| POST | `/api/auth/refresh` | 50 | ⚪ no DB | Supabase auth endpoints |
| GET | `/api/auth/me` | 80 | ⚪ no DB | Supabase auth endpoints |

#### `backtest.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| POST | `/api/backtest/run` | 16 | ⚪ no DB | compute-only; reads the caller's packs, writes nothing |
| POST | `/api/backtest/kpis` | 47 | ⚪ no DB | compute-only; reads the caller's packs, writes nothing |
| POST | `/api/backtest/trade-zoom` | 70 | ⚪ no DB | compute-only; reads the caller's packs, writes nothing |
| POST | `/api/backtest/trade-replay` | 141 | ⚪ no DB | compute-only; reads the caller's packs, writes nothing |
| POST | `/api/backtest/analyze` | 211 | ⚪ no DB | compute-only; reads the caller's packs, writes nothing |

#### `bar_cache_admin.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/bar-cache/targets` | 61 | 🟡 fleet-wide read | coverage over the shared cache |
| POST | `/api/bar-cache/add-ticker` | 145 | 🔴 **fleet-wide WRITE** | global bar-cache control plane; no user column anywhere |
| POST | `/api/bar-cache/verify` | 173 | 🟡 fleet-wide read | Polygon-vs-cache diff; read-only |
| POST | `/api/bar-cache/targets` | 324 | 🔴 **fleet-wide WRITE** | global bar-cache control plane; no user column anywhere |
| DELETE | `/api/bar-cache/targets/{symbol}/{timeframe}` | 345 | 🔴 **fleet-wide WRITE** | global bar-cache control plane; no user column anywhere |
| POST | `/api/bar-cache/backfill` | 353 | 🔴 **fleet-wide WRITE** | global bar-cache control plane; no user column anywhere |
| GET | `/api/bar-cache/backfill/status` | 422 | 🟡 fleet-wide read | in-process status dict |
| POST | `/api/bar-cache/maintain` | 427 | 🔴 **fleet-wide WRITE** | global bar-cache control plane; no user column anywhere |

#### `dashboard.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/dashboard/summary` | 17 | ✅ user-scoped | load_strategies_db_lite/load_portfolios_db → get_current_user_id() |
| GET | `/api/dashboard/health` | 88 | ✅ user-scoped | load_strategies_db_lite/load_portfolios_db → get_current_user_id() |
| GET | `/api/dashboard/positions` | 117 | ✅ user-scoped | load_strategies_db_lite/load_portfolios_db → get_current_user_id() |
| GET | `/api/dashboard/activity` | 148 | ✅ user-scoped | load_strategies_db_lite/load_portfolios_db → get_current_user_id() |
| GET | `/api/dashboard/equity-curve` | 174 | ✅ user-scoped | load_strategies_db_lite/load_portfolios_db → get_current_user_id() |
| GET | `/api/dashboard/daily-pnl` | 242 | ✅ user-scoped | load_strategies_db_lite/load_portfolios_db → get_current_user_id() |
| GET | `/api/dashboard/market-regime` | 309 | ✅ user-scoped | load_strategies_db_lite/load_portfolios_db → get_current_user_id() |

#### `data.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/data/bars/{symbol}` | 16 | 🟡 fleet-wide read | market data (Polygon/cache), no user rows |
| GET | `/api/data/source` | 59 | 🟡 fleet-wide read | market data (Polygon/cache), no user rows |
| GET | `/api/data/timeframes` | 66 | 🟡 fleet-wide read | market data (Polygon/cache), no user rows |

#### `data_health.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/admin/data-health` | 66 | 🟡 fleet-wide read | admin client over live_bars + all strategies |

#### `dev_tasks.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/dev-tasks` | 484 | 🟡 fleet-wide read | whole board |
| POST | `/api/dev-tasks` | 507 | 🔴 **fleet-wide WRITE** | the team board — admin client, RLS enabled with NO policy |
| PATCH | `/api/dev-tasks/{task_id}` | 525 | 🔴 **fleet-wide WRITE** | the team board — admin client, RLS enabled with NO policy |
| DELETE | `/api/dev-tasks/{task_id}` | 630 | 🔴 **fleet-wide WRITE** | the team board — admin client, RLS enabled with NO policy |
| POST | `/api/dev-tasks/{task_id}/run-request` | 636 | 🔴 **fleet-wide WRITE** | the team board — admin client, RLS enabled with NO policy |
| POST | `/api/dev-tasks/{task_id}/stamp` | 726 | 🔴 **fleet-wide WRITE** | the team board — admin client, RLS enabled with NO policy |
| POST | `/api/dev-tasks/{task_id}/steps/complete` | 873 | 🔴 **fleet-wide WRITE** | the team board — admin client, RLS enabled with NO policy |
| POST | `/api/dev-tasks/{task_id}/steps/raise-issue` | 922 | 🔴 **fleet-wide WRITE** | the team board — admin client, RLS enabled with NO policy |
| POST | `/api/dev-tasks/{task_id}/steps/stamp` | 970 | 🔴 **fleet-wide WRITE** | the team board — admin client, RLS enabled with NO policy |
| GET | `/api/dev-tasks/{task_id}/poll` | 1021 | 🟡 fleet-wide read | whole board |
| GET | `/api/dev-tasks/comments/recent` | 1079 | 🟡 fleet-wide read | whole board |
| GET | `/api/dev-tasks/{task_id}/comments` | 1106 | 🟡 fleet-wide read | whole board |
| POST | `/api/dev-tasks/{task_id}/comments` | 1112 | 🔴 **fleet-wide WRITE** | the team board — admin client, RLS enabled with NO policy |

#### `execution_types.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/execution-types` | 10 | ✅ user-scoped | _save_config → save_settings_db (per-user settings row) |
| GET | `/api/execution-types/config` | 32 | ✅ user-scoped | _save_config → save_settings_db (per-user settings row) |
| PUT | `/api/execution-types/config` | 38 | ✅ user-scoped | _save_config → save_settings_db (per-user settings row) |
| GET | `/api/execution-types/{slug}` | 45 | ✅ user-scoped | _save_config → save_settings_db (per-user settings row) |
| PUT | `/api/execution-types/{slug}/toggle` | 60 | ✅ user-scoped | _save_config → save_settings_db (per-user settings row) |
| PUT | `/api/execution-types/{slug}/params` | 72 | ✅ user-scoped | _save_config → save_settings_db (per-user settings row) |
| POST | `/api/execution-types/{slug}/variations` | 85 | ✅ user-scoped | _save_config → save_settings_db (per-user settings row) |
| DELETE | `/api/execution-types/variations/{var_id}` | 119 | ✅ user-scoped | _save_config → save_settings_db (per-user settings row) |
| GET | `/api/execution-types/variations` | 129 | ✅ user-scoped | _save_config → save_settings_db (per-user settings row) |
| GET | `/api/execution-types/{slug}/code` | 139 | ✅ user-scoped | _save_config → save_settings_db (per-user settings row) |
| GET | `/api/execution-types/{slug}/scenarios` | 160 | ✅ user-scoped | _save_config → save_settings_db (per-user settings row) |
| POST | `/api/execution-types/{slug}/generate-scenarios` | 312 | ✅ user-scoped | _save_config → save_settings_db (per-user settings row) |
| POST | `/api/execution-types/{slug}/simulate` | 383 | ✅ user-scoped | _save_config → save_settings_db (per-user settings row) |

#### `health_last10.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/strategy-health-last10` | 232 | 🟡 fleet-wide read | admin client; no sid ownership check |
| GET | `/api/strategy-health-last10/{sid}` | 254 | 🟡 fleet-wide read | admin client; no sid ownership check |

#### `m_session.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/m-session/rules` | 214 | ⚪ no DB | static rules |
| GET | `/api/m-session/state` | 268 | 🟡 fleet-wide read | shared M-session state |
| PUT | `/api/m-session/canvas` | 305 | 🔴 **fleet-wide WRITE** | M-session canvas/order/marks — admin client, no user column |
| PUT | `/api/m-session/order` | 332 | 🔴 **fleet-wide WRITE** | M-session canvas/order/marks — admin client, no user column |
| POST | `/api/m-session/marks` | 387 | 🔴 **fleet-wide WRITE** | M-session canvas/order/marks — admin client, no user column |
| DELETE | `/api/m-session/marks/{event_key:path}` | 423 | 🔴 **fleet-wide WRITE** | M-session canvas/order/marks — admin client, no user column |

#### `mass_builder.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| POST | `/api/mass-builder/run` | 15 | ✅ user-scoped *(via RLS only)* | by-id via get_client(); mass_searches has 4 RLS policies |
| GET | `/api/mass-builder/progress/{search_id}` | 44 | ✅ user-scoped *(via RLS only)* | by-id via get_client(); mass_searches has 4 RLS policies |
| POST | `/api/mass-builder/cleanup-orphans` | 92 | 🔴 **fleet-wide WRITE** | admin client sweeps EVERY user's status=running mass_searches → orphaned |
| POST | `/api/mass-builder/resume/{search_id}` | 110 | ✅ user-scoped *(via RLS only)* | by-id via get_client(); mass_searches has 4 RLS policies |
| POST | `/api/mass-builder/cancel/{search_id}` | 180 | ✅ user-scoped *(via RLS only)* | by-id via get_client(); mass_searches has 4 RLS policies |
| GET | `/api/mass-builder/results` | 207 | ✅ user-scoped *(via RLS only)* | by-id via get_client(); mass_searches has 4 RLS policies |
| GET | `/api/mass-builder/results/{search_id}` | 218 | ✅ user-scoped *(via RLS only)* | by-id via get_client(); mass_searches has 4 RLS policies |
| DELETE | `/api/mass-builder/results/{search_id}` | 232 | ✅ user-scoped *(via RLS only)* | by-id via get_client(); mass_searches has 4 RLS policies |

#### `mentions.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/mentions` | 38 | 🟡 fleet-wide read | all mentions |
| POST | `/api/mentions/seen-all` | 82 | 🔴 **fleet-wide WRITE** | task_mentions — admin client, no user column |
| POST | `/api/mentions/{mention_id}/seen` | 94 | 🔴 **fleet-wide WRITE** | task_mentions — admin client, no user column |

#### `model_parity.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/admin/model-parity/models` | 125 | 🟡 fleet-wide read | selects every strategy, no user filter |
| GET | `/api/admin/model-parity/parity` | 427 | 🟡 fleet-wide read | selects every strategy, no user filter |

#### `monitor.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/monitor/status` | 19 | ✅ user-scoped | monitor_status upsert keyed on user_id |
| POST | `/api/monitor/start` | 30 | ✅ user-scoped | monitor_status upsert keyed on user_id |
| POST | `/api/monitor/stop` | 50 | ✅ user-scoped | monitor_status upsert keyed on user_id |
| GET | `/api/monitor/engine-state` | 73 | ✅ user-scoped | monitor_status upsert keyed on user_id |

#### `packs.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/packs/confluence-groups` | 27 | ✅ user-scoped | confluence/general/risk packs — get_current_user |
| PUT | `/api/packs/confluence-groups` | 35 | ✅ user-scoped | confluence/general/risk packs — get_current_user |
| GET | `/api/packs/confluence-groups/templates` | 49 | ✅ user-scoped | confluence/general/risk packs — get_current_user |
| GET | `/api/packs/confluence-groups/triggers/{direction}` | 71 | ✅ user-scoped | confluence/general/risk packs — get_current_user |
| GET | `/api/packs/general` | 88 | ✅ user-scoped | confluence/general/risk packs — get_current_user |
| PUT | `/api/packs/general` | 96 | ✅ user-scoped | confluence/general/risk packs — get_current_user |
| GET | `/api/packs/general/templates` | 108 | ✅ user-scoped | confluence/general/risk packs — get_current_user |
| GET | `/api/packs/risk-management` | 124 | ✅ user-scoped | confluence/general/risk packs — get_current_user |
| PUT | `/api/packs/risk-management` | 132 | ✅ user-scoped | confluence/general/risk packs — get_current_user |
| GET | `/api/packs/risk-management/templates` | 144 | ✅ user-scoped | confluence/general/risk packs — get_current_user |
| GET | `/api/packs/stop-loss` | 161 | ✅ user-scoped | confluence/general/risk packs — get_current_user |
| GET | `/api/packs/stop-loss/templates` | 170 | ✅ user-scoped | confluence/general/risk packs — get_current_user |
| PUT | `/api/packs/stop-loss` | 182 | ✅ user-scoped | confluence/general/risk packs — get_current_user |
| GET | `/api/packs/take-profit` | 199 | ✅ user-scoped | confluence/general/risk packs — get_current_user |
| GET | `/api/packs/take-profit/templates` | 208 | ✅ user-scoped | confluence/general/risk packs — get_current_user |
| PUT | `/api/packs/take-profit` | 220 | ✅ user-scoped | confluence/general/risk packs — get_current_user |
| GET | `/api/packs/time-exit` | 236 | ✅ user-scoped | confluence/general/risk packs — get_current_user |
| PUT | `/api/packs/time-exit` | 244 | ✅ user-scoped | confluence/general/risk packs — get_current_user |
| GET | `/api/packs/time-exit/templates` | 256 | ✅ user-scoped | confluence/general/risk packs — get_current_user |
| POST | `/api/packs/parity-test` | 377 | ✅ user-scoped | confluence/general/risk packs — get_current_user |

#### `portfolios.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/portfolios` | 167 | ✅ user-scoped | db.py portfolio helpers scope on get_current_user_id() |
| POST | `/api/portfolios` | 312 | ✅ user-scoped | db.py portfolio helpers scope on get_current_user_id() |
| GET | `/api/portfolios/{portfolio_id}` | 355 | ✅ user-scoped | db.py portfolio helpers scope on get_current_user_id() |
| PUT | `/api/portfolios/{portfolio_id}` | 361 | ✅ user-scoped | db.py portfolio helpers scope on get_current_user_id() |
| DELETE | `/api/portfolios/{portfolio_id}` | 386 | ✅ user-scoped | db.py portfolio helpers scope on get_current_user_id() |
| POST | `/api/portfolios/{portfolio_id}/duplicate` | 412 | ✅ user-scoped | db.py portfolio helpers scope on get_current_user_id() |
| POST | `/api/portfolios/{portfolio_id}/compute` | 438 | ✅ user-scoped | db.py portfolio helpers scope on get_current_user_id() |
| GET | `/api/portfolios/{portfolio_id}/trades` | 548 | ✅ user-scoped | db.py portfolio helpers scope on get_current_user_id() |
| POST | `/api/portfolios/{portfolio_id}/requirements/check` | 582 | ✅ user-scoped | db.py portfolio helpers scope on get_current_user_id() |
| POST | `/api/portfolios/{portfolio_id}/health` | 623 | ✅ user-scoped | db.py portfolio helpers scope on get_current_user_id() |
| GET | `/api/portfolios/{portfolio_id}/anomalies` | 679 | ✅ user-scoped | db.py portfolio helpers scope on get_current_user_id() |
| GET | `/api/portfolios/{portfolio_id}/account` | 717 | ✅ user-scoped | db.py portfolio helpers scope on get_current_user_id() |
| POST | `/api/portfolios/{portfolio_id}/account/deposit` | 726 | ✅ user-scoped | db.py portfolio helpers scope on get_current_user_id() |
| DELETE | `/api/portfolios/{portfolio_id}/account/ledger/{entry_id}` | 755 | ✅ user-scoped | db.py portfolio helpers scope on get_current_user_id() |
| POST | `/api/portfolios/{portfolio_id}/worst-case` | 782 | ✅ user-scoped | db.py portfolio helpers scope on get_current_user_id() |
| POST | `/api/portfolios/{portfolio_id}/capital-utilization` | 806 | ✅ user-scoped | db.py portfolio helpers scope on get_current_user_id() |
| GET | `/api/portfolios/{portfolio_id}/open-positions` | 857 | ✅ user-scoped | db.py portfolio helpers scope on get_current_user_id() |
| POST | `/api/portfolios/preview` | 870 | ✅ user-scoped | db.py portfolio helpers scope on get_current_user_id() |
| POST | `/api/portfolios/recommendations` | 902 | ✅ user-scoped | db.py portfolio helpers scope on get_current_user_id() |

#### `r_session.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/r-session/heartbeats` | 176 | 🟡 fleet-wide read | reads deploy log / service state off disk + system_settings |
| POST | `/api/r-session/heartbeat` | 219 | 🔴 **fleet-wide WRITE** | set_system_setting(heartbeat_key) — global system_settings key |
| GET | `/api/r-session/deploy-log` | 308 | 🟡 fleet-wide read | reads deploy log / service state off disk + system_settings |
| GET | `/api/r-session/service` | 357 | 🟡 fleet-wide read | reads deploy log / service state off disk + system_settings |

#### `recompute_jobs.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| POST | `/api/jobs/recompute` | 28 | ✅ user-scoped | strategy_ids filtered .eq(user_id); job routes check ownership |
| GET | `/api/jobs/{job_id}` | 107 | ✅ user-scoped | strategy_ids filtered .eq(user_id); job routes check ownership |
| POST | `/api/jobs/{job_id}/cancel` | 122 | ✅ user-scoped | strategy_ids filtered .eq(user_id); job routes check ownership |
| GET | `/api/jobs` | 138 | ✅ user-scoped | strategy_ids filtered .eq(user_id); job routes check ownership |
| GET | `/api/jobs/cron-stats/algo-history` | 153 | ✅ user-scoped | strategy_ids filtered .eq(user_id); job routes check ownership |

#### `replay_sim.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/strategy-health-sim/queue` | 223 | 🟡 fleet-wide read | shared SIM queue |
| GET | `/api/strategy-health-sim/{sid}` | 237 | 🟡 fleet-wide read | any sid |
| POST | `/api/strategy-health-sim/{sid}/run` | 253 | 🔴 **fleet-wide WRITE** | INSERT replay_sim_requests for ANY sid; only a 404-if-missing check |
| GET | `/api/strategy-health-sim/{sid}/requests` | 278 | 🟡 fleet-wide read | any sid |
| GET | `/api/strategy-health-sim/{sid}/optin` | 287 | 🟡 fleet-wide read | any sid |
| PUT | `/api/strategy-health-sim/{sid}/optin` | 297 | 🔴 **fleet-wide WRITE** | UPSERT replay_sim_optin for ANY sid — arms/disarms nightly SIM |

#### `requirements.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/requirements` | 19 | ✅ user-scoped | db.py requirement helpers scope on get_current_user_id() |
| POST | `/api/requirements` | 26 | ✅ user-scoped | db.py requirement helpers scope on get_current_user_id() |
| GET | `/api/requirements/{req_id}` | 34 | ✅ user-scoped | db.py requirement helpers scope on get_current_user_id() |
| PUT | `/api/requirements/{req_id}` | 44 | ✅ user-scoped | db.py requirement helpers scope on get_current_user_id() |
| DELETE | `/api/requirements/{req_id}` | 58 | ✅ user-scoped | db.py requirement helpers scope on get_current_user_id() |

#### `resampled_store_admin.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/admin/resampled-store/coverage` | 68 | 🟡 fleet-wide read | coverage over the shared resampled store |

#### `run_history.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/run-history` | 27 | 🟡 fleet-wide read | admin client; run_history has no user column |

#### `settings.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/settings` | 22 | ✅ user-scoped | per-user settings row |
| PUT | `/api/settings` | 36 | ✅ user-scoped | per-user settings row |

#### `strategies.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/strategies` | 151 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| GET | `/api/strategies/models` | 403 | ⚠️ unauthenticated (no DB) | no Depends() at all — returns the static model registry, touches no DB |
| POST | `/api/strategies` | 432 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| GET | `/api/strategies/{strategy_id}` | 544 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| PUT | `/api/strategies/{strategy_id}` | 651 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| DELETE | `/api/strategies/{strategy_id}` | 697 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| POST | `/api/strategies/{strategy_id}/run-hifi-pass2` | 723 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| POST | `/api/strategies/append-recent-trades-bulk` | 1113 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| POST | `/api/strategies/run-hifi-pass2-bulk` | 1196 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| POST | `/api/strategies/{strategy_id}/run-parity` | 1251 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| GET | `/api/strategies/{strategy_id}/gate-parity` | 1274 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| GET | `/api/strategies/{strategy_id}/window-1s` | 1295 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| GET | `/api/strategies/{strategy_id}/live-gate-telemetry` | 1339 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| GET | `/api/strategies/{strategy_id}/pine-export` | 1396 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| GET | `/api/strategies/{strategy_id}/pine-readiness` | 1431 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| GET | `/api/strategies/{strategy_id}/grace-shadow-comparison` | 1453 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| POST | `/api/strategies/run-parity-bulk` | 1487 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| POST | `/api/strategies/{strategy_id}/parity-check` | 1553 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| POST | `/api/strategies/{strategy_id}/duplicate` | 1623 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| POST | `/api/strategies/bulk-delete` | 1649 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| PATCH | `/api/strategies/{strategy_id}/forward-test-start` | 1702 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| PATCH | `/api/strategies/{strategy_id}/snapshot-subscription` | 1774 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| GET | `/api/strategies/{strategy_id}/trades` | 1841 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| GET | `/api/strategies/{strategy_id}/cache-coverage` | 1872 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| GET | `/api/strategies/{strategy_id}/algo-trades` | 1979 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| GET | `/api/strategies/{strategy_id}/divergence-data` | 2032 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| GET | `/api/strategies/admin/divergence-summary` | 2216 | ✅ user-scoped | load_strategies_admin(user_id) — scoped despite the /admin path |
| GET | `/api/strategies/{strategy_id}/forward-test` | 2372 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| GET | `/api/strategies/{strategy_id}/kpis` | 2425 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| GET | `/api/strategies/{strategy_id}/chart-data` | 2793 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| GET | `/api/strategies/{strategy_id}/confluence-chart` | 2978 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| GET | `/api/strategies/{strategy_id}/chart-data-cache` | 3191 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| GET | `/api/strategies/{strategy_id}/cache-bars` | 3452 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| GET | `/api/strategies/{strategy_id}/trigger-analysis` | 3573 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| POST | `/api/strategies/{strategy_id}/update` | 3656 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| POST | `/api/strategies/{strategy_id}/refresh` | 3745 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| POST | `/api/strategies/{strategy_id}/manual-exit` | 3797 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |
| GET | `/api/strategies/{strategy_id}/trade-zoom` | 3867 | ✅ user-scoped | get_strategy_by_id_db via get_client() → RLS; writes stamp user_id |

#### `strategy_health.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/admin/strategy-health` | 184 | 🟡 fleet-wide read | admin client, selects every strategy — Kevin's reported symptom |
| GET | `/api/admin/strategy-health/backlog` | 865 | 🟡 fleet-wide read | admin client, selects every strategy — Kevin's reported symptom |
| GET | `/api/admin/strategy-health/by-hour` | 1551 | 🟡 fleet-wide read | admin client, selects every strategy — Kevin's reported symptom |
| GET | `/api/admin/strategy-health/by-deploy` | 1938 | 🟡 fleet-wide read | admin client, selects every strategy — Kevin's reported symptom |
| GET | `/api/admin/strategy-health/bar-parity` | 2586 | 🟡 fleet-wide read | admin client, selects every strategy — Kevin's reported symptom |

#### `strategy_models_admin.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/admin/strategy-models/live` | 164 | ⚪ no DB | static model registry |
| GET | `/api/admin/strategy-models/backtest` | 170 | ⚪ no DB | static model registry |
| GET | `/api/admin/strategy-models/live/{model_id}` | 176 | ⚪ no DB | static model registry |
| GET | `/api/admin/strategy-models/backtest/{model_id}` | 182 | ⚪ no DB | static model registry |

#### `strategy_notes.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/strategy-notes` | 22 | 🟡 fleet-wide read | all notes |
| GET | `/api/strategy-notes/{sid}` | 30 | 🟡 fleet-wide read | notes for any sid |
| POST | `/api/strategy-notes/{sid}` | 36 | 🔴 **fleet-wide WRITE** | INSERT strategy_notes for ANY sid |

#### `system_settings.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/admin/system-settings` | 67 | 🟡 fleet-wide read | global settings table |
| GET | `/api/admin/system-settings/{key}` | 108 | 🟡 fleet-wide read | global settings table |
| PATCH | `/api/admin/system-settings/{key}` | 124 | 🔴 **fleet-wide WRITE** | system_settings upsert on `key` — the Data Worker reads it within ~30s |

#### `trade_snapshots.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| POST | `/api/admin/trade-snapshots` | 93 | ✅ user-scoped | every query .eq(user_id); _all_sids scoped |
| GET | `/api/admin/trade-snapshots` | 113 | ✅ user-scoped | every query .eq(user_id); _all_sids scoped |
| POST | `/api/admin/trade-snapshots/diff` | 126 | ✅ user-scoped | every query .eq(user_id); _all_sids scoped |
| DELETE | `/api/admin/trade-snapshots/{snap_id}` | 144 | ✅ user-scoped | every query .eq(user_id); _all_sids scoped |

#### `update_jobs.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| POST | `/api/update-jobs/run` | 25 | ✅ user-scoped *(via RLS only)* | by-id via get_client(); update_jobs has 4 RLS policies |
| GET | `/api/update-jobs/progress/{job_id}` | 143 | ✅ user-scoped *(via RLS only)* | by-id via get_client(); update_jobs has 4 RLS policies |
| POST | `/api/update-jobs/cancel/{job_id}` | 174 | ✅ user-scoped *(via RLS only)* | by-id via get_client(); update_jobs has 4 RLS policies |
| GET | `/api/update-jobs` | 195 | ✅ user-scoped *(via RLS only)* | by-id via get_client(); update_jobs has 4 RLS policies |
| GET | `/api/update-jobs/{job_id}` | 205 | ✅ user-scoped *(via RLS only)* | by-id via get_client(); update_jobs has 4 RLS policies |
| DELETE | `/api/update-jobs/{job_id}` | 217 | ✅ user-scoped *(via RLS only)* | by-id via get_client(); update_jobs has 4 RLS policies |
| POST | `/api/update-jobs/cleanup-orphans` | 232 | 🔴 **fleet-wide WRITE** | admin client sweeps EVERY user's status=running update_jobs → orphaned |

#### `webhook_groups.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/webhook-groups` | 19 | ✅ user-scoped | db.py webhook helpers scope on get_current_user_id() |
| POST | `/api/webhook-groups` | 26 | ✅ user-scoped | db.py webhook helpers scope on get_current_user_id() |
| GET | `/api/webhook-groups/{group_id}` | 33 | ✅ user-scoped | db.py webhook helpers scope on get_current_user_id() |
| PUT | `/api/webhook-groups/{group_id}` | 43 | ✅ user-scoped | db.py webhook helpers scope on get_current_user_id() |
| DELETE | `/api/webhook-groups/{group_id}` | 57 | ✅ user-scoped | db.py webhook helpers scope on get_current_user_id() |
| POST | `/api/webhook-groups/{group_id}/duplicate` | 67 | ✅ user-scoped | db.py webhook helpers scope on get_current_user_id() |
| POST | `/api/webhook-groups/{group_id}/test/{event_type}` | 81 | ✅ user-scoped | db.py webhook helpers scope on get_current_user_id() |
| GET | `/api/webhook-groups/meta/event-types` | 161 | ✅ user-scoped | db.py webhook helpers scope on get_current_user_id() |

#### `webhooks.py`

| method | path | line | verdict | basis |
|---|---|---|---|---|
| GET | `/api/webhooks/templates` | 19 | ✅ user-scoped | db.py webhook helpers scope on get_current_user_id() |
| POST | `/api/webhooks/templates` | 26 | ✅ user-scoped | db.py webhook helpers scope on get_current_user_id() |
| GET | `/api/webhooks/templates/{template_id}` | 33 | ✅ user-scoped | db.py webhook helpers scope on get_current_user_id() |
| PUT | `/api/webhooks/templates/{template_id}` | 43 | ✅ user-scoped | db.py webhook helpers scope on get_current_user_id() |
| DELETE | `/api/webhooks/templates/{template_id}` | 57 | ✅ user-scoped | db.py webhook helpers scope on get_current_user_id() |
| POST | `/api/webhooks/test` | 74 | ✅ user-scoped | db.py webhook helpers scope on get_current_user_id() |
| GET | `/api/webhooks/delivery-log` | 105 | ✅ user-scoped | db.py webhook helpers scope on get_current_user_id() |


---

## 8. Follow-up tasks this audit implies (flagged, NOT fixed)

Per the rails, each of these is its own task for M to open — nothing here was changed.

1. **Block the §5.1 + §5.2 + §5.3 routes in the dev environment** (16 routes). Highest priority:
   the two `cleanup-orphans` **boot** sweeps, because they fire without a click on every dev deploy.
2. **Confirm the `DEV_USER_ID` default UUID** `19d47e46-…` in the two committed scripts — whose
   account is it, and does anything inherit that default?
3. **Decide the board posture** (§5.5): shared-by-design, but dev-origin writes should be
   identifiable.
4. **`replay_sim` ownership check** — `/run` and `/optin` should verify `strategies.user_id`
   matches the caller, the way `admin_parity/snapshot` already does at `admin_parity.py:75`.
   That one is a real scoping bug independent of the env split.
5. **Strategy Health page scoping** (§4.1) — decide whether fleet-wide is intended (it is an admin
   view) or whether it should scope like the badge does. Kevin's report is the trigger.
6. **`GET /api/strategies/models` has no auth dependency** — harmless today (static registry, no
   DB) but it is the only unauthenticated route in the app and should be intentional, not accidental.
