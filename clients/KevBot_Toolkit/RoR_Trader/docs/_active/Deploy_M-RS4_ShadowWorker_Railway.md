# Deploy — M-RS4 shadow-worker on Railway (Step F rollout) — 2026-06-30

Click-by-click for standing up the continuous-resident backtest engine service, then the
phased, gated rollout. **Nothing here is irreversible until we arm writes** (a single env
var), and rollback is one flag flip. The shadow-worker is its OWN Railway service (crash /
CPU / RAM isolation from the live worker, data-worker, and batch-worker).

Two independent safety gates, both default-SAFE:
- `RORT_BACKTEST_LANE_MODE` — `button` (default) = service idles inert; `shadow` = resident
  lane runs.
- `RORT_SHADOW_DRY_RUN` — `1` (default) = computes + logs "would-write", writes NOTHING;
  `0` = actually writes settled `backtest_<model>` trades.
So `shadow` + `dry_run=1` runs the whole pipeline on real dev data writing nothing — that's
the bring-up state we watch first.

---

## 1. Create the service (mirror the batch-worker)
The shadow-worker uses the same repo + the same mold as `batch-worker`; the only differences
are the Dockerfile path and the env vars. In the Railway project:

1. **New → Empty Service** (or **New → GitHub Repo** and pick this repo, same as batch-worker).
   Name it **`shadow-worker`**.
2. Open the new service → **Settings**:
   - **Source**: the same GitHub repo + branch the other services deploy from (**`dev`**).
     (Confirm against the batch-worker's Source — match it exactly.)
   - **Build**: set **Dockerfile Path** = `clients/KevBot_Toolkit/RoR_Trader/Dockerfile.shadow-worker`
     (match the path style the batch-worker uses for `Dockerfile.batch-worker` — if theirs is
     just `Dockerfile.batch-worker`, the repo root is set to the RoR_Trader dir and you'd use
     `Dockerfile.shadow-worker`). **Easiest: open batch-worker → Settings, copy its Root
     Directory + Dockerfile Path exactly, then swap `batch-worker`→`shadow-worker`.**
   - **Start Command**: leave blank (the Dockerfile's `CMD` runs `python src/shadow_worker.py`).
   - **Healthcheck**: none needed (the Dockerfile has a container HEALTHCHECK on
     `/tmp/shadow_worker_alive`).
3. **Variables**: the simplest correct path is **"Add Reference" / shared variables** — give it
   the SAME database/env vars the batch-worker has (SUPABASE_*, SUPABASE_CONNECTION_STRING,
   POLYGON keys aren't needed but harmless, etc.). Mirror batch-worker's variable list. Then add
   the shadow-worker-specific ones in step 2 of §2 below.

> If anything about Source/Root/Dockerfile is unclear, **do NOT guess** — screenshot the
> batch-worker's Settings → Source + Build panels and the shadow-worker's, and I'll diff them.

## 2. First deploy = INERT, then DRY-RUN (safe)
1. Deploy with **no** extra vars first. Expected logs:
   `[shadow_worker] up ... lane_mode=button` then `inert — idling`. ✅ proves build + boot.
2. Now flip it into the dry-run resident lane. Set Variables (I'll set these myself once the
   service exists if you give me access; otherwise set them here):
   - `RORT_BACKTEST_LANE_MODE=shadow`
   - `RORT_SHADOW_DRY_RUN=1`  (writes nothing)
   - `RORT_SHADOW_SHARD=TSLA`  (start with ONE symbol)
   Redeploy. Expected logs:
   `RESIDENT lane active — dry_run=True shard={'TSLA'}` · `tracking N strategies (N eligible)` ·
   `cycle would-write M trades`. ✅ proves the full pipeline on real dev data, zero writes.

## 3. Validate dry-run vs the truth (before arming)
With dry-run running, confirm the lane it WOULD produce matches a from-cold recompute. Offline
gate is already GREEN (`_shadow_manager_validate.py` 4/4 byte-identical). On dev, spot-check a
canary by running a known-answer recompute and eyeballing counts in the logs. (Queries in §6.)

## 4. Arm writes for ONE canary symbol (the real cutover)
Only after §3 looks right:
- `RORT_SHADOW_DRY_RUN=0` (now it writes settled `backtest_<model>` trades).
- Keep `RORT_SHADOW_SHARD=TSLA`.
Redeploy. Watch one RTH session. Compare the shadow-written trades to (a) a from-cold
`full_recompute` and (b) the algo/alert lanes (§6). Idempotent: writes are de-duped by the
`(strategy_id, entry_fill_ts, exit_fill_ts)` unique index, so a restart can't double-write.

## 5. Expand → fleet
Widen `RORT_SHADOW_SHARD` (comma-separated symbols) or unset it (= all), each step gated by a
recompute comparison. Multiple instances → shard by symbol so a symbol's bars load once per
instance.

## 6. Monitoring & validation
**Log lines to watch:** `RESIDENT lane active`, `tracking N strategies`, `cycle wrote/would-write
M`, any `sid=… poll failed`.

**Per-canary parity (run a from-cold recompute, compare counts):**
```
PYTHONPATH=. ../.venv/bin/python _shadow_manager_validate.py <SID> 1   # offline, byte gate
```
**DB sanity (shadow lane is populating, none stuck provisional):**
```sql
SELECT data_source, provisional, count(*), max(entry_fill_ts)
FROM trades WHERE strategy_id = <SID>
GROUP BY data_source, provisional ORDER BY 1,2;
```

## 7. Rollback (instant, safe)
Set `RORT_BACKTEST_LANE_MODE=button` (or `RORT_SHADOW_DRY_RUN=1`) and redeploy → the lane stops
writing immediately; the existing append/cron lane is untouched the whole time. To remove the
provisional column: `ALTER TABLE trades DROP COLUMN IF EXISTS provisional;` (only after the
feature is abandoned). The service can also be paused with `SHADOW_WORKER_DISABLED=1`.

## Notes / who does what
- **I (Claude) flip the flags + monitor** ([[feedback_solo_ownership_flag_authority]]): once the
  service exists and I have variable access, I set lane_mode / dry_run / shard and watch the logs.
- **You (Kevin) create the Railway service** (§1) — the one UI step I can't do. Do it WITH me on
  the line so I can diff Settings against batch-worker before first deploy.
- The shadow lane is ADDITIVE and gated — the live worker, alerts, and the existing backtest
  lane are never touched until §4, and that's reversible.
