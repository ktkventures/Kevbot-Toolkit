# SOP — Autonomous Iteration ("RoR Mode")

**Last updated:** 2026-06-08
**Purpose:** Define the rules of engagement when Claude iterates autonomously
through the divergence-hunting roadmap. The goal is fast progress without
introducing drift, lost work, or regressions that aren't caught immediately.

Companion to `docs/Roadmap_Divergence_Hunting.md` — that doc lists *what*
to work on; this one defines *how* to work on it autonomously.

---

## TL;DR — what Claude does without asking, what requires sign-off

**Auto-allowed (Claude proceeds without checkpoint):**
- All read-only DB queries, analysis scripts, dashboard observation
- Feature-branch commits, doc updates, iteration log writes
- Creating test strategies with `claude_iter_*` tag via the unified factory
- Click Update All Data and Update New Data on **any** strategy including
  production canaries — these go through the same service functions Kevin's
  UI clicks call, no parity risk

**Ask-first (Claude pauses and requests approval):**
- Push to dev (triggers Railway redeploy + worker restart)
- Modify pack manifest files in `user_packs/`
- Modify `strategy_factory.py` or any service function in `src/api/services/`
- Modify production strategies' configs (anything without `claude_iter_*` tag)

**Never without explicit instruction:**
- Force-restart worker, modify `worker.py` / `ralph_engine.py`
- Revert prior commits
- Delete strategies without `claude_iter_*` tag
- Delete trades or alerts directly
- Modify CI/CD, Railway env vars, Supabase schema

---

## The parity guarantees

Every autonomous action that touches BT data, alerts, or strategies must go
through the same service functions the UI buttons call. The unification work
on 2026-06-04 (task #50) collapsed all paths through these single entry
points:

| Operation | Required service function | File |
|---|---|---|
| Strategy creation | `build_strategy_config()` → `insert_strategy_admin()` | `src/strategy_factory.py:107` + `src/db.py` |
| Update All Data | `update_all_backtest_trades_for_strategy()` | `src/api/services/forward_test_service.py` |
| Update New Data | `append_new_backtest_trades_for_strategy()` | `src/api/services/forward_test_service.py:1264` |
| Bulk UAD | `start_update_job_async(_, 'all', sids, user_id)` | `src/update_jobs.py` |

**What would break parity (NEVER do these autonomously):**
- Direct `c.table('trades').insert/update/delete()` for BT trades
- Direct `c.table('strategies').insert()` for new strategies
- Direct `c.table('strategies').update({'config': ...})` for config edits
- Running `run_unified_backtest()` standalone and writing results manually

The result of using the service functions is **bit-for-bit identical** to
clicking the buttons in Streamlit or Next.js. That's the parity contract.

### Identifying autonomous-created strategies

Test strategies that Claude creates carry a `tag` field set to
`claude_iter_<iteration_id>` so they can be:

- Filtered out of production canary cohort measurements
- Found and deleted en masse at the end of an iteration
- Audited via `SELECT id, name, tag FROM strategies WHERE tag LIKE 'claude_iter_%'`

Production strategies must NEVER have this tag.

---

## Measurement standards — "did this fix help?"

Every iteration produces a verdict using a fixed protocol. The protocol uses
the **stability window** + **tolerance tier** standards defined below.

### Stability window (the apples-to-apples filter)

When measuring combined % to compare pre-change vs post-change, always clip
the event window to the *settled* region. The dashboards now apply this
automatically; manual analysis scripts must respect the same boundaries.

| Boundary | Default | Meaning |
|---|---|---|
| **Upper (now-tail)** | `now - 15 min` | BT lane lag-minutes setting holds the BT engine back from the most-recent bars. Events in this window have artificially high phantom counts. |
| **Lower for deploys (warmup)** | `deploy_ts + 30 min` | Worker restart after deploy → snapshot resume → warmup. Stateful stops (swing/ATR) need ~30 bars to populate buffers. |
| **Minimum window** | ≥ 2 hours of stable data | Below this, mark "low confidence" and don't act on it. |

### Tolerance tiers (the pair window)

Always report at multiple tolerances so the right signal is visible. The
By Hour and By Deploy dashboards offer a tolerance selector with these
options:

| Tolerance | When to use | Notes |
|---|---|---|
| **±5s** (standard) | Default. Tight pairing — anything outside is treated as unpaired. | 96% of paired pairs land sub-second; this is the right strictness for engine-fidelity work. |
| **±10s** | When investigating "almost-paired" cases | Captures pairs that just barely missed the strict window. Useful for diagnosing late-exit patterns. |
| **±60s** | Only for >1m timeframes (1Min, 5Min, etc.) | Mostly noise for 10s timeframes (most of our canary cohort). Use sparingly. |

### Verdict thresholds

Compare baseline vs post-change combined % within the same stability window
and tolerance:

| Change in combined % | Verdict | Action |
|---|---|---|
| **+5 pts or more** OR phantom count down ≥20% | **Improved** | Log success in iteration log, commit on feature branch, advance to next hypothesis |
| Within ±2 pts | **No effect** | Log, revert change on feature branch, try a different hypothesis |
| Down 2-5 pts | **Inconclusive** | Re-measure with a wider window or larger cohort. If still down → treat as "Made worse." |
| **Down 5+ pts** | **Made worse** | **Revert immediately**, log, escalate to Kevin |

### Statistical confidence requirements

- Minimum **50 alerts** in the measurement window
- Minimum **2 hours** of stable data
- Minimum **3 strategies** (when measuring a cohort)

Below any of these → abort the iteration, mark inconclusive, ask for direction
or expand the window.

### Layer 4 (individual trade inspection)

When aggregate combined % is inconclusive, drill into specific trades. The
existing tools support this:

- `src/_probe_bar_diagnostics.py` — per-bar live vs BT comparison
- `src/_divergence_walkthrough.py` — Layer 1 + Layer 2B walkthrough on
  individual sids
- Strategy detail page → Chart & Trades tab — visual per-trade view

Use Layer 4 to answer "what specifically about this trade made it phantom?"
which often reveals the structural bug.

---

## Escalation triggers

Stop autonomous iteration and surface to Kevin when:

- 3 consecutive "no effect" results on the same hypothesis
- Any "made worse" result
- Encountering an unknown error or unexpected DB state
- Need to touch a file outside the planned change scope (e.g., turns out
  the fix requires modifying `strategy_factory.py`, which is ask-first)
- Battle Plan window or any other active diagnostic test is in effect
- Less than 50 alerts in measurement window (insufficient data)
- Production canary combined % drops > 5 pts during autonomous work
  (regression detector)
- 5+ hours of autonomous work elapsed without checkpoint with Kevin
- Discovery of a bug that affects production data integrity (always escalate)

---

## Iteration logging

Every iteration produces a structured log at
`docs/iterations/YYYY-MM-DD_iter_<id>.md`:

```markdown
# Iteration <id> — <hypothesis from roadmap>

**Date:** YYYY-MM-DD
**Hypothesis:** <H# from Roadmap_Divergence_Hunting.md>
**Test bucket:** A / B / C
**Branch:** feat/claude-iter-<id>

## Pre-change baseline
- Cohort: <which strategies>
- Window: <stability-windowed timestamps>
- Tolerance: ±5s
- Combined %: <number>
- Alert count: <number>

## Change
- Files modified: <list>
- Approach: <1-2 sentence summary>
- Why this should help: <reasoning>

## Post-change measurement
- Window: <stability-windowed timestamps>
- Tolerance: ±5s (and ±10s if relevant)
- Combined %: <number>
- Δ vs baseline: <+X pts>
- Alert count: <number>

## Verdict: improved | no effect | inconclusive | made worse

## Next
- <if improved: next hypothesis to try>
- <if not: what alternative to try>
- <if escalated: reason for handoff>
```

Commits in the iteration are tagged `[claude-iter-<id>]` in the message for
`git log` traceability.

---

## Safety nets

1. **Backup branch before any production-touching work:**
   `git branch claude-iter-pre-<date>-backup` before starting any session
   that may modify pack files, strategy configs, or other production data.

2. **Feature branch always** — `feat/claude-iter-<id>` for all work.
   Never commit to dev without explicit per-iteration Kevin approval (which
   only comes after a verified "improved" verdict).

3. **Daily session digest** — at session end, post a summary of:
   - Iterations attempted + verdicts
   - Current cohort state (canary cohort combined % snapshot)
   - Open blockers
   - What's queued next

4. **Hard stop on production data writes** — analysis scripts that update
   the trades or strategies tables must call service functions, not raw
   `c.table().update()` for production rows. Test strategies (`claude_iter_*`
   tagged) are the only DB writes Claude does autonomously.

---

## Playwright MCP setup (browser access for visual verification)

Once configured, Claude can navigate the admin dashboards directly to verify
the data Claude computed in scripts matches what the UI renders. This catches
drift between the analysis layer and the display layer.

### Installation (Kevin runs these)

1. **Install the Playwright MCP server:**
   ```bash
   npx -y @playwright/mcp@latest --help
   ```
   This downloads on first run; subsequent invocations are fast.

2. **Add to Claude Code MCP config** (typically `~/.config/claude/mcp.json`
   or via the Claude Code CLI):
   ```json
   {
     "mcpServers": {
       "playwright": {
         "command": "npx",
         "args": ["-y", "@playwright/mcp@latest"]
       }
     }
   }
   ```

3. **Start the Next.js dev server** (so Claude has a target):
   ```bash
   cd frontend && npm run dev
   ```
   This runs on `http://localhost:3000` by default. Production Railway URL
   also works if you want Claude to verify the live dashboards instead.

4. **Restart Claude Code** so the MCP server is loaded.

### What Claude can do with browser access

- Navigate to `/admin/strategy-health` and screenshot the V3/V4 tabs
- Click filter options (Origin, Trigger Pack, etc.) and verify filter logic
- Compare dashboard numbers vs the Python script outputs for drift detection
- Inspect specific strategy detail pages for Layer 4 trade-level investigation
- Verify a fix's UI impact (e.g., does the stability-window indicator render?)

### Boundaries (additions to authorization tiers)

- ✅ Auto: read-only navigation, screenshots, comparing UI to data layer
- ⚠️ Ask first: clicking destructive UI elements (Delete buttons, Cancel jobs)
- 🛑 Never: any action that creates/modifies production data via UI clicks
  (always use service functions instead)

---

## Starting an iteration — the standard flow

1. **Pick a hypothesis** from `Roadmap_Divergence_Hunting.md` H1-H8
2. **Capture baseline:**
   - Read current combined % on relevant cohort
   - Apply stability window + tolerance standards
   - Save to iteration log
3. **Create branch:** `git checkout -b feat/claude-iter-<id>`
4. **Make the change** (within authorization tier; ask first if outside)
5. **Test via the appropriate bucket:**
   - Bucket A → UAD on test strategy, re-measure
   - Bucket B → deploy, wait, re-measure post-deploy window
   - Bucket C → run replay harness, compare outputs
6. **Apply verdict thresholds**
7. **If improved:** log success, request Kevin's review of the branch for
   merge to dev (push to dev is ask-first)
8. **If not improved:** revert on the branch (no production impact), log,
   queue next hypothesis
9. **If escalated:** stop, summarize, hand off to Kevin

---

## Document lineage

- `docs/Roadmap_Divergence_Hunting.md` — what to work on (hypothesis catalog,
  phase plan)
- `docs/SOP_Autonomous_Iteration.md` — this doc; how to work on it
- `docs/SOP_Strategy_Health_Check.md` — 4-layer analysis methodology
- `docs/Known_Bugs.md` — active bug log
- `docs/iterations/` — per-iteration logs (Claude appends, Kevin reviews)
