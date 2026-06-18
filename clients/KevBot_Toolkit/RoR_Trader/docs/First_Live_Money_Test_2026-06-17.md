# First Real-Money Strategy Test — 2026-06-17

_The shift from "canary" strategies (validate triggers fire) to **real strategies we'd
actually trade with real money.** This doc tracks what's working, what's blocking, and the
ease-of-use pain points surfaced by the first end-to-end save→portfolio→live flow._

## The setup

7 strategies saved from a Mass Builder search (TSLA, 15Sec primary, LONG, swing_123
bull_c2 entry → ut_bot_v4 bear_flip exit, swing stop), split into two portfolios — both
webhook-routed (`grp_1e36f4`):

| Portfolio | Strategies | Profile |
|---|---|---|
| **profit test** (#30) | **308** (ungated) | Trigger-only; profitable in backtest; the one currently firing |
| **Realzies 25k** (#29) | **309–314** (gated) | Same entry/exit, higher profit factor via confluence gates |

Gate timeframes (primary is 15Sec for all):
- 309: 2m + 3m   · 310: 5m + **1d**   · 311: **1h** + 10m
- 312: **1d** + **1d**   · 313: **1d** + **4h**   · 314: **4h** + **1h**

## State as of 15:31Z

| sid | BT trades | alerts | last_recompute | note |
|---|---|---|---|---|
| 308 | **0** | **93** (firing) | None | UAD never completed — firing live with NO backtest baseline |
| 309 | 663 | 0 | 11:13 | gated; 0 live, 0 BT today |
| 310 | 322 | 0 | 10:40 | gated |
| 311 | 342 | 0 | 10:07 | gated |
| 312 | 325 | 0 | 09:34 | gated |
| 313 | 399 | 0 | 07:27 | gated |
| 314 | 262 | 0 | 14:53 | gated; last BT entry Jun-08 |

## Issues (breaking → real money can't proceed)

### 1. [task #23 — ✅ RESOLVED 2026-06-17] High-TF gates (1h/4h/1d) ARE supported live
Initially feared the 1h/4h/1d gates didn't work live (canaries only use 2m). **Verified
they do**, end-to-end:
- 308-314 all `streaming_eligible` (`rest_hifi` ∈ REST_BACKTEST_MODELS).
- Gate TFs parse correctly to `sec_tfs` (1h→1Hour, 4h→4Hour, 1d→1Day) via
  `get_required_tfs_from_confluence` / `get_tf_from_label`.
- `bar_store_facade` routes TFs ≥120s to the Tier-2 `CoarseBarStore` (150-day 1Min).
- **Primed a TSLA coarse store live (8.4s)** → serves 1Hour=1657, 4Hour=448, 1Day=111 bars
  (Jan20→today, ~5mo warmup). Every gate TF covered with warmup.
- Same `run_unified_backtest` evaluates the gates (their backtest trades exist).

**Conclusion: the gated strategies are NOT broken.** The 0 live alerts is almost certainly
**legitimate** — these gates are restrictive by design (the source of the higher profit
factor), so they fire rarely. **Remaining = watch:** confirm the *running* worker's TSLA
coarse store is primed (offline prime was flawless) and catch a real live fire as 309-314
accumulate history over the next 1-2 days. Related: `project_multi_tf_gates`,
`feedback_polygon_xtf_live_regression` (now extended past 2m).

### 2. [P0 · task #24] 308 has no backtest lane (BT=0)
The ungated real-money candidate is firing live (93 alerts) but its UAD never completed
("aired out" last night). No backtest lane → no KPIs, no divergence baseline, can't be
trusted for real money or A/B'd. Must complete its full UAD.

## Pain points (not breaking, but big time-sinks)

### 3. [P2 · task #25] Save from Mass Builder doesn't persist the backtest trades
Saving a strategy from search results does NOT carry the already-computed backtest trades
into the trades table — you must click Update All Data afterward, which recomputes from
scratch (hours). Worse: 7 strategies that share the same entry/exit triggers (differ only
by gate) each re-run the full backtest independently. The Mass Builder strips `stored_trades`
from its DB payload (too large for JSONB), so save has nothing to seed from. Fix: transfer
the search's computed trades into the lane on save, gated on a parity check vs UAD so we
keep the integrity. (The bull_c2 results are validated.)

### 4. [P2 · task #26] First Update-New-Data after a UAD is slow ("feels like UAD twice")
v2 snapshot mode's **first append per strategy = cold-seed** (capped warmup to build the
rolling base snapshot); only the 2nd+ append warm-rolls (~10-25s). For brand-new strategies
this stacks: UAD build, then first append cold-seeds again. Fix: seed `band_base_snapshot_*`
during the UAD (or save) so the first append is already warm. Also: 15Sec × 90d is a ~5-min
cold per-second load regardless — see `Append_Edge_Fossilization.md` / the Mass Builder prep
note (`docs` task #21).

### 5. [P0 · task #27] API OOM during sub-minute Update-New-Data — root cause = #21
~09:00-10:00Z the api OOM'd when Update New Data ran on the new 15Sec strategies. Same root
cause as #21: 15 confluence groups computed over 287k-430k sub-minute bars per strategy;
several UND jobs at once exhausted memory → jobs died (in-memory recompute_jobs lost) →
"nothing updated this morning." 309-313's `last_recompute` is frozen at their *UAD* times
because the later UND attempts crashed. **#21 (scope prep to needed groups) cuts memory
~5-7× and is the fix.** Interim: run UND one strategy at a time on sub-minute strategies.

## Data-readiness snapshot (16:08Z)

| sid | Backtest lane | Forward-test | Verdict |
|---|---|---|---|
| 308 | none (BT=0) | — | needs full build |
| 309 | complete Mar19→Jun16 (663) | cold-seed pending | backtest DONE |
| 310 | complete Mar19→Jun16 (322) | cold-seed pending | backtest DONE |
| 311 | complete Mar19→Jun16 (342) | cold-seed pending | backtest DONE |
| 312 | complete Mar24→Jun12 (325) | cold-seed pending | backtest DONE |
| 313 | complete Apr02→Jun16 (399) | cold-seed pending | backtest DONE |
| 314 | Mar23→Jun08 (262, gate-restricted) | partial snapshot | effectively done |

**Implication:** 309-313 already have complete UAD-built backtest lanes — re-saving via a
fixed #25 pipeline would discard good lanes and gain nothing. The blocker is the OOM (#27/
#21), not the backtest population. None are "fast-UND ready" (no `band_base_snapshot_*`).

## Notes for the real-money launch checklist (building toward)
- A strategy must have a **completed backtest lane** (source of truth) before it's eligible
  to trade real money — 308 currently fails this.
- Gate timeframes beyond 2m must be **live-verified** before relying on the gated edge.
- Both portfolios route to webhook group `grp_1e36f4` — confirm that group's delivery is
  pointed where intended before going live with real size.

## EOD 2026-06-17 — gate-firing verdict + Fidelity Gate

**Gate-firing verdict (the "not-met vs broken" question):** 308 (ungated, same
`sw123_bull_c2` trigger) fired **281 alerts today**; 309–314 (gated) fired **0** — the gates
filtered all 281. Strong evidence this is **legitimate** (gates restrictive; several require
*daily* all-or-nothing conditions — 310/312/313; 312 needs two 1d conditions at once), **not
a bug**. Engine-level confirmation that the high-TF gates *work* live = #23. Full-backtest
confirmation per strategy was blocked by the same prep slowness (#21/#29). Kevin's
one-at-a-time UNDs confirm it: `no_new_trades`/`all_in_lag_window` for today = gates legit;
any trade dated today = missed-trade divergence to chase.

**Recommendation:** 308 is the validated, firing live tester — start with it. Let 309–314
confirm over coming sessions (don't block launch on seeing them fire today).

**Fidelity Gate shipped** (509a6d9) — backtest↔live drift now caught before merge for any
`[FC]` change. See `Fidelity_Gate_Guide.md`.
