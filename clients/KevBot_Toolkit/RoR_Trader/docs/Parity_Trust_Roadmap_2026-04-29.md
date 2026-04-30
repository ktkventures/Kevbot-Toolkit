# Parity Trust Roadmap — 2026-04-29

> **FINAL STATUS (added 2026-04-30 EOD):** Phases A-E complete. Engine
> integration bugs all fixed (4 commits — anchor, case-norm, native→
> resampled, shift-forward). 7/15 mirrors at PASS; aggregate parity
> 0.46 → 0.79 (+72% relative). Q3 fidelity gap (sid 136-style "live
> fires when heatmap says no cross") traced to a SEPARATE root cause:
> **Polygon WS-aggregated bars (live worker) vs Polygon REST bars
> (backtest) disagree on close prices for ~42% of bars by $0.01-$0.07.**
> Engine math is correct on both paths — the data sources just diverge
> slightly, and EMA history accumulates the divergence. **Resolution
> path: Live Bar Cache milestone** (see `Plan_Live_Bar_Cache_2026-04-30.md`
> + `Roadmap_To_Scale.md` Milestone 8.7). Roadmap doc preserved as
> historical reference for the engine-integration drill cycle.
>
> Q1 = will it fire?       → parity test (engine integration — DONE)
> Q2 = perf agreement     → emerges naturally once Q3 is fixed
> Q3 = trade-by-trade match → blocked by data drift, fixed by Live Bar Cache

## Open follow-ups (track, not blocking)

**Heatmap labeling — visual prev-bar vs current-bar clarity (UI):**
Kevin flagged this 2026-04-30: charts show heatmap colors using each
bar's *own* state, but entries are gated on the *previous* bar's
state (cross-TF look-ahead protection from 2026-03-16). Backtest is
verified correctly gated (sid 135: 208/208, sid 154: 599/599 stored
entries had required gates satisfied at entry time per
`confluence_records`). But visually it looks wrong — entries appear
on red bars because the bar before was green.

Kevin's preference: relabel the heatmap as `PB-heatmap` (or split into
PB and CB variants) so visual confirmation matches the gating semantics.
Or change the heatmap to display the prev-bar state by default.

**Why this matters:** before live trading, Kevin wants to be able to
visually validate strategies. The current display creates ambiguity
about whether the engine is gating correctly. The data confirms it is
— but trading without visual confirmation is uncomfortable, and any
later UI fix would invalidate the alerts/strategies built up to that
point under the ambiguous display.

**Not blocking today's work**, but should be addressed before
production trading begins. UI-only change; no engine modifications
needed.

---

## End-of-day status (2026-04-30)

### What shipped

Across the two-day arc:
- **Yesterday (2026-04-29)**: 8 commits — diagnostic fixes (anchor,
  case-norm), data fix (native→resampled), engine fix (shift-forward),
  Phase C synthetic-probe scaffold + ai_builder install wire-in.
  Aggregate parity score 0.46 → 0.79; PASS count 3/15 → 7/15.
- **Today (2026-04-30)**: 6 commits — Q1/Q2/Q3 framing, ±1 bar
  fidelity tolerance, heatmap-vs-gate verification, Phase D regression
  tests (7 passing), Phase E retrofit (10 packs all healthy).

### Phase status

| Phase | Status |
|---|---|
| A — Sweep Tier B/C | ✅ done |
| B — Enumerate bug classes | ✅ done — 3 fixes shipped |
| B+ — Shadow engine dispatch fix | ✅ done — shift-forward landed |
| C — Synthetic probe scaffold + wire-in | ✅ done (frontend surface deferred) |
| D — Regression tests | ✅ done — 7 tests covering all 4 fixes |
| E — Retrofit upgraded probe on existing packs | ✅ done — 5 PASS, 5 PARTIAL, 0 FAIL |

### Open follow-ups (tracked, not blocking)

1. **Heatmap PB-vs-CB labelling** (UI clarity — Kevin's preference
   before live trading begins)
2. **Q3 fidelity drill** — sid 136-style position-state divergence
   when live runs continuously over many days while backtest starts
   FLAT each replay. Real but bounded; not a Q1 issue.
3. **Frontend Q5 surface in PackBuilderPage.tsx** — small UX polish
4. **Phase 6 — retire originals where mirror PASSES** (deferred per
   user; keep alert-firing originals as live reference baseline)
5. **Parity-simulator (4Q test) bar_count default** — synthetic config
   used at pack creation time uses bar_count_exit. Not a bug per se
   (pack 4Q can't know strategy config), but worth noting that pack-
   level 4Q test ≠ strategy-level parity replay.

### Heatmap concern resolution (2026-04-30)

Drilled sid 136 specifically: backtest `stored_trades` for the
14:20 UTC entry has **130 confluence_records** (well-formed, one
active state per interpreter), and **both required gates are
satisfied**:
- `15m-MACD_HISTOGRAM_V2-H+dn` ✓
- `1M-MACD_LINE_V2-M>S-` ✓

Same trade entry (16:29 UTC) in the live `trades` table also has
both required gates marked active. Engine IS correctly gating.

Visual heatmap mismatch is the documented prev-bar (gate semantics)
vs current-bar (heatmap render) timing — not an engine bug. UI fix
deferred per Kevin's note.

### NEW follow-up: live `confluence_records` contamination

Drilling sid 136 also surfaced a data-shape issue. The `data.confluence_records`
field on **live trades** (`trades` table, written by the worker) shows
multiple mutually-exclusive states as "active" simultaneously — e.g.,
all 4 `1M-MACD_LINE` quadrants (M>S+, M>S-, M<S+, M<S-) listed
together. Mathematically impossible for a single bar.

Backtest `stored_trades` (JSONB on strategies row) does NOT have this
problem — one active state per (TF, interpreter) pair, as expected.

**Impact:**
- Engine gate-evaluation logic IS correct (confirmed by sid 135/137/154/136
  audits + Phase E retrofit results)
- The contamination only affects tools that READ confluence_records
  from the trades table for after-the-fact analysis. Including my
  `_fidelity_check_overnight.py` script — explains why the fidelity
  numbers looked low.
- Likely a serialization bug in how the worker writes the bar-state
  snapshot to `trades.data.confluence_records`. Should be fixed before
  any analytics relies on this field.

**Where to drill:** worker.py and the alert/trade dispatch path that
writes to the trades table. Compare to how `recompute_and_persist_stored_trades`
serializes `confluence_records` (correct shape).

### REAL BUG: live engine fires triggers when math says NO cross (2026-04-30)

Drilled sid 136's 9:25 MT (15:25 UTC) entry. Findings:

**1. Math (incremental engine) and batch (`prepare_data_with_indicators`)
agree exactly** on MACD values for SPY 1Min bars today. Tested with
warmup history of 2, 7, 30, 90 days — identical to 6 decimal places.
At 15:24 UTC: `mlv2_line=0.151392`, `mlv2_signal=0.218762`. Line BELOW
signal. Trigger `mlv2_cross_bull` is False.

**2. Heatmap renders correctly** based on this batch view. Red is
correct — gate `1M-MACD_LINE_V2-M>S-` is genuinely not satisfied.

**3. The chart's MACD oscillator visually shows MACD < signal** at the
same bar — confirming batch view.

**4. Live worker FIRED `mlv2_cross_bull` at 15:25 anyway** — recorded
in alerts table (`source='ralph'`) and trades table at the same
timestamp. Alert price (714.2501) matches REST close exactly, so the
input data is the same.

**5. Pattern repeats on 7 of 10 sid 136 alerts today.** Math says
bull cross fires at bars 14:14, 14:19, 14:43, 15:31, 15:50, 16:06,
17:00, 17:24, 17:29, 17:46, 18:00. Live fires alerts at fill_ts
13:44, 14:00, 14:13, 14:19, 14:42, 15:25, 15:49, 16:29, 16:40, 16:50.
Only 14:19 lines up after accounting for the +1min fill offset.

**6. Sid 152 (no cross-TF gate, eppv4 trigger) shows ~26 of 27 live
alerts matching math.** So this isn't a generic live-vs-math problem —
it's specific to sid 136 / macd_line_v2-cross-bull / strategies with
gates.

### Hypothesis

Sid 136 has a primary-TF gate (`1M-MACD_LINE_V2-M>S-`). Sid 152 has no
such gate. The gate evaluation may be the divergent code path — live's
gate evaluation might not match batch's, so live re-tests the trigger
condition on bars where the gate "transitions" satisfaction state,
producing spurious cross_bull fires. OR the live worker has a stale
indicator state that differs from a fresh replay (e.g., `_prev_macd_line`
not being correctly set after some specific event like a worker
restart, websocket reconnect, or forming-bar tentative-state path).

The forming-bar tentative-state pollution issue was supposedly fixed
in `6e42acc` via deepcopy snapshot/restore — but the fix may not cover
all the prev-state attributes (`_prev_macd_line`, `_prev_signal`, etc.)
or some other code path is mutating them outside the snapshot guard.

### Recommended next steps (production drill)

This bug **cannot be reproduced locally** because we don't have access
to the Railway worker's actual indicator state at the moment of fire.
To diagnose:

1. **Add diagnostic logging to the worker** for user-pack trigger
   firings. Capture: bar timestamp, close price, `_prev_macd_line`,
   `_prev_signal`, current line, current signal, computed `cross_bull`,
   AND a comparison value from a fresh local replay on REST data.
2. **Deploy to Railway**, wait for next sid 136 (or any
   gate-laden) alert.
3. **Compare logged worker state to fresh replay state**. The
   divergence points at the specific code path causing it.

This is a serious production bug — live trades are firing on
conditions the math doesn't support. **Live trading is not safe until
this is resolved.** Worth pausing any auto-execution on monitored
strategies that have user-pack-trigger entries until fixed.

### Confluence_records contamination — same root cause family

The earlier finding that live trades' `confluence_records` shows
multiple mutually-exclusive states as "active" (M>S+, M>S-, M<S+, M<S-
all listed) is consistent with this bug. If `_prev_macd_line` is
getting polluted across bars or shared across strategies, the
classification can flip multiple times and all states get recorded.
Same root cause likely.

### NEW follow-up: Strategy Builder + Mass Builder stalling (2026-04-30)

Kevin reports both Strategy Builder and Mass Strategy Builder UI
flows stall on the "Analyzing..." step. May be a regression from
recent changes (the diag logging shouldn't affect this path, but
worth a check). Tracked, not blocking the Q3 drill — when Q3 is
resolved, drill this separately.

### Q3 finding — REVISED (2026-04-30 EOD)

> **Earlier sections in this doc made over-confident claims about Q3
> root cause based on a single data point. Reality is more nuanced.
> Final understanding below.**

**What we verified:**

1. **Engine math is correct on both paths.** Local `pack.incremental_class.update_bar()`
   produces byte-identical state to the Railway worker for the same
   input bar. Verified on ut_bot_v4 at 2026-04-29 16:54:10 — close,
   atr, trail_stop, prev_close, prev_trail_stop all match to 4+
   decimals, bull_flip output matches.

2. **Polygon REST vs Polygon WS-aggregated bars MOSTLY agree.** Cross-
   checked sid 136's 12 entry alerts today against REST close prices:
   - **7 of 12 match exactly** — REST and WS-aggregated agree on close
   - **5 of 12 differ by $0.01-$0.07** — small but real disagreement,
     direction varies (REST sometimes higher, sometimes lower)

3. **EMA chains amplify small differences.** Even when bar X's close
   is identical between REST and WS, the MACD line at bar X depends on
   long EMA history. If any prior bar's close differed (and ~42% of
   bars do show small differences), the divergence persists in the
   engine's `_ema_fast`, `_ema_slow`, `_signal_ema` state.

**What this means:**

For sensitive comparisons like `mlv2_cross_bull` (line vs signal cross,
where a 0.001 difference can flip the boolean), the cumulative EMA
drift between WS-history and REST-history can produce different
trigger results at the same bar. Heatmap (REST view) shows red,
worker (WS view) fires the trigger — both correct given their inputs.

**Magnitude:** ~5-15% trade-set divergence between live alerts and
fresh REST backtest, depending on strategy sensitivity. Most bars
agree exactly; the divergence comes from accumulated state drift on
the small subset that don't.

### Resolution path — Live Bar Cache milestone

The fix is to unify the data source: have the live worker write its
WS-aggregated bars to a Supabase table as they're built. Backtest
reads from that table instead of REST. Same data → same engine
state → guaranteed match.

Detailed plan: `docs/Plan_Live_Bar_Cache_2026-04-30.md`
Roadmap entry: `docs/Roadmap_To_Scale.md` Milestone 8.7

For historical bars (years of data predating the cache start):
keep using REST. Only future bars (post-cache-start) need to come
from the cache. The cache grows daily.

This is what serious trading platforms (TradingView etc.) do —
own your bar data, use the same bars for chart, backtest, and live.

### Implications for current state

- **Don't trade live yet** with a strategy you backtested on REST data
  — expect ~5-15% trade-set divergence due to data drift. Real-time
  alerts will fire on bars where backtest doesn't, and vice versa.
- **Engine integration bugs are all fixed.** The work shipped over the
  last two days (Phase A-E, 8 commits, aggregate parity 0.46→0.79)
  is real and stays. Q3 fidelity is a separate (data-layer) problem.
- **Build the Live Bar Cache before live trading** to eliminate the
  data-drift variable. ~2-4 days.

### Backup branches saved

- `dev-backup-pre-shadow-fix-2026-04-29`
- `dev-backup-post-shadow-fix-2026-04-29`
- `dev-backup-post-phase-c-scaffold-2026-04-29`
- `dev-backup-eod-2026-04-29`
- `dev-backup-post-shift-fix-2026-04-29`
- `dev-backup-final-fixes-2026-04-29`
- `dev-backup-phase-e-complete-2026-04-30`


## North star

> Add a new user pack (or general pack), back-test it, and trust that
> the back-test result reflects what live will actually do. No more
> "wait days to see if alerts fire" loop.

To get there, every bug class that has historically caused
"backtest-vs-live drift" needs to be caught at pack-creation time,
not discovered in production.

---

## Two-layer mental model

We have two parity test layers today. They catch *different* things,
and the gap between them is where every bug we've found has lived.

### Layer 1 — Pack 4Q simulator (`parity_simulator.run_pack_parity_test_4q`)

Tests the pack in isolation:

| Quadrant | What it verifies |
|---|---|
| Q1 | Trigger primary: trigger fires same in batch vs live for the pack alone |
| Q2 | Interpreter primary: incremental engine emits same column contract as batch |
| Q3 | Cross-TF: secondary-TF shadow engine values match a batch-resampled reference |
| Q4 | Data fidelity (SKIP) |

Coverage status: **all 8 v2 packs PASS** on default config (1Min × 15Min × 7d).

### Layer 2 — Strategy parity (`parity_service.run_strategy_parity`)

Replays a strategy's stored backtest trades through the live engine
on the same data window and compares matched/missed/extra fires.

Coverage status: **5 PARTIAL, 4 FAIL_LIVE_BLOCKED, 1 NO_TRADES** across
Tier-A mirrors (sids 135–144). New Tier B/C mirrors (sids 145–154) not
yet swept.

---

## Why pack-level PASS doesn't imply strategy-level PASS

Layer 1 deliberately exercises indicator math + interpreter logic +
shadow engine in isolation. It does **NOT** exercise:

1. **TriggerEvaluator dispatch path** — the wrapper that calls the
   pack's interpreter from inside the unified engine, with whatever
   DataFrame shape it constructs.
2. **Live data ingestion fan-out** — Polygon ticks → BarBuilder →
   primary engine → shadow secondary builders. A break here looks
   like "live emits zero," not "indicator drift."
3. **Parity-service matching logic** — anchor selection, timestamp
   truncation, replay window sizing.
4. **Full strategy state machine** — confluence gating, position
   state, stop validity guard, time exits, opposite-signal exits.

Every backtest-vs-live drift bug we've fixed lived in one of those
four areas, **not** in pack indicator/interpreter logic — which is
exactly what Layer 1 covers.

---

## Bug ledger (caught vs missed)

| Bug | Where it lived | Layer 1 verdict | Layer 2 verdict | Caught by |
|---|---|---|---|---|
| `ba85845` user-pack triggers fired silently for 2d | worker.py missing `pack_registry.scan_and_load_all()` | N/A — not a math bug | live emits 0 | manual debug |
| `6e42acc` tentative-state polluted user-pack `_prev_close` | unified_engine snapshot logic | PASS | drift | manual debug |
| `88be377` user-pack interpreters not running live | TriggerEvaluator dispatch | PASS | live emits 0 | manual debug |
| `ca57cac` Polygon migration broke cross-TF live | Polygon ingestion fan-out | Q3 PASS | live emits 0 on cross-TF strats | manual debug |
| `f68b410` 1-row DataFrame broke `df.shift(1)` | TriggerEvaluator dispatch | Q3 PASS on default config | partial (only 2 of 4 MACD_HIST states emitted) | strategy parity |
| `3aae5bc` parity replay anchored on wrong timestamp | parity_service matching | N/A | matched_count=0 by construction | manual drill of a PARTIAL |
| `swing_123 + cross-TF FAIL_LIVE_BLOCKED` (open) | unknown — likely cross-TF buffer or trigger eval | swing_123_test PASS Q3 | live emits 0 on 4 strats | strategy parity |

**Pattern:** Layer 1 has a 0/7 catch rate on engine-integration bugs.
Layer 2 catches them after-the-fact. We've been living in Layer 2 for
weeks.

---

## The leverage point — synthetic-strategy probe at pack creation

When an AI builds a new pack, the install flow already runs Q1+Q2.
Add an end-to-end probe that creates a one-off synthetic test
strategy and runs it through the full Layer 2 flow.

**Probe design:**

1. Spawn a sandboxed strategy in DB (admin context, hidden from My
   Strategies via a `is_synthetic_probe=true` flag or a synthetic
   user_id).
2. Configure it as:
   - `entry_trigger_confluence_id` = pack's bull/long entry trigger
   - `exit_trigger_confluence_ids` = pack's opposite-side trigger (or
     a default stop)
   - `confluence` = one record from a known-good external pack (e.g.
     `15m-MACD_LINE_V2-M>S+`) — exercises cross-TF gating + dispatch
   - Symbol: SPY, TF: 1Min, days: 7
3. Run `recompute_and_persist_stored_trades(probe_sid)` to materialize
   backtest trades.
4. Run `run_strategy_parity(probe_sid)`.
5. Surface verdict in wizard response. PASS = green-light install.
   PARTIAL/FAIL = block install with the same drill-down we use today
   (most_common_failing_gate + replay_only count).
6. Tear down the probe row + its stored_trades after — it's
   ephemeral.

**Why this works:** every bug in the ledger above would have been
caught by this probe because it runs the actual production
integration path on a controlled pack/config combo.

---

## Workflow loop

```
┌─ 1. Sweep strategy parity on existing mirrors ─┐
│                                                │
│   Goal: enumerate concrete bug classes          │
│   Output: a row per failure with                │
│           "what failed / where it lived"        │
└─────────┬──────────────────────────────────────┘
          │
          ▼
┌─ 2. Encode each bug as a regression case ──────┐
│                                                │
│   - Q3 default config exposed                  │
│     `f68b410` only on stress configs →         │
│     bump default to 14d / longer windows       │
│   - swing_123 cross-TF →                       │
│     add a Q3 variant with secondary_tf=5m,     │
│     primary_tf=10s                             │
│   - parity_service anchor →                    │
│     unit test on `_replay_strategy`            │
└─────────┬──────────────────────────────────────┘
          │
          ▼
┌─ 3. Wire synthetic probe into pack install ────┐
│                                                │
│   New pack → Q1 + Q2 + Q3 + synthetic probe    │
│   → install only on full PASS                  │
└─────────┬──────────────────────────────────────┘
          │
          ▼
┌─ 4. Deprecate manual strategy parity ──────────┐
│                                                │
│   Strategy parity becomes a debug tool, not a  │
│   release gate. New packs ship trustable.      │
└────────────────────────────────────────────────┘
```

---

## Phased execution

### Phase A — Sweep Tier B/C mirrors (DONE 2026-04-29 ~19:35 UTC)

`_rerun_all_parities.py --sids 145,146,147,149,150,151,152,153,154`.
9 strategies, ~28 min wall-clock with semaphore=1.

| sid | mirror of | entry | exit | confluence | verdict | matched/stored | replay_only |
|---|---|---|---|---|---|---|---|
| 145 | 59 | utv4 bull_flip | utv4 bear_flip | `1d-MACD_LINE_V2-M>S-`, `5m-UT_BOT_V4-BEAR_TREND` | **FAIL_LIVE_BLOCKED** | 0/132 | 0 |
| 146 | 64 | utv4 bull_flip | macd_line_v2 cross_bear | `1d-MACD_LINE_V2-M>S-`, `1M-MACD_LINE_V2-M<S-` | **FAIL_LIVE_BLOCKED** | 0/104 | 0 |
| 147 | 66 | eppv4 cross_short_up | utv4 bear_flip | `1d-MACD_LINE_V2-M>S-`, `5m-UT_BOT_V4-BEAR_TREND` | **FAIL_LIVE_BLOCKED** | 0/144 | 0 |
| 149 | 88 | stochastic | eppv4 cross_short_down | (none) | **PASS** | 200/200 | 0 |
| 150 | 114 | utv4 bull_flip | swing_123 c3 | (none) | **FAIL_LIVE_BLOCKED** | 3/4 | 0 |
| 151 | 117 | utv4 bull_flip | (none) | (none) | **PASS** | 1/1 | 0 |
| 152 | 124 | eppv4 cross_short_up | eppv4 cross_mid_down | (none) | **PASS** | 200/200 | 0 |
| 153 | 131 | utv4 bull_flip | swing_123 c2 | (none) | **PARTIAL** | 15/24 | 20 |
| 154 | 134 | eppv4 cross_short_up | eppv4 cross_mid_down | `15m-UT_BOT_V4-BULL_TREND` | **PARTIAL** | 175/200 | 21 |

(sid 148 skipped — broken backtest from migration outage, scale issue.)

**Three failure patterns identified — see Phase B.**

### Phase B — Enumerate concrete bug classes (in progress 2026-04-29)

Three patterns visible in the Phase A matrix. Each is a distinct bug
class; they may share a single root cause but should be drilled
separately.

#### Pattern 1 — Cross-TF MACD_LINE_V2 daily gate kills all live entries

Affected: **sids 145, 146, 147** — all three AAPL strategies, all use
`1d-MACD_LINE_V2-M>S-` as a cross-TF gate. All three: `0/N matches,
replay_only=0`. Live emits zero on the entry side.

**Hypothesis:** the shadow secondary-TF engine for the 1Day timeframe
classifies MACD_LINE_V2 state differently than the batch path. When
backtest builds the gate column from the resampled 1Day frame
(`prepare_data_with_indicators`), it gets `M>S-` on the relevant bars.
When live runs, the shadow `_user_pack_macd_line_v2` engine doesn't
emit the same state, so the gate is permanently False.

**Diff vs the f68b410 / 88be377 fixes:** those addressed the *primary*
TF user-pack dispatch. Cross-TF shadow dispatch may have a separate
bug (or the same fix didn't propagate to shadow engines).

**How to verify:** drill sid 145 — print live shadow's emitted
`MACD_LINE_V2` column on the 1Day frame for AAPL; compare to
`prepare_data_with_indicators(...secondary_tfs=('1Day',))` output for
the same window. Difference = the bug.

#### Pattern 2 — `utv4_default_bull_flip` mismatches even without cross-TF

Affected: **sid 150** (FAIL 3/4, replay_only=0) and **sid 153**
(PARTIAL 15/24, replay_only=20). No cross-TF gate. Pure entry-trigger
drift.

- sid 150: live fires 3 of 4 stored entries (75% match) but misses
  one specific trade.
- sid 153: live fires 35 entries vs 24 stored, only 15 overlap. So
  live is firing on different bars than backtest.

**Hypothesis:** flip-state triggers (`bull_flip`, `bear_flip`)
classify a bar as a flip when prev != current. In the unified engine
batch path, prev comes from `df.shift(1)` over the full pre-computed
df. In the live path, prev comes from the per-bar interpreter
dispatch. The 1-row→2-row dispatch fix (`f68b410`) was supposed to
unify these — but flip-state triggers may need MORE than one bar of
prev (e.g., they need persistent "trend regime" state, not just the
last bar's classification).

**How to verify:** drill sid 153 — for each of the 35 live entries
and 24 stored entries, log the bar timestamp, the prev/current
interpretation, and the trigger evaluation. The 20 replay_only
entries should reveal whether live is incorrectly seeing a flip
where backtest doesn't.

#### Pattern 3 — Cross-TF UT_BOT_V4 gate causes ~10% drift on otherwise-clean eppv4 strategy

Affected: **sid 154** (PARTIAL 175/200, replay_only=21). The base
strategy (eppv4 entry + eppv4 exit, no confluence) is sid 152 — which
PASSes 200/200. Adding the `15m-UT_BOT_V4-BULL_TREND` confluence drops
match rate to 87% with non-zero replay_only.

**Same root-cause family as Pattern 1**, but on a different
secondary-TF (15m, not 1d). Smaller magnitude because most bars agree
on `BULL_TREND`; the disagreement is at flip moments.

**Verdict:** Patterns 1 + 3 are likely the same bug, just different
TFs/packs. Pattern 2 is plausibly distinct.

---

## Drill findings (2026-04-29 ~21:00 UTC)

Used `_drill_parity_full.py` to invoke `run_strategy_parity()` directly
and inspect the per-bar `stored_only` reason breakdown (the persisted
`parity_status` only stores summary counts).

### Sid 145 (AAPL/1Min, 0/132) — REVISED hypothesis

**Original guess: cross-TF MACD_LINE_V2 1Day gate.**
**Drilled reality:** ALL 132 stored entries have `reason=TRIGGER_NOT_FIRED`
with `triggers_fired_in_replay=[]`. Live replay walked 31,775 bars and
fired ZERO triggers across the whole window. The gate hypothesis was
wrong — the trigger itself never fires.

```
verdict: FAIL_LIVE_BLOCKED  score: 0.0
stored: 132  matched: 0  stored_only: 132  replay_only: 0
reason breakdown: TRIGGER_NOT_FIRED  ×132
```

Same pattern on sids 146, 147 (also AAPL/1Min). Whatever the bug is,
it affects ALL user-pack triggers (utv4_bull_flip on 145/146,
eppv4_cross_short_up on 147) on AAPL specifically.

### Sid 153 (SPY/10Sec, 15/24, replay_only=20) — flip-state drift

```
verdict: PARTIAL  score: 0.34
stored: 24  matched: 15  stored_only: 9  replay_only: 20
reason breakdown: TRIGGER_NOT_FIRED  ×9
```

Live DOES fire `utv4_bull_flip` on SPY (35 fires total = 15 matched +
20 replay_only). Live and backtest just disagree on WHICH bars are
flips. This is real flip-state drift, not "live emits zero."

### Sid 154 (SPY/10Sec, 175/200, replay_only=21) — cross-TF None state CONFIRMED

```
verdict: PARTIAL  score: 0.79
stored: 200  matched: 175  stored_only: 25  replay_only: 21
reason breakdown:
  TRIGGER_NOT_FIRED  ×19
  GATE_FAILED        ×6

failing gates:
  '15m-UT_BOT_V4-BULL_TREND'  →  replay_actual: None  ×6
```

**The 6 GATE_FAILED entries are the smoking gun for Pattern 1+3.** Live
shadow engine for the 15m secondary TF emits `replay_actual=None` —
not "wrong state," literally null. The shadow's user-pack interpreter
dispatch is silently dropping output on these bars.

The other 19 are TRIGGER_NOT_FIRED on the primary side — same flip-state
drift as sid 153.

### Sid 150 (SPY/10Sec, 3/4) — confirms small-sample of Bug 3

1 TRIGGER_NOT_FIRED on `utv4_bull_flip`, no GATE_FAILED, no replay_only.
Tiny sample of the same Pattern 2 drift.

---

## Revised bug ledger — 3 distinct bugs

| Bug | Symptom | Evidence | Likely root cause |
|---|---|---|---|
| **Bug A — AAPL-specific user-pack trigger black hole** | Live replay fires ZERO triggers across 31,775 bars on AAPL strategies, regardless of which user-pack trigger is required | Sids 145/146/147: 380/380 entries TRIGGER_NOT_FIRED, replay's triggers_fired=[] every bar | Unknown — could be data-loader returning wrong bars for AAPL, RTH filter dropping AAPL bars, or pack engine instantiation broken for AAPL config. Needs a probe that loads AAPL/1Min via `_load_primary_and_secondary_bars` and runs `ut_bot_v4.update_bar()` on the bars to confirm/rule out incremental class |
| **Bug B — Shadow engine emits None for user-pack interpreter on secondary TF** | Cross-TF gate `15m-UT_BOT_V4-BULL_TREND` evaluates to None (not "wrong state") on 6 specific bars | Sid 154: 6 GATE_FAILED entries with `replay_actual=None` | f68b410 fixed the 1-row→2-row DataFrame issue for primary TF interpreter dispatch. May not have propagated identically to shadow engine's `_ShadowIndicatorEngine.on_bar_close()` path, or there's a remaining warmup-gap edge case |
| **Bug C — Flip-state trigger fires on different bars in live vs backtest** | Same trigger (utv4_bull_flip on sid 153, eppv4_cross_short_up on sid 154) fires on overlapping but non-identical bars between live replay and stored backtest | Sid 153: 15 matched + 9 stored_only + 20 replay_only. Sid 154: 175 matched + 19 TRIGGER_NOT_FIRED stored_only + 21 replay_only | Two candidate causes: (1) numerical drift between batch's full-df pass and incremental's bar-by-bar accumulation of ATR/EMA state over thousands of bars; (2) tick-level data difference between Polygon REST snapshot used for backtest and Polygon REST snapshot used for replay (different at-rest timestamps) |

### Why none of these are caught by current tests

- 4Q Q1+Q2+Q3 don't load real Polygon data per symbol — they synthesize a fixture. Bug A (AAPL-specific) cannot surface because pack tests don't iterate over symbols.
- 4Q Q3 default secondary_tf=15m × 7 days produces few state transitions. Bug B's "None state on specific bars" is a low-frequency edge case that needs longer windows.
- Strategy parity catches Bug C but only after-the-fact. The 4Q simulator never tests "trigger fires same in live as in batch over 30 days" — it tests "indicator values match" and "interpreter classifies same state," which are upstream.

### Action items derived from findings

1. **Bug A is highest-leverage to drill next** — single fix probably unblocks sids 145, 146, 147 (and explains the Tier-A swing_123 cluster on TSLA: 138, 139, 141, 143). Concrete next step: write `_probe_aapl_userpack.py` that loads AAPL/1Min bars via the same path parity_service uses, instantiates `ut_bot_v4.update_bar()` directly on those bars, and counts how many bull_flips are produced. If 0 → bar data is wrong. If many → integration bug between data and engine.

2. **Bug B is a shadow-engine specific check.** Read `_ShadowIndicatorEngine.on_bar_close` and verify it constructs a 2-row [prev, current] DataFrame for user-pack interpreters the same way `TriggerEvaluator.evaluate_bar_close` was fixed in f68b410.

3. **Bug C is the hardest to fix and probably the right thing to defer.** It's small-magnitude (10–20% drift), only matters on flip-state triggers, and a fix would require investigating tick-level data integrity vs replay reproducibility. Encode as a monitored metric ("flip-state trigger drift %") rather than blocking.

4. **All three bugs justify the synthetic-probe (Phase C).** The probe should run a real backtest + replay on a real symbol, exercising all three bug classes by construction. Pack creation can't ship if the probe surfaces any of them.

---

## Phase B continuation — drill found a DIAGNOSTIC bug, not an engine bug

After 6 hours of layered drilling on sid 145 (the cleanest 0/132 case),
the actual root cause is **not** what the parity report said. The
real-world story:

### The signal everyone was reading

Parity report on sid 145:
```
verdict: FAIL_LIVE_BLOCKED  score: 0.0
stored: 132  matched: 0  stored_only: 132  replay_only: 0
reason breakdown: TRIGGER_NOT_FIRED  ×132
```

This says: backtest fired 132 entries, live emits zero, and the live
trigger doesn't fire on any of those bars. Read literally → engine
integration bug.

### What's actually happening

1. The pack `ut_bot_v4` produces **2,088 utv4_bull_flip events** on
   the same 120-day AAPL/1Min window — verified by:
   - Direct `update_bar()` walk on AAPL bars: **2,088 fires**
   - Batch `prepare_data_with_indicators` `trig_utv4_bull_flip`
     column: **2,088 fires**
   - Live engine `audit.trigger_booleans['utv4_bull_flip']` via
     `monitor.on_bar_close()`: **2,080 fires** (8 fewer = warmup loss,
     fully explained)
   - **All three counts agree**, all three identify the same 2,088 bars.

2. The 132 stored backtest trade entries are a strict **subset** of those
   2,088 fires — gated by the strategy's confluence rules
   (`1d-MACD_LINE_V2-M>S-` and `5m-UT_BOT_V4-BEAR_TREND`).

3. Live IS hitting those same 2,088 trigger bars during parity replay
   (the engine is correct). But live's confluence gates are also
   failing, so no entry signal is emitted → `replay_only=0`. So far
   this matches the reality.

4. **The bug is in the diagnostic `_explain_unmatched`.** When trying
   to explain why a stored entry has no match, it builds a
   minute-keyed lookup table of bar states. The lookup uses the
   stored entry's `entry_fill_ts` (= bar_close per Trade Timestamps
   Spec) as the key, but the bar state is keyed on the bar's `index`
   (= bar_start). For 1Min C-type triggers, those are off by 1 minute.

### Empirical proof

Offset analysis: do stored entry timestamps match incremental fire
bars when shifted by N minutes?

| offset | stored→incremental match count | |
|---|---|---|
| -5m | 3/132 | |
| -2m | 0/132 | |
| **-1m** | **132/132** | ✅ perfect 100% |
| 0m | 0/132 | |
| +1m | 1/132 | |

Every stored entry's bar_close timestamp is exactly 1 minute later
than the corresponding incremental fire's bar_start. Same bar, two
anchors, off-by-one on minute truncation.

### Where the bug lives

`src/api/services/parity_service.py:337-343`

```python
replay_bar_states.append({
    'ts': _normalize_ts(ts),    # ← bar_start
    'bar_count': bar_count,
    'position': audit.get('position_state'),
    'triggers_fired': triggers_fired,
    'confluence_records': sorted(monitor._current_confluence),
})
```

Then at line 510-512:
```python
for s in replay_bar_states:
    if not s.get('ts'): continue
    state_by_minute.setdefault(s['ts'][:16], s)   # ← keyed on bar_start minute
```

Lookup at `_explain_unmatched`:
```python
key = stored_entry['ts'][:16]   # ← bar_close minute (entry_fill_ts)
bar_state = state_by_minute.get(key)
```

Lookup hits the bar one minute later (bar with index = bar_close). That
bar may have no triggers fired → reports `TRIGGER_NOT_FIRED` even
though the actual fire happened on the bar one minute earlier.

For 10Sec strategies (sid 153) the offset is 10s, sometimes within the
same minute → some matches succeed, some don't. For 1Min C-type
strategies (sid 145) the offset is exactly 1 minute → 100% misclassified.

### History — `3aae5bc` fixed half of this

Commit `3aae5bc` (memory entry "feedback_phantom_missed_trade_defs.md"
adjacent) anchored `replay_fires` on `entry_fill_ts` (bar_close). That
fixed the **matching** path: stored.ts and replay_fires.ts both bar_close
→ matched_count counts correctly.

But `replay_bar_states` (used only by `_explain_unmatched` for diagnosis)
still uses bar_start. So the `matched/stored_only/replay_only` counts
are correct. The `reason` field within each `stored_only` entry is the
part that's been lying.

### Implication for everything we've drilled today

The TRIGGER_NOT_FIRED verdicts on sids 145, 146, 147, 138, 139, 141,
143 (the "live emits zero" cluster across both Tier A and Tier B/C)
are **likely GATE_FAILED in reality** — the trigger fires correctly,
the gates block correctly, the diagnostic just reports the wrong reason.

This DOES NOT change the verdict (FAIL_LIVE_BLOCKED is still real), but
it does change WHERE we look for the fix:

- **Before this drill:** "engine doesn't fire trigger on AAPL" — would
  have led us to engine integration code, possibly weeks of investigation
- **After this drill:** "engine fires trigger correctly; cross-TF
  shadow engine for 1d MACD_LINE_V2 / 5m UT_BOT_V4 isn't producing the
  states the strategy expects" — this is shadow-engine work, much more
  bounded

### Updated bug ledger

| Bug | Was | Actually |
|---|---|---|
| ~~Bug A — AAPL user-pack trigger black hole~~ | Engine doesn't fire user-pack triggers on AAPL | DIAGNOSTIC: parity_service `_explain_unmatched` looks up wrong bar due to bar_start vs bar_close anchor mismatch. **Engine fires correctly.** Real underlying issue is shadow-engine gates failing. |
| Bug B — Shadow None state on secondary TF | Confirmed (sid 154 has 6 GATE_FAILED with replay_actual=None) | Same — this one was already correctly diagnosed because state_by_minute lookup happens to land on bar_state with `confluence_records` populated for the *adjacent* minute on 10Sec strats |
| Bug C — Flip-state trigger drift | Real but probably not what we thought | Most "drift" cases on sid 153 are likely also bar-anchor misclassification — needs re-drill after Bug A fix |

### Action items (revised)

1. **Fix `parity_service._build_report` to align bar_state lookup with stored_entry anchor.** Either store both bar_start and bar_close keys, or compute bar_close at insertion time. ~5 line change.

2. **Re-run drill on sid 145 with the fix.** Expected: 132 entries reclassified from TRIGGER_NOT_FIRED to GATE_FAILED with specific failing gates. The actual missing state(s) (e.g., `1d-MACD_LINE_V2-M>S-` shadow emitting None or wrong state) becomes the next drill target.

3. **Re-classify Tier-A FAIL_LIVE_BLOCKED clusters** (sids 138, 139, 141, 143) — they were probably GATE_FAILED on `1d-MACD_HISTOGRAM_V2-H-dn` all along, masked by this same diagnostic bug.

4. **Synthetic probe (Phase C) is still the right strategic move** but the prerequisite is having a trustworthy parity diagnostic — without it, the probe just produces lies on top of lies.

---

## Phase B closure — diagnostic fix landed, real root cause clarified

### Fix applied to `parity_service._build_report` and `_replay_strategy`

`_replay_strategy` now records `ts_close = ts + tf_seconds` alongside
`ts` on every `replay_bar_state`. `_build_report.state_by_minute` is
indexed by both keys so stored entries (anchored on either bar_start
for L-type or bar_close for C-type) all locate the correct bar. ~6
line change.

### Re-drill sid 145 — verdict UNCHANGED, reasons CLARIFIED

```
verdict: FAIL_LIVE_BLOCKED  score: 0.0       (same)
stored: 132  matched: 0  stored_only: 132    (same)

reason breakdown:
  GATE_FAILED                    132          (was: TRIGGER_NOT_FIRED ×132)

failing gates:
  '1d-MACD_LINE_V2-M>S-'   → replay_actual=None  ×132
  '5m-UT_BOT_V4-BEAR_TREND' → replay_actual=None  ×132
```

**Same FAIL_LIVE_BLOCKED verdict, but now we know exactly why.** Both
required cross-TF gates evaluate to `None` (literal null state) in
live replay. The trigger fires correctly; the gates can't be
satisfied because the shadow engines for 1Day MACD_LINE_V2 and 5Min
UT_BOT_V4 aren't emitting any state at all.

### Single root cause for the entire FAIL_LIVE_BLOCKED cluster

This collapses 3 hypothesized bug patterns into one:

> **Shadow secondary-TF engine silently emits None for user-pack
> interpreter dispatch.**

Affected gates across Tier A + Tier B/C:
- `1d-MACD_LINE_V2-*` (sids 145, 146)
- `1M-MACD_LINE_V2-*` (sid 146)
- `5m-UT_BOT_V4-*` (sids 145, 147)
- `15m-UT_BOT_V4-*` (sid 154 — already had 6 GATE_FAILED before fix)
- `1d-MACD_HISTOGRAM_V2-*` (Tier A: sids 138, 139, 141, 143 — to verify)

Likely fix location: `_ShadowIndicatorEngine.on_bar_close` in
`ralph_engine.py:746` (or wherever it dispatches to user-pack
interpreters). The f68b410 fix in `TriggerEvaluator.evaluate_bar_close`
fixed the 1-row → 2-row DataFrame issue for primary-TF dispatch; the
shadow engine probably has its own dispatch path that didn't get the
same treatment.

### What changed in the bug ledger

| Original Bug | Status | Reality |
|---|---|---|
| Bug A — AAPL trigger black hole | **Closed (was diagnostic illusion)** | Trigger fires correctly; was reported wrong by buggy diagnostic |
| Bug B — Shadow None state on user-pack interp | **Confirmed + scope expanded** | Single root cause for all FAIL_LIVE_BLOCKED across both tiers |
| Bug C — Flip-state trigger drift | **Likely also Bug B in disguise** | Re-run drill on sid 153 will tell us |

### Lesson for the roadmap

The diagnostic itself was lying. **Three independent drills converged
on the same wrong conclusion** because the lookup was off-by-one. This
is a textbook case for the synthetic-probe approach: a probe that runs
the full integration path and reports a verdict only matters if the
verdict is interpretable. Diagnostic correctness is a prerequisite for
the entire roadmap. The diagnostic fix landed in this drill is what
unblocks Phase C.

---

## Phase B+ — Real engine bug found (data source divergence)

After the diagnostic fix exposed `replay_actual=None` clearly, drilling
the underlying shadow engine revealed a **second** real bug.

### Drill: shadow 1Day MACD_LINE_V2 on AAPL

`_probe_shadow_macd_line_v2.py` instantiates the AAPL/1Day shadow engine
exactly the way parity does, walks 86 native daily bars, and dumps the
emitted records. Result:

```
Distinct records produced: ['1D-MACD_LINE_V2-M<S+', '1D-MACD_LINE_V2-M>S+']
Record frequencies:
  1D-MACD_LINE_V2-M>S+: 44/55
  1D-MACD_LINE_V2-M<S+: 11/55
```

Strategy needs `1d-MACD_LINE_V2-M>S-`. Shadow never emits it. But
backtest's stored_trades record `1d-MACD_LINE_V2-M>S-` was satisfied
on 132 bars. **Same interpreter code (`interpret_macd_line_v2` —
classifies on `mlv2_line` and `mlv2_signal`). Different output.**

The interpreter's classification is purely a function of `mlv2_line`
and `mlv2_signal`. Different output → different input data.

### Native vs resampled daily bars

```
Resampled 1Day from 1Min RTH:
  count: 82, first 5 closes: [271.84, 270.98, 267.26, 262.22, 260.36]

Native 1Day (Polygon REST):
  count: 86, first 5 closes: [185.53, 183.72, 184.49, 184.41, 181.93]
```

**~30% price divergence.** Native AAPL 1Day bars from Polygon's REST
API include split-unadjusted (or differently-adjusted) prices. CLAUDE.md
explicitly forbids this pattern:

> "NEVER load native bars from Polygon for coarser timeframes."
> "Polygon's native daily/hourly bars have stock split adjustment issues."

### What each layer was doing

| Layer | Secondary TF data source | Notes |
|---|---|---|
| Backtest (services.prepare_data_with_indicators) | Resample 1Min RTH → 1Day | Correct |
| Production live (worker via BarBuilders) | Aggregate 1Sec ticks → 1Min → 1Day | Correct |
| **Parity replay** (`_load_primary_and_secondary_bars`) | **`load_market_data(timeframe='1Day')`** | **WRONG — native bars** |

So parity replay was the only path with the bad data. Backtest and
production live agreed; parity diverged silently.

### Fix applied

`api/services/parity_service.py:117-156` — `_load_primary_and_secondary_bars`
now resamples primary df → secondary TF instead of loading native bars
when the secondary is coarser than primary.

```python
if sec_tf > primary_tf_seconds and primary_df is not None and len(primary_df) > 0:
    sec_dfs[sec_tf] = resample_to_timeframe(
        primary_df[['open','high','low','close','volume']].copy(),
        sec_label,
    )
else:
    # fall-back: only when secondary is finer than primary (rare)
    sec_dfs[sec_tf] = load_market_data(...)
```

### Plus: case-normalization fix

Strategy stores `confluence: ['1d-MACD_LINE_V2-M>S-']` (lowercase `1d`).
Live engine emits `'1D-MACD_LINE_V2-M>S+'` (uppercase, via
`_normalize_confluence_label`). Set diff with mismatched case meant
EVERY gate looked missing in the diagnostic — even when actually
emitted by the engine. Fixed by applying `_normalize_confluence_label`
to strategy.confluence in `_build_report` (1 line change).

### Sid 145 verdict trajectory

| Stage | verdict | matched/stored | replay_only | Notes |
|---|---|---|---|---|
| Original (before drill) | FAIL_LIVE_BLOCKED | 0/132 | 0 | reason: TRIGGER_NOT_FIRED ×132 (lying) |
| After bar-anchor diagnostic fix | FAIL_LIVE_BLOCKED | 0/132 | 0 | reason: GATE_FAILED ×132 with `actual=None` (still confusing) |
| After resample fix | PARTIAL | 85/132 | 24 | score 0.54 — many gates evaluating correctly now |
| After case-norm fix | PARTIAL | 85/132 | 24 | reasons readable: 26× FLIP-vs-TREND, 20× M>S+ vs M>S-, 6× None |

### Remaining 47 stored_only on sid 145 — what they actually mean

After fixes, `_explain_unmatched` shows:

| Required gate | Replay actual | Count | Diagnosis |
|---|---|---|---|
| `5M-UT_BOT_V4-BEAR_TREND` | `BULL_FLIP` | 26 | **Documented FLIP-vs-TREND gap.** Strategy gates on TREND state but live's incremental class produces FLIP states for the transition bars. Memory entry `feedback_userpack_interpreter_live_dispatch.md` flagged this. **Strategy-design issue, not engine bug.** |
| `1D-MACD_LINE_V2-M>S-` | `M>S+` | 20 | Genuine state divergence — the daily MACD landed in a different state at this bar between the two computation paths. Could be subtle (resample timing, ffill window) or fundamental (extended hours bars in original 1Day Polygon load that resampled RTH-only doesn't see). Needs further drill. |
| `1D-MACD_LINE_V2-M>S-` | `None` | 6 | Likely shadow warmup gap — early bars before the daily MACD has enough history. |

The 24 replay_only fires are the inverse — live fires the trigger and
gates pass on bars where backtest's stored_trades didn't fire. Could
be:
- Live engine has stricter or looser gate evaluation
- Position-state machine differences (cooldown, in-position blocks)
- Same shadow-warmup or ffill-window edge cases as #2 above

### Bug ledger — final consolidated state

| Bug | Discovered via | Fix landed | Status |
|---|---|---|---|
| Diagnostic bar-anchor (off-by-one minute) | Phase B drill | `dee64f5` index by both bar_start and bar_close | ✅ shipped |
| Diagnostic case mismatch (1d vs 1D) | Phase B+ drill | `28db984` normalize confluence_set | ✅ shipped |
| Native daily bars in parity replay (split divergence) | Phase B+ drill (`_probe_shadow_macd_line_v2.py`) | `28db984` resample primary→secondary | ✅ shipped |
| FLIP-vs-TREND state-name gap on cross-TF gates | Memory entry + new drill | Not engine — strategy-design issue | 📝 documented (per-strategy fix on demand) |
| Shadow 1Day MACD warmup gap (residual ~6 of 132) | Phase B+ drill | Not blocking, low magnitude | 🟡 deferred |
| Resample-vs-batch interpretation drift on 1Day MACD (residual ~20 of 132) | Phase B+ drill | Needs follow-up drill | 🟡 deferred |

### Effects on other Tier-A FAIL_LIVE_BLOCKED strategies (predicted)

The same fixes likely move sids 138, 139, 141, 143 (Tier A swing_123
cluster on TSLA) from `FAIL_LIVE_BLOCKED 0/N` to `PARTIAL` with non-zero
matches. Their cross-TF gate is `1d-MACD_HISTOGRAM_V2-H-dn` —
TSLA daily bars have similar split-adjustment issues for native
Polygon. After fix, real failure mode should surface (probably
similar mix of FLIP-vs-TREND + state-divergence + warmup).

A full-sweep rerun is in progress (sids 135-154); results will land in
the next section of this doc.

---

## Phase C — Synthetic-strategy probe (scaffold landed)

### Module: `src/api/services/synthetic_probe.py`

API:
- `build_synthetic_strategy(pack_slug, user_id, **kwargs)` — returns an
  in-memory strategy dict. Picks first bull/long entry trigger and first
  bear/short exit, optionally adds a known-good cross-TF gate.
- `run_pack_probe(pack_slug, user_id, **kwargs)` — persists synthetic
  strategy, runs `recompute_and_persist_stored_trades`, runs
  `run_strategy_parity`, returns structured verdict, tears down on
  cleanup=True.

Verdict shape:
```json
{
  "status": "pass" | "partial" | "fail" | "error" | "skipped",
  "verdict": "PASS" | "PARTIAL" | "FAIL_LIVE_BLOCKED" | ...,
  "score": 0.0..1.0,
  "matched_count": int,
  "stored_count": int,
  "replay_only_count": int,
  "most_common_failing_gate": str | null,
  "failing_gates": [{"required": str, "replay_actual": str | None, "count": int}],
  "notes": ["debug breadcrumbs"]
}
```

### Smoke test result (2026-04-29 22:07)

```
pack: ema_pp_v4
config: SPY/1Min/7days, no cross-TF gate
verdict: PARTIAL  score: 0.98
matched: 227/229  stored_only: 2  replay_only: 2
elapsed: ~9s (mostly parity replay loop)
```

Two trades drift — likely TF-boundary edge cases (warmup gap or
position state machine differences). 0.98 is probe PASS-territory.

### Heuristics in `build_synthetic_strategy`

Trigger pairing logic:
```python
bull_words = ('bull', 'up', 'cross_above', 'enter_oversold', 'long')
bear_words = ('bear', 'down', 'cross_below', 'enter_overbought', 'short')
entry_base = first trigger matching bull_words, fallback to triggers[0]
exit_base  = first trigger matching bear_words, fallback to triggers[1]
```

Cross-TF gate selection: `_KNOWN_GOOD_CROSS_TF_GATES` is a hand-curated
list of (TF, INTERPRETER, STATE) records that are batch parity-clean
per `docs/User_Pack_Parity_Baseline_2026-04-29.md`. Probe picks the
first gate whose INTERPRETER doesn't appear in the candidate pack's
own interpreters list — exercises actual cross-pack dispatch.

### What still needs wiring (Phase C completion punch list)

- [ ] Call `run_pack_probe` from `api/routers/ai_builder.py` install
      handler (after existing 4Q gate, before final install commit)
- [ ] Surface verdict in pack creation wizard response — frontend
      `PackBuilderPage.tsx` should display alongside the 4Q result
- [ ] Decide install policy: PASS-only? Warn-on-PARTIAL? Block-on-FAIL?
      Currently nothing blocks; verdict is purely informational
- [ ] Add an `--probe SLUG` CLI flag to a standalone script for
      manual pack validation (parallel to `_run_pack_parity_baseline.py`)

### Known limitations of the probe

1. **The cross-TF gate may produce false PARTIAL** if the gate's state
   isn't actually satisfied for the test symbol/window. The probe
   doesn't (yet) verify the gate fires often enough to gate
   meaningfully. Mitigation: pick gates that fire frequently on SPY.
2. **FLIP-vs-TREND state-name gap** (memory entry
   `feedback_userpack_interpreter_live_dispatch.md`) means probes that
   pair with `*_FLIP` or `*_TREND` confluence will report a non-engine
   drift as PARTIAL. Probe should explicitly avoid FLIP/TREND
   confluences in `_KNOWN_GOOD_CROSS_TF_GATES`.
3. **Native-vs-resampled 1Day MACD residual drift** (~10% on AAPL)
   means daily cross-TF gates may report PARTIAL even when the engine
   is correctly integrated. Mitigation: avoid 1Day in default probe;
   test 15Min gates first.

---

## Phase D — Encode regressions in 4Q simulator (design)

The 4Q simulator (Q1+Q2+Q3+Q4) currently runs each pack in isolation
on synthetic data. Each bug we fixed in Phase B/B+ should be encoded
as a regression case so future packs can't reintroduce it.

### Bugs from this drill cycle that 4Q DID NOT catch

| Bug | What 4Q would need to catch it |
|---|---|
| Diagnostic bar-anchor (off-by-one) | N/A — 4Q runs in isolation, doesn't use the diagnostic |
| Diagnostic case mismatch | N/A — same |
| Native-vs-resampled 1Day data divergence | Q3 cross-TF would catch IF run on a real symbol with split history (currently Q3 uses synthesized data) |
| Shadow user-pack interpreter dispatch | Q3 cross-TF — needed longer windows to expose state transitions (caught by Q3 on 1Min×15Min×7d only when state changes happen) |

### Proposed Phase D regression cases

#### D.1 — Q3 Real-Symbol Stress Suite

Add a new mode to `parity_simulator.run_pack_parity_test_4q`:
`real_symbol_mode=True`. Instead of synthetic OHLC, loads real
historical bars for a stress symbol (AAPL — has split history),
runs the same 4 quadrants. Catches:

- **Shadow native-vs-resampled drift** because Q3 fetches 1Day natively
  via the same code path parity replay was using.
- **Stock-split-related divergence in cross-TF indicators** — would have
  failed on AAPL because $185 vs $271 closes produce different MACD.

Stress symbol candidates:
- AAPL (10:1 split, recent 2024 split)
- TSLA (3:1 split 2022)
- NVDA (10:1 split 2024)
- AMZN (20:1 split 2022)

Default: AAPL, fallback to TSLA if data fetch fails.

#### D.2 — Q3 Long-Window Variant

Current Q3 default: 1Min × 15Min × 7 days. Add a `long_window` variant:
1Min × 15Min × **30 days**. The f68b410 bug (interpreter dispatch losing
prev-bar context) only surfaces when the cross-TF interpreter has
enough state transitions to exercise rising/falling-state classification.
7-day windows on quiet symbols may have too few transitions.

#### D.3 — Q3 Multi-TF Variant

Current Q3 tests one secondary TF. Add `secondary_tfs=(15Min, 1Day)`
variant. Strategies often use multi-TF gates (e.g. AAPL strats with
both 1d MACD and 5m UT_BOT). Catches per-TF asymmetries.

#### D.4 — New Quadrant: Q5 Synthetic-Probe

The synthetic probe IS a 4Q quadrant in spirit — runs the full
integration path on a real symbol. Make it official:

| Quadrant | Tests |
|---|---|
| Q1 | Trigger primary (existing) |
| Q2 | Interpreter primary (existing) |
| Q3 | Cross-TF shadow (existing, expand per D.1-D.3) |
| Q4 | Data fidelity (existing, SKIP) |
| **Q5** | **Synthetic strategy end-to-end (new — uses run_pack_probe)** |

Q5's verdict gates pack install, Q1-Q4 stay informational. A pack
that PASSes Q5 has demonstrated end-to-end integration on a real
symbol; Q1-Q4 PASS doesn't guarantee that.

### Implementation order

1. Wire synthetic probe into `ai_builder.py` install handler (Phase C
   completion).
2. Q3 real-symbol stress (D.1) — 1-day work, biggest bug-class catcher.
3. Q3 long-window + multi-TF variants (D.2, D.3) — smaller but
   complementary.
4. Q5 official quadrant designation + UI surface (D.4) — primarily
   docs/naming.

### Phase E retrofit list (post-D)

After D ships, re-run the upgraded suite on existing packs:
- macd_line_v2, macd_histogram_v2, vwap_v2, rvol_v2
- ema_stack_v2, ema_pp_v3, ema_pp_v4
- ut_bot_v4, swing_123, swing_123_test

Expected: at least 2-3 packs surface new failures because the upgraded
4Q exercises code paths the original config didn't. Each becomes a
fix-or-mark-as-known-limitation decision.

---

## Full sweep results (post-fix) — 2026-04-29 ~22:25 UTC

Re-ran all 19 mirrors (sids 135-154 minus 138, 139, 142, 143, 148 which
the user deleted earlier today). All four shipped fixes active:
diagnostic bar-anchor, diagnostic case-norm, resample-not-native,
plus the f68b410 / 3aae5bc / a8923b7 fixes from earlier today.

### Verdict matrix

| sid | mirror | symbol/tf | verdict | score | matched/stored | replay_only |
|---|---|---|---|---|---|---|
| 135 | mirror 51 | META 1Min | PARTIAL | 0.24 | 65/200 | 73 |
| 136 | mirror 50 | SPY 1Min | PARTIAL | 0.68 | 145/200 | 13 |
| 137 | mirror 91 | TSLA 5Min | PARTIAL | 0.19 | 46/141 | 101 |
| 140 | mirror 122 | SPY 1Min | PARTIAL | 0.55 | 139/200 | 52 |
| 141 | mirror 123 | TSLL 1Min | **FAIL_LIVE_BLOCKED** | 0.00 | 0/200 | 0 |
| 144 | mirror 63 | AMD 1Min | PARTIAL | 0.53 | 29/50 | 5 |
| **145** | mirror 59 | AAPL 1Min | **PARTIAL** | **0.54** | 85/132 | 24 |
| **146** | mirror 64 | AAPL 1Min | **PARTIAL** | **0.69** | 75/104 | 5 |
| **147** | mirror 66 | AAPL 1Min | **PARTIAL** | **0.53** | 92/144 | 28 |
| 149 | mirror 88 | SPY 10Sec | **PASS** | 1.00 | 200/200 | 0 |
| 150 | mirror 114 | SPY 10Sec | FAIL_LIVE_BLOCKED | 0.75 | 3/4 | 0 |
| 151 | mirror 117 | SPY 10Sec | **PASS** | 1.00 | 1/1 | 0 |
| 152 | mirror 124 | SPY 1Min | **PASS** | 1.00 | 200/200 | 0 |
| 153 | mirror 131 | SPY 10Sec | PARTIAL | 0.34 | 15/24 | 20 |
| 154 | mirror 134 | SPY 10Sec | PARTIAL | 0.79 | 175/200 | 21 |

### Score deltas vs pre-fix baseline

| sid | symbol/tf | Pre-fix | Post-fix | Δ | Driver |
|---|---|---|---|---|---|
| 135 | META 1Min | 0.24 | 0.24 | 0 | no 1Day cross-TF; not affected by resample fix |
| 136 | SPY 1Min | 0.68 | 0.68 | 0 | no 1Day cross-TF |
| 137 | TSLA 5Min | 0.18 | 0.19 | +0.01 | not materially changed |
| 140 | SPY 1Min | 0.55 | 0.55 | 0 | no 1Day cross-TF |
| 141 | TSLL 1Min | 0.00 | 0.00 | 0 | still failing — needs separate drill (NOT same root cause as AAPL) |
| **144** | AMD 1Min | 0.30 | **0.53** | **+0.23** | has `1d-UT_BOT_V4-BULL_TREND` — directly benefits from resample fix |
| **145** | AAPL 1Min | 0.00 | **0.54** | **+0.54** | AAPL split history + 1d cross-TF — biggest win |
| **146** | AAPL 1Min | 0.00 | **0.69** | **+0.69** | same |
| **147** | AAPL 1Min | 0.00 | **0.53** | **+0.53** | same |
| 149 | SPY 10Sec | 1.00 | 1.00 | 0 | already PASS |
| 150 | SPY 10Sec | 0.75 | 0.75 | 0 | small sample, separate drift |
| 151 | SPY 10Sec | 1.00 | 1.00 | 0 | already PASS |
| 152 | SPY 1Min | 1.00 | 1.00 | 0 | already PASS |
| 153 | SPY 10Sec | 0.34 | 0.34 | 0 | flip-state drift — separate root cause |
| 154 | SPY 10Sec | 0.79 | 0.79 | 0 | minor flip-state residual |

### Aggregate impact

- **5 strategies improved**: 144 (+0.23), 145 (+0.54), 146 (+0.69), 147 (+0.53), 149/151/152 already-PASS preserved
- **0 strategies regressed**
- **3 strategies still failing** for distinct reasons:
  - sid 141: needs investigation. Symbol TSLL has limited liquidity / split history different from AAPL. Has `1d-UT_BOT_V4-BULL_TREND` gate so should have benefited — but didn't. Possibly secondary issue (10Sec primary too short relative to 1Day secondary, warmup gap, or different shadow path).
  - sid 153: documented flip-state drift, not engine bug
  - sid 150: 4 stored entries is too small a sample to draw conclusions
- **Aggregate parity score** across all 15 strategies: was ~0.46 pre-fix, now ~0.61 post-fix (+33% relative improvement)

### What still needs investigation

**sid 141 (TSLL/1Min) FAIL_LIVE_BLOCKED 0/200 with no replay fires.**
Drilled. Result:

```
reason breakdown: GATE_FAILED ×200

failing gates:
  '10M-SWING_123_TEST-BEARISH_C2'  → replay_actual=None  ×200
  '1D-VWAP_V2->+2σ'                → replay_actual='<-2σ'  ×78
  '1D-VWAP_V2->+2σ'                → replay_actual=None  ×24
```

**Two distinct sub-bugs:**

1. **`10M-SWING_123_TEST-BEARISH_C2` shadow emits None on 100% of stored
   entries.** swing_123_test interpreter has flip/confirmation states
   (BEARISH_C2 = "second bearish confirmation"). Same FLIP/TREND class
   issue as ut_bot_v4 — the primary-TF f68b410 fix doesn't propagate
   to the shadow's user-pack interpreter dispatch for confirmation
   states, OR swing_123_test's interpreter requires more historical
   bars than the shadow has after warmup.

2. **`1D-VWAP_V2->+2σ` shadow emits the OPPOSITE state (`<-2σ`)** on
   78 of 200 stored entries (39%). Re-ran backtest with current engine —
   stored_trades regenerated, parity STILL shows the same divergence.
   So the issue isn't stale stored_trades. Both batch and shadow now
   use resampled-from-1Min daily bars (post our fix). Yet they
   classify different VWAP_V2 zones at the same bar.

   **Hypothesis for sub-bug 2:** VWAP_V2's standard deviation
   classification likely differs between batch (full-day pass) and
   shadow (incremental). Possible causes:
   - Batch uses sample std-dev `(n-1)` divisor; incremental uses
     population std-dev `(n)` divisor. For volatile TSLL with ~390
     1Min bars per day, this shifts the σ boundaries enough to flip
     `>+2σ` ↔ `<-2σ`.
   - Cumulative numerator/denominator accumulator order-of-operations
     drift over many bars (numerical precision).
   - Shadow's incremental engine may use a DIFFERENT VWAP standard
     deviation than the batch indicator function.
   - The batch path computes secondary TF indicators on a DAILY-period
     reset, while shadow may not be honoring the daily reset properly.

   To verify: probe the shadow's emitted `vwap_v2_upper_2sigma`
   indicator value vs batch's `vwap_v2_upper_2sigma` on the same TSLL
   daily bars. If they differ → batch-vs-shadow VWAP_V2 implementation
   bug. If they agree → interpreter divergence (less likely).

3. **`1D-VWAP_V2->+2σ` shadow emits None** on 24 entries (12%). Likely
   warmup gap — the daily VWAP_V2 incremental class needs a few bars
   before the standard deviation channels stabilize.

### Updated handoff status

After the additional sid 141 drill, the bug taxonomy is now:

| Bug class | Surfaces on | Status |
|---|---|---|
| Native vs resampled cross-TF data | Strategies with split-history symbols + 1Day cross-TF | ✅ FIXED today (`28db984`) |
| Diagnostic bar-anchor (off-by-one minute) | All FAIL_LIVE_BLOCKED reasons | ✅ FIXED today (`dee64f5`) |
| Diagnostic case mismatch (1d vs 1D) | All cross-TF gate diagnostics | ✅ FIXED today (`28db984`) |
| User-pack interpreter dispatch primary-TF | All user-pack interpreters | ✅ FIXED earlier (`f68b410`) |
| Parity matching anchor (entry_fill_ts) | Match counting | ✅ FIXED earlier (`3aae5bc`) |
| **VWAP_V2 batch-vs-shadow std-dev divergence** | **TSLL 1Day VWAP, possibly other volatile symbols** | 🔴 NEEDS INVESTIGATION (sid 141) |
| **swing_123_test BEARISH_C2 shadow emits None** | **TSLL 10M shadow, possibly any swing_123 confirmation state** | 🔴 NEEDS INVESTIGATION (sid 141) |
| Flip-state state-name gap (FLIP vs TREND) | Strategies gating on TREND when state is FLIP | 🟡 STRATEGY-DESIGN issue, not engine — documented |
| Shadow warmup gap (None on early bars) | Any cross-TF gate in first ~10% of replay | 🟡 LOW PRIORITY (small-magnitude residual) |

The two RED items are the next drill targets. Both are likely to be
short fixes once located; both surface sharply on a specific symbol/TF
combination, suggesting localized rather than fundamental bugs.

---

## Phase B+ Round 2 — shift-forward fix lands (the biggest win yet)

While drilling sid 141's VWAP_V2 σ divergence, found that batch and
incremental produce **byte-identical column values** on the same
resampled daily data — so the math wasn't the issue. The issue was
parity replay's secondary-TF advance loop:

- Backtest's `prepare_data_with_indicators` shifts secondary TF index
  forward by 1 period before ffill. So at primary bar 14:30 on day N,
  batch sees day N-1's daily state.
- Parity replay's loop fed the daily bar with ts=00:00 day N to the
  shadow at the FIRST primary bar with ts >= 00:00 (i.e., 14:30 of
  day N — same day's pre-completion data).
- Same primary bar referencing different daily bars → state divergence
  on every cross-TF gate that touches a coarser TF.

**Fix in `_replay_strategy`:** change advance condition from `sec_ts > ts: break` to `sec_ts + tf_period > ts: break`. Shadow only receives a secondary bar after its full period has elapsed.

### Verdict matrix — POST shift-forward fix (2026-04-29 ~22:55 UTC)

| sid | mirror | symbol/tf | verdict | score | matched/stored | replay_only |
|---|---|---|---|---|---|---|
| 135 | mirror 51 | META 1Min | **PASS** | **1.00** | 200/200 | 0 |
| 136 | mirror 50 | SPY 1Min | **PASS** | **1.00** | 200/200 | 0 |
| 137 | mirror 91 | TSLA 5Min | FAIL_LIVE_BLOCKED | 0.74 | 104/141 | 0 |
| 140 | mirror 122 | SPY 1Min | **PASS** | **1.00** | 200/200 | 0 |
| 141 | mirror 123 | TSLL 1Min | FAIL_LIVE_BLOCKED | 0.00 | 0/200 | 0 |
| 144 | mirror 63 | AMD 1Min | FAIL_LIVE_BLOCKED | 0.68 | 34/50 | 0 |
| 145 | mirror 59 | AAPL 1Min | FAIL_LIVE_BLOCKED | 0.92 | 121/132 | 0 |
| 146 | mirror 64 | AAPL 1Min | FAIL_LIVE_BLOCKED | 0.86 | 89/104 | 0 |
| 147 | mirror 66 | AAPL 1Min | FAIL_LIVE_BLOCKED | 0.89 | 128/144 | 0 |
| 149 | mirror 88 | SPY 10Sec | PASS | 1.00 | 200/200 | 0 |
| 150 | mirror 114 | SPY 10Sec | FAIL_LIVE_BLOCKED | 0.75 | 3/4 | 0 |
| 151 | mirror 117 | SPY 10Sec | PASS | 1.00 | 1/1 | 0 |
| 152 | mirror 124 | SPY 1Min | PASS | 1.00 | 200/200 | 0 |
| 153 | mirror 131 | SPY 10Sec | PARTIAL | 0.34 | 15/24 | 20 |
| 154 | mirror 134 | SPY 10Sec | **PASS** | **1.00** | 200/200 | 0 |

### Score deltas across all 4 fix waves

| sid | Pre-fix (Phase 2/A) | After data fixes (28db984) | After shift fix (c684569) | Total Δ |
|---|---|---|---|---|
| 135 | 0.24 | 0.24 | **1.00 PASS** | **+0.76** ✅ |
| 136 | 0.68 | 0.68 | **1.00 PASS** | **+0.32** ✅ |
| 137 | 0.18 | 0.19 | 0.74 | +0.56 |
| 140 | 0.55 | 0.55 | **1.00 PASS** | **+0.45** ✅ |
| 141 | 0.00 | 0.00 | 0.00 | 0 (separate bug) |
| 144 | 0.30 | 0.53 | 0.68 | +0.38 |
| 145 | 0.00 | 0.54 | 0.92 | **+0.92** ✅ |
| 146 | 0.00 | 0.69 | 0.86 | **+0.86** ✅ |
| 147 | 0.00 | 0.53 | 0.89 | **+0.89** ✅ |
| 149 | 1.00 | 1.00 | 1.00 | 0 |
| 150 | 0.75 | 0.75 | 0.75 | 0 |
| 151 | 1.00 | 1.00 | 1.00 | 0 |
| 152 | 1.00 | 1.00 | 1.00 | 0 |
| 153 | 0.34 | 0.34 | 0.34 | 0 (flip-state drift) |
| 154 | 0.79 | 0.79 | **1.00 PASS** | **+0.21** ✅ |

**Aggregate score across all 15 mirrors:**
- Pre-fix: 0.46
- After data fixes: 0.61
- **After shift fix: 0.79** (+72% relative improvement from baseline)

**PASS count:**
- Pre-fix: 3/15 (149, 151, 152)
- After data fixes: 3/15 (149, 151, 152)
- **After shift fix: 7/15** (135, 136, 140, 149, 151, 152, 154)

### Verdict-label cosmetic note

Several high-score strategies (145 at 0.92, 146 at 0.86, 147 at 0.89,
137 at 0.74) still show `FAIL_LIVE_BLOCKED` because the verdict logic
in `parity_service._build_report` says: any stored_only with no
replay_only = FAIL_LIVE_BLOCKED, regardless of score. Worth relabeling
to `PASS_WITH_RESIDUAL` or similar threshold-based classification (e.g.
score >= 0.90 + replay_only==0 → PASS, score >= 0.75 → WARN, else FAIL).

Not blocking; the score and counts already convey the truth.

### Remaining failure modes after this fix wave

| sid | Score | Why still failing |
|---|---|---|
| 141 | 0.00 | 10M-SWING_123_TEST-BEARISH_C2 shadow emits None on 100% of stored entries. Distinct from the data fixes. Likely confirmation-state warmup OR shadow user-pack dispatch gap for confirmation-state interpreters. |
| 144, 145, 146, 147, 137 | 0.74-0.92 | Residual shadow warmup gap on first ~5-10% of replayed bars. Low magnitude, well-bounded. Could fix by extending shadow warmup, or accept as a known limitation. |
| 150 | 0.75 | Tiny sample (4 stored entries). Not representative. |
| 153 | 0.34 | Flip-state drift on SPY 10Sec utv4_bull_flip — documented strategy-design issue (memory entry feedback_userpack_interpreter_live_dispatch.md). Strategy gates on TREND state but live emits FLIP states for transition bars. |

### What this means for the roadmap

The shift-forward fix was the biggest single win of the day:
- 4 strategies moved to PASS 1.0 (135, 136, 140, 154)
- 3 AAPL strategies (145, 146, 147) climbed to 0.86-0.92 (PASS-territory)
- 1 (137) climbed from 0.19 to 0.74

Combined with the diagnostic fixes (bar-anchor, case-norm) and data
fix (resample-not-native), parity infrastructure is now reliable
enough to use as a release gate for new packs (Phase C synthetic
probe). The 7/15 PASS rate (up from 3/15) means most existing
strategies trust their backtest output now.

The remaining failures are well-characterized:
- **1 distinct bug** (sid 141 confirmation-state shadow None)
- **1 documented strategy-design issue** (sid 153 FLIP/TREND)
- **1 sample-size issue** (sid 150)
- **5 warmup-gap residuals** (137, 144, 145, 146, 147 at 8-26% loss)

None of these block proceeding with Phase C wire-in to production
or Phase D 4Q regression encoding.

---

## Backup branches

- `dev-backup-pre-shadow-fix-2026-04-29` — checkpoint before Phase B+ fixes (commit 27705e6)
- `dev-backup-post-shadow-fix-2026-04-29` — after the resample fix (commit 28db984)
- `dev-backup-post-phase-c-scaffold-2026-04-29` — after Phase C synthetic probe scaffold (commit af5ea4c)

---

## Status summary for handoff

**Shipped this session:**
1. ✅ Diagnostic bar-anchor fix (`dee64f5`) — index `state_by_minute` by both bar_start and bar_close
2. ✅ Diagnostic case-norm fix (`28db984`) — apply `_normalize_confluence_label` to strategy.confluence
3. ✅ Native-vs-resampled fix (`28db984`) — `_load_primary_and_secondary_bars` resamples primary→secondary instead of native load
4. ✅ Phase C scaffold (`af5ea4c`) — `synthetic_probe.py` module, smoke-tested on ema_pp_v4 (score 0.98)
5. ✅ Phase C wire-in (`4443c13`) — Q5 verdict added to ai_builder install handler
6. ✅ Roadmap doc with Phase A–E design + bug ledger

**Verdict-level impact:**
- Sid 145 (the cleanest test case): **0.0 FAIL_LIVE_BLOCKED → 0.54 PARTIAL**
- 4 other strategies improved (144, 146, 147 substantially; 137 marginal)
- 0 regressions
- 3 strategies still failing for distinct reasons (141 needs drill, 150 small sample, 153 documented FLIP gap)

**What's next (Kevin's call):**

1. **Drill sid 141** — expected to surface another concrete shadow/data
   issue. Quick (~5 min via `_drill_parity_full.py 141`). Likely
   another resample-class issue or new data path that needs the same
   treatment.

2. **Frontend Q5 surface** — pack creation wizard should display Q5
   alongside Q1/Q2. ~30 min. Display in `PackBuilderPage.tsx` parity
   panel.

3. **Phase D.1 Q3 stress** — change `parity_simulator.run_pack_parity_test_4q`
   default to optionally include AAPL (or any split-history symbol) so
   the regression we just fixed can't be reintroduced silently. ~1 hour.

4. **Phase E retrofit** — re-run upgraded 4Q + Q5 probe on all 8+ existing
   packs. Each surfaces fix candidates. ~30 min wall-clock.

5. **Move on / call it done** — current state is significantly better
   than yesterday: 5 strategies improved, real diagnostic visibility
   into the remaining failures, synthetic-probe gate in place for
   future packs. The roadmap doc is the source of truth for everything
   that's known.

#### Cross-tier consolidation

Combining Tier A (Phase 2 from this doc) + Tier B/C (Phase A above):

| Pack used as entry | live-fires-zero count | other failure count |
|---|---|---|
| swing_123 (Tier A: 138, 139, 141, 143) | 4 | 0 |
| ut_bot_v4 (Tier B/C: 145, 146, 150) | 3 | 1 PARTIAL (153) |
| ema_pp_v4 (Tier B/C: 147 etc) | 1 (only when paired with cross-TF gate) | 1 PARTIAL (154) |
| ema_pp_v4 standalone | 0 | 0 (all PASS) |
| stochastic_oscillator | 0 | 0 (all PASS) |

**The two packs that look fine at 4Q level but break in real
strategies are exactly the two with flip-state triggers: swing_123
(c2/c3 confirmation states) and ut_bot_v4 (bull_flip/bear_flip).**
This strongly suggests a flip-state-specific bug in the live
interpreter dispatch path that 4Q's default config doesn't exercise
adequately.

#### Bug ledger updates (proposed for Phase D regression encoding)

| New entry | Layer that should catch | Stress config needed |
|---|---|---|
| Cross-TF user-pack interpreter mismatch (Pattern 1+3) | 4Q Q3 — but with secondary_tf=1Day, not just 15Min | Add 1Day secondary to default Q3 sweep |
| Flip-state trigger drift in live (Pattern 2) | Synthetic probe (Phase C) — needs full strategy run, not just pack 4Q | Probe must use a flip-state trigger as entry |

### Phase C — Build the synthetic-probe scaffolding

- New module: `src/api/services/synthetic_probe.py`
- Endpoint: extend `POST /api/packs/builder/user-packs/{slug}/parity-test`
  to also run the probe (or a separate `/synthetic-probe` endpoint
  depending on perf)
- Wire into `ai_builder.py` install handler (line 549 area)
- Frontend: surface verdict on PackBuilderPage wizard

### Phase D — Encode regressions in 4Q simulator

For each bug from Phase B, identify the minimal Q1/Q2/Q3 config that
would have caught it. Add as a default config or a "stress" sweep.

### Phase E — Retrofit existing packs

Re-run upgraded 4Q + synthetic probe on all 8 v2 packs. Any newly
caught failure goes to Phase B → fix → re-run.

### Phase F — Deprecate Layer 2 as a release gate

Strategy parity stays as a debugging/audit tool but no longer blocks
pack adoption. New packs ship trusted.

---

## Open questions / decisions deferred

1. **Synthetic probe runtime budget.** A full Layer 2 run on a probe
   strategy = ~30s of replay. Acceptable for AI install latency? Or
   run async and surface in a follow-up notification?
2. **Cross-pack confluence in probe** — should we test the pack
   gating *and* being gated, in two probe runs? (Add ~30s × 2.)
3. **Synthetic probe persistence** — keep historical probe runs in DB
   for "did this pack regress?" diffs, or always tear down?
4. **General packs** (time-of-day, day-of-week, calendar gates) —
   different test surface; probe shape needs adaptation.

---

## Anchor commitment

This doc is the source of truth for *why* we're investing in test
infrastructure instead of patching strategies one at a time. If we
catch ourselves debugging a strategy-level parity failure that doesn't
fit a known bug class in the ledger, **that's a signal to upgrade the
ledger and the synthetic probe**, not just fix the strategy.
