# Parity Trust Roadmap — 2026-04-29

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
