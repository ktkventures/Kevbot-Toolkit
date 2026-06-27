# Design: TradingView Export Readiness

**Status:** Draft · 2026-06-26
**Owner:** TV-export workstream (branch `feature/tv-export-emitter-coverage`)
**Related:** `src/pine_generator.py`, `src/_tv_emitter_audit.py`, [[project_tradingview_export]]

---

## 1. Why this doc exists

We can now generate copy-paste Pine Script from a RoR strategy (Strategy Detail →
TradingView tab) and have validated it is **trade-for-trade faithful** to the RoR
backtest on clean data (sid 308 TSLA 15Sec: 178/178 entries, exits and intrabar
stop fills matched to the cent). That makes TV a viable live-trading route while
the in-app live model's divergences are worked out.

The open question is **scalability**. The Pine generator is a *hand-written port*
of the engine — a second implementation in a second language. Today it covers
64/68 strategies. But every genuinely-new indicator pack or new stop/exit method
needs a new emitter, and our AI pack-creation pipeline will keep minting new
packs. Without structure, that becomes an endless stream of patches discovered
late (at export time) and of uncertain faithfulness.

This doc defines the **readiness contract** and the **visibility + enforcement**
layer that makes the per-pack cost bounded, checked, and validated — instead of a
silent backlog.

---

## 2. The core constraint: two implementations

Strategy behavior lives in **two** places that must agree:

| | Language | Drives |
|---|---|---|
| **Engine** (`unified_engine.py`, packs, stop/time-exit modules) | Python | In-app backtest + live trading |
| **Pine emitter** (`pine_generator.py`) | Pine v6 (generated) | TradingView export |

The Pine is **not generated from** the engine — it re-implements the same logic by
hand. This is the root cause of both:
- the **recurring per-pack cost** (a new pack has no Pine port until someone writes
  one), and
- **divergence bugs** (e.g. the EOD re-entry churn: the Pine emitted fine and was
  still wrong, because nothing checked it against the engine).

**Design stance:** we are *not* trying to eliminate the hand-written port (that
would mean single-source codegen — see §8, a future north star). We are making the
hand-work **bounded, visible, and validated**. The emitter itself is cheap
(~30–40 lines, one clear contract). The expensive parts are *discovering the gap
late* and *trusting faithfulness* — and those are what we fix here.

---

## 3. What "TV-export ready" means

A **strategy** is TV-export ready iff **every** component it uses resolves to a
faithful emitter:

1. **Entry trigger** — its pack emits the entry trigger base.
2. **Exit trigger** — its pack emits the exit trigger base, OR exit is
   `opposite_signal` (resolvable), OR there is no signal exit (stop/time only).
3. **Every confluence gate** — each gate pack emits its required state.
4. **Stop method** — a stop emitter exists for `stop_config.method`.
5. **Time exit method** — a time-exit emitter exists for `time_exit_config.method`
   (if any).

A **pack** is TV-export ready iff its emitter covers **every** trigger base and
**every** gate state the pack declares (not just the ones some strategy happens to
use today). This is the contract that was *half*-met when UT Bot shipped as a gate
emitter with no primary-trigger method — the gap that became Fix A.

### Two pack families (both must be covered)

| Family | Examples | Emitter type |
|---|---|---|
| **User packs** (candle/indicator-based) | ut_bot_v4, ema_pp_v3, macd_line_v2, swing_123 | Pack emitter: `emit_primary_setup` + `emit_trigger` + `emit_gate_function` |
| **General packs** (mechanical, often time-based) | swing/ATR/fixed/% stops; eod_exit, max_hold_bars | Stop emitter / time-exit emitter |

A strategy is only trade-ready via TV when **both** families are satisfied. The
readiness surfaces (§5) must therefore cover stop/take-profit and time-exit config,
not just the indicator packs.

---

## 4. The emitter contract

Every **pack emitter** subclasses `PineEmitter` and implements:

```
emit_primary_setup() -> str          # indicator seeding (vars, recurrence)
emit_trigger(base)   -> str | None   # edge-event expression for a trigger base
emit_gate_function(state, fn_name) -> str | None   # boolean gate function body
```

plus `slug`, `helpers` (shared Pine helpers needed), and `params` (pulled from the
pack definition, so parameter changes need **no** code change).

**Completeness rule:** an emitter must return non-`None` for *every* trigger base
and *every* gate state its pack declares. A `None` for a declared base/state is a
contract violation and should fail the readiness check (not silently drop).

**Stop emitters** and **time-exit emitters** are simpler functions (see §7) that
should be promoted to registries so the pattern is uniform with pack emitters.

---

## 5. Readiness check + UI surfaces

### 5.1 The check (one reusable function, not a script)

Promote `_tv_emitter_audit.py` into library functions:

```
tv_export_readiness(strategy) -> {ready: bool, missing: [Gap, ...]}
pack_tv_readiness(pack)       -> {ready: bool, missing: [Gap, ...]}
```

where each `Gap` is structured and human-readable, e.g.:
- `{kind: "trigger", pack: "sr_channels", base: "support_broken"}`
- `{kind: "gate", pack: "sr_channels", state: "BELOW_SUPPORT"}`
- `{kind: "stop", method: "chandelier"}`
- `{kind: "time_exit", method: "time_of_day_exit"}`

Expose via an API endpoint so the frontend can render it without re-running Python
ad hoc.

### 5.2 Three badge surfaces

1. **User-pack detail page** — `TradingView export: ✅ ready` / `⚠️ trigger
   'X' / gate 'Y' not wired`. So the moment a pack exists, you can see if it can be
   exported, and exactly what's missing.
2. **Stop/Take-Profit + Time-exit config** — same badge for the *general* pack
   family (`✅ swing / ATR / fixed / % supported`, `⚠️ no emitter for 'chandelier'`).
   These are a distinct concern from candle packs and deserve their own surface.
3. **Strategy Detail → TradingView tab** (exists) — the **rollup**: ready only if
   entry pack + exit pack + every gate + stop method + time-exit method all
   resolve. On `⚠️`, list the exact gaps so the user knows what blocks live-via-TV.

### 5.3 "Ready" means *validated*, not just *emits*

"Emits without error" is necessary but **not** sufficient — the EOD churn emitted
fine and was wrong. Each pack/method carries a **parity canary**: one clean
strategy whose TV output is checked trade-for-trade against the RoR backtest
(the sid 308 method). The badge reflects **canary status**, so `✅` means
"validated against TV," not merely "produced output." Canaries live alongside the
emitter and run in the fidelity/parity suite.

---

## 6. Enforcement at creation (AI pack pipeline)

When the AI pack-creation process mints a new pack, the pack is born
**`TV-export: pending`** and stays there until:
1. a Pine emitter exists that satisfies the completeness rule (§4), **and**
2. its parity canary passes (§5.3).

This turns "silent un-exportable pack discovered months later" into "loud at
creation." The end-state (short of full codegen): the same AI that writes the
indicator logic also writes its Pine emitter + canary in the same step, since it
already has the indicator's math in hand.

---

## 7. Structural cleanups (reduce patch-churn now)

Two low-risk changes that make the three concern-types uniform:

1. **Registries for stop + time-exit methods.** Today `_emit_stop` and
   `_emit_time_exit` are `if/elif` chains. Convert to dict registries
   (`_STOP_EMITTERS`, `_TIME_EXIT_EMITTERS`) keyed by method name, mirroring
   `EMITTERS`. Adding a method becomes "register a function," and the readiness
   check can enumerate supported methods directly from the registry keys.
2. **Formalize + self-check the contract.** Document the pack-emitter contract on
   the `PineEmitter` base class, and add a self-check that asserts an emitter
   covers every base/state its pack declares. This is what would have caught the
   UT Bot "gate but no trigger" gap automatically.

Both are engine-free (`pine_generator.py` only) and carry no conflict risk with
concurrent engine work. **Output must stay byte-identical** for already-passing
strategies (guard with the existing audit + a byte-diff on sid 303).

---

## 8. North star (not now): single-source codegen

The only way to fully erase the per-pack hand-port — and the divergence risk — is
to define each indicator/trigger/gate **once** in a declarative form and generate
*both* the engine implementation and the Pine from it. That eliminates "two
implementations that must agree" by construction.

This is a large architectural lift against an existing imperative Python engine and
is **explicitly out of scope** until the per-pack pain (with §5–§7 in place) clearly
justifies it. Captured here so the option isn't forgotten.

---

## 9. Roadmap

| Stage | Scope | Risk | Value |
|---|---|---|---|
| **1** | Readiness function + API + three badge surfaces (§5) | Low | High — gaps visible at creation, not export |
| **2** | Parity canary per pack/method; badge = validated (§5.3) | Low–med | High — faithfulness, catches divergences |
| **3** | Registries for stop/time-exit + contract self-check (§7) | Low | Med — uniform, fewer footguns |
| **4** | Enforce `TV-export: pending` in AI pack pipeline (§6) | Med | High — sustainable at scale |
| **5** | (Optional, future) single-source codegen (§8) | High | Removes divergence class entirely |

Recommended order: **3 (cleanup) → 1 (visibility) → 2 (canaries) → 4 (enforce)**.
Stage 3 is in flight on the feature branch.

---

## 10. Current state (2026-06-26)

- **Coverage:** 64/68 strategies export (`_tv_emitter_audit.py`).
- **Shipped emitters:** ut_bot_v4, ema_pp_v3/v4, ema_stack_v2, macd_line_v2,
  macd_histogram_v2, supertrend, rvol_v2, rsi_zones_2, stochastic, swing_123,
  bollinger, ml_v2, vwap_v2; stops: atr / fixed_dollar / percentage / swing;
  time exits: eod_exit, max_hold_bars.
- **Remaining gaps:** `sr_channels` (sid 292/293/174 — pack being reworked,
  deferred); `webhook_entry` (sid 7, another user — intentionally not Pine-portable).
- **Validated against TV:** ut_bot primary (sid 269), swing-stop (3/3 byte-exact),
  full strategy incl. swing entry + cross-pack exit + swing stop + eod_exit (sid 308
  TSLA 15Sec, trade-for-trade to the cent, EOD included).
- **EOD re-entry churn — FIXED in lockstep, behind a Railway flag (2026-06-26).**
  The engine used to take new entries inside the flat-by window (entering only to
  be force-flat next bar); the Pine faithfully mirrored it. Now a time-window guard
  in both `check_entry`/`check_entry_intrabar` (reusing `check_time_exit` with
  bars_held=0, so `max_hold_bars` — a duration, not a window — is unaffected) and
  the Pine (`entryGated = ... and not eodExit`, window exits only) suppresses
  entries once a be-flat window would fire. **Gated by `RORT_SUPPRESS_EOD_REENTRY`
  (default OFF = legacy churn).** Both the engine and pine_generator read the same
  flag, so generated Pine always matches the engine's *active* state — flipping the
  flag changes both together (re-export Pine after a flip). Default-OFF means the
  fix merges + ships with **zero backtest change**; enabling it **shifts `eod_exit`
  backtest P&L** (churn trades vanish) and so is a deliberate flip coordinated with
  the fidelity suite / UAD sequencing. Fidelity-parity suite is unaffected *by
  construction* either way (it measures ON==OFF + cache parity = differences, which
  an unconditional behavior change shifts equally); confirm it stays 18/18.
