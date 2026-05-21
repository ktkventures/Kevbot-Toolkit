# Spec — Pack Live Test (canary-based pack health monitoring)

**Status:** DRAFT (2026-05-21). Awaiting Kevin sign-off on §7 open
decisions before implementation.

Companion to:
- `SOP_Test_Strategy_Creation.md` — manual procedure this automates.
- The User Packs liveness metrics shipped 2026-05-21 (commit 8d21d11)
  — `last_triggered` / `triggered_7d` / `last_gated` per pack. This
  spec's status panel is built on those.

## 1. Why

`SWING_123_TEST` was silently broken as a cross-TF confluence gate
for **8+ days** — strategies gating on it fired ~0 live alerts while
the backtest computed 100-230 entries/day. It was found only by
manual investigation.

Two existing pack-validation surfaces did **not** catch it:
- **Signal Validation tab** — runs a historical backtest preview.
  Confirms the pack's interpreter produces output. The bug wasn't in
  the backtest path — backtest computed the cross-TF gate fine.
- **Parity Simulator tab** — 4-quadrant parity test with primary +
  secondary TF. But it runs a *simulated* live replay in-process; it
  does not exercise the **actual deployed live worker's** secondary-
  TF shadow path — which is exactly where the bug lived.

The gap: nothing confirms a pack works in the **real production live
engine**. This spec fills it with always-on canary strategies + a
per-pack "Live Test" tab.

## 2. Concept

For each user pack, maintain up to **two permanent canary
strategies** that exercise the pack in its two roles:

| Canary | Role | Confirms |
|---|---|---|
| `{slug}` trigger canary | pack used as the **entry trigger** | the pack's trigger fires live alerts |
| `{slug}` gate canary | pack used as a **cross-TF confluence gate** | the pack's interpreter emits on secondary-TF shadows live |

Canaries are **not** auto-deleted (revised from the 2026-05-21 rough
plan per Kevin). They are tagged and kept alive as always-sniffing
sentinels — if a pack breaks later, the canary's alert history is
the forensic record. The user may optionally delete a canary.

Sprawl is bounded: **exactly two canaries per pack, max** — creation
is idempotent (re-running "Create canaries" links to the existing
pair rather than duplicating).

## 3. Canary construction

### 3.1 Must use the canonical creation path

The canaries are created through the **same flow a real strategy
uses** — `POST /api/strategies` (the Strategy Builder save path) —
NOT a bespoke insert. Rationale: a canary built through a different
path could pass while the real creation path is broken. The canary
is only a trustworthy diagnostic if it is indistinguishable from a
user-created strategy. The creator assembles a normal strategy
`config` dict and posts it like the Strategy Builder does.

### 3.2 Fixed parameters (for comparability)

All canaries share fixed config so their results are comparable
pack-to-pack and over time:

| Field | Value | Rationale |
|---|---|---|
| symbol | `SPY` | most liquid; always has live data |
| primary timeframe | `10Sec` | fast bars — minutes to first fire |
| direction | `LONG` | simplest |
| trading_session | `Extended Hours` | maximises observation window |
| risk / balance | platform defaults | irrelevant — profitability is not measured |

**Gate canary secondary TF — `2Min`** (per SOP gotchas):
- Must be **≥ 60s** so it routes through the `on_polygon_bar`
  cross-TF fan-out (`ralph_engine.py` ~line 2002) — the path the
  `SWING_123_TEST` bug lived on. A `< 60s` gate would exercise the
  different `on_second_bar` path and miss this class of bug.
- Must not collide with another strategy's **primary** TF on the
  same symbol hub (collision → the shadow isn't built —
  `finalize_shadow_engines` `has_real` check). `2Min` is no
  strategy's primary today; `3Min` is the fallback if that changes.

### 3.3 Trigger/state selection

- **Trigger canary** — uses the pack's most permissive / highest-
  frequency trigger so it fires often (fast signal). No confluence
  gate. If the pack declares no triggers (gate-only packs like
  `sr_channels`), the trigger canary is skipped.
- **Gate canary** — uses a **known-good, high-frequency entry
  trigger from a *different* pack** (proposed: `ut_bot_v4` bull
  flip) + the pack-under-test as a `2Min` cross-TF gate on its most
  permissive interpreter state (e.g. `NEUTRAL`, which holds most of
  the time). This isolates the variable: the entry trigger is known
  to fire; if the canary stays silent, the gate is the cause —
  exactly the sid 111 diagnostic pattern. If the pack declares no
  interpreter outputs, the gate canary is skipped.

### 3.4 Tagging

Every canary carries the tag **`pack-canary`** (Kevin's "alert
test" intent; final name is §7.1). This makes them filterable on
the Strategies page via the existing Phase 36 tag UI and lets the
worker / cleanup tooling recognise them. The canary name is
`PACKTEST · {pack name} · {trigger|gate}`.

## 4. The "Live Test" tab

A new tab on the **user-pack detail page** (alongside Signal
Validation, Parity Simulator, etc.). Three sections:

### 4.1 Canary launcher
- "Create canaries" button → creates the (up to) two canaries for
  this pack via §3. Idempotent.
- Once created: shows the two canary strategy cards with links to
  their detail pages, and a "delete canary" affordance.

### 4.2 Live status panel
Per canary, the current live health — built directly on the
liveness metrics shipped 2026-05-21:
- Trigger canary: `Triggered: 3m ago (12× today)` — green/amber/red.
- Gate canary: `Gated: 5m ago` — green/amber/red.
- A **verdict line** per canary:
  - 🟢 `Pack fires as a trigger` / `Pack gates correctly`
  - 🟡 `No fires yet — give it a few minutes` (just created)
  - 🔴 `Entry trigger fired but gate never opened — likely cross-TF
    dispatch issue` (the SWING_123_TEST signature)

### 4.3 Alert↔backtest alignment report
For each canary, compare what fired live vs what the backtest/algo
lane computed — the diagnostic Kevin asked for:
- entries: alert count vs algo-lane count (same window)
- timing: mean / p95 delta between an alert's `fill_ts` and the
  matching algo trade's `entry_fill_ts`
- a one-line read: `Alerts align within ~1.2s avg of backtest —
  healthy` or `12 algo entries, 0 alerts — divergence`.
This reuses the divergence-pairing logic already used by the
Divergence tab; scoped to the one canary.

### 4.4 Free-form issue flagging
A notes area on the tab to record any observed complication for the
pack (manual, persisted per pack). Low-priority; can be a Phase 4
nicety.

## 5. Architecture

| Component | File(s) | Work |
|---|---|---|
| Canary creator | new `src/api/services/pack_canary_service.py` | builds the two config dicts, posts via the canonical create path, stamps the tag |
| Endpoints | `src/api/routers/ai_builder.py` | `POST /user-packs/{slug}/canaries` (create), `GET /user-packs/{slug}/canaries` (status + alignment) |
| Liveness reuse | `_compute_user_pack_metrics` (already shipped) | status panel data |
| Alignment | reuse Divergence pairing in `forward_test_service` / divergence endpoint | per-canary timing delta |
| Frontend tab | `frontend/src/views/UserPacksPage.tsx` | "Live Test" tab — launcher, status panel, alignment report |

## 6. Phasing

| Phase | Scope | Effort |
|---|---|---|
| **1** | Canary creator service + create/get endpoints. Idempotent. Tag stamping. | ~3h |
| **2** | "Live Test" tab — launcher + live status panel + verdict line | ~2-3h |
| **3** | Alert↔backtest alignment report section | ~2h |
| **4** (optional) | Batch "create canaries for all packs"; free-form notes | ~2h |

Each phase ships independently. Phase 1+2 deliver the core value
(push-button confirmation a pack fires live). Phase 3 adds the
fidelity read.

## 7. Open decisions (need sign-off)

1. **Tag name** — proposed `pack-canary`. Kevin floated "alert
   test". Either works; `pack-canary` is more specific. Pick one.
2. **Gate canary's borrowed entry trigger** — proposed `ut_bot_v4`
   bull flip (high-frequency, known-good). Acceptable, or prefer a
   different known-good pack?
3. **Extended Hours session** — proposed so canaries observe the
   widest window. Confirm (vs `RTH`).
4. **Phase 4 free-form notes** — worth building, or drop it? The
   structured verdict + alignment report may be enough.

## 8. Out of scope

- Backtest-logic validation — the **Signal Validation** tab already
  covers "does the pack's interpreter produce output."
- Simulated backtest↔live parity — the **Parity Simulator** tab
  already covers that. This spec is specifically the *real deployed
  worker* confirmation those two cannot give.
- Profitability of canaries — irrelevant; canaries exist to confirm
  *firing*, not to make money.
