# Design — Gate-Fidelity Hardening (Kevin's 2026-07-09 ideas + assessment)

## The recurring pattern (Kevin's observation, confirmed)
Nearly every divergence bug this month is a CONFLUENCE-GATE bug: 313 (4H session
contamination), 329 (1M own_records fidelity), 272 (10m stoch freeze), 338 (session
mismatch), 325/330/331 (fossilized 4h), 310/340 (interp-blind topology). The common
root: **live and backtest construct/serve the same secondary-TF gate state via TWO
DIFFERENT PATHS**, so they can diverge, and we fix each divergence as it surfaces.
Today's fixes (#49 coarse-RTH reload, #38/329 shadow coverage) are CLASS-level +
kill-switched — but they are still PATCHES on a dual-path architecture.

## The structural fix = COLLAPSE THE TWO PATHS (Kevin's idea #1, reframed)
Idea #1 ("pre-resample & store coarse bars like the 1Sec/1Min bar_cache") — its real
value is NOT resample COST, it's **eliminating divergence-by-construction**:
- Gates evaluate on the PREVIOUS COMPLETED coarse bar (already settled).
- If BOTH lanes read that completed coarse bar from ONE canonical resampled store
  (3m/5m/15m/30m/1h/4h/1d…, RTH-and-session-keyed, built once from 1Min), then live's
  gate state == backtest's gate state BY CONSTRUCTION. 313's session contamination
  becomes IMPOSSIBLE (no live-built vs backtest-built 4H bar to disagree).
- This is the M-RS2 (Shared Bar Store) natural extension: bar_cache phase-1 = 1Sec/1Min;
  phase-2 = canonical resampled coarse store, session-keyed, both lanes read it.
- Nuance: live REAL-TIME still needs a forming-bar for chart/tip, but GATES don't (they
  use completed bars) — so the store fully covers the gate-fidelity class.
- Effort: weeks; fidelity risk (store must be byte-identical to on-the-fly resample,
  gated by the parity suite). Right LONG-TERM hardening; not a today item.
- Admin page (Kevin's framing): a "Resampled Bar Store" admin view (like Bar Cache) to
  inspect/manage the canonical coarse bars per (sym, tf, session) + freshness.

## Near-term observability (Kevin's idea #2, HIGH value / lower effort)
On the Strategy Detail "Confluence/Gate Parity" tab (just fixed in #48), add PER-GATE
CHARTS: render each gate's indicator on its NATIVE TF (e.g. the 4H STRAT_ASSISTANT with
4H candles + the INSIDE/TWO_DOWN classification), live-lane vs backtest-lane side by side.
Then a session-contaminated coarse bar (313's phantom 20:00 AH bucket) is VISIBLE at a
glance instead of needing a probe. This is the visual complement to #48's per-gate table
— turns "diagnose with an agent" into "see it on the page." Medium effort; ships the
"make bugs visible faster" goal. Candidate next after the current bug-hunt batch.

## Honest whack-a-mole vs hardening assessment (for Kevin)
- Today's fixes ARE class fixes with kill-switches (not pure instance patches) — each
  generalizes to N strategies and can't be reintroduced silently (flag + gate-parity panel
  now surfaces the class).
- BUT the ARCHITECTURE is dual-path, so new gate divergences remain POSSIBLE until the
  paths are unified. The end-state that stops the whack-a-mole is idea #1 (single canonical
  gate-bar source). Until then: each class we fix is one fewer way the two paths can differ,
  and the fleet gate-parity tool (#48) + a gate-fidelity CI check catch regressions early.
- RECOMMENDATION: finish the current bug-hunt batch (get the 21 tradeable) → ship idea #2
  (visual gate charts) as the observability layer → then scope idea #1 as the M-RS2-phase-2
  structural fix that retires the class. That sequence gets tradeable NOW and hardened NEXT.
