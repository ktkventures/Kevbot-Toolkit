# Append-Edge Fossilization — mechanics, fix, and open design questions

_Last updated: 2026-06-16_

## What "fossilization" is

The backtest/algo-history lane is built by two different write paths:

1. **Full Update All Data (UAD)** — a complete recompute of the entire history from bars.
   This is the **source of truth**; it is the same engine path the interactive backtest
   uses.
2. **Update New Data (append)** — an incremental catch-up that runs periodically.

An append does **two** things, which together cover the entire elapsed window with **no
hole**:

- **Windowed append**: computes and INSERTs all new trades from `last_recompute_until_ts`
  → now. INSERT-only (dedup never overwrites an existing row). This fills the *whole* gap
  since the last append — not just the last 120 minutes.
- **Edge-band replace** (the B2 fix): *rewrites* the trailing ~120 min
  (`[now − lag − 120, now − lag]`), because the data edge is the one place trades can come
  out **under-counted** while bars are still settling.

**A fossil** is a trade (or missing trade) that was generated **wrong at the unsettled
edge** during a *prior* append, then **frozen** by INSERT-only dedup because no later
append ever overwrote it. The edge-band replace heals the most-recent edge on every
append. But a fossil that has drifted **deeper than the 120-min replace band** — from
*repeated, frequent* edge-nibbling under the old (pre-B2) behavior — will **not** be
re-touched by a normal append. Only a **full UAD** scrubs those.

## The counterintuitive part

- **Infrequent appends are *cleaner*, not dirtier.** Fossilization is caused by *frequent*
  edge-nibbling — each append under-counting at its own raw edge and leaving residue. One
  big catch-up append computes the whole window at once against fully-settled data, so it
  introduces no deep fossils.
- The only cost of *infrequent* appends is **recency lag** — the backtest reference trails
  the live engine. Not gaps, not fossils.
- A strategy that simply **hasn't been appended since this morning** has **no** deep
  fossils. Appending it now fills the full day in a single clean pass.

## Why frequent cron is probably SAFE (with the B2 fix)

The worry: a 1-minute cron would nibble the edge constantly and compound fossils. But the
edge-band replace **rewrites the trailing 120 min every append**. A fossil can only form
if a trade's *final* settlement lands **outside** the 120-min replace band before a later
append corrects it. Polygon settlement is far faster than 120 min, so the band fully
covers it. Frequent, standardized cron appends are therefore expected to be **self-healing
and possibly cleaner** than ad-hoc manual appends (Kevin's intuition). This needs to be
**verified** before turning on a tight cron — see open questions.

## Current behavior knobs

- `_ALGO_HISTORY_LAG_MINUTES = 15` — appends only write trades older than `now − 15min`
  (avoids writing against a still-forming edge). The backtest reference is therefore always
  ≥15 min behind live.
- `_APPEND_EDGE_BAND_MINUTES = 120` — width of the rewrite band.
- `APPEND_EDGE_BAND_ENABLED` (api: **true**) — master switch for the band replace.
- `APPEND_EDGE_BAND_MODE` = `capped` | `snapshot` (api: **snapshot**) — band recompute
  method. `capped` = ~55–90s cold warmup; `snapshot` = ~10–25s warm-roll from a rolling
  lagged base snapshot. Both validated byte-identical to full warmup on samples.

## How to prove append ≡ full UAD (the A/B protocol)

To isolate "did the append match the source of truth" from elapsed-time / settlement
confounds, run them **back-to-back on the same strategy over the same window**:

1. Append the strategy (snapshot mode) → record trades.
2. **Immediately** full-UAD the same strategy → record trades.
3. Diff entry/exit timestamps + prices over the identical `[start, now]` window.

Doing the UAD hours after the append is **not** a clean diff — the two cover different
windows (more settled bars by then). Prefer a few hours into the session so the early bars
have settled, then run the two back-to-back.

## Open design questions / TODO

- **Verify frequent-cron safety** before enabling a tight cron (currently
  `ALGO_HISTORY_CRON_ENABLED=false`). Simulate N rapid appends and confirm the band-replace
  keeps the lane fossil-free.
- **Reduce the 15-min lag** (or add a fast unsettled-edge path) if we want near-real-time
  backtest reference for the **phantom-trade alert** use case (alert the user when the live
  engine is in a position the backtest never took → "you may be in a phantom trade,
  consider flattening").
- **#16: full-UAD lane-wipe guard.** A transient Polygon read-timeout during full UAD can
  delete-then-insert-nothing and wipe a lane to 0 (hit sid 303 on 2026-06-16). The append
  path has a no-wipe guard; the full-UAD path does **not** yet. Must fix before any
  unattended fleet-wide UAD.
- **One full UAD as a clean baseline.** Strategies repeatedly edge-nibbled under old
  behavior may carry deep fossils. A single full UAD floors them fossil-free; appends keep
  them clean thereafter.
