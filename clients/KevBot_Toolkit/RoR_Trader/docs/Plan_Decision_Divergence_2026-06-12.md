# Decision-Layer Divergence — Findings (2026-06-11 PM) + Attack Plan (2026-06-12)

With the bar layer at parity (gap healer + lens fixes, iter 0611a; ribbon 100% across
all rows), the residual live↔BT divergence is in the DECISION layer. This doc captures
today's evidence and the plan. NO code shipped for this yet (Kevin: no pushes during
fleet append / measurement windows).

## Evidence (window 14:45–18:06Z, six strategies: 302/303/301/299/277/293)

Per-edge nearest-match deltas (alert fill_ts − BT edge fill_ts):

1. **Entries are CLEAN fleet-wide.** Median 0.0s on every strategy; ≥94% within ±5s
   (302: 76/79, 301: 52/54, 299: 40/42, 293: 35/37, 277: 15/16, 303: 33/34).
2. **Signal-vs-signal exits are CLEAN.** When BOTH engines exit via the trigger,
   deltas are ~100% within ±5s, zero early (117 matched signal exits, 0 early).
3. **The dominant residual masquerades as "early stops":** 30–50% of BT *stop* exits
   pair with a live exit alert 10–50s EARLY — but the live alert is the **C-type exit
   SIGNAL** (e.g. `utv4_bear_flip`), not a stop. Anatomy (302 examples): live flips
   and exits at a better price 1–5 bars before BT shows any flip; BT rides and stops
   out later. So: **live's exit trigger fires on marginal bars where the batch
   reference does not flip.** Same class on eppv3 cross exits for the cohort.
4. **Gated strategies additionally show phantom ROUND-TRIPS** (303: 78 alert pairs vs
   37 BT trades) producing the "on for a stint / off for a stint" alert-history
   pattern Kevin observed (17:53–18:11Z gap): one marginal phantom entry desyncs the
   position state machine, suppressing/shifting several subsequent real signals until
   resync. Cascades, not independent misses.
5. **Decision-time data noise is NOT the driver:** ~92% of alert bars verify clean vs
   REST; the #57H `would_fire_post_correction` reclassifier marks only 1–2 alerts per
   strategy False in the window.

## ⚡ EVENING PROBE RESULTS (2026-06-11 ~20:30Z) — MAJOR REFRAME

Probed the 302 divergent exits end-to-end (bar_diagnostics live values vs batch
pipeline over the SAME engine-consumed cache bars):

- **Live and batch AGREE on everything at the divergent bars**: trail values match to
  ≤$0.007 (mostly exact), batch interpreter state = BEAR_FLIP at the flip bar, batch
  `trig_utv4_bear_flip` = True, and the live alert fired exactly at that bar's close.
  S2 (state drift) RULED OUT for these cases. Decision-time noise ruled out earlier
  (92% bars verified clean).
- **Therefore: in these cases the LIVE ALERT WAS CORRECT and the BACKTEST is wrong** —
  the rest_hifi BT lane's trade engine rode through a flip its own enriched data
  confirms, exiting later via stop_loss (worse price). "Phantom-heavy divergence" on
  this class is actually backtest-side misses, not live-side noise.
- Trade-level anatomy (302, 14:44–14:52Z): entries identical in all 4 trades; 2 trades
  stop-exit identically in both engines; 2 trades diverge exactly as above.
- `check_exit` (unified_engine ~2520) consumes per-bar `trigger_booleans` from the
  BT run's OWN IncrementalIndicatorEngine — so the BT lane's incremental
  `utv4_bear_flip` must have been False at bars where the batch column says True →
  **incremental-vs-batch FLIP parity gap inside the rest_hifi backtest lane** (the
  documented S1, now localized to the BT side, not live).
- Local repro snag: `run_unified_backtest` for 302 returns 0 trades in the local env
  (pack/confluence resolution differs from worker/API env — silent). Fix the local
  repro first tomorrow (suspect: user-pack incremental param resolution needs the
  same context worker.py sets up), then instrument the BT run's trigger booleans at
  the flip bars to pin the exact incremental-flip semantics bug.

## Ranked suspects

- **S1 — user-pack FLIP-state incremental-vs-batch parity gap (KNOWN, unfixed).**
  [[feedback-userpack-interpreter-live-dispatch]] (88be377) documented: "FLIP-state
  parity gap remains" for user packs. `utv4_bear_flip` / eppv3 crosses are exactly
  this class. Today's evidence is consistent with the incremental pack flipping on
  bars where the batch (shifted) computation does not.
- **S2 — path-dependent trailing-stop state drift.** UT Bot's trail is a ratchet; any
  tiny historical divergence (decision made during a miss→heal lag window, tentative
  forming-bar state, correction ordering) persists in the stop level until the next
  trend reset → marginal crosses resolve earlier live. Asymmetric early-only bias fits.
- **S3 — forming-bar/grace evaluation of C-type triggers** (`fire_on_partial_bucket`
  at strategy grace): partial-bucket close ≠ final close on marginal bars. Would
  produce ≤1-bar-early fires; observed spread (1–5 bars) says S3 alone is not enough.
- **S4 — gate interplay on phantom entries** (gated cohort only): same marginal-cross
  class on entry trigger × 2m gate state.

## Plan (2026-06-12)

1. **Decision ledger probe (measurement first, no engine changes).** New
   `_flip_divergence_probe.py`: for each live exit-signal alert lacking a BT signal
   exit within ±2 bars, pull the live engine's per-bar values from `bar_diagnostics`
   (#57G sources 'live'/'live_corrected' — declared diag columns incl. utv4 trailing
   stop) and the batch values from the parity-clean cache pipeline; diff the inputs
   (trail stop level, prev close, ATR) at the divergent bar. Output classifies each
   case S1 (batch-vs-incremental semantics) vs S2 (state drift magnitude) vs S3
   (alert stamped at grace-fire on partial bucket).
2. **Quantify S2 drift:** per-bar |live trail − batch trail| time series from
   bar_diagnostics for 302/303 — if drift grows between corrections and resets on
   recompute, S2 confirmed and bounded-recompute cadence is the lever.
3. **S3 split:** bucket divergent alerts by fill_ts offset within their bar (boundary
   +grace vs intra-bar) — grace-fires on partial buckets are suppressible for C-type
   EXITS specifically (entries already verified clean) if they dominate.
4. **Fix order:** S1 pack semantics fix (or gate-on-TREND workaround per the memory
   note) → S2 re-anchor cadence → S3 C-type partial suppression. One change at a
   time, each verified via the probe + a clean RTH window before the next.
5. **Gated round-trips re-measure AFTER the exit-signal fix** — phantom entries on
   gated strategies are partly the same marginal-cross class; fix may collapse the
   stint pattern on 303. Then re-run Gate Parity analysis card for what remains.

## Open follow-ups parked from today (iter 0611a)

- dashboard.py:53 null-kpis guard (EOD push, with this doc + iteration log).
- GATE_DIAG log flood strip (post-confirmation).
- apply_rest_correction shadow branch doesn't re-derive confluence records.
- Pre-existing warmup-vs-update_bar `*_prev` divergence (one-bar transient after
  full replays).
- Full-history flat-row cleanup (159k rows, only last-3d cleaned); 267 TSLA not cleaned.
