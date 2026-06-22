# Note — General-confluence gating tz fix + gen-gated strategies (2026-06-22)

## What was wrong
Live general-pack gating (session / time-of-day / calendar) evaluated the bar
timestamp in **UTC** instead of **ET**, so mid-session (e.g. 16:01 UTC = 12:01 ET)
was read as past the 16:00 close → `OUT_OF_SESSION` → the entry gate silently
blocked every session/time-gated strategy. Confirmed live: strategies
**318, 319, 320, 322, 323, 324, 326** (general-confluence gated) had **0 alerts**
while their ungated twin **321** fired 81×.

## Fix (deployed)
Commit **`99c25cf`**, deployed **2026-06-22 16:28 UTC**. Converts the timestamp to
America/New_York before reading hour/minute/weekday/day, in BOTH the live
(`unified_engine._eval_gp_scalar`) and backtest (`general_packs._eval_time_window`
+ day/calendar) paths — kept consistent. Validated offline: 16:01 UTC → IN_SESSION;
live state == backtest state at open/mid/close/overnight.

Also commit **`38c527e`** (frontend, no engine impact): general confluences now
shown on the My Strategies card with a `[GEN]` badge (they were hidden before — that
was a display gap, NOT data loss; 320–324 were created base/general-only by design).

## Should 318–326 fire alerts correctly now? → YES
- **Entries:** the gate is fixed → they fire during the real ET session.
- **Exits (signal / stop):** never gated by general confluences → unaffected.
- **Time-based exits (EOD 5m, etc.):** ALREADY tz-correct — `time_exit_packs.py`
  converts bar_time to US/Eastern (lines 139–172). Not affected by this bug.
- **Positions:** tracked normally once entered.
So the alerts themselves are trustworthy once the Worker finishes redeploying the
new code. (Live confirmation in progress — watching for the first 318/319/320 alert
post-deploy.)

## ⚠️ DO NOT trust the DIVERGENCE data on 318–326 until UAD re-run
The fix corrected **backtest** gen-gating too, so the **saved backtests** for these
strategies (computed under the old UTC logic) are now **stale**. Their
live-vs-backtest divergence will look noisy/wrong until their backtest lane is
recomputed. **Kevin will run Update All Data on them tonight** (deferred now to avoid
bogging the system). Until then: set 318–326 aside in the Strategy Health divergence
view, or filter them out.

## Deploy times today (for filtering divergence noise)
- **16:28 UTC** — `99c25cf` general-pack tz fix (engine; affects gen-gated strategies).
- `38c527e` — frontend display only (no trade/alert impact).
Earlier: the mass-builder fixes (a55cf95, cdcb841, 7fb84ae, 17d4801) are backtest-only.

## TODO (Kevin, tonight)
- Run **Update All Data** on 318–326 to realign their backtest lane with the
  corrected ET gating → then their divergence metrics are trustworthy again.
