# Plan — 2026-05-29 RTH validation of Phases A+B+C

## Where we are

End-of-day 2026-05-28 we shipped Phases A+B+C of the `ws_rest_spliced` refinement:
- Phase A: snapshot_state split into in-memory (default, includes user packs via deepcopy) vs persistent (with pickle probe, drops unpicklable). Unlocks `apply_last_bar_correction` to actually restore UT Bot v4 `_trail_stop` + similar state instead of silently rebuilding from scratch.
- Phase B: bounded rolling snapshot buffer (K=10) on `IncrementalIndicatorEngine`. New `apply_bar_correction` method allows correcting any bar in the last K, not just the latest.
- Phase C: per-second history retention on `BarBuilder`. `rest_verifier` splits per-second list from aggregation. `SymbolHub.apply_rest_correction` splices per-second when both sides have data; falls back to whole-bar splice otherwise.

Plus two follow-up fixes:
- L-type bar_time alignment (intra-bar stop_loss alerts had `bar_time = fill_ts`, not bar boundary). Aligned at both verifier and engine layers.
- New `gap_fill_unverified` verification status to distinguish EH/AH gap-fill cases from genuine REST failures.

**EH validation was inconclusive** — Polygon REST data is sparse in EH (many sub-minute windows have zero trades), so `rest_unavailable` rate spiked from artifacts not real bugs. Also discovered a structural divergence: live engine gap-fills empty bars while backtest drops them, so EH/AH strategies diverge fundamentally regardless of REST quality.

## Tomorrow's plan

### Pre-market (before 13:30 UTC / 09:30 ET)

- [ ] Kevin reviews the EOD memory + this plan
- [ ] Confirm TSLA canary strategies got created overnight (per yesterday's discussion):
  - 2-3 TSLA 10Sec with relaxed triggers (less frequent firing, easier to spot real drift)
  - 1-2 TSLA 1Min (control group — REST settle well within bar)
  - 1 TSLA 5Min (upper-TF reference — should be near-zero divergence)
- [ ] Quick smoke: Railway worker still on commit `1ba3482`, no boot errors overnight

### Phase 1 — RTH opens (13:30-14:30 UTC / 09:30-10:30 ET)

The actual validation window for Phases A+B+C. With dense RTH data, both WS and REST will have aggregates for the same bars, so the verification signal will be meaningful (not contaminated by EH gap-fill artifacts).

- [ ] At T+15 min into RTH: run `python _divergence_walkthrough.py --window-hours 1 --denom`
- [ ] Compare against `Baseline_Metrics_2026-05-28_post-M8.md` (pre-Phase-A+B+C reference):
  - Target: phantom rate ≤ 20% on SPY 10Sec (from 34-56% baseline)
  - Target: `drift_uncorrected` ≤ 2% (from 7.9%)
  - Target: `corrected` materially > 0 (was ~4% baseline; expect higher because user packs now actually survive snapshot restore)
- [ ] Watch for new `"REST correction applied: sym=X tf=Ys bar=Z K=N path=per_second|bar_level"` log markers
- [ ] If numbers improved materially → Phase A+B+C validated, move to Phase 2
- [ ] If numbers stayed flat or worsened → triage what's blocking. Possibilities:
  - Phase A unlock didn't help as much as expected → user pack state wasn't actually the cumulative driver
  - Phase B K=10 still too small → bump to 20 via env, see if it moves
  - Phase C per-second path not firing → check logs for path=bar_level vs path=per_second ratio

### Phase 2 — Cross-symbol/TF validation (14:30-15:30 UTC / 10:30-11:30 ET)

With TSLA canaries firing:
- [ ] Per-symbol divergence rates (filter the walkthrough output by sid)
- [ ] **1Min/5Min strategies should be near-zero divergence.** If they're not, something else is wrong beyond TF/cadence.
- [ ] **10Sec strategies (SPY + TSLA) should be similar.** If SPY is materially worse than TSLA 10Sec, suggests symbol-specific issue (data feed quirk, SPY-only WS aggregation issue, etc.)
- [ ] Spot-check the L-type alignment fix: count post-fix stop_loss verifications by status. Should see drift_uncorrected drop materially from yesterday's 100% rate.

### Phase 3 — Strategy Detail page indicator load timing (15:30-16:00 UTC / 11:30-12:00 ET)

Phase A's snapshot-resume fix should also speed up Strategy Detail page loads (the LEF resume path stops degrading to cold-start for user-pack strategies).

- [ ] Click through sid 151 Strategy Detail, time the chart+indicator render
- [ ] Compare to Kevin's recent recollection (he flagged this in yesterday's session)
- [ ] If meaningfully faster → side benefit confirmed
- [ ] If unchanged → Phase A's effect on the resume path needs investigation

### Phase 4 — AH gap-fill structural decision (after 20:00 UTC / 16:00 ET)

The bigger structural fix Kevin deferred from yesterday. With RTH validation behind us and AH data showing the gap-fill phenomenon clearly, decide on the path:

Options documented in `Known_Bugs.md` "Live ↔ backtest divergence on gap-fill bars" entry:
1. **Backtest gap-fills too** — match live's calendar-bar semantics. Backtest results change.
2. **Live stops gap-filling** — match backtest. Changes max_hold_bars meaning.
3. **Live keeps gap-fill internally but doesn't fire triggers on gap-fill bars** — subtle.

My (Claude's) recommendation yesterday was option 1, but Kevin specifically wanted to make this call awake and after RTH validation. So this is a discussion phase, not necessarily a shipping phase.

If we ship option 1 today:
- Add gap-fill aggregation to `data_loader.resample_to_timeframe` (controllable via flag so historical backtests can re-run)
- Re-run baselines on a few strategies to quantify the behavior change

## Rollback paths (in case Phase A+B+C makes things worse in RTH)

- `REST_VERIFY_ENABLED=false` on Railway worker → verifier no-op (existing flag)
- `INDICATOR_SNAPSHOT_BUFFER_K=1` → buffer effectively K=1, reverts Phase B behavior
- Revert commit `45a3a12` (Phases A+B+C) but keep follow-up fixes (`0f47daa`, `1ba3482`)
- Full revert to `backup-2026-05-28-pre-phase-abc` if catastrophic

## Reference state at start of session

- Live model on all 41 strategies: `ws_rest_spliced`
- Worker deployment HEAD: `1ba3482` (or whatever's latest by morning)
- Verification statuses in use: NULL / verified / corrected / drift_uncorrected / rest_unavailable / gap_fill_unverified
- Snapshot buffer K=10, per-second history K=10
- Sweeper: 5-min cadence, 30-min lookback, 180s retry max_wait
- Verifier max_wait: 120s initial, 180s on sweeper retry
- Backup branches: `backup-2026-05-28-post-m8-ws-rest-spliced`, `backup-2026-05-28-pre-phase-abc`

## Memory + doc cross-references

- EOD recap: `memory/project_session_2026-05-28.md`
- ws_rest_spliced canary timeline: `memory/project_ws_rest_spliced_canary.md`
- Per-second splice context: `memory/project_per_second_splice_idea.md` (now mostly shipped as Phase C)
- Investigation SOP: `docs/SOP_Divergence_Investigation.md`
- Baseline metrics anchor: `docs/Baseline_Metrics_2026-05-28_post-M8.md`
- Divergence cluster findings: `docs/Divergence_Investigation_Log_2026-05-28.md`
- Known bugs (gap-fill, snapshot drop, etc.): `docs/Known_Bugs.md`
- Implementation plan that ran today: `~/.claude/plans/breezy-dreaming-umbrella.md`
