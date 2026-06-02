# Plan — Backtest Snapshot User-Pack Drop Fix

> **Status:** ✅ PHASE 2 SHIPPED 2026-06-01 at `f006937` — verified working on sid 263/268/270/271/272/273. Phantom rates dropped from 24%→8% (sid 263) and 14%→0% (sid 268). Phase 1 (warmup workaround) rolled back via `BACKTEST_SNAPSHOT_WARMUP_BARS=0` env. Phase 1 dead code remains in `data_worker_engine.py` pending 24h stability proof; remove after.
> **Phase 2 commit:** `f006937` on dev. Backup at `dev-backup-pre-phase2-codec`.
> **Backup branch (pre-fix baseline):** `backup-2026-05-29-layer1-solid` at `d8ba188` (Phase A+B+C Layer 1 verified solid).
> **Related docs:** `docs/Known_Bugs.md` (top entry recharacterized — gate-mode is silent stale-state not codec), `docs/User_Pack_Roadmap.md` (per-pack health register), `memory/project_session_2026-05-29.md` + `project_session_2026-06-01.md` + `project_session_2026-06-02.md` (full timeline).

## Quantified impact — empirical confirmation (sid 263, 2026-05-29)

The "Update All Data" button on sid 263 triggered a full recompute via `recompute_and_persist_stored_trades` — the warm-engine path that doesn't go through the broken snapshot. Compared to the same time-window before/after:

| Metric | BEFORE (snapshot-resume bug active) | AFTER (full recompute) | Δ |
|---|---|---|---|
| Phantom rate (3h, raw) | 95 | **3** | **-97%** |
| Missed rate (1h, raw) | 26 | **3** | **-88%** |
| Paired count (3h, raw) | 2 | **94** | **47×** |
| Unpaired alerts (3h) | 20 | **3** | **-85%** |
| Unpaired backtest edges (3h) | 26 | **4** | **-85%** |
| Layer 3 entry avg delta | 8.5s | **3.6s** | -58% (tighter) |
| Layer 3 entry ≤5s | 82.1% | **91.5%** | +9.4pp |
| KPI total_r | **−34.56R** | **+37.98R** | **+72.5R swing** |

The KPI swing confirms the bug isn't just dropping trades — it's producing actively wrong trades from cold-state user packs. After the fix, the strategy is genuinely profitable in the backtest where today's broken state shows it as a loser.

Residual ~3-5% phantoms after recompute are likely the known `phase2_signal_exit` cases + cross-event mispairings — separate, lower-priority issues. **This bug accounts for the overwhelming majority of fleet-wide divergence.**

The new snapshot saved AFTER the recompute STILL has 0 user packs — bug is structurally present every save. Strategy will drift back to phantom-heavy over 24-48h. Manual "Update All Data" clicks buy ~1 day of clean state per strategy until Phase 1 ships.

## Bug recap (one paragraph)

`unified_engine.py:3312` (`serialize_backtest_snapshot`) calls `snapshot_state(persistent=True)`. The persistent branch at `unified_engine.py:861` runs `pickle.dumps(deep)` as a probe to verify each user-pack engine is picklable. All 16 production user packs are loaded via `importlib` with synthetic module names that pickle cannot resolve → probe fails → pack is dropped from the snapshot's `packs_out` dict with a one-time warning. The b64-encoded envelope is persisted to `strategies.config.engine_snapshot_b64`. On every worker tick that resumes from this snapshot, `apply_backtest_snapshot` calls `restore_state` with an empty `packs` dict; `restore_state`'s `if packs:` check skips assignment; `_user_pack_engines` retains the fresh values from `UnifiedStrategy.__init__`. Every backtest snapshot-resume tick processes 1-N new bars with a COLD UT Bot v4 (and all other user-pack engines), so indicator state cannot accumulate across ticks. Result: backtest misses flips that live (continuously-warm) engine detects, producing fleet-wide phantoms in Layer 2 and outliers in Layer 3.

Verified empirically by decoding `strategies.config.engine_snapshot_b64` for sid 263: `packs in snapshot (0 entries): []`. UT Bot v4 from-scratch on Polygon REST 1Sec→10Sec for 2h matches live `bar_engine_states` trail_stop to 0.001 at every critical bar — the indicator math is correct; the bar data is consistent; the bug is purely in snapshot persistence.

## Two-phase fix sequence

### Phase 1 — Option 4: Warmup-window replay (interim patch, ship same-day)

**Purpose:** Stop the fleet bleeding. Restores backtest correctness without touching the snapshot format. Self-contained, behind a feature flag for fast rollback.

**Change scope:** ~30 LOC in `src/data_worker_engine.py`. No protocol changes. No DB changes. No per-pack changes.

**Mechanism:**
1. In `run_store_fed_window`, before the existing `df = df[df.index > last_bar_for_filter]` slice, retain the previous `WARMUP_BARS` bars (default 20).
2. Build a `warmup_df` containing `(last_bar_ts - WARMUP_BARS × tf, last_bar_ts]` (closed-right; bars the engine has already processed once).
3. Run `run_unified_backtest` on `warmup_df` first with `resume_snapshot=envelope`, `include_open_position=False` — this warms `_user_pack_engines` against bars where we already know the engine's prior outputs.
4. Then call `run_unified_backtest` on the post-snapshot `df` slice with the warmed-up packs (via a second `resume_snapshot` capture if needed, or by sharing the strategy instance across calls).
5. Suppress trades from the warmup pass — only commit trades from the post-snapshot slice.

**Feature flag:** `BACKTEST_SNAPSHOT_WARMUP_BARS` env var. Default 20. Set to 0 → disables warmup, falls back to current behavior. Trivial rollback.

**Cost:** ~200ms per tick × N strategies. Worker has headroom — Phase A+B+C planning estimated 400MB peak vs ~3GB available.

**Limitation:** This is a workaround, not a fix. The snapshot still drops user packs; the warmup hides the consequence. If a pack needs >20 bars of warmup (e.g., a 50-period EMA cross), the workaround leaks. Fine for the current pack set (UT Bot v4 needs ~10, MACD ~26, EMA stacks ~21) but unbounded packs would fall through.

**Verification:**
- Run Layer 2 phantom count on sid 263 + 268 before and after deploy
- Phantom rate should drop to ≤5% on both
- Existing alerts continue to pair within ±5s on Layer 3
- Worker CPU/memory metrics stay within Phase A+B+C bounds

**Risk register:**
- Snapshot envelope gets called twice if we don't carefully reset state between warmup and live phases. Mitigation: explicit `strat.position.reset()` between calls; assert position is FLAT before live phase.
- Polygon REST `_fetch_rest_bar` is called for warmup bars that we ALREADY had. Wasted REST calls. Mitigation: warmup pulls from the store (which has them), not REST. Already free.
- Cold start of a user pack at the warmup boundary still produces SPURIOUS flips during early warmup bars. Mitigation: increase WARMUP_BARS if observed. The trades from those warmup bars are not committed anyway (filtered before write).

**Rollback path:** Set `BACKTEST_SNAPSHOT_WARMUP_BARS=0` on Railway worker env. Effective on next worker boot. Or git revert the commit. Worker restart should not cause data loss; existing trades remain.

### Phase 2 — Option 2: Pack state protocol (real fix, ship next 1-2 days)

**Purpose:** Remove pickle from the pack-state path. Each pack publishes an explicit serializable state contract. Future packs cannot ship without implementing it.

#### Step 1 — Protocol module
New file `src/user_pack_state_protocol.py` (~40 LOC):
```python
class PackStateProtocol:
    """Each user pack's incremental engine class implements:
       - serialize_state() -> dict[str, JsonValue]
       - @classmethod restore_state(state: dict) -> instance
       - state_version: int (class attr)
    """
    state_version: int = 1
    def serialize_state(self) -> dict: ...
    @classmethod
    def restore_state(cls, state: dict): ...
```
Plus a register-time validator in `pack_registry.py` that rejects packs missing these methods, with a clear error pointing at the pack's file.

#### Step 2 — Engine integration
In `unified_engine.py`:
- `snapshot_state(persistent=True)` replaces the pickle-probe loop with:
  ```python
  for slug, eng in self._user_pack_engines.items():
      packs_out[slug] = {"v": eng.state_version, "state": eng.serialize_state()}
  ```
- `restore_state(packs)` becomes:
  ```python
  for slug, payload in packs.items():
      cls = pack_registry.get_pack(slug).incremental_class
      self._user_pack_engines[slug] = cls.restore_state(payload["state"])
  ```
- Drop the `if packs:` guard — packs is always non-empty for a strategy that uses user packs.

#### Step 3 — Per-pack implementation
For each of the 16 packs in `user_packs/<slug>/indicator_incremental.py`:
- Add `state_version = 1` class attr
- Implement `serialize_state()` returning a JSON-compatible dict of internal floats/ints/strings
- Implement `@classmethod restore_state(cls, state)` to rebuild
- Round-trip test: `restore_state(eng.serialize_state())` produces an instance that, fed the same next bar, produces bit-identical output

Pack-by-pack checklist (canary first, then high-traffic, then rest):

| Pack | Priority | State to serialize (initial list — finalize during impl) |
|---|---|---|
| ut_bot_v4 | 1 (canary) | `_trail_stop`, `_position`, `_atr`, `_prev_close`, `_init` |
| macd_line_v2 | 2 | `_fast_ema`, `_slow_ema`, `_signal_ema`, `_prev_macd` |
| ema_pp_v4 | 3 | EMA periods + values, position state |
| ema_pp_v3 | 4 | same shape as v4 |
| ema_stack_v2 | 5 | EMA values dict |
| macd_histogram_v2 | 6 | same shape as macd_line_v2 |
| rsi_zones / rsi_zones_2 / rsi_zones_3 | 7-9 | RSI values, prior zone |
| stochastic_oscillator | 10 | K/D windows |
| supertrend | 11 | trend, ATR, trail values |
| swing_123_test | 12 | swing point history |
| sr_channels | 13 | recent S/R levels |
| vwap_v2 | 14 | cumulative VP, cumulative V |
| rvol_v2 | 15 | volume MA window |
| strat_assistant | 16 | TBD |
| bollinger_bands | 17 | MA + StdDev windows |

#### Step 4 — AI pack-creation defenses
Update `pack_builder.py` and the column-contract validator (from 2026-04-27 session) to include `serialize_state`/`restore_state` as required methods. New AI-built packs cannot register without the protocol.

#### Step 5 — Migration of existing strategies
Existing `engine_snapshot_b64` blobs have empty `packs` dicts. On the first post-fix tick:
- `restore_state` is called with empty packs
- Engine falls back to its pre-existing init state for user packs
- After processing the tick, `snapshot_state(persistent=True)` now correctly serializes all packs
- The NEW snapshot has the full state — bug is self-healing within one tick per strategy

This means no migration script is needed. Old snapshots gracefully degrade to "cold start the next tick" which is current behavior — strictly no worse than today's bug. Then they self-heal.

If Phase 1 (warmup workaround) is still active when Phase 2 lands, the warmup window covers the cold tick of the migration; user sees no degradation at all.

#### Step 6 — Tests
- `test_snapshot_state_protocol.py`: unit test per pack, round-trip
- `test_snapshot_resume_parity.py`: integration test — run 100 bars, save snapshot at bar 50, restore + run remaining 50, assert trade list bit-identical to non-snapshot run

#### Step 7 — Deploy
- Push to dev → Railway auto-deploys worker
- Watch Worker logs for the new register-time validation; any pack missing the protocol fails loudly
- Run Layer 2 health check on sid 263 + 268 + SPY 10Sec cohort
- Phantom rate should drop sustainably (not regress over the next hour as we observed in the Option 4 experiment)
- After 24h of stable operation: remove the Phase 1 warmup workaround

## Rollout sequence

```
T+0          Phase 1 ships → fleet bleeding stops; phantom rate drops on canaries
T+24h        Verify Phase 1 stable; phantom rate ≤5% on 10Sec
T+24h        Begin Phase 2 protocol module + UT Bot v4 implementation
T+48h        UT Bot v4 round-trip verified; remaining packs in flight
T+72h        All 16 packs implemented + tested
T+72h        Phase 2 ships → snapshots now persist user-pack state
T+72h-96h    Verify Phase 2 sustained (no regression hour-over-hour)
T+96h        Remove Phase 1 workaround
```

Aggressive timeline. Slip is fine — Phase 1 protects fleet correctness while Phase 2 takes whatever time it needs.

## Verification metric — definition of done

Both must hold for 24h continuous:
- Layer 1: `drift_uncorrected` ≤ 2% on 10Sec strategies; ≤ 0.5% on 1Min+ (unchanged from today)
- Layer 2: phantom rate (global-fair) ≤ 5% on 10Sec; ≤ 2% on 1Min+
- Layer 3: median 0s entry + exit; ≥ 95% within ±5s on entry; ≥ 90% on exit

Plus structural:
- Decoded snapshot for any active strategy shows non-empty `packs` dict
- Layer 4 spot-check on 3 random strategies (Kevin's choice) — no surprise gaps in trade list

## Risks not yet mitigated

- **JSON serialization edge cases.** If a pack's state includes numpy arrays, datetimes, or complex objects, JSON will throw. Mitigation: state contract must use only `(int, float, str, bool, list, dict, None)`. Validator enforces this. Numpy values coerce to Python via `.item()` before serialize.
- **State version mismatch on resume.** If a pack version bumps and old snapshots have `v=1` but new code expects `v=2`. Mitigation: each `restore_state` reads `state.get('v', 1)` and migrates if known. Otherwise falls back to cold start (warmup workaround covers it during transition).
- **Two-pack interaction edge cases.** If Pack A's state references Pack B's output, the restore order matters. Mitigation: register-time validator checks for cross-pack dependencies and rejects.
- **Round-trip parity test missing a subtle field.** Mitigation: each pack's test must cover at least one cold-to-warm transition where the indicator's behavior changes (e.g., first trail flip). If the round-trip differs after restore, the missing state field is identified immediately.

## What this plan deliberately does NOT do

- Change the LIVE engine snapshot path (`persistent=False`, in-memory mode) — Phase A already fixed it.
- Change Layer 1 bar fidelity logic — verified solid 2026-05-29.
- Touch the trade-pairing logic in `/api/admin/strategy-health/backlog` — that's downstream and works correctly.
- Add a DB migration — self-healing on first tick (see Step 5).
- Rewrite any built-in indicators (EMA, MACD, ATR, etc.) — those serialize fine via the standard `IndicatorState` class.

## Backup strategy at each milestone

- Pre-Phase 1: `backup-2026-05-29-layer1-solid` (already saved at d8ba188)
- Pre-Phase 2 protocol module: `backup-pre-phase-2-protocol` before touching `unified_engine.py`
- Pre-per-pack migration: per-pack branches OR a single `phase-2-packs` working branch with one commit per pack (preferred — preserves bisectability)
- Pre-removal of Phase 1: `backup-pre-warmup-removal`

## Open questions for Kevin

1. **Phase 1 warmup-window size.** Default 20 bars covers ATR(10), MACD(26 — borderline). If we have a known longer-warmup pack (RSI(50)?), bump default to 60. Confirm before ship.
2. **Phase 2 protocol — should `restore_state` accept the strategy params too?** Current spec is state-only. If a pack needs to re-validate its params on restore (e.g., to detect a config change), it would need both. Probably YAGNI — the snapshot fingerprint already catches config changes.
3. **Mass Builder interaction.** Mass builder uses `run_unified_backtest` directly with `resume_snapshot=None` (always cold-start). Phase 2 doesn't change that path. Confirm mass-built strategies' first snapshot will populate `packs` correctly on the FIRST persistent save (after creation).

## References

- `src/unified_engine.py:840-896` — snapshot_state, restore_state
- `src/unified_engine.py:3287-3395` — serialize / deserialize / apply backtest snapshot
- `src/data_worker_engine.py:323-390` — run_store_fed_window (tick path)
- `src/data_worker_engine.py:613-654` — run_startup_catchup (catchup path)
- `src/pack_registry.py` — pack loading, registration
- `docs/Known_Bugs.md` — bug entry "Backtest snapshot drops ALL user-pack state"
- `memory/project_session_2026-05-29.md` — full investigation chain
- `memory/reference_health_check_sop.md` — 4-layer health check methodology
