# Divergence Monitoring & Investigation — 2026-06-22 (post-weekend UND)

**Context:** First fleet-wide `update_new_data` (UND) since ~Thursday, after weekend
changes (mass-builder fixes, gen-gate tz fix, new test strategies). Goal: establish a
**clean post-UND baseline** before touching anything, so we don't break the well-performing
strategies with an untested change. Kevin observed on the Strategy Health page: only **321
= 96.4% combined** (good); almost everything else = **0% combined**, split into Phantoms
and Misses.

**Locked terms (Kevin):** *Phantom* = alert-only (live fired, algo/backtest didn't).
*Missed* = algo-only (backtest fired, live didn't).

**Deploy stance: HOLD.** No pushes until UND finishes. Fixes may be staged + validated
locally, NOT deployed. Reason: don't perturb the 96% performer or muddy divergence mid-run.

---

## Triage table (as of ~19:30 UTC, 15:30 ET)

| sid | symbol/tf | kind | secondary TFs | live alerts | last alert | category |
|-----|-----------|------|---------------|------------|-----------|----------|
| 321 | TSLA 15Sec | ungated | — | 189 | 19:27 | ✅ healthy (96.4%) — control |
| 325 | TSLA 30Sec | cross-TF | 4h, 30m | 17 | 16:31 | **Phantom** — stale BT lane |
| 330 | TSLA 30Sec | cross-TF | 4h, 30m | 17 | 16:31 | **Phantom** — stale BT lane |
| 331 | TSLA 30Sec | cross-TF | 10m, 4h | 16 | 16:29 | **Phantom** — stale BT lane |
| 333 | TSLA 30Sec | cross-TF | 3m, 15m | 11 | 19:25 | firing; partial Missed (stale lane) |
| 327 | TSLA 30Sec | cross-TF | **2m, 5m** | **0** | — | **Missed / live-silent** — CANDIDATE BUG |
| 328 | TSLA 30Sec | cross-TF | **5m** | **0** | — | **Missed / live-silent** — CANDIDATE BUG |
| 329 | TSLA 30Sec | cross-TF | **1M, 5m** | **0** | — | **Missed / live-silent** — CANDIDATE BUG |
| 334 | TSLA 15Sec | GEN | session | **0** | — | **Missed** — gen-gate bug (CONFIRMED) |
| 335 | TSLA 15Sec | GEN | tod 11:30–14:00 | **0** | — | Missed — gen-gate bug (+ out of window today) |
| 336 | TSLA 15Sec | GEN | tod 14:00–16:00 | **0** | — | **Missed** — gen-gate bug (CONFIRMED) |

All 11 are **monitored** by the worker (65 total monitored) and **processing bars** — so
zero-alert cases are genuine live silence, not "not watching."

---

## BUG #1 — Live general-pack gating completely non-functional (CONFIRMED, fix staged)

**Affects:** every gen-gated strategy live (334, 336 now; 318–326 before deletion; 335 too
but it's also out-of-window today so untestable until tomorrow 11:30 ET).

**Symptom:** live bar 7599 (15:09 ET) showed `pos=FLAT signals=0 triggers=['utv4_bull_flip']`
— entry fired, strategy warm + flat, but gate suppressed it. Zero live alerts ever.

**Root cause:** `ralph_engine.py` ~line 1279, live bar-close path:
```python
bar_ts = bar.get('timestamp')          # BarBuilder.to_dict() → ISO STRING
if isinstance(bar_ts, datetime):       # FALSE for a str → eval SKIPPED
    self._current_confluence |= _evaluate_general_packs(self._general_packs, bar_ts)
```
`BarBuilder.to_dict()` emits `'timestamp': self.bar_start.isoformat()` (a **string**), so the
`isinstance(..., datetime)` guard is always False live → `_evaluate_general_packs` is never
called → no `GEN-*` record enters `_current_confluence` → the gate's
`confluence_set.issubset(_current_confluence)` always fails → `signals=0`, no entry, no alert.
The **backtest** path uses a datetime DataFrame index, so it worked — which is why the earlier
**tz fix (99c25cf) only ever helped backtest**, never live. Also explains the empty
`GATE_DIAG` (it only logs *after* an entry fires, which never happens).

**Verification chain (all confirmed):**
1. Backtest gating correct: 334 406/407, 335 157/157, 336 132/133 entries inside window.
2. Worker on fixed code (deploy 564afa65, 18:09Z — includes 99c25cf tz + d61946b pack-match).
3. Reproduced exact worker pack path offline (`load_general_packs_admin` → filter → eval at a
   **datetime**): 334 PASS, 336 PASS, 335 BLOCK (correct, out of window).
4. Re-simulated with a **string** timestamp (the real live condition): old guard → ALL
   gen-gated BLOCKED; **fix** → 334 PASS, 336 PASS, 335 BLOCK. ✅

**Fix (STAGED, uncommitted, NOT deployed):** `ralph_engine.py` ~1279 — coerce string→datetime
via `pd.Timestamp(bar_ts).to_pydatetime()` before the guard (matches the pd.Timestamp parse
pattern used elsewhere for `bar['timestamp']`). File compiles; offline-validated. `_gp_to_market`
in `_eval_gp_scalar` handles tz (naive→UTC→ET) so parsed aware/naive both resolve correctly.

**Will UND fix it? NO.** UND recomputes the *backtest* lane; live still won't fire until this
deploys. After deploy, 334 (session) + 336 (14:00–16:00) should fire on the next in-window
`utv4_bull_flip`. Re-arm monitor post-deploy to confirm first live gated alert.

---

## CATEGORY B — Cross-TF Phantoms (325, 330, 331): likely stale backtest lane

These **did** fire live (16–17 alerts each up to ~16:31 UTC), so their cross-TF gate
(4h/30m/10m secondaries) works live. Phantom status ⇒ live alerts have no matching backtest
trade because the **backtest lane is stale (no UND since ~Thursday)**. **Expectation: UND
recomputes the backtest lane → phantoms largely resolve.** RE-ASSESS AFTER UND — do not treat
as a code bug yet. All three are currently FLAT (not stuck in a position); quiet-since-16:31 =
entry simply hasn't fired/passed, not a hang.

333 (3m/15m) is firing (last 19:25) — its "Missed" is likely the same stale-lane effect
(backtest has trades live hasn't matched yet). RE-ASSESS post-UND.

---

## BUG #2 CANDIDATE — Cross-TF live-silent (327, 328, 329): INCONCLUSIVE

**Symptom:** 0 live alerts ever, despite being monitored + processing bars + having large
backtest lanes (357–581 trades). Currently FLAT, `triggers=none` on the sampled bar.

**Shared factor:** all three gate on **2m and/or 5m** secondaries (327: 2m+5m, 328: 5m,
329: 1M+5m), whereas every *firing* cross-TF strategy uses 3m/10m/15m/30m/4h. The 2m/5m
clustering is suspicious. Note also 329 uses **`1M`** (uppercase) vs others lowercase — a
possible TF-label-case mismatch (`_normalize_confluence_label` / `_LABEL_TO_TF_SECONDS`).

**History:** live cross-TF confluence has regressed before — Phase 31 (Polygon migration)
silently broke Phase 30L cross-TF live confluence (secondary builders never fed after warmup);
fixed 2026-04-27 (ca57cac). Worth checking the 2m/5m shadow builders are actually producing
SWING_123 records live.

**CLASSIFIED (19:35 UTC, monitor caught it): GATE-BLOCKED, not entry-silent.**
Bars 3850–3851 showed all three `pos=FLAT signals=0 triggers=['sw123_bull_c2']` (329 also
`sw123_bear_c2`). So the **entry trigger fires live while FLAT, and the cross-TF gate suppresses
it** (`signals=0`, no alert) — exactly the gen-gate signature. Backtest fires (357–581 trades),
live gate blocks → all Missed.

**Ruled out:** TF-label resolution is fine — `2m`→120s, `5m`→300s, `1M`→60s all resolve in
`_LABEL_TO_TF_SECONDS`, so secondary shadow builders should be created. Label normalization
can't be the universal cause either (the firing group 325/330/331 also relies on it).

**Pattern (record, don't over-read):** every SILENT strategy's largest secondary TF is **≤ 5m**;
every FIRING cross-TF strategy has a secondary **≥ 15m** (4h / 15m). Suggests something specific
to small/near-primary secondary TFs (primary = 30Sec), but warmup magnitude argues against it
(4h over 7d = ~12 bars yet fires; 5m over 7d = ~546 bars yet silent).

**Root cause NOT yet pinned.** Two live possibilities remain, distinguishable only by inspecting
the live `_current_confluence` for these strategies: (a) the 2m/5m secondary `SWING_123` records
are **absent/mislabelled** in the mtf buffer → gate can never pass (real bug); (b) the confluence
is **genuinely not met** live at those bars (correct block, just tight alignment). Backtest firing
357–581× makes (a) more likely, but not proven.

**Plan (post-UND batch):** `GATE_DIAG` is useless here (only logs *after* an entry fires). Add a
companion diagnostic that, when a gated strategy's entry trigger fires but is suppressed
(`FLAT` + trigger true + no entry signal), logs **needed confluence_set vs present
`_current_confluence`** — so on the next deploy we instantly see whether the 2m/5m records are
absent. Stage with the gen-gate fix; deploy together after UND.

DO NOT change until UND finishes AND root cause is confirmed (these may share data/builders with
the healthy ones; avoid collateral damage).

---

## Action log / decisions
- **HOLD all deploys until UND completes.** (Kevin, 2026-06-22.)
- Gen-gate live fix STAGED locally (`ralph_engine.py` ~1279), validated, **uncommitted**.
- Monitoring: background watch for (a) 327/328/329 trigger fires, (b) UND progress on the
  phantom lanes. Re-assess the whole table once UND finishes.

## After-hours live test for the gen-gate fix (2026-06-22 ~17:00 ET)
RTH strategies 334/335/336 can't confirm the gen-gate fix after the 16:00 close. Created
**sid 337** (`AH-TEST TOD 16:00-20:00 ExtHours (utbot)`): TSLA 15Sec, ut_bot entry,
`trading_session='Extended Hours'`, gated on a NEW pack `time_of_day-16:00-20:00`
(`GEN-TIME_OF_DAY-16:00-20:00-IN_WINDOW`, verified emitting at 16:53 ET). After the gen-gate
fix deploys + worker reload, 337 should fire on the next after-hours TSLA ut_bot bull_flip →
that's the live confirmation tonight. If it stays silent in-window, the fix didn't take.

## Post-UND checklist (when update_new_data completes)
1. Re-pull the triage table — confirm phantoms (325/330/331/333) resolved (BT lane refreshed).
2. Deploy Bug #1 gen-gate fix → confirm 334/336 fire first live gated alert (log deploy time).
3. Resolve 327/328/329 classification → fix cross-TF live gating if confirmed (check 2m/5m
   shadow builders + `1M` label case).
4. Verify 321 stays healthy (no regression from any change).
