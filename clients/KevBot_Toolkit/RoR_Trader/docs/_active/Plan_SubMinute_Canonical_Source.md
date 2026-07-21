# Plan — Sub-Minute Canonical Bar + State Source (kill the 339-class state-path divergence at the root)

**Status: BUILT 2026-07-21 (flag-OFF, branch `feat/submin-canonical-source`) — see
`Impl_SubMinute_Canonical_Source.md` for what shipped, the gates, and the arming SOP.**
⚠️ One premise below was FALSIFIED during the build: the backtest does NOT derive sub-minute
secondaries from 1Sec Hi-Fi — the services secondary loop UPSAMPLES the primary df into a
pseudo-series at primary cadence (339's "10s" bt ribbon was computed on 30s bars). The cure
therefore includes an offline-lane construction change (component C), and **arming the flag
changes 339's backtest on the next recompute** — intended: one canonical bar, both lanes.

**Status when scoped (original):** SCOPED 2026-07-21. The label-drift fix
(`RORT_TF_LABEL_SEC_FIX=1`) revived 339's sub-minute gate, but *unmasked*
[[project_submin_state_path_divergence]]: the live 10s UT_BOT ribbon flips 1–3 bars later than the
vectorized backtest ribbon, capping 339 at ~14% vs its backtest → **alive but not tradable.** This
doc scopes the structural cure Kevin asked for: give sub-minute TFs the same **stable canonical
source** the coarse TFs already have.

## What we CONFIRMED about the bar architecture (2026-07-21, not assumption)
| Layer | Store | Timeframes | Source | State |
|---|---|---|---|---|
| REST base | `bar_cache` | **1Sec + 1Min only** | Polygon REST | this is all the Bar Cache UI shows |
| Coarse canonical | `resampled_bar_cache` | **120s → 1d** | resampled from **1Min** (`source_1min_span_hash`) | **BUILT + ARMED + serving live** (`RORT_RESAMPLED_STORE_READ/SERVE/SERVE_LIVE=1` on Worker; READ/SERVE on api+batch) |
| WS live | `live_bars` | all TFs incl. sub-minute | WS aggregation | decision-time, not canonical |
| **Sub-minute canonical** | **— none —** | **10s / 15s / 30s** | — | **THE GAP** |

**So Kevin's instinct is exactly right, just precisely located:** coarse TFs already read ONE
canonical stable bar (`resampled_bar_cache`) in both lanes — that's why 314's 4h/1h fix stuck and
coarse strategies are stable. **Sub-minute is the one octave with no canonical source:** the store
floor is 120s because you cannot resample 10s/30s from a 1Min source. Live builds sub-minute from WS
aggregation; the backtest derives it from 1Sec Hi-Fi — two independent reconstructions.

## The problem, precisely (and an honest nuance)
The 339 divergence has TWO contributors, and the evidence says which dominates:
- **PRIMARY = path-dependent STATE derivation.** The 10s UT_BOT ATR trailing stop is path-dependent;
  live derives the ribbon **incrementally**, the backtest **vectorized** → they flip at different
  bars. Proof it's the state, not the bars: the **corrected-bar ceiling is ALSO 14.3%** — feeding
  REST-healed bars doesn't fix it, so it's not a WS/REST tip issue.
- **SECONDARY = no shared bar.** WS-aggregated 10s vs 1Sec-derived 10s can differ at the tip; a
  second-order contributor that a canonical shared bar removes.

**M-RS5b (`RORT_CANONICAL_FINE_TF_STATE=1`) already derives canonical gate state at each close — but
stops at ≥60s** (the `tf_s < 60` boundary in `services.py`). Sub-minute never gets canonical state →
the incremental path stands → drift.

## The cure — extend the PROVEN machinery one octave down, sourced from 1Sec
Two components; both are extensions of armed, working systems (not new inventions):

**A. Canonical sub-minute STATE (the load-bearing fix) — extend M-RS5b below 60s.**
Derive the sub-minute gate ribbon canonically at each close (the same vectorized construction both
lanes use), instead of live-incremental vs backtest-vectorized. This is the actual cure for the
path-dependent flip (the corrected-bar test says state, not bars, is the driver). Remove/extend the
`tf_s < 60` exclusion so `CANONICAL_FINE_TF_STATE` covers 10s/30s, reading from →

**B. Canonical sub-minute BAR store (the foundation + Kevin's stable-source completion) — extend
M-RS2-P2 to sub-minute, sourced from stored 1Sec.**
Add 10s/15s/30s to `resampled_bar_cache`, resampled from the **1Sec** base (analogous to how ≥2m is
resampled from 1Min), with a `source_1sec_span_hash` provenance. Both lanes read ONE bar → removes
the tip variance AND completes the single-canonical-source architecture to its finest TF. **This
store is ALSO the cost fix** (see below) — compute the sub-minute bars once, not per-lane-per-cycle.
Mirror `resampled_bar_store.py` + the consumers (`_resampled_store_consumer*.py`).

## ⚠️ The #1 risk — sub-minute is HEAVIER (and starved a prior attempt)
`services.py:~1094-1119` documents it: `tf<60s` bars are 1Sec-derived, and a prior re-arm attempt
(`Bug_Hunt_Wave1_2026-07-06 'Re-arm attempt FAILED'`) hit **starvation** — sub-minute slots vs the
shadow-worker's 600s-pass watchdog. 1Sec→10s is ~6 bars/min (6× the coarse store's density). So:
- **The store (component B) is the mitigation, not just architecture:** persist once, both lanes
  read — avoids re-deriving 1Sec→10s repeatedly (which is what starved before).
- **Bound the write cost:** populate incrementally + session-scoped; do NOT full-rebuild sub-minute
  every cycle. Watch the shadow-worker pass time against its watchdog.
- **Scope the TF set tightly:** only the sub-minute TFs strategies actually gate on (today: 10s,
  30s). Don't build 1s/5s speculatively.

## Extension points (verify in the session)
- `resampled_bar_store.py` + `_resampled_store_consumer*.py` — the ≥2m pipeline to mirror for 1Sec→sub-minute.
- `services.py` `tf_s < 60` / `tf_s >= 60` boundaries (~1119/1224) — where sub-minute is excluded from the canonical path.
- `bar_store.py:112` — already resamples the 1s series to '10Sec' etc.; the resampling primitive exists.
- The `CANONICAL_FINE_TF_STATE` gate — extend below 60s.
- `bar_cache.py` note ("Resampling stays caller-side; cache stores 1Min and 1Sec for L3").

## Validation gates (non-negotiable, mirror the label-drift discipline)
1. **`fidelity_parity_suite.py` canary 267 = 18/18 byte-identical, flag OFF and flag ON.** ≥1Min
   strategies MUST be unchanged (this only adds a sub-minute path).
2. **339 goes from ~14% → a real ceiling matching its settled backtest @10s** — the harness (now
   GEN-aware + fail-loud) is the fast proof; then confirm live-vs-settled after a recompute.
3. **No regression on the coarse store** (314 = 92.3, 340/267 = 100 stay put).
4. **Shadow-worker pass time stays under its watchdog** (the starvation guard) — measure it, don't assume.
5. **Multi-leg live check** (the label-drift lesson): validate the sub-minute path LIVE via
   XTF_BLOCK_DIAG present-sets, not harness-only — real-monitor coverage topology only exists live.

## Flag-gating / reversibility
Default-OFF kill-switch flag (e.g. `RORT_CANONICAL_SUBMIN_STATE` + reuse the resampled-store serve
flags extended to sub-minute). Byte-identical when OFF. Live-engine + offline-lane change → arm on
**Worker + api + batch-worker** (offline lanes must derive the same canonical bar). Backup branch
first. Reversible via flag flip.

## Sequencing / prioritization
- **A before B is tempting but B underpins A** (canonical state derives from canonical bars). Build
  **B (the sub-minute store) first**, prove byte-parity, then **A (canonical state reading from it)**.
- Kevin's call to prioritize this stands, with eyes open: it's a **real structural build** (heavier
  than the label-drift fix), justified because it (a) makes 339 + the whole sub-minute-secondary
  class tradable and (b) completes the single-canonical-source architecture that kills gate-
  construction drift for good. If the sub-minute class is just 339 today, weigh it against grinding
  the ≥1min fleet to 90% first — but the architecture is durable either way.

## Sibling (small, not this session) — Bar Cache UI
The Bar Cache page shows only `bar_cache` (1Sec/1Min), so the canonical coarse source
(`resampled_bar_cache`) and WS bars (`live_bars`) are invisible — which is why it *looked* like the
coarse source didn't exist. Surface those two tables on the page (visual observability) so the
canonical stack is legible at a glance.

## Launch prompt
> Work `docs/_active/Plan_SubMinute_Canonical_Source.md`: build component B (sub-minute canonical
> `resampled_bar_cache` from 1Sec, byte-parity vs current 1Sec-derived), then component A (extend
> `CANONICAL_FINE_TF_STATE` below 60s to read it). Flag-gated default-OFF; 267 18/18 OFF and ON;
> prove 339 ~14%→matches settled backtest via the harness then live; watch shadow-worker pass time
> vs its watchdog (prior sub-minute attempt starved). Validate the sub-minute path LIVE via
> XTF_BLOCK_DIAG, not harness-only.

## Non-goals
The Bar Cache UI (sibling); speculative TFs (1s/5s) no strategy gates on; any change to the ≥2m
store or the ≥60s canonical path (byte-identical must hold).
