# Flag inventory — the read-vs-set intersection (source 1 × source 2)

**Board:** #165 Phase 0 step 4 (AUDIBLE) · **Artifact task:** #265 · **Coupled to:** #162, #161
**Author:** E·auto (run `r1785551188-165`) · **Date:** 2026-07-31 · **Base:** `origin/dev` @ `2cad3782`
**Rails:** READ-ONLY. Nothing was flipped, set, unset, restarted or redeployed.

**Companion:** `Env_Config_Parity_Inventory.md` (same commit) holds the per-flag classification
table for all **136** code-declared `RORT_*` tokens — class, reason, read site, default. **That
file is §2 of this inventory; this file is the fold-in of source 2 and the analysis it unlocks.**
Read them together; neither is complete alone.

---

## 0. What changed since the predecessor

The predecessor was complete on sources 1, 3 and 4 and **blocked on source 2** (`railway variables`
denied at the headless permission layer). M captured source 2 from an attended session on 07-31
and posted it to #265: **114 `RORT_*` settings across 5 services, 67 distinct flags.**

This document folds that in and answers the question the predecessor could not:

> **Which flags does a service *read in code* but *not have set* — so it silently runs its
> compiled default?**

That intersection is the finding. **The raw partial list is not**: 65 of the 67 flags are set on
some services and not others, and most of that is correct. `RORT_NIGHTLY_RECOMPUTE*` belongs on
batch-worker; `RORT_HEALTHCHECK_*` belongs on Worker. **A flag absent from a service that never
reads it is not a gap**, and no such row is reported here as one.

---

## 1. Method — and its honest limit

**Source 1 (per-service).** Each Railway service has exactly one Python entrypoint (from its
Dockerfile `CMD`):

| Service | Entrypoint | Modules in import closure | Distinct `RORT_*` read sites reachable |
|---|---|---|---|
| Worker | `src/worker.py` (+ `src/engine_health.py` as `HEALTHCHECK`) | 56 | 75 |
| Data Worker | `src/data_worker.py` | 62 | 82 |
| batch-worker | `src/batch_worker.py` | 56 | 84 |
| shadow-worker | `src/shadow_worker.py` | 55 | 88 |
| api | `api.main:app` (uvicorn) | 100 | 75 |

The closure is computed by AST walk over every `Import`/`ImportFrom` node — **including
function-level (lazy) imports**, which this codebase uses heavily — resolved against the module
tree under `src/`. A flag counts as "read by service X" when some module in X's closure calls
`os.getenv`/`os.environ.get`/`os.environ[...]` (or the `_os.` / `_os_bt.` aliases) on it.
Reproducible: `tools/flag_read_closure.py` (committed with this doc).

**⚠️ The limit, stated up front.** Import reachability is an **upper bound on reads, not a proof
of execution**. A module can be imported and its flag-reading function never called on that
service's actual path. So:

- **"reads it" is a claim I verify by hand before calling anything a gap.** Every finding below
  names the read site *and* the call chain from the service's entrypoint.
- **"does not read it" is strong** — if the module is not even importable from the entrypoint,
  the flag cannot be read. That direction is what proves the inert settings in §3.

Counts in the table above are therefore upper bounds and are used only to scope the hand-check.

---

## 2. 🔴 The finding — `RORT_WARMUP_PREV_CHAIN` is armed on Worker and on nothing else

```
RORT_WARMUP_PREV_CHAIN=1     Worker
                             MISSING on: Data Worker, batch-worker, shadow-worker, api
```

**Read site:** `unified_engine.py:72`, `_warmup_prev_chain()`, default `"0"` = OFF.
`unified_engine` is the **shared** engine module — it is in the import closure of all five
services, and the offline lanes are the ones that execute the code this flag guards.

**What the flag does, from its own docstring:**

> `warmup()` and the `recompute_from_history` full-replay path never populated the prev-value
> chain … Every NON-incremental record derivation (`_derive_confluence_records` after a warmup
> seed, the non-RTH session reload that runs on EVERY close, the `RORT_MTF_STATE_REFRESH`
> refresher, rebroadcast recompute, gap-heal) therefore handed `prev={}` to the user-pack
> interpreter dispatch … shift-based interpreters (MACD_HISTOGRAM_V2) silently emitted NO record.
> **Flag ON:** a warmed engine's state is identical to sequential `update_bar` ingestion and
> derive == incremental. **Flag OFF: byte-identical legacy.**

**`warmup()` and `recompute_from_history` are exactly what batch-worker and shadow-worker do.**
A recompute *is* a full replay; a shadow re-true *is* a warmup seed. So today:

- **Worker** (live) runs with the prev-chain fix → derive == incremental.
- **batch-worker / shadow-worker / api** (every offline lane) run the **legacy** path → derive
  hands `prev={}` and shift-based user-pack interpreters can emit no record.

That is a **standing live↔backtest divergence source, by configuration**, in precisely the class
of strategies the divergence hunt has been chasing (user-pack / MACD-histogram gates). It is not
visible to anyone reading the flag list per-service, because "armed on Worker" reads as armed.

**Classification: `must-match`, currently MISMATCHED.** It is the only row in this inventory
where a `must-match` flag is armed on one engine service and absent on the rest.

**Flagging it, not fixing it** (rails). **Do not arm it as a side-effect of the split** — it is a
behaviour change to the offline lanes and needs its own validation, and #165's own ordering says
production must not change before #161 parity lands.

**Owner:** M to route — either a dedicated arm task (validate the offline lanes under the flag),
or an explicit written decision that the offline lanes are meant to stay on legacy. Either is
fine; silence is not.

---

## 3. The two items M flagged — determined

### 3.1 `RORT_RECOMPUTE_PARALLELISM` 6 vs 8 — **not a conflict; the `8` is inert**

```
RORT_RECOMPUTE_PARALLELISM   batch-worker=6   shadow-worker=8
```

**Read sites:** `batch_worker.py:267` and `recompute_jobs.py:732`. Nothing else reads it.

**`recompute_jobs` is not reachable from `shadow_worker.py`.** It appears in the import closure of
`batch-worker` and `api` only. `shadow_worker.py` / `shadow_manager.py` contain no import of
`recompute_jobs` at module or function level (verified by grep as well as by AST closure), and no
subprocess spawn. **So shadow-worker never reads this variable.** The `=8` has no effect on
anything; only batch-worker's `6` does.

- **batch-worker `6`** → `may-differ`. **Reason:** worker-pool sizing matched to the service's
  CPU/RAM. Recompute results are order-independent; throughput is not. Parallelism has previously
  been a saturation lever, so it must be sized to the box rather than copied between environments.
- **shadow-worker `8`** → **not `may-differ` — Class B (set but unread).** It is a dead setting on
  that service. Recorded in §4. **The right fix is to delete it, not to reconcile it to 6** —
  reconciling would enshrine the illusion that shadow-worker has a recompute pool.

Same determination, same evidence, for two more shadow-worker settings: **`RORT_RECOMPUTE_SUPERVISOR=1`**
(`recompute_jobs.py:764`) and **`RORT_COMPUTE_REMOTE=1`** (`compute_jobs_store.py:30`, also
unreachable from `shadow_worker`) are inert on shadow-worker.

### 3.2 `RORT_RESAMPLED_STORE_READ` / `_SERVE` missing on shadow-worker — **split verdict**

| Flag | shadow-worker reads it? | Verdict |
|---|---|---|
| `RORT_RESAMPLED_STORE_READ` | **YES** | 🔴 **Real gap** |
| `RORT_RESAMPLED_STORE_SERVE` | **No** | ✅ Correctly absent |

**`RORT_RESAMPLED_STORE_READ` — shadow-worker READS it.** Call chain, verified by hand:

```
shadow_worker.py  →  shadow_manager.py:614  from services import compute_secondary_columns
                  →  services.py:598        _coarse_secondary_store_swap(strat, sec_tf, primary_df, session)
                  →  services.py:219        if not rbs.read_enabled() …      ← RORT_RESAMPLED_STORE_READ
```

`_coarse_secondary_store_swap` is "CONSUMER #2 (M-RS2 Phase 2): serve a COARSE (≥1Hour)
secondary's OHLCV from the canonical resampled store instead of resampling the primary `df`".
With the flag unset it returns `None` immediately and the shadow lane **resamples the coarse
secondary from the primary** while Worker, Data Worker, batch-worker and api **serve it from the
canonical store**.

The swap is compare-and-fallback (it returns the store bars only when they are byte-identical to
the resample), so the two sources *should* agree — but "should agree" is the claim the store's
comparator exists to test, and it is only tested where the flag is on. Worse, the docstring names
a case where they provably differ: *"a SUB-MINUTE primary's summed volume diverges (documented
1Min != Σsub-minute), which the comparator catches → fallback"*. On the services with the flag ON
that mismatch is **detected and logged**; on shadow-worker it is invisible because the comparison
never runs. **Shadow-worker is the one engine service sourcing coarse secondaries differently
from the services it must agree with — a divergence source by construction, exactly as M
suspected.** `must-match`, currently MISMATCHED.

**`RORT_RESAMPLED_STORE_SERVE` — shadow-worker does NOT read it.** Its read site is
`strategy_data.py:236` (`_resampled_store_serve_enabled`), called at `strategy_data.py:362` inside
`_build_coarse_secondary_from_1min`, which is called only from `plan_windowed_load`
(`strategy_data.py:592`) inside **`load_strategy_data`** (`strategy_data.py:611`). The shadow lane
does not use `load_strategy_data`: `shadow_manager` prepares data through
`services.prepare_strategy_window_df` (services.py:1538), which takes the
`_secondary_snapshot_load_extend` + `prepare_data_with_indicators` route instead. `shadow_manager`
/ `shadow_worker` contain no call to `load_strategy_data` or `get_strategy_trades` (the only
mention is a docstring reference). The other read site,
`api/routers/resampled_store_admin.py:63`, is an api endpoint. **Correctly absent — not a gap.**

---

## 4. Class B — set in Railway, not read by that service

**NOT EMPTY — 6 confirmed members, 2 of them dead repo-wide.**

| Variable | Set on | Read sites in code | Verdict |
|---|---|---|---|
| `RORT_SECONDARY_TF_CACHE=1` | **batch-worker, shadow-worker, api** | **none, repo-wide** (`src/`, `frontend/`, `tools/` all clean) | 🔴 **Dead variable, and wider than the predecessor knew.** F1 recorded batch-worker only; source 2 shows it on **three** services. Still required by `local_update.py:118`'s parity manifest and `_arbitrate_primary_trigger.py:42`, so the #161 gate enforces a no-op on all three. The live reader is the *different* variable `RORT_SECONDARY_TF_SNAPSHOT`. |
| `RORT_SHADOW_POLL_DEBUG=1` | shadow-worker | **none, repo-wide** | 🔴 **New dead variable.** No reader in any file. Either its reader was refactored out or it never landed. Unlike the marker below it has no alternative explanation. |
| `RORT_SHADOW_DEPLOY_MARKER=fix1b-91903ce` | shadow-worker | **none, repo-wide** | ⚪ **Dead by design — benign.** The value is a commit-shaped tag, and changing a variable is the standard way to force a Railway redeploy. Almost certainly a deliberate redeploy trigger, not a lost flag. **Label it as such** so the next audit does not re-raise it. |
| `RORT_RECOMPUTE_PARALLELISM=8` | shadow-worker | `batch_worker.py:267`, `recompute_jobs.py:732` — neither reachable from `shadow_worker` | 🟠 Inert. §3.1. |
| `RORT_RECOMPUTE_SUPERVISOR=1` | shadow-worker | `recompute_jobs.py:764` — not reachable from `shadow_worker` | 🟠 Inert. §3.1. |
| `RORT_COMPUTE_REMOTE=1` | shadow-worker | `compute_jobs_store.py:30` — not reachable from `shadow_worker` | 🟠 Inert. §3.1. Note `batch_worker.py:258` sets it via `setdefault` in-process anyway. |
| `RORT_PARITY_SELF_HEAL=1` | batch-worker | `fidelity_parity_suite.py:130` — a **standalone script** (`python src/fidelity_parity_suite.py`), not imported by `batch_worker.py` and not spawned by it | 🟡 **Undetermined, not a defect.** Inert *unless* the suite is invoked inside the batch-worker container (e.g. `railway run`/exec), in which case the variable is doing exactly its job. **Needs one line from whoever runs the parity suite** — that is an ops fact, not a code fact. |

**Withdrawn from this class after checking:** `RORT_HEALTHCHECK_BOUNDED_BOOT` and
`RORT_HEALTHCHECK_FEED_FRESHNESS` on Worker. They are read in `engine_health.py`, which is not
imported by `worker.py` — it is Worker's container `HEALTHCHECK` command
(`Dockerfile.worker:25 CMD python src/engine_health.py`), a second process in the same container
inheriting the same environment. **Correctly set.** Recorded because the first pass of the closure
missed the healthcheck entrypoint and would otherwise have reported two false gaps.

---

## 5. Class A — read by a service, not set on it → runs the compiled default

**NOT EMPTY.** Two populations, and they must not be conflated.

**A1 — the ~100 flags set on no service at all.** Unchanged from the predecessor (§3 there): for
these the **code default IS the production value**, and a PR that edits a default is a production
config change with no Railway trace. Under a two-environment split each fleet inherits the default
from **its own deployed commit**, so a dev fleet running ahead of `main` silently runs a different
config the moment anyone edits a default. **Recommendation stands: treat "changed a `RORT_*`
default" as a release-note-worthy class of change, the same way an arm is.**

**A2 — the intersection this step exists for: 46 flag×service pairs where a service reaches the
read site but the flag is not set on it, while it *is* set somewhere else.** Someone decided each
of these flags needed an explicit value; on these services it does not have one. Full list below,
grouped by whether the absence is explained.

### A2a — explained by design (not gaps)

| Group | Flags | Why absence is correct |
|---|---|---|
| Live-only decision path | `RORT_MTF_PB_DEFER`, `RORT_MTF_PB_PREV_EPOCH`, `RORT_MTF_FINE_INCREMENTAL_AUTHORITY`, `RORT_MTF_SESSION_SHADOWS`, `RORT_MTF_COARSE_RTH_RELOAD`, `RORT_MTF_STATE_REFRESH_S`, `RORT_INTERP_AWARE_SHADOWS`, `RORT_CANONICAL_FINE_TF_STATE`, `RORT_CANONICAL_PRIMARY_CLOSE`, `RORT_GRACE_FINAL_CLOSE_ELIGIBLE`, `RORT_SESSION_LABEL_GATE`, `RORT_TF_LABEL_SEC_FIX`, `RORT_SUBMIN_DERIVE_BARS`, `RORT_SHADOW_RETRUE_FORCE_FULL`, `RORT_RESAMPLED_STORE_SERVE_LIVE`, `RORT_PRIMARY_STATE_RESYNC_S`, `RORT_PRIMARY_STATE_RESYNC_APPLY` | Read sites are in `ralph_engine`'s live `StrategyMonitor` / MTF refresher machinery, reached only from `worker.py`'s live loop. The offline services import `ralph_engine` but never enter those paths. Worker-only is correct. |
| Live bar construction | `RORT_BAR_DUP_GUARD` | `ralph_engine.py:257`, BarBuilder — builds bars from the live tick/aggregate stream. Offline lanes load finished bars; there is nothing to de-duplicate. |
| Live-transient guard | `RORT_GATE_FAIL_CLOSED` | `unified_engine.py:52`, and its own docstring settles it: *"Backtest derives records per bar (non-empty for gated strategies), so this is effectively a LIVE-transient guard."* Worker-only is deliberate and documented. **Contrast with §2 — same module, opposite verdict; the docstring is the difference.** |
| Store writer / sweeper singletons | `RORT_RESAMPLED_STORE_WRITE`, `RORT_RESAMPLED_STORE_MAINTAIN_S`, `RORT_SETTLE_SWEEPER`, `RORT_SETTLE_SWEEP_INTERVAL_S`, `RORT_BARCACHE_WRITETHROUGH` | Set on Data Worker only. These are **singleton writers against shared tables**; exactly one service must own each. A second owner is a hazard, not parity. |
| Nightly scheduler | `RORT_NIGHTLY_SETTLE_RETRUE`, `..._WINDOW`, `RORT_PARITY_SETTLED_MIN_TDAYS` | batch-worker only — it owns the nightly. Data Worker reaches `settle_sweeper` for the maintain hook but does not run the nightly. |
| api-side only | `RORT_RECOMPUTE_PARALLELISM`, `RORT_RECOMPUTE_SUPERVISOR` absent on api | api can enqueue recomputes but does not run the pool; batch-worker owns it. |
| Shadow-lane only | `RORT_SHADOW_CONFIG_CACHE_TTL_S`, `RORT_PREP_BAR_COUNT_WARMUP` | Read from `shadow_manager` / the shadow prep path; no other service enters it. |

### A2b — unexplained, and therefore owed an answer

| Flag | Set on | Missing on | Why it is not obviously fine |
|---|---|---|---|
| 🔴 `RORT_WARMUP_PREV_CHAIN` | Worker | Data Worker, batch-worker, shadow-worker, api | **§2.** Shared `unified_engine`; the guarded code is warmup + full replay, i.e. the offline lanes' main job. |
| 🔴 `RORT_RESAMPLED_STORE_READ` | Worker, Data Worker, batch-worker, api | **shadow-worker** | **§3.2.** Verified read on the shadow path; different coarse-secondary source from every peer. |
| 🟠 `RORT_APPEND_SUPERSEDE_OPEN` | Worker, batch-worker, api | Data Worker, shadow-worker | `forward_test_service.py:48`. Data Worker runs streaming backtest catchup and shadow-worker writes settled trades — both plausibly append. Default OFF, so they would use legacy supersede semantics. **Needs a path check.** |
| 🟠 `RORT_HIFI_NO_PREBAR_FILL` | Worker, batch-worker, api | Data Worker, shadow-worker | `backtest_service.py:36`, fill policy (#113/#112). Same two services, same reasoning — it changes fill prices, hence KPIs. |
| 🟠 `RORT_HIFI_INCREMENTAL_LOAD` | **shadow-worker only** | Worker, Data Worker, batch-worker, api | Read at `api/routers/strategies.py:805` inside `run_hifi_pass2`, which is called from `forward_test_service` (4 sites) and `data_worker_engine.py:851` — so batch-worker, Data Worker and api all execute it. Default is `"1"`, so behaviour agrees today **by luck of the default**. The arm is recorded on one service and relied upon by four. |
| 🟠 `RORT_RESUME_INHERIT_POSITION`, `RORT_RIGHTSIZE_WARMUP`, `RORT_UPDATE_ALL_SKIP_ALGO` | Worker, batch-worker, shadow-worker, api | **Data Worker** | Set on 4 of 5; Data Worker is the lone omission and it does run backtest catchup. Warmup sizing especially is never a per-service knob. |
| 🟠 `RORT_UAD_GUARD_EMPTY_LANE` | Worker, batch-worker, api | Data Worker, shadow-worker | Default-ON in code, so armed everywhere anyway — but there is no Railway variable to disarm it on two services in an emergency. |
| 🟠 `RORT_PRIME1S_CACHE_MAX_ROWS` | Worker, batch-worker, api | Data Worker, shadow-worker | A memory bound (#68). Unset = `0` = **unbounded** on those two. Sized to the box, so `may-differ` — but "absent" here means "no bound at all", which is a different thing from "a different bound". |
| 🟠 `RORT_SCOPE_CONFLUENCE_GROUPS`, `RORT_SECONDARY_TF_SNAPSHOT`, `RORT_ENFORCE_1MIN_GATE`, `RORT_COARSE_SECONDARY_FROM_1MIN`, `RORT_USE_DOUGH_CACHE` | batch-worker, shadow-worker, api | **Worker, Data Worker** | All five are code-default-ON except `RORT_ENFORCE_1MIN_GATE` (default `"0"`). **`RORT_ENFORCE_1MIN_GATE` is the one to look at**: armed on the three offline services, absent on Worker — the mirror image of §2, and it is the coarse-gate contract. `strategy_data.py`/`data_loader.py` are reachable from `worker.py`. **Needs a path check before it is called either way.** |

**Every 🟠 row is "needs a determination", not "is a defect".** They are listed because an absent
section reads identically to a check that was never run — and because I could hand-verify two
chains in this run, not twelve. The verification method is in §1 and the tool is committed;
finishing the 🟠 rows is mechanical, not novel.

---

## 6. Class C — set on some services but not others

**NOT EMPTY — and now determinable, which it was not before.** 65 of 67 flags are in this shape.
**That count is not the finding**, per §0. The finding is the subset where the shape is *not*
explained by which services execute the read site — which is exactly §5's A2b list, headed by
`RORT_WARMUP_PREV_CHAIN` and `RORT_RESAMPLED_STORE_READ`.

**Differing-value subclass:** exactly **one** flag is set to different values anywhere —
`RORT_RECOMPUTE_PARALLELISM` (batch-worker `6`, shadow-worker `8`) — and §3.1 shows it is not a
live conflict because shadow-worker cannot read it.

---

## 7. Predecessor findings — status after source 2

| # | Was | Now |
|---|---|---|
| **F0** | Source 2 blocked, `railway variables` denied at the headless layer | ✅ **CLEARED** by M from an attended session. The permission asymmetry is real and worth narrowing to `--set` so a read-only listing is not blocked, but it no longer blocks this inventory. |
| **F1** | `RORT_SECONDARY_TF_CACHE` dead on batch-worker | 🔺 **WIDENED** — dead on **batch-worker, shadow-worker AND api**. Still enforced by the #161 parity manifest. |
| **F2** | data-worker and shadow-worker had no committed flag record | ✅ **CLEARED** — both captured (9 and 27 settings). The `tools/preflight/expected_flags.json` service-key defect it also reported is unchanged and still open. |
| **F3** | `replay_harness.ARMED_FLAGS` omits `RORT_BAR_DUP_GUARD` | ⚪ **Unchanged** — source 2 confirms `RORT_BAR_DUP_GUARD=1` really is on Worker, so the mirror really is incomplete. Still likely harmless (the harness bypasses BarBuilder), still undocumented. |
| **F4** | `RORT_SUPPRESS_EOD_REENTRY` maybe missing on Worker | ✅ **CLEARED — it is set on all five services.** The suspected live-lane gap does not exist. |
| **F5** | `expected_flags.json` `_meta.populated = false`, drift check never ran | 🟢 **NOW UNBLOCKED** — the data to populate it is in this document and in #265. Still not populated; still an E-attended action. |
| **F6** | `system_settings` global keys ambiguous under two fleets | ⚪ **Unchanged** — still a split prerequisite for M's scoping. |

**New findings from this pass:** the §2 headline (`RORT_WARMUP_PREV_CHAIN`), the §3.2 gap
(`RORT_RESAMPLED_STORE_READ` on shadow-worker), and three new Class-B members
(`RORT_SHADOW_POLL_DEBUG`, plus `RORT_RECOMPUTE_PARALLELISM`/`_SUPERVISOR`/`RORT_COMPUTE_REMOTE`
inert on shadow-worker).

---

## 8. For #165 Phase A — the rollback baseline

M's capture doubles as the **pre-change record** that Phase A step 11 diffs against to prove the
rename moved nothing. The numbers that must survive the split, per service:

```
Worker 34   Data Worker 9   batch-worker 26   shadow-worker 27   api 18      total 114
```

**Nothing in this document should be acted on as part of the split.** Every 🔴/🟠 row is a
pre-existing condition of the current production fleet. Fixing one during the rename would
destroy the very thing the rollback record is for — and #165's own framing is that `main` is
*"the version whose divergences we have characterized and accepted"*, not the version with none.
**Characterizing them is this document; accepting or fixing them is a separate decision.**

---

## 9. Status against the step's done-when

- **Every entry classified with a reason** — all 136 code-declared tokens in the companion file;
  all 67 set flags crossed against them here, each with its verdict and evidence.
- **The read-vs-set intersection is stated** — §5 A2, 46 pairs, split into explained (A2a) and
  owed-an-answer (A2b), with the method's upper-bound limit declared in §1.
- **All three mismatch classes explicit, including any that are empty:**
  - **Class A — NOT EMPTY** (§5: ~100 unset-anywhere + 46 read-but-unset pairs)
  - **Class B — NOT EMPTY** (§4: 6 members, 2 dead repo-wide, 1 undetermined, 2 withdrawn on check)
  - **Class C — NOT EMPTY** (§6: 65 of 67; exactly 1 differing-value member, resolved)
  - *No class is empty. Had one been, it would say so here rather than be omitted.*
- **Nothing flipped, set, unset, restarted or redeployed.** No credential values recorded.
