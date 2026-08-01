# Environment split — pre-change rollback record

**Board #165 · Phase A step 7 · captured 2026-07-31 / 2026-08-01 UTC · written by E·auto (`r1785554434-165`)**

---

## 0. What this file is, in one sentence

This is the **"before" photograph** of the single live Railway environment and the two git refs,
taken immediately prior to the V5 environment split — so that every later step of #165 can be
**undone against a record rather than against a memory**, and so step 11 ("verify production is
unchanged") has something concrete to diff against.

**Freeze point:** the state described here is the state as of the deploys listed in §2, all of
which are `dev @ 2cad3782` (except `shadow-worker`, see §2.1). Nothing in #165 Phase A had been
executed when this was taken: `main` was still the 2026-03 placeholder, there was still exactly
one Railway environment, and every service still pointed at `dev`.

**This file is the durable artifact.** Board comments are not — they are not versioned, not
diffable, and not present in a checkout. Everything needed to restore is reproduced here in full,
deliberately, rather than referenced.

---

## 1. Provenance — and the one thing that is *not* verified

Read this section before trusting any number below it.

| Section | Source | Verified by E·auto? |
|---|---|---|
| §3 git refs | `git fetch origin` + `git rev-parse`, run in this worktree | ✅ **Yes — re-derived first-hand** |
| §2 deploy IDs / branch map / domains | **attended M session**, `railway status` / deploy listing, 2026-08-01 UTC, posted to board #165 | ⚠️ **No — transcribed. See below.** |
| §4 per-service `RORT_*` values | **attended M session**, `railway variables --kv` × 5 services, 2026-07-31, posted to board #265 | ⚠️ **Not re-pulled from Railway. Cross-confirmed against a second independent transcription — see §5.** |

### ⚠️ The honest bound: E could not re-pull from Railway

The step SOP asked for the flag values to be **re-pulled and confirmed against live**. That was
not possible: **the `railway` CLI is denied to headless E at the permission layer** — 7 of 7
invocation forms refused, confirmed across three separate runs on 07-31/08-01. Every
Railway-gated fact in this file therefore originates from an **attended** capture, not from a
first-hand read by the agent that wrote the file.

What was done instead, and what it is worth:

- The flag capture was **cross-checked flag-for-flag and value-for-value against a second,
  independently-transcribed durable source** (`Env_Config_Parity_Inventory.md`, §5 below).
  **Zero mismatches across all 67 distinct flags.** Two independent transcriptions of the same
  live read agreeing exactly rules out transcription error. It does **not** rule out the values
  having *changed on Railway* since 2026-07-31.
- **Therefore:** treat §4 as accurate as of **2026-07-31**, not as of the moment you read it.
  If any arm, deploy or var-set happened between that capture and the start of Phase A step 8,
  this record is stale by exactly that delta. **No arm is recorded in `Deploy_Log.md` after the
  capture**, but the correct guard is procedural, not evidentiary:

> **Before the fast-forward (step 8) and before the rename (step 9), an attended session should
> re-run `railway variables --kv` once per service and diff against §4.** That is a two-minute
> check and it converts this record from "very probably current" to "current". It is the one
> action this record cannot self-certify.

---

## 2. Railway — deployments, branch mapping, domains

Single environment, 8 services. Captured attended, 2026-08-01 UTC.

| Service | Last deploy | Deployment ID (rollback target) | Branch @ commit | Generated domain |
|---|---|---|---|---|
| `api` | SUCCESS | `c3743525-75a0-4bc5-a188-7faab3fe5a90` | `dev` @ `2cad3782` | **`api-dev-2c9d.up.railway.app`** |
| `Worker` | SUCCESS | `2fe7608e-f5e9-46ce-8bc1-06214a53604d` | `dev` @ `2cad3782` | **`worker-dev-06f0.up.railway.app`** |
| `frontend` | SUCCESS | `c6e25308-3409-4bac-aa3d-8fbedd8f85e4` | `dev` @ `2cad3782` | **`frontend-dev-e01a.up.railway.app`** |
| `RoR Trader - Streamlit` | SUCCESS | `7b1aa3b1-d086-44ee-acf4-b75cb67f2080` | `dev` @ `2cad3782` | `ror-trader-app-dev.up.railway.app` |
| `flat-file-cron` | SUCCESS | `482175b8-3e52-487b-9b96-453b476cd214` | `dev` @ `2cad3782` | `flat-file-cron-dev.up.railway.app` |
| `Data Worker` | SUCCESS | `6c9cb9a9-5d48-4e9c-b073-2350c1182112` | `dev` @ `2cad3782` | — (no public domain) |
| `batch-worker` | SUCCESS | `bdff3c68-c68e-4b9b-8db7-6c47627e7db1` | `dev` @ `2cad3782` | — (no public domain) |
| `shadow-worker` | SUCCESS | `f7513e4b-5af3-4f95-aacc-bc407f8982fe` | ⚠️ **none reported** | — (no public domain) |

### 2.1 Two things to carry forward, not to fix here

- **`shadow-worker` reports no branch and no commit**, and its last deploy is **2026-07-23** while
  every other service is 07-31. Recorded as observed. **Not treated as an error** — it may simply
  be a service deployed by `railway up` rather than by git trigger, which is exactly how
  [[project_mrs4_deploy_mechanism]] says the shadow worker is deployed. The consequence for
  rollback is concrete and worth stating: **`shadow-worker` has no branch to repoint and its
  restore path is its deployment ID above, nothing else.**
- **`frontend` carries no `RORT_*`** and so is absent from the 5-service flag capture in §4 — but
  it **does** have a generated domain, and the domain is precisely what step 9 tests. A
  flag-only record would have silently dropped it. It is in §2 for that reason.

### 2.2 Why the domains are in this record at all

Every live service domain has **`-dev` baked into the hostname**. The environment rename in step 9
is *expected* to leave hostnames untouched — but that is **expected behaviour, not tested
behaviour**, and step 9 is the test. §2 is the only "before" the test can compare against. If a
rename regenerates or invalidates a hostname, the three bolded domains above are the ones that
break something: `api-dev-2c9d` is the board API this very chain runs against.

---

## 3. Git refs — before

```
origin/dev    2cad378289b3edfd0253901d36f971289b1cc54f
origin/main   fd4beeada7d70f1caa5b97ed73dde9064eca5ba0
```

Re-derived first-hand in this worktree, 2026-08-01:

| Ref | Tip date | Subject |
|---|---|---|
| `origin/dev` | 2026-07-31 16:56 -0600 | `Merge pull request #214 from ktkventures/docs/deploy-log-wave38-0731` |
| `origin/main` | **2026-03-20** 14:54 -0600 | `Update PRD: Phase 31A-31E verified, ready for merge to main` |

**The fast-forward in step 8 is clean, and this is the proof:**

```
git rev-list --count origin/main..origin/dev   →  1772
git rev-list --count origin/dev..origin/main   →  0
git merge-base origin/main origin/dev          →  fd4beeada7d70f1caa5b97ed73dde9064eca5ba0
```

`main` is a strict ancestor of `dev`: **1,772 commits behind, 0 ahead.** The merge base *is*
`main`'s tip. So step 8 is a true fast-forward with nothing to reconcile and nothing on `main`
that would be lost — which is also the arithmetic proof of Kevin's "`main` is a placeholder"
call: it has been untouched since **2026-03-20**, four months.

**Remote:** `git@github.com:ktkventures/Kevbot-Toolkit.git` (fetch and push). Recorded because
Phase R repoints Railway at a different repo — this is the "before" for that too.

**To roll back step 8:** `main` is restored by pointing it back at
`fd4beeada7d70f1caa5b97ed73dde9064eca5ba0`. Because the FF adds only commits and moves no other
ref, step 8 in isolation is the cheapest step in the chain to undo.

---

## 4. Per-service `RORT_*` values — verbatim

**114 settings across 5 services · 67 distinct flags · Worker = 34.**
`railway variables --kv`, attended, 2026-07-31. Booleans, integers, durations and mode strings
only — **no credentials are captured here, and none belong in this file.** The `frontend`,
`Streamlit` and `flat-file-cron` services carry no `RORT_*` and are not listed in this section
(they are in §2).

> **34 on `Worker` is the number that must survive the split.** It is the count step 11 checks.

#### `Worker` — 34 settings

```
RORT_APPEND_SUPERSEDE_OPEN=1
RORT_BAR_DUP_GUARD=1
RORT_CANONICAL_FINE_TF_STATE=1
RORT_CANONICAL_PRIMARY_CLOSE=1
RORT_CANONICAL_SUBMIN_STATE=1
RORT_GATE_FAIL_CLOSED=1
RORT_GRACE_FINAL_CLOSE_ELIGIBLE=1
RORT_HEALTHCHECK_BOUNDED_BOOT=1
RORT_HEALTHCHECK_FEED_FRESHNESS=1
RORT_HIFI_NO_PREBAR_FILL=1
RORT_HOTRELOAD_BOOT_PARITY=1
RORT_INTERP_AWARE_SHADOWS=1
RORT_MTF_COARSE_RTH_RELOAD=1
RORT_MTF_FINE_INCREMENTAL_AUTHORITY=1
RORT_MTF_PB_DEFER=1
RORT_MTF_PB_PREV_EPOCH=1
RORT_MTF_SESSION_SHADOWS=1
RORT_MTF_STATE_REFRESH_S=120
RORT_PRIMARY_STATE_RESYNC_APPLY=0
RORT_PRIMARY_STATE_RESYNC_S=0
RORT_PRIME1S_CACHE_MAX_ROWS=2000000
RORT_RESAMPLED_STORE_READ=1
RORT_RESAMPLED_STORE_SERVE=1
RORT_RESAMPLED_STORE_SERVE_LIVE=1
RORT_RESUME_INHERIT_POSITION=1
RORT_RIGHTSIZE_WARMUP=1
RORT_SESSION_LABEL_GATE=1
RORT_SHADOW_RETRUE_FORCE_FULL=1
RORT_SUBMIN_DERIVE_BARS=3000
RORT_SUPPRESS_EOD_REENTRY=1
RORT_TF_LABEL_SEC_FIX=1
RORT_UAD_GUARD_EMPTY_LANE=1
RORT_UPDATE_ALL_SKIP_ALGO=1
RORT_WARMUP_PREV_CHAIN=1
```

#### `Data Worker` — 9 settings

```
RORT_BARCACHE_WRITETHROUGH=1
RORT_CANONICAL_SUBMIN_STATE=1
RORT_RESAMPLED_STORE_MAINTAIN_S=900
RORT_RESAMPLED_STORE_READ=1
RORT_RESAMPLED_STORE_SERVE=1
RORT_RESAMPLED_STORE_WRITE=1
RORT_SETTLE_SWEEPER=1
RORT_SETTLE_SWEEP_INTERVAL_S=300
RORT_SUPPRESS_EOD_REENTRY=1
```

#### `batch-worker` — 26 settings

```
RORT_APPEND_SUPERSEDE_OPEN=1
RORT_CANONICAL_SUBMIN_STATE=1
RORT_COARSE_SECONDARY_FROM_1MIN=1
RORT_COMPUTE_REMOTE=1
RORT_ENFORCE_1MIN_GATE=1
RORT_HIFI_NO_PREBAR_FILL=1
RORT_NIGHTLY_RECOMPUTE=1
RORT_NIGHTLY_RECOMPUTE_AT=00:20
RORT_NIGHTLY_SETTLE_RETRUE=1
RORT_NIGHTLY_SETTLE_RETRUE_WINDOW=5
RORT_PARITY_SELF_HEAL=1
RORT_PARITY_SETTLED_MIN_TDAYS=4
RORT_PRIME1S_CACHE_MAX_ROWS=2000000
RORT_RECOMPUTE_PARALLELISM=6
RORT_RECOMPUTE_SUPERVISOR=1
RORT_RESAMPLED_STORE_READ=1
RORT_RESAMPLED_STORE_SERVE=1
RORT_RESUME_INHERIT_POSITION=1
RORT_RIGHTSIZE_WARMUP=1
RORT_SCOPE_CONFLUENCE_GROUPS=1
RORT_SECONDARY_TF_CACHE=1
RORT_SECONDARY_TF_SNAPSHOT=1
RORT_SUPPRESS_EOD_REENTRY=1
RORT_UAD_GUARD_EMPTY_LANE=1
RORT_UPDATE_ALL_SKIP_ALGO=1
RORT_USE_DOUGH_CACHE=1
```

#### `shadow-worker` — 27 settings

```
RORT_BACKTEST_LANE_MODE=shadow
RORT_CANONICAL_SUBMIN_STATE=1
RORT_COARSE_SECONDARY_FROM_1MIN=1
RORT_COMPUTE_REMOTE=1
RORT_ENFORCE_1MIN_GATE=1
RORT_HIFI_INCREMENTAL_LOAD=1
RORT_PREP_BAR_COUNT_WARMUP=1
RORT_RECOMPUTE_PARALLELISM=8
RORT_RECOMPUTE_SUPERVISOR=1
RORT_RESUME_INHERIT_POSITION=1
RORT_RIGHTSIZE_WARMUP=1
RORT_SCOPE_CONFLUENCE_GROUPS=1
RORT_SECONDARY_TF_CACHE=1
RORT_SECONDARY_TF_SNAPSHOT=1
RORT_SHADOW_CONFIG_CACHE_TTL_S=60
RORT_SHADOW_DEPLOY_MARKER=fix1b-91903ce
RORT_SHADOW_DRY_RUN=0
RORT_SHADOW_FAIR_ORDER=1
RORT_SHADOW_KPI_ASYNC=1
RORT_SHADOW_MAX_ADVANCE_S=0
RORT_SHADOW_POLL_DEBUG=1
RORT_SHADOW_POLL_WORKERS=1
RORT_SHADOW_RESIDENT_FRAME=1
RORT_SHADOW_SIDS=all
RORT_SUPPRESS_EOD_REENTRY=1
RORT_UPDATE_ALL_SKIP_ALGO=1
RORT_USE_DOUGH_CACHE=1
```

#### `api` — 18 settings

```
RORT_APPEND_SUPERSEDE_OPEN=1
RORT_CANONICAL_SUBMIN_STATE=1
RORT_COARSE_SECONDARY_FROM_1MIN=1
RORT_COMPUTE_REMOTE=1
RORT_ENFORCE_1MIN_GATE=1
RORT_HIFI_NO_PREBAR_FILL=1
RORT_PRIME1S_CACHE_MAX_ROWS=2000000
RORT_RESAMPLED_STORE_READ=1
RORT_RESAMPLED_STORE_SERVE=1
RORT_RESUME_INHERIT_POSITION=1
RORT_RIGHTSIZE_WARMUP=1
RORT_SCOPE_CONFLUENCE_GROUPS=1
RORT_SECONDARY_TF_CACHE=1
RORT_SECONDARY_TF_SNAPSHOT=1
RORT_SUPPRESS_EOD_REENTRY=1
RORT_UAD_GUARD_EMPTY_LANE=1
RORT_UPDATE_ALL_SKIP_ALGO=1
RORT_USE_DOUGH_CACHE=1
```

### 4.1 Counts, for the step-11 diff

| Service | Settings |
|---|---|
| `Worker` | **34** |
| `shadow-worker` | 27 |
| `batch-worker` | 26 |
| `api` | 18 |
| `Data Worker` | 9 |
| **Total** | **114** (67 distinct flags) |

---

## 5. How the flag values were confirmed without Railway

The 114 settings above were parsed out of the board #265 capture programmatically (not by eye)
and compared against the **`Set on` / `Value`** cells of §2 of
`docs/_active/Env_Config_Parity_Inventory.md` @ `50ee559f` — a **separate transcription of the
same live read, made by a different run**, in a different format (per-flag rows rather than
per-service blocks).

**Result: zero mismatches.**

- **65 of 67** distinct flags have a `LIVE@07-31` row in inventory §2. For every one of them,
  both the **set of services** carrying the flag and the **value** agree exactly with §4 above.
- The remaining **2** — `RORT_SHADOW_DEPLOY_MARKER=fix1b-91903ce` and `RORT_SHADOW_POLL_DEBUG=1`
  — have no §2 row **by construction**: §2 is keyed on code read sites, and these two have zero
  mentions anywhere in the repo. They are named with their values in inventory §7 class B
  (finding **F8**), and those values match §4. **So all 67 are corroborated.**
- The declared per-service counts (34 / 27 / 26 / 18 / 9) were checked against the parsed line
  counts, and the totals against 114 / 67. All agree.

**What this establishes and what it does not.** It establishes that the values in §4 are a
faithful transcription of what Railway returned on 2026-07-31. It does **not** establish that
Railway still returns them — see the warning in §1.

### 5.1 ⚠️ The inventory is not on `dev`

`Env_Config_Parity_Inventory.md` @ `50ee559f` lives **only** on the branch
`docs/env-config-parity-inventory-265-r1785552074`. It is **not merged into `dev`**, so it will
**not** be carried onto `main` by the step-8 fast-forward.

That is exactly why §4 reproduces every value in full rather than pointing at it. **A rollback
record whose data lives on an unmerged branch is not a rollback record.** Merging that branch is
worth doing on its own merits, but this file does not depend on it.

---

## 6. Restore procedure — per step of Phase A

Ordered so that undoing later steps first is the natural read.

| If this needs undoing | Restore to | Using |
|---|---|---|
| Step 12 — dev user account wiring | n/a (additive) | Remove the dev-side account; no production object is touched |
| Step 11 — new `dev` environment variables | n/a (additive) | Delete the environment; production is a different environment |
| Step 10 — new `dev` environment created | n/a (additive) | Delete the environment |
| **Step 9 — environment rename + repoint to `main`** | the 8 services in **§2**: same deployment IDs, same domains, all on branch **`dev`** | Rename back; repoint each service's branch to `dev`; verify **every domain in §2 verbatim**, including `frontend` |
| **Step 8 — `main` fast-forwarded to `dev`** | `origin/main` → **`fd4beeada7d70f1caa5b97ed73dde9064eca5ba0`** | See §3 |
| Any service deploy that goes bad at any point | the **deployment ID** for that service in §2 | Railway rollback to that deployment |

**The two facts that make a rollback possible at all**, and which existed nowhere durable before
this file: the **deployment IDs** (§2) — without them a Railway rollback has no target — and the
**domains** (§2) — without them the rename has nothing to be checked against.

---

## 7. What this record deliberately does NOT cover

Stated so that no one mistakes its silence for coverage.

1. **Non-`RORT_` Railway variables** — database URLs, API keys, `PORT`, service tokens. They are
   **credentials and must never enter git.** They are unchanged by a rename and are not part of
   any rollback this chain performs. Their existence is inventoried, without values, in
   `Env_Config_Parity_Inventory.md` §6.
2. **`system_settings` rows** (5, live-read 2026-07-31) — inventoried in
   `Env_Config_Parity_Inventory.md` §4. They live in **Supabase, which is shared and not split**,
   so the split cannot change them and there is nothing to restore.
3. **`src/.env`** — the local workstation mirror. Gitignored, holds credentials, out of scope;
   the parity question it raises is board **#161**.
4. **The ~100 flags whose production value is a code default.** Only **65** of the 127 flags with
   a real read site are set anywhere; the rest run their compiled default. **Those defaults are
   production configuration with no Railway trace, and no `railway variables` diff will ever show
   a change to one.** This record cannot capture them, and after the split the two fleets inherit
   defaults from **their own deployed commit**. Inventory §3 makes the recommendation: treat
   "changed a `RORT_*` default" as a release-note class of change, the way an arm is.
5. **Anything about fidelity or divergence.** Per #165's own framing, those are measurable only
   against live market data and the production DB. This record protects **configuration**; it
   says nothing about whether the promoted code behaves the same.

---

## 8. Provenance trail

| Item | Where it came from |
|---|---|
| Deployment IDs, branch map, domains | Board **#165**, attended M capture, 2026-08-01 |
| Per-service `RORT_*` values | Board **#265**, attended M capture (`railway variables --kv`), 2026-07-31 |
| Cross-check second source | `docs/_active/Env_Config_Parity_Inventory.md` @ `50ee559f` (branch `docs/env-config-parity-inventory-265-r1785552074`, **not on `dev`**) |
| Git refs, FF arithmetic, tip metadata, remote | Re-derived first-hand, this worktree, 2026-08-01 |
| Written by | E·auto, run `r1785554434-165`, branch `docs/env-split-rollback-record-165` |
