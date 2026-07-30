# Live verification record — auto-push at run end (board #195, step 5)

**Written by** M·auto, run `r1785383117-195`, 2026-07-30 ~03:50Z.
**Status of #195 at the time of writing:** built (V4.23), merged to `dev` in Wave 13
(`1c8a6354`), loop restarted onto V4.24. **Not yet proven by a live push.**

This file exists because the hermetic suite and the live proof are different claims,
and only the first one was on the record. The suite (`src/test_dispatcher_push_at_run_end_195.py`,
97 checks) proves *the instruction exists*. Only a real dispatched run whose branch
reaches `origin` unaided proves *the instruction fires*.

---

## Act 1 — the loop actually holds the new code

Not "the file on disk is current" — the *running process* must have been started after it.

```
dispatcher.py written : 2026-07-29 21:39:10 -0600
loop pid 188535 started: 2026-07-29 21:39:18 -0600   (+8s)
git diff origin/dev -- tools/team_dispatcher/dispatcher.py : empty
dispatcher.py header  : V4.24 (includes V4.23 = #195 push leg)
```

The 8-second gap is the whole point: a loop started *before* the file landed would
report the same clean `git diff` while running the old code from memory. Preflight (5)
("dispatcher loop alive and file matches dev") agrees, but the process-start-vs-mtime
comparison is the stronger check and is what should be quoted in future.

**Act 1: PASS.**

## Act 2 — the live push

### Why the two runs in flight could not both be the subject

Only two runs were ever claimed under a V4.23+ loop (everything earlier in
`run_history` shows `pushed_branch = NULL` because the leg did not exist in the
process that reaped them):

| run | task | lane | can it prove the push? |
|-----|------|------|------------------------|
| `r1785382865-212` | #212 | R | **No.** A release session has push authority and opens its own PRs, so its branch is already on `origin` by reap time → the leg correctly refuses with `exists-on-origin`. A refusal is not a proof. |
| `r1785383117-195` | #195 | M | **Yes** — this run. |

This is worth remembering: **R runs can never be the subject of this verification.**
The subject has to be a lane that genuinely cannot push.

### Why this run's *registered* worktree cannot be the subject either

```
worktree : /home/kevin/projects/Kevbot-Toolkit/clients/KevBot_Toolkit/RoR_Trader
branch0  : dev
```

M·auto is dispatched into the **main checkout, on `dev`**. Both rails fire there and
both are right: `dev` is in `PROTECTED_BRANCHES`, and `branch == branch0` means the run
did not create it. So the registered-worktree leg is structurally dead for M·auto, and
the **fresh-worktree leg** (charter §4: `git worktree add …`) is the only one that can
ever carry M's work. That is the leg under test.

### The subject

```
worktree : /home/kevin/projects/Kevbot-wt-verify195   (created 03:49Z, after dispatch at 03:45Z)
branch   : verify/auto-push-live-195                  (cut from origin/dev per board #153)
```

Every rail in `push_worktree_branch()` was evaluated by hand before the run ended —
without pushing, because a human push destroys the experiment:

| rail | required | actual |
|------|----------|--------|
| kill switch `NO_PUSH` | absent | absent |
| worktree on record | exists | yes |
| `branch0`/`worktrees0` present | yes | yes (30-entry snapshot) |
| not detached | named branch | `verify/auto-push-live-195` |
| not protected | ∉ {dev, main, master} | ok |
| clean tree | `status --porcelain` empty | empty |
| ahead of `origin/dev` | ≥ 1 | 1 |
| new to origin | `ls-remote` empty | empty |
| fresh worktree | ∉ `worktrees0` | ∉ (created 4 min after dispatch) |

### The one rail that decides it: rival ambiguity

`fresh_worktrees()` declines a worktree if **any other run still in flight** also
predates it — ownership is inferred, so it is inferred conservatively. Run #212 was
dispatched at 03:41Z, four minutes before this worktree existed, so:

- if #212 has **already reaped** when this run reaps → no rival → **push fires**;
- if #212 is **still active** → `SKIP … owner ambiguous` → correct refusal, no proof.

**This is a real property of the design, not a wrinkle of tonight.** On a busy loop the
conservative rail will decline pushes whenever lanes overlap, which is most of the
time. That is the safe direction to fail — pushing a peer's half-finished branch would
be genuinely harmful, and preflight (2) still catches the remainder — but it does mean
**#195 closes the loop for quiet periods and leaves it open for busy ones**. Worth a
follow-up (ownership recorded at worktree-creation time rather than inferred at reap),
which is out of scope for step 5 and belongs on the board.

---

## How to confirm — the check that closes #195

The one-shot contract means the verifying run cannot watch its own reap: the push
happens *after* the process is gone. So the confirmation is deliberately a one-liner
for the next reader, and it is exactly the check step 5's own SOP names:

```bash
git ls-remote --heads origin verify/auto-push-live-195     # a sha = the loop pushed it
grep "push\[r1785383117-195\]" /tmp/dispatcher.log         # the decision, in its own words
```

and in the DB, the rail that feeds #193's dashboard:

```sql
select run_id, pushed_branch, pushed_at from run_history where run_id = 'r1785383117-195';
```

`pushed_branch` non-null there is the stronger signal — it proves the push *and* the
`run_history` write, which is the half the dashboard depends on.

Three outcomes, all informative:

1. **sha present + `pushed_branch` set** → #195 is PROVEN. Close it. Delete this
   worktree and branch; they were scaffolding.
2. **`SKIP … owner ambiguous`** → the leg ran and declined correctly; #212 was still in
   flight. Re-run step 5 on a quiet loop; the code is not implicated.
3. **no `push[...]` line at all** → the leg did not execute. That is the only outcome
   that reopens the build.

Preflight (2) is the passive version of the same check: if it stops listing this branch
without anyone pushing it, the loop pushed it.
