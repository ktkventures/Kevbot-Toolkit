# Diagnosis — headless runs get 401 on `/steps/complete` (board #217, chain step 2)

**Author:** M·auto · 2026-07-30 · **Mode:** DIAGNOSE ONLY — no fix in this branch.
**Blocks:** #202 (the STEP-TICK CONTRACT) · **Next:** step 3, M picks the fix.

---

## TL;DR

**The leading hypothesis is FALSIFIED.** The variable is not *which API base URL each
run resolved*. It is **whether the run went through HTTP at all**.

Neither "successful tick" ever called the endpoint:

| run | what the board shows | what actually happened |
|---|---|---|
| `r1785369528-202` | `#202 step 2 … complete (by M·auto)` | **in-process Python call** to the real handler — `DT.complete_step(202, {'actor':'M·auto'}, user=None)`. No HTTP, no ASGI, so `Depends(get_current_user)` never ran. |
| `r1785377136-184` | `#184 step 1 … complete (by M·auto)` | **never called the endpoint.** The run *authored* the chain and constructed step 1 already `done=True` with hand-written `completed_at`/`completed_by`, written straight to PostgREST with the service-role key. |
| `r1785389489-211` / `r1785391015-211` | 401 | the only runs that actually made the HTTP call. |

So the real scoreboard is **0 for 1**: every run that has ever issued
`POST /api/dev-tasks/<id>/steps/complete` over HTTP has been refused. The endpoint has
never once worked for a headless run. The two "successes" are two *different* bypasses of
the HTTP layer, and the 401 is the correct, reproducible baseline.

A weaker version of the base-URL hypothesis does survive and is worth fixing on its own
merits: **there is no base URL anywhere in the agent's world** (§3). But making it
deterministic fixes nothing by itself — the deployed dev API 401s regardless (§2).

---

## 1. Evidence

### 1.1 The failing run really did call HTTP, and had to *guess* the host

From the run transcript (`368ff216-…jsonl`), in order:

```bash
# it had to go looking for a URL — the prompt gave it a bare path
grep -rn "up.railway.app\|\.railway\.app" src/.env docs/_active/Session_Charters.md

curl -s -X POST "https://api-dev-2c9d.up.railway.app/api/dev-tasks/211/steps/complete" \
     -H "Content-Type: application/json" -d '{"actor":"M·auto"}' -w "\nHTTP %{http_code}\n"
# → 401 {"detail":"Missing Bearer token"}

# then tried the local fallback
curl -s -m 4 -o /dev/null -w "8000:%{http_code}\n" http://localhost:8000/health
# → nothing listening
```

That is not a run that fumbled a URL. It found the only reachable API, called it
correctly, was refused, and **declined to work around it** — exactly per contract.

### 1.2 Reproduced live, just now

```
$ curl -s -o /dev/null -w "%{http_code}\n" https://api-dev-2c9d.up.railway.app/api/dev-tasks
401
$ curl -s https://api-dev-2c9d.up.railway.app/api/dev-tasks
{"detail":"Missing Bearer token"}
$ curl -s -o /dev/null -w "%{http_code}\n" https://api-dev-2c9d.up.railway.app/health
200
```

The service is up; `DEV_BYPASS_AUTH` is **off** on the deployed dev API. Confirmed, not
inferred.

### 1.3 The `#202` "success" was an in-process call, not HTTP

Verbatim from `122e4d9b-…jsonl`:

```bash
cd .../RoR_Trader/src && timeout 90 ../.venv/bin/python -c "
from dotenv import load_dotenv; load_dotenv('.env')
from api.routers import dev_tasks as DT
row = DT.complete_step(202, {'actor':'M·auto'}, user=None)
..."
```

And the run said so in its own report:

> **The tick I did was an in-process route call, not HTTP.** Nothing is listening on
> :8000/:8001 right now, so I invoked `complete_step(202, {"actor":"M·auto"})` directly.

Note `user=None` — it passed the `Depends(get_current_user)` parameter explicitly, which
is what makes the call possible outside ASGI. This is a **faithful** write (it *is* the
handler: `_next_assignee`, `kevin_final`, the system comment all ran) that **bypassed the
auth layer entirely** rather than satisfying it.

### 1.4 The `#184` "success" never touched the endpoint at all

`3a1a9eac-…jsonl` shows the run building the chain in a scratchpad script and stamping
step 1 done at construction time:

```python
def step(owner, title, body, mode="execute", stamp=None, done=False, by=None):
    s = {...}
    if done:
        s["completed_at"] = NOW; s["completed_by"] = by or "M·auto"
```

…then writing the whole `checklist` through a service-role PostgREST helper
(`board.py`, reading `RoR_Trader/src/.env`). Its report says it plainly: *"Authored the
9-step chain on #184 and marked step 1 done."*

That write could **not** have gone through the API even if it had tried:
`dev_tasks.py::_prepare_checklist_patch` rejects a newly-inserted step that starts
completed (`400 — a newly inserted step cannot start completed`). It went direct to
PostgREST, so no guard fired. It also skipped `_next_assignee`, `kevin_final` and the
hand-off comment — the low-fidelity path #217's description warns about.

The board's own timestamps corroborate the pairing (log mtimes are MDT, `completed_at` is
UTC):

```
#202 step 2  completed_at 2026-07-30T00:12:11Z   r1785369528-202.log  2026-07-29 18:14 MDT
#184 step 1  completed_at 2026-07-30T02:12:08Z   r1785377136-184.log  2026-07-29 20:13 MDT
```

### 1.5 Neither run had anything the other lacked, credential-wise

There is no per-run variable in the environment. `spawn()` passes **no `env=`** at all:

```python
subprocess.Popen([CLAUDE_BIN, "-p", prompt, "--output-format", "json"],
                 cwd=agent_worktree(agent), stdout=lf, ...)
```

Every run inherits the same dispatcher environment, and every run can reach the main
checkout's `src/.env` by absolute path (both bypass runs did). So:

- **service-role key — universally available** to every headless run.
- **Supabase *user* JWT — universally absent** from every headless run.

That asymmetry *is* the bug. The service-role key is not a user JWT and never can be:
`deps.py` decodes the bearer and requires a `sub` claim; the service key has none, so
presenting it yields `401 No subject in token` rather than success.

---

## 2. Why "make the base URL deterministic" is not, on its own, the fix

The hypothesis was worth testing and it cost nothing to test. But:

- the deployed dev API returns 401 to an unauthenticated caller (§1.2, measured);
- nothing is listening on `:8000` / `:3000` / `:8001` on this box — `ss -ltn` shows no
  such listener, and both the 07-29 runs that checked found none either;
- `frontend/.env.local` points at `https://api-dev-2c9d.up.railway.app`. The
  `localhost:8000` value survives only in `.env.local.example` and a `.bak-` file.

So there is no configuration of "which base URL" under which the call succeeds today.
A local API *would* succeed (`src/.env` has `DEV_BYPASS_AUTH=true`, and `get_current_user`
short-circuits before the bearer check), but no run has ever had one, and "the tick works
only when a human happens to be running uvicorn" is not a contract.

---

## 3. How the base URL is chosen today

**It isn't.** The dispatcher's tick block emits a bare path:

```
POST /api/dev-tasks/{task['id']}/steps/complete   body: {"actor": "..."}
```

No host, no env var, no note. `NEXT_PUBLIC_API_URL` lives in `frontend/.env.local`, which
is a *frontend* file the agent has no reason to read; nothing in `CLAUDE.md`, the charter,
or the prompt names an API host. The 07-29 run resolved it by grepping the repo for
`railway.app`.

**Is it deterministic per run? No** — it is whatever each agent's search happens to turn
up, and it varies with which files exist in that run's worktree. It is the same shape of
gap as the `#171` assignee contract, which names `PATCH /api/dev-tasks/<id>` with no host
either — the difference is that agents satisfy #171 by writing the `assignee` *column*
straight through PostgREST, which is legitimate because assignee is plain data. Completion
is not plain data: it is `done` + `completed_at` + `completed_by` + `_next_assignee` +
`kevin_final` + a system comment, all server-owned. That is precisely why #171 "works"
headless and #202 does not.

---

## 4. Options the diagnosis supports, with blast radius

Ordered by what the evidence actually backs. **(a)/(b) are the real candidates; (c) is
necessary-but-insufficient; (d) is the only path with a proven success and belongs on the
table.**

### (a) Service token in the agent's env
Dispatcher plumbs a credential into `spawn(env=…)` that the endpoint accepts.

- **Server change still required.** There is no service-token concept in the API today —
  `grep` over `src/api/` finds no `X-Service-Key`, no `SERVICE_TOKEN`, nothing. So this is
  not "config only"; it is (b) plus env plumbing.
- The alternative — **minting a real Supabase user JWT** — is worse than it looks.
  `deps.py::_validate_with_supabase` has a documented **degraded mode**: if Supabase is
  slow or unreachable it falls back to trusting the JWT's `exp` claim alone. A
  self-signed token would be rejected on the happy path but accepted whenever Supabase
  is degraded — a genuinely bad failure mode. A *real* minted session (admin magiclink)
  means a headless run holding a live human-equivalent session token, refreshable, with
  full RLS-user reach.
- **Blast radius:** a bearer that authenticates as a *user* is accepted by **every**
  `Depends(get_current_user)` route in the API — strategies, backtests, alerts, the lot.
  Widest of the four options.

### (b) Service-role path on the endpoint
Accept the service-role key as bearer on the three step-action routes
(`/steps/complete`, `/steps/stamp`, `/steps/raise-issue` — all three take an `actor`),
attributing the write to `payload["actor"]`.

- **Fits the handler as written.** `complete_step` never reads `user` — attribution
  already comes from `actor` in the body. `user=Depends(get_current_user)` is a pure gate.
- **Marginal blast radius is near zero *if scoped to those routes*.** Anyone holding the
  service-role key already has unrestricted PostgREST access to every table including
  `dev_tasks` — that is how the dispatcher does all its board I/O. Letting that same key
  through three endpoints grants no capability its holder lacks; it grants *better*
  fidelity, because the server-owned logic runs instead of being hand-rolled.
- **The way to get this wrong** is to bless the service key inside `get_current_user`
  itself. That would silently open every authenticated route in the product to the key,
  which is the (a)-sized blast radius with none of (a)'s intent. If we take (b), the check
  belongs in the step routes (or a `require_actor` dep used only by them), never in the
  shared dep. **This is the auth-surface judgement step 4 should be asked to make.**
- Cost: the key must then also reach the *deployed* API's env — it is already there
  (`_admin()` uses it server-side), so no new secret distribution.

### (c) Make the base URL deterministic
Name the host in the tick block / agent env instead of leaving each run to grep for it.

- **Does not fix the 401** (§2) — it is not "the entire fix at no security cost". Today it
  changes a guess into a reliable 401.
- **But it is still required.** Whatever (a)/(b) lands, the agent needs to know where to
  send the call. Ship it *with* the chosen fix, not instead of it. Zero security cost,
  ~3 lines in `build_prompt` plus one constant.

### (d) A local tick path that calls the real handler in-process
A tiny `tools/team_dispatcher/` entry point that does what `r1785369528-202` did by hand:
load `src/.env`, `from api.routers import dev_tasks as DT`, call `DT.complete_step(...)`.

- **The only option with a demonstrated success on this board.** Full server-owned
  fidelity; no HTTP; no auth surface added anywhere; nothing new to deploy.
- **Cost 1 — it moves the auth boundary rather than passing it.** Whoever can run the
  script can tick anything. Given they already hold the service-role key, that is close to
  a distinction without a difference, but it should be a *decision*, not a side effect.
- **Cost 2 — worktree fragility.** Agent worktrees do not carry `src/.env` or `.venv`
  (both gitignored; verified absent in `Kevbot-wt-briefgen-211` and `Kevbot-wt-msession-220`).
  The script would have to resolve them by absolute path back to the main checkout, which
  is exactly the kind of coupling that breaks quietly.
- **Cost 3 — it does not generalise** to any agent not on this machine.

**Recommendation for step 3 (M's call, not made here):** **(b) + (c)** — a service-role
path scoped to the step-action routes, plus a deterministic host in the prompt. (b) is the
only option that adds real capability without widening what a leaked key can already do,
and (c) is needed by every option anyway. (d) is the sound fallback if we would rather add
no auth surface at all and accept the machine-local coupling.

---

## 5. Loose end worth its own row (not actioned here)

`#184` step 1 shows that **chain-authoring writes bypass the completion guard**. A run
creating a chain through PostgREST can set `done`/`completed_at`/`completed_by` on a step
at construction time — the `400 a newly inserted step cannot start completed` guard only
fires on the API path, and chain authoring does not use the API path. Same class as the
`step.stamp = None` divergence #184 itself flagged. Not this task's scope; flagging it so
it is not lost.

---

## Scope note

Diagnosis only. No code changed, no auth widened, no fix implemented. One doc, one commit,
branch `diag/steps-complete-401-217`, cut from `origin/dev` @ `94d9affb`.
