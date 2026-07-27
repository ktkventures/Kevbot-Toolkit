# Design — Environment split: dev/main + market-data gateway

**Status:** SCOPING. M-authored proposal. **E must review before any build** — the gateway
is engine architecture and E owns that path.
**Board:** V5 (#162) with subtasks V5.1 / V5.2 / V5.3.
**Author:** M · **Date:** 2026-07-27 · **Origin:** Kevin, during the 07-27 divergence scare.

---

## 1. Why

RoR Trader runs ONE environment. `dev` *is* production — every merge deploys to the fleet
that trades. Three costs follow, and we currently pay all three by hand.

### 1.1 Blast radius is managed manually
A bad merge reaches the money path immediately. Our mitigation is a **manual PR freeze**:
on 2026-07-27 we froze all merges for roughly six hours because we could not tell whether
the fleet was healthy. That freeze is blast-radius management done by hand, and it stops
all other work while it holds. A `main` that deploys independently makes it unnecessary.

*(This is Kevin's argument and it is the strongest one for this vision. The freeze is a
visible, recurring, measurable cost — not a hypothetical.)*

### 1.2 We cannot run a simultaneous A/B
Feature flags let us restore prior behavior **sequentially** — Friday's logic replayed on
Monday's market. That is an anecdote, not an experiment: the two readings differ in market
regime, settle state, and bar finalization, all at once. 2026-07-27 consumed an entire
trading day for exactly this reason — every available comparison was across days, and each
one was confounded.

Two fleets running different logic against the **same bars in the same session** make the
divergence between them attributable to logic and nothing else.

*(Also Kevin's argument. M initially undersold it and was wrong to.)*

### 1.3 Ingest and decisioning are coupled
`ralph_engine.py` is the only module that opens the Polygon WebSocket. So a second fleet
today means a second WS connection and a second data subscription (~$200/mo). That cost is
an artifact of the current structure, not a law.

---

## 2. Current architecture (verified 2026-07-27)

| Fact | Evidence |
|------|----------|
| `ralph_engine.py` is the sole Polygon WS consumer | only file matching `WebSocketClient` / `wss://` under `src/` |
| `live_bars` (`source=ws_agg`) is an already-materialized shared bar stream | queried directly; ~28 rows / 8 min across 4 symbols |
| ws_agg write-lag ≈ **3.5s median**, p90 ~4–6s | sampled repeatedly through the 07-27 session |
| Update-All-Data already runs fleet-wide, in-prod, parallelized | `src/update_jobs.py` + `src/api/routers/update_jobs.py`; UI at `/admin/update-jobs` |
| Lane separation by `data_source` is established precedent | `backtest_rest_hifi`, `cache_cache_locked` |

That fourth row matters for scoping: the "run a job across a large catalog of strategies on
parallel tracks" infrastructure **already exists** and should be reused, not rebuilt.

---

## 3. Target architecture

```
                 ┌──────────────────────────┐
  Polygon WS ───►│  market-data gateway     │   ONE subscription, ONE connection
                 │  (normalize + fan-out)   │   owns bar construction
                 └────────┬─────────┬───────┘
                          │         │           symmetric, sub-second delivery
                 ┌────────▼───┐ ┌───▼────────┐
                 │ main fleet │ │ dev fleet  │  different code / flags
                 │ (money)    │ │ (candidate)│  same bars, same instant
                 └────────┬───┘ └───┬────────┘
                          │         │
                 ┌────────▼─────────▼────────┐
                 │  shared Supabase          │  lane-tagged by environment
                 └───────────────────────────┘
```

**Symmetric delivery is the load-bearing property.** If `main` reads the socket while `dev`
reads the DB, every timing difference observed is latency asymmetry rather than logic. Since
divergence questions are frequently *timing* questions (07-27's signal was a dispersion
change: p90 6.9s → 13.9s), an asymmetric setup would actively mislead.

---

## 4. Phasing

### Prerequisite — #161 environment-parity assertion
On 07-27 we found `local_update.py` running **6 of 26** batch-worker flags and **pandas
2.3.3 against prod's 3.0.3**, with its own smoke test green throughout. We cannot presently
verify that ONE environment matches prod.

**Standing up a second environment before this is fixed makes things worse, not better** —
it multiplies silent-drift surface and renders every dev-vs-main comparison ambiguous. You
would observe a gap and be unable to say whether it was logic or environment. That is the
same epistemic hole that cost 07-27, duplicated.

### V5.1 — cheap pilot (dev fleet on `live_bars`)
No second subscription, no refactor, available now. A second engine consumes the already-
materialized `live_bars` stream, runs a different flag set, writes under a distinct
environment tag.

- **Answers:** do the two fleets take the same *trades*?
- **Does NOT answer:** do they take them at the same *time*? (~3.5s DB lag vs socket.)
- **Done when:** one session compared, with a written verdict on whether fleet-vs-fleet
  divergence is a usable signal. That verdict decides whether V5.2/V5.3 earn their cost.

May run in parallel with #161 **provided no conclusions are drawn** until parity is asserted.

### V5.2 — market-data gateway
Extract WS ingest from `ralph_engine.py` into a fan-out service. Retires the second-
subscription question permanently and delivers the symmetric timing V5.1 cannot.

### V5.3 — promote `dev` → `main`, spin a fresh `dev`
Cut the new `main` from current known-good `dev`. Do **not** resurrect the existing
placeholder `main` — that means inheriting an unknown baseline and spending a week
discovering its contents.

> `main` is not "the version with no divergences." Friday was not clean either.
> `main` is **the version whose divergences we have characterized and accepted.**

**Trigger for this step:** real-money execution going live. That is when insulation earns
its cost.

---

## 5. Decisions already taken

| Question | Decision | Reasoning |
|---|---|---|
| Second data subscription? | **No, if V5.2 lands** | Gateway = one connection, N consumers. V5.1 needs none either. |
| Same Polygon/Massive API for both? | **Yes** | Data is identical; the difference is treatment, not source. The constraint is the WS *connection limit*, which the gateway removes. |
| Shared Supabase or separate? | **Shared, lane-tagged** | Separate DBs destroy the comparison this vision exists for — no shared strategies, history, or measurement surface — and would duplicate every config and reset the divergence baseline. Multi-lane-in-one-DB is established precedent. |
| Where do CI gates run? | **Both PR→dev and dev→main** | The promotion gate tests a merge result that never previously existed. Path-filter so frontend-only changes never pay the engine-suite cost. |
| Which `main`? | **Promote current `dev`** | Known baseline over archaeology. |

---

## 6. Open questions for E

1. **Transport** for fan-out — Redis pub/sub, in-process, message queue? Must beat the
   current direct-socket latency or it is dev-only (Kevin's requirement is sub-second).
2. **Back-pressure** when one consumer falls behind — drop, buffer, or disconnect?
3. **Cutover** — does the live fleet switch to the gateway immediately, or run alongside the
   direct socket during validation? (Alongside is safer and gives a built-in A/B of the
   gateway itself.)
4. **Who writes `live_bars`?** Does the gateway own that write, or does each fleet?
5. **Environment tagging** — extend `data_source`, or add an orthogonal column? Must follow
   existing lane precedent, and the money lane must be unmistakable.
6. **Failure mode** — the gateway is a single point of failure for both environments.
   What happens to the money path when it dies?

---

## 7. What this vision does NOT solve

**It does not fix measurement.** The 07-27 scare was a settle-state artifact. Had two
environments existed that morning, *both* would have read depressed at T+0 and the
difference between them would still have been uninterpretable.

The measurement fix is **#160 — daily T+0 health snapshot**, which is roughly a day of work
against this vision's weeks. Keep the two separate so neither gets sold as the other's
solution.

Nor does it give a place to validate *fidelity* before promoting: divergence can only be
measured against live market data and the production DB. A dev environment with its own
database cannot reproduce it. This split buys code safety, blast-radius control, and the
A/B — not pre-promotion fidelity validation.

---

## 8. Cost

| Path | Recurring | One-time |
|---|---|---|
| Second data subscription (no gateway) | ~$200/mo | low |
| Gateway (V5.2) | $0 extra | real refactor of the most safety-critical path |
| Second Railway environment (V5.3) | Railway cost only | moderate |

The gateway converts a permanent monthly cost into a one-time engineering cost, and is the
correct architecture independently of the money. That is the recommendation.
