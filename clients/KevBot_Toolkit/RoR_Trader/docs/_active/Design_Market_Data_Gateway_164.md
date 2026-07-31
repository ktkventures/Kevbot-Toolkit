# Design — V5.2 market-data gateway (board #164)

**Status:** DESIGN. Nothing built. No flags, no migration, no Railway, no probes run.
**Author:** E·auto (run `r1785538375-164`) · **Date:** 2026-07-31
**Supersedes §3 and §6 of** `Design_Environment_Split_Gateway.md` (M-authored scoping) —
that doc asked the six questions; this one answers them and **changes two of its
assumptions**. Companion evidence: `V51_Pilot_Feed_Noise_Floor_2026-07-31.md` (#163).

---

## 0. The two answers that change the shape of the task

Everything below follows from these. Read them first.

### 0.1 The gateway must publish the TICK STREAM, not built bars

M's diagram has the gateway "owns bar construction". **It must not.** Bar construction in
this engine is *strategy-parameterized* — session filter, per-strategy grace deadlines that
fire on a **still-forming bucket** (`fire_on_partial_bucket`, `ralph_engine.py:1690`),
`skip_volume`, dup-guard, `close_on_boundary`, and five `live_model` variants. A gateway
that builds bars must know every consumer's strategy set, so it changes every time a
strategy changes — which destroys the one property Kevin asked this component for
("ingest becomes a stable, rarely-touched component").

The gateway normalizes, sequences and fans out **Polygon's `A` per-second aggregates
(and `AM`)**. Consumers keep `BarBuilder` exactly as it is today.

### 0.2 We cannot run the gateway "alongside" the current socket — and that is what makes the migration safe

The connection budget is **ONE** (finding #2 on the task). A separate gateway service and
the current in-process socket **cannot both connect**; over the limit Polygon authenticates,
confirms the subscribe, and then **silently delivers nothing** (observed 2026-04-15). So the
obvious "run both, compare" plan is not available.

The way out is to invert it: **the gateway is born *inside* the current worker process**
(§6). Extraction into its own service is the *last* step, not the first. This also produces
the transport measurement in §1 for free, on live data, with zero money-path exposure.

---

## 1. Transport, and its latency budget

### 1.1 What exists today

`ralph_engine`'s WS loop parses the frame and calls `hub.on_second_bar(...)` **on the same
stack** (`ralph_engine.py:7005-7021`). Ingest→decision is a function call: **zero hop,
zero serialization, zero queue.** Any transport we introduce is a pure addition to the money
path. That is the central risk in this task and the reason for a hard number.

The `~3.2–3.4s` figure everyone quotes is **not** this path — it is `written_at − bar_close`
for the *database* (`V51_Pilot_Feed_Noise_Floor` §5), i.e. Polygon's own emit delay plus a
fire-and-forget Supabase write that no decision waits on. **Do not spend that 3.2s as if it
were headroom.** The decision path's current cost is ~0 and the budget below is absolute.

### 1.2 The mechanism

**Redis pub/sub over Railway private networking**, one channel per symbol, fire-and-forget,
**no persistence, no consumer groups, no ack**.

Why not the alternatives:

| Option | Verdict |
|---|---|
| **Redis pub/sub** | **Chosen.** Managed on Railway; private-network hop; pub/sub semantics already *are* the back-pressure policy we want (§1.4) — a subscriber that cannot keep up is disconnected, never queued. |
| ZeroMQ PUB/SUB | Lower theoretical latency (no broker hop), but adds a peer-discovery problem across Railway service restarts and a native dependency to the most safety-critical path. Not worth the milliseconds. |
| Postgres `LISTEN/NOTIFY` | Rejected. Puts the money path's data plane inside the database we already know adds seconds. Explicitly ruled out by the task. |
| DB round-trip (`live_bars`) | Rejected — that is V5.1, and #163 measured it at **100.0%** of decisions arriving after live had already decided. |
| Kafka / NATS / SQS | Rejected. Durable-queue semantics are the *wrong* semantics: a replayed stale bar is worse than a dropped one (§1.4). |

### 1.3 The budget — this is the make-or-break number

Measured at the consumer as `t_consumer_receives − t_gateway_parses_frame`:

| statistic | budget | rationale |
|---|---|---|
| p50 | **≤ 5 ms** | 0.05% of a 10Sec bar. |
| p99 | **≤ 50 ms** | 0.5% of a 10Sec bar; ~1.5% of the *shortest* strategy grace window in use. |
| p99.9 | **≤ 250 ms** | Hard ceiling. Above this a decision can cross a grace deadline. |
| any sample | **> 1 s ⇒ ship-blocker** | Kevin's sub-second requirement is on the whole path, and Polygon already owns most of it. |

**These are budgets, not measurements. Nothing is measured yet — see §6 phase 2 for the
probe that produces the real distribution on live data before any money-path change.**

**Ship gate:** the gateway path may only carry the live fleet when a full RTH session shows
p99 within budget **and zero `gw_seq` gaps** (§3.3). Failing either, it is dev-only — which
is exactly the outcome the task description anticipates.

### 1.4 When a consumer stalls: drop and re-warm. Never buffer.

**Policy: the transport never queues on the consumer's behalf.** Redis client output-buffer
limits are set so a subscriber that falls behind is **disconnected**, not backlogged.

Rationale: a bar delivered late is not worth having. A buffered burst is *actively harmful* —
it replays a minute of stale bars into a live decision path, which is the 2026-05-19 lag
spiral in a new costume. The correct behaviour on a stall is the one the engine already
implements for restarts:

1. subscriber detects the gap (`gw_seq` discontinuity, §3.3),
2. suppresses alerts for bars older than its grace window (**see the open ruling in §5.3**),
3. re-warms from REST via the existing `_queue_gap_heal_if_gap` / `warmup_seed` reconcile
   path (`live_bars_writer.py:218` — built precisely because "worker restarts leave a
   ~1-3 min hole"),
4. resumes.

No new correction machinery. The stall path *is* the restart path (§5).

---

## 2. What the gateway publishes

**Normalized per-second aggregates, verbatim, plus the minute stream — never built bars.**

Payload per message (one Polygon event, one channel per symbol):

```
{ ev, sym, s, e, o, h, l, c, v, vw, n,     # Polygon's fields, unaltered
  gw_seq,        # monotonic per-symbol sequence — gap detection (§3.3)
  gw_recv_ns }   # gateway's monotonic receive stamp — latency measurement (§1.3)
```

The gateway owns: connection, auth, subscribe, frame parse, symbol normalization (the
crypto `X:` → `BASE/QUOTE` rewrite at `ralph_engine.py:6975`), sequencing, and fan-out.
It owns **nothing** that decides anything.

**Why not built bars, in one line each:**

- **Sub-minute is 95% of decisions** (10Sec 63% · 15Sec 32% · 1Min 5%) and sub-minute bars
  are built from `A` events *inside the hub* — a bar-publishing gateway would have to
  replicate five `live_model` variants and every strategy's grace deadline.
- **Some decisions have no bar at all.** `fire_on_partial_bucket` decides on a bucket that
  has not closed. A built-bar gateway is structurally incapable of serving those.
- **Migration provability.** Consumers keep `BarBuilder` byte-identical, so the gateway can
  be proven behaviour-neutral by diff and by the replay harness. If the gateway also builds,
  every comparison is confounded by the rebuild.
- **Stability.** Bar-building logic changed at least five times in the last quarter (#279,
  `RORT_BAR_DUP_GUARD`, sub-min canonical, volume-integrity, PB-defer). Ingest changed
  approximately never. Only one of those belongs in a single point of failure.

**Consequence to accept explicitly:** every consumer pays its own aggregation CPU. That is
already true today per-process and is the price of symmetric, strategy-correct construction.

---

## 3. The 6.7% gap — does the gateway inherit it?

**No — with one caveat that must be gated, and one new risk the gateway introduces.**

### 3.1 What the 6.7% actually is

#163 measured **97/1414 (6.9%)** of live deciding bars absent from `live_bars`
(15Sec **13.9%**, 10Sec 3.6%, 1Min 2.8%). Crucially: **the engine decided on those bars.**
They existed in memory. The gap is not an ingest gap — it is a **materialization** gap,
entirely downstream of the decision, in two distinct classes:

- **(a) Write-path loss.** `live_bars_writer.write_bar()` is fire-and-forget onto a
  **2-worker thread pool**, and `_write_bar_sync` swallows every failure at `logger.warning`
  (`live_bars_writer.py:120-129`). Nothing retries, nothing counts the loss, and the write is
  gated behind `LIVE_BAR_CACHE_WRITE_ENABLED`. A decision is never blocked by it and never
  learns it failed.
- **(b) No bar was ever completed.** Grace fires (`fire_on_partial_bucket`) decide on a
  *partial bucket*. `write_bar` is only called on a **completed** bar
  (`ralph_engine.py:6050-6056`), so a decision taken at grace on a bucket that closes late,
  differently, or not at all leaves no row — correctly, because no such bar existed.

**Both classes are consumer-side and post-decision. A tick fan-out neither causes nor cures
them.** The gateway inherits the gap **only if it publishes bars sourced from `live_bars`** —
which is precisely the shape §0.1 rejects.

### 3.2 The caveat — this is not yet decomposed

The (a)/(b) split has **not** been measured; 15Sec's 13.9% versus 10Sec's 3.6% is
unexplained and that asymmetry matters. **Probe requested (post-close, read-only, ~5 min):**
join the 97 missing decisions against `strategies.live_model` and the fire mode, and report
the split by `live_model` × timeframe. If it is mostly (b), the number is not a defect at all
— it is the forming-bar decision class, already characterized in
`project_forming_bar_fidelity_nonohlc_packs`. If it is mostly (a), `live_bars` is losing
writes under load and that is its own bug, worth a board task, and **independent of this
task**. Either way the gateway answer above is unchanged; the probe decides whose problem it is.

### 3.3 The new risk the gateway DOES introduce — and its gate

Today an in-process function call **cannot drop a bar.** A transport can. A dropped `A`
event does not produce a missing bar — it produces a **silently wrong** one (short volume,
missing extreme), which is strictly worse in kind, and lands exactly on the known
sub-minute volume-gate trap (`project_fine_tf_construction_falsified`).

Mitigation is mandatory, not optional:

- `gw_seq` — monotonic per-symbol sequence on every message.
- Consumer-side gap detector on `gw_seq`; a gap routes to the existing
  `_queue_gap_heal_if_gap` REST heal and **marks the affected bar** rather than silently
  using it.
- **Cutover gate: one full RTH session with zero unhealed `gw_seq` gaps**, alongside the
  §1.3 latency gate, before the money fleet consumes the bus.

---

## 4. Minute bars: `AM` or built?

**Decision: the gateway CARRIES `AM`. It does not decide who closes on it — that stays #279.**

Split the question in two, because it is two questions:

**Should the gateway subscribe `AM`?** **Yes.** The 2026-05-18 reason for dropping it was that
`AM` and `ws_agg` became *two writers racing the same `live_bars` PK*, contaminating the
`first_*` decision-time columns (`ralph_engine.py:6857`). **The gateway removes that reason
by construction** — it is a single ingest owner publishing two distinct streams with
distinct `ev` tags; nothing races because nothing writes. It costs one channel and no
decision changes.

**Should 1Min+ builders close on `AM` instead of on the next `A` event?** **Not in this
task.** It is a genuine improvement — `AM` is clock-driven, so it retires #279's ~28s p90
sparse-session close delay — but it is a *decision-logic* change: every monitor currently
runs a `ws_agg` live_model, and moving the close changes which bar a strategy decides on and
therefore backtest parity. It gets its own flag, its own arm, and its own validation, on
**#279**, and only after the gateway is proven behaviour-neutral. Absorbing it here would
mean shipping a transport change and a logic change in the same diff on the money path.

**Restatement:** `AM` is mutable for ~15 min. **It fits `ws_rest_spliced`'s existing policy
unchanged** — act on the fast value, converge to the authoritative one, never retract, stamp
the delta. The gateway republishes a restated `AM` as an ordinary message with the same
`(sym, s)` and a later `gw_seq`; the consumer's existing correction path
(`last_was_correction` → `recompute_from_history`) already handles a re-delivered bar.
**No second correction channel is created.**

---

## 5. Restart behaviour

### 5.1 The finding that hurts: one connection means no zero-downtime deploy

Because the connection budget is **one**, a new gateway instance **cannot** connect before
the old one disconnects — it would authenticate and then receive silence. So the gateway
**cannot be rolling-deployed or blue/green'd.** Every gateway deploy is a
**stop-then-start hard outage** for both fleets simultaneously, lasting reconnect + auth +
subscribe + first-event.

This is the strongest argument in the whole design for `§2`'s "publish ticks, change
nothing else": **the fewer reasons the gateway has to deploy, the smaller this cost is.**
It is also why the gateway must not own bar-building — every builder change would become a
fleet-wide bar outage.

Operational consequence, for the promotion policy settled on #263: **the gateway deploys on
manual promotion only, outside RTH.** An RTH gateway deploy is strictly more expensive than
today's worker restart (which already costs ~5 edges per active strategy per
`feedback_rth_varset_arm_cost`) because it now hits every consumer at once.

### 5.2 What consumers do while it is down

Identical to §1.4, and it reuses machinery that already exists because worker restarts
already leave holes:

- Consumers **never** depend on the gateway for warmup — REST warmup is unchanged.
- On reconnect, the `gw_seq` gap triggers `_queue_gap_heal_if_gap` / `warmup_seed`
  reconcile.
- The gateway publishes a `gw_epoch` on connect; a changed epoch tells every consumer
  "you missed an unknown amount" without them having to infer it.

### 5.3 In-flight decisions — **this one needs Kevin's ruling**

A bar that arrives after its grace deadline can be fired late or dropped. **Recommendation:
prefer a missed alert to a late alert at a stale price** — suppress alerts for bars older
than the strategy's grace window, count the suppression, and surface it. That is a
behavioural change to the money path under an outage, and per
`feedback_no_silent_defaults` it must be a stated decision rather than an emergent one.
**Ruling needed before build** (this is step 3 of the chain and I am not pre-empting it —
noting only that the money-path failure mode Kevin asked to have considered first *is* this
one, and it belongs in that ruling).

---

## 6. Migration path — four phases, no flag day, and the money path moves last

The one-connection constraint (§0.2) forbids "run the new service alongside the old socket".
So the process boundary is introduced **last**, after everything else has been proven:

**Phase 1 — extract in-process. Zero transport, zero risk.**
Move the WS loop out of `ralph_engine`'s `_run_polygon_ws` into a `market_data_gateway`
module that publishes to an **in-process** bus the hub subscribes to. Same process, same
stack, same latency (~0). Provable by diff plus the replay harness; nothing to flag, nothing
to arm. This is where the whole refactor's risk actually lives, and it is paid with the money
path unchanged.

**Phase 2 — dual-publish, and get the §1.3 number for free.**
The same module *also* publishes to Redis. **Prod still consumes in-process.** A dev
consumer subscribes to Redis. Comparing the two consumers' receive stamps on live data
yields the real transport distribution — **the measurement §1.3 currently only budgets** —
at zero money-path exposure. Also validates `gw_seq` gap-freedom (§3.3) over full sessions.

**Phase 3 — flip the consumer, not the process.**
Only if phase 2 meets both gates: prod's hub consumes the bus instead of the in-process
call, behind `RORT_GATEWAY_CONSUME` (`inproc` | `bus`, default `inproc`). Still one process,
so rollback is a flag flip and the in-process path stays resident. **This is the step that
actually spends the latency**, and by now its cost is measured, not assumed.

**Phase 4 — extract the service.**
Move the gateway module into its own Railway service. Prod has been consuming the bus for
weeks, so the only new variables are the process boundary and §5.1's deploy outage. Dev and
prod now consume the same stream at the same instant — **the symmetry V5.1 cannot provide**,
which is the entire point of the task.

Each phase is independently revertible and none is a flag day. Phases 1–3 need no second
Polygon connection at any point; **phase 4 needs one brief cutover window** in which the old
socket closes before the new service connects — a scheduled, out-of-hours outage of one
reconnect, not a purchase.

---

## 7. `live_bars` — who writes it? (carried from `Design_Environment_Split_Gateway.md` §6 Q4)

**The gateway does not write `live_bars`, and should not.** `live_bars` is a record of *what
a deciding process saw* — that is what makes it forensically useful, and it is per-consumer
by definition. Two fleets consuming the same ticks can still build different bars (different
flags is the entire purpose), and a gateway-owned write would erase exactly the difference
the A/B exists to observe. Each fleet keeps writing its own rows under its own environment
tag (#163's `pilot_alerts` precedent applies to the write-isolation question).

The gateway may optionally record its **raw** stream for forensics; that is a separate
table, a separate decision, and not required for V5.2.

---

## 8. Honest limits

- **Every latency figure in §1.3 is a budget. None is measured.** Phase 2 exists to replace
  them with real numbers before anything on the money path moves.
- **The 6.7% is not decomposed** (§3.2) — the probe is named and not run. The gateway
  conclusion does not depend on the split; the ownership of the residual bug does.
- **Redis on Railway private networking has not been latency-tested by us at all.** If phase
  2 shows p99 outside budget, §1.2's table is re-opened with real data rather than argued
  again from first principles — ZeroMQ is the fallback, not a co-recommendation.
- This design assumes the #263 topology (two Railway environments, two Supabase projects,
  gateway is a service *inside* prod). If that changes, §6 phase 4 changes with it.
- **#161 environment-parity remains unasserted.** Nothing here is a conclusion about logic
  divergence, and none may be drawn from a two-fleet comparison until it is.
