# Design note — Replay-simulated last-10 basis (SIM basis) for no-algo-history strategies

**Board #120 (V1.16) · Author: E · Status: DESIGN FIRST — do NOT build without M/Kevin
sign-off on the compute budget (§7).** Tier 2 of Kevin's 07-25 simulated-basis idea.
Companion to F's V2.8 (#119, ALGO basis) and the existing Last-10 instrument (#70/#122).
Sibling initiative: `Design_Provisional_Backtest_Tail.md` (V1.7).

---

## 0. Kevin's requirements — VERBATIM (07-25, captured per M's directive)

> **(1) TRUST IS THE HARD REQUIREMENT, speed is nice-to-have:** the simulated basis must
> be an EMBEDDED version of what the live model actually does — replay harness qualifies
> (real engine, decision-time bars), and every known divergence from live (bar-replay not
> provider-event; no live close/publish topology) must be ENUMERATED, TESTED, and SURFACED
> in the UI next to the score, not implied. Any new divergence introduced by optimization
> must join that list with tests.
>
> **(2) ON-DEMAND acceptable and maybe preferred:** a per-strategy button on strategy
> health (Kevin clicks when curious) → queued replay job → cached result table → page
> reads cache. Fits the dispatcher/Run-button architecture; nightly-for-all becomes
> optional.
>
> **(3) BOUNDED WINDOW idea:** only need enough history to produce the last ~10 trades —
> BUT indicator warmup is sacred (never reduce, memory feedback_indicator_warmup): E sizes
> the minimum window as warmup-preamble + trade-window, not naive truncation. Sequencing
> unchanged: after Phase C + Run button per Kevin priority.

These three points are the acceptance criteria. §5 answers (2), §6 answers (3), §7 is the
sign-off ask, and **§8 (the divergence ledger) is the whole deliverable for (1).**

---

## 1. The core reframe — SIM is a 4th *lane*, not a new instrument

The Last-10 instrument already exists and is proven (`src/api/routers/health_last10.py`).
It does exactly one thing: take a strategy's **last 10 completed BACKTEST-lane trades** as
the reference edge set, and score each entry/exit that pairs (±tol, greedy two-pointer walk
`_greedy_pair`, verbatim port of `strategy_health._pair_phantom_missed`) against **some
lane's timestamps** → `n / 2·trade_count`.

- Basis today = the **ALERT** lane (`fired` + `theo`).
- V2.8 (#119, F) adds the **ALGO** lane (bt edges vs `trades.data_source=algo`).
- **V1.16 (this note) adds the SIM lane** = bt edges vs the **replay harness's
  would-have-fired edges**.

So the *pairing/scoring machinery is already built and does not change*. The only new thing
SIM needs is **a cached list of the replay lane's would-have-fired entry/exit timestamps
per strategy** — because producing that list is the expensive part (it runs the real
engine), and it must never run on page load. Everything else is a `basis=sim` parameter on
the existing endpoint.

**This is the key simplification: the results table stores the replay *edges*, not a
score.** Pairing against the current last-10 bt set happens cheaply at read time, so the
SIM score stays fresh as new backtests land (up to the staleness rule in §4.3), and we
reuse F's exact greedy walk instead of re-deriving one.

**Why bt-lane is still the reference for "no-algo-history" strategies:** a backtested
strategy always has a bt lane (that's what health scores against); "no algo history" means
the *alert/algo* lanes are empty/quiet, so the alert and V2.8 columns read n/a or misleading
zeros. SIM answers the question those columns can't: *would the live ENGINE have taken these
bt trades, given the decision-time bars it actually saw?* — logic-would-fire evidence,
**never** delivery evidence (same framing as V2.8's algo basis).

---

## 2. What the replay produces today, and the one enrichment needed

`replay_harness.replay(sid, since, until, sb, canonical_edge=True, corrected=False)` returns
`{'entries': [...], 'exits': [...], 'diag': {...}}` where entries/exits are **fill
datetimes** (`replay_harness.py:474,513`) — decision-time bars driven through the real
`StrategyMonitor` / `SymbolHub` / `_ShadowIndicatorEngine`, intra-bar 1Sec ticks feeding
`on_tick` for stops/targets, under the ARMED prod flag stack (`ARMED_FLAGS`,
`replay_harness.py:100-123`).

- **For scoring, timestamps are sufficient** — `_greedy_pair` only needs timestamps. So v1
  SIM basis can ship with the harness exactly as-is.
- **For the modal (per-trade rows like the Last-10 modal), we want enrichment**: side,
  entry/exit price, exit reason. The monitor emits these on the signal/fill dict; today
  `replay()` discards everything but the timestamp (`fill = close_ts.to_pydatetime()`,
  line 511). Capturing the 3–4 extra fields the monitor already produces is a small,
  contained harness change — **BUT per Kevin req (1) it is a new surface and must ship with
  a test** (assert enriched fills' timestamps are byte-identical to the current
  timestamp-only path on a canary; enrichment must not perturb the edge set). Recommend
  enrichment in v1.1 (modal polish), not v1.0 (score column).

**Job efficiency note:** the *job* calls `replay()` directly and stores its edges. It does
**NOT** call `get_strategy_health`. The 31s canary run in §7 spent most of its wall-clock in
the 3-tolerance × 2-basis dashboard-scoring loop (visible as ~4× repeated identical
`trades`/`alerts`/`strategies`/`shadow_heartbeats` fetches) — that whole path is `main()`
presentation and is skipped by the job.

---

## 3. Architecture — on-demand primary, nightly-for-all optional

Per Kevin req (2), **on-demand is the primary path**; nightly-for-all is an optional later
sweep. Three moving parts:

```
  [Strategy Health page]                          [runner: off-hours, prod-load-light]
        │  ▲                                              │
  click │  │ read cache (never computes)                  │ claim → replay() → write cache
  "run  │  │                                              │
   SIM" ▼  │                                              ▼
   POST /api/strategy-health-sim/{sid}/run   ┌────────────────────────────────┐
        │  enqueue request row               │ replay_sim_requests (queue)    │
        └───────────────────────────────────►│  → replay_sim_basis (results)  │
   GET  /api/strategy-health-last10/{sid}     └────────────────────────────────┘
        ?basis=sim  ── reads replay_sim_basis, pairs vs live last-10 bt edges
```

**Runner host — the real decision (see §7).** The replay harness is a heavy,
prod-DB-reading, off-hours-only tool (`feedback_local_analysis_starves_live_worker`). Two
candidates:

- **(A) Local machine, dispatcher-pattern (RECOMMENDED for on-demand).** Same host the
  harness runs on today and the same host the Run-button dispatcher
  (`tools/team_dispatcher/dispatcher.py`) already polls from — Railway can't reach Kevin's
  machine, so button presses are *declarative* (a row/tag the local poller claims). A SIM
  request is the exact same shape: the page writes a `replay_sim_requests` row; a small
  local poller (new, or a mode added to the dispatcher) claims it, runs `replay()`, writes
  `replay_sim_basis`. Matches Kevin's "fits the dispatcher/Run-button architecture" word for
  word. Cost: latency = next poll + run time; the machine must be on (fine for an
  on-demand-when-curious tool).
- **(B) Railway batch-worker, `compute_jobs_store` job type (for reliable nightly-for-all).**
  Always-on; already runs the nightly `full_recompute` via `compute_jobs_store.enqueue(uid,
  sids, "full_recompute", …)` fired from a UTC-timed thread (`batch_worker.py:155-230`).
  A `replay_sim_basis` job type would slot in the same way. Cost: adds heavy replay load to
  a prod Railway service — **must be gated to off-hours only** (post-close → pre-open) and
  serialized, or it starves the live worker's DB egress.

**Recommendation:** ship **(A)** for the on-demand button (Phase 1). Treat nightly-for-all
as a **separate, later, opt-in** decision (Phase 2) and, if wanted, prefer **(B)** for it
because reliability of an unattended nightly needs an always-on host. Do not build (B) until
Kevin/M want nightly-for-all — Kevin explicitly demoted it to optional.

---

## 4. Schema

Additive, admin-only via service-role (same RLS posture as `dev_tasks` / `run_history` /
`compute_jobs`). Migration lives in `src/migrations/`.

### 4.1 `replay_sim_basis` — the results cache (health page reads this)

One row per completed replay per strategy; the read path takes the newest `ok` row per sid.
Keep history (don't upsert-in-place) so we can see when a score moved and why.

```sql
CREATE TABLE IF NOT EXISTS replay_sim_basis (
    id              BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    strategy_id     BIGINT      NOT NULL,          -- not FK'd to strategies (soft; sid may be deleted)
    symbol          TEXT        NOT NULL,          -- provenance
    timeframe_secs  INTEGER     NOT NULL,          -- provenance

    -- the window actually replayed (§6). window_since already excludes warmup;
    -- warmup depth is implicit (harness 7d fine / 60d coarse) and NOT reducible.
    window_since    TIMESTAMPTZ NOT NULL,
    window_until    TIMESTAMPTZ NOT NULL,
    warmup_days     INTEGER     NOT NULL,          -- the preamble the harness used (audit)

    -- the reference bt edge set this replay was sized to cover (§4.3 staleness).
    bt_span_start   TIMESTAMPTZ,                   -- first of the last-10 bt entries
    bt_span_end     TIMESTAMPTZ,                   -- last of the last-10 bt exits
    bt_trade_count  INTEGER,                       -- how many completed bt trades (denom basis)

    -- the replay lane's would-have-fired edges (the whole point of the cache).
    -- v1.0: ["2026-07-24T13:41:07Z", ...]. v1.1 enrichment: [{"ts":..,"side":..,
    -- "price":..,"reason":..}, ...] (§2). Reader pairs on ts either way.
    replay_entries  JSONB       NOT NULL DEFAULT '[]',
    replay_exits    JSONB       NOT NULL DEFAULT '[]',

    -- TRUST provenance (Kevin req 1) — everything the UI surfaces next to the score.
    corrected       BOOLEAN     NOT NULL DEFAULT false,  -- decision-time (false) vs healed (true)
    coverage        REAL,                          -- fraction of primary closes in-window that had
                                                   --   live_bars decision-time rows (§8 D4). <1 ⇒ partial.
    divergences     JSONB       NOT NULL DEFAULT '[]',   -- the applicable ledger entries + run-specific
                                                   --   footnotes (UNRES keys, LABEL-DRIFT tokens, window-cap)
    flags_fp        TEXT,                          -- fingerprint of ARMED_FLAGS at compute time (§8 D9)
    engine_sha      TEXT,                          -- git sha of the engine code that produced this

    status          TEXT        NOT NULL DEFAULT 'ok',   -- ok | partial | unres | error
    compute_secs    REAL,                          -- budget telemetry (§7)
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    requested_by    TEXT                           -- button author; NULL = nightly/organic
);
CREATE INDEX IF NOT EXISTS idx_replay_sim_basis_sid
    ON replay_sim_basis (strategy_id, computed_at DESC);
ALTER TABLE replay_sim_basis ENABLE ROW LEVEL SECURITY;   -- service-role only, no public policies
```

### 4.2 `replay_sim_requests` — the queue (only if runner-host = local, option A)

Mirrors `run_history`'s lifecycle exactly (`requested → running → terminal`). If nightly-only
via the batch-worker (option B) is ever chosen instead, this table is unnecessary —
`compute_jobs` already is the queue.

```sql
CREATE TABLE IF NOT EXISTS replay_sim_requests (
    id            BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    strategy_id   BIGINT      NOT NULL,
    requested_by  TEXT,                            -- button author; NULL = nightly sweep
    requested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at    TIMESTAMPTZ,
    finished_at   TIMESTAMPTZ,
    outcome       TEXT        NOT NULL DEFAULT 'requested', -- requested|running|ok|error|ignored
    result_id     BIGINT,                          -- → replay_sim_basis.id on success
    log_tail      TEXT
);
CREATE INDEX IF NOT EXISTS idx_replay_sim_requests_pending
    ON replay_sim_requests (outcome, requested_at);
ALTER TABLE replay_sim_requests ENABLE ROW LEVEL SECURITY;
```

### 4.3 Staleness rule (read path)

The cache is sized to a specific last-10 bt set. When newer bt trades land, the last-10 set
shifts. The reader compares the *current* last-10 bt span to the cached `bt_span_end`:

- current last-10 fully inside `[window_since, window_until]` → **fresh**, score it.
- current last-10 extends past `window_until` → **stale**: pair only the covered subset,
  shrink the denominator to the covered trades, and surface a `stale · N of 10 covered ·
  re-run` chip. Never silently score against a window that doesn't cover the trade.

---

## 5. Read path — the `basis=sim` parameter (F owns; symmetric with V2.8)

`health_last10.py` already pairs bt edges against a lane. SIM adds a lane whose source is the
cache instead of the `alerts` table:

```
GET /api/strategy-health-last10/{sid}?basis=sim
  → row = latest ok replay_sim_basis for sid   (None ⇒ {status:'none'}, page shows "run SIM")
  → replay lane edges = row.replay_entries ∪ row.replay_exits  (timestamps)
  → paired = _greedy_pair(bt_edge_ts, replay_edge_ts, tol)     ← the SAME walk F built
  → points/denom exactly like the fired/theo bases
  → ALSO return: row.divergences, row.coverage, row.corrected, row.computed_at, staleness
```

The endpoint change is small and read-only and mirrors V2.8's `basis=algo` addition — this
is the natural F seam (the endpoint has been F's since #70/#122; it "touches no engine
paths"). The **producer half (job + table + harness enrichment) is E's**. Clean split =
"E designs job+schema, F renders."

**UI (F), Kevin req (1) — surfaced not implied:** SIM column styled like Last-10 but
visually distinct (the V2.8 SIM/ALGO badge). The divergence ledger (§8) and the
coverage/decision-time/staleness provenance render **next to the score** (tooltip or modal
header), never buried. A partial/UNRES run shows its status chip, never a bare number.

---

## 6. Bounded window (Kevin req 3) — warmup is sacred, only the trade span is bounded

The harness already loads warmup ENDING at `since`, never now-anchored: primary
`warmup_asof(sym, tf_s, session, since)` = 7 days (`replay_harness.py:280`); secondary
shadows `days=7 if sec_tf < 3600 else 60` (`:290`); coarse 1Min deep seed `days=60`
(`:349`). **That preamble is the sacred warmup and is NOT touched** — "bounded window" bounds
only the `[since, until]` *decision/trade* span:

```
since  = session_floor( first entry of the last-10 bt trades )   # start-of-session, not mid-bar
until  = last exit of the last-10 bt trades  + small pad         # ensure the last trade's exit lands
```

The harness then auto-loads warmup `[since − 7d/60d, since]`. **The dominant variable cost is
the `until − since` span** (the 1Sec SIP tick load spans the whole window,
`replay_harness.py:385`, plus per-close secondary rebuilds); warmup is a fixed per-run cost.
This is why the bounded window matters and why it is safe: it shrinks the expensive part
without shrinking warmup.

**The catch — and it lands squarely on the target audience.** "No-algo-history" strategies
are *quiet*, so their last-10 bt trades can span **many trading days/weeks**, which is a long
span = a large 1Sec load = the expensive case. Active strategies bound to hours. So the
budget is bimodal (§7), and the mitigation lever is the **span cap** below.

**v1 sizing rule (recommended):** continuous replay over `[since, until]` with a **hard span
cap** (proposed `SIM_MAX_SPAN_TRADING_DAYS`, default 5). If the last-10 span exceeds the cap,
replay only the most-recent trades that fit → a *partial* basis (last K of 10, denom shrinks
like the alert basis already does) + a surfaced `window-capped` divergence. This bounds
worst-case cost and pays warmup exactly once.

**Future optimization (Phase 2, gated):** per-trade-neighborhood windows (10 tight brackets,
each warmup-preamble + one trade) instead of one continuous span. Cheaper 1Sec load for very
sparse strategies, BUT it re-seeds engine state at every bracket → amplifies the session-open
convergence divergence (§8 D5). Per Kevin req (1) this optimization may not ship until its
new divergence is enumerated (it is, D11) AND its test passes (bounded-neighborhood edge set
== continuous-window edge set on a canary, or the delta is characterized and surfaced).

---

## 7. Compute budget — THE SIGN-OFF ASK

**Grounded measurement (E, 07-25 post-close, canary 267 fine-TF, 2h window):**
**31s wall-clock end-to-end**, and most of that was the `main()` dashboard-scoring loop the
*job does not run* (≈4× repeated identical fleet fetches). Replay-only is meaningfully under
31s for a 2h window.

**Extrapolation (order-of-magnitude, to be tightened during build):**

| Case | Span | Est. per-strategy cost |
|------|------|------------------------|
| Active strat, last-10 within ~1 session | ≤ 6.5h | ~30–90s |
| Multi-day, at the 5-day span cap | ≤ 5 trading days | ~2–6 min (1Sec load dominates; DB-egress-sensitive) |
| Quiet strat, uncapped (NOT recommended) | weeks | many min + OOM risk on the 1Sec load |

**On-demand (Phase 1):** one strategy per click, off-hours → 30s–6min. Trivially affordable;
no fleet-wide budget question. **Recommend approving Phase 1 as-is.**

**Nightly-for-all (Phase 2, optional):** cost = (# no-algo-history strategies) ×
(avg per-strategy). If ~20–40 such strategies × ~2min avg ⇒ **~40–80 min of serialized
off-hours compute + sustained prod-DB egress**. This is the number that needs an explicit
budget stamp. Guardrails to sign off *with* it: **(a)** off-hours-only window (post-close →
pre-open), **(b)** strictly serialized (never concurrent with the nightly recompute's DB
load), **(c)** the 5-day span cap, **(d)** skip strategies whose cache is fresher than N days.

**Ask to M/Kevin:** approve Phase 1 (on-demand, local, option A) now; **defer** the Phase 2
nightly budget until there's appetite for precompute-for-all, and decide host (A vs B) then.

---

## 8. THE DIVERGENCE LEDGER (Kevin req 1 — enumerated · tested · surfaced)

Every known way the SIM basis differs from what LIVE actually does. Each entry states the
divergence, the direction of its bias, how it's **tested**, and how it's **surfaced** in the
UI. This ledger is stored per-run in `replay_sim_basis.divergences` (the applicable subset +
run-specific footnotes) and rendered next to the score. **New optimizations MUST add their
row here with a test before arming.**

**Structural (inherent to a bar/engine replay):**

- **D1 · Bar-replay, NOT provider-event replay.** Consumes already-aggregated
  `live_bars.first_*`; bypasses WS parsing, `WsAggMinute`/`BarBuilder` construction,
  reconnect/dup/correction ordering, the flush-vs-fanout race, and grace firing
  (`replay_harness.py:64-72`, Brandon audit). Certifies *"the decision LOGIC is faithful
  GIVEN the recorded bars"*, not *"the bars were constructed/routed correctly."* Live-path
  construction defects (P0 `≥60s` partial force-close, P1 default-model routing, P1 grace
  suppression) live BELOW this harness. *Bias:* SIM can't see construction bugs → SIM may
  score a trade live actually mis-constructed. *Test:* the harness scope-limit is asserted by
  the existing suite; label this divergence permanently ON every SIM row. *Surface:* "bar
  replay — not provider-event; construction defects invisible" tooltip line.
- **D2 · No live close/publish topology (single-monitor hub).** The replay runs ONE
  strategy's monitor in isolation; it cannot see the real fleet's close/publish timing and
  ordering contention (memory: single-monitor hub can't see live close/publish topology).
  *Bias:* SIM assumes ideal scheduling. *Test:* n/a (topology is out of scope by
  construction) — labeled ON every row. *Surface:* "single-monitor — no fleet publish
  contention".
- **D3 · No operational loss modeled → SIM ≥ live by construction.** Stalls, starved bar
  closes (PRIMARY-RESYNC GIL), lost alert saves are absent — that's the POINT (this is a
  *ceiling*), but it means SIM is **logic-would-fire evidence, NEVER delivery evidence**
  (identical to V2.8's algo framing). *Diagnostic value:* SIM-high + fired-low = delivery/ops
  loss; SIM-low = logic/state divergence (a real bug). *Surface:* the badge label itself
  ("SIM — would-fire, not delivery").

**Coverage / data preconditions:**

- **D4 · Decision-time `live_bars` coverage dependency.** `decision_time_bars` reads
  `live_bars` keyed by (symbol, timeframe_seconds) (`replay_harness.py:178-183`) — NOT by
  strategy. If the strategy's symbol+TF was subscribed live over the window (common — any
  strategy on NVDA/SPY/etc.), coverage exists even for a brand-new strategy. If the symbol+TF
  was never live over the window (novel symbol, exotic TF), there are **no decision-time bars
  → the replay can't produce edges**. *Bias:* missing/partial, not wrong. *Test:* unit —
  a sid whose symbol has zero in-window `live_bars` yields `status='unres'`/`coverage=0`,
  never a silent 0-score. *Surface:* `coverage` chip ("decision-time bars: 72% of closes");
  `coverage<1` ⇒ partial badge.

**State-seeding residuals (the harness's own characterized ~1-trade/day floor):**

- **D5 · Session-open / warmup-seed convergence.** Gates seed as-of `since`; live carries
  continuous state, so first-of-window entries can be late (memory: 328's 07-10 first entry
  3.5 min late). A bounded window makes this worse because every window starts cold →
  **see D11.** *Test:* canary (267/314) bounded-window score vs the known full-day ceiling;
  assert within the characterized 1-trade band. *Surface:* footnote when the first in-window
  edge is unpaired.
- **D6 · Rapid re-entry clusters.** When bt double-enters 60–90s apart, the replay takes only
  the 2nd (exit-timing → re-entry-eligibility) — a characterized ~1-trade/day residual.
  *Test:* the `scratchpad/char_328.py`-style trade-by-trade check on a canary. *Surface:*
  footnote when a bt entry inside a cluster is unpaired.

**Fail-loud markers (never a silent zero — already implemented in the harness):**

- **D7 · UNRES.** A required ribbon the replay could not resolve prints `UNRES` and the run
  is `status='unres'`, never scored 0 (`replay_harness.py:51-52,591`). *Surface:* "SIM
  unavailable — unresolved ribbon <key>", not a number.
- **D8 · LABEL-DRIFT.** A resolved-but-unmatchable sub-minute ribbon (`'10Sec-'` vs required
  `'10S-'`) prints a LABEL-DRIFT footnote — a live-engine bug suspect (339's class), not a SIM
  artifact. *Surface:* footnote flagging it as a suspected live bug, spun out separately.

**Model-fidelity provenance:**

- **D9 · Flag-stack drift.** The harness hard-codes `ARMED_FLAGS` (`replay_harness.py:100`).
  If prod flags change and the cache isn't recomputed, SIM reflects a stale engine. *Test:*
  compute `flags_fp` from the armed set; the read path warns/invalidates if the live armed
  fingerprint differs. *Surface:* "engine config as of <computed_at> (flags fp <hash>)"; a
  changed fingerprint shows a `re-run` chip.
- **D10 · Refresher is a MODEL of the live 120s re-true.** The harness simulates the armed
  120s MTF refresher with force-full re-trues from canonical bars as-of each tick
  (`replay_harness.py:34-36,408-423`) — a faithful model, but re-true *timing* differs from
  live's. *Surface:* one-line "120s refresher simulated (armed model)".
- **D11 · [NEW — introduced by the bounded-window optimization, Kevin req 1].** Bounding to
  warmup-preamble + trade-window re-seeds state at the window start rather than carrying it
  continuously from the true session/day start → amplifies D5. *Mitigation:* `since` is
  floored to a session boundary (§6), and warmup depth is never reduced. *Test (gate before
  the optimization counts as trusted):* bounded-window edge set == full-day-window edge set
  on canaries 267 (fine) and 314 (coarse), OR the delta is characterized and surfaced.
  *Surface:* "window bounded to last-10 span (+session-floored warmup)"; if the span cap
  (§6) truncated the set, the `window-capped · K of 10` chip.

**Corrected vs decision-time:** the basis declares which OHLC it used (`corrected`), default
decision-time (the honest ceiling). Not a divergence per se, but surfaced so a `corrected`
run (convergent with bt) is never confused with the decision-time run.

---

## 9. Test plan (maps 1:1 to the ledger; Kevin req 1 = "TESTED")

1. **Reuse-parity:** SIM `_greedy_pair` results == the existing walk on shared fixtures
   (SIM reuses F's function; assert no fork). Include the V2.9 dup-timestamp case.
2. **Fail-loud (D4/D7/D8):** zero-coverage sid → `unres`; unresolved ribbon → `unres`;
   label-drift → footnote. **No path yields a silent 0-score.**
3. **Staleness (§4.3):** bt trades landing past `window_until` → partial score + `stale`
   chip, never a full-denom score against an uncovered window.
4. **Bounded-window fidelity (D5/D11):** canary 267 + 314 bounded-window edge set vs full-day
   — within the characterized band or the delta surfaced.
5. **Enrichment non-perturbation (§2, if v1.1):** enriched fills' timestamps byte-identical
   to timestamp-only path on a canary.
6. **Flag fingerprint (D9):** changed armed set ⇒ read path flags the cache stale.
7. **Budget telemetry:** `compute_secs` recorded; a span-cap breach produces `window-capped`
   not an unbounded run.

---

## 10. Open questions for M / Kevin (need answers before build)

1. **Runner host for on-demand:** confirm option **A** (local, dispatcher-pattern). ✅ = build
   `replay_sim_requests` + a local poller (or a dispatcher mode). ❌/nightly-only = skip the
   queue table, use `compute_jobs` (option B).
2. **Phase 2 (nightly-for-all):** approve/defer, and if approved, the budget in §7 +
   host (A vs B). Recommend **defer** (Kevin demoted it to optional).
3. **Span cap value:** `SIM_MAX_SPAN_TRADING_DAYS=5` OK? (bounds the quiet-strategy worst
   case; larger = more complete last-10 for quiet strategies at higher cost.)
4. **Enrichment in v1.0 or v1.1?** (score column needs only timestamps; the modal wants
   side/price/reason — a small tested harness change.)
5. **Corrected mode exposure:** decision-time only (recommended, the honest ceiling), or also
   offer a `corrected` toggle like the harness's `--corrected`?

---

## 11. Build outline (once §10 answered — NOT started; design-first per the stamp)

- **E (engine/producer):** migration `src/migrations/replay_sim_basis.sql`
  (+ `replay_sim_requests.sql` if option A); a thin `replay_sim_job.py` that wraps
  `replay_harness.replay()` (skips `get_strategy_health`), sizes the bounded window (§6),
  records provenance + ledger + `compute_secs`, writes the cache; the local poller or
  dispatcher mode; harness enrichment (§2) if v1.1; the §9 tests.
- **F (frontend/consumer):** `basis=sim` on `health_last10.py` (symmetric with V2.8's
  `basis=algo`); the "run SIM" button → `POST …/run`; the SIM badge + provenance/ledger
  rendering next to the score.

Sequencing (Kevin): after Phase C + the Run button, per his stated priority.
