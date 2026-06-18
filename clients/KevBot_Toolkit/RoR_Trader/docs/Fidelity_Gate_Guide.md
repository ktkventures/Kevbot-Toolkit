# Fidelity Gate — Plain-English Guide & Journey

_A readable companion to `Fidelity_Gate.md` (the technical spec). This is the doc to
**read** to understand what the validation does, and the **living log** of what it
protects against. If something feels missing, flag it — we add to the "Coverage" and
"Change Log" sections as we harden against new bugs._

---

## 1. What this is, in one paragraph

The Fidelity Gate is a **safety net that catches backtest↔live divergence before it ships.**
It took weeks to get the backtest engine and the live engine to produce the *same trades
within ~5 seconds*. The danger is that a future code change silently shifts how a trigger
or gate resolves — and we don't notice until it's costing real money. The Gate's job: any
change that could move a trade gets compared against a **frozen, known-good reference**, and
if anything shifts, it **fails loudly** instead of slipping through.

## 2. The fear it addresses

> "I just really don't want to introduce subtle divergences that are invisible to us until
> it starts biting us in the butt." — the whole reason this exists.

A gate that *cries wolf* (false alarms) is as bad as no gate — you'd stop trusting it. So the
Gate is built to be **deterministic**: same inputs → same result, every time, no matter when
it runs.

## 3. How it works — three layers

**Tier 1 — Golden replay (built, fast, offline).** We "freeze" a strategy: its raw price
bars + the exact gate/trigger columns + the exact trades it produced, saved to disk. Later,
we feed those frozen bars back through the *real* engine and check the result is
**byte-for-byte identical** to the frozen reference. If a code change shifted a gate, the
bytes won't match → fail. It needs no live data and runs on every push (the fast synthetic
version) + on demand (the real-bar version).

**Tier 2 — Live before/after (built, slower).** Before a risky change, snapshot the current
trusted trades for the live strategies. Make the change. Regenerate and diff. Catches drift
on *real recent data* that the frozen fixtures might not cover.

**Tier 3 — Post-deploy monitor (planned, optional).** After deploying, watch incoming live
alerts + the by-deploy divergence report (per strategy) to catch "this deploy made strategy
X start diverging." Slowest (needs data to accumulate), so it's a standardized follow-up,
not a per-push check.

## 4. When do we run it? (the `[FC]` rule)

- **Fidelity-critical (`[FC]`)** change — anything that *could* change how indicators /
  triggers / gates / trades resolve, or how backtest-vs-live data is loaded. **Must pass the
  Gate before merging.** Tag the commit `[FC]`.
- **Surface** change — UI, dashboards, docs, non-engine stuff. No gate needed.

Example: disabling the divergence *tab* (June 17) was **surface** — it changed no math.
The upcoming prep-scoping work (#21) is **fidelity-critical** — it goes through the Gate.

## 5. What it currently protects against (coverage)

The frozen reference set covers one strategy per indicator "pack" + each gate-timeframe
class, so a change that breaks any of these fails the Gate:
- **Every in-use pack** (swing, ut_bot, bollinger, vwap, rvol, supertrend, stochastic,
  strat_assistant, ema variants, macd variants, rsi) — committed subset 308/309/313 +
  regenerable broad set.
- **Gate timeframes** from 2m up through 1-day (the high-TF gates that are hardest).
- **Trade-level parity** — entries, exits, prices, exit reasons.
- **Loud-failure behaviors** — e.g. the swing-stop now *rejects* an empty buffer instead of
  silently falling back (a past divergence source); the Gate asserts the loud rejection.

## 6. How to run it
```
# Fast synthetic suite (runs in CI on every push automatically):
cd src && python test_unified_parity.py

# Real-bar golden replay (committed subset):
cd src && python -m fidelity.golden

# Live before/after around an [FC] change:
cd src && python -m fidelity.fidelity_gate --snapshot   # before
cd src && python -m fidelity.fidelity_gate --check      # after

# Regenerate / add a fixture:
cd src && python -m fidelity.capture <strategy_id> 30
```

## 7. Change Log (the journey)

- **2026-06-17 — Gate created.** Tier-1 golden + Tier-2 live command + per-push CI built and
  committed (509a6d9). Committed reference: strategies 308/309/313, validated byte-identical.
  Synthetic suite green (31/7/6). Fixed 2 stale tests (swing-stop loud rejection; legacy
  bar-open vs unified bar-close timestamp convention).
- **2026-06-17 — Determinism hardening identified (#30).** Found the golden replay could
  drift if run *in the same hour* the fixture was captured (the engine clips bars newer than
  "now", so a same-session replay clips a different tail). Fix queued: pin an explicit
  start/end date in each fixture + trim the last hour of bars, so the window never shifts and
  the Gate is identical no matter when it runs. (For normal next-day/nightly use it's already
  deterministic.)
- **2026-06-17 — Tier-3 post-deploy monitor proposed (#31).** Standardized follow-up to watch
  incoming alerts + by-deploy divergence per strategy after a deploy.

## 8. Known gaps / not-yet-covered (tell me if you want any of these added)
- **#30 window pinning** — until done, don't run the real-bar golden in the *same session* a
  fixture was captured (it'll false-alarm on the trailing bars). Normal use is fine.
- **Broad fixture set is regenerable, not committed** — only the 308/309/313 subset is in git
  (size). The full per-pack set is rebuilt locally via `capture.py` when needed.
- **Tier-3 (post-deploy live monitor)** — designed, not built.
- **Confluence-group edits** — if you change a group's parameters, the golden reference for
  affected strategies must be re-blessed (re-captured). Not yet automated.
- **New bug classes we want to protect against** — add them here as we hit them, then add a
  test/fixture so the Gate catches them next time.
