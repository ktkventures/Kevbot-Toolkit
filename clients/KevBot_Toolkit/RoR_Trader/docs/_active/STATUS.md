# RoR Trader — STATUS (read me first)

**Updated: 2026-06-18.** This is the single doc to open for "what are we doing and why."
It's the live priority list + current state. Deeper detail lives in the linked docs (kept
in their current locations for now — see Doc Map). New observations get logged here (and in
`Roadmap_Divergence_Hunting.md` for divergence specifics).

## North star
Backtest and live produce the **same trades within ~5s**, so we can **trade reliably**.
Current focus: the first real-money strategies (308–314, TSLA 15Sec).

## Current state (2026-06-18)
**Working:**
- High-TF gates (1h/4h/1d) verified supported live (#23).
- Fidelity Gate shipped — regression net vs backtest↔live drift (`Fidelity_Gate_Guide.md`).
- Divergence-data OOM that crashed the site: fixed + deployed (#28).
- 308 is firing live (281 alerts/day); it's the intended ungated live tester.

**Broken / under investigation:**
- **308 backtest path is broken** — UAD persisted 0 trades; direct windowed run truncates at
  Apr-13 (1,896 trades) instead of running to now; yet the direct prepare path works (2,593).
  So 308 shows 0-paired / all-phantom on Strategy Health. **Real bug, P0.**
- **309/310/313 missed trades** — after UND, the backtest shows entries live didn't fire =
  **gate decision-timing divergence** (live evaluates intra-bar/at-grace; backtest is
  bar-close). This is the documented gated root cause, not a new bug. Diagnose via the
  **Gate Parity tab** (already built).

## Priorities

### P0 — divergences blocking live trading
1. **308 broken backtest path** — UAD writes 0 + windowed run truncates at Apr-13. Fix so
   308 has a correct full-window lane (it's the live tester).
2. **309/310/313 missed-trade divergence** — gate decision-timing (live grace vs backtest
   bar-close). Use the Gate Parity tab. Contributing documented classes: sub-minute
   drift-cascade (live misses clusters), user-pack gates failing on secondary TFs, EH/AH
   session-filter inflating "missed."
3. **`alerts.live_model` NULL on new strategies** → rest-verifier silently skips 308–314
   (they may be unmeasured/mispaired — could itself create phantom/TBD noise).
4. **#16** — guard full-UAD so a transient fetch failure can't silently zero a lane.

### P1 — infra that unblocks investigation + speed
5. **#30** pin the Fidelity Gate fixture window (deterministic golden).
6. **#21 DONE (2026-06-18, `1bbad56`)** scope prep to needed confluence groups. Group-loop
   scoping = ~2× prep + narrower df (OOM relief), byte-identical (golden 3/3 + parity guard).
   Flag `RORT_SCOPE_CONFLUENCE_GROUPS=1` live on api; live engine deliberately left full.
   Profile verdict: trade engine = 76% of a recompute, prep = 24% — so interp/trigger
   scoping (the rest) is marginal (~7-10s on shared path) and was **dropped**. Next real
   recompute lever = trade-engine throughput (separate, risky — needs explicit decision).
7. **#29 PARTIAL DONE (2026-06-18, `13c4786`)** chart speed = scope chart prep (#21) + trim
   returned df to visible window. **Decouple REJECTED** (empirically): daily/4h built from a
   cheap 1Min load ≠ from the 15Sec primary (close Δ0.055, volume Δ527k) → would break
   chart↔backtest heatmap parity. **Residual:** 365d×15Sec ≈ 5.7M-bar *load* for
   sub-minute+daily charts has no fidelity-safe quick fix — deferred architectural item
   (entangled with secondary-source consistency across backtest/live/chart). Side-note:
   1Min vs 15Sec feeds report different volume for the same span — worth a later look.

### P2 — known divergence draggers (documented in Roadmap_Divergence_Hunting / Known_Bugs)
8. **B4 1Min early-fire (#5)** — the "1-minute bars and above" timing item (265/269).
9. 292 SR-channels pack (WIP), 275 dead-live (0 alerts/22 BT), 271 rare 5m-gate under-pass.
10. Streaming-tick backtest under-fires ~7%; backtest session-filter on primary_df path;
    manual-backfill +17% vs UAD; ralph_engine watchdog for secondary-TF shadow engines.

### P3 — larger builds (discuss scope first) + cleanup
11. **Live-Mode ground-truth embedding** (#4) — per-bar telemetry on ralph_engine so gate
    parity is read, not reconstructed.
12. **Tier-3 post-deploy monitor** (#31); Mass Builder #22 (HIFI cache bug) / #25
    (save-persists-trades) / #26 (cold-seed).
13. **Docs cleanup phase 2** — actually move active docs into `_active/`, archive done specs.

## Doc map (current locations; move = phase 2)
**Active (read these):**
- `Roadmap_Divergence_Hunting.md` — divergence root causes + draggers (has some stale 6/11
  entries; freshness pass needed).
- `Known_Bugs.md` — ~18 active documented bugs (2 weeks old; needs triage).
- `First_Live_Money_Test_2026-06-17.md` — the 308–314 launch log.
- `Fidelity_Gate_Guide.md` (reader) + `Fidelity_Gate.md` (spec) — the regression gate.
- `Append_Edge_Fossilization.md` — UND/append mechanics.
- `Fleet_UAD_Report_2026-06-12.md` — historical UAD log (reference).

**Outdated / big-picture (don't drive daily work):** `Roadmap_To_Scale.md`, the PRD, older
Implementation_Spec_Phase_* docs (candidates for a `done/` or `archive/` folder in phase 2).

## Convention
- New observation that needs action → add to the right Priority bucket above (and to
  `Roadmap_Divergence_Hunting.md` if it's a divergence specifically).
- This doc is the entry point; keep it current at each session's end.
