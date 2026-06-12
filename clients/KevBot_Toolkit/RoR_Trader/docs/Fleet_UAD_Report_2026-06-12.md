# Fleet UAD Overnight Report — 2026-06-11/12

**Context:** Kevin launches fleet-wide Update All Data (~47 strategies) after extended
hours close (~00:00Z). Every lane gets rebuilt on post-fix code (gap healer + lens
faithfulness + snapshot lineage guard + HiFi signal-exit fix, all deployed 2026-06-11).
Claude monitors overnight, appends one standardized block per completed strategy +
judgment, and flags residual-bug theories. Measurement window per strategy: today's
RTH 14:45–20:00Z, alerts↔BT pairing at ±5s (`_uad_fleet_report.py --sid N`).

**How to read a block:** combined% = paired/(paired+phantom+missed) at ±5s. Entries
and signal exits should sit at med 0.0s with ≥90% within ±5s on liquid RTH for
ungated strategies; stop exits must show early≈0 (early stops were the HiFi-rewrite
signature — eliminated). Exit mix ~20% stops is the healed profile for trigger
strategies (57% was the rewriter profile). ⚠ lines list exact timestamps worth
checking on the chart.

**Known benign patterns today (NOT bugs):**
- Misses inside deploy-restart windows (14:33, 14:38, 15:27-15:29, 16:28, 21:33 Z):
  the live engine was down — alerts genuinely don't exist there.
- After-hours gray cells on genuinely tradeless windows (matches backtest dropna).
- A few pre-flush-hook after-hours WS stragglers (e.g. SPY 20:38:40Z) pending the
  next restart's 2h reconcile.

**Today's deploy timeline + which side each change affects (CRITICAL CONTEXT):**

| time (Z) | deploy | side affected |
|---|---|---|
| 14:33 + 14:38 | gap healer + lens faithfulness (f723af8/1cfbd12) | LIVE engine (heals WS-missed bars) + display |
| 16:28 | deploy-hole cache reconcile + flush_stale_bars gap-heal hook (3964eab/1d77665) | LIVE display + thin-symbol heal latency |
| 18:19 | API restart (no code) | none (thread-starvation fix) |
| 21:33 | snapshot lineage guard + HiFi signal-exit fix (5b6f7a9/3016abb) | BACKTEST lane only |
| 21:36 | docs only (cf633fb) | none |

**Alert/live-engine health today:** alerts flowed ALL day; the live model needed no
correctness fixes after the morning gap healer — the afternoon/evening hunt was
entirely BACKTEST-lane (snapshot fossilization + HiFi rewriter), where the live
alerts turned out to be RIGHT. Implications for reading today's data:
- Live alerts BEFORE ~14:38Z = old engine (sparse bars era); AFTER = gap-healer era.
  Today is therefore a MIXED-ERA alert day; tomorrow is the first pure-era day.
- Alert gaps during the 5 deploy windows are structural (engine down ~1-3 min).
- The fleet UAD rewrites every BACKTEST lane with all fixes → divergence metrics
  improve from BOTH sides: honest reference (backtest fixes) + better live decisions
  (post-14:38 alerts).
- Mass Strategy Builder VERIFIED safe: it calls the shared
  `backtest_service._hifi_resolve_trades` (mass_builder.py:419) → inherits the
  signal-exit fix automatically. Same for any path using the shared service fns.

**Bug-theory watchlist going in (from docs/Plan_Decision_Divergence_2026-06-12.md):**
B2 full-UAD live-edge writes (nil overnight); B3 marginal decision-time crosses
(~irreducible, small); B4 grace-fire residue; B5 gated-cohort phantom round-trips
(gate interplay — the main open question; watch gated strategies' phantom clusters).

---

## SAMPLE / FORMAT REFERENCE (pre-fleet-UAD)

### sid 302 — PACKTEST · UT Bot V4 · trigger  ✅ HEALTHY (the gold standard)
- SPY 10Sec | no gate | lane fresh: 2026-06-11T22:14:01 | anchor: None (full UAD; anchor stamps on first append)
- window 06-11T14:45 → 06-11T20:00Z: **±5s: paired=229 phantom=7 missed=13 → combined 92.0%** | ±10s: 232/4/10 → 94.3%  (alerts 119E/117X, BT trades 121)
- entries:      n=121 med=+0.0s ±5s=114/121 (94%) ±10s=117/121 (97%) early=4 late=3 unmatched=0
- signal exits: n=94 med=+0.0s ±5s=91/94 (97%) ±10s=91/94 (97%) early=2 late=1
- stop exits:   n=25 med=+0.0s ±5s=24/25 (96%) ±10s=24/25 (96%) early=0 late=1 unmatched=2
- exit mix: 96 signal / 25 stop (21% stops)
- ⚠ unmatched BT exits: ['15:27:40(utv4_bear_flip)', '16:29:10(utv4_bear_flip)']

**Judgment:** Healthy across the board. Both unmatched exits fall inside deploy-restart
windows (15:27 + 16:28 deploys) — genuinely missed live alerts, expected today only.
The 13 misses largely cluster around the same restarts. Residual phantom=7 over 5+
hours ≈ B3 marginal-cross territory. This is the profile to compare everything against.

---

## FLEET RESULTS (appended as strategies complete)

### sid 307 — TEST-GAPHEAL-KO-10s-2m-0611  ✅ GOOD (canary, gated, thin tape)
- KO 10Sec | gate: 2m-UT_BOT_V4-BULL_TREND | lane fresh: 2026-06-12T00:28:34 | anchor: None
- window 06-11T14:45 → 06-11T20:00Z: **±5s: paired=67 phantom=14 missed=6 → combined 77.0%** | ±10s: 72/9/1 → 87.8%  (alerts 40E/41X, BT trades 36)
- entries:      n=36 med=+0.0s ±5s=35/36 (97%) ±10s=35/36 (97%) early=1 late=0 unmatched=0
- signal exits: n=29 med=+0.0s ±5s=29/29 (100%) ±10s=29/29 (100%) early=0 late=0
- stop exits:   n=7 med=+8.0s ±5s=2/7 (29%) ±10s=7/7 (100%) early=0 late=5 unmatched=0
- exit mix: 29 signal / 7 stop (19% stops)
- ⚠ orphan alerts: cluster 15:56–15:59Z (3 entry/exit round-trips)

**Judgment:** Strong for a gated thin-tape canary created today. Entries 97%, signal
exits 100%. TWO findings:
1. **NEW PATTERN — thin-tape stop-fill latency:** all 7 stops pair LATE (med +8s,
   100% within ±10s, ZERO early). On sparse tape the live engine can only see a stop
   cross when the next trade prints (per-second data gaps), while HiFi resolves the
   exact 1s cross. Structural, bounded ≤10s, NOT the early-stop bug. Candidate small
   optimization, low priority (thin symbols only).
2. The 15:56–15:59Z phantom round-trip cluster predates the flush-hook deploy
   (21:33) — KO's engine bars were sparse-era then (heals only via sweep). Likely an
   era artifact that should not recur tomorrow; verify on tomorrow's pure-era data
   before treating as a B5 gate bug.

### sid 306 — TEST-GAPHEAL-DIA-10s-2m-0611  🟡 OK (canary-birth artifacts depress the headline)
- DIA 10Sec | gate: 2m-UT_BOT_V4-BULL_TREND | lane fresh: 2026-06-12T00:50:31 | anchor: None
- window 06-11T14:45 → 06-11T20:00Z: **±5s: paired=76 phantom=18 missed=46 → combined 54.3%** | ±10s: identical (not latency — existence)
- entries:      n=39 med=+0.0s ±5s=37/39 (95%) early=1 late=1 unmatched=22
- signal exits: n=31 med=+0.0s ±5s=30/31 (97%) early=0 late=1
- stop exits:   n=11 med=+0.0s ±5s=9/11 (82%) early=1 late=1 unmatched=19
- exit mix: 41 signal / 20 stop (33% stops)
- ⚠ unmatched BT entries cluster 14:46–14:55Z; orphan alerts 17:11/17:29/18:14/18:28Z

**Judgment:** Matched-pair quality is excellent (95/97/82% — med 0.0s everywhere), so
the engine agreement is fine. The 54% headline is existence artifacts unique to this
canary's birth: the missed cluster (14:46–14:55Z) is the window where DIA's brand-new
hub was warming/healing on a just-subscribed symbol (DIA had ~0 WS bars until ~14:42)
— BT has full REST history there, live alerts couldn't exist yet. ±10s = ±5s confirms
no latency tail. Orphan round-trips (17:11+) = same gated/era bucket as 307's. NOT a
new bug class; expect clean numbers tomorrow on its first full pure-era day.

### sid 305 — PACKTEST · VWAP v2 · gate  🔴 GATE DIVERGENCE (first real B5 case — with a suspect mechanism)
- SPY 10Sec | gate: 2m-VWAP_V2->+2σ | lane fresh: 2026-06-12T00:58:04 | anchor: None
- window 06-11T14:45 → 06-11T20:00Z: **±5s: paired=19 phantom=85 missed=1 → combined 18.1%** | ±10s identical (existence, not latency)
- entries: n=10 ±5s 90% · signal exits: n=9 100% · stop: n=1 100% — matched pairs are PERFECT
- exit mix: 9 signal / 1 stop | live took 52 round-trips, BT only 10
- ⚠ orphan-alert round-trips throughout (17:55–18:01Z sample)

**Judgment:** Classic gate-state divergence: when both engines trade, they agree to the
second — but live's `2m-VWAP_V2->+2σ` gate passes ~5x more often than BT's. TWO prior
breadcrumbs converge here: (1) VWAP v2 was already the known phantom-heavy laggard
pack (pre-existing roadmap item), and (2) ralph has a DOCUMENTED volume caveat: ≥60s
builders fed by ws_agg fan-out double-count volume during forming periods — "affects
volume-based interpreters (VWAP_V2, RVOL_V2) only" (code comment at
_fanout_to_primary_builders_gt_60). A 2x-volume skew on the live 2m bars shifts VWAP
bands → gate opens when BT's doesn't. **TOP RESIDUAL BUG THEORY: live 2m VWAP/RVOL
gate bars carry inflated volume.** Testable tomorrow: compare live-cache 2m volume vs
REST 2m volume; if ~2x on forming-period closes, that's the fix target. Expect the
same signature on any RVOL_V2-gated strategy tonight (291 was rv2 per GATE_DIAG logs).

### sid 304 — PACKTEST · VWAP v2 · trigger  🔴 VWAP TIMING SCATTER (refines the volume theory)
- SPY 10Sec | no gate | lane fresh: 2026-06-12T01:07:52 | anchor: None
- window: **±5s: paired=14 phantom=27 missed=22 → combined 22.2%** | ±10s: 21/20/15 → 37.5%
- entries: n=16 ±5s 50% (7 EARLY) · signal exits: n=9 ±5s 44% (late-leaning) · stops: n=4 med +5.5s
- divergence is BIDIRECTIONAL (27 phantom / 22 missed) and TIMING-scattered — unlike 305's
  one-sided gate flood and unlike every price-based pack so far (med 0.0s everywhere)

**Judgment + synthesis (304+305 together):** VWAP v2 diverges as BOTH trigger and gate,
while price-based packs (UT Bot 302: 92%, eppv3 entries everywhere: med 0.0s) are clean.
Refined theory: **live bar VOLUME ≠ REST volume, and VWAP integrates volume cumulatively
from session start** — so any per-bar volume discrepancy (WS undercount of
condition-excluded trades; the documented ws_agg fan-out double-count on ≥60s builders;
healed bars carrying REST volume mid-stream) accumulates in the live VWAP line and
shifts every band crossing by a few seconds-to-bars, both directions. Matches 304's
scatter (crosses early AND late) + 305's gate flood (+2σ band shifted).
**Tomorrow's test:** per-bar volume diff (live cache vs REST) at 10s and 120s + recompute
VWAP over each series and measure band deltas. Fix direction if confirmed: make live
VWAP volume-source REST-converging (verifier already corrects OHLC — volume correction
may simply need to be included in apply_rest_correction's effective_bar), and kill the
fan-out double-count. NOTE: RVOL_V2 strategies should show the same signature (watch 291).

### sid 303 — PACKTEST · UT Bot V4 · gate  🔴 GATE FLOOD CONFIRMED ON A PRICE-BASED GATE (B5 is real & distinct)
- SPY 10Sec | gate: 2m-UT_BOT_V4-BULL_TREND | lane fresh: 2026-06-12T01:17:14 | anchor: None
- window: **±5s: paired=103 phantom=135 missed=9 → combined 41.7%** | ±10s: 42.3% (existence, not latency)
- entries: n=52 ±5s 96% med 0.0s · signal exits: n=44 98% · stops: n=12 83% — matched pairs near-perfect
- exit mix: 44 signal / 12 stop (21%) | live took ~2.2x the BT's round-trips (120E alerts vs 56 BT trades)
- ⚠ phantom round-trips ALL DAY (14:52Z onward), not era-limited; 4 unmatched BT entries

**Judgment:** This isolates B5 cleanly and rules out volume as its cause: UT Bot's 2m
gate is PRICE-based, yet live's gate passes ~2.2x more often than BT's — the original
H1 gate-mode divergence (live ~2.4x), still alive after all bar-layer fixes. Contrast
matrix now: trigger-only UT Bot (302) 92% ✓ · gated UT Bot (303) 42% w/ perfect matched
pairs · VWAP anything (304/305) broken separately (volume). KEY TENSION: the ribbon
says batch-vs-batch 2m gate STATES match 100% on identical bars — so the live ENGINE's
emergent gate state (incremental shadow `_mtf_confluence` read at decision time) must
differ from both. Suspects, in order: (a) incremental-vs-batch UT Bot state on the 2m
shadow engine (trail seeded/reset differently → BULL_TREND when batch says BEAR),
(b) decision-timing: grace fires read the gate buffer at a moment whose PB differs
from bar-close PB. **Decomposition tool already exists:** the Gate Parity Analysis
card (theoretical replay, per-live-entry PB/CB classification) on tomorrow's pure-era
data. This is tomorrow's #1 investigation — the biggest remaining divergence class.

### sid 302 — fleet rebuild ⚠️ PARTIAL LANE (operational finding, not a parity result)
- The fleet pass rebuilt 302 at 01:30 with trades covering ONLY 06-01→06-05 — todays
  healed trades were deleted and not replaced (siblings 303/304/305, identical configs,
  rebuilt through today). A silently truncated run: job reports success, lane is wrong.
- ⚠️ OPERATIONAL BUG THEORY (NEW): bulk-UAD per-strategy runs can complete with a
  truncated data window (transient fetch failure?) with NO error surfaced. Future fix:
  UAD should assert lane coverage (newest trade ≥ run start − lag) before declaring a
  strategy done. Also discovered: lanes read EMPTY mid-rebuild (delete→reinsert gap) —
  never measure during a UAD.
- REMEDIATION QUEUED: re-run 302's single UAD after the fleet job completes (avoiding
  concurrent load), then re-verify against its 22:08 gold standard (92.0%/94.3%).
- My monitoring now checks lane coverage explicitly on every block (newest-trade ts).

### sid 301 — PACKTEST · Swing 1-2-3 · gate  🟡 GATE FLOOD (milder — mostly-open gate)
- SPY 10Sec | gate: 2m-SWING_123-NEUTRAL | coverage ✓ (23:59) | anchor: 23:00
- **±5s: 155/80/11 → 63.0%** | ±10s: 65.0% | entries 93% · signal exits 97% · stops 94%, all med 0.0s, zero early
- exit mix 66/17 (20% stops)

**Judgment:** Same B5 signature as 303 but milder (~1.4x phantom ratio vs 2.2x) —
consistent with SWING_123-NEUTRAL being a mostly-open gate: the more often the gate
SHOULD be open, the less room for live/BT gate-state disagreement. Reinforces the
gate-state divergence model. Matched-pair quality again flawless. Deploy-window
unmatched exits (15:27/16:29) as expected.

### sid 300 — PACKTEST · Swing 1-2-3 · trigger  ✅ HEALTHY (90.9%)
- SPY 10Sec | no gate | coverage ✓ | **±5s: 310/12/19 → 90.9%** (±10s identical)
- entries 96% · signal exits 98% · stops 90% — med 0.0s everywhere
- ⚠ unmatched items cluster at deploy windows (15:28, 16:28–29) + one at RTH close edge

**Judgment:** Second ungated price-based strategy at ~91% (302: 92.0%) — the healthy
profile is reproducible across packs (UT Bot, Swing 1-2-3). Ungated + price-based =
~90%+ is now the established baseline. Nothing new.

### sid 299 — PACKTEST · SuperTrend · gate  🟡 GATE FLOOD (mild — 73.2%)
- SPY 10Sec | gate: 2m-SUPERTREND-BULL_TRENDING | coverage ✓ | anchor 23:06
- **±5s: 153/49/7 → 73.2%** | ±10s 74.9% | entries 95% · signal exits 97% · stops 100% — med 0.0s
- exit mix 65/15 (19%) | phantom ratio ~1.3x

**Judgment:** Best gated result tonight. Gate-severity ladder now: SUPERTREND 73% >
SWING-NEUTRAL 63% > UT_BOT 42% (> VWAP 18%, volume-confounded) — all with flawless
matched pairs. B5's magnitude tracks how often the specific 2m gate state flips near
boundaries (UT Bot trend flips most on 2m SPY; SuperTrend trends persist). Same single
mechanism, pack-dependent exposure. Stops 100% — third strategy with zero early stops.

### sid 298 — PACKTEST · SuperTrend · trigger  🟢 OK (68.8% on small n)
- SPY 10Sec | no gate | coverage ✓ | **±5s: 33/8/7 → 68.8%** | ±10s 76.1%
- entries 85% (n=20, 3 outliers) · signal exits 100% · stops 70% ±5s → 80% ±10s, med +4.5s LATE lean
- exit mix 9/11 (55% stops — SuperTrend flip exits are rare; not a rewriter signature, stops pair clean/zero-early)

**Judgment:** Small-sample (20 trades) — each event = 2pts, so 68.8% here ≠ a new class;
±10s already lifts to 76%. Two soft notes: stop fills lean late (+4.5s med — echoes
KO's stop-latency pattern, here on volatile moments rather than thin tape), and one
orphan round-trip at 14:49Z (mixed-era morning). No new bug class.

### sid 297 — PACKTEST · Strat Assistant · gate  🔴 EXTREME GATE FLOOD (11.0% — ladder extreme)
- SPY 10Sec | gate: 2m-STRAT_ASSISTANT-INSIDE | coverage ✓
- **±5s: 23/185/1 → 11.0%** | matched pairs LITERALLY 100% on all three categories (12E/7sig/4stop)
- live ~9x the BT round-trips

**Judgment:** The ladder's extreme end, and the strongest confirmation of the B5 model:
INSIDE-bar is the flippiest possible 2m state (re-evaluated every bar, alternates
constantly) → maximum live/BT gate-state disagreement, while every co-decided trade
matches to the second. Gate-severity ladder final form: INSIDE 11% < UT_BOT 42% <
SWING-NEUTRAL 63% < SUPERTREND 73% (< ungated ~91%). Phantom ratio ∝ gate flip
frequency — a textbook dose-response curve pointing at one mechanism in the live 2m
gate-state path.

### sid 296 — PACKTEST · Strat Assistant · trigger  ✅ BEST OF NIGHT (96.6%)
- SPY 10Sec | no gate | coverage ✓ | **±5s: 599/10/11 → 96.6%** | ±10s 97.6% — on n=305 trades!
- entries 98% · signal exits 99% · stops 96% — med 0.0s, zero early stops
**Judgment:** New fleet record, and on the LARGEST sample (305 trades, 5x most others).
Ungated + price-based at scale = 96.6% — exceeds the 95% healthy-system target. The
same pack whose gated twin (297) read 11% — gate on/off is worth 85 points. B5 is
unambiguously the whole story for gated SPY strategies.

---
## 🔧 OVERNIGHT FIX SHIPPED (local commit d6a9be8, deploys pre-market)

**B5 prime suspect REPRODUCED AND FIXED at ~03:00Z.** Unit test: feed a 2m shadow a
rising series (gate records bullish) → REST correction crashes the bar, flipping MACD
bearish → pre-fix, `_mtf_confluence` stayed BULLISH (stale); the live gate trades up
to 2 minutes on pre-correction state after every correction (~23/window on SPY).
Fix: re-derive records after the shadow recompute in apply_rest_correction (the 0609a
fan-out fix never reached this path). 44 tests green. Deploys pre-market with the
fleet-completion batch; tomorrow's RTH gated phantom ratios are the verdict.
---

### sid 295 — PACKTEST · Stochastic Oscillator · gate  🔴 GATE FLOOD (17.0% — fits the ladder)
- gate: 2m-STOCH-OVERBOUGHT_BEARISH | coverage ✓ | **±5s: 26/127/0 → 17.0%**
- matched pairs 100%/100%/100% — perfect; live ~6x BT round-trips; ZERO missed
**Judgment:** Slots into the ladder exactly where an oscillator overbought state
belongs (flippy on 2m): INSIDE 11% < STOCH-OB 17% < UT_BOT 42% < SWING 63% <
SUPERTREND 73%. Notably ZERO missed — live's stale gate only ever OVER-passes
(consistent with stale-records: a gate stuck open floods; corrections rarely leave
it stuck closed because the next 2m close refreshes it). All evidence keeps pointing
at the one fixed mechanism (d6a9be8).

### sid 294 — PACKTEST · Stochastic Oscillator · trigger  ✅ HEALTHY (82.2%)
- no gate | coverage ✓ | **±5s: 88/13/6 → 82.2%** | ±10s 84.0%
- entries 93% · signal exits 100% · stops 100% (n=30, zero early/late!) — stop-heavy mix (68%) is
  the pack's character (oscillator triggers cut early, stops do the work), pairs perfectly
- unmatched items all at the 16:28–16:30 deploy window
**Judgment:** Healthy. Gated twin (295) at 17% — another 65-point gate toggle. Ungated
band now: 296: 96.6, 302: 92.0, 300: 90.9, 294: 82.2, 298: 68.8(small-n).

### sid 293 — PACKTEST · SR Channels · gate  🔴 GATE FLOOD (45.6% — ladder-consistent)
- gate: 2m-SR_CHANNELS-ABOVE_RESISTANCE | coverage ✓ | **±5s: 109/113/17 → 45.6%** | ±10s 47.5%
- entries 90% · signal exits 91% · stops 100% | ~1.8x phantom ratio
**Judgment:** Sits beside UT_BOT (42%) on the ladder — ABOVE_RESISTANCE flips at
comparable frequency. Unmatched items at deploy windows (15:27-29, 16:31-33). Same
mechanism; nothing new. (Note: 293 was the flat strategy in the afternoon cohort
measurement — the fleet rebuild + tonight's lens shows it was gate-bound all along.)

### sid 292 — PACKTEST · SR Channels · trigger  🔴 PACK-LEVEL SCATTER (13.5% — known laggard pack)
- no gate | coverage ✓ | **±5s: 30/99/93 → 13.5%** | ±10s 18.9%
- entries 27% ±5s (23 EARLY + 13 late!) · signal exits med +15s (8%!) · stops 40% — bidirectional
  timing scatter everywhere, BREAKS the ungated-healthy pattern
**Judgment:** NOT the gate bug (no gate) — this is the 304-VWAP profile: trigger-level
timing scatter. Crucially, SR Channels was ALREADY the other known laggard pack
("Phase 4 pack cleanup: VWAP v2 + SR Channels migration completion" — pre-existing
roadmap). So the two ungated outliers tonight are exactly the two pre-known problem
packs. Class: pack-specific incremental-vs-batch parity (SR channel levels are
lookback/pivot-anchored — window-sensitive like VWAP's cumulative anchor, mechanism
differs from the volume theory). Action: fold into the Phase-4 pack cleanup with
VWAP — measure each pack's incremental-vs-batch parity directly (the parity simulator
exists for exactly this). Not a new engine bug.

### sid 291 — PACKTEST · Relative Volume v2 · gate  🔴 PREDICTION CONFIRMED (1.0% — worst of fleet)
- gate: 2m-RVOL_V2-EXTREME | coverage ✓ | **±5s: 1/103/1 → 1.0%** — BT took ONE trade, live took 52
**Judgment:** The volume theory's forecast ("expect the same signature on any
RVOL-gated strategy") lands exactly: live's volume-inflated 2m bars read RVOL-EXTREME
~constantly (the documented fan-out double-count alone ≈ 2x volume → "extreme"
relative volume nearly always true), while BT's honest volume almost never crosses
EXTREME. Worst combined% of the fleet, and the cleanest possible confirmation that
live 2m VOLUME is broken. Volume-integrity fix (correct volume in REST corrections +
kill fan-out double-count) is now co-#1 priority with the deployed gate-records fix —
note RVOL/VWAP gates compound BOTH bugs.

### sid 290 — PACKTEST · Relative Volume v2 · trigger  ✅ HEALTHY (87.1%) — and it LOCALIZES the volume bug
- no gate | coverage ✓ | **±5s: 379/33/23 → 87.1%** | ±10s 89.7% (n=201!)
- entries 95% · signal exits 95% · stops 84%→96% ±10s
**Judgment:** The twist that completes the volume picture: RVOL TRIGGER (10s primary)
healthy at 87% while RVOL GATE (2m) was 1.0%. Localization: sub-minute builders are
per-second-fed (NO fan-out double-count) → 10s volume ≈ honest; the 2m gate bars are
fan-out-built (doubled volume) → broken. And why is VWAP broken even at 10s (304)?
VWAP is CUMULATIVE session volume — small per-bar deltas accumulate; RVOL is a RATIO
(numerator/denominator share the bias → robust). So: **the volume fix is specifically
(a) the ≥60s fan-out double-count and (b) cumulative-anchor packs' sensitivity** —
10s per-bar volume is approximately fine. Fix (a) likely rescues RVOL/VWAP GATES
entirely; VWAP TRIGGER may also need volume in REST corrections.

### sid 289 — PACKTEST · RSI Zones 2 · gate  🔴 GATE FLOOD (5.9% — rare-state gate)
- gate: 2m-RSI2-EXTREME_OVERBOUGHT | coverage ✓ | **±5s: 6/96/0 → 5.9%** | matched 100% (n=6 edges)
**Judgment:** Rare-state gate (EXTREME_OVERBOUGHT ≈ almost never true in BT — 3 trades
all day) → any stale-open window dominates → near-total flood, zero missed. Fits the
established mechanism at the rare-state extreme. Nothing new.

### sid 288 — PACKTEST · RSI Zones 2 · trigger  ✅ HEALTHY (94.3%)
- no gate | coverage ✓ | **±5s: 183/2/9 → 94.3%** | ±10s 96.4% — only 2 phantoms all day
- entries 99% · signal exits 97% · stops 88% | unmatched at deploy windows
**Judgment:** Second-best of the fleet. Gated twin (289): 5.9% — the largest gate
toggle yet (88 points). Pattern complete: every healthy pack's gated twin is broken
in proportion to its gate's state rarity/flip-rate.

### sid 287 — PACKTEST · MACD Line v2 · gate  🔴 GATE FLOOD (42.6% — ladder-consistent)
- gate: 2m-MACD_LINE_V2-M>S+ | coverage ✓ | **±5s: 80/106/2 → 42.6%** | pairs 95/100/100%
**Judgment:** MACD crossover-state gate lands beside UT_BOT (42%) — comparable 2m flip
frequency. Mechanism fingerprint intact (perfect pairs, one-sided flood). Nothing new.

### sid 286 — PACKTEST · MACD Line v2 · trigger  ✅ HEALTHY (92.4%)
- no gate | coverage ✓ | **±5s: 146/2/10 → 92.4%** | ±10s 94.9% — 2 phantoms all day
- entries 97% · signal exits 98% · stops 94% | gated twin 287: 42.6%
**Judgment:** Healthy band, 7th pack family confirming the dichotomy. Nothing new.

### sid 285 — PACKTEST · MACD Histogram v2 · gate  🔴 GATE FLOOD (30.4%)
- gate: 2m-MACD_HIST_V2-H+up | **±5s: 55/125/1 → 30.4%** | pairs 96/100/100%
**Judgment:** Histogram-direction gate (flippier than line-cross) sits between INSIDE
(11%) and UT_BOT/MACD-line (42%) — ladder-consistent. Nothing new.

### sid 284 — PACKTEST · MACD Histogram v2 · trigger  ✅ HEALTHY (95.2% on n=277)
- no gate | coverage ✓ | **±5s: 534/7/20 → 95.2%** | ±10s 96.6% | stops 17/17 perfect
**Judgment:** Above the 95% target on the second-largest sample. Gated twin: 30.4%.
Healthy band now has TWO strategies >95% (296, 284). Nothing new.

### sid 283 — PACKTEST · EMA Stack v2 · gate  🟡 GATE FLOOD (54.6% — ladder-consistent)
- gate: 2m-EMA_STACK_V2-SML | **±5s: 125/95/9 → 54.6%** | pairs 93/94/100%
**Judgment:** EMA-stack ordering persists more than crosses, less than SuperTrend —
sits between SWING (63%) and UT_BOT (42%). Ladder intact. Nothing new.

### sid 282 — PACKTEST · EMA Stack v2 · trigger  ✅ NEW FLEET RECORD (96.5% / 98.8% ±10s)
- no gate | coverage ✓ | **±5s: 82/1/2 → 96.5%** | ONE phantom all day; exits 21/21 + stops 20/20 perfect
**Judgment:** New record. Gated twin: 54.6%. Eighth family, same story.

### sid 281 — PACKTEST · EMA PP v4 · gate  🔴 GATE FLOOD (37.0% — ladder-consistent)
- gate: 2m-EMA_PP_V4-PSML | **±5s: 81/135/3 → 37.0%** | pairs 95/97/100%
**Judgment:** 4-way price/EMA ordering state (PSML) flips often → between HIST (30%)
and UT_BOT (42%). Ladder intact. Nothing new.

### sid 280 — PACKTEST · EMA PP v4 · trigger  ✅ HEALTHY (95.4% on n=214)
- no gate | **±5s: 412/4/16 → 95.4%** | ±10s 97.2% | 4 phantoms all day
**Judgment:** Third strategy above the 95% target. Gated twin: 37%. Ninth family, same story.

### sid 279 — PACKTEST · EMA PP v3 · gate  🔴 GATE FLOOD (37.0% — identical twin of 281)
- gate: 2m-EMA_PP_V3-PSML | **±5s: 81/135/3 → 37.0%** — numbers identical to v4 gate (281)
**Judgment:** v3/v4 gates produce byte-identical results (same PSML state on same data)
— good internal consistency check of the measurement itself. Nothing new.

### sid 278 — PACKTEST · EMA PP v3 · trigger  ✅ HEALTHY (95.4% — identical twin of 280)
**Judgment:** Byte-identical to v4 trigger (280). Consistency check passes again.

### sid 277 — PACKTEST · Bollinger Bands · gate  🔴 GATE FLOOD (35.4% — ladder-consistent)
- gate: 2m-BB-SQUEEZE_UPPER | **±5s: 70/124/4 → 35.4%** | pairs 92/97/100%
**Judgment:** Squeeze-band state flips often → sits with HIST/PP gates (30-37%).
Ladder intact across all 10 gated strategies now. Nothing new.

### sid 276 — PACKTEST · Bollinger · trigger  ✅ HEALTHY (89.9% / 95.5% ±10s) — 10th family, same story.

### sid 275 — TEST-P2-stoch-10s-TSLA · 10m gate  🔴 GATE FLOOD (45.1%) — cross-SYMBOL confirmation:
TSLA + a 10m gate floods identically (pairs 92/96/100%). B5 is symbol- and TF-general.

### sid 274 — TEST-P2-multipack-10s-TSLA  ⚫ KNOWN DEAD-LIVE (0 alerts vs 55 BT trades)
**Judgment:** This is the pre-existing roadmap carry-over ("sid 274 genuinely silent
live — investigate"). Still fully dead on the live side — zero alerts all day. NOT new
tonight, but tonight confirms it persists post-all-fixes. Suspect its 5m gate state
(M<S+) never passes live or its monitor isn't evaluating. Keep on roadmap, low priority
(test strategy).

### sid 273 — TEST-P2-utbotv4-10s-TSLA · trigger  ✅ HEALTHY (91.2%)
**Judgment:** Ungated TSLA at 91% — the healthy band holds cross-symbol. Nothing new.

### sid 272 — TEST-P2-stoch-10s-SPY · 10m gate  🔴 GATE FLOOD (40.5%) — signal exits 100%.
### sid 271 — TEST-P2-multipack-10s-SPY · 5m gate  🔴 GATE FLOOD (9.0%) — rare-state 5m gate (M<S+),
8 BT trades vs 65 live round-trips. NOTE: same config family as dead-live 274 (TSLA) but
SPY version DOES alert — 274's deadness is symbol/instance-specific, not config-wide.
### sid 270 — TEST-P2-utbotv4-10s-SPY · trigger  ✅ HEALTHY (92.3% / 94.7% ±10s).

### sid 269 — SPY-CANARY-1m-Control  🟠 1MIN EARLY-FIRE PATTERN (21.4%, small n) — possible B4 evidence
- SPY 1Min | no gate | **±5s: 15/36/19 → 21.4%** | ±10s identical
- entries 62% with 5 EARLY · signal exits 50% with 6 EARLY — early-dominant, unlike any 10s strategy
**Judgment:** First 1Min strategy, and the first EARLY-dominant profile. On 1Min TF a
grace-fire on the forming bucket can be up to 60s early vs bar close (invisible at 10s
where max-early=10s ≈ tolerance) — this is the predicted B4 signature surfacing exactly
where it should. Small n (17 BT trades); check other 1Min strategies tonight before
concluding. If 1Min strategies are systematically early-heavy → B4 (forming-bucket
C-type evaluation) gets promoted to a real fix target.

### sid 268 — SPY-CANARY-10s-NoConf  ✅ HEALTHY (92.3% / 94.7%).
### sid 267 — TSLA-CANARY-10s · SWING-NEUTRAL gate  🟡 GATE FLOOD (61.9%) — matches the
SPY SWING-NEUTRAL ladder slot (63%) cross-symbol. Mechanism is symbol-independent.
### sid 266 — TSLA-CANARY-5m-Control  ⚪ inconclusive (n=4; pairs 100%).

### sid 265 — TSLA-CANARY-1m-Control  🔴 B4 CONFIRMED (17.2% — systematic EARLY firing at 1Min)
- entries med **−30s**, 8/16 early, ZERO late · signal exits med **−60s**, 9/16 early, ZERO late
**Judgment + B4 PROMOTION:** Second independent 1Min strategy (after SPY 269) with the
early-only profile — medians −30s/−60s are exactly the forming-bucket grace-fire
magnitudes (fire at bar_start+grace instead of bar CLOSE: invisible at 10s TF where
max-early≈10s, a −30..−60s systematic error at 1Min). **B4 (grace-fire C-type
evaluation on the forming bucket) is now a CONFIRMED bug for 1Min+ strategies** — but
note it's also a DESIGN tension: Kevin's sub-5s latency goal vs C-type bar-close
semantics. At 10s TF the current behavior approximates both; at 1Min+ they conflict.
DECISION FOR KEVIN: should 1Min+ C-type triggers wait for bar close (parity-correct,
up to ~60s "slower" than today's behavior) or keep grace-fires (fast, structurally
early vs backtest)? The backtest could alternatively model grace-fires — but that
reintroduces look-ahead. Recommend: wait-for-close on 1Min+ C-type (it matches the
documented C-type contract; today's early fires are arguably wrong fills).

### sid 263 — TSLA-CANARY-10s-NoConf  ✅ HEALTHY (91.2%).
### sid 194 — TSLL 1Min Mass #30 · dual gate  ⚪ tiny n (1 BT trade / 5 alert pairs) — 1Min + dual
gate + B4 early-fire + gate flood compound; too small to score. Production 1Min+gated
strategies inherit BOTH confirmed bugs.

### sid 174 — TSLA 1Min Mass #2 · DUAL gate  🔴 EXTREME GATE FLOOD (0% — not truncation)
- Recheck: lane rebuilt with history through 06-10; BT legitimately took ZERO trades
  on 06-11 (dual gate: SWING BULL_C2 AND SR BELOW_SUPPORT both required) while live
  fired 52 alert-pairs. Dual rare gates = flood probability compounds. Same B5
  mechanism at its theoretical maximum; 1Min B4 early-fire stacks on top.

### sid 136 — SPY 1Min Mass #11 · dual gate  🔴 B4+B5 COMPOUND (8.6%, small n)
- 1Min + dual gate: early-fire medians (−60s/−120s) + flood — both confirmed bugs stacked.
Production 1Min Mass strategies = the most-affected class; should benefit most from
tomorrow's fixes.

---
# ☀️ MORNING SUMMARY (compiled ~08:15Z)

## Fleet table (RTH 06-11 14:45–20:00Z, ±5s combined%)

| band | strategies | range |
|---|---|---|
| ✅ Ungated, price-based (10s) | 296: **96.6** · 282: **96.5** · 280/278: **95.4** · 284: **95.2** · 288: **94.3** · 286: 92.4 · 270/268: 92.3 · 302: 92.0 · 273/263: 91.2 · 300: 90.9 · 276: 89.9 · 294: 82.2 · 298: 68.8(n=20) | **~90–97%** |
| 🟡 Gated (B5 flood, by gate flip-rate) | 299: 73.2 · 301: 63.0 · 267: 61.9 · 283: 54.6 · 293: 45.6 · 275: 45.1 · 287: 42.6 · 303: 41.7 · 272: 40.5 · 281/279: 37.0 · 277: 35.4 · 285: 30.4 | 30–73% |
| 🔴 Rare-state / dual gates | 295: 17.0 · 297: 11.0 · 271: 9.0 · 136: 8.6 · 289: 5.9 · 291: 1.0 · 174: 0.0 | 0–17% |
| 🔴 Laggard packs (any config) | VWAP: 304: 22.2 / 305: 18.1 · SR Channels: 292: 13.5 | known pre-existing |
| 🟠 1Min (B4 early-fire) | 269: 21.4 · 265: 17.2 (medians −30/−60s, early-ONLY) | confirmed bug |
| ⚪ Special | 307(KO): 77.0 · 306(DIA): 54.3 (canary-birth) · 266: n=4 · 194: n=1 · 274: dead-live (known) | — |

## Ranked residual bugs (after tonight's fixes)

1. **B5 gate-state flood — FIXED overnight (d6a9be8, deployed 08:01Z).** Stale
   `_mtf_confluence` after REST corrections; reproduced via unit test; dose-response
   evidence across 17 gated strategies. **VERDICT = TODAY'S RTH:** gated phantom
   ratios should collapse toward the ungated band. If residual remains → Gate Parity
   Analysis card decomposition + the gate-state telemetry instrument (build today).
2. **Volume integrity (≥60s fan-out double-count)** — localized precisely (RVOL
   trigger 87% vs RVOL gate 1%; VWAP cumulative-anchor sensitivity). Fix: skip_volume
   in fan-out accept_second_bar + volume in REST corrections. Today.
3. **B4 1Min+ early-fire** — confirmed, quantified (−30/−60s medians). DESIGN DECISION
   for Kevin: wait-for-close on 1Min+ C-type (recommended) vs keep grace-fires.
4. **Bulk-UAD silent truncation** (302, one-off so far) — coverage assertion in UAD.
5. **Pack cleanup:** VWAP v2 + SR Channels incremental parity (pre-existing Phase 4).
6. Minor: thin-tape stop-fill late-lean (≤10s); sid 274 dead-live; after-hours sweep
   lookback widening.

## Overnight ops log
- Fleet UAD: 47 strategies, 00:11→07:44Z, one truncation (302, re-healed 08:0xZ post-fix).
- B5 fix: reproduced → fixed → 44 tests green → deployed 08:01Z pre-market.
- All lanes now rebuilt on post-HiFi-fix code with clean snapshot lineages.
- TODAY'S PLAN: (1) watch gated cohort live vs ungated post-open — B5 verdict;
  (2) volume fix (small, test like gap-healer); (3) Kevin decides B4 policy;
  (4) gate-state telemetry + lens Live mode; (5) backtest-reference toggle.

### sid 302 — RE-HEALED ✅ (08:16Z): lane covers through 08:01Z, numbers BYTE-IDENTICAL to the
gold standard (92.0% / 94.3%, paired=229/7/13). Truncation fully recovered; remains a
one-off. All 47 fleet lanes now verified-fresh on post-fix code.

---
# 🎯 B5 VERDICT TRACKING (today's RTH, gate fix live since 08:01Z)

### 307 (KO · gated) — TODAY 13:45–15:05Z: **91.7%** (was 77.0% yesterday)
- 100% on ALL categories (11E/7sig/4stop) · 2 phantoms in 80 min · stop late-lean GONE
- First gated strategy reading in the ungated band. n=24 events — direction unmistakable.

### 306 (DIA · gated) — TODAY: **84.8%** (was 54.3% yesterday; canary-birth artifacts gone)
- 100% deltas everywhere · 3 phantoms / 2 missed in 80 min on a thin gated symbol

### 305/304 (VWAP pair) — TODAY: tiny n (quiet +2σ morning); 304 trigger 62.5%/85.7% ±10s (n=8).
VWAP verdict belongs to the VOLUME fix (this afternoon), not the gate fix — don't read these yet.

---
# 🔧 TRUE B5 ROOT CAUSE — FIXED & DEPLOYED 15:35Z (711e72b)

303's verdict measurement exposed it: 27.7% today (canaries 84-92%) → probe showed
21/29 live entries fired with the batch PB gate CLOSED → GATE_DIAG records carried
ALL FOUR 2M-UT_BOT states simultaneously → **own_records pollution loop**: monitors
publish their WHOLE merged confluence set (own states + every other TF's records)
into _mtf_confluence when their primary TF serves as a gate TF; round-trips
accumulate until the buffer holds all states of everything → subset checks always
pass → gates permanently open on multi-strategy hubs (SPY). Lone-canary hubs (KO/DIA)
have pure shadows → clean → why they improved and SPY didn't. Fix: own-TF-prefix
filter at all 3 publish sites. The d6a9be8 correction-staleness fix stands but was
minor. **A/B verdict: window A 13:45–15:30Z (pre-fix) vs window B 15:45Z→close
(post-fix), same day, same market.** Re-append job 1014679494 extends lanes.

### ⚡ EARLY TELL (16:05Z, 25 min post-fix): gated alert RATES collapsed at the deploy
Pre-fix (13:45-15:30Z) vs post-fix first 15 min, ENTRY alerts: 277: 46→0 · 279: 28→0 ·
267: 40→1 · 272: 33→4. Gates are CLOSING for the first time. (Post-fix pairing reads
0% right now purely because appended lanes lag 15 min — no BT reference yet; formal
A/B at 17:10Z after the next append.) Re-append progressing fast: 266-280 done by 16:02.

### 🎯 B5 A/B INTERIM (17:12Z measurement)
- **303: PRE 28% (23/58/2) → POST 75% (6/0/2) — ZERO PHANTOMS post-fix** on the worst offender
- 307: 86→82% steady · 306: 50% (n=4) · 297: 0/4/2 tiny-n (watch — flippiest gate)
- Controls untouched: 302 95% · 296 90% · 284 100%
- Most cohort slots ended pre-15:45 (no post-fix coverage yet) → append round 3 fired
  (1020424457); full verdict at ~18:00Z with 2h+ post-fix windows.

### 🏆 B5 FINAL A/B (18:08Z): GATED 40.3% → 69.0%, phantoms 577 → 44 (~10x rate drop)
285: 16→100% · 277: 38→100% · 287: 60→91% · 303: 28→89% (0 phantoms) · 267: 54→85% ·
283: 82→78% · 281/279: 68→57% (n small). RESIDUAL WATCH (EOD): 297 INSIDE 0% (n=8),
272 10m-gate 21% (n=19), 307 over-close? (7 missed). CONTROL DIP in the same later
window (302: 95→72) = time-correlated NOT fix-correlated (ungated paths untouched) —
suspect append-edge/regime artifacts 16:30-17:10Z; re-measure on settled EOD data.
VERDICT: own-records pollution loop was THE gate bug. Next: volume fan-out fix.

# 🔧 VOLUME-INTEGRITY FIX DEPLOYED 18:11Z (48e07ee)
Sub-bar volume dedup (AM+ws_agg same-minute double-feed into >=120s builders) +
skip_volume on the per-second chart-visual path. RVOL/VWAP A/B: pre-fix windows today
vs post-18:15Z. Watch sids 291 (RVOL gate, was 1.0%), 305/304 (VWAP), 290 (control,
RVOL trigger — should stay ~87%). Append round 4 running (1024021332). Verdict at EOD.

### 📐 18:55Z INTERIM + NEW FINDING: B2 PROMOTED (append-edge fossilization)
Extended-window reads softened FLEET-WIDE incl. CONTROLS (302/296/284: 95-100% →
74-78%) — controls touch neither fix, so the afternoon pollution is measurement-side:
each same-day append round writes unsettled-REST edge trades that are NEVER rewritten
(insert-only dedup) → 4 rounds today = 4 fossilized edge bands. Gate-fix verdict
STANDS (pre-fix baselines far below current reads; 17:10 clean snapshot 89-100%).
297 = genuine residual (0%, n=14, consistent). Volume cohort n=0 yet.
FIXES QUEUED: B2 — append should re-verify/replace its last edge band each run;
tonight's full UAD washes today's fossils; tomorrow = single-append cadence clean day.

---
# 🌆 END-OF-DAY WRAP (Friday 2026-06-12, written 19:50Z)

## What happened today, in order
1. **09:35 MT — B5 TRUE fix deployed** (own-records pollution): gated cohort 40.3%→69.0%
   on clean windows; 285/277 → 100%, 287 → 91%, 303 → 89% w/ zero phantoms. VERIFIED.
2. **12:11 MT — volume dedup + skip_volume** (defense-in-depth; the 2x theory was wrong
   in detail — kept as guard).
3. Mechanical audit found the REAL ≥120s corruption: **single-minute clobbering of
   closed 2m bars** (volume 0.50x exactly, ranges collapsed) — explains the post-gate-fix
   residuals (285's 11:04/11:07 phantoms — Kevin-spotted, verified genuine — plus 297, 272).
4. **13:03 MT — clobber fix deployed… and silently stopped ≥120s closes in prod**
   (passed 40 unit tests; production seeded-history/dual-feed path differs). Caught via
   the cache audit within ~40 min; **REVERTED 13:50 MT** (92c9528). Gates back on
   clobbered-but-live 2m bars — known-bad beats frozen.
5. **TBD classification shipped** (Kevin's design) in by-hour: uncovered alerts no longer
   count as phantoms; auto-convert on append. Kills the afternoon sag artifact.

## Contaminated windows today (for any retro analysis)
- 19:03–19:50Z: frozen ≥120s gate buffers (clobber-fix deploy → revert).
- Append-edge fossil bands at ~15:39/16:53/17:14/18:25Z (B2).
- Pre-09:35 MT: gate-flood era. 09:35–13:03 MT: cleanest gate-fix windows.

## WEEKEND QUEUE (markets closed Sat/Sun — all offline-buildable, Monday verifies)
1. **Clobber re-fix with faithful repro** (seeded builder + dual AM/ws_agg feed sim —
   find why prod ≥120s closes stopped; unit tests passed, prod didn't). TOP PRIORITY.
2. **Gate-state telemetry + lens Live mode** — would have caught today's freeze in
   minutes; the "show me what the engine sees" instrument.
3. B4 wait-for-close on 1Min+ C-type (approved).
4. B2 append edge-band re-verify.
5. by-deploy TBD (same change as by-hour).
6. Cache cleanup: delete corrupted ≥120s ws/ws_agg rows (lens falls back to clean 10s
   resample) — fixes historical red ribbon rows on 2m lines.
7. Pack parity (VWAP/SR) via parity simulator.

## The week's scoreboard
Six confirmed root causes found; five fixed-and-verified (WS gaps, HiFi rewriter,
snapshot lineage, gate corrections-staleness, gate own-records pollution); one
characterized with fix pending re-land (2m clobber). Ungated price strategies:
90–97% at ±5s. Gated strategies: flood eliminated, residuals mapped to the clobber.
Measurement integrity: TBD class + coverage clipping + benign-pattern taxonomy.

### 🔬 LIVE BISECTION RESULT (20:19–20:30Z): freeze is DETERMINISTIC under the guard
Re-landed 247afec's guard under telemetry: 1Min rows flowed (9), 2m rows = ZERO for
10+ min → reverted again (36dc2df). FACTS BANKED: (a) freeze reproduces 2-for-2 under
the guard, recovers 2-for-2 on revert; (b) ≥120s-specific (1Min healthy — not warmup,
not feed); (c) the faithful fan-out-level repro PASSES with the guard — the freezing
ingredient lives in the CALLER layer (on_polygon_bar / on_second_bar / seed overlap),
not in accept_second_bar itself; (d) the clobber repro stands as the re-land gate
(reproduces 0.50x exactly; clobber was DESIGNED double-count "healing", obsoleted by
the dedup). WEEKEND: caller-level repro (drive on_polygon_bar + on_second_bar with
realistic per-second + AM streams over a seeded hub). ALSO FOUND: telemetry writes
silently rejected by `bar_diagnostics_source_check` (table CHECK allows only
live/live_corrected/backtest/algo) — needs Kevin to run one SQL line.

### 20:55Z — Gate telemetry LIVE (constraint fixed by Kevin, validated)

Kevin extended `bar_diagnostics_source_check` to allow `live_gate` at ~20:48Z.
Validation (direct DB read):

- **Rows flowing**: every SPY-cohort sid (277–303) writing one `live_gate` row per
  2m bar (20:48, 20:50, 20:52), written 1–37s after bar close by the flush loop.
- **Content is healthy**: exactly ONE state per interpreter (15 records for the
  15-interpreter 2m group) — no accumulation, no cross-TF pollution. This is the
  B5 fix visible in ground truth: pre-fix this set would have grown unboundedly.
- **States actually change bar-to-bar** (proves it's live, not fossilized):
  RSI_ZONES NEUTRAL_BULLISH→NEUTRAL_BEARISH at 20:50, RVOL MINIMAL→HIGH at 20:52,
  STRAT_ASSISTANT TWO_UP→TWO_DOWN at 20:50.
- All sids share identical record sets per bar — expected (same SPY 2m shadow engine).

**What this unlocks**: Monday RTH we have a per-bar record of what the live gates
actually saw, queryable against the backtest lens — phantom/missed diagnosis goes
from inference to direct comparison. Weekend item: Alert-lens "Live mode" frontend
reads these rows.
