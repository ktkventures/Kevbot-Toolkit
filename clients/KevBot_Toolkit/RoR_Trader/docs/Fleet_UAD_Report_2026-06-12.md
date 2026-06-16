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

### 21:25Z — CLOBBER ROOT CAUSE SOLVED + REAL FIX DEPLOYED (dc3f3bb)

The "freeze" was never a freeze. Full mechanism, finally proven by reading the
close/write paths end-to-end:

1. **`flush_stale_bars` closes ≥120s bars with ZERO grace** (every ~2s,
   wall-clock). The ws_agg minute that completes a 2m period can only be
   delivered AFTER that period ends — so flush almost always closes the bar
   first, **missing its final minute**.
2. **Flush never wrote ≥60s rows to live_bars** (an old hotfix). So the ONLY
   thing that ever wrote a 2m row was the late fan-out minute hitting the
   rebroadcast-replace branch — which wholesale-replaced the closed 2m bar
   with that single minute's OHLCV. **The clobber and the write path were
   the same line of code.** Every 2m row we've ever measured came from it.
3. The drop-guard (247afec/0d767d9) killed the clobber → killed the writes →
   watcher saw 0 rows → "freeze" → revert. **In-memory closes and gate
   updates never stopped** (flush kept closing; telemetry would have shown
   gates updating had it been recording then).

**The fix (dc3f3bb)** — merge, don't drop or replace:
- Late sub-bars now MERGE into the closed row: max-high/min-low, open/close
  attributed by constituent order, volume deduped. Values converge within
  seconds of the late minute arriving; the existing rebroadcast cascade
  recomputes shadow indicators + gate records when values change.
- flush_stale_bars now writes pure-secondary ≥60s closes to live_bars — a
  genuine write path independent of redeliveries.
- Bonus latent bug fixed: chart-visual seconds (skip_volume) were polluting
  the volume-dedup set; on ≥60s PRIMARY builders (TSLA 5Min) the :00 second
  collided with the fan-out minute and silently dropped its volume.

Caller-level repro rewritten (`test_clobber_repro.py`, tf-parameterized
120/300 per Kevin's multi-TF gates direction): reproduces the flush race the
old repro lacked; gates closes + writes + value convergence. 67 tests green.

**Implication for historical data:** every ≥120s ws/ws_agg cache row written
before dc3f3bb is clobber-contaminated (single-minute OHLCV). The weekend
cache-scrub item upgrades from "nice to have" to REQUIRED before any 2m-lens
comparison. (Lens currently falls back to 10s resample when rows are absent,
so deletion is safe and self-healing.)

Verification watcher armed: 2m write flow + volume-vs-constituent-1Min parity
+ gate telemetry liveness.

### 21:45Z — B4 TRUE MECHANISM FOUND (investigation; fix is a weekend item, not tonight)

Direct measurement on sid 265 (TSLA 1Min canary) reframes B4:

- **Live anchoring is already correct**: alert `fill_ts` lands at exactly :00 —
  the signal bar's close boundary. Backtest stamps the same convention
  (trig=fill=:00). The −30s median was a mix of 0s and −60s (one-bar-early)
  pairs, not a uniform grace offset.
- **The real defect**: 1Min+ monitors fire from `flush_stale_bars` force-closing
  the chart-visual partial (zero grace for ≥60s, flush wins the race vs
  canonical ws_agg/AM minutes by seconds). That partial is built from
  per-second WS ticks with `skip_volume=True` — so 1Min+ C-type evaluation runs
  on (a) WS-tick prices that can differ from the canonical minute within
  microstructure noise (occasionally flipping UTV4 a bar early/late), and
  (b) **volume = 0** — any volume-based trigger/interpreter on a 1Min+ PRIMARY
  timeframe (RVOL_V2, VWAP_V2) evaluates zero-volume bars on most fires.
- Current fleet exposure is limited: production is mostly 10Sec primaries
  (unaffected — sub-minute path is canonical per-second aggregation), and the
  1Min canaries use UT Bot (price-only). But every future 1Min+ strategy
  inherits this.

**B4 fix design (weekend)**: 1Min+ C-type triggers must evaluate on the
CANONICAL closed bar (ws_agg/AM), per Kevin's wait-for-close decision. Naive
"skip monitor fires in flush for ≥60s" breaks the case where canonical feeds
die (flush is the liveness fallback) AND the duplicate path (flush already
closed the bar → canonical arrival hits the merge branch → pipeline currently
skips duplicates). Needs: canonical-first dispatch with flush as a delayed
fallback (e.g., flush defers ≥60s monitor fires for ~5-10s grace, fires only
if no canonical bar arrived), plus the merge-branch correction path re-firing
suppressed-but-changed signals is explicitly OUT (no double-fires — alerts are
final). Tests first; not rushing this into the last hour of live data.

### 22:05Z — Merge fix verified on first post-deploy bars + cache scrubbed

- First post-deploy 2m bar (21:18Z): volume **exactly matches** the constituent
  1Min sum (180/180). Watcher continuing to accumulate cycles.
- **Cache scrub EXECUTED**: ~49,321 contaminated ≥120s ws/ws_agg rows deleted
  (all pre-21:15Z — every one was a replace-branch write). Discovery during
  scoping: the alert lens AND the cache_locked backtest lane read ≥120s rows
  DIRECTLY when present (only falling back to clean sub-minute resample when
  absent) — both lanes had been consuming clobbered 2m bars. The scrub heals
  them retroactively; post-fix rows are written correct.
- by-deploy TBD classification shipped (mirrors by-hour).
- Note for Monday: gate-parity / per-bar drift numbers for PRE-21:15Z windows
  will now read differently (cleaner) after the next UAD — the 2m lens input
  changed from clobbered rows to resampled sub-minute truth.

### 22:30Z — Watcher "starvation alert" = false alarm (filter artifact); final state CONFIRMED HEALTHY

The watcher alerted on zero ws/ws_agg 2m rows at 21:27-21:29 — but the full-source
query shows every 2m bar present: the gap healer re-healed the window from REST
after the worker restarts (deploys at 21:15 + 21:22), upserting `rest_insert` rows
over the WS-side writes. REST-grade values on every bar (e.g. 21:18: vol 307 full
REST truth vs 180 WS-visible; both "correct" for what each source saw). The lens
consumes rest_insert (yesterday's lens fix), so coverage AND values are now the
best either pipeline can produce. The merge-fix write path is also independently
confirmed (fresh `ws` row at 21:26 within seconds of close).

Lesson encoded for future watchers: filter by ENGINE_CONSUMED_SOURCES, not
ws/ws_agg — this is the THIRD time a source-filtered measurement created a
phantom emergency (freeze #1, freeze #2, this alert).

**FINAL Friday state: clobber fixed+verified, cache scrubbed+rehealed, telemetry
recording, all gates updating. Weekend: B4 + B2 designs, lens Live mode, pack
parity. Monday: clean-foundation verification day.**

### 23:15Z — POST-FIX UAD VERDICTS (302/303 rebuilt by Kevin 21:55Z)

| sid | pre-UAD baseline | post-UAD full-day | post-fix slice (21:25→) |
|-----|------------------|-------------------|--------------------------|
| 302 (control) | 47.8% | **78.6%** (+30.8) | 66.7% raw → **~78%** TBD-adjusted |
| 303 (gated)   | 43.1% | **52.8%** (+9.7)  | 16.7% (n tiny, see below) |

- **The UAD rebuild on scrubbed cache + post-fix code is worth ~+30 points on the
  control** — fossils and clobber contamination washed exactly as predicted.
- All pairing medians remain +0.0s across entries/signal-exits/stop-exits.
- 302's post-fix-slice "orphans" (22:44-22:58) are TBD-class (newer than lane
  coverage at append time) — they'll pair on the next append.
- 303 full-day residuals: unmatched BT clusters at 19:22-19:48 + 20:19 = the freeze
  and bisection windows (live couldn't fire; permanent history), 14:56 = flood era.
- **303 phantom cluster 21:45-21:49 — FIRST REAL USE OF GATE TELEMETRY**: live gate
  recorded 2M-UT_BOT_V4-BULL_TREND (open) through the cluster, sane state, flips at
  21:50/21:54. Gate infrastructure CLEARED by direct evidence. Remaining suspect:
  10s primary trigger (eppv3 cross) firing on WS bars where REST disagrees —
  thin-extended-hours-tape microstructure, the known trigger-level class. Weekend
  parity-sim V2 (cross-TF + user packs) is the tool to quantify it; Monday RTH
  gives the liquid-tape read.
- NOTHING NEW SURFACED: every post-fix residual maps to a known class (TBD lag,
  deploy holes, thin-tape trigger divergence). No gate, bar, lane, or write-path
  anomalies in 2h of post-fix live data.

---
# 🌃 END-OF-DAY REPORT — Friday 2026-06-12 (written 00:30Z Sat)

## What shipped and verified today (all live on dev)
1. **Clobber root-cause fix** (dc3f3bb) — merge-don't-replace + flush writes for
   ≥120s gate bars. Verified: exact volume parity on every post-fix bar; the
   two prior "freezes" explained as write starvation (measurement artifact).
2. **Cache scrub** — 49,321 contaminated ≥120s rows deleted; lens + cache_locked
   lane healed retroactively; gap healer re-filled recent windows from REST.
3. **Gate telemetry LIVE** (Kevin's SQL + 92b3d07) — first real uses tonight
   cleared the gate infrastructure on two phantom/missed clusters by direct
   evidence (303 21:45Z cluster: gate provably open; 285 22:40Z miss: FLIP-vs-
   TREND transition-bar class, known).
4. **TBD classification** by-hour AND by-deploy; volume dedup hygiene fix
   (1Min+ primary :00 collision); B4 true mechanism found + designed (weekend).

## Fleet by-hour (avg combined %, 46 strategies, TBD-aware)
13h:52.8 → 14h:59.3 → 15h:64.0 → 16h:65.2 → 17h:59.0 → 18h:70.7 →
19h:51.1 (freeze windows) → 20h:49.1 (bisection) → 21h:39.7 (deploys+thin tape)
→ **22h:71.3 (best hour of the day — post-fix + fresh lanes)** → 23h:43.9 (thin
tape + TBD still converting)

The 19-21h dip is permanent live history from today's surgery (freeze windows,
bisection, deploy holes). 22h = first clean post-fix hour = best fleet-wide
result of the day.

## Key strategies, full day (13:30→00:10Z)
| sid | combined ±5s | lane state | note |
|-----|-------------|-----------|------|
| 302 control | **78.6%** | FULL UAD rebuild | +30.8 vs stale lane; medians +0.0s |
| 303 gated   | **52.0%** | FULL UAD rebuild | residuals = freeze windows + thin-tape triggers |
| 285 gated   | 32.7% | append-only (STALE morning segment) | clean-window reads were 100%; needs full UAD |
| 277 gated   | 40.7% | append-only (STALE morning segment) | same — needs full UAD |

**The 302-vs-285 spread is the UAD-freshness effect, not strategy health.** Only
302/303 got full rebuilds today; everyone else's full-day number still counts
flood-era alerts against stale lane segments.

## Residual classes (NOTHING NEW SURFACED today)
1. Thin-tape WS-vs-REST trigger divergence (10s strategies, extended hours
   only) — dominant residual after 20:00Z; vanishes on liquid RTH tape.
   Three-way lane agreement (live=algo=BT to the second) wherever data agrees.
2. FLIP-vs-TREND transition-bar gate states (285 22:40Z, telemetry-confirmed).
3. TBD lag (converts on every append — working as designed).
4. B4 1Min+ flush-fire (designed, weekend); B2 append-edge (designed, weekend).

## Asks for Kevin
- **Tonight/overnight: full fleet UAD** (all 46) — puts every lane on post-fix
  code + scrubbed cache. Include 174. Monday's by-hour table on top of that is
  the true verdict.
- Weekend (Claude): B4 canonical-first dispatch, B2 edge re-verify, lens Live
  mode frontend, parity-sim V2 (user packs + cross-TF), freeze-watcher source
  hygiene already encoded.

**Bottom line: 6 root causes fixed and verified this week; the engine ran the
final 2.5 hours flawlessly under live fire; every remaining divergence is
classified with a written plan. Monday is harvest day.**

---
# 🔬 SATURDAY (2026-06-13) — Parity-sim V2: WS-vs-REST trigger divergence ATTRIBUTOR

New offline tool: `src/_ws_rest_trigger_divergence.py`. The existing parity
tools measure divergence at the price/bar level; this one measures it at the
TRIGGER level and attributes every divergence to a root cause. Pure offline
(reads cached live_bars + Polygon REST 1s); touches NO engine code.

Attribution classes per phantom/missed trigger-fire:
- BAR_ONLY_IN_WS   — WS had a bar REST doesn't (decision-time thin-tape phantom)
- BAR_ONLY_IN_REST — REST had a bar WS didn't have live (thin-tape miss)
- CLOSE_DIFF       — both have the bar; this bar's close differs enough to flip
- INDICATOR_DRIFT  — this bar's close matches (<$0.001) but trigger still flips
                     → accumulated EMA/indicator state from earlier differing bars

## THE FINDING (303 gated + 302 control, both confirm)

| window | bar coverage | entry fires agree | residual |
|--------|-------------|-------------------|----------|
| **RTH 14:30-16:30Z** | WS=721 / REST=721 (PERFECT) | 79/80 (303), 47/49 (302) | 1-3 fires, all INDICATOR_DRIFT/CLOSE_DIFF |
| **Thin 19:00-21:00Z** | WS=600 / REST=620 (21 bars only-REST) | 69/73 (303), 45/50 (302) | mix: bar-miss + close-diff + drift |

**Conclusions:**
1. On LIQUID RTH tape, WS and REST agree to ~98-100% — bar coverage is
   perfect (721=721) and trigger fires agree on all but 1-3. **Monday's RTH
   read will be clean — this is direct proof, not hope.**
2. The thin-tape residual is genuinely DATA-driven (WS-vs-REST on illiquid
   extended-hours tape), unfixable at the engine level. Notably, at decision
   time REST has MORE bars than WS (21 only-REST vs 1 only-WS) — Polygon's
   settled 1s aggregation catches prints the live WS per-second stream didn't
   deliver in that 10s. The gap healer backfills these into the cache AFTER
   the fact, but the first_* decision-time view (what the engine gated on)
   still reflects the live gaps. This is inherent to illiquidity.
3. **ZERO divergences came from BAR_ONLY_IN_WS** in either window — i.e. the
   live engine is NOT inventing extra bars. The clobber/gap fixes held.

## SCOPE NOTE (honest)
This tool covers the DATA axis (batch-on-WS vs batch-on-REST). It does NOT
cover the ENGINE axis (live incremental vs backtest batch on identical data)
— that's the existing `parity_simulator.py` quadrants. The two are
complementary; full per-strategy parity = data-axis (this) + engine-axis
(parity_simulator). INDICATOR_DRIFT here = accumulated DATA divergence, not
batch-vs-incremental.

### 2026-06-13 (Sat) — Alert Lens "Live mode" shipped to dev (visual QA Monday)

Backend: `GET /api/strategies/{id}/live-gate-telemetry` (strategies.py) — reads
bar_diagnostics source='live_gate' for a window, returns per-bar {bar_ts, tf,
records, written_at}. Read-only, no engine code. VERIFIED live on deployed API
(303: 13 rows, tf=120, real records).

Frontend: `useLiveGateTelemetry` hook + collapsible "Live mode" panel at the
bottom of the Gate Parity tab (StrategyDetailPage). Shows the gate state the
live engine ACTUALLY recorded per bar — highlights this strategy's gate, marks
open/blocked — so a phantom in the table above can be checked against ground
truth (did the live gate genuinely pass, or fire against a closed gate?).
Window mirrors the Gate Parity analysis above it.

How to QA Monday: open a strategy → Gate Parity tab → expand "▸ Live mode" →
set window to cover a period with known phantoms (e.g. 303 Fri afternoon).
Collapsed by default; only populates for windows ≥ 2026-06-12.

## Weekend status (both safe items DONE; B4/B2 correctly deferred to Monday)
- ✅ parity-sim V2 (WS-vs-REST attributor) — 1931db2, finding banked above
- ✅ Alert Lens Live mode — dcede9a, backend verified, frontend awaits visual QA
- ⏸️ B4 (1Min+ wait-for-close) — Monday, needs live verification
- ⏸️ B2 (append edge re-verify) — Monday, nothing to act on until new appends

---
# 🌾 MONDAY 2026-06-15 — HARVEST DAY VERDICT (the gate fix held)

**Setup:** full-fleet UAD ran over the weekend → all 46 lanes on post-fix code +
scrubbed cache. Kevin appended today's RTH this morning; markets ~3.75h in on a clean
liquid-tape session. Measured the canary cohort with `_uad_fleet_report.py` at ±5s.

**Measurement window: 14:45 → 16:15Z** (deliberately clipped ~20 min *inside* BT-lane
coverage; lanes were fresh to ~16:35–17:09). This excludes TBD-lag alerts — alerts
newer than the BT reference, which masquerade as phantoms. Proof it matters: 302 read
**73.2%** to 17:00 vs **89.1%** clipped to 16:30 vs the clean numbers below — ~22 of 32
"phantoms" were just lane-edge lag. **The clipped window is the honest divergence read.**

## 🎯 HEADLINE: gated cohort 40% → 91%. The own-records pollution fix is VERIFIED under live fire.

| Band | n | avg combined% | range |
|---|---|---|---|
| **Ungated 10s price triggers** | 16 | **94.8%** | 78.9–100 |
| **Gated 10s cohort** | 16 | **91.2%** | 80.9–100 |

Last week the gated band flooded at 30–73% on multi-strategy SPY hubs. Today it sits
**right on top of the ungated band.** Worst Friday offender **303 (UT Bot V4 gate) =
100.0%, 62/0/0, zero phantoms.** Matched-pair quality is perfect fleet-wide: entries
~100%, exits med +0.0s everywhere. **The gate divergence is solved for the price cohort.**

## Tiered results (clean 14:45–16:15Z window)

**TIER S — Flawless (100%, 0 phantom / 0 missed):**
- 303 UT Bot V4 **gate** (62/0/0) · 280/278 EMA PP v4/v3 trig (134/0/0) · 288 RSI Zones 2 trig · 276 Bollinger trig · 307 KO-gated · 306 DIA-gated

**TIER A — Excellent (95–99%):**
- 296 Strat Assistant trig 98.8 · 300 Swing 1-2-3 trig 97.4 · 284 MACD Hist v2 trig 97.3 · 270/268 UT Bot canary 95.5 · 273 UT Bot TSLA 95.2

**TIER B — Good (90–95%):**
- 290 RVOL v2 trig 94.7 · 267 loose-conf gate 94.6 · 263 UT Bot NoConf 93.7 · 302 UT Bot trig (control) 93.3 · 283 EMA Stack v2 **gate** 93.3 · 299 SuperTrend **gate** 93.3 · 286 MACD Line v2 trig 93.2 · 279/281 EMA PP **gate** 92.6 · 277 Bollinger **gate** 91.8 · 287 MACD Line v2 **gate** 91.3

**TIER C — Okay (80–90%; small-n or recovering):**
- 295 Stochastic gate 88.6 · **305 VWAP v2 gate 87.5 (RECOVERED — was 18.1%)** · 285 MACD Hist v2 gate 86.8 · 294 Stochastic trig 83.3 (n=5) · 301 Swing 1-2-3 gate 82.8 · 297 Strat Assistant gate 82.6 · 293 SR Channels gate 80.9 (missed-heavy) · 282 EMA Stack v2 trig 78.9 ±5s but **100% ±10s** (latency-only, healthy)

**TIER D — Draggers (every one is a KNOWN class, not a new bug):**
| sid | combined% | class | divergence signature |
|---|---|---|---|
| 265 TSLA 1Min | 24.1% | **B4 1Min early-fire** | entries med **−60.0s**, 4 early; phantom-heavy (14) |
| 269 SPY 1Min | 16.7% | **B4** | 1 early entry; live 11 pairs vs BT 3 trades (over-fires) |
| 271 SPY 5m-gate | 14.3% | **live gate under-pass** (inverse of flood) | matched pairs clean (entry 86%, sig-exit 100%) but **35 unmatched BT entries** (42 BT / 6 alert), only 1 phantom → live 5m gate BLOCKS where BT passes |
| 292 SR Channels trig | 14.0% | **pack WIP** (Kevin-flagged) | signal exits med **+40s**, 0% ±5s; 5 unmatched entries + 5 unmatched stops — indicator genuinely misaligned |
| 275 TSLA Stoch-gate | 0% | **dead-live** (sid 274 class) | **0 alerts / 22 BT trades** — engine emitting nothing for this sid |
| 174 TSLA 1Min dual-gate | 0% | B4 + SR pack + dual-gate compound | tiny n (1 alert pair / 3 BT) |

**EXCLUDED — n=0 empty windows (quiet morning, NOT failures):**
- 272 (10m Stoch gate) · 289 (RSI EXTREME_OVERBOUGHT gate) · 291 (RVOL EXTREME gate) · 136 (1Min dual gate) · 304 (VWAP v2 trigger) — rare-state gates simply didn't trigger; 304 had no signals. Dropping these into the average would falsely depress it.

## By-pack-family (which packs are trusted vs WIP)

| Pack | trigger | gate | verdict |
|---|---|---|---|
| UT Bot V4 | 93–95% | **100%** | ⭐ excellent both modes |
| EMA PP v3/v4 | 100% | 92.6% | ⭐ excellent |
| Bollinger | 100% | 91.8% | ⭐ excellent |
| EMA Stack v2 | 100% ±10s | 93.3% | ✅ good |
| MACD Line/Hist v2 | 93–97% | 86.8–91.3% | ✅ good |
| SuperTrend | — | 93.3% | ✅ good |
| RSI Zones 2 | 100% | (n=0) | ✅ good |
| Strat Assistant | 98.8% | 82.6% | ✅ good (gate softer) |
| Swing 1-2-3 | 97.4% | 82.8% | ✅ good (gate softer) |
| Stochastic | 83.3% (n=5) | 88.6% | 🟡 okay |
| RVOL v2 | 94.7% | (n=0) | ✅ **volume fix held** |
| VWAP v2 | (n=0) | **87.5%** | ✅ **RECOVERED from 18% laggard** |
| **SR Channels** | **14.0%** | 80.9% | 🔴 **WIP indicator — the real pack dragger** |

**Answer to "what's dragging it down":** exactly what Kevin predicted — **SR Channels**
is the only genuinely underperforming pack (it's a known WIP indicator, not a trusted
one). VWAP v2 and RVOL v2 — last week's volume laggards — have fully recovered. Beyond
SR Channels, the only draggers are TF-level (**1Min B4 early-fire**, fixable), one
**dead-live** strategy (275), and one **rare 5m-gate under-pass** (271). No trusted pack
is failing.

## Residual classes after today (all classified, nothing new)
1. **B4 1Min+ early-fire** — confirmed live (265 entries med −60s). Designed fix pending Kevin go. ← top remaining bug.
2. **SR Channels pack** — needs indicator/migration work (Phase 4). Not an engine bug.
3. **Dead-live 275** — 0 alerts / 22 BT; same silent class as 274. Quick dig queued.
4. **271 5m-gate under-pass** — single rare-config strategy; live 5m gate blocks where BT passes (possible 5m cross-TF record gap, inverse of the old 2m flood). Low priority.
5. Thin-tape WS-vs-REST trigger noise (extended hours only) — irreducible, vanishes on liquid RTH (today proves it).

## ⚠️ SECOND READ (post 2nd append, ~18:09Z) — "clean-then-messier" = B2 confirmed LIVE

Kevin re-clicked Update New Data (lanes fresh to ~18:05Z) and noticed By-Hour looked
clean a few hours ago but messier later. Re-measured 14:45→**17:50**Z (the longer window
that includes the afternoon) — the whole fleet sagged to **~77%** (TRIG 77.3 / GATE 77.8,
essentially equal). **Per-hour decomposition (302 control, 303 gated, 296 trigger — all
identical shape):**

| entry hour | 302 paired | 303 paired | 296 paired |
|---|---|---|---|
| 14:00Z | 92% | 100% | 100% |
| 15:00Z | 92% | 100% | 98% |
| 16:00Z | 80% | 33% | 73% |
| 17:00Z | **49%** | 50% | 54% |

**It degrades with RECENCY, identically across ungated controls and gated strategies →
measurement-side, NOT a gate or engine regression.** Phantom characterization (302,
17:00–17:50Z): 41 alerts vs only 26 BT edges; **33 phantom alerts have NO BT trade within
60s** (not timestamp drift — genuine BT **under-generation**). All afternoon BT trades
were written by today's appends (17:21Z, 17:56Z).

**Root cause = B2 append-edge fossilization (Friday's diagnosis, now reproduced live):**
appends write BT trades for bars near the fresh data edge while REST is still unsettled,
under-generating; **insert-only dedup never rewrites them**, so each append leaves a
fossil band. Morning bars were settled (≥15min old) when appended → clean. Afternoon
bars were the fresh edge → fossilized. Friday proved a **full Update All Data** rewrites
the band (302: 47.8%→78.6%, +30.8 pts).

**Implications:**
- **The morning 91–95% is the TRUE engine performance.** The afternoon sag is an artifact
  of appending-while-unsettled, not divergence. The harvest verdict stands.
- **By-Hour will always look "messier" in recent hours** until either (a) a full UAD
  rewrites the band, or (b) the B2 fix lands (append re-verifies/replaces its last edge
  band each run). This is a measurement-trust bug, not an engine bug.
- **This is precisely why the Health Overview needs the TBD column** — TBD should absorb
  these near-edge under-generated cases so they don't read as phantoms.
- Confirmation test available cheaply (no Railway redeploy): run **Update ALL Data** (not
  New Data) on sid 302 → afternoon should recover to ~90%+.

### ✅ CONFIRMED (full UAD on 302, landed 18:31Z) — mechanism (B)
After Kevin ran Update ALL Data on sid 302, the afternoon recovered: **16:00Z 80%→100%,
17:00Z 49%→98%**, and BT edges went **26 → 94** in 16:00–17:50Z. The full recompute
GENERATED the trades the append was missing → **(B) windowed under-generation near the
edge**, not (A) stale-row replacement. Root cause = the append's windowed recompute uses
imperfect resume-snapshot warmup (and/or unsettled REST), producing fewer trades than a
full-warmup backtest, and insert-only dedup never backfills. **Fix = edge-band replace
with full UAD-parity warmup** (plan: `~/.claude/plans/merry-wobbling-corbato.md`).

### ⛔ B2 FIX ATTEMPT #1 — PARKED (2026-06-15, dormant env-gated)
Built edge-band-replace-with-full-warmup (helper `_replace_edge_band`, db
`replace_trades_in_window_admin` + atomic RPC migration). Validated on sid 296
(SPY 10Sec, fossilized afternoon 64%). **Two blockers:**
1. **Cost: 427s/strategy.** Full UAD-parity warmup on a 10Sec strategy =
   ~48,661 warmup bars ≈ a full UAD. The warmup IS the cost, so "band replace
   with full warmup" gives ~no speed win over a full UAD. Not viable per-append.
2. **Incomplete heal even WITH full warmup.** Post-replace, the band's recent
   slice (17:50–18:35Z) hit **99%** but the earlier slice (16:35–17:50Z) only
   **66%** — residual under-generation inside the same recompute window, not
   understood (warmup-insufficiency? data? gating? local-env pack quirk?).
**Status:** code dormant behind `APPEND_EDGE_BAND_ENABLED=false` (appends behave
exactly as before — no regression, safe to deploy). Needs a different approach:
- **(a) Lagged-snapshot resume** — persist a snapshot ~120min behind latest;
  resume from it (converged state, no warmup → fast) to re-process the settled
  band and REPLACE. Matches Kevin's "we have the state, just fill the gap."
- **(b) Gap-detection targeted recompute** (Kevin's 2026-06-05 P0 design) —
  cheap cron detects fossil windows (alerts ≫ BT in a bucket), queues a targeted
  recompute only there. Doesn't slow normal appends.
- First: understand the 66%/99% split (does full UAD on 296 also show it, or is
  it recompute-specific?).
**Immediate trust win instead → Part B dashboard:** a "settled-through" window +
TBD column makes the dashboard compute its headline on settled data only, so the
clean-then-messier artifact is invisible WITHOUT the engine fix.

### ✅ B2 FIX ATTEMPT #2 — RESOLVED via capped warmup (2026-06-15, validated, dormant)
The attempt-#1 blockers are solved:
1. **Mechanism is correct** — attempt-#1's "66%" was a measurement artifact (band was
   ~[17:33,19:38]; I'd measured [16:00,17:50], so mostly pre-band fossils). Entry-range
   proof: band-replace rows densely cover their window; pre-band fossils correctly
   untouched. Where the band covers, it heals 100%.
2. **Cost fixed by a TF-safe warmup cap.** Full UAD-parity warmup ≈ a full UAD (427s).
   Capping the band recompute's warmup at 5 trading days (scaled UP for coarse TFs so
   the coarsest TF always gets ≥300 bars — never under-warms) cuts it to ~55–83s and is
   **byte-identical to full warmup**, validated on three strategies covering the fleet's
   TF range: **296 ungated 10Sec (103=103), 303 2m gate (23=23), 272 10m gate (27=27)** —
   zero trade differences each. Regression tests: no new failures (4 pre-existing env
   failures unchanged).

**Shipped (local, DORMANT):** `_replace_edge_band` (hoisted above append early-returns,
both lanes) + `db.replace_trades_in_window_admin` (atomic RPC `append_edge_band_replace_rpc.sql`
+ client-side fallback) + TF-safe `_uad_warmup_bars(cap_days=...)`. Gated OFF behind
`APPEND_EDGE_BAND_ENABLED=false` → appends behave exactly as before until activated.
30-min cron throttle; manual `force=True` heals immediately.

**Known limit (documented):** an extreme TF spread (e.g. 10Sec primary + a DAILY gate)
can't be served cheaply by any warmup cap — that needs the **lagged-snapshot resume**
follow-up (TF-agnostic, no warmup; reuses the existing pack-modular snapshot
serialization). Not in the current fleet.

**EOD deploy sequence:** (1) commit + push after-hours (markets closed) → deploys dormant
code, zero risk; (2) set `APPEND_EDGE_BAND_ENABLED=true` on Railway to activate (single
reversible flag); (3) run Update All Data overnight for a clean baseline; (4) tomorrow's
appends use the band-replace → stay clean. Backup branch `dev-backup-pre-b2-fix`.

**✅ DEPLOYED 2026-06-15 EOD:** commit `ed758da` → dev; **`api` deploy SUCCESS** (clean
boot, no import errors). Flag `APPEND_EDGE_BAND_ENABLED=true` set on **`api` only** (manual
Update New Data path) — NOT on Data Worker (its `run_startup_catchup` appends on restart →
would band-replace-storm; auto-append cron is off anyway: `ALGO_HISTORY_CRON_ENABLED=false`).
RPC migration run (atomic advisory-locked path live). **Tomorrow plan (tasks #12-14):**
overnight UAD (full-recompute, flag-independent) → AM test Update New Data on 1-2 strategies
(296+303), watch Combined% on Health Overview → fleet-wide once clean. Snapshot-resume v2
(8× faster, cron-reliable) is the next build — resume returns 0 trades in 3 spikes, needs
`run_unified_backtest` resume-path instrumentation (tasks #10/#11).

### 🩹 Overview over-filtering fix (2026-06-16, commit 37a7dad)
During the partial fleet UAD, Kevin saw sid 304 (30 alerts/15 BT) show BLANK
combined%/paired/phantom/missed in the Health Overview while By-Hour showed it fine.
Root cause: the Overview's apples-to-apples cutoff was a fleet-wide MIN of every strategy's
min(last_bt, last_alert) — sid 305 (stopped firing at 16:11) dragged it back, excluding
every other strategy's later events into TBD. **Fix:** classify per-strategy by the
strategy's OWN coverage (`last_recompute_until_ts` || last_bt_processed), exactly like
By-Hour; `_pair_phantom_missed` returns a 4th value (tbd) split by `coverage_unix`. The
fleet-wide global-fair cutoff + the apples-to-apples toggle are RETIRED from the display
(TBD makes it apples-to-apples naturally per strategy). Verified live: 304 → 54.1% (20/9/8),
305/306 legit-empty (0 activity), 307 → 100%. Read-side only; UAD/append path untouched.

---

## EOD 2026-06-16 — v2 snapshot-resume PROD-VALIDATED; fossilization documented

**v2 snapshot-resume shipped + validated in prod.** `APPEND_EDGE_BAND_MODE=snapshot` (api)
warm-rolls a rolling lagged base snapshot instead of cold warmup. Offline byte-identical to
capped on 302/303/272 (0 diffs); the earlier "resume = 0 trades" was a TEST ARTIFACT
(`prepare_data_with_indicators` ignored a past `end_date` → snapshot landed at "now" →
resume strip nuked the band; fixed by a clip in `get_strategy_trades_for_window`, commit
411e5e7). Live confirmation on controls 296/303: across two consecutive appends the base
snapshot **rolled forward** (296: 20:12→20:43, 303: 20:11→20:44) — the warm-roll signature
— both lanes' band stamps moved in lockstep with `last_recompute`, no wipe, pairing held
95%/92%. (Could not capture `APPEND_ELAPSED` from Railway logs — the `rsi_zones` warning
flood drowns the EDGE-BAND lines in the 500-line snapshot buffer; confirmed via config
fingerprints instead.)

**Confidence that append ≡ full UAD: ~90/100** for recently-UAD'd strategies. Holds back
from ~98 on: (1) edge-bar settlement differences (full UAD re-fetches all bars fresh;
append reuses cached `live_bars`), (2) Tier-3 always-start-flat boundary in rare
position-open-at-boundary cases, (3) offline diffs used the same cached bars both sides
(proves engine math, not a fresh re-fetch). The A/B that converts 90→measured: append then
**immediately** full-UAD the same strategy back-to-back, diff the identical window (task
#18). Running UAD hours after the append is NOT a clean diff (different settled windows).

**Fossilization fully documented** in `docs/Append_Edge_Fossilization.md`. Key points:
an append fills the WHOLE gap (windowed append, INSERT-only) + REWRITES the trailing 120min
(edge-band replace) — no hole. A "fossil" = edge under-count frozen by INSERT-only dedup,
drifted deeper than the 120-min band; only a full UAD scrubs deep fossils. Counterintuitive:
infrequent appends are CLEANER (one settled-data pass); frequent edge-nibbling is what
compounds fossils — but a tight cron is probably SAFE because the 120-min band covers
Polygon settlement (verify before enabling — task #19). Phantom-trade live alert use case
logged (task #20): needs reduced `_ALGO_HISTORY_LAG_MINUTES` (15) for near-real-time.

**Plan locked:** tonight = Update-New-Data across fleet (append, safe — no-wipe guard).
Do NOT run unattended fleet-wide full UAD tonight (#16 lane-wipe-on-fetch-failure still
unguarded — hit 303 today). Tomorrow = A/B back-to-back on 2–3 diverse samples (296 ungated
10Sec, 303 gated 2m, 272 coarse 10m) a few hours into session once early bars settle.
Backup branch `dev-backup-2026-06-16-v2-snapshot` @ 411e5e7.
