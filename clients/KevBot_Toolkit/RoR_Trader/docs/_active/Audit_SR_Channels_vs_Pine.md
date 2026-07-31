# Audit — `sr_channels` pack vs. the Pine source

**Board #73 step 2 (E·auto, 2026-07-31).** Clause-by-clause mapping of
`reference-indicators/sr_channels.pine` (LonesomeTheBlue, "Support Resistance
Channels", v6) against the shipped pack in `user_packs/sr_channels/`.

**Verdict: REBUILD against the Pine as spec. Do not patch.** Reasoning in §5.

Scope note: *why the AI pack builder diverged* is **#146**, deliberately not
investigated here.

---

## 0 · What was compared

| | |
|---|---|
| Pine source | `reference-indicators/sr_channels.pine` — 193 lines |
| Shipped pack | `indicator_incremental.py` 309 L · `indicator.py` 49 L · `interpreter.py` 37 L · `manifest.json` 72 L |
| Audit instrument | `tools/sr_channels_pine_reference.py` — faithful transliteration, Pine line numbers in comments |

The mapping below is not a reading — it is **measured**. The instrument and the
shipped pack were run over identical deterministic synthetic bars and diffed
(§4). Anyone can reproduce it; it touches no DB and no network.

---

## 1 · Clause-by-clause mapping

Legend: **✅ faithful** · **⚠️ divergent** · **❌ absent** · **➕ our invention, not in the Pine**

### 1.1 Inputs

| Pine | Ours | |
|---|---|---|
| L6 `prd` 10 (4–30) | `pivot_period` 10 (4–30) | ✅ |
| L7 `ppsrc` High/Low \| Close/Open | `pivot_source` idem | ✅ |
| L8 `ChannelW` 5 (1–8) | `channel_width_pct` 5 (1–8) | ✅ |
| L9 `minstrength` 1 (minval 1, no max) | `min_strength` 1 (1–10) | ✅ cap is ours; harmless |
| L10 `maxnumsr = input(6) − 1` | `max_num_sr` 6, used directly | ✅ **equivalent by a different route** — Pine loops `0..min(9, N−1)` = `min(10,N)` channels; we take `min(10,N)` directly. Same count for every N. Do not "fix" the off-by-one. |
| L11 `loopback` 290 (100–400) | `loopback` 290 (100–400) | ✅ |
| L12-14 `res_col`, `sup_col`, **`inch_col`** | `resistance_color`, `support_color` | ⚠️ `inch_col` (price-inside-channel grey) **absent** — see §3 |
| L15-16 `showpp`, `showsrbroken` | — | ❌ display toggles, non-material |
| L17-22, L24-28 MA1 / MA2 overlays | — | ❌ display-only, default off, non-material |

### 1.2 Pivot detection

| Pine | Ours | |
|---|---|---|
| L31-32 `src1/src2` from ppsrc | `_src_high` / `_src_low` | ✅ |
| L33-34 `ta.pivothigh(src1, prd, prd)` | `_confirm_pivot_at` | ⚠️ **DIVERGENT — see §2.1.** Pine's tie-break is asymmetric; ours demands a **unique** extremum over the whole `2·prd+1` window. Our pivot set is a **provable subset** of Pine's. |
| L37-38 pivot labels | — | ❌ display-only |
| L48-50 keep pivots, `bar_index` as loc | `_build_channels` walks pivot-bar indices | ✅ **equivalent window.** Pine stores the *confirmation* index (= pivot + prd) and windows `[cur−loopback, cur]`; we window pivot bars `[cur−prd−loopback, cur−prd]`. Identical set. |
| L49 `unshift(pivotvals, ph ? ph : pl)` — **ONE value per bar** | we append **both** `ph` and `pl` | ⚠️ **DIVERGENT — see §2.3.** When a high and a low confirm on the same bar the Pine **discards the low**. |
| L51-56 drop pivots older than loopback | window clamp | ✅ |
| pivot array **order** — newest-first (`unshift`) | oldest-first (ascending buffer scan) | ⚠️ **DIVERGENT — see §2.2.** Order is load-bearing. |

### 1.3 Channel width

| Pine | Ours | |
|---|---|---|
| L41-43 `cwidth = (highest(300) − lowest(300)) · ChannelW/100` | same, `max(highs)/min(lows)` over 300 | ✅ formula and sources match (Pine's arg-less `ta.highest/ta.lowest` default to `high`/`low`) |
| `ta.highest(300)` is `na` until 300 bars exist | partial window from bar 0 (`max(0, idx−299)`) | ⚠️ **warmup divergence** — we emit channels before the Pine would. §2.5 |

### 1.4 Channel construction — `get_sr_vals` (L59-76)

| Pine | Ours | |
|---|---|---|
| L65 `wdth = cpp <= hi ? hi−cpp : cpp−lo` | identical | ✅ |
| L66-72 `if cpp<=hi: lo:=min(lo,cpp) else: hi:=max(hi,cpp)` | `if cpp<lo: lo=cpp` / `if cpp>hi: hi=cpp` (independent) | ✅ **provably equivalent** — case-split on `cpp<=hi` gives the same result in both branches |
| L74 `numpp += 20` per member | `pp_count += 20` | ✅ |
| iteration order determines the grown `hi`/`lo` | — | ⚠️ inherits the §2.2 ordering divergence |

### 1.5 Strength — bar touches (L99-107)

| Pine | Ours | |
|---|---|---|
| L104 `high[y] ∈ [l,h] or low[y] ∈ [l,h]` (precedence: `and` binds tighter) | identical condition | ✅ |
| L103 `for y = 0 to loopback` → `loopback+1` bars ending at current | `lb+1` bars ending at `current_idx` | ✅ |
| L107 `strength = numpp + touches` | `ch_strengths[k] = pp_count + touch_count` | ✅ |

### 1.6 Selection & ranking (L109-143)

| Pine | Ours | |
|---|---|---|
| L110 reset `suportresistance` | fresh list each build | ✅ |
| L117 pick max with `> stv and >= minstrength·20` | identical gate, `best_val` init −1 | ✅ |
| L130-132 zero out every channel overlapping the winner | `used[k]=True; ch_strengths[k]=-1` | ✅ |
| L135 stop at 10 | `min(10, max_num_sr)` rounds | ✅ (see L10 row) |
| ties resolved to the **lowest array index** | idem — but the array order differs | ⚠️ inherits §2.2 |
| L138-143 bubble sort by strength | `sr_levels.sort(key=-strength)` | ✅ **and both are dead code.** The greedy pick is non-increasing by construction, so the branch never fires — **verified empirically: `sort_ever_fired = False` on all 3 seeds.** Note the Pine's sort body is itself buggy (`stren[x]` is never assigned `tmp`, L141-142). **Do not port that bug.** |

### 1.7 Price-vs-channel state (L169-192)

| Pine | Ours | |
|---|---|---|
| L172-177 `not_in_a_channel` over `0..min(9,maxnumsr)` | `close<=top and close>=bot` over the level list | ✅ condition equivalent |
| L180 breakout checked **only when not in a channel** | same guard | ✅ |
| L182 `close[1] <= top and close > top` → resistance broken | identical | ✅ |
| L185 `close[1] >= bot and close < bot` → support broken | identical | ✅ |
| **the array these read is FROZEN between pivot bars** | **recomputed every bar** | 🔴 **DIVERGENT — the big one. §2.4** |
| L189-190 two `alertcondition`s | `src_resistance_broken`, `src_support_broken` | ✅ trigger surface matches |

### 1.8 Rendering (L146-167)

| Pine | Ours | |
|---|---|---|
| L162-167 up to `maxnumsr+1` **boxes**, `extend.both`, one per channel | two numeric columns for the **nearest channel only** | 🔴 **ABSENT — §3** |
| L158 colour: resistance if **both** bounds > close, support if **both** < close, else grey | — | 🔴 absent |

### 1.9 Our inventions (no Pine counterpart)

| Ours | |
|---|---|
| `src_nearest_top` / `src_nearest_bot` — nearest channel by **midpoint distance** | ➕ Pine has no "nearest channel" concept at all |
| `src_num_channels`, `src_in_channel`, `src_res_broken`, `src_sup_broken` | ➕ harmless scalars |
| `src_enter_sr_zone` / `src_exit_sr_zone` triggers | ➕ no Pine equivalent |
| Interpreter states `ABOVE_RESISTANCE / IN_RESISTANCE / BETWEEN_LEVELS / IN_SUPPORT / BELOW_SUPPORT` | ➕ **and two conflict with the Pine's own semantics — §2.6** |
| `trigger_levels` → `src_nearest_top_prev` / `src_nearest_bot_prev` for L-type | ➕ **internally inconsistent — §2.7** |

---

## 2 · The divergences, named

### 2.1 Pivot detection is strictly under-inclusive 🔴
`_confirm_pivot_at` (L96-117) requires the centre bar to be the **unique**
max/min across `[p−prd, p+prd]` — any equal value anywhere in the 21-bar window
kills the pivot. TradingView's `ta.pivothigh` uses an **asymmetric** tie-break
(one side admits equality). Whichever side that is, our rule rejects a superset
of what Pine rejects, so **our pivots are a strict subset of Pine's** —
confirmed by exhaustive check in the instrument.

Measured miss rate on quantised prices:

| tick | Pine pivot highs | ours | **missed** |
|---|---|---|---|
| $0.01 | 44 | 36 | **18.2 %** |
| $0.05 | 43–45 | 20 | **53.5–55.6 %** |

Equal highs are not an edge case — they are what quantised prices *do*, and
they get denser as the bar interval shrinks. Our sub-minute lanes are the worst
case. Everything downstream is built on this input.

> ⚠️ **The one clause this audit could not settle offline:** *which* side of
> `ta.pivothigh` admits equality. The instrument parameterises it
> (`pivot_tie="left_strict" | "right_strict"`); the measured gap is
> materially the same either way (§4). Settle it against a live TradingView
> chart during the rebuild — it is a one-chart check, and it is the only
> open question in the whole mapping.

### 2.2 Pivot iteration order is reversed 🔴
Pine's `pivotvals` is **newest-first** (`array.unshift`, L49). We build `pv` by
scanning the buffer **ascending** = oldest-first (L133-139). `get_sr_vals`
grows `hi`/`lo` as it walks, so the resulting channel bounds and `numpp` are
**order-dependent** — and selection ties resolve to the lowest index, which is
now a different pivot. Same pivot set, different channels.

### 2.3 Simultaneous high+low: Pine keeps only the high ⚠️
Pine L49 unshifts **one** value per confirmation bar: `bool(ph) ? ph : pl`. If
both confirm on the same bar the **low is silently dropped**. We append both
(L134-139). Rare on High/Low source, much less rare on Close/Open.

### 2.4 We recompute every bar; the Pine freezes between pivots 🔴 **— the big one**
The entire S/R rebuild in Pine sits inside `if bool(ph) or bool(pl)` (L88), and
`suportresistance` is a `var` — **persistent**. Channels change **only on a bar
that confirms a new pivot** (measured: 88 of 1400 bars = 6.3 %). Between those,
the levels are constants and price moves against a fixed line.

`update_bar` calls `_build_channels(current_idx)` on **every bar** (L245). Both
inputs to the channel set move every bar — `cwidth` (rolling 300-bar range) and
the touch counts (rolling `loopback` window) — so **our levels drift under the
price continuously**.

This is what makes a "break" mean something different in each implementation.
In Pine, price crosses a level that is standing still. In ours, the level is
sliding while price crosses it, so we manufacture breaks that never happened
and erase ones that did. It also explains §2.7: an L-type intra-bar level is
only well-defined if the level holds still for the duration of the bar.

It is also **the reason the rebuild is cheaper**: doing the O(P × loopback)
scoring on 6 % of bars instead of 100 % of them.

### 2.5 Warmup: we emit channels the Pine would not ⚠️
`ta.highest(300)` returns `na` until 300 bars exist, so Pine produces no
channels at all before then. We use a partial window from bar 0. The manifest
declares no minimum-bar requirement. Per [[feedback_indicator_warmup]] this
should be an explicit, declared warmup — never a silent partial result.
(The buffer sizing itself is **correct**: `_max_buf = max(300, loopback+2·prd+2)`
= 312 at defaults, and the pivot-confirmation index tracks correctly through
the trim. ✅)

### 2.6 The interpreter's five states contradict the Pine ⚠️
* **`IN_RESISTANCE` / `IN_SUPPORT`** split "price inside a channel" by whether
  close is above or below the channel **midpoint**. Pine's own colour rule
  (L158) says a channel is resistance when **both** bounds are above close,
  support when **both** are below, and **neither** when price is inside it —
  that is the grey `inch_col` case. We assert a directional bias exactly where
  the Pine declines to.
* **`ABOVE_RESISTANCE` / `BELOW_SUPPORT`** are computed against the **nearest**
  channel only, so price is labelled `ABOVE_RESISTANCE` while sitting beneath
  five stronger channels. The natural reading — above *all* channels — is not
  what the code does.

### 2.7 C-type and L-type triggers fire off different levels ⚠️
`src_nearest_top_prev` is emitted as *the previous bar's* nearest top (L229-230,
L302-303) and `trigger_levels` points the L-type intra-bar cross at it. But the
C-type break inside `update_bar` (L266-272) is evaluated against **this** bar's
freshly-rebuilt levels. Same logical trigger, two different levels. §2.4 is the
root cause; freezing the channel set collapses this automatically.

---

## 3 · Rendering — Kevin's "may need to install new shapes"

He is right, and it is a **frontend (F) gap, not an engine one.**

The Pine draws **up to 6 boxes at once**, extended both directions, each
recoloured per bar into one of three states (resistance / support /
price-inside). What our chart stack can express:

| primitive | where | what it does |
|---|---|---|
| `column_color_map` → line series | Next.js chart | one **line** per column, **one static colour** |
| `plot_config.band_fills` | **legacy Streamlit `app.py` only** (`get_band_fills_for_group`) | fill between **exactly two** named columns, **one static colour key** |

The production Next.js chart (`frontend/src/charts/`) constructs only
`addLineSeries` / `addCandlestickSeries` / `addHistogramSeries`, and consumes
**neither** `column_color_map` nor `band_fills`. So today `sr_channels` plots as
**two flat lines — the nearest channel only** — and 5 of 6 channels are simply
invisible.

Three separable gaps:

1. **N channels, not 1.** Needs `src_ch{1..6}_top/bot` columns (12) or a
   multi-band primitive. Expressible in the manifest schema as it stands.
2. **Per-bar, per-channel colour.** `fill_color_key` is a *static* plot_schema
   key. Not expressible at all — needs a new primitive (a per-bar colour column
   per band, mirroring how `candle_color_column` already carries hex strings).
3. **The Next.js chart honours neither.** Even `band_fills` — shipped and used
   by `bollinger_bands` and `stochastic_oscillator` — renders only in the legacy
   Streamlit chart.

**Recommendation: do not fold this into #73.** The engine rebuild is
self-contained and testable against the Pine's numbers; the chart primitive is
an F-lane schema + renderer change that would otherwise gate an engine fix on
frontend work. Item 3 is worth raising on its own — two shipped packs already
believe they render bands and do not.

Also still open and out of scope here: `pine_generator.py` L833 has
`sr_channels` **NOT YET PORTED** for TradingView export. A faithful Python
rebuild makes that port mechanical, so it should follow — not lead.

---

## 4 · Measured gap

Instrument: `tools/sr_channels_pine_reference.py`; harness reproduced in the
task thread. 1400 deterministic synthetic bars/seed, first 400 discarded as
warmup, 1000 bars scored, pack defaults on both sides (= Pine defaults).

**Trigger fidelity — the numbers that matter for a signal:**

| seed | trigger | shared fires | **spurious (ours, no Pine)** | **missed (Pine, no ours)** |
|---|---|---|---|---|
| 7 | resistance_broken | 67 | **16 = 19.3 %** | 13 = 16.2 % |
| 7 | support_broken | 68 | **21 = 23.6 %** | 6 = 8.1 % |
| 13 | resistance_broken | 48 | **10 = 17.2 %** | 10 = 17.2 % |
| 13 | support_broken | 51 | **10 = 16.4 %** | 8 = 13.6 % |
| 21 | resistance_broken | 70 | **23 = 24.7 %** | 14 = 16.7 % |
| 21 | support_broken | 89 | **33 = 27.0 %** | 15 = 14.4 % |

**Roughly one in five of our break signals is not a Pine break, and we miss a
comparable share of the real ones.** `in_channel` disagrees on 3.0–9.8 % of all
bars (we miss 24–48 % of the bars Pine calls in-channel).

Switching the unresolved pivot tie-break (§2.1) moves these by under a point —
**the gap is structural, not a tie-break detail.**

**Cost:**

| | per-bar `update_bar` |
|---|---|
| shipped pack | 0.32 – 0.39 ms |
| faithful (freeze-based) | 0.048 – 0.069 ms |
| | **5 – 7× cheaper** |

The faithful implementation is not a performance concession. It is **5–7×
faster**, because §2.4 means we do the expensive scoring on every bar where the
Pine does it on 6 %. Against [[feedback_sub_second_latency]] that is the
direction we want.

**Also confirmed:** the Pine's L138-143 sort never fires on any seed — dead
code, and its `stren[x]`-never-assigned bug is unreachable. Do not port it.

---

## 5 · The call — REBUILD, not patch

**Rebuild `sr_channels` against the Pine as spec, behind a default-OFF flag.**

1. **The gap is structural, not incremental.** Of the divergences, four
   (§2.1–2.4) sit at the *bottom* of the pipeline — pivot detection, pivot
   ordering, pivot bookkeeping, and when the whole thing recomputes. Every
   number the pack emits is downstream of all four. There is no patch that
   fixes one without re-deriving everything above it.
2. **Patching costs more than rebuilding.** The one change that matters most —
   freezing the channel set between pivot bars (§2.4) — inverts the control
   flow of `update_bar`/`_build_channels`, which is the bulk of the 309-line
   file. Once that is restructured, the ordering and pivot fixes are trivially
   absorbed. Patch and rebuild converge on the same diff; the rebuild just
   arrives with the Pine's line numbers attached.
3. **The measurement instrument already exists and is the oracle.** §4 is a
   ready-made differential test: assert the rebuild is byte-identical to the
   transliteration across seeds. A patched implementation would need the same
   harness — so the harness is sunk cost either way, and it only pays off
   against a from-scratch port.
4. **Zero blast radius, and it will never be this cheap again.** 0 of 24
   strategies reference `sr_channels`; two confluence *groups* define it
   (`sr_channels_default`, config/confluence_groups.json) and no strategy uses
   them. No baseline to preserve, no migration, nothing to re-validate. There
   are **no tests** on this pack today — a rebuild starts the test file that a
   patch would also have to start.
5. **It is faster** (§4) — so faithfulness costs nothing at runtime.

### Recommended shape for step 4 (E, next)
* Port `tools/sr_channels_pine_reference.py` into `indicator_incremental.py` as
  the pack's own logic, keeping the Pine line-number comments.
* **Freeze the channel set between pivot bars** (§2.4) — the load-bearing change.
* Pivot detection to TradingView's asymmetric rule, after settling §2.1 against
  a live chart.
* Newest-first pivot ordering (§2.2); one unshift per bar (§2.3).
* Declare the 300-bar warmup explicitly (§2.5); never emit a partial-window
  channel.
* Keep the outputs' *names* (`src_*`) so no manifest/registry churn; redefine
  `IN_RESISTANCE`/`IN_SUPPORT`/`ABOVE_RESISTANCE`/`BELOW_SUPPORT` to the Pine's
  own semantics (§2.6), including the third "inside = neither" state.
* Differential test vs. the transliteration across seeds; that is the gate.
* Default-OFF flag; one branch, one PR.

### Explicitly deferred (raise, don't absorb)
* **Multi-channel rendering + per-bar band colour** — F lane, §3. Including the
  standalone finding that the Next.js chart honours no `band_fills` at all.
* **`pine_generator` TradingView export** for `sr_channels` — follows the
  rebuild.
* **Why the builder simplified it** — **#146**, untouched by design.

---

*E·auto, run `r1785477931-73`, branch `audit/sr-channels-pine-73`.*
