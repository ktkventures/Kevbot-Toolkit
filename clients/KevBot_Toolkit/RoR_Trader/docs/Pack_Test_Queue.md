# Pack Test Queue

Working list of user packs to build through the wizard for end-to-end validation. Each pack tests a different aspect of the pipeline. We work through them one at a time so issues can be diagnosed and the pipeline strengthened iteratively.

## Process

1. Use the prompt below in Pack Builder Step 1 (Pack Info)
2. Run Step 2 (Generate Structure) — review the AI's proposed parameters/outputs/triggers
3. Refine if needed in Step 3
4. Run Step 4 (Generate & Validate Code)
5. Install via Step 5
6. Test in Strategy Builder with the new pack
7. If issues arise, Claude fixes directly in `user_packs/<slug>/` and documents lessons in `pack_builder_context.md` + adds validation rules

## Completed

- ✅ **swing_123_test** — Pattern-based pack with candle coloring + L-type intra-bar entry on previous close reclaim. Validated: pattern detection, candle color override, level column flow through engine, strict cross check.

---

## Next Up

### Pack 1: Stochastic Oscillator (Recommended next)

**Why this one:** Classic oscillator. Tests oscillator pane rendering and standard level-cross L-type without pattern complexity.

**Pack Info to enter in Wizard:**
- **Name:** Stochastic Oscillator
- **Category:** Momentum
- **Display Type:** oscillator
- **Description:**
```
Stochastic oscillator with %K and %D lines. Shows overbought above 80 and oversold below 20. I want signals when %K crosses %D and when it enters or leaves the extreme zones.
```

**Pine Script Reference (optional — paste if you have it handy):**
```pinescript
//@version=6
indicator("Stochastic")
k_period = input.int(14)
k_smooth = input.int(3)
d_period = input.int(3)

k = ta.sma(ta.stoch(close, high, low, k_period), k_smooth)
d = ta.sma(k, d_period)

plot(k, color=color.blue)
plot(d, color=color.orange)
hline(80, color=color.red)
hline(50, color=color.gray)
hline(20, color=color.green)
```

**What we're validating (post-install):**
- [ ] Oscillator pane renders in Chart Preview
- [ ] Two lines (K and D) plot correctly
- [ ] Reference lines (80/50/20) appear
- [ ] State distribution reasonable (no single state dominates)
- [ ] Triggers fire at expected zones
- [ ] L-type variants are auto-generated and fire intra-bar
- [ ] Strategy Builder backtest produces clean trades

**Things to watch for (developer eyes only):**
- AI must not use `.shift()` or `.where()` on level columns (validation catches)
- All triggers must be `BOTH/BOTH`
- Trigger prefix should be short like `stoch`
- Level columns must end in `_prev` and be in `indicator_columns` but not `plot_schema`

Bugs found here = improvements to add to context.md, prompts, and validation.

---

## Future Queue

### Pack 2: Donchian Channels
- **Type:** Overlay (price chart)
- **Tests:** Lookback-based highest high / lowest low, breakout L-type entries
- **Triggers:** Cross above N-bar high (breakout), cross below N-bar low (breakdown)

### Pack 3: Keltner Channels
- **Type:** Overlay (price chart)
- **Tests:** EMA + ATR computation, multi-line bands
- **Triggers:** Cross above upper band, cross below lower band, return to midline

### Pack 4: Choppiness Index
- **Type:** Oscillator
- **Tests:** Regime detection states (trending vs choppy), single-line oscillator
- **Triggers:** Cross above 61.8 (entering chop), cross below 38.2 (exiting chop)

### Pack 5: CCI (Commodity Channel Index)
- **Type:** Oscillator
- **Tests:** Unbounded oscillator with 100/-100/200/-200 thresholds
- **Triggers:** Standard level crosses

### Pack 6: SR Channels (Stress Test — Save for last)
- **Type:** Overlay (price chart)
- **Tests:** Multiple dynamic levels, pivot-based detection, complex state logic
- **Why hard:** Tests the limits of the system with multiple simultaneous levels
