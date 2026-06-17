# Fidelity Gate fixtures

Frozen raw-bar + golden gate-column + golden-trade snapshots for the Tier-1
golden replay (`fidelity/golden.py`). See `docs/Fidelity_Gate.md`.

## Committed (durable nightly baseline)
The **live real-money set** — covers ungated + low-TF gate + high-TF gate:
- **sid 308** — TSLA 15Sec, ungated (swing_123 entry)
- **sid 309** — TSLA 15Sec, 2m/3m gates
- **sid 313** — TSLA 15Sec, 1d/4h gates (the high-TF cross-TF case #29 affects most)

## Regenerate the full broad-coverage set (gitignored, ~80 MB)
One fixture per in-use pack + gate-TF class. Regenerate any/all with:
```
cd src && python -m fidelity.capture <sid> 30
```
Full set (sid → coverage):
174 ut_bot_v4(1Min) · 308 swing_123 · 309/311/313 gate-TFs(2m/3m,1h/10m,1d/4h) ·
288 rsi_zones_2 · 290 rvol_v2 · 294 stochastic · 296 strat_assistant ·
298 supertrend · 304 vwap_v2 · 280 ema_pp_v4 · 282 ema_stack_v2 ·
284 macd_histogram_v2 · 286 macd_line_v2 · 276 bollinger_bands · 278 ema_pp_v3

## Run the gate
```
cd src && python -m fidelity.golden            # all present fixtures
cd src && python -m fidelity.golden 308 309 313  # specific
```
