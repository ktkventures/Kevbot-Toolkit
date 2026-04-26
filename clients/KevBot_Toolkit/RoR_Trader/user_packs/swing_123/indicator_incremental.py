"""Swing 1-2-3 — incremental indicator class.

Pure pattern detection: detects C2 reversal candidates and C3
continuation confirmations against the prior bar. No look-back
parameter (always compares to the immediately prior bar). No level
columns — this is a hidden-display pack with discrete state events,
so triggers only get C and CC variants from the registry.

The original batch encoded the pattern as an integer in
`sw123_pattern` (0=neutral, 1=bull C2, 2=bull C3, -1=bear C2,
-2=bear C3). The interpreter reads that column for state
classification, and the trigger function derives bool-per-pattern
from it. Mirrored exactly here.
"""


class Swing123Incremental:
    def __init__(self, **params):
        # No params — original pack has parameters_schema={}
        # State: just the previous bar's prices and the previous bar's
        # pattern code (needed to detect C3, since C3 = "prev was C2").
        self._prev_close = None
        self._prev_high = None
        self._prev_low = None
        self._prev_pattern = 0  # 0 = neutral

    def warmup(self, df) -> None:
        for _, row in df.iterrows():
            self.update_bar({
                "open": row.get("open", 0.0),
                "high": row.get("high", 0.0),
                "low": row.get("low", 0.0),
                "close": row.get("close", 0.0),
                "volume": row.get("volume", 0.0),
            })

    def update_bar(self, bar: dict) -> dict:
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])

        pattern = 0
        candle_color = ""

        if self._prev_close is not None:
            # C2 conditions
            bull_c2 = low < self._prev_low and close > self._prev_close
            bear_c2 = high > self._prev_high and close < self._prev_close

            # C3 conditions: prev bar was C2 of matching direction AND
            # current close broke beyond prev bar's high (bull) or low (bear)
            bull_c3 = (self._prev_pattern == 1) and close > self._prev_high
            bear_c3 = (self._prev_pattern == -1) and close < self._prev_low

            # Priority: C3 > C2 (matches the Pine reference and original batch)
            if bull_c3:
                pattern = 2
                candle_color = "#FFFF00"  # Bullish C3: neon yellow
            elif bear_c3:
                pattern = -2
                candle_color = "#FF33CC"  # Bearish C3: neon pink
            elif bull_c2:
                pattern = 1
                candle_color = "#FFD11A"  # Bullish C2: med yellow
            elif bear_c2:
                pattern = -1
                candle_color = "#FF66B3"  # Bearish C2: med pink

        # Trigger booleans — each pattern code maps to one trigger
        bull_c2_fired = pattern == 1
        bull_c3_fired = pattern == 2
        bear_c2_fired = pattern == -1
        bear_c3_fired = pattern == -2

        # Update state for next bar
        self._prev_close = close
        self._prev_high = high
        self._prev_low = low
        self._prev_pattern = pattern

        return {
            # Indicator columns (interpreter reads these for state classification)
            "sw123_pattern": pattern,
            "sw123_candle_color": candle_color,
            # Trigger booleans (engine reads via user-pack pickup loop)
            "sw123_bull_c2": bool(bull_c2_fired),
            "sw123_bull_c3": bool(bull_c3_fired),
            "sw123_bear_c2": bool(bear_c2_fired),
            "sw123_bear_c3": bool(bear_c3_fired),
        }
