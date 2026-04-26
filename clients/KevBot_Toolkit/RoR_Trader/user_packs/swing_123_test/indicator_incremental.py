"""Swing 1-2-3 Test — incremental indicator class.

Detects the C2 setup (lower low + close above prior close, or its
bearish mirror) and the C3 confirmation (next bar closes beyond the
C2 bar's high/low). Maintains a small ring buffer of the last
`lookback+1` bars so the lookback comparison and the C3 prev-bar
reference are both available without scanning history.

Trigger booleans are computed alongside the indicator columns and
returned in the same dict; the engine's user-pack-pickup loop reads
them under their `{trigger_prefix}_{base}` keys.

The indicator-column prefix (`sw123t_`) and trigger prefix (`s123t_`)
differ — that's a quirk of the original pack manifest preserved for
backward compatibility with any strategies referencing those columns.
"""


class Swing123TestIncremental:
    def __init__(self, **params):
        self.min_c2_wick_pct = float(params.get("min_c2_wick_pct", 0.0))
        self.lookback = int(params.get("lookback_bars", 1))

        # Ring buffer of last `lookback + 1` bars. Index -1 is current,
        # index -(lookback+1) is the C2 reference bar (`lookback` ago),
        # index -2 is the C3 reference bar (previous bar regardless of
        # lookback).
        self._closes: list = []
        self._highs: list = []
        self._lows: list = []

        # Previous-bar C2 booleans — needed to detect C3 on this bar
        self._prev_bull_c2 = False
        self._prev_bear_c2 = False

        # Previous bar's classified state — needed to detect pattern_reset
        self._prev_state = ""

        # Bars seen so far (for the early-warmup guard)
        self._bar_count = 0

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

        self._closes.append(close)
        self._highs.append(high)
        self._lows.append(low)
        max_buf = self.lookback + 1
        if len(self._closes) > max_buf:
            self._closes.pop(0)
            self._highs.pop(0)
            self._lows.pop(0)

        self._bar_count += 1

        # ── C2 detection ──────────────────────────────────────────────
        # Need at least `lookback + 1` bars in history to evaluate C2.
        bull_c2 = False
        bear_c2 = False
        if len(self._closes) >= self.lookback + 1:
            # Index -(lookback+1) is the C1 reference bar
            close_lb = self._closes[-self.lookback - 1]
            high_lb = self._highs[-self.lookback - 1]
            low_lb = self._lows[-self.lookback - 1]

            if close_lb is not None and close_lb != 0:
                bull_c2 = (low < low_lb) and (close > close_lb)
                if self.min_c2_wick_pct > 0 and bull_c2:
                    bull_wick_ext = (close_lb - low) / close_lb
                    bull_c2 = bull_wick_ext >= self.min_c2_wick_pct

                bear_c2 = (high > high_lb) and (close < close_lb)
                if self.min_c2_wick_pct > 0 and bear_c2:
                    bear_wick_ext = (high - close_lb) / close_lb
                    bear_c2 = bear_wick_ext >= self.min_c2_wick_pct

        # ── C3 detection ──────────────────────────────────────────────
        # Prev bar was a C2 of matching direction AND current close
        # broke beyond that bar's high (bull) or low (bear).
        bull_c3 = False
        bear_c3 = False
        if len(self._closes) >= 2:
            high_prev = self._highs[-2]
            low_prev = self._lows[-2]
            bull_c3 = self._prev_bull_c2 and (close > high_prev)
            bear_c3 = self._prev_bear_c2 and (close < low_prev)

        # ── State classification (priority: C3 > C2 > NEUTRAL) ───────
        if bull_c3:
            current_state = "BULLISH_C3"
        elif bear_c3:
            current_state = "BEARISH_C3"
        elif bull_c2:
            current_state = "BULLISH_C2"
        elif bear_c2:
            current_state = "BEARISH_C2"
        else:
            current_state = "NEUTRAL"

        # Early-bar guard — match the original interpreter's behavior of
        # treating bars within the first `lookback + 1` as None.
        if self._bar_count < self.lookback + 1:
            current_state = ""

        # ── Trigger booleans ──────────────────────────────────────────
        c2_detected_bull = current_state == "BULLISH_C2"
        c2_detected_bear = current_state == "BEARISH_C2"
        c3_confirmed_bull = current_state == "BULLISH_C3"
        c3_confirmed_bear = current_state == "BEARISH_C3"

        active_states = ("BULLISH_C2", "BULLISH_C3", "BEARISH_C2", "BEARISH_C3")
        pattern_reset = (
            self._prev_state in active_states
            and current_state == "NEUTRAL"
        )

        # ── Candle color (priority: C3 > C2 > none) ──────────────────
        candle_color = ""
        if bull_c3:
            candle_color = "#FFFF00"  # Bullish C3: neon yellow
        elif bear_c3:
            candle_color = "#FF33CC"  # Bearish C3: neon pink
        elif bull_c2:
            candle_color = "#FFD11A"  # Bullish C2: med yellow
        elif bear_c2:
            candle_color = "#FF66B3"  # Bearish C2: med pink

        # ── Update state for next bar ────────────────────────────────
        self._prev_bull_c2 = bool(bull_c2)
        self._prev_bear_c2 = bool(bear_c2)
        self._prev_state = current_state

        # Level columns: emit current bar's close. The engine adds the
        # 1-bar lag automatically by caching at this bar's close and
        # using on the next bar for intra-bar reachability checks.
        # Do NOT shift here — that would cause a 2-bar lag.
        return {
            # Indicator columns (consumed by interpreter + L-type level cache)
            "sw123t_bull_c2": bool(bull_c2),
            "sw123t_bear_c2": bool(bear_c2),
            "sw123t_bull_c3": bool(bull_c3),
            "sw123t_bear_c3": bool(bear_c3),
            "sw123t_candle_color": candle_color,
            "sw123t_bull_entry_level_prev": close,
            "sw123t_bear_entry_level_prev": close,
            # Trigger booleans (engine reads via user-pack pickup loop)
            "s123t_bullish_c2_detected": bool(c2_detected_bull),
            "s123t_bullish_c3_confirmed": bool(c3_confirmed_bull),
            "s123t_bearish_c2_detected": bool(c2_detected_bear),
            "s123t_bearish_c3_confirmed": bool(c3_confirmed_bear),
            "s123t_pattern_reset": bool(pattern_reset),
        }
