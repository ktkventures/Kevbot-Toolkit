"""SuperTrend — incremental indicator class.

Per-bar Wilder ATR + ratcheted up/down bands + trend direction. The
per-bar `update_bar` method maintains the rolling state that the batch
calculation produces in a single vectorized loop. Both code paths
share `indicator.py` as a thin wrapper so they cannot drift apart —
the parity simulator verifies bit-equivalence.

Trigger booleans (bull_flip, bear_flip, near_stop_bull, near_stop_bear)
are computed alongside the indicator state and returned in the same
dict so the engine can pick them up via the user-pack-trigger merge in
TriggerEvaluator.
"""


class SupertrendIncremental:
    def __init__(self, **params):
        self.atr_period = int(params.get("atr_period", 10))
        self.multiplier = float(params.get("atr_multiplier", 3.0))
        self._alpha = 1.0 / self.atr_period

        # Indicator state
        self._first = True
        self._atr = 0.0
        self._prev_close = 0.0
        self._up = 0.0          # ratcheted upper band
        self._dn = 0.0          # ratcheted lower band
        self._trend = 1
        self._st_line = 0.0
        self._prev_st_line = 0.0

        # Trigger state
        self._prev_trend = 1
        self._prev_was_far_bull = False
        self._prev_was_far_bear = False

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

        # Capture previous st_line (exposed as the `_prev` column for L-type
        # entries; engine adds the 1-bar lag automatically).
        out_prev_line = self._prev_st_line

        if self._first:
            tr = high - low
            self._atr = tr
            src = (high + low) / 2.0
            self._up = src - self.multiplier * tr
            self._dn = src + self.multiplier * tr
            self._trend = 1
            self._st_line = self._up
            self._prev_close = close
            self._first = False
            bull_flip = False
            bear_flip = False
            near_stop_bull = False
            near_stop_bear = False
        else:
            # Wilder true range + RMA-smoothed ATR
            tr = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close),
            )
            self._atr = self._atr + self._alpha * (tr - self._atr)

            # Basic bands from hl2 ± multiplier·ATR
            src = (high + low) / 2.0
            basic_up = src - self.multiplier * self._atr
            basic_dn = src + self.multiplier * self._atr

            # Ratchet bands — they only move in the direction of the trend
            new_up = (
                max(basic_up, self._up)
                if self._prev_close > self._up else basic_up
            )
            new_dn = (
                min(basic_dn, self._dn)
                if self._prev_close < self._dn else basic_dn
            )

            # Trend flips when close crosses the *previous* bar's band
            # (matches the batch loop which references self._up / self._dn
            # before they're overwritten — those are the previous-bar
            # values at this point in the sequence).
            if self._trend == -1 and close > self._dn:
                new_trend = 1
            elif self._trend == 1 and close < self._up:
                new_trend = -1
            else:
                new_trend = self._trend

            # Trigger detection happens BEFORE updating state — compare
            # against the previous-bar values still held by self.
            bull_flip = self._prev_trend == -1 and new_trend == 1
            bear_flip = self._prev_trend == 1 and new_trend == -1

            # Near-stop proximity (within 0.5 × ATR of the active line)
            threshold = 0.5 * self._atr
            new_st_line = new_up if new_trend == 1 else new_dn
            now_near_bull = (close - new_st_line) <= threshold and new_trend == 1
            now_near_bear = (new_st_line - close) <= threshold and new_trend == -1
            near_stop_bull = self._prev_was_far_bull and now_near_bull
            near_stop_bear = self._prev_was_far_bear and now_near_bear

            # Update "was far" tracking for next bar
            far_bull = (close - new_st_line) > threshold and new_trend == 1
            far_bear = (new_st_line - close) > threshold and new_trend == -1
            self._prev_was_far_bull = far_bull
            self._prev_was_far_bear = far_bear

            # Commit indicator state
            self._up = new_up
            self._dn = new_dn
            self._trend = new_trend
            self._st_line = new_st_line
            self._prev_close = close

        # Save prev_trend for next-bar flip detection
        self._prev_trend = self._trend
        # Save current line as next bar's `_prev`
        self._prev_st_line = self._st_line

        return {
            "st_line": self._st_line,
            "st_direction": self._trend,
            "st_atr": self._atr,
            "st_line_prev": out_prev_line,
            "st_bull_flip": bool(bull_flip),
            "__st_bull_flip": bool(bull_flip),  # alias for interpreter
            "st_bear_flip": bool(bear_flip),
            "__st_bear_flip": bool(bear_flip),  # alias for interpreter
            "st_near_stop_bull": bool(near_stop_bull),
            "__st_near_stop_bull": bool(near_stop_bull),  # alias for interpreter
            "st_near_stop_bear": bool(near_stop_bear),
            "__st_near_stop_bear": bool(near_stop_bear),  # alias for interpreter
        }
