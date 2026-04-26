"""MACD Line v2 — incremental indicator class.

Three EMAs (fast/slow/signal) with alpha = 2/(N+1):
  macd_line   = EMA(close, fast) - EMA(close, slow)
  macd_signal = EMA(macd_line, signal)
  macd_hist   = macd_line - macd_signal

Mirrors the built-in `macd_line` template's algorithm exactly.

Triggers:
  - cross_bull  / cross_bear: macd_line crosses macd_signal
  - zero_cross_up / zero_cross_down: macd_line crosses zero

No level columns (MACD values aren't price levels), so only C and CC
runtime variants get materialized.
"""


class MacdLineV2Incremental:
    def __init__(self, **params):
        self.fast = int(params.get("fast_period", 12))
        self.slow = int(params.get("slow_period", 26))
        self.signal_p = int(params.get("signal_period", 9))

        self._a_fast = 2.0 / (self.fast + 1)
        self._a_slow = 2.0 / (self.slow + 1)
        self._a_sig = 2.0 / (self.signal_p + 1)

        self._first = True
        self._ema_fast = 0.0
        self._ema_slow = 0.0
        self._signal_ema = 0.0
        self._prev_macd_line = 0.0
        self._prev_signal = 0.0

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
        close = float(bar["close"])

        if self._first:
            self._ema_fast = close
            self._ema_slow = close
            macd_line = 0.0
            self._signal_ema = 0.0
            self._first = False
        else:
            self._ema_fast = self._a_fast * close + (1.0 - self._a_fast) * self._ema_fast
            self._ema_slow = self._a_slow * close + (1.0 - self._a_slow) * self._ema_slow
            macd_line = self._ema_fast - self._ema_slow
            self._signal_ema = (
                self._a_sig * macd_line + (1.0 - self._a_sig) * self._signal_ema
            )

        macd_hist = macd_line - self._signal_ema

        # Trigger detection — current vs previous values
        prev_line = self._prev_macd_line
        prev_sig = self._prev_signal

        cross_bull = (macd_line > self._signal_ema) and (prev_line <= prev_sig)
        cross_bear = (macd_line < self._signal_ema) and (prev_line >= prev_sig)
        zero_cross_up = (macd_line > 0) and (prev_line <= 0)
        zero_cross_down = (macd_line < 0) and (prev_line >= 0)

        # Save state for next bar
        self._prev_macd_line = macd_line
        self._prev_signal = self._signal_ema

        return {
            "mlv2_line":   macd_line,
            "mlv2_signal": self._signal_ema,
            "mlv2_hist":   macd_hist,
            "mlv2_cross_bull":      bool(cross_bull),
            "mlv2_cross_bear":      bool(cross_bear),
            "mlv2_zero_cross_up":   bool(zero_cross_up),
            "mlv2_zero_cross_down": bool(zero_cross_down),
        }
