"""MACD Histogram v2 — incremental indicator class.

Same MACD core as macd_line_v2 (fast/slow/signal EMAs), exposing the
histogram value and four histogram-based triggers:

  - flip_pos / flip_neg: hist crosses zero
  - momentum_shift_up: hist > prev_hist AND prev_hist < prev2_hist
    (V-shape: was falling, now rising — change in momentum direction)
  - momentum_shift_down: hist < prev_hist AND prev_hist > prev2_hist
    (inverted-V: was rising, now falling)

The momentum_shift triggers need a 2-bar history of histogram values.

Note: this is a stand-alone implementation rather than a subclass of
macd_line_v2 because user packs are validated independently and can't
import each other's modules. Same algorithm, different prefix and
triggers.
"""


class MacdHistogramV2Incremental:
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

        # Histogram history for trigger detection
        self._prev_hist = 0.0
        self._prev2_hist = 0.0

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

        # Triggers — use prev_hist and prev2_hist
        prev_h = self._prev_hist
        prev2_h = self._prev2_hist

        flip_pos = (macd_hist > 0) and (prev_h <= 0)
        flip_neg = (macd_hist < 0) and (prev_h >= 0)
        momentum_up = (macd_hist > prev_h) and (prev_h < prev2_h)
        momentum_down = (macd_hist < prev_h) and (prev_h > prev2_h)

        # Shift histogram history forward
        self._prev2_hist = self._prev_hist
        self._prev_hist = macd_hist

        return {
            "mhv2_line":   macd_line,
            "mhv2_signal": self._signal_ema,
            "mhv2_hist":   macd_hist,
            "mhv2_flip_pos":             bool(flip_pos),
            "mhv2_flip_neg":             bool(flip_neg),
            "mhv2_momentum_shift_up":    bool(momentum_up),
            "mhv2_momentum_shift_down":  bool(momentum_down),
        }
