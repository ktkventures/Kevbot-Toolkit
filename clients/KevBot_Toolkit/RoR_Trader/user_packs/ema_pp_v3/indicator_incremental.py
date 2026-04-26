"""EMA Price Position v3 — incremental indicator class.

Three EMAs + price-vs-EMA classification + four cross triggers
(price crossing short, price crossing mid). Uses L0-type level
columns: the cross level is the CURRENT bar's EMA value (no `_prev`
suffix in the column name). EMAs are slow-moving so L0 is
appropriate for these.

The level columns are emitted under bare names (no _prev). At
runtime, the engine reads the current bar's value directly.
"""


class EmaPpV3Incremental:
    def __init__(self, **params):
        self.short_period = int(params.get("short_period", 8))
        self.mid_period = int(params.get("mid_period", 21))
        self.long_period = int(params.get("long_period", 50))

        self._a_short = 2.0 / (self.short_period + 1)
        self._a_mid = 2.0 / (self.mid_period + 1)
        self._a_long = 2.0 / (self.long_period + 1)

        self._first = True
        self._ema_short = 0.0
        self._ema_mid = 0.0
        self._ema_long = 0.0

        # Previous-bar values for trigger detection
        self._prev_close = 0.0
        self._prev_short = 0.0
        self._prev_mid = 0.0
        self._prev_long = 0.0

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
            self._ema_short = close
            self._ema_mid = close
            self._ema_long = close
            self._first = False
        else:
            self._ema_short = self._a_short * close + (1.0 - self._a_short) * self._ema_short
            self._ema_mid = self._a_mid * close + (1.0 - self._a_mid) * self._ema_mid
            self._ema_long = self._a_long * close + (1.0 - self._a_long) * self._ema_long

        s, m = self._ema_short, self._ema_mid
        ps, pm = self._prev_short, self._prev_mid
        pc = self._prev_close

        # Bar-close triggers: price crossing each EMA
        cross_short_up   = (close > s) and (pc <= ps)
        cross_short_down = (close < s) and (pc >= ps)
        cross_mid_up     = (close > m) and (pc <= pm)
        cross_mid_down   = (close < m) and (pc >= pm)

        # Save state for next bar
        self._prev_close = close
        self._prev_short = s
        self._prev_mid = m
        self._prev_long = self._ema_long

        return {
            "eppv3_short": s,
            "eppv3_mid":   m,
            "eppv3_long":  self._ema_long,
            "eppv3_cross_short_up":   bool(cross_short_up),
            "eppv3_cross_short_down": bool(cross_short_down),
            "eppv3_cross_mid_up":     bool(cross_mid_up),
            "eppv3_cross_mid_down":   bool(cross_mid_down),
        }
