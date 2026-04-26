"""EMA Stack v2 — incremental indicator class.

Three EMAs with alpha = 2/(N+1) on close. Six stack-order states
based on the relative ordering of S/M/L. Two cross-pair triggers
plus their mid-cross counterparts.

Critical fix vs the built-in `ema_stack`:
  The built-in's trigger logic at unified_engine.py:1119-1134 hardcodes
  the period names `ema_8`, `ema_21`, `ema_50` regardless of the
  user's confluence-group configuration. So if the group has periods
  [9, 21, 200], the live engine computes `ema_9` and `ema_200` but the
  trigger evaluator reads `ema_8` (returns 0) and the cross condition
  never fires. This pack uses the user's actual short/mid/long params
  by name, so the triggers fire correctly regardless of period choice.
"""


class EmaStackV2Incremental:
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

        # Previous-bar values for cross detection
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

        s, m, l_ = self._ema_short, self._ema_mid, self._ema_long
        ps, pm, pl = self._prev_short, self._prev_mid, self._prev_long

        # Triggers — unlike the built-in, these use the actual configured
        # periods via instance state rather than hardcoded ema_8/ema_50
        # column lookups.
        cross_bull = (s > m) and (ps <= pm)
        cross_bear = (s < m) and (ps >= pm)
        mid_cross_bull = (m > l_) and (pm <= pl)
        mid_cross_bear = (m < l_) and (pm >= pl)

        # Save state for next bar
        self._prev_short = s
        self._prev_mid = m
        self._prev_long = l_

        return {
            "esv2_short": s,
            "esv2_mid":   m,
            "esv2_long":  l_,
            "esv2_cross_bull":      bool(cross_bull),
            "esv2_cross_bear":      bool(cross_bear),
            "esv2_mid_cross_bull":  bool(mid_cross_bull),
            "esv2_mid_cross_bear":  bool(mid_cross_bear),
        }
