"""EMA Price Position v4 — incremental indicator class.

Same EMA algorithm and bar-close cross triggers as ema_pp_v3, but
uses L1-type level columns (`_prev` suffix). The engine adds the
1-bar lag automatically by caching the column at bar N's close and
using it on bar N+1's intra-bar reachability check.

Per the doc: emit the CURRENT bar's value under the `_prev`-named
column. Do NOT shift here — the engine handles the lag.

L1 vs L0 distinction matters for indicators where the line moves
during the signal bar in a way that would make the L0 fill price
unrealistic. For slow EMAs the difference is small; v4 is provided
for symmetry with the built-in ema_price_position_v2 template.
"""


class EmaPpV4Incremental:
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

        s, m, l_ = self._ema_short, self._ema_mid, self._ema_long
        ps, pm = self._prev_short, self._prev_mid
        pc = self._prev_close

        cross_short_up   = (close > s) and (pc <= ps)
        cross_short_down = (close < s) and (pc >= ps)
        cross_mid_up     = (close > m) and (pc <= pm)
        cross_mid_down   = (close < m) and (pc >= pm)

        # Save state for next bar
        self._prev_close = close
        self._prev_short = s
        self._prev_mid = m
        self._prev_long = l_

        # The `_prev`-named columns expose the CURRENT bar's EMA
        # values; the engine caches them at bar N's close and uses
        # them on bar N+1 for intra-bar reachability.
        return {
            "eppv4_short_prev": s,
            "eppv4_mid_prev":   m,
            "eppv4_long_prev":  l_,
            "eppv4_cross_short_up":   bool(cross_short_up),
            "eppv4_cross_short_down": bool(cross_short_down),
            "eppv4_cross_mid_up":     bool(cross_mid_up),
            "eppv4_cross_mid_down":   bool(cross_mid_down),
        }
