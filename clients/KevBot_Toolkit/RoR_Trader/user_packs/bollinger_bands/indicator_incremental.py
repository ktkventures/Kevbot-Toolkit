"""Bollinger Bands — incremental indicator class.

Maintains a rolling window of the last `period` closes and recomputes
SMA + sample stdev each bar. The batch indicator (`indicator.py`) is a
thin wrapper that replays bars through this same class, so backtest
and live cannot drift apart — the parity simulator verifies bit-
equivalence.

Trigger booleans (cross_upper / cross_lower / cross_basis_up /
cross_basis_down / squeeze_on / squeeze_off) are computed alongside
the indicator state and returned in the same dict.
"""


def _isnan(x):
    return x != x


class BollingerBandsIncremental:
    def __init__(self, **params):
        self.period = int(params.get("bb_period", 20))
        self.mult = float(params.get("bb_mult", 2.0))
        self.squeeze_threshold = float(params.get("squeeze_threshold", 0.04))

        # Rolling window of closes; trimmed to `period`.
        self._closes = []

        # Previous-bar values used for crossover detection on this bar.
        self._prev_close = None
        self._prev_upper = None
        self._prev_basis = None
        self._prev_lower = None
        self._prev_bw = None

        # Previous-bar level values exposed as `*_prev` columns. The engine
        # uses these as the L-type cross level (it adds the 1-bar lag
        # automatically by caching at bar N's close and applying on N+1).
        self._out_prev_upper = float("nan")
        self._out_prev_basis = float("nan")
        self._out_prev_lower = float("nan")

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

        # Capture values to expose as `_prev` BEFORE updating state.
        out_prev_upper = self._out_prev_upper
        out_prev_basis = self._out_prev_basis
        out_prev_lower = self._out_prev_lower

        # Append + trim rolling window
        self._closes.append(close)
        if len(self._closes) > self.period:
            self._closes.pop(0)

        # Pandas rolling default is min_periods=period, so the first
        # period-1 bars yield NaN. Match that exactly.
        if len(self._closes) < self.period:
            self._prev_close = close
            # No level yet — keep _prev outputs at their last (NaN) value
            return {
                "bb_upper": float("nan"),
                "bb_basis": float("nan"),
                "bb_lower": float("nan"),
                "bb_bandwidth": float("nan"),
                "bb_upper_prev": out_prev_upper,
                "bb_basis_prev": out_prev_basis,
                "bb_lower_prev": out_prev_lower,
                "bb_cross_upper": False,
                "__bb_cross_upper": False,  # alias for interpreter
                "bb_cross_lower": False,
                "__bb_cross_lower": False,  # alias for interpreter
                "bb_cross_basis_up": False,
                "__bb_cross_basis_up": False,  # alias for interpreter
                "bb_cross_basis_down": False,
                "__bb_cross_basis_down": False,  # alias for interpreter
                "bb_squeeze_on": False,
                "__bb_squeeze_on": False,  # alias for interpreter
                "bb_squeeze_off": False,
                "__bb_squeeze_off": False,  # alias for interpreter
            }

        # SMA + sample stdev (ddof=1) — matches pandas .rolling().std() default.
        n = len(self._closes)
        mean = sum(self._closes) / n
        var = sum((x - mean) ** 2 for x in self._closes) / (n - 1)
        std = var ** 0.5

        basis = mean
        upper = basis + self.mult * std
        lower = basis - self.mult * std
        bw = (upper - lower) / basis if basis > 0 else float("nan")

        # Trigger detection — uses the previous bar's close vs the
        # previous bar's bands (matches the pandas .shift(1) semantics
        # in the original batch trigger function).
        cross_upper = False
        cross_lower = False
        cross_basis_up = False
        cross_basis_down = False
        squeeze_on = False
        squeeze_off = False

        if (self._prev_close is not None
                and self._prev_upper is not None
                and self._prev_basis is not None
                and self._prev_lower is not None):
            cross_upper = (close > upper) and (self._prev_close <= self._prev_upper)
            cross_lower = (close < lower) and (self._prev_close >= self._prev_lower)
            cross_basis_up = (close > basis) and (self._prev_close <= self._prev_basis)
            cross_basis_down = (close < basis) and (self._prev_close >= self._prev_basis)

        if (self._prev_bw is not None
                and not _isnan(bw)
                and not _isnan(self._prev_bw)):
            squeeze_on = (bw < self.squeeze_threshold) and (self._prev_bw >= self.squeeze_threshold)
            squeeze_off = (bw >= self.squeeze_threshold) and (self._prev_bw < self.squeeze_threshold)

        # Save state for next bar
        self._prev_close = close
        self._prev_upper = upper
        self._prev_basis = basis
        self._prev_lower = lower
        self._prev_bw = bw
        # `_prev` outputs for the NEXT bar = the values we just computed
        self._out_prev_upper = upper
        self._out_prev_basis = basis
        self._out_prev_lower = lower

        return {
            "bb_upper": upper,
            "bb_basis": basis,
            "bb_lower": lower,
            "bb_bandwidth": bw,
            "bb_upper_prev": out_prev_upper,
            "bb_basis_prev": out_prev_basis,
            "bb_lower_prev": out_prev_lower,
            "bb_cross_upper": bool(cross_upper),
            "__bb_cross_upper": bool(cross_upper),  # alias for interpreter
            "bb_cross_lower": bool(cross_lower),
            "__bb_cross_lower": bool(cross_lower),  # alias for interpreter
            "bb_cross_basis_up": bool(cross_basis_up),
            "__bb_cross_basis_up": bool(cross_basis_up),  # alias for interpreter
            "bb_cross_basis_down": bool(cross_basis_down),
            "__bb_cross_basis_down": bool(cross_basis_down),  # alias for interpreter
            "bb_squeeze_on": bool(squeeze_on),
            "__bb_squeeze_on": bool(squeeze_on),  # alias for interpreter
            "bb_squeeze_off": bool(squeeze_off),
            "__bb_squeeze_off": bool(squeeze_off),  # alias for interpreter
        }
