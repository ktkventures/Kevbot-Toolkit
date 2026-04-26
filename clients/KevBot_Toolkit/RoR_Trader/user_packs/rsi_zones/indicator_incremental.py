"""RSI Zones — incremental indicator class.

Wilder's RSI computed via running EWM (alpha = 1/period, adjust=False)
to match pandas's `.ewm(alpha=alpha, adjust=False).mean()` exactly.

Trigger booleans for all 12 zone-cross events are computed alongside
the RSI value and returned in the same dict so the engine's user-pack
pickup loop can read them directly.

No level columns: RSI is an oscillator value, not a price. Triggers
are bar-close events (the RSI value at bar close compared to the
previous bar's RSI vs threshold). Therefore only C and CC variants
get materialized by the registry.
"""


class RsiZonesIncremental:
    def __init__(self, **params):
        self.period = int(params.get("rsi_period", 14))
        self.extreme_ob = float(params.get("extreme_overbought_level", 80.0))
        self.ob = float(params.get("overbought_level", 70.0))
        self.midline = float(params.get("midline", 50.0))
        self.os = float(params.get("oversold_level", 30.0))
        self.extreme_os = float(params.get("extreme_oversold_level", 20.0))

        self._alpha = 1.0 / self.period

        # State
        self._prev_close = None
        self._avg_gain = None  # seeded by first valid delta
        self._avg_loss = None
        self._bar_count = 0     # for the "blank first period bars" rule
        self._prev_rsi = float("nan")

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
        self._bar_count += 1

        # ── RSI computation ──────────────────────────────────────────
        rsi = float("nan")
        if self._prev_close is not None:
            delta = close - self._prev_close
            gain = delta if delta > 0 else 0.0
            loss = -delta if delta < 0 else 0.0

            if self._avg_gain is None:
                # First valid delta seeds the EWM (matches pandas
                # adjust=False: y_0 = x_0 for the first non-NaN value).
                self._avg_gain = gain
                self._avg_loss = loss
            else:
                self._avg_gain = (1 - self._alpha) * self._avg_gain + self._alpha * gain
                self._avg_loss = (1 - self._alpha) * self._avg_loss + self._alpha * loss

            # Original batch blanks the first `period` bars to NaN.
            # bar_count is 1-indexed, so emit RSI starting at bar_count > period.
            if self._bar_count > self.period:
                if self._avg_loss == 0:
                    rsi = 100.0
                else:
                    rs = self._avg_gain / self._avg_loss
                    rsi = 100.0 - (100.0 / (1.0 + rs))

        self._prev_close = close

        # ── Trigger detection (zone crossings) ──────────────────────
        prev_rsi = self._prev_rsi
        both_valid = not _isnan(rsi) and not _isnan(prev_rsi)

        cross_into_ob = both_valid and rsi >= self.ob and prev_rsi < self.ob
        cross_into_os = both_valid and rsi <= self.os and prev_rsi > self.os
        exit_ob = both_valid and rsi < self.ob and prev_rsi >= self.ob
        exit_os = both_valid and rsi > self.os and prev_rsi <= self.os
        cross_into_extreme_ob = both_valid and rsi >= self.extreme_ob and prev_rsi < self.extreme_ob
        cross_into_extreme_os = both_valid and rsi <= self.extreme_os and prev_rsi > self.extreme_os
        cross_above_mid = both_valid and rsi > self.midline and prev_rsi <= self.midline
        cross_below_mid = both_valid and rsi < self.midline and prev_rsi >= self.midline

        # Save for next bar
        self._prev_rsi = rsi

        # The four "exit_*" triggers are aliases of the underlying zone
        # crosses with different semantic labels — the original pack
        # exposed them so users could pick the label that matches their
        # strategy's intent (long-side vs short-side). We mirror that.
        exit_long_overbought = cross_into_ob
        exit_short_oversold = cross_into_os
        exit_long_midline_down = cross_below_mid
        exit_short_midline_up = cross_above_mid

        return {
            "rsi_value": rsi,
            # Trigger booleans (engine reads via user-pack pickup loop)
            "rsi_cross_into_overbought":          bool(cross_into_ob),
            "rsi_cross_into_oversold":            bool(cross_into_os),
            "rsi_exit_overbought":                bool(exit_ob),
            "rsi_exit_oversold":                  bool(exit_os),
            "rsi_cross_into_extreme_overbought":  bool(cross_into_extreme_ob),
            "rsi_cross_into_extreme_oversold":    bool(cross_into_extreme_os),
            "rsi_cross_above_midline":            bool(cross_above_mid),
            "rsi_cross_below_midline":            bool(cross_below_mid),
            "rsi_exit_long_overbought":           bool(exit_long_overbought),
            "rsi_exit_short_oversold":            bool(exit_short_oversold),
            "rsi_exit_long_midline_down":         bool(exit_long_midline_down),
            "rsi_exit_short_midline_up":          bool(exit_short_midline_up),
        }


def _isnan(x):
    return x != x
