"""RSI Zones 2 — incremental indicator class.

Same Wilder RSI algorithm as `rsi_zones`, just with the rsi_zones_2
column name (`rsi2_rsi`) and trigger naming convention (`exit_*_zone`,
`exit_*_midline_cross_*`). Unlike rsi_zones, this variant does NOT
NaN-blank the first `period` bars — it emits the EWM value from the
first bar where there's a non-zero gain or loss.

Triggers are oscillator-value crosses (no price-level), so only C and
CC variants get materialized by the registry.
"""


class RsiZones2Incremental:
    def __init__(self, **params):
        self.period = int(params.get("rsi_period", 14))
        self.extreme_ob = float(params.get("extreme_overbought_level", 80.0))
        self.ob = float(params.get("overbought_level", 70.0))
        self.os = float(params.get("oversold_level", 30.0))
        self.extreme_os = float(params.get("extreme_oversold_level", 20.0))
        # midline is hardcoded to 50 in this variant (matches the
        # original interpreter; not exposed as a parameter)
        self.midline = 50.0

        self._alpha = 1.0 / self.period

        self._prev_close = None
        self._avg_gain = None
        self._avg_loss = None
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

        rsi = float("nan")
        if self._prev_close is not None:
            delta = close - self._prev_close
            gain = delta if delta > 0 else 0.0
            loss = -delta if delta < 0 else 0.0

            if self._avg_gain is None:
                self._avg_gain = gain
                self._avg_loss = loss
            else:
                self._avg_gain = (1 - self._alpha) * self._avg_gain + self._alpha * gain
                self._avg_loss = (1 - self._alpha) * self._avg_loss + self._alpha * loss

            if self._avg_loss == 0:
                rsi = 100.0
            else:
                rs = self._avg_gain / self._avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))

        self._prev_close = close

        # ── Trigger detection ───────────────────────────────────────
        prev_rsi = self._prev_rsi
        in_ob_now = (not _isnan(rsi)) and rsi >= self.ob
        in_ob_prev = (not _isnan(prev_rsi)) and prev_rsi >= self.ob
        in_extreme_ob_now = (not _isnan(rsi)) and rsi >= self.extreme_ob
        in_extreme_ob_prev = (not _isnan(prev_rsi)) and prev_rsi >= self.extreme_ob
        in_os_now = (not _isnan(rsi)) and rsi <= self.os
        in_os_prev = (not _isnan(prev_rsi)) and prev_rsi <= self.os
        in_extreme_os_now = (not _isnan(rsi)) and rsi <= self.extreme_os
        in_extreme_os_prev = (not _isnan(prev_rsi)) and prev_rsi <= self.extreme_os
        above_mid_now = (not _isnan(rsi)) and rsi > self.midline
        above_mid_prev = (not _isnan(prev_rsi)) and prev_rsi > self.midline

        cross_into_ob = in_ob_now and (not in_ob_prev)
        cross_into_extreme_ob = in_extreme_ob_now and (not in_extreme_ob_prev)
        cross_into_os = in_os_now and (not in_os_prev)
        cross_into_extreme_os = in_extreme_os_now and (not in_extreme_os_prev)
        exit_ob = (not in_ob_now) and in_ob_prev
        exit_os = (not in_os_now) and in_os_prev
        cross_above_mid = above_mid_now and (not above_mid_prev)
        cross_below_mid = (not above_mid_now) and above_mid_prev

        self._prev_rsi = rsi

        return {
            "rsi2_rsi": rsi,
            "rsi2_cross_into_overbought":         bool(cross_into_ob),
            "rsi2_cross_into_extreme_overbought": bool(cross_into_extreme_ob),
            "rsi2_cross_into_oversold":           bool(cross_into_os),
            "rsi2_cross_into_extreme_oversold":   bool(cross_into_extreme_os),
            "rsi2_exit_overbought_zone":          bool(exit_ob),
            "rsi2_exit_oversold_zone":            bool(exit_os),
            "rsi2_cross_above_midline":           bool(cross_above_mid),
            "rsi2_cross_below_midline":           bool(cross_below_mid),
            "rsi2_exit_long_overbought":          bool(cross_into_ob),
            "rsi2_exit_short_oversold":           bool(cross_into_os),
            "rsi2_exit_long_midline_cross_down":  bool(cross_below_mid),
            "rsi2_exit_short_midline_cross_up":   bool(cross_above_mid),
        }


def _isnan(x):
    return x != x
