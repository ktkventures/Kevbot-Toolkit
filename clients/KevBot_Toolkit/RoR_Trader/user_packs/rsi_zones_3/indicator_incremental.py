"""RSI Zones 3 — incremental indicator class.

Same Wilder RSI core as the other RSI variants, but with an added
smoothing layer (SMA over `smoothing` bars) on top of the raw RSI.
The smoothed series (`rsi_z3_signal`) is what drives state
classification and triggers — that way the rendered line on chart
matches the trigger logic.

Note: original trigger_prefix in the manifest was `rsi`, which
collided with the `rsi_zones` pack's prefix. Renamed here to `rsi3`
so both can coexist at runtime without overwriting each other's
INTRABAR_LEVEL_MAP and INTERPRETERS entries.
"""


def _isnan(x):
    return x != x


class RsiZones3Incremental:
    def __init__(self, **params):
        self.period = int(params.get("rsi_period", 14))
        self.overbought = float(params.get("overbought_level", 70.0))
        self.oversold = float(params.get("oversold_level", 30.0))
        self.smoothing = int(params.get("smoothing", 1))

        self._alpha = 1.0 / self.period

        # Wilder RSI state
        self._prev_close = None
        self._avg_gain = None
        self._avg_loss = None

        # Smoothing window — rolling SMA over last `smoothing` raw RSI values
        self._rsi_window: list = []

        # Trigger detection state
        self._prev_signal = float("nan")

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

        # ── Wilder RSI ───────────────────────────────────────────────
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
                # Match pandas: when avg_loss is exactly 0, RSI is 100
                # (only if avg_gain > 0; otherwise NaN). The original used
                # `rs = NaN` when avg_loss is 0, making rsi NaN — but then
                # the .where() override didn't apply since the original
                # code used `replace(0, NaN)` which leaves rsi as NaN
                # whenever avg_loss is zero. We mirror that.
                rsi = float("nan")
            else:
                rs = self._avg_gain / self._avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))

        self._prev_close = close

        # ── SMA smoothing of RSI ────────────────────────────────────
        # Original batch uses `.rolling(smoothing, min_periods=smoothing).mean()`,
        # which yields NaN until the window is full. We mirror that.
        signal = float("nan")
        if not _isnan(rsi):
            self._rsi_window.append(rsi)
            if len(self._rsi_window) > self.smoothing:
                self._rsi_window.pop(0)

        if len(self._rsi_window) >= self.smoothing and self.smoothing >= 1:
            # All entries in _rsi_window are non-NaN by construction
            # (only appended above when rsi is valid). Mean is well-defined.
            signal = sum(self._rsi_window) / len(self._rsi_window)

        # ── Trigger detection on the SMOOTHED signal ────────────────
        prev_signal = self._prev_signal
        both_valid = (not _isnan(signal)) and (not _isnan(prev_signal))

        cross_into_ob = (
            both_valid and signal >= self.overbought and prev_signal < self.overbought
        )
        cross_into_os = (
            both_valid and signal <= self.oversold and prev_signal > self.oversold
        )
        exit_ob_zone = (
            both_valid and signal < self.overbought and prev_signal >= self.overbought
        )
        exit_os_zone = (
            both_valid and signal > self.oversold and prev_signal <= self.oversold
        )

        self._prev_signal = signal

        return {
            "rsi_z3": rsi,
            "rsi_z3_signal": signal,
            # Trigger booleans — prefixed with `rsi3_` to avoid collision
            # with the `rsi_zones` pack at runtime registration time.
            "rsi3_cross_into_overbought": bool(cross_into_ob),
            "rsi3_cross_into_oversold":   bool(cross_into_os),
            "rsi3_exit_overbought_zone":  bool(exit_ob_zone),
            "rsi3_exit_oversold_zone":    bool(exit_os_zone),
            "rsi3_overbought_long_exit":  bool(exit_ob_zone),  # alias
            "rsi3_oversold_short_exit":   bool(exit_os_zone),  # alias
        }
