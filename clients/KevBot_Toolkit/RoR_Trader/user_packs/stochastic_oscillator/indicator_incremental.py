"""Stochastic Oscillator — incremental indicator class.

Three-stage rolling computation:
  1. raw_k = (close - lowest_low_over_k_period) / (highest_high - lowest_low) * 100
  2. stoch_k = SMA of raw_k over k_smooth bars
  3. stoch_d = SMA of stoch_k over d_period bars

Maintains three small ring buffers (highs, lows, raw_ks, stoch_ks)
sized by the relevant lookback window. Pandas `.rolling().min()` and
`.rolling().mean()` return NaN until each window is full — mirrored
exactly here.

Triggers are oscillator-value events (no price level), so only C and
CC variants get materialized by the registry.
"""


def _isnan(x):
    return x != x


class StochasticOscillatorIncremental:
    def __init__(self, **params):
        self.k_period = int(params.get("k_period", 14))
        self.k_smooth = int(params.get("k_smooth", 3))
        self.d_period = int(params.get("d_period", 3))
        self.overbought = float(params.get("overbought", 80.0))
        self.oversold = float(params.get("oversold", 20.0))
        self.midline = float(params.get("midline", 50.0))

        # Buffers
        self._highs: list = []
        self._lows: list = []
        self._raw_k_buf: list = []
        self._stoch_k_buf: list = []

        # Trigger detection state
        self._prev_k = float("nan")
        self._prev_d = float("nan")

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

        # Append + trim high/low buffers (raw_k window = k_period)
        self._highs.append(high)
        self._lows.append(low)
        if len(self._highs) > self.k_period:
            self._highs.pop(0)
            self._lows.pop(0)

        # Raw %K — only valid once we have k_period bars in the window
        raw_k = float("nan")
        if len(self._highs) >= self.k_period:
            ll = min(self._lows)
            hh = max(self._highs)
            hl_range = hh - ll
            if hl_range == 0:
                raw_k = 50.0  # neutral when range is zero (matches batch)
            else:
                raw_k = (close - ll) / hl_range * 100.0

        # Append raw_k to its buffer (only when valid — pandas rolling on
        # a NaN-leading series propagates NaN, but the resulting SMA only
        # becomes non-NaN once k_smooth consecutive non-NaN values exist).
        if not _isnan(raw_k):
            self._raw_k_buf.append(raw_k)
            if len(self._raw_k_buf) > self.k_smooth:
                self._raw_k_buf.pop(0)

        # stoch_k = SMA of last k_smooth raw_k values
        stoch_k = float("nan")
        if len(self._raw_k_buf) >= self.k_smooth:
            stoch_k = sum(self._raw_k_buf) / len(self._raw_k_buf)

        # Append stoch_k to its buffer
        if not _isnan(stoch_k):
            self._stoch_k_buf.append(stoch_k)
            if len(self._stoch_k_buf) > self.d_period:
                self._stoch_k_buf.pop(0)

        # stoch_d = SMA of last d_period stoch_k values
        stoch_d = float("nan")
        if len(self._stoch_k_buf) >= self.d_period:
            stoch_d = sum(self._stoch_k_buf) / len(self._stoch_k_buf)

        # ── Trigger detection ───────────────────────────────────────
        prev_k = self._prev_k
        prev_d = self._prev_d
        kd_valid = (not _isnan(stoch_k)) and (not _isnan(stoch_d)) \
            and (not _isnan(prev_k)) and (not _isnan(prev_d))

        # Zone membership
        in_oversold = (not _isnan(stoch_k)) and stoch_k <= self.oversold
        in_overbought = (not _isnan(stoch_k)) and stoch_k >= self.overbought
        in_midrange = (not _isnan(stoch_k)) and stoch_k > self.oversold and stoch_k < self.overbought
        in_oversold_prev = (not _isnan(prev_k)) and prev_k <= self.oversold
        in_overbought_prev = (not _isnan(prev_k)) and prev_k >= self.overbought

        # K crosses D
        k_above_d = kd_valid and (stoch_k > stoch_d) and (prev_k <= prev_d)
        k_below_d = kd_valid and (stoch_k < stoch_d) and (prev_k >= prev_d)

        # Midline crosses
        k_valid = (not _isnan(stoch_k)) and (not _isnan(prev_k))
        cross_above_mid = k_valid and (stoch_k > self.midline) and (prev_k <= self.midline)
        cross_below_mid = k_valid and (stoch_k < self.midline) and (prev_k >= self.midline)

        triggers = {
            "sto_k_cross_above_d_oversold":   bool(k_above_d and in_oversold),
            "sto_k_cross_below_d_overbought": bool(k_below_d and in_overbought),
            "sto_k_cross_above_d_midrange":   bool(k_above_d and in_midrange),
            "sto_k_cross_below_d_midrange":   bool(k_below_d and in_midrange),
            "sto_enter_overbought":           bool(in_overbought and not in_overbought_prev),
            "sto_exit_overbought":            bool((not in_overbought) and in_overbought_prev),
            "sto_enter_oversold":             bool(in_oversold and not in_oversold_prev),
            "sto_exit_oversold":              bool((not in_oversold) and in_oversold_prev),
            "sto_k_cross_above_midline":      bool(cross_above_mid),
            "sto_k_cross_below_midline":      bool(cross_below_mid),
        }

        # Update state
        self._prev_k = stoch_k
        self._prev_d = stoch_d

        return {
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            **triggers,
        }
