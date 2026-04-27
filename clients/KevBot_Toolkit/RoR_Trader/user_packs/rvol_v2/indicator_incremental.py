"""RVOL v2 — incremental indicator class.

Relative Volume = current_volume / SMA(volume, period). Mirrors the
built-in `rvol` template's algorithm exactly:
- Maintain a rolling window of the last `vol_sma_period` volumes
- Until at least 5 volumes have been seen, emit rvol = 0 (matches the
  engine's existing behavior — see unified_engine._update_indicators)
- After 5+, emit rvol = volume / sma where sma = mean of buffer
- Three threshold-cross triggers: spike (cross above 1.5x), extreme
  (cross above 3.0x), fade (cross below 1.0x)

No level columns — the rvol value isn't a price level the engine could
detect intra-bar via high/low. So only C and CC variants get
materialized at runtime.
"""


class RvolV2Incremental:
    def __init__(self, **params):
        self.period = int(params.get("vol_sma_period", 20))
        self._spike_threshold = 1.5
        self._extreme_threshold = 3.0
        self._fade_threshold = 1.0
        self._min_bars_for_emit = 5

        self._buffer: list = []
        self._sum = 0.0
        self._count = 0
        self._prev_rvol = 0.0

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
        volume = float(bar["volume"])

        # O(1) rolling sum
        if len(self._buffer) == self.period:
            self._sum -= self._buffer.pop(0)
        self._buffer.append(volume)
        self._sum += volume
        self._count = min(self._count + 1, self.period)

        if self._count >= self._min_bars_for_emit:
            vol_sma = self._sum / len(self._buffer)
            rvol = volume / vol_sma if vol_sma > 0 else 0.0
        else:
            vol_sma = 0.0
            rvol = 0.0

        # Threshold-cross triggers (using prev_rvol vs current)
        prev = self._prev_rvol
        spike = rvol > self._spike_threshold and prev <= self._spike_threshold
        extreme = rvol > self._extreme_threshold and prev <= self._extreme_threshold
        fade = rvol < self._fade_threshold and prev >= self._fade_threshold

        self._prev_rvol = rvol

        return {
            "rv2_rvol": rvol,
            "rv2_vol_sma": vol_sma,
            "rv2_spike": bool(spike),
            "__rv2_spike": bool(spike),  # alias for interpreter
            "rv2_extreme": bool(extreme),
            "__rv2_extreme": bool(extreme),  # alias for interpreter
            "rv2_fade": bool(fade),
            "__rv2_fade": bool(fade),  # alias for interpreter
        }
