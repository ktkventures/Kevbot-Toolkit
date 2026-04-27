"""UT Bot V4 — incremental indicator class.

Maintains the rolling state needed to compute the ATR trailing stop
one bar at a time. Mirrors the built-in utbot logic in
unified_engine._update_indicators (Wilder ATR + recursive trailing
stop) so the parity simulator can verify backtest-vs-live agreement.

The contract this class implements is documented in
pack_builder_context.md under "Live-Mode Incremental Class". For the
batch path, indicator.py is a thin wrapper that replays bars through
this same class — so by construction the two paths cannot diverge.
"""


class UtBotV4Incremental:
    def __init__(self, **params):
        # Same defaults as the manifest's parameters_schema
        self.atr_period = int(params.get("atr_period", 10))
        self.key_value = float(params.get("key_value", 1.0))
        self._alpha_w = 1.0 / self.atr_period  # Wilder smoothing factor

        # Rolling state — initialized on the first bar
        self._first = True
        self._atr = 0.0
        self._trail_stop = 0.0
        self._prev_close = 0.0
        self._prev_trail_stop = 0.0  # last bar's stop, exposed as `_prev`

    def update_bar(self, bar: dict) -> dict:
        """Process one bar; return new indicator + trigger values.

        Returns a dict whose keys are a subset of the manifest's
        ``indicator_columns`` plus the two trigger booleans
        (``utv4_bull_flip``, ``utv4_bear_flip``). The engine merges
        all of these into its `current` state dict.
        """
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])

        # The "_prev" column reflects whatever the engine cached at the
        # END of the previous bar. We expose the value we computed last
        # tick BEFORE updating to the new one — that's the one a
        # cross-detector should compare against on the new bar.
        prev_stop_for_output = self._prev_trail_stop

        if self._first:
            tr = high - low
            self._atr = tr
            self._trail_stop = close - self.key_value * tr
            bull_flip = False
            bear_flip = False
            # First-bar prev-close = current-close, matches built-in
            self._prev_close = close
            self._first = False
        else:
            # Wilder true range
            prev_c = self._prev_close
            tr = max(
                high - low,
                abs(high - prev_c),
                abs(low - prev_c),
            )
            # Wilder smoothing: ATR_n = ATR_{n-1} + (1/N) * (TR - ATR_{n-1})
            self._atr = self._atr + self._alpha_w * (tr - self._atr)

            n_loss = self.key_value * self._atr
            prev_stop = self._trail_stop

            # Pine UT Bot trailing-stop recursion (verbatim translation):
            #   if src > prev_stop AND src[1] > prev_stop:  max(prev_stop, src - nLoss)
            #   elif src < prev_stop AND src[1] < prev_stop: min(prev_stop, src + nLoss)
            #   elif src > prev_stop:                       src - nLoss
            #   else:                                       src + nLoss
            if close > prev_stop and prev_c > prev_stop:
                new_stop = max(prev_stop, close - n_loss)
            elif close < prev_stop and prev_c < prev_stop:
                new_stop = min(prev_stop, close + n_loss)
            elif close > prev_stop:
                new_stop = close - n_loss
            else:
                new_stop = close + n_loss

            self._trail_stop = new_stop

            # Trigger detection — Pine's `buy = src > xATRTrailingStop and crossover(src, xATRTrailingStop)`
            # Crossover collapses to: prev_close <= prev_stop AND close > new_stop.
            # (See indicator.py for the same logic on the batch path.)
            bull_flip = prev_c <= prev_stop and close > new_stop
            bear_flip = prev_c >= prev_stop and close < new_stop

            self._prev_close = close

        # Save current stop for next tick's `_prev` output
        self._prev_trail_stop = self._trail_stop

        return {
            "utv4_trailing_stop": self._trail_stop,
            "utv4_trailing_stop_prev": prev_stop_for_output,
            "utv4_bull_flip": bool(bull_flip),
            "utv4_bear_flip": bool(bear_flip),
            # Mirror the batch path's `__`-prefixed columns so the
            # interpreter (which reads `df["__utv4_bull_flip"]`) sees
            # the same data live as it does in backtest. Without this
            # alias, Q2/Q3 parity caps at ~0.87 because flip bars
            # mis-classify to BULL_TREND/BEAR_TREND in live mode.
            # Fixed 2026-04-27 after parity simulator surfaced the gap.
            "__utv4_bull_flip": bool(bull_flip),
            "__utv4_bear_flip": bool(bear_flip),
        }

    def warmup(self, df) -> None:
        """Bulk-seed state by replaying through update_bar."""
        for _, row in df.iterrows():
            self.update_bar({
                "open": row.get("open", 0.0),
                "high": row.get("high", 0.0),
                "low": row.get("low", 0.0),
                "close": row.get("close", 0.0),
                "volume": row.get("volume", 0.0),
            })
