"""VWAP v2 — incremental indicator class.

Cumulative volume-weighted average price with ±1σ and ±2σ bands.
Mirrors the built-in `vwap` template's algorithm exactly, including
the 30-minute session-gap reset (resets cumulative state when bar
timestamps span a >30-minute gap, which signals a new RTH session).

Triggers:
- cross_above / cross_below: price crosses the VWAP line
- enter_upper_extreme / enter_lower_extreme: price crosses the ±2σ bands
- return_to_vwap: price was outside ±1σ on the previous bar and is now
  within ±0.5σ — composite condition that doesn't fit a simple level
  cross, so no L-type variant for it.

L-type level columns: VWAP and its bands move slowly bar-to-bar, so
they're L0-type (current bar's level, no `_prev` suffix in the column
name). The engine handles them via INTRABAR_LEVEL_MAP using the
column's current value.
"""

import math

import pandas as pd


_VWAP_SESSION_GAP_SECONDS = 30 * 60  # 30 minutes — matches built-in


def _to_epoch(ts):
    """Convert a timestamp to UNIX epoch float.

    Returns None if ts is None. The batch wrapper passes a pandas
    Timestamp / datetime (`.timestamp()` works directly), but the live
    engine bar path passes an ISO string — coerce that via pd.Timestamp,
    mirroring the built-in vwap (unified_engine line ~913).

    Note: the pack sandbox (pack_spec.SAFE_BUILTINS) does not expose
    `hasattr` / `getattr` / `AttributeError`, so type-test with the
    allowed `isinstance` rather than duck-typing on `.timestamp`.
    """
    if ts is None:
        return None
    if isinstance(ts, str):
        try:
            return pd.Timestamp(ts).timestamp()
        except (ValueError, TypeError):
            return None
    return ts.timestamp()


class VwapV2Incremental:
    def __init__(self, **params):
        self.sd1_mult = float(params.get("sd1_mult", 1.0))
        self.sd2_mult = float(params.get("sd2_mult", 2.0))

        # Cumulative aggregates for the current session.
        self._cum_pv = 0.0
        self._cum_vol = 0.0
        self._cum_sq_dev_vol = 0.0
        self._prev_ts = None
        self._first = True

        # Previous-bar values for trigger detection
        self._prev_close = None
        self._prev_vwap = float("nan")
        self._prev_sd1_upper = float("nan")
        self._prev_sd1_lower = float("nan")
        self._prev_sd2_upper = float("nan")
        self._prev_sd2_lower = float("nan")

    def warmup(self, df) -> None:
        for ts, row in df.iterrows():
            self.update_bar({
                "open": row.get("open", 0.0),
                "high": row.get("high", 0.0),
                "low": row.get("low", 0.0),
                "close": row.get("close", 0.0),
                "volume": row.get("volume", 0.0),
                "timestamp": ts,
            })

    def update_bar(self, bar: dict) -> dict:
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        volume = float(bar["volume"])
        ts = bar.get("timestamp")

        tp = (high + low + close) / 3.0

        # Session-gap detection: reset cumulative state if >30 min from prev bar.
        if self._prev_ts is not None and ts is not None:
            now_e = _to_epoch(ts)
            prev_e = _to_epoch(self._prev_ts)
            if now_e is not None and prev_e is not None:
                if (now_e - prev_e) > _VWAP_SESSION_GAP_SECONDS:
                    self._cum_pv = 0.0
                    self._cum_vol = 0.0
                    self._cum_sq_dev_vol = 0.0

        if self._first:
            self._cum_pv = 0.0
            self._cum_vol = 0.0
            self._cum_sq_dev_vol = 0.0
            self._first = False

        self._cum_pv += tp * volume
        self._cum_vol += volume

        if self._cum_vol > 0:
            vwap = self._cum_pv / self._cum_vol
            sq_dev = volume * (tp - vwap) ** 2
            self._cum_sq_dev_vol += sq_dev
            std = math.sqrt(self._cum_sq_dev_vol / self._cum_vol)
        else:
            vwap = close
            std = 0.0

        sd1_upper = vwap + self.sd1_mult * std
        sd1_lower = vwap - self.sd1_mult * std
        sd2_upper = vwap + self.sd2_mult * std
        sd2_lower = vwap - self.sd2_mult * std

        self._prev_ts = ts

        # Trigger detection — use previous bar's values
        prev_close = self._prev_close
        prev_vwap = self._prev_vwap
        prev_sd2u = self._prev_sd2_upper
        prev_sd2l = self._prev_sd2_lower
        prev_sd1u = self._prev_sd1_upper
        prev_sd1l = self._prev_sd1_lower

        prev_valid = (
            prev_close is not None
            and not _isnan(prev_vwap) and prev_vwap != 0
        )

        cross_above = prev_valid and close > vwap and prev_close <= prev_vwap
        cross_below = prev_valid and close < vwap and prev_close >= prev_vwap

        enter_upper_extreme = (
            prev_valid
            and not _isnan(prev_sd2u)
            and close > sd2_upper and prev_close <= prev_sd2u
        )
        enter_lower_extreme = (
            prev_valid
            and not _isnan(prev_sd2l)
            and close < sd2_lower and prev_close >= prev_sd2l
        )

        # return_to_vwap: previous close was outside ±1σ AND current
        # close is within ±0.5σ (the @V zone). Half-σ is computed from
        # current bands.
        return_to_vwap = False
        if prev_valid and not _isnan(prev_sd1u) and not _isnan(prev_sd1l):
            half_sd = (sd1_upper - vwap) * 0.5
            at_upper = vwap + half_sd
            at_lower = vwap - half_sd
            was_extreme = (prev_close > prev_sd1u) or (prev_close < prev_sd1l)
            now_at_vwap = at_lower <= close <= at_upper
            return_to_vwap = was_extreme and now_at_vwap

        # Save state for next bar
        self._prev_close = close
        self._prev_vwap = vwap
        self._prev_sd1_upper = sd1_upper
        self._prev_sd1_lower = sd1_lower
        self._prev_sd2_upper = sd2_upper
        self._prev_sd2_lower = sd2_lower

        return {
            "vwapv2_value": vwap,
            "vwapv2_sd1_upper": sd1_upper,
            "vwapv2_sd1_lower": sd1_lower,
            "vwapv2_sd2_upper": sd2_upper,
            "vwapv2_sd2_lower": sd2_lower,
            "vwapv2_cross_above":         bool(cross_above),
            "vwapv2_cross_below":         bool(cross_below),
            "vwapv2_enter_upper_extreme": bool(enter_upper_extreme),
            "vwapv2_enter_lower_extreme": bool(enter_lower_extreme),
            "vwapv2_return_to_vwap":      bool(return_to_vwap),
        }


def _isnan(x):
    return x != x
