"""Tier-2 coarse-bar (1-Minute) store for the data-worker.

`CoarseBarStore` is the long-window companion to `SymbolBarStore`
(Tier-1, 90-min 1-second). It holds a rolling **150-day series of
1-Minute OHLCV** per symbol so cross-TF confluence indicators
(15Min/1Hour/1Day secondaries) have enough warmup to converge — Tier-1
holds only 90 minutes, which is structurally insufficient for any
secondary TF ≥ 2Min.

Spec: `docs/Spec_Data_Worker_Service.md` + the Phase 2.5 plan
(`~/.claude/plans/fluttering-wondering-comet.md`).

Design notes:
- In-memory only; rebuilt from Polygon REST on data-worker restart
  (mirrors Tier-1's no-persistence stance, spec §10.4).
- `upsert_bars` is **last-write-wins** so the per-cycle refresh's late
  prints overwrite the prior provisional 1Min bar.
- Trim by a calendar-days window (vs Tier-1's minutes window).
- DataFrame shape matches Tier-1 / `data_loader._polygon_bars_to_df` so
  `data_loader.resample_to_timeframe` consumes it directly.
"""
from __future__ import annotations

import threading

import pandas as pd

_RESAMPLE_AGG = {
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
    'volume': 'sum', 'trade_count': 'sum',
}


class CoarseBarStore:
    """Rolling in-memory 1-Minute OHLCV series for a single symbol."""

    def __init__(self, symbol: str, window_days: int = 150):
        self.symbol = symbol
        self._window = pd.Timedelta(days=window_days)
        self._lock = threading.Lock()
        self._bars = pd.DataFrame(
            columns=['open', 'high', 'low', 'close', 'volume',
                     'vwap', 'trade_count'])
        self._bars.index = pd.DatetimeIndex([], name='timestamp', tz='UTC')
        self._total_appended = 0
        self._total_revised = 0
        self._last_bar_ts = None

    def upsert_bars(self, df: pd.DataFrame) -> dict:
        """Merge a batch of 1-Minute bars into the series (last-write-wins).

        Re-fetched minutes overwrite their prior value — that is how the
        per-cycle refresh corrects provisional 1Min bars for late prints.

        Returns {appended, revised, total_bars} for instrumentation.
        """
        if df is None or len(df) == 0:
            with self._lock:
                return {'appended': 0, 'revised': 0,
                        'total_bars': len(self._bars)}

        df = df[~df.index.duplicated(keep='last')]

        with self._lock:
            prior = self._bars
            appended = 0
            revised = 0
            if len(prior) > 0:
                overlap = df.index.intersection(prior.index)
                for ts in overlap:
                    old = prior.at[ts, 'close']
                    new = df.at[ts, 'close']
                    if round(float(old), 6) != round(float(new), 6):
                        revised += 1
                appended = len(df.index.difference(prior.index))
            else:
                appended = len(df)

            if len(prior) == 0:
                combined = df.copy()
            else:
                combined = pd.concat([prior, df])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined.sort_index(inplace=True)
            self._bars = combined
            self._trim()

            self._total_appended += appended
            self._total_revised += revised
            if len(self._bars) > 0:
                self._last_bar_ts = self._bars.index[-1]
            return {'appended': appended, 'revised': revised,
                    'total_bars': len(self._bars)}

    def get_1min(self, start=None, end=None) -> pd.DataFrame:
        """A copy of the 1-Minute series, optionally sliced to [start, end]."""
        with self._lock:
            bars = self._bars.copy()
        if start is not None:
            bars = bars[bars.index >= pd.Timestamp(start)]
        if end is not None:
            bars = bars[bars.index <= pd.Timestamp(end)]
        return bars

    def get_timeframe(self, tf: str) -> pd.DataFrame:
        """Resample the 1-Minute series up to `tf` (e.g. '15Min','1Day').

        '1Min' returns the raw series. Coarser TFs delegate to
        `data_loader.resample_to_timeframe`. Sub-minute TFs are nonsensical
        here (caller should route to Tier-1); returns the raw 1Min frame
        defensively rather than raising.
        """
        with self._lock:
            bars = self._bars.copy()
        if len(bars) == 0 or tf in ('1Min', '1min'):
            return bars
        from data_loader import resample_to_timeframe
        return resample_to_timeframe(bars, tf)

    def _trim(self) -> None:
        """Drop bars older than the rolling window. Caller must hold _lock."""
        if len(self._bars) == 0:
            return
        cutoff = self._bars.index[-1] - self._window
        self._bars = self._bars[self._bars.index >= cutoff]

    def coverage(self) -> dict:
        """Cheap {first_ts, last_ts, count} — no memory_usage scan.

        first_ts / last_ts are tz-aware Timestamps (or None when empty).
        Mirrors `SymbolBarStore.coverage()`.
        """
        with self._lock:
            n = len(self._bars)
            if n == 0:
                return {'first_ts': None, 'last_ts': None, 'count': 0}
            return {'first_ts': self._bars.index[0],
                    'last_ts': self._bars.index[-1], 'count': n}

    def stats(self) -> dict:
        """Snapshot of size / span / memory — feeds the metrics log."""
        with self._lock:
            n = len(self._bars)
            if n == 0:
                return {
                    'symbol': self.symbol, 'bar_count': 0,
                    'first_ts': None, 'last_ts': None, 'span_days': 0,
                    'mem_bytes': 0,
                    'total_appended': self._total_appended,
                    'total_revised': self._total_revised,
                }
            first = self._bars.index[0]
            last = self._bars.index[-1]
            mem = int(self._bars.memory_usage(deep=True).sum())
            return {
                'symbol': self.symbol, 'bar_count': n,
                'first_ts': first.isoformat(), 'last_ts': last.isoformat(),
                'span_days': round((last - first).total_seconds() / 86400, 1),
                'mem_bytes': mem,
                'total_appended': self._total_appended,
                'total_revised': self._total_revised,
            }
