"""In-memory canonical per-symbol 1-second bar layer for the data-worker.

`SymbolBarStore` holds one rolling 1-second OHLCV DataFrame per symbol and
serves resampled timeframes on demand. It is the "shared bar layer" of
docs/Spec_Data_Worker_Service.md §3.1 — ingested once per symbol, every
consumer (Phase 2: streaming engines) subscribes and resamples.

Phase 1 scope: store the 1s series + resample. No engines, no DB.

Design notes:
- The series is **in-memory only** — REST is canonical and always
  re-fetchable, so on a data-worker restart the series is simply rebuilt
  from REST (spec §10.4 / §9). No DB table.
- `upsert_bars` is **last-write-wins**: a second re-fetched by the
  reconciliation sweep overwrites its earlier provisional value — that
  is how late-print corrections land (spec §10.2).
- A rolling-window `_trim` caps memory; the Phase 1 measurement gate
  depends on it actually bounding growth.
- DataFrame column shape matches `data_loader._polygon_bars_to_df` /
  `ralph_engine.BarBuilder.history`, so a Phase 2 streaming engine can
  consume `get_timeframe()` output directly.
"""
from __future__ import annotations

import threading

import pandas as pd

# OHLCV columns aggregated when resampling (kept if present on the frame).
_RESAMPLE_AGG = {
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
    'volume': 'sum', 'trade_count': 'sum',
}


class SymbolBarStore:
    """Rolling in-memory 1-second OHLCV series for a single symbol."""

    def __init__(self, symbol: str, window_minutes: int = 90):
        self.symbol = symbol
        self._window = pd.Timedelta(minutes=window_minutes)
        self._lock = threading.Lock()
        # Empty frame with the canonical column set + a UTC DatetimeIndex.
        self._bars = pd.DataFrame(
            columns=['open', 'high', 'low', 'close', 'volume',
                     'vwap', 'trade_count'])
        self._bars.index = pd.DatetimeIndex([], name='timestamp', tz='UTC')
        self._total_appended = 0
        self._total_revised = 0
        self._last_bar_ts = None

    def upsert_bars(self, df: pd.DataFrame) -> dict:
        """Merge a batch of 1-second bars into the series (last-write-wins).

        Re-fetched seconds overwrite their prior value — this is how the
        reconciliation sweep corrects provisional bars for late prints.

        Returns {appended, revised, total_bars} for instrumentation.
        """
        if df is None or len(df) == 0:
            with self._lock:
                return {'appended': 0, 'revised': 0,
                        'total_bars': len(self._bars)}

        # Defensive: a fetch could in theory return a duplicated second.
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

            # Skip the concat when prior is empty — concatenating an empty
            # frame triggers a pandas FutureWarning and is needless work.
            if len(prior) == 0:
                combined = df.copy()
            else:
                combined = pd.concat([prior, df])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined.sort_index(inplace=True)
            self._bars = combined
            self._trim()  # under the lock — see _trim docstring

            self._total_appended += appended
            self._total_revised += revised
            if len(self._bars) > 0:
                self._last_bar_ts = self._bars.index[-1]
            return {'appended': appended, 'revised': revised,
                    'total_bars': len(self._bars)}

    def get_1s(self, start=None, end=None) -> pd.DataFrame:
        """A copy of the 1-second series, optionally sliced to [start, end]."""
        with self._lock:
            bars = self._bars.copy()
        if start is not None:
            bars = bars[bars.index >= pd.Timestamp(start)]
        if end is not None:
            bars = bars[bars.index <= pd.Timestamp(end)]
        return bars

    def get_timeframe(self, tf: str) -> pd.DataFrame:
        """Resample the 1s series up to timeframe `tf` (e.g. '10Sec','1Min').

        '1Sec' returns the raw series. '1Min' is handled directly (it is
        absent from data_loader._RESAMPLE_RULES); coarser TFs delegate to
        data_loader.resample_to_timeframe.
        """
        with self._lock:
            bars = self._bars.copy()
        if len(bars) == 0 or tf in ('1Sec', '1s'):
            return bars
        if tf in ('1Min', '1min'):
            agg = {k: v for k, v in _RESAMPLE_AGG.items() if k in bars.columns}
            return bars.resample('1min').agg(agg).dropna(subset=['open'])
        from data_loader import resample_to_timeframe
        return resample_to_timeframe(bars, tf)

    def coverage(self) -> dict:
        """Cheap {first_ts, last_ts, count} — no memory_usage scan.

        The Phase 2 streaming tick calls this every cycle to confirm the
        store spans the strategy's resume window before running the
        engine; `stats()` is too heavy for that (it does a deep
        memory_usage scan). first_ts / last_ts are tz-aware Timestamps
        (or None when empty).
        """
        with self._lock:
            n = len(self._bars)
            if n == 0:
                return {'first_ts': None, 'last_ts': None, 'count': 0}
            return {'first_ts': self._bars.index[0],
                    'last_ts': self._bars.index[-1], 'count': n}

    def _trim(self) -> None:
        """Drop bars older than the rolling window. Caller must hold _lock."""
        if len(self._bars) == 0:
            return
        cutoff = self._bars.index[-1] - self._window
        self._bars = self._bars[self._bars.index >= cutoff]

    def stats(self) -> dict:
        """Snapshot of size / span / memory — feeds the Phase 1 metrics log."""
        with self._lock:
            n = len(self._bars)
            if n == 0:
                return {
                    'symbol': self.symbol, 'bar_count': 0,
                    'first_ts': None, 'last_ts': None, 'span_seconds': 0,
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
                'span_seconds': (last - first).total_seconds(),
                'mem_bytes': mem,
                'total_appended': self._total_appended,
                'total_revised': self._total_revised,
            }
