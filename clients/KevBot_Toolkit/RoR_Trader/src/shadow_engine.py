"""M-RS4 Phase 3 — the continuous-resident backtest engine core (Step C).

`ResidentStrategyEngine` holds ONE live `UnifiedStrategy` in memory and applies each
new SETTLED bar incrementally via `process_bar` — NEVER serializing/re-windowing in
steady state. That is the byte-identity fix proven by `_resident_replay_harness.py`
(Step A): a single warmed engine fed bar-by-bar == a from-cold full recompute, whereas
the Data Worker's snapshot-resume tick (`data_worker_engine.run_store_fed_window`) drops
~4% at chunk boundaries. This module is the thing that supersedes that tick.

Design (Plan_M-RS4_Phase3_ContinuousBacktestEngine.md §2-3, decisions §6):
  - per-STRATEGY engine state (its own indicators/triggers/position/ATR/risk);
  - per-SYMBOL bar feed (the manager loads a symbol's bars once and fans them out);
  - bootstrap ONCE from a bounded from-cold window (warm the live engine — no snapshot,
    so byte-identical by construction), then stay resident;
  - serialization is for CRASH RECOVERY only (Step D), never the steady path.

The per-bar input construction (bar dict, MTF confluence records, pre-computed user-pack
columns) is lifted VERBATIM from `unified_engine.run_unified_backtest`'s loop so a
resident feed is identical to the one-shot loop. The Step A/PATH-D harness guards this:
if run_unified_backtest's loop body ever changes, the byte-identity test goes red.

This module is pure-compute: it does NO IO (no bar_cache reads, no DB writes). The
`ResidentEngineManager` (service integration, lands next) owns the bar_cache poll and the
`trades`-table writes; it calls `engine.feed(df_new)` and persists the returned trades.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from unified_engine import UnifiedStrategy, INTRABAR_LEVEL_MAP

# Built-in interpreters are computed incrementally by the engine; everything else
# referenced by a trigger arrives as a pre-computed DataFrame column. Mirrors the
# set in run_unified_backtest — keep in sync (the PATH-D harness enforces it).
_BUILTIN_INTERPS = {
    'EMA_STACK', 'EMA_PRICE_POSITION', 'EMA_PRICE_POSITION_V2',
    'MACD_LINE', 'MACD_HISTOGRAM', 'VWAP', 'RVOL', 'UTBOT', 'UTBOT_V2',
}


class ResidentStrategyEngine:
    """One strategy's resident engine: warmed once, fed new settled bars forever.

    Usage (the manager):
        eng = ResidentStrategyEngine(strat, enabled_general_packs)
        eng.feed(warmup_plus_history_df)      # bootstrap: warm live, returns its trades
        ...later, each poll...
        new_trades = eng.feed(new_settled_bars_df)   # only bars with ts > last_bar_ts

    `feed` accepts any df sharing the prepared-df schema (OHLCV + pre-computed
    indicator/interp/trigger/MTF columns from `prepare_data_with_indicators`). Bars at
    or before `last_bar_ts` are skipped, so the manager can hand overlapping windows
    safely (idempotent on the engine side). Returns the raw trade dicts process_bar
    emitted for the newly-applied bars (same shape run_unified_backtest collects).
    """

    def __init__(self, strategy: dict, general_packs=None):
        self.strategy = strategy
        self.engine = UnifiedStrategy(strategy, general_packs)
        self.last_bar_ts: Optional[pd.Timestamp] = None
        # Column detection is df-schema dependent; resolved on the first feed and
        # cached (the prepared-df schema is stable across a strategy's lifetime).
        self._cols_ready = False
        self._user_interp_cols: List[str] = []
        self._user_trig_cols: List[str] = []
        self._user_indicator_cols: set = set()

    def _detect_cols(self, df: pd.DataFrame) -> None:
        te = self.engine.trigger_eval
        self._user_interp_cols = [
            ik for ik in te.required_interpreters
            if ik not in _BUILTIN_INTERPS and ik in df.columns
        ]
        required_trig_set = {f'trig_{t}' for t in te.required_triggers}
        self._user_trig_cols = [
            col for col in df.columns
            if col in required_trig_set and col.startswith('trig_')
        ]
        self._user_indicator_cols = set()
        for base_trigger in te.required_triggers:
            for suffix in ('_ib', '_lc', '_cc', '_hm', '_hl'):
                if base_trigger.endswith(suffix):
                    base = base_trigger[:-len(suffix)]
                    if base in INTRABAR_LEVEL_MAP:
                        level_col = INTRABAR_LEVEL_MAP[base].get('column', '')
                        if level_col and level_col in df.columns:
                            self._user_indicator_cols.add(level_col)
                    break
        self._cols_ready = True

    def _build_inputs(self, row, secondary_tf_map):
        """The bar dict + mtf_records + user_pack_data for one row — verbatim from
        run_unified_backtest's per-bar body."""
        bar = {
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row.get('volume', 0)),
            # timestamp filled by caller (it owns the df index)
        }
        mtf_records = None
        if secondary_tf_map:
            mtf_set = set()
            for tf_label, cols in secondary_tf_map.items():
                for col in cols:
                    val = row.get(col)
                    if val is not None and pd.notna(val):
                        base_interp = col.rsplit('__', 1)[0]
                        mtf_set.add(f"{tf_label}-{base_interp}-{val}")
            if mtf_set:
                mtf_records = mtf_set

        user_pack_data = None
        if self._user_interp_cols or self._user_trig_cols or self._user_indicator_cols:
            up_interps, up_triggers, up_indicators = {}, {}, {}
            for col in self._user_interp_cols:
                val = row.get(col)
                if val is not None and pd.notna(val):
                    up_interps[col] = str(val)
            for col in self._user_trig_cols:
                val = row.get(col)
                up_triggers[col[5:]] = bool(val) if pd.notna(val) else False
            for col in self._user_indicator_cols:
                val = row.get(col)
                if val is not None and pd.notna(val):
                    up_indicators[col] = float(val)
            if up_interps or up_triggers or up_indicators:
                user_pack_data = {'interps': up_interps, 'triggers': up_triggers,
                                  'indicators': up_indicators}
        return bar, mtf_records, user_pack_data

    def feed(self, df: pd.DataFrame, secondary_tf_map: Optional[dict] = None) -> List[dict]:
        """Apply every bar in `df` with index strictly after `last_bar_ts`.

        Returns the trade dicts process_bar emitted across those bars (closed and/or
        opened, same shape as run_unified_backtest's internal `trades` list). Advances
        `last_bar_ts` to the last applied bar.
        """
        if df is None or len(df) == 0:
            return []
        if not self._cols_ready:
            self._detect_cols(df)

        if self.last_bar_ts is not None:
            df = df[df.index > self.last_bar_ts]
            if len(df) == 0:
                return []

        out: List[dict] = []
        idx = df.index
        for i in range(len(df)):
            row = df.iloc[i]
            bar, mtf_records, user_pack_data = self._build_inputs(row, secondary_tf_map)
            bar['timestamp'] = idx[i]
            bar_trades, _ind, _interp, _trig = self.engine.process_bar(
                bar, mtf_records=mtf_records, partial=False,
                user_pack_data=user_pack_data)
            if bar_trades:
                out.extend(bar_trades)
        self.last_bar_ts = idx[-1]
        return out

    @property
    def in_position(self) -> bool:
        return self.engine.position.state.status == 'IN_POSITION'
