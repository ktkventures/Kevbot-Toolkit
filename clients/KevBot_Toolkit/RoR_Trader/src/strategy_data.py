"""Unified strategy data loader — anchors every read path to the
strategy's `backtest_start_date` visible window and applies a consistent
warmup floor.

Why this exists: prior to this module, eight different read paths
(get_strategy_trades, prepare_forward_test_data, Mass Builder data load,
append_new_backtest_trades cold-start, _do_recompute, chart loader,
streaming engine, Tier-2 coarse store) each computed their own window
and warmup independently — `*2` multiplier here, 100 bars there, 365
days for charts, snapshot resume in streaming, etc. The result: Mass
Builder preview KPIs diverged from Update All Data, and cross-TF
strategies could be dark for days after a cold-start.

This module is the single point of truth for:
- **Visible window**: the user-facing time range — anchored at
  `strategy.backtest_start_date` (set at save time). Fixed start,
  growing end. KPIs / trades / charts trim to this window.
- **Warmup buffer**: extra data loaded BEFORE the visible window so
  indicators have time to converge before the first trade. Floor is
  `visible_days * WARMUP_MULTIPLIER` (v1); a v1+1 enhancement will
  union with `longest_indicator_lookback_days` for precision.

Migrate any new data-loading code through here. See
`docs/Audit_Warmup_Window_Alignment.md` for the audit that motivated
this and `/home/kevin/.claude/plans/piped-wondering-tower.md` for the
shipped scope.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants (tunable; see plan: piped-wondering-tower.md) ───────────

WARMUP_MULTIPLIER = 2
"""Warmup buffer = visible_days * this. v1 floor; v1+1 will union with
indicator-derived longest lookback. Keeping `*2` matches the prior
behavior of `prepare_forward_test_data:834` so live KPIs don't shift
out from under existing strategies."""

LEGACY_FALLBACK_DAYS = 90
"""Visible-window length for strategies that don't have
`backtest_start_date` set (e.g. created via the API endpoint before
this field was stamped). Resolved as `forward_test_start - this`."""

DEFAULT_DATA_DAYS = 90
"""Last-resort default when a strategy has neither `backtest_start_date`
nor `forward_test_start` (very legacy / synthetic strategies)."""


# ── Bundle ───────────────────────────────────────────────────────────

@dataclass
class StrategyDataBundle:
    """Result of `load_strategy_data`. Carries the loaded df + the
    visible-window anchor so callers can trim deterministically."""

    df: pd.DataFrame                  # full loaded data (warmup + visible)
    visible_start: pd.Timestamp       # the anchor — trim point for trades/KPIs
    visible_end: pd.Timestamp         # typically now
    warmup_start: pd.Timestamp        # earliest bar loaded
    secondary_tf_map: dict            # standard MTF map (from get_secondary_tf_map)
    anchor_source: str                # how visible_start was resolved
    warmup_days: float                # the warmup window size (for logging/diag)


# ── Window resolution ────────────────────────────────────────────────

def _parse_iso_to_tz_aware(s) -> Optional[pd.Timestamp]:
    if s is None or s == "":
        return None
    try:
        ts = pd.Timestamp(s)
    except (ValueError, TypeError):
        return None
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts


def resolve_visible_window(
    strat: dict,
) -> Tuple[pd.Timestamp, pd.Timestamp, str]:
    """Resolve (visible_start, visible_end, anchor_source).

    Resolution order:
      1. `lookback_mode='Date Range'` + `lookback_start_date` →
         user-pinned [lookback_start, lookback_end]. Visible_end is
         the user's lookback_end, not now.
      2. `strategy.backtest_start_date` (preferred for the common case;
         set at save time by Mass Builder / Streamlit save /
         API create_strategy)
      3. `forward_test_start - LEGACY_FALLBACK_DAYS days` (legacy
         strategies that pre-date `backtest_start_date`)
      4. `now - data_days` (last-resort; strategies with neither anchor)
    """
    now = pd.Timestamp.now(tz="UTC")

    # 1. Explicit date-range lookback — user-pinned window
    if strat.get("lookback_mode") == "Date Range":
        ls = _parse_iso_to_tz_aware(strat.get("lookback_start_date"))
        le = _parse_iso_to_tz_aware(strat.get("lookback_end_date"))
        if ls is not None and le is not None:
            return ls, le, "lookback_range"

    # 2. Mass-Builder / Streamlit / API-stamped backtest_start_date
    bsd = _parse_iso_to_tz_aware(strat.get("backtest_start_date"))
    if bsd is not None:
        return bsd, now, "backtest_start_date"

    # 3. Legacy strategies — derive from forward_test_start
    fts = _parse_iso_to_tz_aware(strat.get("forward_test_start"))
    if fts is not None:
        return (fts - pd.Timedelta(days=LEGACY_FALLBACK_DAYS),
                now,
                "forward_test_start_fallback")

    # 4. Final fallback — relative to now
    data_days = int(strat.get("data_days") or DEFAULT_DATA_DAYS)
    return (now - pd.Timedelta(days=data_days),
            now,
            "data_days_fallback")


def compute_warmup_days(strat: dict, visible_days: float) -> float:
    """Warmup buffer in days. v1: `visible_days * WARMUP_MULTIPLIER`.

    TODO v1+1: union with indicator-derived longest lookback. Walk the
    strategy's `resolve_strategy_requirements` to extract longest period
    (200-bar EMA on 1Day → 200 calendar days needed), then return
    `max(visible_days * WARMUP_MULTIPLIER, longest_indicator_lookback)`.
    See unified_engine.py:430-622 (resolve_strategy_requirements) +
    unified_engine.py:291-335 (_INDICATOR_PARAM_SPEC) for the hook.
    """
    _ = strat  # unused in v1; reserved for v1+1
    return max(1.0, float(visible_days) * float(WARMUP_MULTIPLIER))


# ── Loader ────────────────────────────────────────────────────────────

def load_strategy_data(
    strat: dict,
    *,
    data_feed: str = "sip",
    model_override: Optional[str] = None,
    secondary_tfs_override: Optional[tuple] = None,
    required_confluence_ids: Optional[set] = None,
    force_scope: bool = False,
) -> StrategyDataBundle:
    """Resolve window, compute warmup, load bars, return bundle.

    Calls `services.prepare_data_with_indicators` internally with the
    extended `[warmup_start, visible_end]` window. The bundle carries
    `visible_start` so callers can trim trades / equity-curve / charts
    consistently.

    `secondary_tfs_override`: explicit list of secondary TFs to compute,
    bypassing the auto-derivation from `strat['confluence']`. Used by
    Mass Builder (loads MTF data for ALL candidates' potential
    confluences, not just one strategy's).

    Emits an INFO log line per call so we can audit which strategies
    are hitting the legacy fallbacks vs the primary anchor.
    """
    # Local import — services.py imports indirectly from here in places,
    # so keep the import lazy to avoid circulars.
    from services import prepare_data_with_indicators, get_secondary_tf_map
    from data_loader import (
        get_required_tfs_from_confluence, get_tf_from_label,
    )

    visible_start, visible_end, anchor_source = resolve_visible_window(strat)
    visible_days = max(
        1.0, (visible_end - visible_start).total_seconds() / 86400.0)
    warmup_days = compute_warmup_days(strat, visible_days)
    warmup_start = visible_start - pd.Timedelta(days=warmup_days)

    if secondary_tfs_override is not None:
        secondary_tfs = tuple(secondary_tfs_override)
    else:
        # Resolve secondary TFs from the strategy's confluence list (same
        # convention used by services.get_strategy_trades).
        req_labels = get_required_tfs_from_confluence(
            strat.get("confluence", []))
        secondary_tfs = tuple(sorted(get_tf_from_label(lbl)
                                      for lbl in req_labels))

    logger.info(
        "[StrategyData] sid=%s symbol=%s tf=%s anchor=%s "
        "visible=[%s, %s] visible_days=%.1f warmup_days=%.1f "
        "warmup_start=%s sec_tfs=%s",
        strat.get("id"), strat.get("symbol"), strat.get("timeframe"),
        anchor_source,
        visible_start.isoformat(), visible_end.isoformat(),
        visible_days, warmup_days, warmup_start.isoformat(),
        secondary_tfs,
    )

    # Hand to the existing data-prep machinery using explicit start/end.
    # prepare_data_with_indicators (services.py:185) accepts start_date /
    # end_date and prefers them over `days`.
    df = prepare_data_with_indicators(
        strat["symbol"],
        seed=strat.get("data_seed", 42),
        start_date=warmup_start.to_pydatetime(),
        end_date=visible_end.to_pydatetime(),
        timeframe=strat.get("timeframe", "1Min"),
        data_feed=data_feed,
        session=strat.get("trading_session", "RTH"),
        secondary_tfs=secondary_tfs,
        strat=strat,
        model_override=model_override,
        required_confluence_ids=required_confluence_ids,
        force_scope=force_scope,
    )

    sec_tf_map = get_secondary_tf_map(df) if len(df) > 0 else {}

    # Seed the secondary-TF SNAPSHOT (RORT_SECONDARY_TF_SNAPSHOT). This is a
    # FULL warmup-extended load (start=warmup_start), so `df` carries the
    # COMPLETE secondary series — safe to persist. Keyed by model_override so
    # coarse-gate Update-New appends can take the fast windowed path. Only
    # fires for long-cycle (1Hour+) secondaries; inert when the kill-switch is
    # OFF. Non-fatal on any error.
    try:
        import services as _svc
        if (_svc._secondary_snapshot_enabled() and secondary_tfs
                and _svc._has_long_cycle_secondary_tf(strat) and len(df) > 0):
            _svc._secondary_snapshot_persist(
                strat, df, secondary_tfs, model_override)
    except Exception:
        pass

    return StrategyDataBundle(
        df=df,
        visible_start=visible_start,
        visible_end=visible_end,
        warmup_start=warmup_start,
        secondary_tf_map=sec_tf_map,
        anchor_source=anchor_source,
        warmup_days=warmup_days,
    )


# ── Trim helpers ──────────────────────────────────────────────────────

def _trade_entry_column(trades: pd.DataFrame) -> Optional[str]:
    """The trades_df has used various column names over time. Return the
    first one that exists; None if neither does."""
    for col in ("entry_fill_ts", "entry_time", "entry_trigger_ts"):
        if col in trades.columns:
            return col
    return None


def trim_trades_to_visible(
    trades: pd.DataFrame, visible_start: pd.Timestamp,
) -> pd.DataFrame:
    """Drop trades whose entry is before `visible_start`."""
    if trades is None or len(trades) == 0:
        return trades
    col = _trade_entry_column(trades)
    if col is None:
        return trades
    entry_series = pd.to_datetime(trades[col], utc=True, errors="coerce")
    mask = entry_series >= visible_start
    return trades[mask].copy()


def trim_df_to_visible(
    df: pd.DataFrame, visible_start: pd.Timestamp,
) -> pd.DataFrame:
    """Drop bars whose index is before `visible_start`. Use for KPI
    denominators / equity-curve plotting on a bar-indexed df."""
    if df is None or len(df) == 0:
        return df
    if not hasattr(df.index, "tz") or df.index.tz is None:
        # Make the comparison work for naive-index dfs.
        anchor = visible_start.tz_convert(None) if visible_start.tz \
            else visible_start
        return df[df.index >= anchor].copy()
    return df[df.index >= visible_start].copy()
