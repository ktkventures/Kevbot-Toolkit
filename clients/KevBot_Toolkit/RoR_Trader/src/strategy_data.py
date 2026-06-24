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
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants (tunable; see plan: piped-wondering-tower.md) ───────────

WARMUP_MULTIPLIER = 2
"""Warmup buffer = visible_days * this. LEGACY path (kill-switch OFF).
Keeping `*2` matches the prior behavior of `prepare_forward_test_data:834`
so live KPIs don't shift out from under existing strategies. The
right-sized path (M-RS1, RORT_RIGHTSIZE_WARMUP=1) replaces this — see
`compute_warmup_days` + `docs/_active/Plan_M-RS1_Warmup_Rightsizing.md`."""

# ── M-RS1 right-sized warmup (per-TF; kill-switch RORT_RIGHTSIZE_WARMUP) ──
PRIMARY_WARMUP_BARS = 1200
"""Warmup bars for the PRIMARY (fastest) TF. Generous — covers long-EMA
(EMA-200 ≈ 6× period) convergence. Cheap in calendar terms because the
primary is the highest bars/day TF (1200 × 30Sec ≈ 1.5 trading days)."""

SECONDARY_WARMUP_BARS = 250
"""Warmup bars for each SECONDARY (coarse) TF. == ralph_engine.
_SHADOW_WARMUP_TARGET_BARS so the backtest warms its secondaries exactly
like the LIVE engine does (parity bonus). A single global target would
backfire here: 1200 bars on a 4h TF = ~2.4yr of calendar warmup."""

_TRADING_TO_CALENDAR = 365.0 / 252.0
"""Inflate trading-day counts → calendar days (weekends/holidays)."""


def _rightsize_warmup_enabled() -> bool:
    """M-RS1 kill-switch. Default OFF — legacy `visible_days * 2` until
    byte-identical-proven, then flip ON. Instant rollback by unsetting."""
    return os.getenv("RORT_RIGHTSIZE_WARMUP", "0") == "1"

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
    """Warmup buffer in days.

    LEGACY (RORT_RIGHTSIZE_WARMUP unset): `visible_days * WARMUP_MULTIPLIER`.
    This scales with the visible window, which is wrong — indicators need a
    FIXED bar count regardless of how long the backtest runs. For an old
    coarse-gate strategy (visible ≈ 540d) this loads ~1,620d ≈ 555k 1-min
    bars and the engine chews all of it (~2/3 trimmed away). See cProfile in
    `docs/_active/Recompute_Scalability_Findings.md`.

    RIGHT-SIZED (M-RS1, RORT_RIGHTSIZE_WARMUP=1): size warmup PER-TF to the
    bars each TF needs to converge, then take the max calendar span:
      - PRIMARY (fastest TF): PRIMARY_WARMUP_BARS (generous; cheap in days).
      - each SECONDARY (coarse): SECONDARY_WARMUP_BARS (== live's 250-bar
        standard). A single global target would over-warm the coarse TF.
    Independent of visible_days. Byte-identical-gated before default ON —
    see `docs/_active/Plan_M-RS1_Warmup_Rightsizing.md`.
    """
    if not _rightsize_warmup_enabled():
        return max(1.0, float(visible_days) * float(WARMUP_MULTIPLIER))

    # Right-sized, per-TF. Resolve secondaries from confluence (same
    # convention as load_strategy_data / services.get_strategy_trades).
    from data_loader import (
        BARS_PER_DAY, get_required_tfs_from_confluence, get_tf_from_label,
    )

    primary_tf = strat.get("timeframe", "1Min")
    try:
        sec_tfs = [get_tf_from_label(lbl)
                   for lbl in get_required_tfs_from_confluence(
                       strat.get("confluence", []))]
    except Exception:
        sec_tfs = []

    def _days(bars: float, tf: str) -> float:
        bpd = BARS_PER_DAY.get(tf, 390) or 390
        return math.ceil(bars / bpd * _TRADING_TO_CALENDAR)

    days = _days(PRIMARY_WARMUP_BARS, primary_tf)
    for tf in sec_tfs:
        days = max(days, _days(SECONDARY_WARMUP_BARS, tf))
    return max(1.0, float(days))


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
