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
    # graduated to default-ON 2026-07-03 (flag-graduation; env remains the kill switch)
    return os.getenv("RORT_RIGHTSIZE_WARMUP", "1") == "1"


# ── Coarse-secondary-from-1Min (the ~360× warmup blow-up fix) ──────────
COARSE_SECONDARY_SECONDS = 3600
"""A secondary TF at/above this (1Hour/4Hour/1Day) is "coarse": converging
its indicator needs ~250 of its own bars = a long calendar span, which the
default path pays by loading the sub-minute PRIMARY across that whole span
and resampling UP (~360× blow-up for a 10Sec primary + 1Day gate). When
RORT_COARSE_SECONDARY_FROM_1MIN is on we instead build the coarse secondary
from 1Min (cache-accelerated) and inject it, sizing the primary warmup short."""


def _coarse_secondary_from_1min_enabled() -> bool:
    """Kill-switch (default OFF). When ON, coarse (>=1Hour) secondary gates are
    built from 1Min and injected via `secondary_tf_dfs` instead of being
    resampled from the sub-minute primary — source matches LIVE
    `ralph_engine._load_warmup_df`, removing a latent backtest-vs-live
    divergence. Instant rollback by unsetting."""
    # graduated to default-ON 2026-07-03 (flag-graduation; env remains the kill switch)
    return os.getenv("RORT_COARSE_SECONDARY_FROM_1MIN", "1") == "1"


def _tf_seconds_safe(tf: str) -> int:
    """Canonical seconds for a TF label ('1Day'->86400). 60 on any miss."""
    try:
        from unified_engine import TIMEFRAME_SECONDS
        return int(TIMEFRAME_SECONDS.get(tf, 60))
    except Exception:
        return 60


def _tf_warmup_days(tf: str, bars: float) -> float:
    """Calendar days needed to warm `bars` bars of `tf` (trading->calendar)."""
    from data_loader import BARS_PER_DAY
    bpd = BARS_PER_DAY.get(tf, 390) or 390
    return math.ceil(bars / bpd * _TRADING_TO_CALENDAR)


def _build_coarse_secondary_from_1min(symbol, coarse_tfs, start, end,
                                      session, data_feed):
    """Build coarse (>=1Hour) secondary OHLCV from a single 1Min load + resample
    — the SAME source LIVE uses (`ralph_engine._load_warmup_df`) and identical
    to what `prepare_data_with_indicators` would resample, only sourced from
    1Min not the sub-minute primary. Returns {canonical_tf: DataFrame[OHLCV]}
    (full series, last bar kept — matches prepare_data's internal resample at
    services.py:360) or None on any miss → caller falls back to the resample
    path. The 1Min load is bar_cache-accelerated when BAR_CACHE_ENABLED."""
    try:
        from data_loader import load_market_data, resample_to_timeframe
        cols = ["open", "high", "low", "close", "volume"]
        df1 = load_market_data(symbol, start_date=start, end_date=end,
                               timeframe="1Min", feed=data_feed, session=session)
        if df1 is None or len(df1) == 0:
            return None
        out = {}
        for tf in coarse_tfs:
            sec = resample_to_timeframe(df1[cols].copy(), tf)
            if sec is not None and len(sec) > 0:
                out[tf] = sec
        return out or None
    except Exception as e:  # noqa: BLE001
        logger.warning("[CoarseSecondary] build-from-1Min failed %s %s: %s",
                       symbol, coarse_tfs, e)
        return None


def _build_native_1min_secondary(symbol, start, end, session, data_feed,
                                 no_backfill=False):
    """Build a 1-MINUTE SECONDARY gate's OHLCV from the NATIVE 1Min bar.

    For the 1-minute-secondary gate class (RORT_ENFORCE_1MIN_GATE): a '1M'/'1m'
    gate on a sub-minute (e.g. 30Sec) primary is a genuine 1-minute secondary.
    Source it from the SAME native 1Min bar LIVE consumes
    (`ralph_engine._load_warmup_df` / a native 1Min BarBuilder) rather than
    resampling UP from the sub-minute primary — no aggregation drift, and it
    parallels how a 1Min PRIMARY is loaded. Returns {'1Min': DataFrame[OHLCV]}
    (full series, cache-accelerated when BAR_CACHE_ENABLED) or None on any miss
    (caller falls back to the resample rule so the gate is never silently
    skipped).

    `no_backfill` (M-RS4 Phase 3 shadow lane): read-only cache access — never
    fetch Polygon / write the cache. The read-only resident shadow-worker passes
    True so this native build honors the same no-write contract as the rest of
    its prep. Default False keeps the recompute callers byte-identical."""
    try:
        from data_loader import load_market_data
        cols = ["open", "high", "low", "close", "volume"]
        df1 = load_market_data(symbol, start_date=start, end_date=end,
                               timeframe="1Min", feed=data_feed, session=session,
                               no_backfill=no_backfill)
        if df1 is None or len(df1) == 0:
            return None
        return {"1Min": df1[cols].copy()}
    except Exception as e:  # noqa: BLE001
        logger.warning("[1MinSecondary] native build failed %s: %s", symbol, e)
        return None


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
        get_required_tfs_from_confluence, get_tf_from_label,
    )

    primary_tf = strat.get("timeframe", "1Min")
    try:
        sec_tfs = [get_tf_from_label(lbl)
                   for lbl in get_required_tfs_from_confluence(
                       strat.get("confluence", []))]
    except Exception:
        sec_tfs = []

    days = _tf_warmup_days(primary_tf, PRIMARY_WARMUP_BARS)
    for tf in sec_tfs:
        days = max(days, _tf_warmup_days(tf, SECONDARY_WARMUP_BARS))
    return max(1.0, float(days))


def plan_windowed_load(
    strat: dict,
    secondary_tfs: tuple,
    visible_start: pd.Timestamp,
    visible_end: pd.Timestamp,
    data_feed: str,
    base_warmup_days: float,
    *,
    allow_coarse_injection: bool = True,
) -> Tuple[float, Optional[dict]]:
    """Per-TF windowed-load planner (M-RS1 + coarse-secondary-from-1Min).

    Given the already-right-sized MAX `base_warmup_days` (from
    `compute_warmup_days`) and the strategy's secondary TFs, decide how to
    load so the sub-minute PRIMARY is NEVER fetched across a coarse (>=1Hour)
    secondary's long calendar span. Returns ``(warmup_days, sec_inject)``:

      - ``sec_inject``: ``{canonical_tf: DataFrame[OHLCV]}`` for the coarse
        (>=1Hour) secondaries, pre-built from a single cheap 1Min load over
        each's own ``SECONDARY_WARMUP_BARS`` span (or ``None`` when there is
        no coarse secondary / the flag is off). Inject via
        ``prepare_data_with_indicators(secondary_tf_dfs=...)``.
      - ``warmup_days``: RE-SIZED to the PRIMARY + any NON-coarse secondaries
        only when a coarse injection is produced (the coarse ones are already
        pre-warmed), so the primary load window stays short. Otherwise it is
        returned unchanged (== ``base_warmup_days``).

    Inert — returns ``(base_warmup_days, None)`` — when the
    coarse-secondary-from-1Min flag is off, there is no coarse secondary, or
    ``allow_coarse_injection`` is False (e.g. Mass Builder's own MTF load).
    This is the SAME machinery `load_strategy_data` uses; the parity-ribbon
    backtest endpoint calls it so its warmup matches the engine bar-for-bar
    without loading the display primary across the ~363d daily-gate span.
    """
    warmup_days = base_warmup_days
    sec_inject = None
    coarse_tfs = tuple(t for t in secondary_tfs
                       if _tf_seconds_safe(t) >= COARSE_SECONDARY_SECONDS)
    if _coarse_secondary_from_1min_enabled() and coarse_tfs and allow_coarse_injection:
        sec_warmup_days = max(_tf_warmup_days(t, SECONDARY_WARMUP_BARS)
                              for t in coarse_tfs)
        sec_start = visible_start - pd.Timedelta(days=sec_warmup_days)
        sec_inject = _build_coarse_secondary_from_1min(
            strat["symbol"], coarse_tfs, sec_start.to_pydatetime(),
            visible_end.to_pydatetime(),
            strat.get("trading_session", "RTH"), data_feed)
        # Re-size the PRIMARY warmup off primary + NON-coarse secondaries only
        # (the coarse ones now come pre-built). Only when right-sizing is on —
        # the legacy `visible_days * 2` window is already short, so leave it.
        if sec_inject and _rightsize_warmup_enabled():
            primary_tf = strat.get("timeframe", "1Min")
            wd = _tf_warmup_days(primary_tf, PRIMARY_WARMUP_BARS)
            for t in secondary_tfs:
                if _tf_seconds_safe(t) < COARSE_SECONDARY_SECONDS:
                    wd = max(wd, _tf_warmup_days(t, SECONDARY_WARMUP_BARS))
            warmup_days = max(1.0, float(wd))
    return warmup_days, sec_inject


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
        normalize_1min_secondary_gate, enforce_1min_gate_enabled,
    )

    # 1-minute-secondary gate (RORT_ENFORCE_1MIN_GATE): primary-aware relabel of
    # any overloaded '1M-' gate → lowercase '1m-' (a genuine 1-minute secondary)
    # on a non-1Min primary, so the standard lowercase-secondary machinery loads,
    # builds, and matches it. Flag OFF / 1Min-primary → unchanged (byte-identical).
    # Also fires the silent-drop tripwire. Work on a shallow copy so the caller's
    # dict is never mutated.
    _norm_conf = normalize_1min_secondary_gate(
        strat.get("confluence"), strat.get("timeframe", "1Min"))
    if _norm_conf is not strat.get("confluence"):
        strat = {**strat, "confluence": _norm_conf}

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

    # Coarse-secondary-from-1Min (RORT_COARSE_SECONDARY_FROM_1MIN): for coarse
    # (>=1Hour) secondary gates, build the secondary OHLCV from 1Min (cache-
    # accelerated) over the long convergence span and inject it via
    # `secondary_tf_dfs`, then re-size the PRIMARY warmup off the primary + any
    # sub-1Hour secondaries ONLY. This avoids loading the sub-minute primary
    # across the coarse span (~363d for a 1Day gate → the ~360× blow-up that
    # hung sid 338's UAD). Source = 1Min → matches LIVE _load_warmup_df, closing
    # a latent backtest-vs-live divergence. Inert when OFF / no coarse secondary.
    # Scope to the normal strategy path — Mass Builder (secondary_tfs_override)
    # has its own MTF-load expectations and is out of scope for this fix.
    warmup_days, sec_inject = plan_windowed_load(
        strat, secondary_tfs, visible_start, visible_end, data_feed,
        warmup_days, allow_coarse_injection=(secondary_tfs_override is None))
    warmup_start = visible_start - pd.Timedelta(days=warmup_days)

    # 1-minute SECONDARY gate (RORT_ENFORCE_1MIN_GATE): when the primary is NOT
    # 1Min and a reclassified '1m' gate resolved to a 1Min secondary, build it
    # from the NATIVE 1Min bar (matches LIVE; no resample-from-primary drift)
    # and inject via secondary_tf_dfs. Inert when the flag is OFF (a 1Min
    # secondary never appears in secondary_tfs then).
    if (enforce_1min_gate_enabled() and "1Min" in secondary_tfs
            and _tf_seconds_safe(strat.get("timeframe", "1Min")) != 60
            and (sec_inject is None or "1Min" not in sec_inject)):
        _native1 = _build_native_1min_secondary(
            strat["symbol"], warmup_start.to_pydatetime(),
            visible_end.to_pydatetime(),
            strat.get("trading_session", "RTH"), data_feed)
        if _native1:
            sec_inject = {**(sec_inject or {}), **_native1}

    logger.info(
        "[StrategyData] sid=%s symbol=%s tf=%s anchor=%s "
        "visible=[%s, %s] visible_days=%.1f warmup_days=%.1f "
        "warmup_start=%s sec_tfs=%s coarse_inject=%s",
        strat.get("id"), strat.get("symbol"), strat.get("timeframe"),
        anchor_source,
        visible_start.isoformat(), visible_end.isoformat(),
        visible_days, warmup_days, warmup_start.isoformat(),
        secondary_tfs,
        (sorted(sec_inject.keys()) if sec_inject else None),
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
        secondary_tf_dfs=sec_inject,
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
