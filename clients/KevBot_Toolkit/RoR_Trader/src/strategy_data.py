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


def _resampled_store_read_enabled() -> bool:
    """Kill-switch (default OFF) for CONSUMER #1 of the M-RS2 Phase-2 resampled bar
    store: read the coarse (>=1Hour) secondary from the canonical store instead of
    resampling 1Min here. Byte-identity-safe by construction (see
    `_coarse_secondary_from_store`). Instant rollback by unsetting."""
    try:
        import resampled_bar_store as rbs
        return rbs.read_enabled()
    except Exception:
        return False


def _coarse_secondary_from_store(symbol, coarse_tfs, start, end, session):
    """CONSUMER #1 (M-RS2 Phase 2): VERIFIED-CANONICAL read of the coarse (>=1Hour)
    secondaries — the output is ALWAYS byte-identical to the flag-OFF resample path
    BY CONSTRUCTION, and the store is VERIFIED against it on the comparable zone.

    v2 (2026-07-10, window-alignment fix — 5th comparator catch): the original
    all-or-nothing compare failed on ANY non-bin-aligned window start, because
    `read_store` floors its low bound (full first bar) while the canonical was
    built from 1Min loaded at the RAW start (partial first bucket) → guaranteed
    first-bucket cell diffs → permanent fallback. Same class at the tail: the
    store's edge bucket is a settled snapshot while the canonical's tip is fresher.
    Neither is drift — both are window-semantics artifacts.

    Design now:
      - OUTPUT: resample of the session-filtered 1Min from the RAW [start, end] —
        the same members the flag-OFF path resamples, from the same loader → flag
        ON == flag OFF bytes ALWAYS, unconditionally (no "only when identical").
      - VERIFY: compare store vs canonical ONLY on the comparable zone —
        bin-ALIGNED head (ts >= floor(start, tf)) through the SETTLED cutoff
        (drop the forming/just-closed tip) — and log green/drift per tf. A head
        gap where the window pre-dates store coverage is reported as coverage,
        not drift. This is the promotion evidence: once VERIFY is green across
        real engine windows for N days, the compute-skip cutover (serve the
        settled zone FROM the store and skip the resample — the speed payoff)
        flips on with the store already field-proven on real windows.

    Returns {canonical_tf: DataFrame[OHLCV]} or None (load failure → caller runs
    its own identical resample path). Inert when the read flag is off."""
    # v3: the OUTPUT no longer comes from this function at all — the caller's own
    # load+resample (the flag-OFF body) is the single output path, so ON == OFF is
    # the SAME code, not an equivalence claim. This function's job is now pure
    # VERIFICATION (see _verify_coarse_secondary_store). Returning None always
    # sends the caller down its own path.
    return None


def _verify_coarse_secondary_store(symbol, out, session, end, tag="#1"):
    """Store VERIFICATION (log-only, never affects output): compare the store
    against the engine's own freshly-resampled coarse secondaries on the comparable
    zone, and log GREEN/DRIFT per tf. Shared by consumer #1 (offline backtest
    secondary build, tag '#1') and consumer #3 (LIVE warmup/reload path via
    `ralph_engine._load_warmup_df`, tag '#3-live'). The zone excludes:
      - the HEAD bucket (its content depends on the caller's window-start loader
        semantics — partial vs full first bucket is a window artifact, not drift);
      - the UNSETTLED tail (the forming/just-closed WS-tip bucket legitimately
        differs from the store's settled snapshot);
      - bars before the store's coverage floor (seed depth = coverage, not drift).
    What remains is like-for-like settled truth: ANY diff there is real drift and
    is logged as a warning (the admin comparator diff-rate picks it up).

    This is the promotion evidence for the compute-skip cutover: once VERIFY runs
    green across real engine windows for N trading days, serving the settled zone
    FROM the store (skipping the resample — the speed payoff) flips on with the
    store already field-proven on the exact windows the engine uses."""
    try:
        import resampled_bar_store as rbs
        for tf, sec in out.items():
            try:
                if sec is None or len(sec) < 3:
                    continue
                head = sec.index[0]
                zone_hi = rbs._settled_cutoff_ts(tf)
                sec_z = sec[(sec.index > head) & (sec.index <= zone_hi)]
                if len(sec_z) == 0:
                    continue
                store_df = rbs.read_store(symbol, tf, session, sec_z.index[0], end)
                if store_df is None or len(store_df) == 0:
                    logger.info("[ResampledStore%s] %s %s %s: VERIFY skipped — "
                                "store uncovered for window", tag, symbol, tf,
                                session)
                    continue
                store_z = store_df[(store_df.index > head)
                                   & (store_df.index <= zone_hi)]
                head_gap = 0
                tail_lag = 0
                if len(store_z):
                    head_gap = int((sec_z.index < store_z.index[0]).sum())
                    if head_gap:
                        sec_z = sec_z[sec_z.index >= store_z.index[0]]
                    # EDGE LAG ≠ drift: bars newer than the store's last write
                    # (maintain hasn't caught up — settled bars keep accruing
                    # between passes) are an operational freshness signal, not
                    # byte-divergence. Trim them from the compare and report
                    # separately so DRIFT stays "values differ" only.
                    tail_lag = int((sec_z.index > store_z.index[-1]).sum())
                    if tail_lag:
                        sec_z = sec_z[sec_z.index <= store_z.index[-1]]
                cols = ["open", "high", "low", "close", "volume"]
                cmp = rbs.compare_store_vs_canonical(
                    store_z, sec_z[cols], symbol=symbol, tf=tf, session=session)
                if cmp["match"]:
                    notes = ""
                    if head_gap:
                        notes += f" (head coverage gap {head_gap} bars)"
                    if tail_lag:
                        notes += f" (edge lag {tail_lag} bars — maintain behind)"
                    logger.info("[ResampledStore%s] %s %s %s: VERIFY GREEN — %d "
                                "settled bars byte-identical%s", tag, symbol, tf,
                                session, len(store_z), notes)
                else:
                    detail = cmp.get("note") or f"{len(cmp['cell_diffs'])} cell diffs"
                    if cmp.get("cell_diffs"):
                        d0 = cmp["cell_diffs"][0]
                        detail += (f" [first: {d0.get('ts')} {d0.get('col')} "
                                   f"store={d0.get('built')} canon={d0.get('canonical')}]")
                    logger.warning("[ResampledStore%s] %s %s %s: VERIFY DRIFT — %s "
                                   "(log-only; output is the engine's own resample)",
                                   tag, symbol, tf, session, detail)
            except Exception as ve:  # noqa: BLE001 — verify must never break trades
                logger.warning("[ResampledStore%s] verify failed %s %s: %s",
                               tag, symbol, tf, ve)
    except Exception as e:  # noqa: BLE001
        logger.warning("[ResampledStore%s] verify unavailable %s: %s", tag, symbol, e)


def _resampled_store_serve_enabled() -> bool:
    """COMPUTE-SKIP kill-switch (default OFF): serve the SETTLED zone of coarse
    secondaries FROM the canonical store and load 1Min only for the edge days —
    the M-RS2-P2 speed payoff. Promotion-gated: arm only after the verify ledger
    is green on real windows (docs/_active/Weekend_Sprint_Ledger.md: fleet-wide
    zero drift, 2026-07-11). Instant rollback by unsetting."""
    return os.getenv("RORT_RESAMPLED_STORE_SERVE", "0") == "1"


def _coarse_secondary_serve_from_store(symbol, coarse_tfs, start, end,
                                       session, data_feed):
    """COMPUTE-SKIP (M-RS2 Phase 2, sprint item 3): build the coarse secondaries
    as  [store bars over the settled whole-days]  +  [fresh 1Min resample for the
    EDGE days only]  — byte-identical to the full deep resample BY CONSTRUCTION:
      - resample bins are UTC-day-anchored and never straddle days (the proven
        window-REPLACE property), so per-day pieces == whole-window resample;
      - the store's whole-day bars are byte-identical to the canonical resample
        (fleet-wide zero-drift ledger, 2026-07-11);
      - edge days (today / any day newer than the store's last SETTLED whole
        day, plus any head day older than store coverage) are resampled from a
        fresh 1Min load exactly as the deep path would.
    The 1Min load collapses from warmup-depth (e.g. 182d for a 4Hour gate) to
    the edge days only. Returns {tf: DataFrame} or None on ANY doubt → caller
    runs the full deep path (== flag-OFF bytes, always)."""
    try:
        import pandas as pd
        import resampled_bar_store as rbs
        from data_loader import load_market_data, resample_to_timeframe
        cols = ["open", "high", "low", "close", "volume"]
        now = datetime.now(timezone.utc)
        out = {}
        lo = pd.Timestamp(start)
        lo = (lo.tz_localize("UTC") if lo.tzinfo is None else lo)
        head_day = lo.floor("1D")
        # Edge cut: last COMPLETE UTC day that is fully settled (yesterday once
        # today began; today is always rebuilt fresh — its bars may be forming).
        edge_day = pd.Timestamp(now).tz_convert("UTC").normalize()
        # HEAD day is rebuilt fresh from the RAW window start, NOT served from
        # the store: the deep path's first-day bars are PARTIAL when `start`
        # falls mid-day — and for sessions whose UTC bucket straddles midnight
        # (winter Extended Hours includes the prior evening's 00:00-01:00Z tail)
        # the store's full-day bar differs from that partial. Caught by the
        # byte-proof 2026-07-11 (TSLA/KO 1Day Extended first-day 'open').
        df1_head = None
        df1_edge = None  # lazy — one load each, shared by every tf
        for tf in coarse_tfs:
            store_df = rbs.read_store(symbol, tf, session, start, end)
            if store_df is None or len(store_df) == 0:
                return None
            # Store piece: whole settled days strictly AFTER the head day and
            # strictly BEFORE the edge day.
            store_piece = store_df[
                (store_df.index >= head_day + pd.Timedelta(days=1))
                & (store_df.index < edge_day)][cols]
            if len(store_piece) == 0:
                return None
            # Coverage: the store must reach the first post-head day the window
            # needs (deeper windows fall back to the deep path). Tolerance is
            # calendar days but markets aren't: a Fri/Sat head day puts the
            # first real bar on Monday 13:30Z (> head+2d), and a long weekend
            # (Jul-4 2026: Fri head → first bar Mon) pushes further — those
            # fell back needlessly (caught by the live-serve validation,
            # 2026-07-13). 5 days rides out any US market gap while still
            # catching genuine under-coverage (a short seed misses by months,
            # e.g. the 182d-seed canary missed by ~218d).
            if store_piece.index[0] > head_day + pd.Timedelta(days=5):
                logger.info("[ResampledStore#skip] %s %s %s: head coverage short "
                            "(store starts %s, window %s) → deep path", symbol,
                            tf, session, store_piece.index[0], lo)
                return None
            # Head piece: fresh 1Min for [start, next midnight) — reproduces the
            # deep path's partial first-day bars exactly (all sessions).
            if df1_head is None:
                df1_head = load_market_data(
                    symbol, start_date=start,
                    end_date=(head_day + pd.Timedelta(days=1)).to_pydatetime(),
                    timeframe="1Min", feed=data_feed, session=session)
            head_piece = None
            if df1_head is not None and len(df1_head) > 0:
                head_1m = df1_head[
                    (df1_head.index >= lo)
                    & (df1_head.index < head_day + pd.Timedelta(days=1))]
                if len(head_1m) > 0:
                    head_piece = resample_to_timeframe(head_1m[cols].copy(), tf)
            # Edge piece: fresh 1Min from edge_day forward, resampled per tf.
            if df1_edge is None:
                df1_edge = load_market_data(
                    symbol, start_date=edge_day.to_pydatetime(), end_date=end,
                    timeframe="1Min", feed=data_feed, session=session)
            edge_piece = None
            if df1_edge is not None and len(df1_edge) > 0:
                edge_1m = df1_edge[df1_edge.index >= edge_day]
                if len(edge_1m) > 0:
                    edge_piece = resample_to_timeframe(edge_1m[cols].copy(), tf)
            pieces = []
            if head_piece is not None and len(head_piece) > 0:
                pieces.append(head_piece[cols])
            pieces.append(store_piece)
            if edge_piece is not None and len(edge_piece) > 0:
                pieces.append(edge_piece[cols])
            sec = pd.concat(pieces)
            sec = sec[~sec.index.duplicated(keep="last")].sort_index()
            out[tf] = sec
        if out:
            logger.info("[ResampledStore#skip] %s SERVED %s from store + edge-day "
                        "resample [%s] (compute-skip)", symbol,
                        sorted(out.keys()), session)
        return out or None
    except Exception as e:  # noqa: BLE001
        logger.warning("[ResampledStore#skip] serve failed %s %s: %s → deep path",
                       symbol, coarse_tfs, e)
        return None


def _build_coarse_secondary_from_1min(symbol, coarse_tfs, start, end,
                                      session, data_feed):
    """Build coarse (>=1Hour) secondary OHLCV from a single 1Min load + resample
    — the SAME source LIVE uses (`ralph_engine._load_warmup_df`) and identical
    to what `prepare_data_with_indicators` would resample, only sourced from
    1Min not the sub-minute primary. Returns {canonical_tf: DataFrame[OHLCV]}
    (full series, last bar kept — matches prepare_data's internal resample at
    services.py:360) or None on any miss → caller falls back to the resample
    path. The 1Min load is bar_cache-accelerated when BAR_CACHE_ENABLED."""
    # COMPUTE-SKIP (RORT_RESAMPLED_STORE_SERVE, default OFF): serve settled
    # whole days from the store + resample only the edge days. Byte-identical
    # by construction; any doubt → None → the deep path below (== OFF bytes).
    if _resampled_store_serve_enabled():
        served = _coarse_secondary_serve_from_store(
            symbol, coarse_tfs, start, end, session, data_feed)
        if served is not None:
            return served
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
        # CONSUMER #1 (M-RS2 Phase 2), v3: output above is the single path for
        # flag ON and OFF alike (ON == OFF is the same code). When the read flag
        # is armed, additionally VERIFY the store against these freshly-resampled
        # secondaries on the settled/aligned zone — log-only promotion evidence
        # for the later compute-skip cutover. Never affects trades.
        if out and _resampled_store_read_enabled():
            _verify_coarse_secondary_store(symbol, out, session, end)
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
