"""Canonical Resampled Bar Store (M-RS2 Phase 2) — session-keyed COARSE bars.

================================================================================
WHY THIS EXISTS (the fidelity north star)
--------------------------------------------------------------------------------
Nearly every divergence bug this month is a CONFLUENCE-GATE bug whose root is:
the LIVE lane and the BACKTEST lane construct/serve the same secondary-TF gate
bar via DIFFERENT code paths, so the two can disagree (313 4H session
contamination, 325/330/331 fossilized-4h, 338 session mismatch, ...).

The structural cure: gates evaluate on the PREVIOUS COMPLETED coarse bar. If BOTH
lanes read that completed bar from ONE canonical store keyed by
(symbol, timeframe, SESSION), the gate state is identical BY CONSTRUCTION — 313's
contamination becomes impossible (there is no separately-built live 4H bar to
disagree). This module is that store's Phase-2 core.

Phase-1 (bar_cache.py) = the 1Sec/1Min REST Bars cache.
Phase-2 (this)         = canonical, session-keyed RESAMPLED coarse bars (2Min..1Day)
                         that both lanes read.

================================================================================
THE ONE DEFINITION (byte-identity is the acceptance criterion)
--------------------------------------------------------------------------------
    canonical(symbol, tf, session, window) =
        resample_to_timeframe( _filter_session( load_1min(symbol, window), session ), tf )

- Resample ALWAYS from 1Min, NEVER native coarse (Polygon native daily has split
  bugs — CLAUDE.md). The canonical resampler is `data_loader.resample_to_timeframe`
  + `_RESAMPLE_RULES`.
- SESSION is a FIRST-CLASS KEY (the 313 lesson). The session filter is
  `data_loader._filter_session` (half-open [start,end) ET), applied to the 1Min
  bars BEFORE resample. NOTE: this is the resample-time filter — deliberately NOT
  `ralph_engine._is_in_session` (equivalent boundaries, separate code path;
  binding to the wrong one would itself be a new divergence).
- The store's TF set is the COARSE bars only: 2Min..1Day (the `_RESAMPLE_RULES`
  keys with tf_seconds >= 120). 1Min is served from `bar_cache`; sub-minute
  (5/10/15/30Sec) derives from native 1Sec. `'1M'` UPPERCASE is the primary-TF
  sentinel in the record namespace and is NEVER a store key; `'1m'` lowercase
  (a real 1-minute secondary) is served from `bar_cache`, not here.
- The cross-TF shift (`secondary_tf_shift`) stays at GATE-EVAL time in the
  consumer, NOT in the store. The store holds raw completed coarse bars.

================================================================================
WINDOW-REPLACE, not upsert (the retracted-bar lesson)
--------------------------------------------------------------------------------
The write path must DELETE+INSERT the affected scope, never blind-upsert: if a
1Min bar is retracted (in cache but no longer in Polygon), the coarse bucket must
be REPLACED so the stale bar is removed (an upsert can't delete). The natural
replace unit is (symbol, tf, session, UTC-day): every store TF divides evenly
into a day and pandas anchors resample bins at UTC midnight, so NO bin straddles a
day boundary — per-UTC-day resample is byte-identical to a whole-window resample
(proven 2026-07-09 across 5Min/1Hour/4Hour/1Day, and asserted for the full
matrix by `_resampled_store_byteident.py`). That makes the incremental unit safe.

================================================================================
M1a SCOPE (this checkout): the store is OFFLINE + READ-ONLY.
--------------------------------------------------------------------------------
`canonical_resampled` and `build_window` are PURE COMPUTE — they read cached 1Min
via `bar_cache.read_bars` (a plain SELECT: no Polygon fetch, no upsert) and
resample in memory. NOTHING here writes the DB. Persistence (the
`resampled_bar_cache` table), the write-through builder, the read-time comparator
wired into live reads, and the consumer cutovers are LATER, separately-gated
phases (M1b+). All flags below default OFF and are inert until then.
================================================================================
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# The store's coarse TF set — 2Min..1Day. Derived from _RESAMPLE_RULES keys with
# tf_seconds >= 120 (excludes 5/10/15/30Sec which derive from native 1Sec, and
# 1Min which is served from bar_cache directly). Kept as an explicit list so the
# store's contract is obvious and testable, not an implicit filter.
COARSE_TFS = ["2Min", "3Min", "5Min", "10Min", "15Min", "30Min",
              "1Hour", "2Hour", "4Hour", "1Day"]

# Sessions that are first-class store keys. RTH / Extended Hours / 24-7 are the
# three the spec names; Pre-Market / After Hours are included because they're the
# same filter machinery and cost nothing to key. '24/7' means "no session filter"
# (all hours) — for equities that is the raw-all-hours resample, matching
# bar_cache.materialize_derived's 24/7-only path.
STORE_SESSIONS = ["RTH", "Extended Hours", "Pre-Market", "After Hours", "24/7"]

OHLCV = ["open", "high", "low", "close", "volume"]

# Extra store columns produced alongside OHLCV.
_META_COLS = ["source_1min_span_hash"]


# =============================================================================
# Feature flags — ALL default OFF (M1a exercises none of the write/live paths)
# =============================================================================

def compare_enabled() -> bool:
    """RORT_RESAMPLED_STORE_COMPARE (default OFF). When ON, every store READ also
    computes the canonical resample on-the-fly and byte-compares (OHLCV exact, per
    bar); log + alarm on ANY diff, per (symbol, tf, session). This is THE gate that
    makes the store subtract divergence instead of adding it — ship the store with
    the comparator ON in shadow BEFORE any consumer trusts it. Inert until a store
    read path exists (M1b+)."""
    return os.environ.get("RORT_RESAMPLED_STORE_COMPARE", "").strip().lower() in (
        "1", "true", "yes", "on")


def writethrough_enabled() -> bool:
    """RORT_RESAMPLED_STORE_WRITE (default OFF). When ON, the builder persists coarse
    bars into `resampled_bar_cache` (window-REPLACE). Inert in M1a (no table, no
    writes)."""
    return os.environ.get("RORT_RESAMPLED_STORE_WRITE", "").strip().lower() in (
        "1", "true", "yes", "on")


def read_enabled() -> bool:
    """RORT_RESAMPLED_STORE_READ (default OFF). When ON, cut-over consumers read the
    store instead of resampling. Inert in M1a (no consumer wired)."""
    return os.environ.get("RORT_RESAMPLED_STORE_READ", "").strip().lower() in (
        "1", "true", "yes", "on")


# =============================================================================
# TF helpers
# =============================================================================

def tf_seconds(tf: str) -> int:
    """Canonical TF string -> seconds ('4Hour'->14400, '1Day'->86400). Reuses
    bar_cache._tf_seconds so the store and the Phase-1 cache agree exactly."""
    from bar_cache import _tf_seconds
    return _tf_seconds(tf)


def is_store_tf(tf: str) -> bool:
    """True if `tf` is a coarse TF this store is responsible for (2Min..1Day)."""
    from data_loader import _RESAMPLE_RULES
    return tf in _RESAMPLE_RULES and tf_seconds(tf) >= 120


# =============================================================================
# The canonical oracle — the single definition the store must reproduce
# =============================================================================

def _load_1min_readonly(symbol: str, start, end) -> Optional[pd.DataFrame]:
    """Read cached 1Min for [start, end] READ-ONLY (bar_cache.read_bars = a plain
    direct-PG SELECT). NEVER fetches Polygon and NEVER writes the cache — the store
    is a read-only consumer of the 1Min layer (the data-worker owns keeping it
    fresh). Returns RAW all-hours 1Min (session filter applied by the caller), or
    None if the DSN is absent / nothing is cached."""
    import bar_cache
    return bar_cache.read_bars(symbol, "1Min", start, end)


def canonical_resampled(symbol: str, tf: str, session: str, start, end,
                        *, one_min_df: Optional[pd.DataFrame] = None
                        ) -> Optional[pd.DataFrame]:
    """THE canonical definition, computed on-the-fly. This is the oracle the store
    must be byte-identical to:

        resample_to_timeframe( _filter_session( load_1min(window), session ), tf )

    `one_min_df` — optional pre-loaded RAW (all-hours) 1Min for the window, so a
    caller resampling many TFs/sessions reads 1Min once. When None, reads it
    read-only. Returns an OHLCV DataFrame (UTC DatetimeIndex) or None."""
    from data_loader import _filter_session, resample_to_timeframe
    if one_min_df is None:
        one_min_df = _load_1min_readonly(symbol, start, end)
    if one_min_df is None or len(one_min_df) == 0:
        return None
    df = one_min_df[OHLCV].copy()
    if session and session != "24/7":
        df = _filter_session(df, session)
    if len(df) == 0:
        return None
    out = resample_to_timeframe(df, tf)
    return out if out is not None and len(out) > 0 else None


# =============================================================================
# Span hash — drift detector for the underlying 1Min
# =============================================================================

def _span_hash(members: pd.DataFrame) -> str:
    """Stable hash of the 1Min rows (ts + OHLCV) that produced one coarse bucket.
    When the underlying 1Min changes (T+2 revision / settle-sweeper correction),
    the hash changes -> that bucket is re-resampled. Hashes the int64-ns index +
    the float64 OHLCV bytes (deterministic for identical values; order-stable
    because the members are ts-sorted)."""
    idx_bytes = members.index.asi8.tobytes()            # ns since epoch, int64
    val_bytes = members[OHLCV].to_numpy(dtype="float64").tobytes()
    return hashlib.sha1(idx_bytes + val_bytes).hexdigest()


# =============================================================================
# Builder — window-REPLACE unit = (symbol, tf, session, UTC-day)
# =============================================================================

def _empty_built() -> pd.DataFrame:
    return pd.DataFrame(columns=OHLCV + _META_COLS)


def build_window(symbol: str, tf: str, session: str, start, end,
                 *, one_min_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Build the store's coarse bars for the window by resampling session-filtered
    cached 1Min PER UTC-DAY (the window-REPLACE unit) and concatenating. Proven
    byte-identical to the whole-window canonical resample (see module docstring).
    Also stamps `source_1min_span_hash` per bucket.

    PURE COMPUTE — no DB writes. Returns a DataFrame indexed by bar-ts (UTC) with
    columns OHLCV + source_1min_span_hash, or an empty frame when the session has
    no bars in the window."""
    from data_loader import _filter_session, resample_to_timeframe, _RESAMPLE_RULES
    rule = _RESAMPLE_RULES.get(tf)
    if rule is None:
        raise ValueError(f"resampled_bar_store: '{tf}' is not a resamplable coarse "
                         f"TF. Store TFs: {COARSE_TFS}")
    if one_min_df is None:
        one_min_df = _load_1min_readonly(symbol, start, end)
    if one_min_df is None or len(one_min_df) == 0:
        return _empty_built()
    df = one_min_df[OHLCV].copy()
    if session and session != "24/7":
        df = _filter_session(df, session)
    if len(df) == 0:
        return _empty_built()

    frames = []
    # Group by UTC calendar day (normalize -> midnight-UTC key). Every store TF
    # divides the day and anchors at midnight, so a day-scoped resample yields the
    # same bins/labels as the whole-window resample — the property that makes
    # per-day window-REPLACE safe.
    for _day, day_df in df.groupby(df.index.normalize()):
        coarse = resample_to_timeframe(day_df.copy(), tf)
        if coarse is None or len(coarse) == 0:
            continue
        # Per-bucket span hash from the same day's 1Min members.
        hashes: dict = {}
        for bstart, members in day_df.resample(rule):
            if len(members) == 0:
                continue
            hashes[bstart] = _span_hash(members)
        coarse = coarse[["open", "high", "low", "close", "volume"]].copy()
        coarse["source_1min_span_hash"] = [hashes.get(ix) for ix in coarse.index]
        frames.append(coarse)

    if not frames:
        return _empty_built()
    out = pd.concat(frames).sort_index()
    return out


# =============================================================================
# Comparator — byte-compare built store bars vs the canonical resample
# =============================================================================

def _num_equal(a, b) -> bool:
    """Exact float equality with NaN==NaN treated as equal. OHLC come from
    first/max/min/last (no arithmetic -> exact); volume is a sum of the same
    members in the same order (-> exact). So exact equality is the right test."""
    import math
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    fa, fb = float(a), float(b)
    if math.isnan(fa) and math.isnan(fb):
        return True
    return fa == fb


def compare_store_vs_canonical(built: Optional[pd.DataFrame],
                               canonical: Optional[pd.DataFrame],
                               *, symbol: str = "", tf: str = "",
                               session: str = "") -> dict:
    """Byte-compare built store bars vs the canonical resample. Returns a dict:
        { match: bool, n_built, n_canonical, index_mismatch: [...], cell_diffs: [...] }
    OHLCV exact per bar. This is the comparator body the shadow mode
    (RORT_RESAMPLED_STORE_COMPARE) and the offline harness both use, so 'green' means
    the same thing everywhere."""
    tag = {"symbol": symbol, "tf": tf, "session": session}
    b = built[OHLCV] if built is not None and len(built) > 0 else None
    c = canonical[OHLCV] if canonical is not None and len(canonical) > 0 else None

    if b is None and c is None:
        return {**tag, "match": True, "n_built": 0, "n_canonical": 0,
                "index_mismatch": [], "cell_diffs": [], "note": "both empty"}
    if b is None or c is None:
        return {**tag, "match": False, "n_built": 0 if b is None else len(b),
                "n_canonical": 0 if c is None else len(c),
                "index_mismatch": ["one side empty"], "cell_diffs": [],
                "note": "one side empty, the other not"}

    if not b.index.equals(c.index):
        only_built = [str(x) for x in b.index.difference(c.index)][:10]
        only_canon = [str(x) for x in c.index.difference(b.index)][:10]
        return {**tag, "match": False, "n_built": len(b), "n_canonical": len(c),
                "index_mismatch": {"only_in_built": only_built,
                                   "only_in_canonical": only_canon},
                "cell_diffs": [], "note": "index (bar set) mismatch"}

    cell_diffs = []
    for i, ts in enumerate(b.index):
        for col in OHLCV:
            av, cv = b[col].iloc[i], c[col].iloc[i]
            if not _num_equal(av, cv):
                cell_diffs.append({"ts": str(ts), "col": col,
                                   "built": float(av) if av == av else None,
                                   "canonical": float(cv) if cv == cv else None})
                if len(cell_diffs) >= 50:  # cap the report
                    break
        if len(cell_diffs) >= 50:
            break

    return {**tag, "match": len(cell_diffs) == 0, "n_built": len(b),
            "n_canonical": len(c), "index_mismatch": [], "cell_diffs": cell_diffs}


def build_and_compare(symbol: str, tf: str, session: str, start, end,
                      *, one_min_df: Optional[pd.DataFrame] = None) -> dict:
    """Convenience: build the store bars AND the canonical from the SAME 1Min input,
    then byte-compare. The offline byte-identity harness's per-cell unit."""
    if one_min_df is None:
        one_min_df = _load_1min_readonly(symbol, start, end)
    built = build_window(symbol, tf, session, start, end, one_min_df=one_min_df)
    canon = canonical_resampled(symbol, tf, session, start, end,
                                one_min_df=one_min_df)
    return compare_store_vs_canonical(built, canon, symbol=symbol, tf=tf,
                                      session=session)
