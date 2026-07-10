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


def base_layer_for_tf(tf: str) -> str:
    """The native bar_cache layer a store TF resamples FROM — the canonical BASE.
    THE SEAM for the future sub-minute phase:
      - COARSE (tf_seconds >= 120): resample from **1Min** (this phase).
      - SUB-MINUTE (tf_seconds < 60): resample from **1Second** (Phase 2b — a
        DIFFERENT base layer: ~20x the volume + the prime1s-incident risk zone; NOT
        seeded this phase). Only sid 339 has a sub-minute secondary gate (['30m','10s'])
        and it's inactive — no rush, but the seam is here so Phase 2b drops in by
        extending this mapping + widening the target filter, with no other rework.
    1Min itself is the base (served from bar_cache directly, never from this store)."""
    s = tf_seconds(tf)
    if s < 60:
        return "1Sec"   # Phase 2b (sub-minute-from-1Second) — reserved seam
    return "1Min"       # coarse (>=120s) — this phase


# =============================================================================
# The canonical oracle — the single definition the store must reproduce
# =============================================================================

def _load_base_readonly(symbol: str, tf: str, start, end) -> Optional[pd.DataFrame]:
    """Read the BASE layer for `tf` (base_layer_for_tf — coarse→1Min, sub-minute→1Sec)
    for [start, end] READ-ONLY (bar_cache.read_bars = a plain direct-PG SELECT). NEVER
    fetches Polygon and NEVER writes the cache — the store is a read-only consumer of
    the base layer (the data-worker owns keeping it fresh). Returns RAW all-hours base
    bars (session filter applied by the caller), or None."""
    import bar_cache
    return bar_cache.read_bars(symbol, base_layer_for_tf(tf), start, end)


def _load_1min_readonly(symbol: str, start, end) -> Optional[pd.DataFrame]:
    """Back-compat convenience — the 1Min base read (coarse TFs). Prefer
    `_load_base_readonly(symbol, tf, ...)` so the base layer follows the TF."""
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
        one_min_df = _load_base_readonly(symbol, tf, start, end)
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
        one_min_df = _load_base_readonly(symbol, tf, start, end)
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
        one_min_df = _load_base_readonly(symbol, tf, start, end)
    built = build_window(symbol, tf, session, start, end, one_min_df=one_min_df)
    canon = canonical_resampled(symbol, tf, session, start, end,
                                one_min_df=one_min_df)
    return compare_store_vs_canonical(built, canon, symbol=symbol, tf=tf,
                                      session=session)


# =============================================================================
# M1b — PERSISTENCE (WINDOW-REPLACE), store read, comparator shadow.
# All gated: store_window is a no-op unless RORT_RESAMPLED_STORE_WRITE. read_store
# is inert until rows exist. Nothing here is wired into a live consumer — the
# consumer cutovers (backtest secondary -> ribbon -> LIVE shadow) are separate,
# each byte-identity-gated. The live-shadow cutover (#3) is HELD for coordinated
# review (it must read the row matching the STRATEGY's session — RTH-4H != Ext-4H;
# that is the 313 fix made structural).
# =============================================================================

def _dsn() -> Optional[str]:
    return os.environ.get("SUPABASE_CONNECTION_STRING")


def _utc_days(start, end) -> list:
    """Inclusive list of UTC-day midnight Timestamps covered by [start, end]. The
    window-REPLACE unit — every day in the requested range is replaced (so a day
    whose 1Min was fully RETRACTED gets its stale coarse rows deleted, which an
    upsert of only the surviving buckets could never do)."""
    d = pd.Timestamp(start).tz_convert("UTC").normalize()
    last = pd.Timestamp(end).tz_convert("UTC").normalize()
    out = []
    while d <= last:
        out.append(d)
        d = d + pd.Timedelta(days=1)
    return out


def store_window(symbol: str, tf: str, session: str, start, end,
                 *, one_min_df: Optional[pd.DataFrame] = None) -> dict:
    """Build the store's coarse bars for the window and PERSIST them WINDOW-REPLACE,
    one atomic transaction PER (symbol, tf, session, UTC-day): DELETE the day scope,
    INSERT that day's freshly-built bars. Delete-then-insert (not upsert) so a
    retracted bucket's stale row is removed. Per-day commit keeps transactions tiny
    (no idle-in-txn bloat — the #7 lesson) and models the real incremental path
    (a settle/revise re-resamples ONE day and replaces it).

    Gated by RORT_RESAMPLED_STORE_WRITE — a no-op when off. Returns
    {days, rows, retracted_days}."""
    if not writethrough_enabled():
        return {"skipped": "RORT_RESAMPLED_STORE_WRITE off", "days": 0, "rows": 0}
    dsn = _dsn()
    if not dsn:
        return {"error": "no DSN", "days": 0, "rows": 0}
    import psycopg
    from datetime import datetime, timedelta, timezone

    tf_s = tf_seconds(tf)
    built = build_window(symbol, tf, session, start, end, one_min_df=one_min_df)
    # Group built rows by UTC-day.
    by_day: dict = {}
    for ts, r in built.iterrows():
        day = pd.Timestamp(ts).tz_convert("UTC").normalize()
        by_day.setdefault(day, []).append((ts, r))

    settle_min = _settle_min()
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    days = _utc_days(start, end)
    total_rows = 0
    retracted = 0
    ins_sql = ("INSERT INTO resampled_bar_cache (symbol, timeframe_seconds, session, "
               "ts, open, high, low, close, volume, source_1min_span_hash, settled, "
               "revised_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)")
    del_sql = ("DELETE FROM resampled_bar_cache WHERE symbol=%s AND "
               "timeframe_seconds=%s AND session=%s AND ts >= %s AND ts < %s")
    with psycopg.connect(dsn, connect_timeout=20, prepare_threshold=None,
                         autocommit=False) as conn:
        with conn.cursor() as cur:
            for day in days:
                day_start = day.isoformat()
                day_end = (day + pd.Timedelta(days=1)).isoformat()
                cur.execute(del_sql, (symbol, tf_s, session, day_start, day_end))
                deleted = cur.rowcount
                rows = by_day.get(day, [])
                if not rows and deleted > 0:
                    retracted += 1  # a day that had rows now has none (retraction)
                payload = []
                for ts, r in rows:
                    bar_end = ts + timedelta(seconds=tf_s)
                    settled = bool(bar_end <= now - timedelta(minutes=settle_min))
                    payload.append((
                        symbol, tf_s, session, ts.isoformat(),
                        float(r["open"]), float(r["high"]), float(r["low"]),
                        float(r["close"]), float(r["volume"]),
                        r.get("source_1min_span_hash"), settled, now_iso))
                if payload:
                    cur.executemany(ins_sql, payload)
                    total_rows += len(payload)
                conn.commit()  # atomic per day
    return {"days": len(days), "rows": total_rows, "retracted_days": retracted}


def _settle_min() -> int:
    """Reuse bar_cache's settle threshold so the store and the 1Min cache agree on
    what 'settled' means."""
    from bar_cache import _settle_min as _sm
    return _sm()


def _settled_cutoff_ts(tf: str, *, now=None):
    """The newest bar-start ts that is fully SETTLED for `tf`: a bar labeled T covers
    [T, T+tf), so it's closed at now>=T+tf and settled at now>=T+tf+settle_min →
    T <= now - settle_min - tf. Bars at/before this are permanent truth; anything
    after is the still-forming / just-closed WS tip (expected to differ)."""
    from datetime import datetime, timedelta, timezone
    if now is None:
        now = datetime.now(timezone.utc)
    return now - timedelta(minutes=_settle_min()) - timedelta(seconds=tf_seconds(tf))


def _trim_settled(df, tf: str, *, now=None):
    """Trim a bar series to the SETTLED bars only (drop the forming/just-closed tip).
    Used to exclude the WS-tip bucket from byte-identity 'green' — the forming bar
    legitimately differs (store snapshot vs fresher canonical) and must not count."""
    if df is None or len(df) == 0:
        return df
    return df[df.index <= _settled_cutoff_ts(tf, now=now)]


def read_store(symbol: str, tf: str, session: str, start, end
               ) -> Optional[pd.DataFrame]:
    """Read persisted coarse bars for (symbol, tf, session) in [start, end] over a
    direct-PG SELECT (same fast path as bar_cache.read_bars). Returns an OHLCV
    DataFrame (UTC DatetimeIndex) or None if no DSN / no rows. This is what the
    comparator and future consumers read."""
    dsn = _dsn()
    if not dsn:
        return None
    import psycopg
    tf_s = tf_seconds(tf)
    # Coarse bars are LABELED at their bin START, which PRECEDES their data (a 1Day
    # bar is labeled 00:00 but aggregates the whole day; a window starting mid-day
    # still wants that bar because it contains in-window 1Min — exactly as the
    # on-the-fly resample produces it). So floor the low bound to the bin boundary,
    # else the first bar is wrongly excluded when `start` falls mid-bin (a bar-set
    # mismatch vs canonical). Store TFs divide the day and anchor at UTC midnight, so
    # epoch-flooring (Timestamp.floor) matches the resample bins. No-op for
    # midnight-aligned windows.
    lo = pd.Timestamp(start)
    if lo.tzinfo is None:
        lo = lo.tz_localize("UTC")
    lo = lo.floor(f"{tf_s}s")
    s = lo.isoformat()
    e = end.isoformat() if hasattr(end, "isoformat") else end
    sql = ("select ts, open, high, low, close, volume from resampled_bar_cache "
           "where symbol=%s and timeframe_seconds=%s and session=%s "
           "and ts between %s and %s order by ts")
    try:
        with psycopg.connect(dsn, connect_timeout=15, prepare_threshold=None,
                             autocommit=True) as conn:
            # BINARY result format — protocol-exact float8. The DEFAULT text format
            # rounds float8 to ~15 sig digits (pooler extra_float_digits), which
            # drops the last ULP of a RESAMPLED volume that carries a long
            # fractional tail (e.g. 1986300.6153870001 -> ...387) — a 1e-10 diff
            # from the on-the-fly canonical. Byte-identity is the acceptance
            # criterion, so the store must return the stored double EXACTLY. (The
            # write is already lossless; binary read makes the round-trip exact and
            # doesn't depend on a session GUC surviving transaction pooling.
            # bar_cache.read_bars is unaffected: native 1Min volumes are
            # integer-valued, so text round-trips them exactly.)
            with conn.cursor(binary=True) as cur:
                cur.execute(sql, (symbol, tf_s, session, s, e))
                rows = cur.fetchall()
    except Exception as ex:  # noqa: BLE001
        logger.warning("resampled_bar_store.read_store failed %s/%s/%s: %s",
                       symbol, tf, session, ex)
        return None
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts")
    df.index.name = None
    for col in OHLCV:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def shadow_check(symbol: str, tf: str, session: str, start, end,
                 *, one_min_df: Optional[pd.DataFrame] = None,
                 persist: bool = False, settled_only: bool = False) -> dict:
    """The read-time comparator SHADOW: (optionally persist, gated), then read the
    STORE back and byte-compare it to the canonical resample. This is stronger than
    the M1a offline check because it validates the PERSISTENCE ROUND-TRIP (Postgres
    tz/float) — the store the consumer would actually read must equal canonical.
    Alarms (logs) on any diff. Returns the comparator dict.

    settled_only: trim BOTH sides to settled bars before comparing (exclude the
    forming/just-closed WS-tip bucket, which legitimately differs). The right 'green'
    criterion during RTH."""
    if persist and writethrough_enabled():
        store_window(symbol, tf, session, start, end, one_min_df=one_min_df)
    stored = read_store(symbol, tf, session, start, end)
    canon = canonical_resampled(symbol, tf, session, start, end,
                                one_min_df=one_min_df)
    if settled_only:
        stored = _trim_settled(stored, tf)
        canon = _trim_settled(canon, tf)
    res = compare_store_vs_canonical(stored, canon, symbol=symbol, tf=tf,
                                     session=session)
    if not res["match"]:
        detail = (res.get("note") or f"{len(res['cell_diffs'])} cell diffs")
        logger.warning("[RESAMPLED-STORE-SHADOW] DIFF %s/%s/%s: %s",
                       symbol, tf, session, detail)
    return res


# =============================================================================
# MAINTENANCE / BACKFILL LOOP (§5) + SHADOW COMPARATOR (§7) + COVERAGE (§8)
# Hardened per the prime1s DOS incident (chunk + circuit breaker + conservative
# cadence — the shared Supabase 522s under mass single-shot load). All WRITES gated
# RORT_RESAMPLED_STORE_WRITE (default OFF; the store is truncatable/reversible). NOT
# wired into any cron/worker — driven by _resampled_store_maintain.py. The heavy
# historical seed runs OFF-PEAK (after the 20:00Z close) so it never contends with
# the trading path. The §7 comparator must be green for N trading days BEFORE any
# consumer READ flag is flipped on in prod.
# =============================================================================

_BACKFILL_CHUNK_DAYS = 30      # day-chunk the seed — bounded write per call
_BREAKER_MAX_FAIL = 3          # consecutive chunk/target failures → open breaker
_CHUNK_PAUSE_S = 0.5           # conservative cadence between chunks (Supabase-kind)


SEED_SYMBOLS = ["TSLA"]                       # this phase (guardrail; extend deliberately)
SEED_SESSIONS = ["RTH", "Extended Hours"]


def _load_timeframes_config() -> dict:
    """Read the persisted Timeframes-page config: settings.enabled_timeframes
    (Record<tfId, {confluenceEnabled, ...}>) from src/settings.json — the same source
    the /confluence-packs/timeframes page saves to. {} when unset (defaults apply)."""
    import json
    path = os.path.join(os.path.dirname(__file__), "settings.json")
    try:
        with open(path) as f:
            return (json.load(f) or {}).get("enabled_timeframes", {}) or {}
    except Exception:
        return {}


def confluence_enabled_coarse_tfs() -> list:
    """The gate-eligible COARSE TF catalog, CONFIG-DERIVED from the Timeframes page:
    every resample target with tf_seconds >= 120 that is CONFLUENCE-ENABLED. Enabling
    a TF on /confluence-packs/timeframes auto-adds it to seed + maintenance (no code
    change — Kevin's 'mechanism to add new timeframes'); disabling drops it. Excludes
    1Min (the base, served from bar_cache), sub-minute (<120s → Phase 2b from
    1Second), and 1Week/1Month (chart-display-only — no resample rule / not
    confluence-enabled). Default (config unset): all resamplable coarse TFs (2Min..1Day)."""
    from data_loader import _RESAMPLE_RULES
    cfg = _load_timeframes_config()
    enabled = {tf: True for tf in _RESAMPLE_RULES if tf_seconds(tf) >= 120}
    for tf, flags in cfg.items():
        if (isinstance(flags, dict) and "confluenceEnabled" in flags
                and tf in _RESAMPLE_RULES and tf_seconds(tf) >= 120):
            enabled[tf] = bool(flags["confluenceEnabled"])
    return sorted((tf for tf, on in enabled.items() if on), key=tf_seconds)


def _capture_symbols() -> list:
    """Symbols from the bar-cache CAPTURE-TARGET authority (the same source the
    settle sweeper uses) — i.e. the symbols whose 1Min base layer is actively
    maintained, which is exactly the store's base-layer requirement. Store
    targets follow it so enabling a new capture symbol auto-joins it to
    seed/maintain/shadow/coverage (the symbol analogue of the Timeframes-page
    TF derivation). Empty list on discovery failure → caller falls back to
    SEED_SYMBOLS (never silently zero targets)."""
    try:
        from bar_cache import get_capture_targets
        seen: list = []
        for t in get_capture_targets(enabled_only=True):
            s = t.get("symbol")
            if s and s not in seen:
                seen.append(s)
        return sorted(seen)
    except Exception as e:  # noqa: BLE001
        logger.warning("resampled_bar_store: capture-symbol discovery failed "
                       "(%s) — falling back to %s", e, SEED_SYMBOLS)
        return []


def default_coarse_targets() -> list:
    """Seed/maintenance target set = capture symbols × confluence_enabled_coarse_tfs()
    × SEED_SESSIONS. FULLY CONFIG-DERIVED both axes (2026-07-10 extension): TFs follow
    the Timeframes page; symbols follow the Bar Cache capture authority (was TSLA-only
    guardrail — extended deliberately after consumer #3's live verify surfaced SPY
    'store uncovered' on real fleet gates)."""
    tfs = confluence_enabled_coarse_tfs()
    syms = _capture_symbols() or list(SEED_SYMBOLS)
    return [(s, tf, sess) for s in syms for tf in tfs for sess in SEED_SESSIONS]


def seed_days_for_tf(tf: str, *, warmup_bars: int = 250, floor_days: int = 90,
                     cap_days: int = 400) -> int:
    """TF-adaptive seed depth: enough calendar days for ~warmup_bars of `tf`, floored
    so FINE TFs still cover a ribbon window and capped for the coarsest. Coarse
    (1Day)≈362, 4Hour≈181, fine (2m/5m)→floor. Fine TFs dominate row count — the floor
    bounds it (a uniform deep window would over-seed them)."""
    import math
    from data_loader import BARS_PER_DAY
    bpd = BARS_PER_DAY.get(tf, 390) or 390
    days = math.ceil(warmup_bars / bpd * 365 / 252)
    return max(floor_days, min(cap_days, days))


class _Breaker:
    """Trip after `max_fail` consecutive failures — the prime1s circuit breaker."""
    def __init__(self, max_fail: int = _BREAKER_MAX_FAIL):
        self.max_fail = max_fail
        self.fails = 0
        self.open = False

    def ok(self):
        self.fails = 0

    def fail(self) -> bool:
        self.fails += 1
        if self.fails >= self.max_fail:
            self.open = True
        return self.open


def backfill_coarse(symbol: str, tf: str, session: str, start, end, *,
                    chunk_days: Optional[int] = None, compare: bool = True,
                    breaker: Optional["_Breaker"] = None,
                    pause_s: Optional[float] = None) -> dict:
    """Day-chunked historical SEED for (symbol, tf, session) over [start, end].
    WINDOW-REPLACE per chunk (idempotent, re-runnable). CIRCUIT BREAKER aborts after
    consecutive chunk failures (Supabase 522 protection). Conservative pause between
    chunks. When `compare`, runs the §7 shadow comparator per chunk and tallies
    diffs. Gated RORT_RESAMPLED_STORE_WRITE. Returns
    {chunks, rows, diff_chunks, aborted}."""
    if not writethrough_enabled():
        return {"skipped": "RORT_RESAMPLED_STORE_WRITE off", "chunks": 0, "rows": 0}
    import time
    step = int(chunk_days or _BACKFILL_CHUNK_DAYS)
    pause = _CHUNK_PAUSE_S if pause_s is None else pause_s
    br = breaker or _Breaker(_BREAKER_MAX_FAIL)
    cur = pd.Timestamp(start)
    endts = pd.Timestamp(end)
    if cur.tzinfo is None:
        cur = cur.tz_localize("UTC")
    if endts.tzinfo is None:
        endts = endts.tz_localize("UTC")
    # CRITICAL: chunk boundaries MUST be UTC-day-aligned. The write unit is
    # WINDOW-REPLACE per UTC-day, so a day split across two chunks would be REPLACED
    # by the 2nd chunk's PARTIAL view of that day (it only loads that day's later
    # 1Min) — silently dropping the day's earlier bars. Flooring `cur` to midnight
    # makes every whole day belong to exactly one chunk (only the final chunk ends
    # mid-day, at `end`=now, and no earlier chunk touches that day). Proven: without
    # this, a 3-day-chunk seed lost 07-08 12:00 (store 9 vs canonical 10).
    cur = cur.normalize()
    chunks = rows = diff_chunks = 0
    aborted = False
    last_error = None
    while cur < endts:
        ce = min(cur + pd.Timedelta(days=step), endts)
        try:
            w = store_window(symbol, tf, session, cur.to_pydatetime(),
                             ce.to_pydatetime())
            rows += w.get("rows", 0)
            chunks += 1
            br.ok()
            if compare:
                res = shadow_check(symbol, tf, session, cur.to_pydatetime(),
                                   ce.to_pydatetime(), settled_only=True)
                if not res["match"]:
                    diff_chunks += 1
                    logger.warning("[RESAMPLED-STORE-SEED] SETTLED DIFF %s/%s/%s [%s..%s]: %s",
                                   symbol, tf, session, cur.date(), ce.date(),
                                   res.get("note") or f"{len(res['cell_diffs'])} diffs")
        except Exception as e:  # noqa: BLE001
            last_error = str(e)[:200]
            logger.warning("[RESAMPLED-STORE-SEED] chunk fail %s/%s/%s [%s..%s]: %s",
                           symbol, tf, session, cur.date(), ce.date(), e)
            if br.fail():
                aborted = True
                logger.error("[RESAMPLED-STORE-SEED] BREAKER OPEN — aborting "
                             "%s/%s/%s after %d fails", symbol, tf, session, br.fails)
                break
        cur = ce
        if pause and cur < endts:
            time.sleep(pause)
    return {"chunks": chunks, "rows": rows, "diff_chunks": diff_chunks,
            "aborted": aborted, "last_error": last_error}


def maintain_coarse(symbol: str, tf: str, session: str, *,
                    recent_days: int = 3) -> dict:
    """Incremental maintenance: re-resample + WINDOW-REPLACE the recent `recent_days`
    window (where the underlying 1Min may have settled / been revised since the store
    was built — the T+2 revision / settle-sweeper class). Bounded + cheap; runs
    alongside the settle-sweeper cadence. Gated. Returns store_window's result."""
    if not writethrough_enabled():
        return {"skipped": "RORT_RESAMPLED_STORE_WRITE off"}
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    # Floor start to UTC midnight so window-REPLACE rebuilds WHOLE days. A mid-day
    # start would partial-replace the start-day — delete the whole UTC-day, then
    # rebuild it from only that day's LATER 1Min → drop its morning bars (the
    # corruption seen 2026-07-10). Whole-day rebuild also refreshes the settled edge.
    # (store_window is deliberately NOT floored globally: a mid-day-window consumer
    # read floors the low bound too, so the store holds complete days but a partial
    # first bar still round-trips — 1Day consumer #1 relies on that.)
    start = (now - timedelta(days=recent_days)).replace(
        hour=0, minute=0, second=0, microsecond=0)
    return store_window(symbol, tf, session, start, now)


def maintain_all_coarse(targets: Optional[list] = None, *,
                        recent_days: int = 3) -> dict:
    """Cron entrypoint: incrementally maintain every target's recent window. Breaker
    across targets. Gated. Returns {targets, rows, errors, aborted}."""
    if not writethrough_enabled():
        return {"skipped": "RORT_RESAMPLED_STORE_WRITE off"}
    targets = targets or default_coarse_targets()
    br = _Breaker(_BREAKER_MAX_FAIL)
    rows = 0
    errors = 0
    aborted = False
    for (s, tf, sess) in targets:
        try:
            w = maintain_coarse(s, tf, sess, recent_days=recent_days)
            rows += w.get("rows", 0)
            br.ok()
        except Exception as e:  # noqa: BLE001
            errors += 1
            logger.warning("maintain_all_coarse %s/%s/%s: %s", s, tf, sess, e)
            if br.fail():
                aborted = True
                logger.error("maintain_all_coarse BREAKER OPEN after %d fails", br.fails)
                break
    return {"targets": len(targets), "rows": rows, "errors": errors,
            "aborted": aborted}


def shadow_compare_targets(targets: Optional[list] = None, *,
                           days: int = 5, settled_only: bool = False) -> list:
    """§7 shadow comparator over targets — READ-ONLY (no writes). For each target,
    reads the store vs the on-the-fly canonical over the last `days` and byte-compares.
    This is the gate that must be GREEN for N trading days before any consumer READ
    flag is flipped on in prod. `settled_only` trims the forming/just-closed WS-tip
    bucket (the right 'green' criterion during RTH). Returns comparator dicts."""
    from datetime import datetime, timedelta, timezone
    targets = targets or default_coarse_targets()
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    out = []
    for (s, tf, sess) in targets:
        stored = read_store(s, tf, sess, start, now)
        canon = canonical_resampled(s, tf, sess, start, now)
        if settled_only:
            stored = _trim_settled(stored, tf, now=now)
            canon = _trim_settled(canon, tf, now=now)
        out.append(compare_store_vs_canonical(stored, canon, symbol=s, tf=tf,
                                               session=sess))
    return out


def resampled_supply_coverage(targets: Optional[list] = None, *,
                              sample_days: int = 5) -> list:
    """§8 admin observability. Per (symbol, tf, session): coverage span (min/max ts),
    freshness (revised_at of the edge bar), row count, and — the MONEY COLUMN — a live
    comparator sample (store vs canonical over the last `sample_days`) so drift is
    visible BEFORE it bites. READ-ONLY. Drives the Resampled Bar Store admin page."""
    from datetime import datetime, timedelta, timezone
    targets = targets or default_coarse_targets()
    dsn = _dsn()
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=sample_days)
    out = []
    for (s, tf, sess) in targets:
        tf_s = tf_seconds(tf)
        cov = {"min_ts": None, "max_ts": None, "rows": 0, "last_revised": None}
        if dsn:
            try:
                import psycopg
                with psycopg.connect(dsn, connect_timeout=15, prepare_threshold=None,
                                     autocommit=True) as c:
                    with c.cursor() as cur:
                        cur.execute(
                            "select min(ts), max(ts), count(*), max(revised_at) "
                            "from resampled_bar_cache where symbol=%s and "
                            "timeframe_seconds=%s and session=%s", (s, tf_s, sess))
                        mn, mx, cnt, rev = cur.fetchone()
                cov = {"min_ts": mn.isoformat() if mn else None,
                       "max_ts": mx.isoformat() if mx else None,
                       "rows": int(cnt or 0),
                       "last_revised": rev.isoformat() if rev else None}
            except Exception as e:  # noqa: BLE001
                logger.warning("resampled_supply_coverage cov %s/%s/%s: %s",
                               s, tf, sess, e)
        # live comparator sample (the drift signal)
        try:
            stored = read_store(s, tf, sess, start, now)
            canon = canonical_resampled(s, tf, sess, start, now)
            cmp = compare_store_vs_canonical(stored, canon, symbol=s, tf=tf,
                                             session=sess)
            n = cmp.get("n_canonical") or cmp.get("n_built") or 0
            diff_cells = len(cmp.get("cell_diffs") or [])
            idx_mm = bool(cmp.get("index_mismatch"))
            diff_rate = (diff_cells / (n * 5)) if n else None
        except Exception as e:  # noqa: BLE001
            cmp = {"match": None}
            diff_cells = idx_mm = diff_rate = None
            logger.warning("resampled_supply_coverage cmp %s/%s/%s: %s",
                           s, tf, sess, e)
        out.append({"symbol": s, "tf": tf, "session": sess, **cov,
                    "comparator_match": cmp.get("match"),
                    "comparator_diff_cells": diff_cells,
                    "comparator_index_mismatch": idx_mm,
                    "comparator_diff_rate": diff_rate,
                    "comparator_sample_days": sample_days})
    return out
