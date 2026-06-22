"""Secondary-TF cache — fast Update-New-Data for coarse-gate strategies.

PROBLEM: strategies with a 1Hour+ secondary TF (e.g. 1d/4h gates) are routed to
the FULL warmup path (`_has_long_cycle_secondary_tf` guard,
forward_test_service.py) because the primary engine snapshot
(`serialize_backtest_snapshot`) only captures the PRIMARY indicator state — the
coarse secondary series would resume into undefined indicators. That forces a
months-long primary load every append (~700s on TSLA 15Sec / 1d/4h).

FIX (this module): cache the resampled secondary OHLCV. On append, load the cache
+ extend it with a SHORT fresh load from the last complete coarse bar onward, then
inject via the existing `secondary_tf_dfs` hook (services.prepare_data_with_indicators).
The primary load shrinks to the primary's own (short) warmup → big speedup.

FIDELITY: the merge is byte-identical to a fresh full resample (proven offline on
TSLA 1d/4h): cached COMPLETE bars + resample-from-boundary == full resample. Because
`resample_to_timeframe` is calendar-aligned and `run_all_indicators` is deterministic,
byte-identical OHLCV ⇒ byte-identical indicators ⇒ gates ⇒ trades. Fingerprint +
model_id keyed (mirrors the primary snapshot); any mismatch ⇒ caller falls back to
full rebuild. Kill-switch: RORT_SECONDARY_TF_CACHE (default OFF).

Cache stores COMPLETE coarse bars only (the last, possibly-forming, bar is dropped
at write time and re-formed on the next extend).
"""
from __future__ import annotations
import base64
import pickle
from typing import Dict, Optional

import pandas as pd

OHLCV = ['open', 'high', 'low', 'close', 'volume']
SECONDARY_CACHE_SCHEMA_VERSION = 1


def _period_seconds(target_tf: str) -> int:
    from unified_engine import TIMEFRAME_SECONDS
    return TIMEFRAME_SECONDS[target_tf]


def build_secondary_ohlcv(primary_df: pd.DataFrame,
                          secondary_tfs: tuple) -> Dict[str, pd.DataFrame]:
    """Resample the secondary OHLCV from a primary DataFrame — the SAME path
    `services.prepare_data_with_indicators` uses (resample_to_timeframe on
    df[OHLCV]). Drops the last (possibly-incomplete) coarse bar so the cache
    holds COMPLETE bars only. Keyed by canonical TF string (e.g. '1Day').
    """
    from data_loader import resample_to_timeframe
    out: Dict[str, pd.DataFrame] = {}
    if primary_df is None or len(primary_df) == 0:
        return out
    for tf in secondary_tfs:
        sec = resample_to_timeframe(primary_df[OHLCV].copy(), tf)
        if len(sec) > 1:
            sec = sec.iloc[:-1]  # drop last (possibly-forming) bar
        out[tf] = sec
    return out


def extend_secondary_ohlcv(cached: Dict[str, pd.DataFrame],
                           recent_primary_df: pd.DataFrame,
                           secondary_tfs: tuple) -> Dict[str, pd.DataFrame]:
    """Merge cached COMPLETE coarse bars with a fresh resample of
    `recent_primary_df` (a short load from the last complete cache bar onward).

    Byte-identical to a fresh full resample: keep cache bars, then append
    resampled-recent bars whose period start is >= (last cache bar + 1 period).
    The last bar of the result may again be incomplete → drop it (callers that
    persist re-cache via build_secondary_ohlcv; callers that consume keep it,
    matching prepare_data which keeps the forming bar then shifts/ffills).
    """
    from data_loader import resample_to_timeframe
    out: Dict[str, pd.DataFrame] = {}
    for tf in secondary_tfs:
        recent = resample_to_timeframe(recent_primary_df[OHLCV].copy(), tf) \
            if recent_primary_df is not None and len(recent_primary_df) else \
            pd.DataFrame(columns=OHLCV)
        c = cached.get(tf) if cached else None
        if c is None or len(c) == 0:
            out[tf] = recent
            continue
        cut = c.index[-1] + pd.Timedelta(seconds=_period_seconds(tf))
        recent = recent[recent.index >= cut]
        out[tf] = pd.concat([c, recent])
    return out


def serialize_secondary_cache(sec_dfs: Dict[str, pd.DataFrame],
                              fingerprint: str,
                              model_id: Optional[str],
                              last_bar_ts: str) -> Optional[str]:
    """Pickle+b64 the secondary OHLCV cache. Returns None on failure (caller
    falls back to full rebuild — never raises)."""
    try:
        envelope = {
            'schema_version': SECONDARY_CACHE_SCHEMA_VERSION,
            'fingerprint': fingerprint,
            'model_id': model_id,
            'last_bar_ts': last_bar_ts,
            # store as records to keep the blob small + version-stable
            'series': {tf: {'index': [t.isoformat() for t in df.index],
                            'data': df[OHLCV].to_dict('list')}
                       for tf, df in sec_dfs.items()},
        }
        return base64.b64encode(pickle.dumps(envelope)).decode('ascii')
    except Exception:
        return None


def deserialize_secondary_cache(
        b64: Optional[str],
        expected_fingerprint: Optional[str] = None,
        expected_model_id: Optional[str] = None) -> Optional[dict]:
    """Reverse serialize. Returns None on: empty, schema/fingerprint/model
    mismatch, or corruption → caller treats as 'no cache, full rebuild'.
    Returns {'last_bar_ts': str, 'series': {tf: DataFrame}}."""
    if not b64:
        return None
    try:
        env = pickle.loads(base64.b64decode(b64))
    except Exception:
        return None
    if not isinstance(env, dict):
        return None
    if env.get('schema_version') != SECONDARY_CACHE_SCHEMA_VERSION:
        return None
    if expected_fingerprint and env.get('fingerprint') != expected_fingerprint:
        return None
    if (expected_model_id and env.get('model_id') is not None
            and env.get('model_id') != expected_model_id):
        return None
    series: Dict[str, pd.DataFrame] = {}
    for tf, payload in (env.get('series') or {}).items():
        idx = pd.DatetimeIndex([pd.Timestamp(t) for t in payload['index']])
        series[tf] = pd.DataFrame(payload['data'], index=idx)
    return {'last_bar_ts': env.get('last_bar_ts'), 'series': series}
