"""Phase 4.5 — Per-strategy multi-grace shadow comparison.

For a given strategy on a sub-minute timeframe with grace shadow data
available, replays the unified backtest engine against each grace
variant's recorded bars (`live_bars_trades.source = 'grace_Ns'`) and
reports per-variant trade counts + close-match % against the highest-
fidelity variant.

Per LEF spec §5.5: "recommendation-first, not raw tables." Returns a
dict shaped for direct UI consumption:
  {
    'available': bool,
    'reason': str if not available else None,
    'symbol': str, 'tf_seconds': int, 'window_start': iso, 'window_end': iso,
    'reference_variant': 'grace_7s',
    'variants': {
      'grace_2s': {'grace': 2, 'trade_count': N, 'close_match_pct': 0.83, ...},
      ...
    },
    'recommendation': {
      'grace': 5,
      'label': 'Balanced',
      'rationale': "...",
    }
  }

Engine-risk assessment (verified 2026-05-20):
  - Reads from live_bars_trades only — does not write
  - Instantiates fresh UnifiedStrategy per variant — no shared state
  - Runs run_unified_backtest which is stateless per call
  - No live-engine touch — purely analytical
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Variants we compare. Anchored to the grace_shadow.py source labels.
# Order matters: lowest-to-highest grace, reference last.
GRACE_VARIANTS: list[tuple[str, float]] = [
    ('grace_2s', 2.0),
    ('grace_3s', 3.0),
    ('grace_4s', 4.0),
    ('grace_5s', 5.0),
    ('grace_6s', 6.0),
    ('grace_7s', 7.0),
]

# Threshold for a "close match" on entry price: $0.01 default. Tighter
# than the spec example ($0.10) since most strategies' entries land
# within a cent already; the laxer threshold would hide drift.
CLOSE_MATCH_TOLERANCE_USD = 0.01

# Symbols + TF the grace shadow currently records (grace_shadow.py).
SUPPORTED_SYMBOLS = {'SPY', 'TSLA'}
SUPPORTED_TF_SECONDS = 10


def fetch_shadow_bars(sb, symbol: str, source: str,
                      tf_seconds: int,
                      start_iso: str, end_iso: str) -> pd.DataFrame:
    """Load grace shadow bars from live_bars_trades for one variant.

    Returns a DataFrame indexed by bar_start (UTC) with OHLCV columns.
    """
    PAGE = 1000
    MAX_ROWS = 200_000
    out: list = []
    offset = 0
    while offset < MAX_ROWS:
        res = sb.table('live_bars_trades').select(
            'bar_start, open, high, low, close, volume'
        ).eq('symbol', symbol.upper()) \
         .eq('source', source) \
         .eq('timeframe_seconds', tf_seconds) \
         .gte('bar_start', start_iso) \
         .lt('bar_start', end_iso) \
         .order('bar_start') \
         .range(offset, offset + PAGE - 1) \
         .execute()
        page = res.data or []
        out.extend(page)
        if len(page) < PAGE:
            break
        offset += PAGE
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    df['bar_start'] = pd.to_datetime(df['bar_start'], utc=True)
    df = df.set_index('bar_start').sort_index()
    # Coerce numeric columns
    for c in ('open', 'high', 'low', 'close', 'volume'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def _run_backtest_on_variant(df: pd.DataFrame, strategy: dict) -> pd.DataFrame:
    """Wrap run_unified_backtest with the call signature this service uses.

    Returns trades_df with at minimum entry_time + entry_price columns,
    or an empty DataFrame on failure.
    """
    if len(df) < 2:
        return pd.DataFrame()
    try:
        from unified_engine import run_unified_backtest
        import general_packs as gp_module
        enabled_gen = gp_module.get_enabled_general_packs(
            gp_module.load_general_packs())
        trades, _enriched = run_unified_backtest(
            df, strategy,
            general_packs=enabled_gen,
            include_open_position=False,
            last_bar_partial=False,
        )
        if not isinstance(trades, pd.DataFrame):
            return pd.DataFrame()
        # Bridge entry_fill_ts → entry_time (same as services.unified_trades)
        if 'entry_fill_ts' in trades.columns and 'entry_time' not in trades.columns:
            trades['entry_time'] = pd.to_datetime(
                trades['entry_fill_ts'], utc=True, errors='coerce')
        return trades
    except Exception as e:
        logger.warning("grace shadow compare: backtest failed for variant: %s", e)
        return pd.DataFrame()


def _compute_close_match(variant_trades: pd.DataFrame,
                         reference_trades: pd.DataFrame) -> float:
    """Fraction of reference trades whose entry price falls within
    CLOSE_MATCH_TOLERANCE_USD of any same-timestamp variant trade.
    Returns 0.0 if reference empty.
    """
    if len(reference_trades) == 0:
        return 0.0
    if len(variant_trades) == 0:
        return 0.0

    matches = 0
    # Build a lookup by rounded-to-bar timestamp for fast nearest-match
    # check. Bars are 10s wide; we match by exact entry_time alignment
    # plus a small tolerance.
    var_by_ts: dict = {}
    for _, t in variant_trades.iterrows():
        ts = t.get('entry_time')
        if pd.isna(ts) or ts is None:
            continue
        var_by_ts[pd.Timestamp(ts)] = float(t.get('entry_price', 0))

    for _, t in reference_trades.iterrows():
        ref_ts = t.get('entry_time')
        if pd.isna(ref_ts) or ref_ts is None:
            continue
        ref_price = float(t.get('entry_price', 0))
        ref_ts_pd = pd.Timestamp(ref_ts)
        # Match if variant has a trade at the same timestamp within $tol.
        # Walk var_by_ts; for tiny lists this is fine.
        for v_ts, v_price in var_by_ts.items():
            if abs((v_ts - ref_ts_pd).total_seconds()) <= 0.001:
                if abs(v_price - ref_price) <= CLOSE_MATCH_TOLERANCE_USD:
                    matches += 1
                break
    return matches / len(reference_trades)


def _pick_recommendation(variants: dict, target_pct: float = 0.95) -> dict:
    """Lowest grace where close_match_pct >= target — that's the
    'fastest tier that still matches reference.' Falls back to the
    reference itself if no variant clears the threshold.
    """
    # variants is {label: {grace, trade_count, close_match_pct, ...}}
    sorted_variants = sorted(
        variants.values(), key=lambda v: v['grace']
    )
    # Skip the reference itself (highest grace by construction)
    for v in sorted_variants[:-1]:
        if v['close_match_pct'] >= target_pct:
            return {
                'grace': v['grace'],
                'label': _grace_to_tier_label(v['grace']),
                'rationale': (
                    f"Grace {v['grace']:.0f}s achieves "
                    f"{v['close_match_pct']*100:.1f}% close-match vs "
                    f"reference (grace 7s) with "
                    f"{v['trade_count']} trades. Lower grace = faster "
                    f"alerts; higher = closer to backtest fidelity."
                ),
            }
    # Fallback: recommend the reference
    ref = sorted_variants[-1]
    return {
        'grace': ref['grace'],
        'label': _grace_to_tier_label(ref['grace']),
        'rationale': (
            f"No faster variant achieved >= {target_pct*100:.0f}% close-match; "
            f"highest-fidelity (grace {ref['grace']:.0f}s) recommended."
        ),
    }


def _grace_to_tier_label(grace: float) -> str:
    """Map a grace seconds value to the named tier from the Phase 5 UI."""
    if grace <= 3:
        return 'Fastest'
    if grace <= 5:
        return 'Balanced'
    return 'Highest-fidelity'


def compare_strategy(strategy: dict, window_hours: int = 6) -> dict:
    """Public API. Returns dict shaped for direct UI consumption.

    `window_hours` defaults to 6 — recent RTH window. Increase for more
    statistical power; decrease for faster turnaround.
    """
    symbol = (strategy.get('symbol') or '').upper()
    tf_str = strategy.get('timeframe') or '1Min'
    # Resolve tf_seconds
    from ralph_engine import TIMEFRAME_SECONDS
    tf_seconds = TIMEFRAME_SECONDS.get(tf_str, 60)

    if symbol not in SUPPORTED_SYMBOLS:
        return {
            'available': False,
            'reason': (
                f"Grace shadow recorded for {sorted(SUPPORTED_SYMBOLS)} only; "
                f"this strategy is on {symbol}."
            ),
        }
    if tf_seconds != SUPPORTED_TF_SECONDS:
        return {
            'available': False,
            'reason': (
                f"Grace shadow recorded at {SUPPORTED_TF_SECONDS}s only; "
                f"this strategy is at {tf_seconds}s."
            ),
        }

    # Window: now - window_hours → now
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=window_hours)
    start_iso = start.isoformat()
    end_iso = end.isoformat()

    from db import get_admin_client
    sb = get_admin_client()

    # Cache trades per variant so we don't re-fetch + re-run for the
    # close-match pass. For typical 6-hour windows this saves 6 round-
    # trips per call; memory cost is at most a few hundred KB of DataFrames.
    variants_out: dict = {}
    per_variant_trades: dict[str, pd.DataFrame] = {}

    for label, grace in GRACE_VARIANTS:
        bars = fetch_shadow_bars(sb, symbol, label, tf_seconds,
                                 start_iso, end_iso)
        if len(bars) == 0:
            variants_out[label] = {
                'grace': grace,
                'trade_count': 0,
                'close_match_pct': 0.0,
                'bars_available': 0,
                'note': 'no shadow bars found for window',
            }
            per_variant_trades[label] = pd.DataFrame()
            continue
        trades = _run_backtest_on_variant(bars, strategy)
        per_variant_trades[label] = trades
        variants_out[label] = {
            'grace': grace,
            'trade_count': int(len(trades)),
            'close_match_pct': None,  # filled below vs reference
            'bars_available': int(len(bars)),
        }

    reference_label = GRACE_VARIANTS[-1][0]
    reference_trades = per_variant_trades.get(reference_label, pd.DataFrame())
    if len(reference_trades) > 0:
        for label, _ in GRACE_VARIANTS:
            variants_out[label]['close_match_pct'] = _compute_close_match(
                per_variant_trades.get(label, pd.DataFrame()),
                reference_trades)
    else:
        for v in variants_out.values():
            v['close_match_pct'] = 0.0
            v['note'] = v.get('note') or 'reference produced no trades'

    recommendation = _pick_recommendation(variants_out)

    return {
        'available': True,
        'reason': None,
        'symbol': symbol,
        'tf_seconds': tf_seconds,
        'window_start': start_iso,
        'window_end': end_iso,
        'window_hours': window_hours,
        'reference_variant': GRACE_VARIANTS[-1][0],
        'variants': variants_out,
        'recommendation': recommendation,
    }
