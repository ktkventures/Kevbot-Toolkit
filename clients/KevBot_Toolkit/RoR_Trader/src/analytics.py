"""
Analytics Module for RoR Trader — Phase 11
============================================

Rolling performance metrics, Markov chain analysis, market regime detection,
edge decay scoring, and rule-based intelligence insights.

All functions operate on a trades DataFrame with at minimum:
    - r_multiple (float)
    - win (bool)
    - exit_time (datetime)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# =============================================================================
# ROLLING PERFORMANCE METRICS
# =============================================================================

def compute_rolling_metrics(
    trades_df: pd.DataFrame,
    window: int = 20,
) -> Dict[str, np.ndarray]:
    """Compute rolling Win Rate, Profit Factor, and Sharpe over a trade window.

    Args:
        trades_df: DataFrame with r_multiple and win columns.
        window: Rolling window size in trades.

    Returns:
        Dict with 'rolling_wr', 'rolling_pf', 'rolling_sharpe' arrays (same
        length as trades_df; NaN for first `window-1` entries).
    """
    r = trades_df["r_multiple"].values.astype(float)
    w = trades_df["win"].values.astype(float)
    n = len(r)

    rolling_wr = np.full(n, np.nan)
    rolling_pf = np.full(n, np.nan)
    rolling_sharpe = np.full(n, np.nan)

    for i in range(window - 1, n):
        sl = slice(i - window + 1, i + 1)
        chunk_r = r[sl]
        chunk_w = w[sl]

        # Win rate
        rolling_wr[i] = float(chunk_w.mean() * 100)

        # Profit factor
        wins_sum = float(chunk_r[chunk_r > 0].sum())
        losses_sum = float(abs(chunk_r[chunk_r < 0].sum()))
        rolling_pf[i] = wins_sum / losses_sum if losses_sum > 0 else np.nan

        # Sharpe (simplified: mean/std of R-multiples over window)
        m = chunk_r.mean()
        s = chunk_r.std(ddof=1)
        rolling_sharpe[i] = float(m / s) if s > 0 else np.nan

    return {
        "rolling_wr": rolling_wr,
        "rolling_pf": rolling_pf,
        "rolling_sharpe": rolling_sharpe,
    }


# =============================================================================
# MARKOV STATE TRANSITIONS
# =============================================================================

def compute_markov_transitions(
    trades_df: pd.DataFrame,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Compute Win/Loss Markov transition matrix.

    Returns:
        (probs, counts) where probs is {W_to_W, W_to_L, L_to_W, L_to_L} as
        floats 0-1, and counts is the raw transition counts.
    """
    wins = trades_df["win"].values
    counts = {"WW": 0, "WL": 0, "LW": 0, "LL": 0}

    for i in range(1, len(wins)):
        prev = "W" if wins[i - 1] else "L"
        curr = "W" if wins[i] else "L"
        counts[prev + curr] += 1

    total_from_w = counts["WW"] + counts["WL"]
    total_from_l = counts["LW"] + counts["LL"]

    probs = {
        "W_to_W": counts["WW"] / max(total_from_w, 1),
        "W_to_L": counts["WL"] / max(total_from_w, 1),
        "L_to_W": counts["LW"] / max(total_from_l, 1),
        "L_to_L": counts["LL"] / max(total_from_l, 1),
    }
    return probs, counts


def compute_streaks(trades_df: pd.DataFrame) -> List[int]:
    """Compute win/loss streak series.

    Returns a list of integers — positive for win streaks, negative for loss
    streaks. One entry per streak (not per trade).
    """
    wins = trades_df["win"].values
    if len(wins) == 0:
        return []

    streaks = []
    current = 1 if wins[0] else -1

    for i in range(1, len(wins)):
        if wins[i] == wins[i - 1]:
            current += 1 if wins[i] else -1
        else:
            streaks.append(current)
            current = 1 if wins[i] else -1
    streaks.append(current)
    return streaks


# =============================================================================
# EDGE DECAY & SCORES
# =============================================================================

def compute_edge_scores(
    trades_df: pd.DataFrame,
    window: int = 20,
    edge_threshold: float = 1.2,
) -> Dict[str, Optional[float]]:
    """Compute Consistency Score, Stability Index, and Trend Strength.

    Args:
        trades_df: DataFrame with r_multiple column.
        window: Rolling window size.
        edge_threshold: Profit Factor threshold for edge.

    Returns:
        Dict with 'consistency', 'stability', 'trend_strength' (0-100 each),
        'edge_status' ('strong'|'moderate'|'critical'|'lost'), and
        'current_pf' (latest rolling PF).
    """
    r = trades_df["r_multiple"].values.astype(float)
    n = len(r)

    if n < window:
        return {
            "consistency": None, "stability": None, "trend_strength": None,
            "edge_status": None, "current_pf": None,
        }

    # Compute rolling PF
    rolling_pf = []
    for i in range(window - 1, n):
        chunk = r[i - window + 1: i + 1]
        wins_sum = float(chunk[chunk > 0].sum())
        losses_sum = float(abs(chunk[chunk < 0].sum()))
        pf = wins_sum / losses_sum if losses_sum > 0 else np.nan
        rolling_pf.append(pf)

    rolling_pf = np.array(rolling_pf)
    valid = rolling_pf[~np.isnan(rolling_pf)]

    if len(valid) == 0:
        return {
            "consistency": None, "stability": None, "trend_strength": None,
            "edge_status": None, "current_pf": None,
        }

    # Consistency: % of windows where PF > threshold
    consistency = round(float(np.sum(valid > edge_threshold) / len(valid) * 100), 0)

    # Stability: 1 - CV (coefficient of variation), capped 0-100
    mean_pf = np.mean(valid)
    std_pf = np.std(valid, ddof=1) if len(valid) > 1 else 0
    if mean_pf > 0 and std_pf > 0:
        stability = round(float(max(0, min(100, (1 - std_pf / mean_pf) * 100))), 0)
    else:
        stability = 0.0

    # Trend Strength: normalized slope of rolling PF linear regression
    x = np.arange(len(valid))
    if len(valid) >= 3:
        slope = float(np.polyfit(x, valid, 1)[0])
        # Normalize: slope of 0.05 per window = 100, negative = 0
        trend_strength = round(float(max(0, min(100, slope * 2000))), 0)
    else:
        trend_strength = 50.0

    # Edge status
    current_pf = float(valid[-1])
    if current_pf >= edge_threshold * 1.25:
        edge_status = "strong"
    elif current_pf >= edge_threshold:
        edge_status = "moderate"
    elif current_pf >= 1.0:
        edge_status = "critical"
    else:
        edge_status = "lost"

    return {
        "consistency": consistency,
        "stability": stability,
        "trend_strength": trend_strength,
        "edge_status": edge_status,
        "current_pf": round(current_pf, 2),
    }


# =============================================================================
# MARKET REGIME DETECTION
# =============================================================================

def detect_market_regimes(
    trades_df: pd.DataFrame,
    window: int = 20,
) -> Dict:
    """Classify trade windows into Favorable / Unfavorable / Neutral regimes.

    Returns:
        Dict with 'regimes' (list of str per trade), 'favorable_pct',
        'avg_regime_duration', 'current_regime', 'current_regime_age'.
    """
    r = trades_df["r_multiple"].values.astype(float)
    n = len(r)

    regimes = ["neutral"] * n

    if n < window:
        return {
            "regimes": regimes,
            "favorable_pct": None,
            "avg_regime_duration": None,
            "current_regime": "neutral",
            "current_regime_age": 0,
        }

    rolling_mean = pd.Series(r).rolling(window, min_periods=window).mean().values
    rolling_std = pd.Series(r).rolling(window, min_periods=window).std().values

    for i in range(n):
        if np.isnan(rolling_mean[i]):
            continue
        threshold = 0.5 * rolling_std[i] if not np.isnan(rolling_std[i]) else 0
        if rolling_mean[i] > threshold:
            regimes[i] = "favorable"
        elif rolling_mean[i] < -threshold:
            regimes[i] = "unfavorable"
        else:
            regimes[i] = "neutral"

    # Stats
    total_classified = sum(1 for r in regimes if r != "neutral" or True)
    favorable_count = sum(1 for r in regimes[window - 1:] if r == "favorable")
    classified_count = len(regimes[window - 1:])
    favorable_pct = round(favorable_count / max(classified_count, 1) * 100, 1)

    # Average regime duration
    regime_lengths = []
    current_regime = regimes[window - 1] if n > window - 1 else "neutral"
    current_len = 1
    for i in range(window, n):
        if regimes[i] == current_regime:
            current_len += 1
        else:
            regime_lengths.append(current_len)
            current_regime = regimes[i]
            current_len = 1
    regime_lengths.append(current_len)
    avg_regime_duration = round(np.mean(regime_lengths), 0) if regime_lengths else 0

    # Current regime info
    cr = regimes[-1]
    cr_age = 1
    for i in range(n - 2, -1, -1):
        if regimes[i] == cr:
            cr_age += 1
        else:
            break

    return {
        "regimes": regimes,
        "favorable_pct": favorable_pct,
        "avg_regime_duration": avg_regime_duration,
        "current_regime": cr,
        "current_regime_age": cr_age,
    }


# =============================================================================
# MARKOV INTELLIGENCE INSIGHTS
# =============================================================================

def generate_markov_insights(
    probs: Dict[str, float],
    edge_scores: Dict,
    regime_info: Dict,
) -> List[str]:
    """Generate 2-4 plain-text insight sentences from Markov analysis.

    Rule-based templates — no LLM call.
    """
    insights = []

    # Momentum insight
    w2w = probs.get("W_to_W", 0)
    if w2w > 0.65:
        insights.append(
            f"Strong momentum detected: {w2w * 100:.1f}% probability of "
            f"consecutive wins suggests positive autocorrelation."
        )
    elif w2w < 0.4:
        insights.append(
            f"Weak momentum: only {w2w * 100:.1f}% probability of consecutive "
            f"wins. Win/loss outcomes appear more random."
        )

    # Loss clustering
    l2l = probs.get("L_to_L", 0)
    if l2l > 0.6:
        insights.append(
            f"Loss clustering detected: {l2l * 100:.1f}% probability of "
            f"consecutive losses. Consider position sizing adjustment during "
            f"losing streaks."
        )

    # Trend insight
    trend = edge_scores.get("trend_strength")
    if trend is not None and trend > 70:
        insights.append(
            "Strong performance trend detected. Current trajectory shows "
            "sustained edge."
        )
    elif trend is not None and trend < 30:
        insights.append(
            "Performance trend is declining. Monitor closely for further "
            "edge degradation."
        )

    # Edge strength
    status = edge_scores.get("edge_status")
    cpf = edge_scores.get("current_pf")
    if status == "strong" and cpf is not None:
        insights.append(
            f"Exceptional recent performance. Profit factor at {cpf:.2f} "
            f"indicates robust edge."
        )
    elif status == "lost" and cpf is not None:
        insights.append(
            f"Edge appears lost: rolling profit factor at {cpf:.2f} is below "
            f"breakeven. Strategy may need re-evaluation."
        )

    # Consistency warning
    consistency = edge_scores.get("consistency")
    if consistency is not None and consistency < 50:
        insights.append(
            f"Edge inconsistency warning: strategy only maintains edge in "
            f"{consistency:.0f}% of rolling windows."
        )

    # Regime insight
    regime = regime_info.get("current_regime")
    fav_pct = regime_info.get("favorable_pct")
    if regime == "unfavorable" and fav_pct is not None:
        insights.append(
            f"Currently in an unfavorable regime. Only {fav_pct:.0f}% of "
            f"analyzed periods were favorable."
        )

    return insights[:4]
