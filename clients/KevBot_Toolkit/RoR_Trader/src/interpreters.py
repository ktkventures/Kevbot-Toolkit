"""
Interpreters for RoR Trader
============================

Interpreters analyze indicator values and price action to output clear,
mutually exclusive condition states (interpretations).

Each interpreter:
1. Requires specific indicator columns to be present
2. Classifies the current state into a categorical output
3. Provides trigger detection for entry/exit signals

Interpreters are SEPARATE from indicators - indicators calculate values,
interpreters classify states.
"""

import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Optional, Set
from dataclasses import dataclass


@dataclass
class InterpreterConfig:
    """Configuration for an interpreter."""
    name: str
    description: str
    category: str
    requires_indicators: List[str]  # Indicator IDs this interpreter needs
    outputs: List[str]              # Possible categorical outputs
    triggers: List[str]             # Available trigger events


# =============================================================================
# INTERPRETER DEFINITIONS
# =============================================================================

INTERPRETERS: Dict[str, InterpreterConfig] = {
    "EMA_STACK": InterpreterConfig(
        name="EMA Stack",
        description="Analyzes EMA alignment (8, 21, 50) relative to price",
        category="Moving Averages",
        requires_indicators=["ema_8", "ema_21", "ema_50"],
        outputs=["SML", "SLM", "MSL", "MLS", "LSM", "LMS"],
        triggers=["ema_cross_bull", "ema_cross_bear", "ema_mid_cross_bull", "ema_mid_cross_bear"]
    ),
    "MACD_LINE": InterpreterConfig(
        name="MACD Line",
        description="MACD line vs Signal line with zero-line context",
        category="MACD",
        requires_indicators=["macd"],
        outputs=["M>S+", "M>S-", "M<S-", "M<S+"],
        triggers=["macd_cross_bull", "macd_cross_bear", "macd_zero_cross_up", "macd_zero_cross_down"]
    ),
    "MACD_HISTOGRAM": InterpreterConfig(
        name="MACD Histogram",
        description="MACD histogram direction and momentum",
        category="MACD",
        requires_indicators=["macd"],
        outputs=["H+up", "H+dn", "H-dn", "H-up"],
        triggers=["macd_hist_flip_pos", "macd_hist_flip_neg", "macd_hist_momentum_shift_up", "macd_hist_momentum_shift_down"]
    ),
    "VWAP": InterpreterConfig(
        name="VWAP Position",
        description="Price position relative to VWAP with SD bands (7 zones)",
        category="Volume",
        requires_indicators=["vwap"],
        outputs=[">+2σ", ">+1σ", ">V", "@V", "<V", "<-1σ", "<-2σ"],
        triggers=["vwap_cross_above", "vwap_cross_below", "vwap_enter_upper_extreme", "vwap_enter_lower_extreme", "vwap_return_to_vwap"]
    ),
    "RVOL": InterpreterConfig(
        name="Relative Volume",
        description="Current volume vs historical average",
        category="Volume",
        requires_indicators=["vol_sma"],
        outputs=["EXTREME", "HIGH", "NORMAL", "LOW", "MINIMAL"],
        triggers=["rvol_spike", "rvol_extreme", "rvol_fade"]
    ),
    "UTBOT": InterpreterConfig(
        name="UT Bot",
        description="ATR trailing stop direction",
        category="Trend",
        requires_indicators=["utbot_stop", "utbot_direction"],
        outputs=["BULL", "BEAR"],
        triggers=["utbot_buy", "utbot_sell"]
    ),
}


# =============================================================================
# EMA STACK INTERPRETER
# =============================================================================

def interpret_ema_stack(df: pd.DataFrame) -> pd.Series:
    """
    Interpret EMA Stack alignment.

    Requires: ema_8, ema_21, ema_50 columns

    Outputs:
    - SML: Price > Short > Mid > Long (Full Bull Stack)
    - SLM: Short > Price > Mid > Long (Price below short)
    - MSL: Short > Mid > Price > Long (Price in middle)
    - MLS: Short > Mid > Long > Price (Price below all, bull order)
    - LSM: Long > Short > Mid or other transitional
    - LMS: Price < Short < Mid < Long (Full Bear Stack)
    """
    def interpret(row):
        p = row['close']
        s = row.get('ema_8', np.nan)
        m = row.get('ema_21', np.nan)
        l = row.get('ema_50', np.nan)

        if pd.isna(s) or pd.isna(m) or pd.isna(l):
            return None

        # Full bull stack: Price > Short > Mid > Long
        if p > s > m > l:
            return "SML"
        # Bull but price below short EMA
        elif s > p > m > l:
            return "SLM"
        # Price between mid and long
        elif s > m > p > l:
            return "MSL"
        # Price below all EMAs but EMAs still bullish order
        elif s > m > l > p:
            return "MLS"
        # Full bear stack: Long > Mid > Short > Price
        elif l > m > s > p:
            return "LMS"
        # Various transitional states
        else:
            return "LSM"

    return df.apply(interpret, axis=1)


def detect_ema_triggers(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Detect EMA-based triggers.

    Returns dict of boolean Series for each trigger.
    """
    triggers = {}

    # Short crosses above Mid (bullish)
    triggers['ema_cross_bull'] = (
        (df['ema_8'] > df['ema_21']) &
        (df['ema_8'].shift(1) <= df['ema_21'].shift(1))
    )

    # Short crosses below Mid (bearish)
    triggers['ema_cross_bear'] = (
        (df['ema_8'] < df['ema_21']) &
        (df['ema_8'].shift(1) >= df['ema_21'].shift(1))
    )

    # Mid crosses above Long (bullish confirmation)
    triggers['ema_mid_cross_bull'] = (
        (df['ema_21'] > df['ema_50']) &
        (df['ema_21'].shift(1) <= df['ema_50'].shift(1))
    )

    # Mid crosses below Long (bearish confirmation)
    triggers['ema_mid_cross_bear'] = (
        (df['ema_21'] < df['ema_50']) &
        (df['ema_21'].shift(1) >= df['ema_50'].shift(1))
    )

    return triggers


# =============================================================================
# MACD LINE INTERPRETER
# =============================================================================

def interpret_macd_line(df: pd.DataFrame) -> pd.Series:
    """
    Interpret MACD Line position.

    Requires: macd_line, macd_signal columns

    Outputs:
    - M>S+: MACD above signal AND above zero (strong bullish)
    - M>S-: MACD above signal BUT below zero (recovering)
    - M<S-: MACD below signal AND below zero (strong bearish)
    - M<S+: MACD below signal BUT above zero (weakening)
    """
    def interpret(row):
        macd = row.get('macd_line', np.nan)
        signal = row.get('macd_signal', np.nan)

        if pd.isna(macd) or pd.isna(signal):
            return None

        above_signal = macd > signal
        above_zero = macd > 0

        if above_signal and above_zero:
            return "M>S+"
        elif above_signal and not above_zero:
            return "M>S-"
        elif not above_signal and not above_zero:
            return "M<S-"
        else:  # below signal, above zero
            return "M<S+"

    return df.apply(interpret, axis=1)


def detect_macd_triggers(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Detect MACD-based triggers."""
    triggers = {}

    # Bullish cross (MACD crosses above signal)
    triggers['macd_cross_bull'] = (
        (df['macd_line'] > df['macd_signal']) &
        (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))
    )

    # Bearish cross (MACD crosses below signal)
    triggers['macd_cross_bear'] = (
        (df['macd_line'] < df['macd_signal']) &
        (df['macd_line'].shift(1) >= df['macd_signal'].shift(1))
    )

    # Zero line cross up
    triggers['macd_zero_cross_up'] = (
        (df['macd_line'] > 0) &
        (df['macd_line'].shift(1) <= 0)
    )

    # Zero line cross down
    triggers['macd_zero_cross_down'] = (
        (df['macd_line'] < 0) &
        (df['macd_line'].shift(1) >= 0)
    )

    return triggers


# =============================================================================
# MACD HISTOGRAM INTERPRETER
# =============================================================================

def interpret_macd_histogram(df: pd.DataFrame) -> pd.Series:
    """
    Interpret MACD Histogram momentum.

    Requires: macd_hist column

    Outputs:
    - H+up: Positive and rising (strengthening bullish)
    - H+dn: Positive but falling (weakening bullish)
    - H-dn: Negative and falling (strengthening bearish)
    - H-up: Negative but rising (weakening bearish)
    """
    def interpret(row, prev_hist):
        hist = row.get('macd_hist', np.nan)

        if pd.isna(hist) or pd.isna(prev_hist):
            return None

        positive = hist > 0
        rising = hist > prev_hist

        if positive and rising:
            return "H+up"
        elif positive and not rising:
            return "H+dn"
        elif not positive and not rising:
            return "H-dn"
        else:  # negative and rising
            return "H-up"

    # Need to pass previous histogram value
    results = []
    prev = np.nan
    for idx, row in df.iterrows():
        results.append(interpret(row, prev))
        prev = row.get('macd_hist', np.nan)

    return pd.Series(results, index=df.index)


def detect_macd_hist_triggers(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Detect MACD histogram triggers."""
    triggers = {}

    # Histogram flips positive (neg → pos)
    triggers['macd_hist_flip_pos'] = (
        (df['macd_hist'] > 0) &
        (df['macd_hist'].shift(1) <= 0)
    )

    # Histogram flips negative (pos → neg)
    triggers['macd_hist_flip_neg'] = (
        (df['macd_hist'] < 0) &
        (df['macd_hist'].shift(1) >= 0)
    )

    # Momentum shift up (was falling, now rising)
    triggers['macd_hist_momentum_shift_up'] = (
        (df['macd_hist'] > df['macd_hist'].shift(1)) &
        (df['macd_hist'].shift(1) < df['macd_hist'].shift(2))
    )

    # Momentum shift down (was rising, now falling)
    triggers['macd_hist_momentum_shift_down'] = (
        (df['macd_hist'] < df['macd_hist'].shift(1)) &
        (df['macd_hist'].shift(1) > df['macd_hist'].shift(2))
    )

    return triggers


# =============================================================================
# VWAP INTERPRETER
# =============================================================================

def interpret_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Interpret VWAP Position using 7 mutually exclusive zones.

    Requires: vwap, vwap_sd1_upper, vwap_sd1_lower, vwap_sd2_upper, vwap_sd2_lower columns
    Falls back to rolling std if SD band columns not present.

    Zones (from KevBot reference):
    - >+2σ: Price above VWAP + 2×SD (extended high)
    - >+1σ: Price between +1σ and +2σ
    - >V:   Price between VWAP+0.5σ and +1σ (above VWAP zone)
    - @V:   Price within ±0.5σ of VWAP (at VWAP)
    - <V:   Price between VWAP-0.5σ and -1σ (below VWAP zone)
    - <-1σ: Price between -1σ and -2σ
    - <-2σ: Price below VWAP - 2×SD (extended low)
    """
    has_bands = all(c in df.columns for c in ['vwap_sd1_upper', 'vwap_sd1_lower', 'vwap_sd2_upper', 'vwap_sd2_lower'])

    def interpret(row):
        vwap_val = row.get('vwap', np.nan)
        price = row['close']

        if pd.isna(vwap_val) or vwap_val == 0:
            return None

        if has_bands:
            sd1_upper = row.get('vwap_sd1_upper', np.nan)
            sd1_lower = row.get('vwap_sd1_lower', np.nan)
            sd2_upper = row.get('vwap_sd2_upper', np.nan)
            sd2_lower = row.get('vwap_sd2_lower', np.nan)

            if pd.isna(sd1_upper):
                return None

            # Compute half-SD for the @V zone (midpoint between VWAP and SD1)
            half_sd = (sd1_upper - vwap_val) * 0.5
            at_upper = vwap_val + half_sd
            at_lower = vwap_val - half_sd
        else:
            # Fallback: use a simple percentage tolerance
            half_sd = vwap_val * 0.001
            at_upper = vwap_val + half_sd
            at_lower = vwap_val - half_sd
            sd1_upper = vwap_val + half_sd * 2
            sd1_lower = vwap_val - half_sd * 2
            sd2_upper = vwap_val + half_sd * 4
            sd2_lower = vwap_val - half_sd * 4

        if price > sd2_upper:
            return ">+2σ"
        elif price > sd1_upper:
            return ">+1σ"
        elif price > at_upper:
            return ">V"
        elif price >= at_lower:
            return "@V"
        elif price >= sd1_lower:
            return "<V"
        elif price >= sd2_lower:
            return "<-1σ"
        else:
            return "<-2σ"

    return df.apply(interpret, axis=1)


def detect_vwap_triggers(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Detect VWAP-based triggers (5 triggers matching KevBot reference)."""
    triggers = {}

    # Price crosses above VWAP
    triggers['vwap_cross_above'] = (
        (df['close'] > df['vwap']) &
        (df['close'].shift(1) <= df['vwap'].shift(1))
    )

    # Price crosses below VWAP
    triggers['vwap_cross_below'] = (
        (df['close'] < df['vwap']) &
        (df['close'].shift(1) >= df['vwap'].shift(1))
    )

    # Enter upper extreme (price enters >+2σ zone)
    if 'vwap_sd2_upper' in df.columns:
        triggers['vwap_enter_upper_extreme'] = (
            (df['close'] > df['vwap_sd2_upper']) &
            (df['close'].shift(1) <= df['vwap_sd2_upper'].shift(1))
        )

        # Enter lower extreme (price enters <-2σ zone)
        triggers['vwap_enter_lower_extreme'] = (
            (df['close'] < df['vwap_sd2_lower']) &
            (df['close'].shift(1) >= df['vwap_sd2_lower'].shift(1))
        )

    # Return to VWAP zone (from extreme back to @V)
    if 'vwap_sd1_upper' in df.columns:
        half_sd = (df['vwap_sd1_upper'] - df['vwap']) * 0.5
        at_upper = df['vwap'] + half_sd
        at_lower = df['vwap'] - half_sd
        prev_half_sd = (df['vwap_sd1_upper'].shift(1) - df['vwap'].shift(1)) * 0.5
        prev_at_upper = df['vwap'].shift(1) + prev_half_sd
        prev_at_lower = df['vwap'].shift(1) - prev_half_sd

        # Was outside ±1σ, now within ±0.5σ (at VWAP)
        was_extreme = (
            (df['close'].shift(1) > df['vwap_sd1_upper'].shift(1)) |
            (df['close'].shift(1) < df['vwap_sd1_lower'].shift(1))
        )
        now_at_vwap = (df['close'] >= at_lower) & (df['close'] <= at_upper)
        triggers['vwap_return_to_vwap'] = was_extreme & now_at_vwap

    return triggers


# =============================================================================
# RVOL INTERPRETER
# =============================================================================

def interpret_rvol(df: pd.DataFrame) -> pd.Series:
    """
    Interpret Relative Volume.

    Requires: rvol column (volume / vol_sma)

    Outputs:
    - EXTREME: > 300% of average
    - HIGH: > 150% of average
    - NORMAL: 75-150% of average
    - LOW: 50-75% of average
    - MINIMAL: < 50% of average
    """
    def interpret(row):
        rvol = row.get('rvol', np.nan)

        if pd.isna(rvol):
            return None

        if rvol > 3.0:
            return "EXTREME"
        elif rvol > 1.5:
            return "HIGH"
        elif rvol > 0.75:
            return "NORMAL"
        elif rvol > 0.5:
            return "LOW"
        else:
            return "MINIMAL"

    return df.apply(interpret, axis=1)


def detect_rvol_triggers(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Detect RVOL-based triggers."""
    triggers = {}

    # Volume spike (crosses above 1.5x)
    triggers['rvol_spike'] = (
        (df['rvol'] > 1.5) &
        (df['rvol'].shift(1) <= 1.5)
    )

    # Extreme volume (crosses above 3x)
    triggers['rvol_extreme'] = (
        (df['rvol'] > 3.0) &
        (df['rvol'].shift(1) <= 3.0)
    )

    # Volume fade (crosses below 1.0)
    triggers['rvol_fade'] = (
        (df['rvol'] < 1.0) &
        (df['rvol'].shift(1) >= 1.0)
    )

    return triggers


# =============================================================================
# UT BOT INTERPRETER
# =============================================================================

def interpret_utbot(df: pd.DataFrame) -> pd.Series:
    """
    Interpret UT Bot direction.

    Requires: utbot_direction column

    Outputs:
    - BULL: Price above trailing stop (bullish trend)
    - BEAR: Price below trailing stop (bearish trend)
    """
    def interpret(row):
        d = row.get('utbot_direction', np.nan)
        if pd.isna(d) or d == 0:
            return None
        return "BULL" if d == 1 else "BEAR"

    return df.apply(interpret, axis=1)


def detect_utbot_triggers(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Detect UT Bot buy/sell triggers (direction flips)."""
    triggers = {}

    d = df['utbot_direction']
    d_prev = d.shift(1)

    # Buy signal: direction flips to bullish
    triggers['utbot_buy'] = (d == 1) & (d_prev != 1)

    # Sell signal: direction flips to bearish
    triggers['utbot_sell'] = (d == -1) & (d_prev != -1)

    return triggers


# =============================================================================
# INTERPRETER & TRIGGER FUNCTION REGISTRIES
# =============================================================================
# Mutable dispatch registries. Built-in functions are registered below;
# user packs register via register_interpreter() / register_trigger_detector().

INTERPRETER_FUNCS: Dict[str, Callable] = {}
TRIGGER_FUNCS: Dict[str, Callable] = {}

# Register built-in interpreter functions
INTERPRETER_FUNCS["EMA_STACK"] = interpret_ema_stack
INTERPRETER_FUNCS["MACD_LINE"] = interpret_macd_line
INTERPRETER_FUNCS["MACD_HISTOGRAM"] = interpret_macd_histogram
INTERPRETER_FUNCS["VWAP"] = interpret_vwap
INTERPRETER_FUNCS["RVOL"] = interpret_rvol
INTERPRETER_FUNCS["UTBOT"] = interpret_utbot

# Register built-in trigger detection functions
TRIGGER_FUNCS["EMA_STACK"] = detect_ema_triggers
TRIGGER_FUNCS["MACD_LINE"] = detect_macd_triggers
TRIGGER_FUNCS["MACD_HISTOGRAM"] = detect_macd_hist_triggers
TRIGGER_FUNCS["VWAP"] = detect_vwap_triggers
TRIGGER_FUNCS["RVOL"] = detect_rvol_triggers
TRIGGER_FUNCS["UTBOT"] = detect_utbot_triggers


def register_interpreter(key: str, func: Callable) -> None:
    """Register an interpreter function for a given key."""
    INTERPRETER_FUNCS[key] = func


def register_trigger_detector(key: str, func: Callable) -> None:
    """Register a trigger detection function for a given key."""
    TRIGGER_FUNCS[key] = func


def unregister_interpreter(key: str) -> None:
    """Remove an interpreter function."""
    INTERPRETER_FUNCS.pop(key, None)


def unregister_trigger_detector(key: str) -> None:
    """Remove a trigger detection function."""
    TRIGGER_FUNCS.pop(key, None)


# =============================================================================
# MAIN INTERPRETER ENGINE
# =============================================================================

def run_all_interpreters(
    df: pd.DataFrame,
    enabled_interpreters: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Run all enabled interpreters on the data.

    IMPORTANT: Indicators must be calculated first using run_all_indicators().

    Args:
        df: DataFrame with OHLCV data AND indicator columns
        enabled_interpreters: List of interpreter keys to run (default: all)

    Returns:
        DataFrame with interpretation columns added
    """
    if enabled_interpreters is None:
        enabled_interpreters = list(INTERPRETERS.keys())

    result = df.copy()

    for interp_key in enabled_interpreters:
        if interp_key in INTERPRETER_FUNCS:
            try:
                result[interp_key] = INTERPRETER_FUNCS[interp_key](df)
            except KeyError:
                # Missing indicator columns — skip this interpreter
                pass

    return result


def detect_all_triggers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect all triggers and add boolean columns.

    Args:
        df: DataFrame with indicator columns

    Returns:
        DataFrame with trigger boolean columns added
    """
    result = df.copy()

    # Collect all triggers from registered detectors
    all_triggers = {}

    for interp_key, trigger_func in TRIGGER_FUNCS.items():
        try:
            triggers = trigger_func(df)
            all_triggers.update(triggers)
        except KeyError:
            # Missing indicator columns — skip this trigger set
            pass

    # Add trigger columns
    for trigger_id, trigger_series in all_triggers.items():
        result[f"trig_{trigger_id}"] = trigger_series.fillna(False)

    return result


def get_confluence_records(row: pd.Series, timeframe: str, interpreters: List[str],
                           general_columns: Optional[List[str]] = None) -> Set[str]:
    """
    Get all confluence records for a single bar.

    Returns a set of strings like {"1M-EMA_STACK-SML", "1M-MACD_LINE-M>S+"}
    General pack columns produce records like {"GEN-TOD_NY_OPEN-IN_WINDOW"}
    """
    records = set()

    for interp in interpreters:
        if interp in row.index and pd.notna(row[interp]):
            record = f"{timeframe}-{interp}-{row[interp]}"
            records.add(record)

    if general_columns:
        for col in general_columns:
            if col in row.index and pd.notna(row[col]):
                pack_id = col[3:]  # Strip "GP_" prefix
                records.add(f"GEN-{pack_id}-{row[col]}")

    return records


def get_available_triggers() -> Dict[str, str]:
    """
    Get all available triggers with display names.

    Returns dict mapping trigger_id -> display_name
    """
    triggers = {
        # EMA triggers
        "ema_cross_bull": "EMA 8 > 21 Cross (Bull)",
        "ema_cross_bear": "EMA 8 < 21 Cross (Bear)",
        "ema_mid_cross_bull": "EMA 21 > 50 Cross (Bull)",
        "ema_mid_cross_bear": "EMA 21 < 50 Cross (Bear)",
        # MACD triggers
        "macd_cross_bull": "MACD Bullish Cross",
        "macd_cross_bear": "MACD Bearish Cross",
        "macd_zero_cross_up": "MACD Zero Cross Up",
        "macd_zero_cross_down": "MACD Zero Cross Down",
        "macd_hist_flip_pos": "MACD Histogram Flip Positive",
        "macd_hist_flip_neg": "MACD Histogram Flip Negative",
        "macd_hist_momentum_shift_up": "MACD Histogram Momentum Shift Up",
        "macd_hist_momentum_shift_down": "MACD Histogram Momentum Shift Down",
        # VWAP triggers
        "vwap_cross_above": "VWAP Cross Above",
        "vwap_cross_below": "VWAP Cross Below",
        "vwap_enter_upper_extreme": "VWAP Enter Upper Extreme (>+2σ)",
        "vwap_enter_lower_extreme": "VWAP Enter Lower Extreme (<-2σ)",
        "vwap_return_to_vwap": "VWAP Return to VWAP Zone (@V)",
        # RVOL triggers
        "rvol_spike": "Volume Spike (1.5x)",
        "rvol_extreme": "Extreme Volume (3x)",
        "rvol_fade": "Volume Fade (<1x)",
    }
    return triggers


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    from mock_data import generate_mock_bars
    from indicators import run_all_indicators
    from datetime import datetime, timedelta

    # Generate test data
    end = datetime.now().replace(hour=16, minute=0)
    start = end - timedelta(days=5)

    bars = generate_mock_bars(["SPY"], start, end, "1Min")
    spy_bars = bars.loc["SPY"]

    print("Running indicators...")
    df = run_all_indicators(spy_bars)

    print("Running interpreters...")
    df = run_all_interpreters(df)

    print("Detecting triggers...")
    df = detect_all_triggers(df)

    print("\nOutput columns:", df.columns.tolist())
    print()

    print("Sample interpretations (last 10 bars):")
    interp_cols = ['EMA_STACK', 'MACD_LINE', 'MACD_HISTOGRAM', 'VWAP', 'RVOL']
    available = [c for c in interp_cols if c in df.columns]
    print(df[available].tail(10))

    print("\nTrigger counts:")
    trigger_cols = [c for c in df.columns if c.startswith('trig_')]
    for col in trigger_cols:
        count = df[col].sum()
        if count > 0:
            print(f"  {col}: {count}")
