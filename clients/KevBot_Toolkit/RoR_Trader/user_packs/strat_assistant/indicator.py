import pandas as pd
import numpy as np


def calculate_strat_assistant(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Calculate Strat bar patterns, strategy combos, and actionable signals.

    Bar Types (comparing current bar to prior):
        1 = Inside bar (high <= prior high AND low >= prior low)
        2 = Two-Up (high > prior high, low NOT lower)
       -2 = Two-Down (low < prior low, high NOT higher)
        3 = Outside bar (high > prior high AND low < prior low)

    Strategy Combos (multi-bar patterns):
        Continuations: 222_BULL, 222_BEAR, 212_MM_BULL, 212_MM_BEAR
        Reversals: 22_REV_BULL, 22_REV_BEAR, 212_REV_BULL, 212_REV_BEAR,
                   322_REV_BULL, 322_REV_BEAR, 32_REV_BULL, 32_REV_BEAR,
                   312_REV_BULL, 312_REV_BEAR, 122_REVSTRAT_BULL, 122_REVSTRAT_BEAR,
                   3_REVSTRAT_BULL, 3_REVSTRAT_BEAR

    Actionable Signals:
        SHOOTER = upper wick >= wick_pct of bar range (bearish)
        HAMMER = lower wick >= wick_pct of bar range (bullish)
        INSIDE = inside bar pattern

    Outputs:
        strat_bar_type: Integer bar classification (1, 2, -2, 3, or 0)
        strat_combo: String combo name or empty
        strat_actionable: String actionable signal or empty
        strat_candle_color: Hex color for bar coloring
    """
    wick_pct = params.get("wick_pct", 0.75)

    result = df.copy()
    high = result["high"].values
    low = result["low"].values
    close = result["close"].values
    opn = result["open"].values
    n = len(close)

    # Colors
    inside_color = "#F6BE00"
    two_up_color = "#22c55e"
    two_down_color = "#ef4444"
    outside_color = "#d946ef"

    # Bar type classification
    bar_type = np.zeros(n, dtype=int)
    colors = [""] * n

    for i in range(1, n):
        is_inside = high[i] <= high[i - 1] and low[i] >= low[i - 1]
        is_outside = high[i] > high[i - 1] and low[i] < low[i - 1]
        is_two_up = high[i] > high[i - 1] and not (low[i] < low[i - 1])
        is_two_down = low[i] < low[i - 1] and not (high[i] > high[i - 1])

        if is_inside:
            bar_type[i] = 1
            colors[i] = inside_color
        elif is_outside:
            bar_type[i] = 3
            colors[i] = outside_color
        elif is_two_up:
            bar_type[i] = 2
            colors[i] = two_up_color
        elif is_two_down:
            bar_type[i] = -2
            colors[i] = two_down_color

    # Strategy combo detection (look back 2-4 bars)
    combos = [""] * n

    for i in range(3, n):
        b0 = bar_type[i]
        b1 = bar_type[i - 1]
        b2 = bar_type[i - 2]
        b3 = bar_type[i - 3] if i >= 4 else 0

        # Continuations
        if b2 == -2 and b1 == -2 and b0 == -2:
            combos[i] = "222_BEAR"
        elif b2 == 2 and b1 == 2 and b0 == 2:
            combos[i] = "222_BULL"
        elif b2 == -2 and b1 == 1 and b0 == -2:
            combos[i] = "212_MM_BEAR"
        elif b2 == 2 and b1 == 1 and b0 == 2:
            combos[i] = "212_MM_BULL"

        # Reversals (check most specific first)
        elif i >= 4 and b3 == -2 and b2 == -2 and b1 == 2 and b0 == -2:
            combos[i] = "2222_BEAR"
        elif i >= 4 and b3 == 2 and b2 == 2 and b1 == -2 and b0 == 2:
            combos[i] = "2222_BULL"
        elif b2 == 1 and b1 == 2 and b0 == -2:
            combos[i] = "122_REVSTRAT_BEAR"
        elif b2 == 1 and b1 == -2 and b0 == 2:
            combos[i] = "122_REVSTRAT_BULL"
        elif b2 == 3 and b1 == 2 and b0 == -2:
            combos[i] = "322_REV_BEAR"
        elif b2 == 3 and b1 == -2 and b0 == 2:
            combos[i] = "322_REV_BULL"
        elif b2 == 3 and b1 == 1 and b0 == -2:
            combos[i] = "312_REV_BEAR"
        elif b2 == 3 and b1 == 1 and b0 == 2:
            combos[i] = "312_REV_BULL"
        elif b1 == 3 and b0 == -2:
            combos[i] = "32_REV_BEAR"
        elif b1 == 3 and b0 == 2:
            combos[i] = "32_REV_BULL"
        elif b2 == 2 and b1 == 1 and b0 == -2:
            combos[i] = "212_REV_BEAR"
        elif b2 == -2 and b1 == 1 and b0 == 2:
            combos[i] = "212_REV_BULL"
        elif b1 == 2 and b0 == -2 and b2 != 1 and b2 != -2 and b2 != 3:
            combos[i] = "22_REV_BEAR"
        elif b1 == -2 and b0 == 2 and b2 != 1 and b2 != 2 and b2 != 3:
            combos[i] = "22_REV_BULL"

        # 3-bar RevStrat (outside bar closing beyond prior range)
        if b0 == 3:
            if close[i] < low[i - 1]:
                combos[i] = "3_REVSTRAT_BEAR"
            elif close[i] > high[i - 1]:
                combos[i] = "3_REVSTRAT_BULL"

    # Actionable signals
    actionable = [""] * n
    for i in range(1, n):
        bar_range = high[i] - low[i]
        if bar_range <= 0:
            continue

        wick_height = bar_range * wick_pct
        shooter_top = high[i] - wick_height
        hammer_bottom = low[i] + wick_height

        is_inside = bar_type[i] == 1

        # Inside bar takes priority over shooter/hammer (matching Pine)
        if is_inside:
            actionable[i] = "INSIDE"
        elif opn[i] < shooter_top and close[i] < shooter_top:
            actionable[i] = "SHOOTER"
        elif opn[i] > hammer_bottom and close[i] > hammer_bottom:
            actionable[i] = "HAMMER"

    result["strat_bar_type"] = bar_type
    result["strat_combo"] = combos
    result["strat_actionable"] = actionable
    result["strat_candle_color"] = colors

    return result
