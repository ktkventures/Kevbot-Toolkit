"""Strat Assistant — incremental indicator class.

Bar-type classification (1=Inside, 2=Two-Up, -2=Two-Down, 3=Outside)
relative to the previous bar, plus a multi-bar combo detector that
needs a 4-deep history of bar types, plus actionable wick-based
signals (Shooter / Hammer / Inside).

Maintains:
  - Previous bar's high/low (for bar-type classification)
  - Ring buffer of last 4 bar_type codes (for combo detection)

Triggers are pure pattern events — no level crosses, so only C and CC
variants get materialized by the registry.
"""


class StratAssistantIncremental:
    def __init__(self, **params):
        self.wick_pct = float(params.get("wick_pct", 0.75))

        # Bar-type classification needs prev bar's high/low
        self._prev_high = None
        self._prev_low = None

        # Combo detection needs prev 4 bar_types
        self._bar_type_buf: list = []

    def warmup(self, df) -> None:
        for _, row in df.iterrows():
            self.update_bar({
                "open": row.get("open", 0.0),
                "high": row.get("high", 0.0),
                "low": row.get("low", 0.0),
                "close": row.get("close", 0.0),
                "volume": row.get("volume", 0.0),
            })

    def update_bar(self, bar: dict) -> dict:
        opn = float(bar["open"])
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])

        # ── Bar type classification ──────────────────────────────────
        bar_type = 0
        candle_color = ""
        if self._prev_high is not None and self._prev_low is not None:
            ph = self._prev_high
            pl = self._prev_low
            is_inside = high <= ph and low >= pl
            is_outside = high > ph and low < pl
            is_two_up = high > ph and not (low < pl)
            is_two_down = low < pl and not (high > ph)

            if is_inside:
                bar_type = 1
                candle_color = "#F6BE00"   # Inside: yellow
            elif is_outside:
                bar_type = 3
                candle_color = "#d946ef"   # Outside: magenta
            elif is_two_up:
                bar_type = 2
                candle_color = "#22c55e"   # 2-up: green
            elif is_two_down:
                bar_type = -2
                candle_color = "#ef4444"   # 2-down: red

        # ── Combo detection ──────────────────────────────────────────
        # We need bar_type history including this bar to evaluate combos.
        b0 = bar_type
        b1 = self._bar_type_buf[-1] if len(self._bar_type_buf) >= 1 else 0
        b2 = self._bar_type_buf[-2] if len(self._bar_type_buf) >= 2 else 0
        b3 = self._bar_type_buf[-3] if len(self._bar_type_buf) >= 3 else 0
        combo = ""

        # Combo logic mirrors the original batch — order matters because
        # most combos are mutually exclusive but the 3_REVSTRAT check at
        # the end can override.
        if len(self._bar_type_buf) >= 2:
            # Continuations
            if b2 == -2 and b1 == -2 and b0 == -2:
                combo = "222_BEAR"
            elif b2 == 2 and b1 == 2 and b0 == 2:
                combo = "222_BULL"
            elif b2 == -2 and b1 == 1 and b0 == -2:
                combo = "212_MM_BEAR"
            elif b2 == 2 and b1 == 1 and b0 == 2:
                combo = "212_MM_BULL"
            # Reversals — most specific first
            elif len(self._bar_type_buf) >= 3 and b3 == -2 and b2 == -2 and b1 == 2 and b0 == -2:
                combo = "2222_BEAR"
            elif len(self._bar_type_buf) >= 3 and b3 == 2 and b2 == 2 and b1 == -2 and b0 == 2:
                combo = "2222_BULL"
            elif b2 == 1 and b1 == 2 and b0 == -2:
                combo = "122_REVSTRAT_BEAR"
            elif b2 == 1 and b1 == -2 and b0 == 2:
                combo = "122_REVSTRAT_BULL"
            elif b2 == 3 and b1 == 2 and b0 == -2:
                combo = "322_REV_BEAR"
            elif b2 == 3 and b1 == -2 and b0 == 2:
                combo = "322_REV_BULL"
            elif b2 == 3 and b1 == 1 and b0 == -2:
                combo = "312_REV_BEAR"
            elif b2 == 3 and b1 == 1 and b0 == 2:
                combo = "312_REV_BULL"
            elif b1 == 3 and b0 == -2:
                combo = "32_REV_BEAR"
            elif b1 == 3 and b0 == 2:
                combo = "32_REV_BULL"
            elif b2 == 2 and b1 == 1 and b0 == -2:
                combo = "212_REV_BEAR"
            elif b2 == -2 and b1 == 1 and b0 == 2:
                combo = "212_REV_BULL"
            elif b1 == 2 and b0 == -2 and b2 != 1 and b2 != -2 and b2 != 3:
                combo = "22_REV_BEAR"
            elif b1 == -2 and b0 == 2 and b2 != 1 and b2 != 2 and b2 != 3:
                combo = "22_REV_BULL"

        # 3_REVSTRAT can override — outside bar closing past prior bar's range
        if b0 == 3 and self._prev_high is not None and self._prev_low is not None:
            if close < self._prev_low:
                combo = "3_REVSTRAT_BEAR"
            elif close > self._prev_high:
                combo = "3_REVSTRAT_BULL"

        # ── Actionable wick signal ───────────────────────────────────
        actionable = ""
        bar_range = high - low
        if bar_range > 0:
            wick_height = bar_range * self.wick_pct
            shooter_top = high - wick_height
            hammer_bottom = low + wick_height
            is_inside = bar_type == 1

            if is_inside:
                actionable = "INSIDE"
            elif opn < shooter_top and close < shooter_top:
                actionable = "SHOOTER"
            elif opn > hammer_bottom and close > hammer_bottom:
                actionable = "HAMMER"

        # ── Trigger booleans ─────────────────────────────────────────
        trig_bull_c2 = bar_type == 2
        trig_bear_c2 = bar_type == -2
        trig_outside_bar = bar_type == 3
        trig_inside_bar = bar_type == 1
        trig_shooter = actionable == "SHOOTER"
        trig_hammer = actionable == "HAMMER"

        # ── Update state for next bar ────────────────────────────────
        self._prev_high = high
        self._prev_low = low
        self._bar_type_buf.append(bar_type)
        if len(self._bar_type_buf) > 4:
            self._bar_type_buf.pop(0)

        return {
            # Indicator columns
            "strat_bar_type": bar_type,
            "strat_combo": combo,
            "strat_actionable": actionable,
            "strat_candle_color": candle_color,
            # Trigger booleans (engine reads via user-pack pickup loop)
            "strat_bull_c2": bool(trig_bull_c2),
            "strat_bear_c2": bool(trig_bear_c2),
            "strat_outside_bar": bool(trig_outside_bar),
            "strat_inside_bar": bool(trig_inside_bar),
            "strat_shooter": bool(trig_shooter),
            "strat_hammer": bool(trig_hammer),
        }
