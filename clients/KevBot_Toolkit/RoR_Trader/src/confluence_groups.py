"""
Confluence Groups for RoR Trader
=================================

A Confluence Group bundles together:
- Indicator parameters (e.g., EMA periods)
- Plot settings (colors, line widths)
- Interpreter outputs (categorical states)
- Triggers (entry/exit signals)

Users can create variations of base templates with different parameters,
and each variation gets unique identifiers for drill-down analysis.

Usage:
    from confluence_groups import (
        load_confluence_groups,
        save_confluence_groups,
        get_enabled_groups,
        get_group_triggers,
    )
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PlotSettings:
    """Visual settings for chart overlays."""
    colors: Dict[str, str] = field(default_factory=dict)
    line_width: int = 1
    visible: bool = True


@dataclass
class TriggerDefinition:
    """A trigger that can fire from this confluence group."""
    id: str                  # Unique trigger ID (includes group prefix)
    name: str                # Display name
    base_trigger: str        # Base trigger type (e.g., "cross_bull")
    direction: str           # "LONG", "SHORT", or "BOTH"
    trigger_type: str        # "ENTRY" or "EXIT"
    execution: str = "bar_close"  # "bar_close" or "intra_bar"
    column_base: Optional[str] = None  # If set, share boolean column with this base trigger


@dataclass
class ConfluenceGroup:
    """
    A configured instance of a Confluence Group Template.

    Terminology:
    - Template: The base indicator type with coded logic (e.g., "ema_stack")
    - Version: User's parameter configuration name (e.g., "Default", "Scalping")
    - Confluence Group: The full combination shown in UI (e.g., "EMA Stack (Scalping)")

    Each group generates unique:
    - Interpreter column: {GROUP_ID} (uppercase)
    - Trigger IDs: {group_id}_{trigger_type}
    """
    id: str                           # Unique identifier (e.g., "ema_stack_scalping")
    base_template: str                # Template ID (e.g., "ema_stack")
    version: str                      # Version name (e.g., "Default", "Scalping")
    description: str                  # User description
    enabled: bool                     # Whether to include in analysis
    is_default: bool                  # Protected from deletion
    parameters: Dict[str, Any]        # Template-specific parameters
    plot_settings: PlotSettings       # Visual settings

    @property
    def template_name(self) -> str:
        """Get the human-readable template name (e.g., 'EMA Stack')."""
        template = TEMPLATES.get(self.base_template)
        return template["name"] if template else self.base_template

    @property
    def name(self) -> str:
        """Get the full Confluence Group name: '{Template} ({Version})'."""
        return f"{self.template_name} ({self.version})"

    def get_interpreter_column(self) -> str:
        """Get the DataFrame column name for this group's interpretation."""
        return self.id.upper()

    def get_trigger_id(self, base_trigger: str) -> str:
        """Get the full trigger ID for a base trigger type."""
        return f"{self.id}_{base_trigger}"

    def get_trigger_name(self, base_trigger: str, base_name: str) -> str:
        """Get the display name for a trigger: '{Confluence Group} > {Trigger}'."""
        return f"{self.name} > {base_name}"


# =============================================================================
# BULL/BEAR/NEUTRAL STATE CLASSIFICATION
# =============================================================================
# Used by Mass Builder to expand synthetic BULL/BEAR selections to real states.
# Keyed by interpreter name (matches `interpreters` list in templates).
# States not listed are treated as NEUTRAL (no direction).

INTERPRETER_DIRECTION_MAP: Dict[str, Dict[str, List[str]]] = {
    "EMA_STACK": {
        "BULL": ["SML", "SLM", "MSL"],
        "BEAR": ["MLS", "LSM", "LMS"],
    },
    "MACD_LINE": {
        "BULL": ["M>S+", "M>S-"],
        "BEAR": ["M<S-", "M<S+"],
    },
    "MACD_HISTOGRAM": {
        "BULL": ["H+up", "H+dn"],
        "BEAR": ["H-dn", "H-up"],
    },
    "VWAP": {
        "BULL": [">+2σ", ">+1σ", ">V"],
        "BEAR": ["<V", "<-1σ", "<-2σ"],
        "NEUTRAL": ["@V"],
    },
    "RVOL": {
        "BULL": ["EXTREME", "HIGH"],
        "BEAR": ["LOW", "MINIMAL"],
        "NEUTRAL": ["NORMAL"],
    },
    "UTBOT_V2": {
        "BULL": ["BULL"],
        "BEAR": ["BEAR"],
    },
    "UTBOT": {
        "BULL": ["BULL"],
        "BEAR": ["BEAR"],
    },
    "EMA_PRICE_POSITION_V2": {
        "BULL": ["PSML", "PSLM", "PMSL", "SPML", "SPLM", "SMPL"],
        "BEAR": ["MLSP", "LMSP", "LSMP", "MSLP", "SLMP", "SMLP",
                 "LMPS", "MLPS", "LSPM", "SLPM", "LPMS", "LPSM",
                 "MPLS", "MPSL", "PLMS", "PLSM", "PMLS"],
    },
    "EMA_PRICE_POSITION": {
        "BULL": ["PSML", "PSLM", "PMSL", "SPML", "SPLM", "SMPL"],
        "BEAR": ["MLSP", "LMSP", "LSMP", "MSLP", "SLMP", "SMLP",
                 "LMPS", "MLPS", "LSPM", "SLPM", "LPMS", "LPSM",
                 "MPLS", "MPSL", "PLMS", "PLSM", "PMLS"],
    },
    "SWING_123": {
        "BULL": ["BULL_C3", "BULL_C2"],
        "BEAR": ["BEAR_C3", "BEAR_C2"],
        "NEUTRAL": ["NEUTRAL"],
    },
}


def expand_direction_to_states(interpreter: str, direction: str) -> List[str]:
    """Given an interpreter and a BULL/BEAR/NEUTRAL direction, return the list of
    actual state labels. Unknown interpreter/direction → empty list.
    """
    return INTERPRETER_DIRECTION_MAP.get(interpreter, {}).get(direction, [])


# =============================================================================
# TEMPLATE DEFINITIONS
# =============================================================================

# These define what parameters and triggers are available for each template type

TEMPLATES: Dict[str, Dict] = {
    "ema_stack": {
        "name": "EMA Stack",
        "category": "Moving Averages",
        "description": "Three EMAs for trend direction and momentum",
        "interpreters": ["EMA_STACK"],
        "trigger_prefix": "ema",
        "parameters_schema": {
            "short_period": {"type": "int", "default": 9, "min": 1, "max": 200, "label": "Short Period"},
            "mid_period": {"type": "int", "default": 21, "min": 1, "max": 200, "label": "Mid Period"},
            "long_period": {"type": "int", "default": 200, "min": 1, "max": 500, "label": "Long Period"},
        },
        "plot_schema": {
            "short_color": {"type": "color", "default": "#22c55e", "label": "Short EMA Color"},
            "mid_color": {"type": "color", "default": "#eab308", "label": "Mid EMA Color"},
            "long_color": {"type": "color", "default": "#ef4444", "label": "Long EMA Color"},
        },
        "outputs": ["SML", "SLM", "MSL", "MLS", "LSM", "LMS"],
        "output_descriptions": {
            "SML": "Full Bull Stack — Short > Mid > Long",
            "SLM": "Short leading, Mid dipped below Long",
            "MSL": "Mid leading, Short pulling back",
            "MLS": "Mid leading, bearish Short",
            "LSM": "Long on top, transitional",
            "LMS": "Full Bear Stack — Long > Mid > Short",
        },
        "triggers": [
            {"base": "cross_bull", "name": "Short > Mid Cross", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
            {"base": "cross_bear", "name": "Short < Mid Cross", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
            {"base": "mid_cross_bull", "name": "Mid > Long Cross", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
            {"base": "mid_cross_bear", "name": "Mid < Long Cross", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
        ],
        "indicator_columns": ["ema_short", "ema_mid", "ema_long"],
    },

    "macd_line": {
        "name": "MACD Line",
        "category": "Momentum",
        "description": "MACD line vs Signal line with zero-line context",
        "interpreters": ["MACD_LINE"],
        "trigger_prefix": "macd",
        "parameters_schema": {
            "fast_period": {"type": "int", "default": 12, "min": 1, "max": 100, "label": "Fast Period"},
            "slow_period": {"type": "int", "default": 26, "min": 1, "max": 100, "label": "Slow Period"},
            "signal_period": {"type": "int", "default": 9, "min": 1, "max": 50, "label": "Signal Period"},
        },
        "plot_schema": {
            "macd_color": {"type": "color", "default": "#2563eb", "label": "MACD Line Color"},
            "signal_color": {"type": "color", "default": "#f97316", "label": "Signal Line Color"},
        },
        "outputs": ["M>S+", "M>S-", "M<S-", "M<S+"],
        "output_descriptions": {
            "M>S+": "MACD above signal, above zero (strong bull)",
            "M>S-": "MACD above signal, below zero (recovering)",
            "M<S-": "MACD below signal, below zero (strong bear)",
            "M<S+": "MACD below signal, above zero (weakening)",
        },
        "triggers": [
            {"base": "cross_bull", "name": "Bullish Cross", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
            {"base": "cross_bear", "name": "Bearish Cross", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
            {"base": "zero_cross_up", "name": "Zero Line Cross Up", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
            {"base": "zero_cross_down", "name": "Zero Line Cross Down", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
        ],
        "indicator_columns": ["macd_line", "macd_signal"],
    },

    "macd_histogram": {
        "name": "MACD Histogram",
        "category": "Momentum",
        "description": "MACD histogram momentum and direction",
        "interpreters": ["MACD_HISTOGRAM"],
        "trigger_prefix": "macd_hist",
        "parameters_schema": {
            "fast_period": {"type": "int", "default": 12, "min": 1, "max": 100, "label": "Fast Period"},
            "slow_period": {"type": "int", "default": 26, "min": 1, "max": 100, "label": "Slow Period"},
            "signal_period": {"type": "int", "default": 9, "min": 1, "max": 50, "label": "Signal Period"},
        },
        "plot_schema": {
            "hist_pos_color": {"type": "color", "default": "#22c55e", "label": "Histogram Positive"},
            "hist_neg_color": {"type": "color", "default": "#ef4444", "label": "Histogram Negative"},
        },
        "outputs": ["H+up", "H+dn", "H-dn", "H-up"],
        "output_descriptions": {
            "H+up": "Positive and rising (accelerating bullish)",
            "H+dn": "Positive but falling (decelerating bullish)",
            "H-dn": "Negative and falling (accelerating bearish)",
            "H-up": "Negative but rising (decelerating bearish)",
        },
        "triggers": [
            {"base": "flip_pos", "name": "Histogram Flip Bullish", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
            {"base": "flip_neg", "name": "Histogram Flip Bearish", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
            {"base": "momentum_shift_up", "name": "Momentum Shift Up", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
            {"base": "momentum_shift_down", "name": "Momentum Shift Down", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
        ],
        "indicator_columns": ["macd_hist"],
    },

    "vwap": {
        "name": "VWAP",
        "category": "Volume",
        "description": "Volume Weighted Average Price with SD bands (7-zone system)",
        "interpreters": ["VWAP"],
        "trigger_prefix": "vwap",
        "parameters_schema": {
            "sd1_mult": {"type": "float", "default": 1.0, "min": 0.5, "max": 5.0, "label": "Inner Band (SD1) Multiplier"},
            "sd2_mult": {"type": "float", "default": 2.0, "min": 0.5, "max": 5.0, "label": "Outer Band (SD2) Multiplier"},
        },
        "plot_schema": {
            "vwap_color": {"type": "color", "default": "#8b5cf6", "label": "VWAP Line Color"},
            "sd1_band_color": {"type": "color", "default": "#c4b5fd", "label": "SD1 Band Color"},
            "sd2_band_color": {"type": "color", "default": "#ddd6fe", "label": "SD2 Band Color"},
        },
        "outputs": [">+2σ", ">+1σ", ">V", "@V", "<V", "<-1σ", "<-2σ"],
        "output_descriptions": {
            ">+2σ": "Price above VWAP + 2×SD (extended high)",
            ">+1σ": "Price between +1σ and +2σ",
            ">V": "Price between VWAP and +1σ (above VWAP zone)",
            "@V": "Price within ±0.5σ of VWAP (at VWAP)",
            "<V": "Price between VWAP and -1σ (below VWAP zone)",
            "<-1σ": "Price between -1σ and -2σ",
            "<-2σ": "Price below VWAP - 2×SD (extended low)",
        },
        "triggers": [
            {"base": "cross_above", "name": "Cross Above VWAP", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
            {"base": "cross_above_ib", "name": "Cross Above VWAP", "direction": "BOTH", "type": "BOTH", "execution": "intra_bar", "exec_variants": {"C": {"enabled": False}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False}}, "column_base": "cross_above"},
            {"base": "cross_below", "name": "Cross Below VWAP", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
            {"base": "cross_below_ib", "name": "Cross Below VWAP", "direction": "BOTH", "type": "BOTH", "execution": "intra_bar", "exec_variants": {"C": {"enabled": False}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False}}, "column_base": "cross_below"},
            {"base": "enter_upper_extreme", "name": "Enter Upper Extreme (>+2σ)", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
            {"base": "enter_upper_extreme_ib", "name": "Enter Upper Extreme (>+2σ)", "direction": "BOTH", "type": "BOTH", "execution": "intra_bar", "exec_variants": {"C": {"enabled": False}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False}}, "column_base": "enter_upper_extreme"},
            {"base": "enter_lower_extreme", "name": "Enter Lower Extreme (<-2σ)", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
            {"base": "enter_lower_extreme_ib", "name": "Enter Lower Extreme (<-2σ)", "direction": "BOTH", "type": "BOTH", "execution": "intra_bar", "exec_variants": {"C": {"enabled": False}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False}}, "column_base": "enter_lower_extreme"},
            {"base": "cross_above_hm", "name": "Cross Above VWAP (HM)", "direction": "BOTH", "type": "BOTH", "execution": "hybrid_market", "column_base": "cross_above"},
            {"base": "cross_above_hl", "name": "Cross Above VWAP (HL)", "direction": "BOTH", "type": "BOTH", "execution": "hybrid_limit", "column_base": "cross_above"},
            {"base": "cross_below_hm", "name": "Cross Below VWAP (HM)", "direction": "BOTH", "type": "BOTH", "execution": "hybrid_market", "column_base": "cross_below"},
            {"base": "cross_below_hl", "name": "Cross Below VWAP (HL)", "direction": "BOTH", "type": "BOTH", "execution": "hybrid_limit", "column_base": "cross_below"},
            {"base": "enter_upper_extreme_hm", "name": "Enter Upper Extreme (HM)", "direction": "BOTH", "type": "BOTH", "execution": "hybrid_market", "column_base": "enter_upper_extreme"},
            {"base": "enter_upper_extreme_hl", "name": "Enter Upper Extreme (HL)", "direction": "BOTH", "type": "BOTH", "execution": "hybrid_limit", "column_base": "enter_upper_extreme"},
            {"base": "enter_lower_extreme_hm", "name": "Enter Lower Extreme (HM)", "direction": "BOTH", "type": "BOTH", "execution": "hybrid_market", "column_base": "enter_lower_extreme"},
            {"base": "enter_lower_extreme_hl", "name": "Enter Lower Extreme (HL)", "direction": "BOTH", "type": "BOTH", "execution": "hybrid_limit", "column_base": "enter_lower_extreme"},
            {"base": "return_to_vwap", "name": "Return to VWAP Zone", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
        ],
        "indicator_columns": ["vwap", "vwap_sd1_upper", "vwap_sd1_lower", "vwap_sd2_upper", "vwap_sd2_lower"],
    },

    "rvol": {
        "name": "Relative Volume",
        "category": "Volume",
        "description": "Current volume relative to historical average",
        "interpreters": ["RVOL"],
        "trigger_prefix": "rvol",
        "parameters_schema": {
            "sma_period": {"type": "int", "default": 20, "min": 5, "max": 100, "label": "SMA Period"},
            "high_threshold": {"type": "float", "default": 1.5, "min": 1.0, "max": 5.0, "label": "High Threshold"},
            "extreme_threshold": {"type": "float", "default": 3.0, "min": 2.0, "max": 10.0, "label": "Extreme Threshold"},
        },
        "plot_schema": {
            "bar_color": {"type": "color", "default": "#64748b", "label": "Volume Bar Color"},
            "high_color": {"type": "color", "default": "#f59e0b", "label": "High Volume Color"},
            "extreme_color": {"type": "color", "default": "#ef4444", "label": "Extreme Volume Color"},
        },
        "outputs": ["EXTREME", "HIGH", "NORMAL", "LOW", "MINIMAL"],
        "output_descriptions": {
            "EXTREME": "Volume > 300% of average",
            "HIGH": "Volume > 150% of average",
            "NORMAL": "Volume 75-150% of average",
            "LOW": "Volume 50-75% of average",
            "MINIMAL": "Volume < 50% of average",
        },
        "triggers": [
            {"base": "spike", "name": "Volume Spike", "direction": "BOTH", "type": "ENTRY", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
            {"base": "extreme", "name": "Extreme Volume", "direction": "BOTH", "type": "ENTRY", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
            {"base": "fade", "name": "Volume Fade", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
        ],
        "indicator_columns": ["vol_sma", "rvol"],
    },

    "utbot_v2": {
        "name": "UT Bot (Confirmed)",
        "category": "Trend",
        "description": "UT Bot with 1-bar confirmation delay — signals fire the bar after the ATR trailing stop crossover to avoid repainting",
        "interpreters": ["UTBOT_V2"],
        "trigger_prefix": "utbot_v2",
        "parameters_schema": {
            "atr_period": {"type": "int", "default": 10, "min": 1, "max": 50, "label": "ATR Period"},
            "atr_multiplier": {"type": "float", "default": 1.0, "min": 0.5, "max": 5.0, "label": "ATR Multiplier"},
        },
        "plot_schema": {
            "buy_color": {"type": "color", "default": "#22c55e", "label": "Buy Signal Color"},
            "sell_color": {"type": "color", "default": "#ef4444", "label": "Sell Signal Color"},
            "trail_color": {"type": "color", "default": "#64748b", "label": "Trailing Stop Color"},
        },
        "outputs": ["BULL", "BEAR"],
        "output_descriptions": {
            "BULL": "Price above trailing stop (bullish)",
            "BEAR": "Price below trailing stop (bearish)",
        },
        "triggers": [
            {"base": "buy", "name": "Buy Signal (Confirmed)", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
            {"base": "buy_ib", "name": "Buy Signal (Confirmed)", "direction": "BOTH", "type": "BOTH", "execution": "intra_bar", "exec_variants": {"C": {"enabled": False}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False}}, "column_base": "buy"},
            {"base": "sell", "name": "Sell Signal (Confirmed)", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
            {"base": "sell_ib", "name": "Sell Signal (Confirmed)", "direction": "BOTH", "type": "BOTH", "execution": "intra_bar", "exec_variants": {"C": {"enabled": False}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False}}, "column_base": "sell"},
            {"base": "buy_hm", "name": "Buy Signal (Confirmed, HM)", "direction": "BOTH", "type": "BOTH", "execution": "hybrid_market", "column_base": "buy"},
            {"base": "buy_hl", "name": "Buy Signal (Confirmed, HL)", "direction": "BOTH", "type": "BOTH", "execution": "hybrid_limit", "column_base": "buy"},
            {"base": "sell_hm", "name": "Sell Signal (Confirmed, HM)", "direction": "BOTH", "type": "BOTH", "execution": "hybrid_market", "column_base": "sell"},
            {"base": "sell_hl", "name": "Sell Signal (Confirmed, HL)", "direction": "BOTH", "type": "BOTH", "execution": "hybrid_limit", "column_base": "sell"},
        ],
        "indicator_columns": ["utbot_stop"],
    },

    "ema_price_position_v2": {
        "name": "EMA Price Position (Confirmed)",
        "category": "Moving Averages",
        "description": "Price position within the EMA stack with 1-bar confirmation delay — crossover signals fire the bar after the cross to avoid repainting",
        "interpreters": ["EMA_PRICE_POSITION_V2"],
        "trigger_prefix": "ema_pp_v2",
        "parameters_schema": {
            "short_period": {"type": "int", "default": 9, "min": 1, "max": 200, "label": "Short Period"},
            "mid_period": {"type": "int", "default": 21, "min": 1, "max": 200, "label": "Mid Period"},
            "long_period": {"type": "int", "default": 200, "min": 1, "max": 500, "label": "Long Period"},
        },
        "plot_schema": {
            "short_color": {"type": "color", "default": "#22c55e", "label": "Short EMA Color"},
            "mid_color": {"type": "color", "default": "#eab308", "label": "Mid EMA Color"},
            "long_color": {"type": "color", "default": "#ef4444", "label": "Long EMA Color"},
        },
        "outputs": [
            "PSML", "PSLM", "PMSL", "PMLS", "PLSM", "PLMS",
            "SPML", "SPLM", "MPSL", "MPLS", "LPSM", "LPMS",
            "SMPL", "MSPL", "SLPM", "LSPM", "MLPS", "LMPS",
            "SMLP", "SLMP", "MSLP", "MLSP", "LSMP", "LMSP",
        ],
        "output_descriptions": {
            "PSML": "Price leading, full bull (P > S > M > L)",
            "PSLM": "Price leading, mid dipped (P > S > L > M)",
            "PMSL": "Price above mid, short fell (P > M > S > L)",
            "PMLS": "Price above mid only (P > M > L > S)",
            "PLSM": "Price above long only (P > L > S > M)",
            "PLMS": "Price above long, bear EMAs (P > L > M > S)",
            "SPML": "Below short, bull EMAs (S > P > M > L)",
            "SPLM": "Below short, mid dipped (S > P > L > M)",
            "MPSL": "Below mid, mid leading (M > P > S > L)",
            "MPLS": "Below mid (M > P > L > S)",
            "LPSM": "Below long (L > P > S > M)",
            "LPMS": "Below long (L > P > M > S)",
            "SMPL": "Between mid and long, bull (S > M > P > L)",
            "MSPL": "Between mid and long (M > S > P > L)",
            "SLPM": "Between long and mid (S > L > P > M)",
            "LSPM": "Between long and mid (L > S > P > M)",
            "MLPS": "Between long and short (M > L > P > S)",
            "LMPS": "Between long and short (L > M > P > S)",
            "SMLP": "Below all, bull EMAs (S > M > L > P)",
            "SLMP": "Below all (S > L > M > P)",
            "MSLP": "Below all (M > S > L > P)",
            "MLSP": "Below all (M > L > S > P)",
            "LSMP": "Below all (L > S > M > P)",
            "LMSP": "Below all, bear EMAs (L > M > S > P)",
        },
        "triggers": [
            {"base": "cross_short_up", "name": "Price > Short EMA (Confirmed)", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
            {"base": "cross_short_up_ib", "name": "Price > Short EMA (Confirmed)", "direction": "BOTH", "type": "BOTH", "execution": "intra_bar", "exec_variants": {"C": {"enabled": False}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False}}, "column_base": "cross_short_up"},
            {"base": "cross_short_down", "name": "Price < Short EMA (Confirmed)", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
            {"base": "cross_short_down_ib", "name": "Price < Short EMA (Confirmed)", "direction": "BOTH", "type": "BOTH", "execution": "intra_bar", "exec_variants": {"C": {"enabled": False}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False}}, "column_base": "cross_short_down"},
            {"base": "cross_mid_up", "name": "Price > Mid EMA (Confirmed)", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
            {"base": "cross_mid_up_ib", "name": "Price > Mid EMA (Confirmed)", "direction": "BOTH", "type": "BOTH", "execution": "intra_bar", "exec_variants": {"C": {"enabled": False}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False}}, "column_base": "cross_mid_up"},
            {"base": "cross_mid_down", "name": "Price < Mid EMA (Confirmed)", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
            {"base": "cross_mid_down_ib", "name": "Price < Mid EMA (Confirmed)", "direction": "BOTH", "type": "BOTH", "execution": "intra_bar", "exec_variants": {"C": {"enabled": False}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False}}, "column_base": "cross_mid_down"},
            {"base": "cross_short_up_hm", "name": "Price > Short EMA (Confirmed, HM)", "direction": "BOTH", "type": "BOTH", "execution": "hybrid_market", "column_base": "cross_short_up"},
            {"base": "cross_short_up_hl", "name": "Price > Short EMA (Confirmed, HL)", "direction": "BOTH", "type": "BOTH", "execution": "hybrid_limit", "column_base": "cross_short_up"},
            {"base": "cross_short_down_hm", "name": "Price < Short EMA (Confirmed, HM)", "direction": "BOTH", "type": "BOTH", "execution": "hybrid_market", "column_base": "cross_short_down"},
            {"base": "cross_short_down_hl", "name": "Price < Short EMA (Confirmed, HL)", "direction": "BOTH", "type": "BOTH", "execution": "hybrid_limit", "column_base": "cross_short_down"},
            {"base": "cross_mid_up_hm", "name": "Price > Mid EMA (Confirmed, HM)", "direction": "BOTH", "type": "BOTH", "execution": "hybrid_market", "column_base": "cross_mid_up"},
            {"base": "cross_mid_up_hl", "name": "Price > Mid EMA (Confirmed, HL)", "direction": "BOTH", "type": "BOTH", "execution": "hybrid_limit", "column_base": "cross_mid_up"},
            {"base": "cross_mid_down_hm", "name": "Price < Mid EMA (Confirmed, HM)", "direction": "BOTH", "type": "BOTH", "execution": "hybrid_market", "column_base": "cross_mid_down"},
            {"base": "cross_mid_down_hl", "name": "Price < Mid EMA (Confirmed, HL)", "direction": "BOTH", "type": "BOTH", "execution": "hybrid_limit", "column_base": "cross_mid_down"},
            {"base": "cross_short_up_lc", "name": "Price > Short EMA (LC)", "direction": "BOTH", "type": "BOTH", "execution": "level_close", "column_base": "cross_short_up"},
            {"base": "cross_short_down_lc", "name": "Price < Short EMA (LC)", "direction": "BOTH", "type": "BOTH", "execution": "level_close", "column_base": "cross_short_down"},
            {"base": "cross_mid_up_lc", "name": "Price > Mid EMA (LC)", "direction": "BOTH", "type": "BOTH", "execution": "level_close", "column_base": "cross_mid_up"},
            {"base": "cross_mid_down_lc", "name": "Price < Mid EMA (LC)", "direction": "BOTH", "type": "BOTH", "execution": "level_close", "column_base": "cross_mid_down"},
            {"base": "cross_short_up_cc", "name": "Price > Short EMA (CC)", "direction": "BOTH", "type": "BOTH", "execution": "close_close", "column_base": "cross_short_up"},
            {"base": "cross_short_down_cc", "name": "Price < Short EMA (CC)", "direction": "BOTH", "type": "BOTH", "execution": "close_close", "column_base": "cross_short_down"},
            {"base": "cross_mid_up_cc", "name": "Price > Mid EMA (CC)", "direction": "BOTH", "type": "BOTH", "execution": "close_close", "column_base": "cross_mid_up"},
            {"base": "cross_mid_down_cc", "name": "Price < Mid EMA (CC)", "direction": "BOTH", "type": "BOTH", "execution": "close_close", "column_base": "cross_mid_down"},
        ],
        "indicator_columns": ["ema_short", "ema_mid", "ema_long"],
    },

    "swing_123": {
        "name": "Swing 1-2-3",
        "category": "Trend",
        "description": "Candle 2 and Candle 3 pattern detection for swing reversal and continuation signals",
        "interpreters": ["SWING_123"],
        "trigger_prefix": "sw123",
        "parameters_schema": {},
        "plot_schema": {
            "bull_c2_color": {"type": "color", "default": "#FFD11A", "label": "Bullish C2 Color"},
            "bull_c3_color": {"type": "color", "default": "#FFFF00", "label": "Bullish C3 Color"},
            "bear_c2_color": {"type": "color", "default": "#FF66B3", "label": "Bearish C2 Color"},
            "bear_c3_color": {"type": "color", "default": "#FF33CC", "label": "Bearish C3 Color"},
        },
        "plot_config": {"candle_color_column": "sw123_candle_color"},
        "outputs": ["BULL_C3", "BULL_C2", "BEAR_C3", "BEAR_C2", "NEUTRAL"],
        "output_descriptions": {
            "BULL_C3": "Bullish Candle 3: Prior bar was C2 and current close > prior high (continuation confirmed)",
            "BULL_C2": "Bullish Candle 2: Made lower low but closed above prior close (reversal candidate)",
            "BEAR_C3": "Bearish Candle 3: Prior bar was C2 and current close < prior low (continuation confirmed)",
            "BEAR_C2": "Bearish Candle 2: Made higher high but closed below prior close (reversal candidate)",
            "NEUTRAL": "No swing pattern detected on this bar",
        },
        "triggers": [
            {"base": "bull_c2", "name": "Bullish Candle 2", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": False, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": False, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": True, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
            {"base": "bull_c3", "name": "Bullish Candle 3", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": False}, "LC": {"enabled": False}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
            {"base": "bear_c2", "name": "Bearish Candle 2", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": False, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": False, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": True, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
            {"base": "bear_c3", "name": "Bearish Candle 3", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": False}, "LC": {"enabled": False}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
            {"base": "bull_c2_cc", "name": "Bullish C2 \u2192 C3 Confirmed (CC)", "direction": "BOTH", "type": "BOTH", "execution": "close_close", "column_base": "bull_c2"},
            {"base": "bear_c2_cc", "name": "Bearish C2 \u2192 C3 Confirmed (CC)", "direction": "BOTH", "type": "BOTH", "execution": "close_close", "column_base": "bear_c2"},
        ],
        "indicator_columns": ["sw123_pattern", "sw123_candle_color"],
        "display_type": "hidden",
    },

    "bar_count": {
        "name": "Bar Count Exit",
        "trigger_prefix": "bar_count",
        "category": "exit",
        "parameters_schema": {
            "candle_count": {"type": "int", "default": 4, "min": 1, "max": 500, "label": "Candle Count"},
        },
        "plot_schema": {},
        "outputs": [],
        "output_descriptions": {},
        "triggers": [
            {"base": "exit", "name": "Exit After N Candles", "direction": "BOTH", "type": "BOTH", "execution": "bar_close", "exec_variants": {"C": {"enabled": True, "reference_bar": 0, "order_type": "market"}, "L": {"enabled": True, "reference_bar": -1, "order_type": "market", "hold_seconds": 0}, "LC": {"enabled": True, "confirm_bar_offset": 0, "bail_action": "exit_market"}, "CC": {"enabled": False, "confirm_bar_offset": 1, "bail_action": "exit_market"}}},
        ],
        "indicator_columns": [],
    },

    # ---- Legacy template aliases ----
    # Old confluence groups reference these names. They use the same interpreters
    # and indicator columns as their v2 counterparts but without confirmation delay.
    "utbot": {
        "name": "UT Bot (Legacy)",
        "category": "Trend",
        "description": "UT Bot ATR trailing stop — legacy version without confirmation delay",
        "interpreters": ["UTBOT"],
        "trigger_prefix": "utbot",
        "parameters_schema": {
            "atr_period": {"type": "int", "default": 10, "min": 1, "max": 50, "label": "ATR Period"},
            "atr_multiplier": {"type": "float", "default": 1.0, "min": 0.5, "max": 5.0, "label": "ATR Multiplier"},
        },
        "plot_schema": {
            "trail_color": {"type": "color", "default": "#64748b", "label": "Trailing Stop Color"},
        },
        "outputs": [],
        "output_descriptions": {},
        "triggers": [
            {"base": "buy", "name": "UT Bot Buy", "direction": "BOTH", "type": "BOTH", "execution": "bar_close"},
            {"base": "sell", "name": "UT Bot Sell", "direction": "BOTH", "type": "BOTH", "execution": "bar_close"},
        ],
        "indicator_columns": ["utbot_stop"],
    },
    "ema_price_position": {
        "name": "EMA Price Position (Legacy)",
        "category": "Moving Averages",
        "description": "Price position within the EMA stack — legacy version without confirmation delay",
        "interpreters": ["EMA_PRICE_POSITION"],
        "trigger_prefix": "ema_pp",
        "parameters_schema": {
            "short_period": {"type": "int", "default": 9, "min": 1, "max": 200, "label": "Short Period"},
            "mid_period": {"type": "int", "default": 21, "min": 1, "max": 200, "label": "Mid Period"},
            "long_period": {"type": "int", "default": 200, "min": 1, "max": 500, "label": "Long Period"},
        },
        "plot_schema": {
            "short_color": {"type": "color", "default": "#22c55e", "label": "Short EMA Color"},
            "mid_color": {"type": "color", "default": "#eab308", "label": "Mid EMA Color"},
            "long_color": {"type": "color", "default": "#ef4444", "label": "Long EMA Color"},
        },
        "outputs": [],
        "output_descriptions": {},
        "triggers": [
            {"base": "cross_short_up", "name": "Price Crosses Above Short EMA", "direction": "BOTH", "type": "BOTH", "execution": "bar_close"},
            {"base": "cross_short_down", "name": "Price Crosses Below Short EMA", "direction": "BOTH", "type": "BOTH", "execution": "bar_close"},
        ],
        "indicator_columns": ["ema_short", "ema_mid", "ema_long"],
    },
}


# =============================================================================
# FILE OPERATIONS
# =============================================================================

def get_config_path() -> Path:
    """Get the path to the confluence groups config file."""
    # Look relative to this file's location
    src_dir = Path(__file__).parent
    project_dir = src_dir.parent
    return project_dir / "config" / "confluence_groups.json"


def _parse_group_list(raw_groups: list) -> List[ConfluenceGroup]:
    """Convert list of dicts to ConfluenceGroup objects with backward compat."""
    groups = []
    for group_data in raw_groups:
        # Parse plot settings
        plot_data = group_data.get("plot_settings", {})
        plot_settings = PlotSettings(
            colors=plot_data.get("colors", {}),
            line_width=plot_data.get("line_width", 1),
            visible=plot_data.get("visible", True),
        )

        # Support both "version" (new) and "name" (legacy) fields
        version = group_data.get("version")
        if not version:
            legacy_name = group_data.get("name", "")
            if "(" in legacy_name and ")" in legacy_name:
                version = legacy_name.split("(")[-1].rstrip(")")
            else:
                version = "Default"

        # Backward compat: old "macd" template -> "macd_line"
        base_template = group_data["base_template"]
        if base_template == "macd":
            base_template = "macd_line"

        # Backward compat: old VWAP parameters
        params = group_data.get("parameters", {})
        if base_template == "vwap" and "std_dev" in params and "sd1_mult" not in params:
            old_std = params.pop("std_dev")
            params["sd1_mult"] = 1.0
            params["sd2_mult"] = float(old_std)
            params.pop("tolerance_pct", None)

        # Guard: skip groups referencing templates not in TEMPLATES
        if base_template not in TEMPLATES:
            print(f"Warning: skipping group '{group_data['id']}' — "
                  f"template '{base_template}' not found "
                  f"(user pack may be removed)")
            continue

        group = ConfluenceGroup(
            id=group_data["id"],
            base_template=base_template,
            version=version,
            description=group_data.get("description", ""),
            enabled=group_data.get("enabled", True),
            is_default=group_data.get("is_default", False),
            parameters=params,
            plot_settings=plot_settings,
        )
        groups.append(group)

    # Migration: add bar_count_default if no bar_count group exists
    if not any(g.base_template == "bar_count" for g in groups):
        groups.append(ConfluenceGroup(
            id="bar_count_default",
            base_template="bar_count",
            version="Default",
            description="Exit after 4 candles if no other exit triggers fire",
            enabled=True,
            is_default=True,
            parameters={"candle_count": 4},
            plot_settings=PlotSettings(colors={}, line_width=1, visible=False),
        ))

    # Migration: add swing_123_default if no swing_123 group exists
    if not any(g.base_template == "swing_123" for g in groups):
        groups.append(ConfluenceGroup(
            id="swing_123_default",
            base_template="swing_123",
            version="Default",
            description="Swing 1-2-3 candle pattern detection with C2/C3 entries",
            enabled=True,
            is_default=True,
            parameters={},
            plot_settings=PlotSettings(
                colors={
                    "bull_c2_color": "#FFD11A",
                    "bull_c3_color": "#FFFF00",
                    "bear_c2_color": "#FF66B3",
                    "bear_c3_color": "#FF33CC",
                },
                line_width=1,
                visible=True,
            ),
        ))

    return groups


def _serialize_group_list(groups: List[ConfluenceGroup]) -> list:
    """Convert ConfluenceGroup objects to list of dicts for storage."""
    return [{
        "id": g.id,
        "base_template": g.base_template,
        "version": g.version,
        "description": g.description,
        "enabled": g.enabled,
        "is_default": g.is_default,
        "parameters": g.parameters,
        "plot_settings": {
            "colors": g.plot_settings.colors,
            "line_width": g.plot_settings.line_width,
            "visible": g.plot_settings.visible,
        },
    } for g in groups]


def load_confluence_groups() -> List[ConfluenceGroup]:
    """
    Load confluence groups from the config file or database.

    Returns list of ConfluenceGroup objects.
    """
    from db import USE_DB
    if USE_DB:
        from db import load_confluence_groups_db
        raw = load_confluence_groups_db()
        if not raw:
            groups = create_default_groups()
            try:
                save_confluence_groups(groups)
            except Exception:
                pass  # Return in-memory defaults even if save fails (e.g. no JWT)
            return groups
        return _parse_group_list(raw)

    config_path = get_config_path()

    if not config_path.exists():
        # Return default groups if no config exists
        return create_default_groups()

    try:
        with open(config_path, 'r') as f:
            data = json.load(f)

        groups = []
        for group_data in data.get("groups", []):
            # Parse plot settings
            plot_data = group_data.get("plot_settings", {})
            plot_settings = PlotSettings(
                colors=plot_data.get("colors", {}),
                line_width=plot_data.get("line_width", 1),
                visible=plot_data.get("visible", True),
            )

            # Support both "version" (new) and "name" (legacy) fields
            version = group_data.get("version")
            if not version:
                # Extract version from legacy "name" field like "EMA Stack (Default)"
                legacy_name = group_data.get("name", "")
                if "(" in legacy_name and ")" in legacy_name:
                    version = legacy_name.split("(")[-1].rstrip(")")
                else:
                    version = "Default"

            # Backward compat: old "macd" template → "macd_line"
            base_template = group_data["base_template"]
            if base_template == "macd":
                base_template = "macd_line"

            # Backward compat: old VWAP parameters (std_dev) → (sd1_mult, sd2_mult)
            params = group_data.get("parameters", {})
            if base_template == "vwap" and "std_dev" in params and "sd1_mult" not in params:
                old_std = params.pop("std_dev")
                params["sd1_mult"] = 1.0
                params["sd2_mult"] = float(old_std)
                params.pop("tolerance_pct", None)

            # Guard: skip groups referencing templates not in TEMPLATES
            # (e.g., a user pack that was removed)
            if base_template not in TEMPLATES:
                print(f"Warning: skipping group '{group_data['id']}' — "
                      f"template '{base_template}' not found "
                      f"(user pack may be removed)")
                continue

            group = ConfluenceGroup(
                id=group_data["id"],
                base_template=base_template,
                version=version,
                description=group_data.get("description", ""),
                enabled=group_data.get("enabled", True),
                is_default=group_data.get("is_default", False),
                parameters=params,
                plot_settings=plot_settings,
            )
            groups.append(group)

        # Migration: add bar_count_default if no bar_count group exists
        if not any(g.base_template == "bar_count" for g in groups):
            groups.append(ConfluenceGroup(
                id="bar_count_default",
                base_template="bar_count",
                version="Default",
                description="Exit after 4 candles if no other exit triggers fire",
                enabled=True,
                is_default=True,
                parameters={"candle_count": 4},
                plot_settings=PlotSettings(colors={}, line_width=1, visible=False),
            ))

        # Migration: add swing_123_default if no swing_123 group exists
        if not any(g.base_template == "swing_123" for g in groups):
            groups.append(ConfluenceGroup(
                id="swing_123_default",
                base_template="swing_123",
                version="Default",
                description="Swing 1-2-3 candle pattern detection with C2/C3 entries",
                enabled=True,
                is_default=True,
                parameters={},
                plot_settings=PlotSettings(
                    colors={
                        "bull_c2_color": "#FFD11A",
                        "bull_c3_color": "#FFFF00",
                        "bear_c2_color": "#FF66B3",
                        "bear_c3_color": "#FF33CC",
                    },
                    line_width=1,
                    visible=True,
                ),
            ))

        return groups

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error loading confluence groups: {e}")
        return create_default_groups()


def save_confluence_groups(groups: List[ConfluenceGroup]) -> bool:
    """
    Save confluence groups to the config file or database.

    Returns True if successful.
    """
    from db import USE_DB
    if USE_DB:
        from db import save_confluence_groups_db
        save_confluence_groups_db(_serialize_group_list(groups))
        return True

    config_path = get_config_path()

    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        data = {"version": "1.0", "groups": _serialize_group_list(groups)}

        with open(config_path, 'w') as f:
            json.dump(data, f, indent=2)

        return True

    except Exception as e:
        print(f"Error saving confluence groups: {e}")
        return False


# =============================================================================
# DEFAULT GROUPS
# =============================================================================

def create_default_groups() -> List[ConfluenceGroup]:
    """Create the default set of confluence groups."""
    defaults = [
        ConfluenceGroup(
            id="ema_stack_default",
            base_template="ema_stack",
            version="Default",
            description="Standard EMA stack with 9/21/200 periods",
            enabled=True,
            is_default=True,
            parameters={
                "short_period": 9,
                "mid_period": 21,
                "long_period": 200,
            },
            plot_settings=PlotSettings(
                colors={
                    "short_color": "#22c55e",
                    "mid_color": "#eab308",
                    "long_color": "#ef4444",
                },
                line_width=1,
                visible=True,
            ),
        ),
        ConfluenceGroup(
            id="macd_line_default",
            base_template="macd_line",
            version="Default",
            description="MACD Line vs Signal with 12/26/9 periods",
            enabled=True,
            is_default=True,
            parameters={
                "fast_period": 12,
                "slow_period": 26,
                "signal_period": 9,
            },
            plot_settings=PlotSettings(
                colors={
                    "macd_color": "#2563eb",
                    "signal_color": "#f97316",
                },
                line_width=1,
                visible=True,
            ),
        ),
        ConfluenceGroup(
            id="macd_histogram_default",
            base_template="macd_histogram",
            version="Default",
            description="MACD Histogram momentum with 12/26/9 periods",
            enabled=True,
            is_default=True,
            parameters={
                "fast_period": 12,
                "slow_period": 26,
                "signal_period": 9,
            },
            plot_settings=PlotSettings(
                colors={
                    "hist_pos_color": "#22c55e",
                    "hist_neg_color": "#ef4444",
                },
                line_width=1,
                visible=True,
            ),
        ),
        ConfluenceGroup(
            id="vwap_default",
            base_template="vwap",
            version="Default",
            description="VWAP with 1σ/2σ standard deviation bands (7-zone system)",
            enabled=True,
            is_default=True,
            parameters={
                "sd1_mult": 1.0,
                "sd2_mult": 2.0,
            },
            plot_settings=PlotSettings(
                colors={
                    "vwap_color": "#8b5cf6",
                    "sd1_band_color": "#c4b5fd",
                    "sd2_band_color": "#ddd6fe",
                },
                line_width=2,
                visible=True,
            ),
        ),
        ConfluenceGroup(
            id="rvol_default",
            base_template="rvol",
            version="Default",
            description="Relative volume with 20-period SMA baseline",
            enabled=True,
            is_default=True,
            parameters={
                "sma_period": 20,
                "high_threshold": 1.5,
                "extreme_threshold": 3.0,
            },
            plot_settings=PlotSettings(
                colors={
                    "bar_color": "#64748b",
                    "high_color": "#f59e0b",
                    "extreme_color": "#ef4444",
                },
                line_width=1,
                visible=True,
            ),
        ),
        ConfluenceGroup(
            id="bar_count_default",
            base_template="bar_count",
            version="Default",
            description="Exit after 4 candles if no other exit triggers fire",
            enabled=True,
            is_default=True,
            parameters={"candle_count": 4},
            plot_settings=PlotSettings(
                colors={},
                line_width=1,
                visible=False,
            ),
        ),
        ConfluenceGroup(
            id="swing_123_default",
            base_template="swing_123",
            version="Default",
            description="Swing 1-2-3 candle pattern detection with C2/C3 entries",
            enabled=True,
            is_default=True,
            parameters={},
            plot_settings=PlotSettings(
                colors={
                    "bull_c2_color": "#FFD11A",
                    "bull_c3_color": "#FFFF00",
                    "bear_c2_color": "#FF66B3",
                    "bear_c3_color": "#FF33CC",
                },
                line_width=1,
                visible=True,
            ),
        ),
    ]

    return defaults


# =============================================================================
# GROUP OPERATIONS
# =============================================================================

def get_enabled_groups(groups: Optional[List[ConfluenceGroup]] = None) -> List[ConfluenceGroup]:
    """Get only the enabled confluence groups."""
    if groups is None:
        groups = load_confluence_groups()
    return [g for g in groups if g.enabled]


def get_enabled_interpreter_keys(groups: Optional[List[ConfluenceGroup]] = None) -> List[str]:
    """Derive interpreter keys that correspond to enabled confluence groups.

    Maps each enabled group's base_template to TEMPLATES[base_template]["interpreters"].
    Returns a deduplicated list of interpreter keys (e.g., ["EMA_STACK", "MACD_LINE"]).
    """
    if groups is None:
        groups = get_enabled_groups()
    keys = []
    seen = set()
    for group in groups:
        template = TEMPLATES.get(group.base_template)
        if template:
            for interp_key in template.get("interpreters", []):
                if interp_key not in seen:
                    keys.append(interp_key)
                    seen.add(interp_key)
    return keys


def get_group_by_id(group_id: str, groups: Optional[List[ConfluenceGroup]] = None) -> Optional[ConfluenceGroup]:
    """Get a specific confluence group by ID."""
    if groups is None:
        groups = load_confluence_groups()
    for g in groups:
        if g.id == group_id:
            return g
    return None


def get_groups_by_template(template: str, groups: Optional[List[ConfluenceGroup]] = None) -> List[ConfluenceGroup]:
    """Get all groups based on a specific template."""
    if groups is None:
        groups = load_confluence_groups()
    return [g for g in groups if g.base_template == template]


def duplicate_group(
    source_group: ConfluenceGroup,
    new_id: str,
    new_version: str,
) -> ConfluenceGroup:
    """
    Create a copy of a confluence group with a new ID and version.

    The duplicate is not a default (can be deleted).
    """
    return ConfluenceGroup(
        id=new_id,
        base_template=source_group.base_template,
        version=new_version,
        description=f"Copy of {source_group.name}",
        enabled=True,
        is_default=False,
        parameters=source_group.parameters.copy(),
        plot_settings=PlotSettings(
            colors=source_group.plot_settings.colors.copy(),
            line_width=source_group.plot_settings.line_width,
            visible=source_group.plot_settings.visible,
        ),
    )


def validate_group_id(group_id: str, existing_groups: List[ConfluenceGroup]) -> bool:
    """Check if a group ID is valid and unique."""
    if not group_id:
        return False
    if not group_id.replace("_", "").isalnum():
        return False
    if any(g.id == group_id for g in existing_groups):
        return False
    return True


def generate_unique_id(base_template: str, existing_groups: List[ConfluenceGroup]) -> str:
    """Generate a unique ID for a new group."""
    base = base_template.replace("_", "_")
    counter = 1
    while True:
        candidate = f"{base}_custom_{counter}"
        if validate_group_id(candidate, existing_groups):
            return candidate
        counter += 1


# =============================================================================
# TRIGGER HELPERS
# =============================================================================

def get_group_triggers(group: ConfluenceGroup) -> List[TriggerDefinition]:
    """
    Get all triggers for a confluence group.

    For bar_close triggers without explicit exec_variants, auto-generates
    execution type variants (_ib, _lc, _cc) based on globally enabled types.
    This enables the modular execution type approach: packs define the signal,
    the system generates the variants.

    Returns list of TriggerDefinition with group-specific IDs and names.
    """
    template = TEMPLATES.get(group.base_template)
    if not template:
        return []

    # Execution type variant labels
    EXEC_VARIANT_LABELS = {
        '_ib': ('[L]', 'intra_bar'),
        '_lc': ('[LC]', 'level_close'),
        '_cc': ('[CC]', 'close_close'),
    }

    triggers = []
    for trig_def in template.get("triggers", []):
        base = trig_def["base"]
        execution = trig_def.get("execution", "bar_close")
        has_exec_variants = "exec_variants" in trig_def

        trigger = TriggerDefinition(
            id=group.get_trigger_id(base),
            name=group.get_trigger_name(base, trig_def["name"]),
            base_trigger=base,
            direction=trig_def["direction"],
            trigger_type=trig_def["type"],
            execution=execution,
            column_base=trig_def.get("column_base"),
        )
        triggers.append(trigger)

        # Auto-generate execution type variants for bar_close triggers
        # that don't already have explicit variants or suffixed siblings.
        # L-type (_ib) only generated for packs with indicator overlay lines
        # (level columns to cross). Pattern-based packs (candle coloring, hidden
        # display) only get CC variants since there's no price level to cross.
        if execution == "bar_close" and not has_exec_variants:
            existing_bases = {t["base"] for t in template.get("triggers", [])}
            display_type = template.get("display_type", "overlay")
            has_candle_color = bool(template.get("plot_config", {}).get("candle_color_column"))
            # L-type requires a price level to cross — only for overlay packs
            # without candle coloring (those have actual indicator lines)
            has_level_columns = (
                display_type == "overlay"
                and not has_candle_color
                and any(not t.get("base", "").endswith(("_ib", "_hm", "_hl", "_lc", "_cc"))
                        for t in template.get("triggers", [])
                        if t.get("level_column"))
            )
            for suffix, (label, exec_type) in EXEC_VARIANT_LABELS.items():
                # Skip L-type for packs without level columns
                if suffix == '_ib' and not has_level_columns:
                    continue
                # Skip LC for packs without level columns (LC = level entry + close confirm)
                if suffix == '_lc' and not has_level_columns:
                    continue
                suffixed_base = base + suffix
                if suffixed_base not in existing_bases:
                    variant = TriggerDefinition(
                        id=group.get_trigger_id(suffixed_base),
                        name=group.get_trigger_name(suffixed_base, f"{trig_def['name']} {label}"),
                        base_trigger=suffixed_base,
                        direction=trig_def["direction"],
                        trigger_type=trig_def["type"],
                        execution=exec_type,
                        column_base=base,
                    )
                    triggers.append(variant)

    return triggers


def get_all_triggers(groups: Optional[List[ConfluenceGroup]] = None) -> Dict[str, TriggerDefinition]:
    """
    Get all triggers from all enabled groups.

    Returns dict mapping trigger_id -> TriggerDefinition
    """
    if groups is None:
        groups = get_enabled_groups()

    all_triggers = {}
    for group in groups:
        for trigger in get_group_triggers(group):
            all_triggers[trigger.id] = trigger

    return all_triggers


def _get_enabled_exec_suffixes() -> set:
    """Get the set of trigger suffixes for enabled execution types.

    Reads the user's execution type config to determine which exec types
    are enabled. Returns suffixes like {'', '_ib', '_lc', '_cc'}.
    Empty string = C-type (no suffix).
    """
    enabled_suffixes = set()
    try:
        from db import USE_DB
        if USE_DB:
            from db import load_settings_db
            settings = load_settings_db()
            et_config = settings.get('execution_types', {})
        else:
            et_config = {}

        # Map execution type slugs to their trigger suffixes
        SLUG_TO_SUFFIXES = {
            'bar_close': {''},          # C-type: no suffix
            'level': {'_ib'},           # L-type: _ib suffix
            'level_close': {'_lc'},     # LC-type: _lc suffix (also _hm, _hl legacy)
            'close_close': {'_cc'},     # CC-type: _cc suffix
        }

        for slug, suffixes in SLUG_TO_SUFFIXES.items():
            cfg = et_config.get(slug, {})
            # Default: C and L enabled, LC and CC disabled
            is_enabled = cfg.get('enabled', slug in ('bar_close', 'level'))
            if is_enabled:
                enabled_suffixes.update(suffixes)
    except Exception:
        # Fallback: C and L enabled
        enabled_suffixes = {'', '_ib'}

    return enabled_suffixes


def get_entry_triggers(direction: str, groups: Optional[List[ConfluenceGroup]] = None) -> Dict[str, str]:
    """
    Get all available triggers, filtered by enabled execution types.

    Triggers are direction-agnostic and type-agnostic — users decide how to
    use them (entry/exit, long/short) in Strategy Builder. The `direction`
    parameter is accepted for backward compatibility but not used for filtering.

    Returns dict mapping trigger_id -> display_name
    """
    all_triggers = get_all_triggers(groups)
    enabled_suffixes = _get_enabled_exec_suffixes()

    result = {}
    for trig_id, trig_def in all_triggers.items():
        # Check if this trigger's execution type suffix is enabled
        base = trig_def.base_trigger
        suffix = ''
        for s in ('_ib', '_hm', '_hl', '_lc', '_cc'):
            if base.endswith(s):
                suffix = s
                break

        # Map legacy suffixes
        if suffix in ('_hm', '_hl'):
            suffix = '_lc'  # HM/HL are LC variants

        if suffix in enabled_suffixes:
            result[trig_id] = trig_def.name

    return result


def get_exit_triggers(groups: Optional[List[ConfluenceGroup]] = None) -> Dict[str, str]:
    """
    Get all exit-eligible triggers, filtered by enabled execution types.

    Returns dict mapping trigger_id -> display_name.
    Includes both EXIT-type triggers and all ENTRY triggers (which can
    serve as opposite-signal exits when the direction is reversed).
    """
    all_triggers = get_all_triggers(groups)
    enabled_suffixes = _get_enabled_exec_suffixes()

    result = {}
    for trig_id, trig_def in all_triggers.items():
        # Determine suffix
        base = trig_def.base_trigger
        suffix = ''
        for s in ('_ib', '_hm', '_hl', '_lc', '_cc'):
            if base.endswith(s):
                suffix = s
                break
        if suffix in ('_hm', '_hl'):
            suffix = '_lc'
        if suffix not in enabled_suffixes:
            continue

        if trig_def.trigger_type == "EXIT":
            result[trig_id] = trig_def.name
        elif trig_def.trigger_type == "ENTRY":
            result[trig_id] = trig_def.name

    return result


def get_all_conditions(groups: Optional[List[ConfluenceGroup]] = None) -> Dict[str, str]:
    """Get all available TF confluence conditions from enabled groups.

    Returns dict mapping condition_id -> display_label.
    Condition IDs follow the format: "{tf_label}-{INTERPRETER}-{STATE}"
    e.g. "5M-EMA_STACK-SML", "1D-MACD_LINE-BULL"
    """
    if groups is None:
        groups = get_enabled_groups()

    result = {}
    for group in groups:
        template = TEMPLATES.get(group.base_template)
        if not template:
            continue
        outputs = template.get("outputs", [])
        interpreters = template.get("interpreters", [])
        if not outputs or not interpreters:
            continue
        interp_key = interpreters[0]
        # Generate conditions for common timeframes
        for tf_label in ('1M', '5M', '15M', '1H', '1D'):
            for state in outputs:
                cond_id = f"{tf_label}-{interp_key}-{state}"
                label = f"{group.base_template} ({group.version}) — {tf_label} {state}"
                result[cond_id] = label

    return result


# =============================================================================
# TEMPLATE HELPERS
# =============================================================================

def get_template(template_id: str) -> Optional[Dict]:
    """Get a template definition by ID."""
    return TEMPLATES.get(template_id)


def get_template_categories() -> Dict[str, List[str]]:
    """Get templates organized by category."""
    categories = {}
    for template_id, template in TEMPLATES.items():
        category = template["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append(template_id)
    return categories


def get_parameter_schema(template_id: str) -> Dict:
    """Get the parameter schema for a template."""
    template = TEMPLATES.get(template_id)
    if template:
        return template.get("parameters_schema", {})
    return {}


def get_plot_schema(template_id: str) -> Dict:
    """Get the plot settings schema for a template."""
    template = TEMPLATES.get(template_id)
    if template:
        return template.get("plot_schema", {})
    return {}


def get_output_descriptions(template_id: str) -> Dict[str, str]:
    """Get the output descriptions for a template."""
    template = TEMPLATES.get(template_id)
    if template:
        return template.get("output_descriptions", {})
    return {}


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Confluence Groups Test")
    print("=" * 60)

    # Load groups
    groups = load_confluence_groups()
    print(f"\nLoaded {len(groups)} groups:")
    for g in groups:
        status = "enabled" if g.enabled else "disabled"
        default = " (default)" if g.is_default else ""
        print(f"  - {g.name} [{g.id}] - {status}{default}")
        print(f"    Template: {g.base_template}")
        print(f"    Parameters: {g.parameters}")

    # Get triggers for a group
    print("\n" + "-" * 40)
    ema_group = get_group_by_id("ema_stack_default", groups)
    if ema_group:
        print(f"\nTriggers for {ema_group.name}:")
        for trig in get_group_triggers(ema_group):
            print(f"  - {trig.id}: {trig.name} ({trig.direction} {trig.trigger_type})")

    # Get all entry triggers for LONG
    print("\n" + "-" * 40)
    print("\nAll LONG entry triggers:")
    for trig_id, trig_name in get_entry_triggers("LONG", groups).items():
        print(f"  - {trig_id}: {trig_name}")

    # Test duplicate
    print("\n" + "-" * 40)
    if ema_group:
        new_id = generate_unique_id("ema_stack", groups)
        dup = duplicate_group(ema_group, new_id, "EMA Stack (Aggressive)")
        dup.parameters["short_period"] = 5
        dup.parameters["mid_period"] = 13
        dup.parameters["long_period"] = 50
        print(f"\nDuplicated to: {dup.name} [{dup.id}]")
        print(f"  Parameters: {dup.parameters}")

    # Save and reload
    print("\n" + "-" * 40)
    if save_confluence_groups(groups):
        print("\nSaved groups to config file")
        print(f"Config path: {get_config_path()}")
