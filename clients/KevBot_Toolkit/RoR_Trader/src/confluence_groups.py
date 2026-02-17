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
        """Get the display name for a trigger: '{Confluence Group}: {Trigger}'."""
        return f"{self.name}: {base_name}"


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
            "SML": "Bullish - Short > Mid > Long",
            "SLM": "Price below Short, above Mid",
            "MSL": "Price below Mid, above Long",
            "MLS": "Price below all EMAs (bull order)",
            "LSM": "Transitional state",
            "LMS": "Bearish - Long > Mid > Short",
        },
        "triggers": [
            {"base": "cross_bull", "name": "Short > Mid Cross", "direction": "LONG", "type": "ENTRY", "execution": "bar_close"},
            {"base": "cross_bear", "name": "Short < Mid Cross", "direction": "SHORT", "type": "ENTRY", "execution": "bar_close"},
            {"base": "mid_cross_bull", "name": "Mid > Long Cross", "direction": "LONG", "type": "ENTRY", "execution": "bar_close"},
            {"base": "mid_cross_bear", "name": "Mid < Long Cross", "direction": "SHORT", "type": "ENTRY", "execution": "bar_close"},
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
            {"base": "cross_bull", "name": "Bullish Cross", "direction": "LONG", "type": "ENTRY", "execution": "bar_close"},
            {"base": "cross_bear", "name": "Bearish Cross", "direction": "SHORT", "type": "ENTRY", "execution": "bar_close"},
            {"base": "zero_cross_up", "name": "Zero Line Cross Up", "direction": "LONG", "type": "ENTRY", "execution": "bar_close"},
            {"base": "zero_cross_down", "name": "Zero Line Cross Down", "direction": "SHORT", "type": "ENTRY", "execution": "bar_close"},
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
            {"base": "flip_pos", "name": "Histogram Flip Bullish", "direction": "LONG", "type": "ENTRY", "execution": "bar_close"},
            {"base": "flip_neg", "name": "Histogram Flip Bearish", "direction": "SHORT", "type": "ENTRY", "execution": "bar_close"},
            {"base": "momentum_shift_up", "name": "Momentum Shift Up", "direction": "LONG", "type": "ENTRY", "execution": "bar_close"},
            {"base": "momentum_shift_down", "name": "Momentum Shift Down", "direction": "SHORT", "type": "ENTRY", "execution": "bar_close"},
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
            {"base": "cross_above", "name": "Cross Above VWAP", "direction": "LONG", "type": "ENTRY", "execution": "bar_close"},
            {"base": "cross_below", "name": "Cross Below VWAP", "direction": "SHORT", "type": "ENTRY", "execution": "bar_close"},
            {"base": "enter_upper_extreme", "name": "Enter Upper Extreme (>+2σ)", "direction": "SHORT", "type": "ENTRY", "execution": "bar_close"},
            {"base": "enter_lower_extreme", "name": "Enter Lower Extreme (<-2σ)", "direction": "LONG", "type": "ENTRY", "execution": "bar_close"},
            {"base": "return_to_vwap", "name": "Return to VWAP Zone", "direction": "BOTH", "type": "EXIT", "execution": "bar_close"},
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
            {"base": "spike", "name": "Volume Spike", "direction": "BOTH", "type": "ENTRY", "execution": "bar_close"},
            {"base": "extreme", "name": "Extreme Volume", "direction": "BOTH", "type": "ENTRY", "execution": "bar_close"},
            {"base": "fade", "name": "Volume Fade", "direction": "BOTH", "type": "EXIT", "execution": "bar_close"},
        ],
        "indicator_columns": ["vol_sma", "rvol"],
    },

    "utbot": {
        "name": "UT Bot",
        "category": "Trend",
        "description": "UT Bot trend-following alerts based on ATR trailing stop",
        "interpreters": ["UTBOT"],
        "trigger_prefix": "utbot",
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
            {"base": "buy", "name": "Buy Signal", "direction": "LONG", "type": "ENTRY", "execution": "bar_close"},
            {"base": "sell", "name": "Sell Signal", "direction": "SHORT", "type": "ENTRY", "execution": "bar_close"},
        ],
        "indicator_columns": ["utbot_stop", "utbot_direction"],
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
            {"base": "exit", "name": "Exit After N Candles", "direction": "BOTH", "type": "EXIT", "execution": "bar_close"},
        ],
        "indicator_columns": [],
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


def load_confluence_groups() -> List[ConfluenceGroup]:
    """
    Load confluence groups from the config file.

    Returns list of ConfluenceGroup objects.
    """
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

        return groups

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error loading confluence groups: {e}")
        return create_default_groups()


def save_confluence_groups(groups: List[ConfluenceGroup]) -> bool:
    """
    Save confluence groups to the config file.

    Returns True if successful.
    """
    config_path = get_config_path()

    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        data = {
            "version": "1.0",
            "groups": []
        }

        for group in groups:
            group_data = {
                "id": group.id,
                "base_template": group.base_template,
                "version": group.version,
                "description": group.description,
                "enabled": group.enabled,
                "is_default": group.is_default,
                "parameters": group.parameters,
                "plot_settings": {
                    "colors": group.plot_settings.colors,
                    "line_width": group.plot_settings.line_width,
                    "visible": group.plot_settings.visible,
                },
            }
            data["groups"].append(group_data)

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
            id="utbot_default",
            base_template="utbot",
            version="Default",
            description="UT Bot with ATR period 10, multiplier 1.0",
            enabled=True,
            is_default=True,
            parameters={
                "atr_period": 10,
                "atr_multiplier": 1.0,
            },
            plot_settings=PlotSettings(
                colors={
                    "buy_color": "#22c55e",
                    "sell_color": "#ef4444",
                    "trail_color": "#64748b",
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

    Returns list of TriggerDefinition with group-specific IDs and names.
    """
    template = TEMPLATES.get(group.base_template)
    if not template:
        return []

    triggers = []
    for trig_def in template.get("triggers", []):
        trigger = TriggerDefinition(
            id=group.get_trigger_id(trig_def["base"]),
            name=group.get_trigger_name(trig_def["base"], trig_def["name"]),
            base_trigger=trig_def["base"],
            direction=trig_def["direction"],
            trigger_type=trig_def["type"],
            execution=trig_def.get("execution", "bar_close"),
        )
        triggers.append(trigger)

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


def get_entry_triggers(direction: str, groups: Optional[List[ConfluenceGroup]] = None) -> Dict[str, str]:
    """
    Get entry triggers for a specific direction.

    Returns dict mapping trigger_id -> display_name
    """
    all_triggers = get_all_triggers(groups)

    result = {}
    for trig_id, trig_def in all_triggers.items():
        if trig_def.trigger_type == "ENTRY":
            if trig_def.direction == direction or trig_def.direction == "BOTH":
                result[trig_id] = trig_def.name

    return result


def get_exit_triggers(groups: Optional[List[ConfluenceGroup]] = None) -> Dict[str, str]:
    """
    Get all exit triggers.

    Returns dict mapping trigger_id -> display_name
    """
    all_triggers = get_all_triggers(groups)

    result = {}
    for trig_id, trig_def in all_triggers.items():
        if trig_def.trigger_type == "EXIT":
            result[trig_id] = trig_def.name

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
