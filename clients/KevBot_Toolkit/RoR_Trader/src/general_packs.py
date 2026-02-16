"""
General Confluence Packs for RoR Trader
=======================================

General Packs provide non-indicator, strategy-wide conditions and optional
triggers.  Examples include time-of-day windows, trading-session filters,
day-of-week rules, and calendar / news-event filters.

Each pack follows the same template / version pattern used by TF Confluence
Packs (confluence_groups.py).  A pack can output:
  - Conditions (categorical states checked on every bar)
  - Triggers  (optional entry/exit signals, e.g. breaking-news alert)

Usage:
    from general_packs import (
        load_general_packs,
        save_general_packs,
        get_enabled_general_packs,
    )
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GeneralPack:
    """
    A configured instance of a General Pack Template.

    Terminology mirrors TF Confluence Packs:
    - Template: The base logic type (e.g., "time_of_day")
    - Version:  User's parameter configuration name (e.g., "NY Open")
    - Pack:     The full combination shown in UI (e.g., "Time of Day (NY Open)")
    """
    id: str
    base_template: str
    version: str
    description: str
    enabled: bool
    is_default: bool
    parameters: Dict[str, Any]

    @property
    def template_name(self) -> str:
        template = TEMPLATES.get(self.base_template)
        return template["name"] if template else self.base_template

    @property
    def name(self) -> str:
        return f"{self.template_name} ({self.version})"

    def get_condition_column(self) -> str:
        return f"GP_{self.id.upper()}"


# =============================================================================
# TEMPLATE DEFINITIONS
# =============================================================================

TEMPLATES: Dict[str, Dict] = {
    "time_of_day": {
        "name": "Time of Day",
        "category": "Time",
        "description": "Filter trades to a specific time-of-day window",
        "parameters_schema": {
            "start_hour": {"type": "int", "default": 9, "min": 0, "max": 23, "label": "Start Hour"},
            "start_minute": {"type": "int", "default": 30, "min": 0, "max": 59, "label": "Start Minute"},
            "end_hour": {"type": "int", "default": 12, "min": 0, "max": 23, "label": "End Hour"},
            "end_minute": {"type": "int", "default": 0, "min": 0, "max": 59, "label": "End Minute"},
        },
        "outputs": ["IN_WINDOW", "OUT_OF_WINDOW"],
        "output_descriptions": {
            "IN_WINDOW": "Current time is within the configured window",
            "OUT_OF_WINDOW": "Current time is outside the configured window",
        },
        "triggers": [],
        "condition_logic": "time_window",
    },

    "trading_session": {
        "name": "Trading Session",
        "category": "Time",
        "description": "Filter trades by market session (pre-market, regular, after-hours)",
        "parameters_schema": {
            "session": {
                "type": "select",
                "default": "regular",
                "options": ["pre_market", "regular", "after_hours", "extended"],
                "label": "Session",
            },
        },
        "outputs": ["IN_SESSION", "OUT_OF_SESSION"],
        "output_descriptions": {
            "IN_SESSION": "Current time is within the selected session",
            "OUT_OF_SESSION": "Current time is outside the selected session",
        },
        "triggers": [],
        "condition_logic": "session_filter",
    },

    "day_of_week": {
        "name": "Day of Week",
        "category": "Calendar",
        "description": "Allow or block trading on specific days of the week",
        "parameters_schema": {
            "monday": {"type": "bool", "default": True, "label": "Monday"},
            "tuesday": {"type": "bool", "default": True, "label": "Tuesday"},
            "wednesday": {"type": "bool", "default": True, "label": "Wednesday"},
            "thursday": {"type": "bool", "default": True, "label": "Thursday"},
            "friday": {"type": "bool", "default": True, "label": "Friday"},
        },
        "outputs": ["ALLOWED_DAY", "BLOCKED_DAY"],
        "output_descriptions": {
            "ALLOWED_DAY": "Today is an allowed trading day",
            "BLOCKED_DAY": "Today is a blocked trading day",
        },
        "triggers": [],
        "condition_logic": "day_filter",
    },

    "calendar_filter": {
        "name": "Calendar Filter",
        "category": "Calendar",
        "description": "Block trading around high-impact economic events",
        "parameters_schema": {
            "avoid_fomc": {"type": "bool", "default": True, "label": "Avoid FOMC Days"},
            "avoid_opex": {"type": "bool", "default": False, "label": "Avoid OpEx Days"},
            "avoid_nfp": {"type": "bool", "default": True, "label": "Avoid NFP Days"},
            "buffer_minutes": {"type": "int", "default": 30, "min": 0, "max": 120, "label": "Buffer (minutes)"},
        },
        "outputs": ["CLEAR", "BLOCKED"],
        "output_descriptions": {
            "CLEAR": "No high-impact events — trading allowed",
            "BLOCKED": "High-impact event window — trading blocked",
        },
        "triggers": [],
        "condition_logic": "calendar_filter",
    },
}


# =============================================================================
# FILE OPERATIONS
# =============================================================================

def get_config_path() -> Path:
    src_dir = Path(__file__).parent
    project_dir = src_dir.parent
    return project_dir / "config" / "general_packs.json"


def load_general_packs() -> List[GeneralPack]:
    config_path = get_config_path()

    if not config_path.exists():
        return create_default_packs()

    try:
        with open(config_path, 'r') as f:
            data = json.load(f)

        packs = []
        for pack_data in data.get("packs", []):
            pack = GeneralPack(
                id=pack_data["id"],
                base_template=pack_data["base_template"],
                version=pack_data.get("version", "Default"),
                description=pack_data.get("description", ""),
                enabled=pack_data.get("enabled", True),
                is_default=pack_data.get("is_default", False),
                parameters=pack_data.get("parameters", {}),
            )
            packs.append(pack)

        return packs

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error loading general packs: {e}")
        return create_default_packs()


def save_general_packs(packs: List[GeneralPack]) -> bool:
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        data = {
            "version": "1.0",
            "packs": []
        }

        for pack in packs:
            pack_data = {
                "id": pack.id,
                "base_template": pack.base_template,
                "version": pack.version,
                "description": pack.description,
                "enabled": pack.enabled,
                "is_default": pack.is_default,
                "parameters": pack.parameters,
            }
            data["packs"].append(pack_data)

        with open(config_path, 'w') as f:
            json.dump(data, f, indent=2)

        return True

    except Exception as e:
        print(f"Error saving general packs: {e}")
        return False


# =============================================================================
# DEFAULT PACKS
# =============================================================================

def create_default_packs() -> List[GeneralPack]:
    defaults = [
        GeneralPack(
            id="tod_ny_open",
            base_template="time_of_day",
            version="NY Open",
            description="First 2.5 hours of regular trading (9:30 - 12:00 ET)",
            enabled=True,
            is_default=True,
            parameters={
                "start_hour": 9, "start_minute": 30,
                "end_hour": 12, "end_minute": 0,
            },
        ),
        GeneralPack(
            id="session_regular",
            base_template="trading_session",
            version="Regular Hours",
            description="Regular trading session only (9:30 - 16:00 ET)",
            enabled=True,
            is_default=True,
            parameters={"session": "regular"},
        ),
        GeneralPack(
            id="dow_weekdays",
            base_template="day_of_week",
            version="All Weekdays",
            description="Trade all weekdays (default)",
            enabled=True,
            is_default=True,
            parameters={
                "monday": True, "tuesday": True, "wednesday": True,
                "thursday": True, "friday": True,
            },
        ),
        GeneralPack(
            id="cal_avoid_fomc_nfp",
            base_template="calendar_filter",
            version="Avoid FOMC & NFP",
            description="Skip trading on FOMC and NFP days",
            enabled=True,
            is_default=True,
            parameters={
                "avoid_fomc": True, "avoid_opex": False,
                "avoid_nfp": True, "buffer_minutes": 30,
            },
        ),
    ]
    return defaults


# =============================================================================
# HELPERS
# =============================================================================

def get_enabled_general_packs(packs: List[GeneralPack]) -> List[GeneralPack]:
    return [p for p in packs if p.enabled]


def get_pack_by_id(pack_id: str, packs: List[GeneralPack]) -> Optional[GeneralPack]:
    for p in packs:
        if p.id == pack_id:
            return p
    return None


def get_template(template_id: str) -> Optional[Dict]:
    return TEMPLATES.get(template_id)


def get_parameter_schema(template_id: str) -> Dict:
    template = TEMPLATES.get(template_id)
    return template.get("parameters_schema", {}) if template else {}


def get_output_descriptions(template_id: str) -> Dict:
    template = TEMPLATES.get(template_id)
    return template.get("output_descriptions", {}) if template else {}


def validate_pack_id(pack_id: str, existing_packs: List[GeneralPack]) -> bool:
    if not pack_id or not pack_id.replace("_", "").isalnum():
        return False
    return not any(p.id == pack_id for p in existing_packs)


def generate_unique_id(template_id: str, existing_packs: List[GeneralPack]) -> str:
    base = template_id
    counter = 1
    candidate = f"{base}_custom"
    while any(p.id == candidate for p in existing_packs):
        counter += 1
        candidate = f"{base}_custom_{counter}"
    return candidate


def duplicate_pack(pack: GeneralPack, new_id: str, new_version: str) -> GeneralPack:
    return GeneralPack(
        id=new_id,
        base_template=pack.base_template,
        version=new_version,
        description=f"Copy of {pack.name}",
        enabled=True,
        is_default=False,
        parameters=dict(pack.parameters),
    )


def format_parameters(params: dict, template_id: str) -> str:
    template = TEMPLATES.get(template_id)
    if not template:
        return str(params)

    schema = template.get("parameters_schema", {})
    parts = []
    for key, value in params.items():
        label = schema.get(key, {}).get("label", key)
        short_label = label.replace(" Hour", "h").replace(" Minute", "m")
        if isinstance(value, bool):
            if value:
                parts.append(short_label)
        else:
            parts.append(f"{short_label}: {value}")
    return " | ".join(parts) if parts else "Default"


# =============================================================================
# CONDITION EVALUATION
# =============================================================================

SESSION_WINDOWS = {
    "pre_market": (4, 0, 9, 30),
    "regular": (9, 30, 16, 0),
    "after_hours": (16, 0, 20, 0),
    "extended": (4, 0, 20, 0),
}


def evaluate_condition(df: pd.DataFrame, pack: 'GeneralPack') -> pd.Series:
    """Evaluate a general pack's condition on each bar and return a Series of output labels."""
    logic = TEMPLATES.get(pack.base_template, {}).get("condition_logic")
    if logic == "time_window":
        return _eval_time_of_day(df, pack.parameters)
    elif logic == "session_filter":
        return _eval_trading_session(df, pack.parameters)
    elif logic == "day_filter":
        return _eval_day_of_week(df, pack.parameters)
    elif logic == "calendar_filter":
        return _eval_calendar_filter(df, pack.parameters)
    return pd.Series("UNKNOWN", index=df.index)


def _eval_time_of_day(df: pd.DataFrame, params: dict) -> pd.Series:
    """Check if each bar's time falls within [start_hour:start_minute, end_hour:end_minute)."""
    sh = params.get("start_hour", 9)
    sm = params.get("start_minute", 30)
    eh = params.get("end_hour", 12)
    em = params.get("end_minute", 0)
    start_minutes = sh * 60 + sm
    end_minutes = eh * 60 + em

    idx = df.index
    if hasattr(idx, 'get_level_values'):
        try:
            idx = idx.get_level_values(-1)
        except Exception:
            pass

    bar_minutes = idx.hour * 60 + idx.minute
    in_window = (bar_minutes >= start_minutes) & (bar_minutes < end_minutes)
    return pd.Series(
        ["IN_WINDOW" if v else "OUT_OF_WINDOW" for v in in_window],
        index=df.index,
    )


def _eval_trading_session(df: pd.DataFrame, params: dict) -> pd.Series:
    """Check if each bar falls within the selected trading session."""
    session = params.get("session", "regular")
    window = SESSION_WINDOWS.get(session, SESSION_WINDOWS["regular"])
    sh, sm, eh, em = window
    start_minutes = sh * 60 + sm
    end_minutes = eh * 60 + em

    idx = df.index
    if hasattr(idx, 'get_level_values'):
        try:
            idx = idx.get_level_values(-1)
        except Exception:
            pass

    bar_minutes = idx.hour * 60 + idx.minute
    in_session = (bar_minutes >= start_minutes) & (bar_minutes < end_minutes)
    return pd.Series(
        ["IN_SESSION" if v else "OUT_OF_SESSION" for v in in_session],
        index=df.index,
    )


def _eval_day_of_week(df: pd.DataFrame, params: dict) -> pd.Series:
    """Check if each bar's day of week is allowed."""
    day_map = {0: "monday", 1: "tuesday", 2: "wednesday", 3: "thursday", 4: "friday"}
    idx = df.index
    if hasattr(idx, 'get_level_values'):
        try:
            idx = idx.get_level_values(-1)
        except Exception:
            pass

    results = []
    for dt in idx:
        day_name = day_map.get(dt.dayofweek, "")
        allowed = params.get(day_name, True)
        results.append("ALLOWED_DAY" if allowed else "BLOCKED_DAY")
    return pd.Series(results, index=df.index)


def _eval_calendar_filter(df: pd.DataFrame, params: dict) -> pd.Series:
    """Simplified calendar filter — blocks known high-impact event days.
    In production this would check an event calendar API. For preview,
    we simulate by blocking the first Wednesday of each month (FOMC proxy)
    and the first Friday (NFP proxy)."""
    avoid_fomc = params.get("avoid_fomc", True)
    avoid_nfp = params.get("avoid_nfp", True)

    idx = df.index
    if hasattr(idx, 'get_level_values'):
        try:
            idx = idx.get_level_values(-1)
        except Exception:
            pass

    results = []
    for dt in idx:
        blocked = False
        if avoid_fomc and dt.dayofweek == 2 and dt.day <= 7:
            blocked = True
        if avoid_nfp and dt.dayofweek == 4 and dt.day <= 7:
            blocked = True
        results.append("BLOCKED" if blocked else "CLEAR")
    return pd.Series(results, index=df.index)


def get_state_transitions(condition_series: pd.Series) -> pd.DataFrame:
    """Extract rows where the condition state changes."""
    shifted = condition_series.shift(1)
    changed = condition_series != shifted
    transitions = condition_series[changed].reset_index()
    transitions.columns = ["time", "state"]
    return transitions
