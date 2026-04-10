"""
Time Exit Packs for RoR Trader
===============================

Time Exit Packs provide mechanical, time-based exit rules that force
position closure independent of price signals.  They are a new category
of optimizable variable — users can compare variations (e.g. "EOD 15min
vs EOD 30min vs no time exit") in the analysis card.

Conceptual distinction from General Packs:
  - General Packs *filter entries* (gate whether trades can open)
  - Time Exit Packs *force exits* (close existing positions)

Each pack follows the same template / version pattern used by
Risk Management Packs and General Packs.

Usage:
    from time_exit_packs import (
        load_time_exit_packs,
        save_time_exit_packs,
        get_enabled_time_exit_packs,
        check_time_exit,
    )
"""

import json
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from typing import Dict, List, Optional, Any
from pathlib import Path

import pandas as pd


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TimeExitPack:
    """
    A configured instance of a Time Exit Pack Template.

    Terminology mirrors other pack types:
    - Template: The base exit method (e.g., "eod_exit")
    - Version:  User's parameter configuration name (e.g., "15min")
    - Pack:     The full combination shown in UI (e.g., "End of Day (15min)")

    Key difference from Risk Management Packs: outputs an exit config
    checked on every bar close, not a price-level stop/target.
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

    def get_exit_config(self) -> dict:
        """Generate the time exit config dict from this pack's parameters.
        The config is stored on strategies and passed to check_time_exit()."""
        template = TEMPLATES.get(self.base_template)
        if not template:
            return {"method": "max_hold_bars", "max_bars": 4}
        builder = template.get("build_exit_config")
        if builder:
            return builder(self.parameters)
        return {"method": self.base_template}


# =============================================================================
# CONFIG BUILDERS
# =============================================================================

def _clamp(val, default, mn=None, mx=None):
    """Clamp a parameter value to schema range."""
    try:
        val = type(default)(val) if val is not None else default
    except (TypeError, ValueError):
        val = default
    if mn is not None:
        val = max(val, mn)
    if mx is not None:
        val = min(val, mx)
    return val


def _build_eod(params: dict) -> dict:
    mins = _clamp(params.get("minutes_before_close", 15), 15, 1, 120)
    return {"method": "eod_exit", "minutes_before_close": mins}


def _build_time_of_day(params: dict) -> dict:
    hour = _clamp(params.get("exit_hour", 15), 15, 0, 23)
    minute = _clamp(params.get("exit_minute", 50), 50, 0, 59)
    return {"method": "time_of_day_exit", "exit_hour": hour, "exit_minute": minute}


def _build_max_hold_bars(params: dict) -> dict:
    bars = _clamp(params.get("max_bars", 4), 4, 1, 500)
    return {"method": "max_hold_bars", "max_bars": bars}


def _build_session_exit(params: dict) -> dict:
    return {
        "method": "session_exit",
        "start_hour": _clamp(params.get("start_hour", 9), 9, 0, 23),
        "start_minute": _clamp(params.get("start_minute", 30), 30, 0, 59),
        "end_hour": _clamp(params.get("end_hour", 16), 16, 0, 23),
        "end_minute": _clamp(params.get("end_minute", 0), 0, 0, 59),
    }


# =============================================================================
# EXIT CHECKERS (called per bar by the engine)
# =============================================================================

# Market close times by session (hour, minute) in Eastern Time
_SESSION_CLOSE = {
    "RTH": (16, 0),
    "regular": (16, 0),
    "extended": (20, 0),
    "after_hours": (20, 0),
    "pre_market": (9, 30),
}


def _parse_bar_time(bar_time) -> Optional[datetime]:
    """Parse bar_time to a datetime in US/Eastern.

    Bar timestamps from the data pipeline are typically UTC.  All time-based
    exit checks compare against Eastern Time clocks (market close = 16:00 ET),
    so we convert here once rather than in each checker.
    """
    if bar_time is None:
        return None
    try:
        import pytz
        _ET = pytz.timezone('America/New_York')
    except ImportError:
        _ET = None

    ts = None
    if isinstance(bar_time, pd.Timestamp):
        ts = bar_time
    elif isinstance(bar_time, datetime):
        ts = pd.Timestamp(bar_time)
    elif isinstance(bar_time, str):
        try:
            ts = pd.Timestamp(bar_time)
        except Exception:
            return None
    else:
        return None

    # Convert to Eastern Time
    if _ET is not None:
        if ts.tzinfo is not None:
            ts = ts.tz_convert(_ET)
        else:
            # Naive timestamps assumed UTC (matches Polygon data convention)
            ts = ts.tz_localize('UTC').tz_convert(_ET)

    return ts.to_pydatetime()


def _check_eod(config: dict, bar_time, session: str) -> Optional[str]:
    """Check if current bar is within N minutes of market close."""
    dt = _parse_bar_time(bar_time)
    if dt is None:
        return None
    close_h, close_m = _SESSION_CLOSE.get(session, (16, 0))
    close_minutes = close_h * 60 + close_m
    bar_minutes = dt.hour * 60 + dt.minute
    minutes_before = config.get("minutes_before_close", 15)
    threshold = close_minutes - minutes_before
    if bar_minutes >= threshold:
        return "eod_exit"
    return None


def _check_time_of_day(config: dict, bar_time) -> Optional[str]:
    """Check if current bar is at or past the configured exit time."""
    dt = _parse_bar_time(bar_time)
    if dt is None:
        return None
    exit_h = config.get("exit_hour", 15)
    exit_m = config.get("exit_minute", 50)
    exit_minutes = exit_h * 60 + exit_m
    bar_minutes = dt.hour * 60 + dt.minute
    if bar_minutes >= exit_minutes:
        return "time_of_day_exit"
    return None


def _check_max_hold_bars(config: dict, bars_held: int) -> Optional[str]:
    """Check if position has been held for max_bars or more."""
    max_bars = config.get("max_bars", 4)
    if bars_held >= max_bars:
        return "max_hold_bars"
    return None


def _check_session_exit(config: dict, bar_time) -> Optional[str]:
    """Check if current bar is outside the configured session window."""
    dt = _parse_bar_time(bar_time)
    if dt is None:
        return None
    start_minutes = config.get("start_hour", 9) * 60 + config.get("start_minute", 30)
    end_minutes = config.get("end_hour", 16) * 60 + config.get("end_minute", 0)
    bar_minutes = dt.hour * 60 + dt.minute
    if bar_minutes < start_minutes or bar_minutes >= end_minutes:
        return "session_exit"
    return None


# Dispatch table for exit checkers
_EXIT_CHECKERS = {
    "eod_exit": _check_eod,
    "time_of_day_exit": _check_time_of_day,
    "max_hold_bars": _check_max_hold_bars,
    "session_exit": _check_session_exit,
}


def check_time_exit(config: dict, bar_time, bars_held: int,
                    session: str = 'RTH') -> Optional[str]:
    """Stateless check — given a time exit config, should we exit now?

    Args:
        config: Time exit config dict with 'method' key (from get_exit_config())
        bar_time: Bar timestamp (ISO string, datetime, or pd.Timestamp)
        bars_held: Number of bars the position has been held
        session: Trading session ('RTH', '24/7', etc.)

    Returns:
        Exit reason string (e.g., 'eod_exit') or None.
    """
    if not config:
        return None

    method = config.get("method")
    if not method:
        return None

    # Check 24/7 compatibility
    template = TEMPLATES.get(method)
    if template and session in ('24/7', '24_7'):
        if not template.get("applies_to_24_7", False):
            return None

    checker = _EXIT_CHECKERS.get(method)
    if not checker:
        return None

    # max_hold_bars only needs bars_held
    if method == "max_hold_bars":
        return checker(config, bars_held)

    # eod_exit needs session for close time
    if method == "eod_exit":
        return checker(config, bar_time, session)

    # Others just need bar_time
    return checker(config, bar_time)


# =============================================================================
# TEMPLATE DEFINITIONS
# =============================================================================

TEMPLATES: Dict[str, Dict] = {
    "eod_exit": {
        "name": "End of Day",
        "category": "Time",
        "description": "Force exit N minutes before market close — prevents overnight gap risk",
        "applies_to_24_7": False,
        "parameters_schema": {
            "minutes_before_close": {
                "type": "int", "default": 15, "min": 1, "max": 120,
                "label": "Minutes Before Close",
            },
        },
        "exit_reason": "eod_exit",
        "exit_summary": "EOD {minutes_before_close}min",
        "build_exit_config": _build_eod,
    },

    "time_of_day_exit": {
        "name": "Time of Day",
        "category": "Time",
        "description": "Force exit at or after a specific clock time",
        "applies_to_24_7": False,
        "parameters_schema": {
            "exit_hour": {
                "type": "int", "default": 15, "min": 0, "max": 23,
                "label": "Exit Hour",
            },
            "exit_minute": {
                "type": "int", "default": 50, "min": 0, "max": 59,
                "label": "Exit Minute",
            },
        },
        "exit_reason": "time_of_day_exit",
        "exit_summary": "Exit at {exit_hour}:{exit_minute:02d}",
        "build_exit_config": _build_time_of_day,
    },

    "max_hold_bars": {
        "name": "Max Hold Bars",
        "category": "Duration",
        "description": "Force exit after holding for a maximum number of bars",
        "applies_to_24_7": True,
        "parameters_schema": {
            "max_bars": {
                "type": "int", "default": 4, "min": 1, "max": 500,
                "label": "Max Bars",
            },
        },
        "exit_reason": "max_hold_bars",
        "exit_summary": "Max {max_bars} bars",
        "build_exit_config": _build_max_hold_bars,
    },

    "session_exit": {
        "name": "Session Window",
        "category": "Time",
        "description": "Force exit if bar falls outside a configured time window",
        "applies_to_24_7": False,
        "parameters_schema": {
            "start_hour": {
                "type": "int", "default": 9, "min": 0, "max": 23,
                "label": "Window Start Hour",
            },
            "start_minute": {
                "type": "int", "default": 30, "min": 0, "max": 59,
                "label": "Window Start Minute",
            },
            "end_hour": {
                "type": "int", "default": 16, "min": 0, "max": 23,
                "label": "Window End Hour",
            },
            "end_minute": {
                "type": "int", "default": 0, "min": 0, "max": 59,
                "label": "Window End Minute",
            },
        },
        "exit_reason": "session_exit",
        "exit_summary": "Window {start_hour}:{start_minute:02d}-{end_hour}:{end_minute:02d}",
        "build_exit_config": _build_session_exit,
    },
}


# =============================================================================
# FILE OPERATIONS
# =============================================================================

def get_config_path() -> Path:
    src_dir = Path(__file__).parent
    project_dir = src_dir.parent
    return project_dir / "config" / "time_exit_packs.json"


def _parse_pack_list(raw_packs: list) -> List[TimeExitPack]:
    """Convert list of dicts to TimeExitPack objects."""
    return [TimeExitPack(
        id=p["id"],
        base_template=p["base_template"],
        version=p.get("version", "Default"),
        description=p.get("description", ""),
        enabled=p.get("enabled", True),
        is_default=p.get("is_default", False),
        parameters=p.get("parameters", {}),
    ) for p in raw_packs]


def _serialize_pack_list(packs: List[TimeExitPack]) -> list:
    """Convert TimeExitPack objects to list of dicts for storage."""
    return [{
        "id": p.id,
        "base_template": p.base_template,
        "version": p.version,
        "description": p.description,
        "enabled": p.enabled,
        "is_default": p.is_default,
        "parameters": p.parameters,
    } for p in packs]


def load_time_exit_packs() -> List[TimeExitPack]:
    from db import USE_DB
    if USE_DB:
        from db import load_time_exit_packs_db
        raw = load_time_exit_packs_db()
        if not raw:
            packs = create_default_packs()
            try:
                save_time_exit_packs(packs)
            except Exception:
                pass  # Return in-memory defaults even if save fails (e.g. no JWT)
            return packs
        return _parse_pack_list(raw)

    config_path = get_config_path()

    if not config_path.exists():
        return create_default_packs()

    try:
        with open(config_path, 'r') as f:
            data = json.load(f)

        return _parse_pack_list(data.get("packs", []))

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error loading time exit packs: {e}")
        return create_default_packs()


def save_time_exit_packs(packs: List[TimeExitPack]) -> bool:
    from db import USE_DB
    if USE_DB:
        from db import save_time_exit_packs_db
        save_time_exit_packs_db(_serialize_pack_list(packs))
        return True

    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        data = {"version": "1.0", "packs": _serialize_pack_list(packs)}

        with open(config_path, 'w') as f:
            json.dump(data, f, indent=2)

        return True

    except Exception as e:
        print(f"Error saving time exit packs: {e}")
        return False


# =============================================================================
# DEFAULT PACKS
# =============================================================================

def create_default_packs() -> List[TimeExitPack]:
    defaults = [
        TimeExitPack(
            id="eod_15min",
            base_template="eod_exit",
            version="15min",
            description="Exit 15 minutes before market close (15:45 ET for RTH)",
            enabled=True,
            is_default=True,
            parameters={"minutes_before_close": 15},
        ),
        TimeExitPack(
            id="eod_30min",
            base_template="eod_exit",
            version="30min",
            description="Exit 30 minutes before market close (15:30 ET for RTH)",
            enabled=True,
            is_default=True,
            parameters={"minutes_before_close": 30},
        ),
        TimeExitPack(
            id="max_hold_4",
            base_template="max_hold_bars",
            version="4 Bars",
            description="Force exit after holding for 4 bars",
            enabled=True,
            is_default=True,
            parameters={"max_bars": 4},
        ),
        TimeExitPack(
            id="max_hold_10",
            base_template="max_hold_bars",
            version="10 Bars",
            description="Force exit after holding for 10 bars",
            enabled=True,
            is_default=True,
            parameters={"max_bars": 10},
        ),
    ]
    return defaults


# =============================================================================
# HELPERS
# =============================================================================

def get_enabled_time_exit_packs(packs: List[TimeExitPack]) -> List[TimeExitPack]:
    return [p for p in packs if p.enabled]


def get_pack_by_id(pack_id: str, packs: List[TimeExitPack]) -> Optional[TimeExitPack]:
    for p in packs:
        if p.id == pack_id:
            return p
    return None


def get_template(template_id: str) -> Optional[Dict]:
    return TEMPLATES.get(template_id)


def get_parameter_schema(template_id: str) -> Dict:
    template = TEMPLATES.get(template_id)
    return template.get("parameters_schema", {}) if template else {}


def validate_pack_id(pack_id: str, existing_packs: List[TimeExitPack]) -> bool:
    if not pack_id or not pack_id.replace("_", "").isalnum():
        return False
    return not any(p.id == pack_id for p in existing_packs)


def generate_unique_id(template_id: str, existing_packs: List[TimeExitPack]) -> str:
    base = template_id
    counter = 1
    candidate = f"{base}_custom"
    while any(p.id == candidate for p in existing_packs):
        counter += 1
        candidate = f"{base}_custom_{counter}"
    return candidate


def duplicate_pack(pack: TimeExitPack, new_id: str, new_version: str) -> TimeExitPack:
    return TimeExitPack(
        id=new_id,
        base_template=pack.base_template,
        version=new_version,
        description=f"Copy of {pack.name}",
        enabled=True,
        is_default=False,
        parameters=dict(pack.parameters),
    )


def validate_parameters(params: dict, template_id: str) -> dict:
    """Validate and clamp parameters against the template schema.
    Returns a sanitized copy of params."""
    schema = get_parameter_schema(template_id)
    if not schema:
        return dict(params)
    validated = {}
    for key, spec in schema.items():
        val = params.get(key, spec.get("default"))
        ptype = spec.get("type")
        if ptype == "int":
            try:
                val = int(val)
            except (TypeError, ValueError):
                val = spec.get("default", 0)
            if "min" in spec:
                val = max(val, spec["min"])
            if "max" in spec:
                val = min(val, spec["max"])
        elif ptype == "float":
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = spec.get("default", 0.0)
            if "min" in spec:
                val = max(val, spec["min"])
            if "max" in spec:
                val = min(val, spec["max"])
        validated[key] = val
    return validated


def format_parameters(params: dict, template_id: str) -> str:
    template = TEMPLATES.get(template_id)
    if not template:
        return str(params)

    schema = template.get("parameters_schema", {})
    parts = []
    for key, value in params.items():
        label = schema.get(key, {}).get("label", key)
        parts.append(f"{label}: {value}")
    return " | ".join(parts) if parts else "Default"


def format_exit_summary(pack: TimeExitPack) -> str:
    """Human-readable summary for API/UI display."""
    template = TEMPLATES.get(pack.base_template)
    if not template:
        return "Unknown"
    try:
        return template["exit_summary"].format(**pack.parameters)
    except (KeyError, ValueError):
        return str(pack.get_exit_config())
