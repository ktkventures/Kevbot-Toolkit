"""
Risk Management Packs for RoR Trader
=====================================

Each Risk Management Pack bundles stop-loss AND take-profit configurations
from a shared set of parameters — analogous to how TF Confluence Packs
output both triggers AND conditions from the same indicator.

For example, an "ATR-Based" pack has a single ATR period but separate
multipliers for the stop and the target.

Usage:
    from risk_management_packs import (
        load_risk_management_packs,
        save_risk_management_packs,
        get_enabled_risk_management_packs,
    )
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RiskManagementPack:
    """
    A configured instance of a Risk Management Pack Template.

    Terminology mirrors TF Confluence Packs:
    - Template: The base risk method (e.g., "atr_based")
    - Version:  User's parameter configuration name (e.g., "Tight")
    - Pack:     The full combination shown in UI (e.g., "ATR-Based (Tight)")

    Key difference: outputs are stop_config and target_config dicts
    rather than triggers/conditions.
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

    def get_stop_config(self) -> dict:
        """Generate the stop-loss config dict from this pack's parameters."""
        template = TEMPLATES.get(self.base_template)
        if not template:
            return {"method": "atr", "atr_mult": 1.5}
        builder = template.get("build_stop")
        if builder:
            return builder(self.parameters)
        return {"method": "atr", "atr_mult": 1.5}

    def get_target_config(self) -> Optional[dict]:
        """Generate the take-profit config dict from this pack's parameters.
        Returns None if the pack doesn't define a target."""
        template = TEMPLATES.get(self.base_template)
        if not template:
            return None
        builder = template.get("build_target")
        if builder:
            return builder(self.parameters)
        return None


# =============================================================================
# CONFIG BUILDERS
# =============================================================================

def _atr_stop(params: dict) -> dict:
    return {"method": "atr", "atr_mult": params.get("stop_atr_mult", 1.5)}

def _atr_target(params: dict) -> Optional[dict]:
    mult = params.get("target_atr_mult")
    if mult is None or mult <= 0:
        return None
    return {"method": "atr", "atr_mult": mult}

def _fixed_stop(params: dict) -> dict:
    return {"method": "fixed_dollar", "dollar_amount": params.get("stop_amount", 1.0)}

def _fixed_target(params: dict) -> Optional[dict]:
    amt = params.get("target_amount")
    if amt is None or amt <= 0:
        return None
    return {"method": "fixed_dollar", "dollar_amount": amt}

def _pct_stop(params: dict) -> dict:
    return {"method": "percentage", "percentage": params.get("stop_pct", 0.5)}

def _pct_target(params: dict) -> Optional[dict]:
    pct = params.get("target_pct")
    if pct is None or pct <= 0:
        return None
    return {"method": "percentage", "percentage": pct}

def _swing_stop(params: dict) -> dict:
    return {
        "method": "swing",
        "lookback": params.get("lookback", 5),
        "padding": params.get("padding", 0.05),
    }

def _swing_target(params: dict) -> Optional[dict]:
    rr = params.get("rr_ratio")
    if rr is None or rr <= 0:
        return None
    return {"method": "risk_reward", "rr_ratio": rr}

def _rr_stop(params: dict) -> dict:
    method = params.get("stop_method", "atr")
    if method == "atr":
        return {"method": "atr", "atr_mult": params.get("stop_atr_mult", 1.5)}
    elif method == "fixed_dollar":
        return {"method": "fixed_dollar", "dollar_amount": params.get("stop_amount", 1.0)}
    elif method == "percentage":
        return {"method": "percentage", "percentage": params.get("stop_pct", 0.5)}
    elif method == "swing":
        return {"method": "swing", "lookback": params.get("lookback", 5), "padding": params.get("padding", 0.05)}
    return {"method": "atr", "atr_mult": 1.5}

def _rr_target(params: dict) -> Optional[dict]:
    rr = params.get("rr_ratio")
    if rr is None or rr <= 0:
        return None
    return {"method": "risk_reward", "rr_ratio": rr}


# =============================================================================
# TEMPLATE DEFINITIONS
# =============================================================================

TEMPLATES: Dict[str, Dict] = {
    "atr_based": {
        "name": "ATR-Based",
        "category": "Volatility",
        "description": "ATR-derived stop and target with independent multipliers",
        "parameters_schema": {
            "stop_atr_mult": {"type": "float", "default": 1.5, "min": 0.5, "max": 5.0, "label": "Stop ATR Mult"},
            "target_atr_mult": {"type": "float", "default": 3.0, "min": 0.0, "max": 10.0, "label": "Target ATR Mult"},
        },
        "stop_summary": "ATR x {stop_atr_mult}",
        "target_summary": "ATR x {target_atr_mult}",
        "build_stop": _atr_stop,
        "build_target": _atr_target,
    },

    "fixed_dollar": {
        "name": "Fixed Dollar",
        "category": "Fixed",
        "description": "Fixed dollar amounts for stop and target",
        "parameters_schema": {
            "stop_amount": {"type": "float", "default": 1.0, "min": 0.01, "max": 100.0, "label": "Stop ($)"},
            "target_amount": {"type": "float", "default": 2.0, "min": 0.0, "max": 100.0, "label": "Target ($)"},
        },
        "stop_summary": "${stop_amount}",
        "target_summary": "${target_amount}",
        "build_stop": _fixed_stop,
        "build_target": _fixed_target,
    },

    "percentage": {
        "name": "Percentage",
        "category": "Fixed",
        "description": "Percentage-based stop and target relative to entry price",
        "parameters_schema": {
            "stop_pct": {"type": "float", "default": 0.5, "min": 0.01, "max": 10.0, "label": "Stop (%)"},
            "target_pct": {"type": "float", "default": 1.0, "min": 0.0, "max": 20.0, "label": "Target (%)"},
        },
        "stop_summary": "{stop_pct}%",
        "target_summary": "{target_pct}%",
        "build_stop": _pct_stop,
        "build_target": _pct_target,
    },

    "swing": {
        "name": "Swing",
        "category": "Structure",
        "description": "Swing-based stop with risk:reward target",
        "parameters_schema": {
            "lookback": {"type": "int", "default": 5, "min": 2, "max": 50, "label": "Lookback"},
            "padding": {"type": "float", "default": 0.05, "min": 0.0, "max": 10.0, "label": "Padding ($)"},
            "rr_ratio": {"type": "float", "default": 2.0, "min": 0.0, "max": 10.0, "label": "R:R Ratio"},
        },
        "stop_summary": "Swing ({lookback} bars, ${padding} pad)",
        "target_summary": "{rr_ratio}R",
        "build_stop": _swing_stop,
        "build_target": _swing_target,
    },

    "rr_ratio": {
        "name": "Risk:Reward",
        "category": "Composite",
        "description": "Any stop method paired with a fixed risk:reward target",
        "parameters_schema": {
            "stop_method": {
                "type": "select",
                "default": "atr",
                "options": ["atr", "fixed_dollar", "percentage", "swing"],
                "label": "Stop Method",
            },
            "stop_atr_mult": {"type": "float", "default": 1.5, "min": 0.5, "max": 5.0, "label": "Stop ATR Mult"},
            "stop_amount": {"type": "float", "default": 1.0, "min": 0.01, "max": 100.0, "label": "Stop ($)"},
            "stop_pct": {"type": "float", "default": 0.5, "min": 0.01, "max": 10.0, "label": "Stop (%)"},
            "lookback": {"type": "int", "default": 5, "min": 2, "max": 50, "label": "Lookback"},
            "padding": {"type": "float", "default": 0.05, "min": 0.0, "max": 10.0, "label": "Padding ($)"},
            "rr_ratio": {"type": "float", "default": 2.0, "min": 0.5, "max": 10.0, "label": "R:R Ratio"},
        },
        "stop_summary": "{stop_method} stop",
        "target_summary": "{rr_ratio}R",
        "build_stop": _rr_stop,
        "build_target": _rr_target,
    },
}


# =============================================================================
# FILE OPERATIONS
# =============================================================================

def get_config_path() -> Path:
    src_dir = Path(__file__).parent
    project_dir = src_dir.parent
    return project_dir / "config" / "risk_management_packs.json"


def load_risk_management_packs() -> List[RiskManagementPack]:
    config_path = get_config_path()

    if not config_path.exists():
        return create_default_packs()

    try:
        with open(config_path, 'r') as f:
            data = json.load(f)

        packs = []
        for pack_data in data.get("packs", []):
            pack = RiskManagementPack(
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
        print(f"Error loading risk management packs: {e}")
        return create_default_packs()


def save_risk_management_packs(packs: List[RiskManagementPack]) -> bool:
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
        print(f"Error saving risk management packs: {e}")
        return False


# =============================================================================
# DEFAULT PACKS
# =============================================================================

def create_default_packs() -> List[RiskManagementPack]:
    defaults = [
        RiskManagementPack(
            id="atr_default",
            base_template="atr_based",
            version="Default",
            description="1.5x ATR stop, 3x ATR target",
            enabled=True,
            is_default=True,
            parameters={"stop_atr_mult": 1.5, "target_atr_mult": 3.0},
        ),
        RiskManagementPack(
            id="atr_tight",
            base_template="atr_based",
            version="Tight",
            description="1x ATR stop, 2x ATR target",
            enabled=True,
            is_default=True,
            parameters={"stop_atr_mult": 1.0, "target_atr_mult": 2.0},
        ),
        RiskManagementPack(
            id="fixed_1_2",
            base_template="fixed_dollar",
            version="$1 / $2",
            description="$1 stop, $2 target",
            enabled=True,
            is_default=True,
            parameters={"stop_amount": 1.0, "target_amount": 2.0},
        ),
        RiskManagementPack(
            id="pct_half_one",
            base_template="percentage",
            version="0.5% / 1%",
            description="0.5% stop, 1% target",
            enabled=False,
            is_default=True,
            parameters={"stop_pct": 0.5, "target_pct": 1.0},
        ),
        RiskManagementPack(
            id="swing_2r",
            base_template="swing",
            version="Swing 2R",
            description="Swing-based stop with 2:1 reward",
            enabled=True,
            is_default=True,
            parameters={"lookback": 5, "padding": 0.05, "rr_ratio": 2.0},
        ),
    ]
    return defaults


# =============================================================================
# HELPERS
# =============================================================================

def get_enabled_risk_management_packs(packs: List[RiskManagementPack]) -> List[RiskManagementPack]:
    return [p for p in packs if p.enabled]


def get_pack_by_id(pack_id: str, packs: List[RiskManagementPack]) -> Optional[RiskManagementPack]:
    for p in packs:
        if p.id == pack_id:
            return p
    return None


def get_template(template_id: str) -> Optional[Dict]:
    return TEMPLATES.get(template_id)


def get_parameter_schema(template_id: str) -> Dict:
    template = TEMPLATES.get(template_id)
    return template.get("parameters_schema", {}) if template else {}


def validate_pack_id(pack_id: str, existing_packs: List[RiskManagementPack]) -> bool:
    if not pack_id or not pack_id.replace("_", "").isalnum():
        return False
    return not any(p.id == pack_id for p in existing_packs)


def generate_unique_id(template_id: str, existing_packs: List[RiskManagementPack]) -> str:
    base = template_id
    counter = 1
    candidate = f"{base}_custom"
    while any(p.id == candidate for p in existing_packs):
        counter += 1
        candidate = f"{base}_custom_{counter}"
    return candidate


def duplicate_pack(pack: RiskManagementPack, new_id: str, new_version: str) -> RiskManagementPack:
    return RiskManagementPack(
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
        s = schema.get(key, {})
        label = s.get("label", key)
        # For composite packs, only show params relevant to selected stop method
        if template_id == "rr_ratio":
            stop_method = params.get("stop_method", "atr")
            if key == "stop_atr_mult" and stop_method != "atr":
                continue
            if key == "stop_amount" and stop_method != "fixed_dollar":
                continue
            if key == "stop_pct" and stop_method != "percentage":
                continue
            if key in ("lookback", "padding") and stop_method != "swing":
                continue
        parts.append(f"{label}: {value}")
    return " | ".join(parts) if parts else "Default"


def format_stop_summary(pack: RiskManagementPack) -> str:
    template = TEMPLATES.get(pack.base_template)
    if not template:
        return "Unknown"
    try:
        return template["stop_summary"].format(**pack.parameters)
    except (KeyError, ValueError):
        return str(pack.get_stop_config())


def format_target_summary(pack: RiskManagementPack) -> str:
    template = TEMPLATES.get(pack.base_template)
    if not template:
        return "None"
    target = pack.get_target_config()
    if target is None:
        return "None"
    try:
        return template["target_summary"].format(**pack.parameters)
    except (KeyError, ValueError):
        return str(target)
