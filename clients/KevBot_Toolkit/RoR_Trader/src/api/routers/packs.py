"""Packs router — confluence groups, general packs, risk management packs.

All three pack types follow the same pattern:
  - GET returns the full list per user (serialized from dataclasses)
  - PUT receives a full list of dicts and replaces the stored data
  - GET /templates returns the static TEMPLATES registry
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, Body
from dataclasses import asdict

from api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/packs", tags=["packs"])


# =============================================================================
# CONFLUENCE GROUPS (TF Packs)
# =============================================================================

@router.get("/confluence-groups")
def get_confluence_groups(user=Depends(get_current_user)):
    """Load user's confluence groups (full list)."""
    from confluence_groups import load_confluence_groups, _serialize_group_list
    groups = load_confluence_groups()
    return _serialize_group_list(groups)


@router.put("/confluence-groups")
def save_confluence_groups_endpoint(
    groups: list = Body(...),
    user=Depends(get_current_user),
):
    """Save full list of confluence groups (replaces existing)."""
    from confluence_groups import (
        save_confluence_groups, _parse_group_list,
    )
    parsed = _parse_group_list(groups)
    save_confluence_groups(parsed)
    return {"status": "saved", "count": len(parsed)}


@router.get("/confluence-groups/templates")
def get_confluence_templates(user=Depends(get_current_user)):
    """Get the static TEMPLATES registry for confluence groups."""
    from confluence_groups import TEMPLATES
    # TEMPLATES is a dict of dicts — JSON-serializable as-is
    return TEMPLATES


@router.get("/confluence-groups/triggers/{direction}")
def get_confluence_triggers(direction: str, user=Depends(get_current_user)):
    """Get entry triggers for a direction (LONG/SHORT) or all triggers.

    Only returns triggers from ENABLED confluence packs.
    """
    from confluence_groups import (
        get_enabled_groups, get_entry_triggers, get_exit_triggers,
    )
    groups = get_enabled_groups()
    if direction.upper() in ("LONG", "SHORT"):
        return get_entry_triggers(direction.upper(), groups)
    elif direction.upper() == "EXIT":
        return get_exit_triggers(groups)
    else:
        from confluence_groups import get_all_triggers
        triggers = get_all_triggers(groups)
        return {k: asdict(v) for k, v in triggers.items()}


# =============================================================================
# GENERAL PACKS
# =============================================================================

@router.get("/general")
def get_general_packs(user=Depends(get_current_user)):
    """Load user's general packs (full list)."""
    from general_packs import load_general_packs
    packs = load_general_packs()
    return [_general_pack_to_dict(p) for p in packs]


@router.put("/general")
def save_general_packs_endpoint(
    packs: list = Body(...),
    user=Depends(get_current_user),
):
    """Save full list of general packs (replaces existing)."""
    from general_packs import save_general_packs, GeneralPack
    parsed = [_dict_to_general_pack(p) for p in packs]
    save_general_packs(parsed)
    return {"status": "saved", "count": len(parsed)}


@router.get("/general/templates")
def get_general_templates(user=Depends(get_current_user)):
    """Get the static TEMPLATES registry for general packs."""
    from general_packs import TEMPLATES
    # Filter out non-serializable callables
    result = {}
    for k, v in TEMPLATES.items():
        entry = {key: val for key, val in v.items() if key != "evaluator"}
        result[k] = entry
    return result


# =============================================================================
# RISK MANAGEMENT PACKS
# =============================================================================

@router.get("/risk-management")
def get_risk_management_packs(user=Depends(get_current_user)):
    """Load user's risk management packs (full list)."""
    from risk_management_packs import load_risk_management_packs
    packs = load_risk_management_packs()
    return [_rm_pack_to_dict(p) for p in packs]


@router.put("/risk-management")
def save_risk_management_packs_endpoint(
    packs: list = Body(...),
    user=Depends(get_current_user),
):
    """Save full list of risk management packs (replaces existing)."""
    from risk_management_packs import save_risk_management_packs, RiskManagementPack
    parsed = [_dict_to_rm_pack(p) for p in packs]
    save_risk_management_packs(parsed)
    return {"status": "saved", "count": len(parsed)}


@router.get("/risk-management/templates")
def get_rm_templates(user=Depends(get_current_user)):
    """Get the static TEMPLATES registry for risk management packs."""
    from risk_management_packs import TEMPLATES
    # Filter out non-serializable callables (build_stop, build_target)
    result = {}
    for k, v in TEMPLATES.items():
        entry = {key: val for key, val in v.items()
                 if not callable(val)}
        result[k] = entry
    return result


# =============================================================================
# STOP LOSS PACKS (filtered view of RM packs)
# =============================================================================

@router.get("/stop-loss")
def get_stop_loss_packs(user=Depends(get_current_user)):
    """Load stop-loss capable packs (filtered from RM packs)."""
    from risk_management_packs import load_risk_management_packs, TEMPLATES
    packs = load_risk_management_packs()
    stop_packs = [p for p in packs if TEMPLATES.get(p.base_template, {}).get("has_stop", True)]
    return [_rm_pack_to_dict(p, include_configs=True) for p in stop_packs]


@router.get("/stop-loss/templates")
def get_stop_loss_templates(user=Depends(get_current_user)):
    """Get stop-loss template definitions."""
    from risk_management_packs import TEMPLATES
    result = {}
    for k, v in TEMPLATES.items():
        if v.get("has_stop", True):
            entry = {key: val for key, val in v.items() if not callable(val)}
            result[k] = entry
    return result


@router.put("/stop-loss")
def save_stop_loss_packs(packs: list = Body(...), user=Depends(get_current_user)):
    """Save stop-loss packs (writes through RM pack pipeline)."""
    from risk_management_packs import save_risk_management_packs, load_risk_management_packs, TEMPLATES
    # Merge: keep existing non-stop packs, replace stop packs
    existing = load_risk_management_packs()
    stop_keys = {k for k, v in TEMPLATES.items() if v.get("has_stop", True)}
    non_stop = [p for p in existing if p.base_template not in stop_keys]
    new_stop = [_dict_to_rm_pack(p) for p in packs]
    save_risk_management_packs(non_stop + new_stop)
    return {"status": "saved", "count": len(new_stop)}


# =============================================================================
# TAKE PROFIT PACKS (filtered view of RM packs)
# =============================================================================

@router.get("/take-profit")
def get_take_profit_packs(user=Depends(get_current_user)):
    """Load take-profit capable packs (filtered from RM packs)."""
    from risk_management_packs import load_risk_management_packs, TEMPLATES
    packs = load_risk_management_packs()
    tp_packs = [p for p in packs if TEMPLATES.get(p.base_template, {}).get("has_target", True)]
    return [_rm_pack_to_dict(p, include_configs=True) for p in tp_packs]


@router.get("/take-profit/templates")
def get_take_profit_templates(user=Depends(get_current_user)):
    """Get take-profit template definitions."""
    from risk_management_packs import TEMPLATES
    result = {}
    for k, v in TEMPLATES.items():
        if v.get("has_target", True):
            entry = {key: val for key, val in v.items() if not callable(val)}
            result[k] = entry
    return result


@router.put("/take-profit")
def save_take_profit_packs(packs: list = Body(...), user=Depends(get_current_user)):
    """Save take-profit packs (writes through RM pack pipeline)."""
    from risk_management_packs import save_risk_management_packs, load_risk_management_packs, TEMPLATES
    existing = load_risk_management_packs()
    tp_keys = {k for k, v in TEMPLATES.items() if v.get("has_target", True)}
    non_tp = [p for p in existing if p.base_template not in tp_keys]
    new_tp = [_dict_to_rm_pack(p) for p in packs]
    save_risk_management_packs(non_tp + new_tp)
    return {"status": "saved", "count": len(new_tp)}


# =============================================================================
# SERIALIZATION HELPERS
# =============================================================================

def _general_pack_to_dict(pack) -> dict:
    """Serialize a GeneralPack dataclass to dict."""
    return {
        "id": pack.id,
        "base_template": pack.base_template,
        "version": pack.version,
        "description": pack.description,
        "enabled": pack.enabled,
        "is_default": pack.is_default,
        "parameters": pack.parameters,
    }


def _dict_to_general_pack(d: dict):
    """Deserialize a dict to GeneralPack dataclass."""
    from general_packs import GeneralPack
    return GeneralPack(
        id=d["id"],
        base_template=d["base_template"],
        version=d.get("version", "Default"),
        description=d.get("description", ""),
        enabled=d.get("enabled", True),
        is_default=d.get("is_default", False),
        parameters=d.get("parameters", {}),
    )


def _rm_pack_to_dict(pack, include_configs: bool = False) -> dict:
    """Serialize a RiskManagementPack dataclass to dict."""
    d = {
        "id": pack.id,
        "base_template": pack.base_template,
        "version": pack.version,
        "description": pack.description,
        "enabled": pack.enabled,
        "is_default": pack.is_default,
        "parameters": pack.parameters,
    }
    if include_configs:
        try:
            d["stop_config"] = pack.get_stop_config()
        except Exception:
            d["stop_config"] = None
        try:
            d["target_config"] = pack.get_target_config()
        except Exception:
            d["target_config"] = None
    return d


def _dict_to_rm_pack(d: dict):
    """Deserialize a dict to RiskManagementPack dataclass."""
    from risk_management_packs import RiskManagementPack
    return RiskManagementPack(
        id=d["id"],
        base_template=d["base_template"],
        version=d.get("version", "Default"),
        description=d.get("description", ""),
        enabled=d.get("enabled", True),
        is_default=d.get("is_default", False),
        parameters=d.get("parameters", {}),
    )
