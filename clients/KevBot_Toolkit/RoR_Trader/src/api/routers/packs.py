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
    """Get the static TEMPLATES registry for confluence groups.

    Enriched with `output_directions`: map of {output_state → "BULL"|"BEAR"|"NEUTRAL"}
    derived from INTERPRETER_DIRECTION_MAP so the frontend can group states
    by direction for the Mass Builder confluence selector.
    """
    from confluence_groups import TEMPLATES, INTERPRETER_DIRECTION_MAP
    enriched = {}
    for key, tmpl in TEMPLATES.items():
        out = dict(tmpl)
        state_direction: dict[str, str] = {}
        for interp in tmpl.get('interpreters', []):
            for direction, states in INTERPRETER_DIRECTION_MAP.get(interp, {}).items():
                for state in states:
                    state_direction[state] = direction
        out['output_directions'] = state_direction
        enriched[key] = out
    return enriched


@router.get("/confluence-groups/triggers/{direction}")
def get_confluence_triggers(direction: str, user=Depends(get_current_user)):
    """Get all available triggers, filtered by enabled execution types.

    Triggers are direction-agnostic and type-agnostic — users decide how to
    use them in Strategy Builder. The `direction` path parameter is kept for
    backward compatibility but all paths return the same full trigger list.
    """
    from confluence_groups import get_enabled_groups, get_entry_triggers
    groups = get_enabled_groups()
    return get_entry_triggers(direction.upper(), groups)


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
# TIME EXIT PACKS
# =============================================================================

@router.get("/time-exit")
def get_time_exit_packs(user=Depends(get_current_user)):
    """Load user's time exit packs (full list)."""
    from time_exit_packs import load_time_exit_packs
    packs = load_time_exit_packs()
    return [_time_exit_pack_to_dict(p) for p in packs]


@router.put("/time-exit")
def save_time_exit_packs_endpoint(
    packs: list = Body(...),
    user=Depends(get_current_user),
):
    """Save full list of time exit packs (replaces existing)."""
    from time_exit_packs import save_time_exit_packs
    parsed = [_dict_to_time_exit_pack(p) for p in packs]
    save_time_exit_packs(parsed)
    return {"status": "saved", "count": len(parsed)}


@router.get("/time-exit/templates")
def get_time_exit_templates(user=Depends(get_current_user)):
    """Get the static TEMPLATES registry for time exit packs."""
    from time_exit_packs import TEMPLATES
    # Filter out non-serializable callables
    result = {}
    for k, v in TEMPLATES.items():
        entry = {key: val for key, val in v.items() if not callable(val)}
        result[k] = entry
    return result


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
    from risk_management_packs import TEMPLATES, format_stop_summary, format_target_summary
    template = TEMPLATES.get(pack.base_template, {})
    d = {
        "id": pack.id,
        "base_template": pack.base_template,
        "template_name": template.get("name", pack.base_template),
        "version": pack.version,
        "description": pack.description,
        "enabled": pack.enabled,
        "is_default": pack.is_default,
        "parameters": pack.parameters,
        "stop_summary": format_stop_summary(pack),
        "target_summary": format_target_summary(pack),
        "supported_exec_types": pack.get_supported_exec_types(),
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


def _time_exit_pack_to_dict(pack) -> dict:
    """Serialize a TimeExitPack dataclass to dict."""
    from time_exit_packs import format_exit_summary
    return {
        "id": pack.id,
        "base_template": pack.base_template,
        "template_name": pack.template_name,
        "version": pack.version,
        "description": pack.description,
        "enabled": pack.enabled,
        "is_default": pack.is_default,
        "parameters": pack.parameters,
        "exit_summary": format_exit_summary(pack),
        "exit_config": pack.get_exit_config(),
    }


def _dict_to_time_exit_pack(d: dict):
    """Deserialize a dict to TimeExitPack dataclass."""
    from time_exit_packs import TimeExitPack
    return TimeExitPack(
        id=d["id"],
        base_template=d["base_template"],
        version=d.get("version", "Default"),
        description=d.get("description", ""),
        enabled=d.get("enabled", True),
        is_default=d.get("is_default", False),
        parameters=d.get("parameters", {}),
    )


# =============================================================================
# PARITY SIMULATOR
# =============================================================================

@router.post("/parity-test")
def run_parity_test(
    body: dict = Body(...),
    user=Depends(get_current_user),
):
    """Run a backtest-vs-live parity test for a pack's entry trigger.

    Body: {
      "pack_id": str,
      "entry_trigger": str,
      "symbol": str = "SPY",
      "timeframe": str = "1Min",
      "days": int = 7,
      "session": str = "RTH",
      "feed": str = "sip",
      "warmup_bars": int = 200,
    }

    Returns the full parity result schema — see parity_simulator
    docstring for field descriptions.

    Long-running: a 7-day SPY/1Min run is ~2s; longer windows or
    sub-second feeds will scale roughly linearly with bar count.
    """
    pack_id = body.get("pack_id")
    entry_trigger = body.get("entry_trigger")
    if not pack_id or not entry_trigger:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail="pack_id and entry_trigger are required",
        )

    from parity_simulator import run_pack_parity_test
    return run_pack_parity_test(
        pack_id=pack_id,
        entry_trigger=entry_trigger,
        symbol=body.get("symbol", "SPY"),
        timeframe=body.get("timeframe", "1Min"),
        days=int(body.get("days", 7)),
        session=body.get("session", "RTH"),
        feed=body.get("feed", "sip"),
        warmup_bars=int(body.get("warmup_bars", 200)),
    )
