"""Strategy Models admin router — per-model usage stats.

Backs the /admin/live-models and /admin/backtest-models frontend pages
(Phase B of plans/synchronous-tickling-yeti.md).  Reads the model
registry from `strategy_models.py` and enriches each entry with a count
of strategies currently selecting it (default-counted when the
strategy.config field is null).

Read-only and admin-scoped (relies on the existing logged-in guard —
RoR_Trader is solo SaaS today, no separate role gating).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_current_user
from strategy_models import (
    BACKTEST_MODELS, LIVE_MODELS,
    get_default_backtest_model, get_default_live_model,
    get_default_algo_model, get_model_status,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/strategy-models",
                   tags=["admin", "strategy-models"])

Kind = Literal['live', 'backtest']


def _registry(kind: Kind) -> Dict[str, Dict[str, Any]]:
    return LIVE_MODELS if kind == 'live' else BACKTEST_MODELS


def _config_field(kind: Kind) -> str:
    return 'live_model' if kind == 'live' else 'backtest_model'


def _default(kind: Kind) -> str:
    return get_default_live_model() if kind == 'live' else get_default_backtest_model()


def _strategies_grouped_by_model(kind: Kind) -> Dict[str, List[Dict[str, Any]]]:
    """Return {model_id: [{id, name}, ...]} for every active strategy.

    A strategy with config.<field> == null counts toward the default
    model (matches how engine + ModelsCard treat missing values today).
    """
    from db import get_admin_client
    c = get_admin_client()
    field = _config_field(kind)
    default_id = _default(kind)
    valid_ids = set(_registry(kind).keys())

    # We don't filter in SQL because the field lives in JSONB and
    # PostgREST's .filter() on JSONB is awkward across SDK versions.
    # Strategy count is small (tens to hundreds); pull and bucket
    # client-side.
    result = c.table('strategies').select('id,name,config').execute()
    rows = result.data or []

    grouped: Dict[str, List[Dict[str, Any]]] = {mid: [] for mid in valid_ids}
    for row in rows:
        cfg = row.get('config') or {}
        selected = cfg.get(field) or default_id
        if selected not in valid_ids:
            # Strategy has a stale ID (e.g. a removed M3 placeholder).
            # Count it under the default so usage totals match strategy
            # count.  The detail page also surfaces stale-id strategies
            # with a warning when present.
            selected = default_id
        grouped[selected].append({'id': row['id'], 'name': row.get('name', '')})
    return grouped


def _strategies_grouped_by_algo_model() -> Dict[str, List[Dict[str, Any]]]:
    """Like `_strategies_grouped_by_model` but buckets by the `algo_model`
    JSONB field. `algo_model` draws from the same BACKTEST_MODELS registry
    as `backtest_model` — this is the second consumer (the cron's
    live-accountability lane). A null field counts toward the algo default.
    """
    from db import get_admin_client
    c = get_admin_client()
    default_id = get_default_algo_model()
    valid_ids = set(BACKTEST_MODELS.keys())

    result = c.table('strategies').select('id,name,config').execute()
    rows = result.data or []

    grouped: Dict[str, List[Dict[str, Any]]] = {mid: [] for mid in valid_ids}
    for row in rows:
        cfg = row.get('config') or {}
        selected = cfg.get('algo_model') or default_id
        if selected not in valid_ids:
            selected = default_id
        grouped[selected].append({'id': row['id'], 'name': row.get('name', '')})
    return grouped


def _list_response(kind: Kind) -> Dict[str, Any]:
    grouped = _strategies_grouped_by_model(kind)
    registry = _registry(kind)
    # The backtest registry doubles as the algo_model registry — surface
    # the algo default + algo usage so the page can badge both roles.
    algo_grouped = _strategies_grouped_by_algo_model() if kind == 'backtest' else {}
    algo_default_id = get_default_algo_model() if kind == 'backtest' else None
    rows = []
    for model_id, meta in registry.items():
        users = grouped.get(model_id, [])
        row = {
            'id': model_id,
            'label': meta.get('label', model_id),
            'available': bool(meta.get('available')),
            'default': bool(meta.get('default')),
            'status': get_model_status(model_id, kind),
            'description': meta.get('description', ''),
            'strategies_using': len(users),
            'sample_strategy_ids': [u['id'] for u in users[:5]],
        }
        if kind == 'backtest':
            algo_users = algo_grouped.get(model_id, [])
            row['algo_default'] = bool(meta.get('algo_default'))
            row['algo_strategies_using'] = len(algo_users)
        rows.append(row)
    return {
        'kind': kind,
        'default_id': _default(kind),
        'algo_default_id': algo_default_id,
        'rows': rows,
    }


def _detail_response(kind: Kind, model_id: str) -> Dict[str, Any]:
    registry = _registry(kind)
    meta = registry.get(model_id)
    if meta is None:
        raise HTTPException(status_code=404,
                            detail=f"Unknown {kind} model: {model_id}")
    grouped = _strategies_grouped_by_model(kind)
    users = grouped.get(model_id, [])
    out = {
        'id': model_id,
        'kind': kind,
        'label': meta.get('label', model_id),
        'available': bool(meta.get('available')),
        'default': bool(meta.get('default')),
        'status': get_model_status(model_id, kind),
        'description': meta.get('description', ''),
        'strategies': sorted(users, key=lambda s: s['id']),
        'strategies_using': len(users),
    }
    if kind == 'backtest':
        algo_users = _strategies_grouped_by_algo_model().get(model_id, [])
        out['algo_default'] = bool(meta.get('algo_default'))
        out['algo_strategies'] = sorted(algo_users, key=lambda s: s['id'])
        out['algo_strategies_using'] = len(algo_users)
    return out


@router.get("/live")
def list_live_models(user=Depends(get_current_user)):
    """List all live_model registry entries with usage counts."""
    return _list_response('live')


@router.get("/backtest")
def list_backtest_models(user=Depends(get_current_user)):
    """List all backtest_model registry entries with usage counts."""
    return _list_response('backtest')


@router.get("/live/{model_id}")
def get_live_model(model_id: str, user=Depends(get_current_user)):
    """Detail for a single live_model — full description + strategies using it."""
    return _detail_response('live', model_id)


@router.get("/backtest/{model_id}")
def get_backtest_model(model_id: str, user=Depends(get_current_user)):
    """Detail for a single backtest_model — full description + strategies using it."""
    return _detail_response('backtest', model_id)
