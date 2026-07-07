"""Model Parity admin router (W2-6c, 2026-07-07).

Backs the Strategy Health "Model Parity" tab — one row per strategy:

1.  RUNTIME-EFFECTIVE model columns (backtest_model / algo_model /
    live_model). These MIRROR the actual engine resolution sites, NOT
    the admin registry defaults (`strategy_models_admin.py` buckets a
    null field under the registry default — that display behavior hid
    the cache_None mislabel for 3 weeks; ledger W2-6):

      - live_model     → ralph_engine.py:1014-1018 (StrategyMonitor.__init__):
                         declared-if-valid else get_default_live_model()
      - backtest_model → data_worker_engine.py:277-279 /
                         forward_test_service.py:415-419:
                         cfg-or-toplevel or the literal 'rest_hifi'
      - algo_model     → forward_test_service.py:1053-1065
                         (_require_algo_model, PR #26 tripwire):
                         algo_model → backtest_model, and ALL-UNSET is an
                         ERROR (the runtime raises rather than defaulting —
                         never mask it as the registry default here)

2.  Per-strategy parity columns computed the SAME way the Strategy
    Detail "Per-Bar Parity Drift · v2" module does (frontend
    ParityDriftRibbonV2.tsx): backtest (REST) lens vs live-cache lens
    ('first' = decision-time WS values, 'latest' = REST-corrected),
    last 300 common-window bars, identical tolerances, pct =
    match/comparable. The bar payloads are produced by the very same
    endpoint functions the detail page calls
    (/strategies/{id}/chart-data and /chart-data-cache) so the numbers
    reconcile with what the detail page shows.

Perf: the parity computation is heavy (two full indicator pipelines per
strategy). The /parity endpoint therefore:
  - accepts at most _MAX_SIDS_PER_REQUEST (8) strategy ids per call —
    the frontend walks the fleet in small batches;
  - caches results server-side keyed (strategy_id, lens) with a ~10 min
    TTL;
  - runs per-sid indexed queries only (each compute goes through the
    per-strategy detail endpoints — no fleet-wide LIKE scans).

Read-only and admin-scoped (existing logged-in guard — solo SaaS).
"""
from __future__ import annotations

import logging
import math
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/model-parity",
                   tags=["admin", "model-parity"])


# ─────────────────────────────────────────────────────────────────────────
# 1. Runtime-effective model resolution (mirrors of the engine sites)
# ─────────────────────────────────────────────────────────────────────────

def _resolve_live_model(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror of ralph_engine.py:1014-1018 (StrategyMonitor.__init__).

        declared = strategy.get('live_model') or strategy.get('config', {}).get('live_model')
        self.live_model = declared if (declared and is_valid_live_model(declared)) \
                          else get_default_live_model()

    We read the raw strategies.config JSONB, so `cfg` IS the source the
    engine's flattened dict was built from.
    """
    from strategy_models import get_default_live_model, is_valid_live_model
    declared = cfg.get('live_model')
    default = get_default_live_model()
    if declared and is_valid_live_model(declared):
        return {"declared": declared, "effective": declared, "source": "explicit"}
    if declared:  # set but stale/invalid — engine falls back to default
        return {"declared": declared, "effective": default,
                "source": "invalid_fallback"}
    return {"declared": None, "effective": default, "source": "default"}


def _resolve_backtest_model(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror of data_worker_engine.py:277-279 and
    forward_test_service.py:415-419:

        bt_model = cfg.get('backtest_model') or strat.get('backtest_model') or 'rest_hifi'

    Note the engines use the LITERAL 'rest_hifi', not the registry
    resolver — they currently agree, but we mirror the literal on
    purpose (the engine is the truth being displayed).
    """
    declared = cfg.get('backtest_model')
    if declared:
        return {"declared": declared, "effective": declared, "source": "explicit"}
    return {"declared": None, "effective": "rest_hifi", "source": "default"}


def _resolve_algo_model(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror of forward_test_service.py:1053-1065 (_require_algo_model,
    the PR #26 cache_None tripwire):

        model = cfg.get('algo_model') or cfg.get('backtest_model') or ...
        if not model: raise ValueError(...)

    The algo lane REFUSES to default. An all-unset strategy is a config
    error the runtime raises on — surface it as source='error' with a
    null effective value, never as the registry default (that masking is
    exactly what hid the 18,871-row cache_None mislabel).
    """
    declared = cfg.get('algo_model')
    if declared:
        return {"declared": declared, "effective": declared, "source": "explicit"}
    bt = cfg.get('backtest_model')
    if bt:
        return {"declared": None, "effective": bt, "source": "fallback_backtest"}
    return {"declared": None, "effective": None, "source": "error"}


@router.get("/models")
def get_fleet_models(user=Depends(get_current_user)):
    """One row per strategy with the runtime-effective model per lane.

    Cheap (single indexed strategies select) — feeds the Model Parity
    tab's model columns for the whole fleet in one request.
    """
    from db import get_admin_client

    c = get_admin_client()
    resp = c.table("strategies").select(
        "id,user_id,name,symbol,timeframe,forward_testing,config"
    ).order("id").execute()

    rows: List[Dict[str, Any]] = []
    for s in (resp.data or []):
        cfg = s.get("config")
        if isinstance(cfg, str):
            import json as _json
            try:
                cfg = _json.loads(cfg)
            except ValueError:
                cfg = {}
        cfg = cfg or {}
        rows.append({
            "strategy_id": s["id"],
            "user_id": s.get("user_id"),
            "name": s.get("name", ""),
            "symbol": s.get("symbol"),
            "timeframe": s.get("timeframe"),
            "forward_testing": bool(s.get("forward_testing")),
            "backtest_model": _resolve_backtest_model(cfg),
            "algo_model": _resolve_algo_model(cfg),
            "live_model": _resolve_live_model(cfg),
        })

    return {
        "now": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }


# ─────────────────────────────────────────────────────────────────────────
# 2. Per-strategy parity — server-side mirror of ParityDriftRibbonV2
# ─────────────────────────────────────────────────────────────────────────
# Tolerances copied verbatim from
# frontend/src/components/ParityDriftRibbonV2.tsx:23-29 — the numbers on
# this tab MUST reconcile with the detail-page ribbon.

_PRICE_ABS_MATCH = 0.01
_PRICE_REL_MATCH = 0.0002
_PRICE_REL_MINOR = 0.001
_NUM_ABS_MATCH = 1e-6
_NUM_REL_MATCH = 0.001
_NUM_REL_MINOR = 0.01
_MAX_BARS = 300          # "recent window" cap, same as the ribbon

_MAX_SIDS_PER_REQUEST = 8   # NEVER fleet-wide in one request
_CACHE_TTL_SEC = 600        # ~10 min

# {(strategy_id, lens): {"computed_at": epoch_sec, "payload": {...}}}
_parity_cache: Dict[tuple, Dict[str, Any]] = {}
_parity_lock = threading.Lock()


def _to_unix_sec(ts: Any) -> Optional[int]:
    """Mirror of ribbon toUnixSec(): ISO string (or epoch) → unix sec."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        v = float(ts)
        return int(v / 1000) if v > 1e12 else int(v)
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return int(dt.timestamp())
    except ValueError:
        return None


def _num(v: Any) -> Optional[float]:
    """Mirror of ribbon num(): null/'' → None; non-finite → None."""
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _compare(a: float, b: float, abs_tol: float,
             rel_match: float, rel_minor: float) -> str:
    d = abs(a - b)
    if d <= abs_tol:
        return "match"
    rel = d / max(abs(b), 1e-9)
    if rel <= rel_match:
        return "match"
    if rel <= rel_minor:
        return "minor"
    return "major"


def _compare_price(a: float, b: float) -> str:
    return _compare(a, b, _PRICE_ABS_MATCH, _PRICE_REL_MATCH, _PRICE_REL_MINOR)


def _compare_num(a: float, b: float) -> str:
    return _compare(a, b, _NUM_ABS_MATCH, _NUM_REL_MATCH, _NUM_REL_MINOR)


def _compute_parity_rows(
    backtest_bars: List[dict],
    alert_bars: List[dict],
    overlay_names: List[str],
    osc_names: List[str],
) -> Optional[Dict[str, Any]]:
    """Server-side mirror of the ParityDriftRibbonV2 useMemo (recent-window
    mode: no custom start/end → clip to the two lenses' overlap and take
    the LAST 300 union timestamps). Metric rows = Bar exists, O/H/L/C,
    then each pack indicator column (overlay + oscillator, deduped, same
    order as the ribbon). State/heatmap rows are a detail-page-only
    feature — not part of this fleet summary.
    """
    bt = [b for b in (backtest_bars or []) if b and b.get("timestamp") is not None]
    al = [b for b in (alert_bars or []) if b and b.get("timestamp") is not None]
    if not bt or not al:
        return None

    def _index(side: List[dict]) -> Dict[int, dict]:
        # sort by time; later duplicates win — same as the ribbon's Map
        out: Dict[int, dict] = {}
        for bar in sorted(side, key=lambda x: _to_unix_sec(x["timestamp"]) or 0):
            t = _to_unix_sec(bar["timestamp"])
            if t is not None:
                out[t] = bar
        return out

    bt_idx = _index(bt)
    al_idx = _index(al)
    if not bt_idx or not al_idx:
        return None

    lo = max(min(bt_idx), min(al_idx))
    hi = min(max(bt_idx), max(al_idx))

    all_times = sorted(set(bt_idx) | set(al_idx))
    if lo <= hi:
        all_times = [t for t in all_times if lo <= t <= hi]
    total = len(all_times)
    truncated = total > _MAX_BARS
    times = all_times[-_MAX_BARS:] if truncated else all_times

    common = bt_only = al_only = 0
    for t in times:
        in_bt, in_al = t in bt_idx, t in al_idx
        if in_bt and in_al:
            common += 1
        elif in_bt:
            bt_only += 1
        else:
            al_only += 1

    defs: List[Dict[str, str]] = [
        {"key": "exists", "label": "Bar exists", "kind": "exists", "col": ""},
        {"key": "open", "label": "Open", "kind": "price", "col": "open"},
        {"key": "high", "label": "High", "kind": "price", "col": "high"},
        {"key": "low", "label": "Low", "kind": "price", "col": "low"},
        {"key": "close", "label": "Close", "kind": "price", "col": "close"},
    ]
    seen: set = set()
    for cname in list(overlay_names or []) + list(osc_names or []):
        if not cname or cname in seen:
            continue
        seen.add(cname)
        defs.append({"key": f"ind:{cname}",
                     "label": cname.replace("_", " "),
                     "kind": "num", "col": cname})

    metrics: List[Dict[str, Any]] = []
    for d in defs:
        match = minor = major = 0
        for t in times:
            b_bar = bt_idx.get(t)
            a_bar = al_idx.get(t)
            if d["kind"] == "exists":
                status = "match" if (b_bar is not None and a_bar is not None) else "major"
            elif b_bar is None or a_bar is None:
                continue  # n/a — not comparable
            else:
                a = _num(b_bar.get(d["col"]))
                b = _num(a_bar.get(d["col"]))
                if a is None or b is None:
                    continue  # n/a
                status = _compare_price(a, b) if d["kind"] == "price" \
                    else _compare_num(a, b)
            if status == "match":
                match += 1
            elif status == "minor":
                minor += 1
            else:
                major += 1
        comparable = match + minor + major
        metrics.append({
            "key": d["key"],
            "label": d["label"],
            "kind": d["kind"],
            "match": match,
            "minor": minor,
            "major": major,
            "comparable": comparable,
            "pct": round(match / comparable * 100, 2) if comparable else None,
        })

    return {
        "metrics": metrics,
        "bars": {
            "common": common,
            "backtest_only": bt_only,
            "live_only": al_only,
            "window_bars": len(times),
            "total_overlap_bars": total,
            "truncated": truncated,
            "window_start": datetime.fromtimestamp(times[0], tz=timezone.utc).isoformat() if times else None,
            "window_end": datetime.fromtimestamp(times[-1], tz=timezone.utc).isoformat() if times else None,
        },
    }


def _compute_strategy_parity(strategy_id: int, lens: str,
                             user: dict) -> Dict[str, Any]:
    """Build both lens payloads through the SAME endpoint functions the
    Strategy Detail page uses, then run the ribbon-mirror comparison.

    - backtest lens: strategies.get_strategy_chart_data (REST/bar_cache
      pipeline, per-strategy data_days window, #29 trim — identical).
    - live lens: strategies.get_strategy_chart_data_from_cache with
      value_type=lens ('first' decision-time WS / 'latest' corrected).

    Reusing the endpoints (rather than re-implementing the pipelines)
    is what guarantees the percentages reconcile with the detail page's
    Per-Bar Parity Drift v2 module. Both are per-sid indexed paths.
    """
    from api.routers.strategies import (
        get_strategy_chart_data,
        get_strategy_chart_data_from_cache,
    )
    from db import get_admin_client, set_admin_user_context

    c = get_admin_client()
    row = c.table("strategies").select("id,user_id,symbol,timeframe,name") \
        .eq("id", strategy_id).maybe_single().execute()
    if not row or not row.data:
        raise HTTPException(status_code=404,
                            detail=f"strategy {strategy_id} not found")
    owner_id = row.data.get("user_id")

    # Admin context: the detail endpoints resolve their DB client through
    # get_client(); admin mode lets the fleet view compute any strategy
    # (same pattern as /chart-data-cache itself). User-scoped helpers
    # (confluence groups) resolve against the strategy owner.
    set_admin_user_context(owner_id)
    ctx_user = {"id": owner_id}

    # NOTE: pass days=None explicitly — calling the endpoint functions
    # directly would otherwise leave the FastAPI Query default object in
    # place (truthy!) and silently override data_days.
    bt_payload = get_strategy_chart_data(strategy_id, days=None, user=ctx_user)
    cache_payload = get_strategy_chart_data_from_cache(
        strategy_id, value_type=lens, days=None, user=ctx_user)

    overlay = bt_payload.get("overlay_indicators") or []
    osc = bt_payload.get("oscillator_indicators") or []

    result = _compute_parity_rows(
        bt_payload.get("chart_data") or [],
        cache_payload.get("chart_data") or [],
        overlay, osc,
    )

    out: Dict[str, Any] = {
        "strategy_id": strategy_id,
        "symbol": row.data.get("symbol"),
        "timeframe": row.data.get("timeframe"),
        "lens": lens,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "cached": False,
    }
    if result is None:
        out["metrics"] = []
        out["bars"] = None
        out["note"] = (
            f"Need both lenses — backtest bars: "
            f"{len(bt_payload.get('chart_data') or [])}, live {lens} bars: "
            f"{len(cache_payload.get('chart_data') or [])}."
            + (" " + cache_payload["note"] if cache_payload.get("note") else "")
        )
    else:
        out.update(result)
    return out


@router.get("/parity")
def get_model_parity(
    sids: str = Query(..., description=(
        "Comma-separated strategy ids — max "
        f"{_MAX_SIDS_PER_REQUEST} per request (the tab walks the fleet "
        "in small batches; a full-fleet request is rejected by design).")),
    lens: str = Query("first", description=(
        "Live-values lens: 'first' = decision-time WS values "
        "(first_close — what the engine SAW), 'latest' = REST-corrected "
        "values (what the cache healed to). Same semantics as the "
        "detail-page v2 ribbon toggle.")),
    refresh: bool = Query(False, description="Bypass the ~10 min cache."),
    user=Depends(get_current_user),
):
    """Per-strategy parity summary rows for the Model Parity tab.

    Heavy: each UNCACHED sid runs the full detail-page chart pipelines
    (backtest + live cache lens). Results cache ~10 min per
    (strategy_id, lens). Per-sid failures are trapped and returned as
    row-level errors so one bad strategy can't 500 the batch.
    """
    if lens not in ("first", "latest"):
        raise HTTPException(status_code=400,
                            detail="lens must be 'first' or 'latest'")
    try:
        sid_list = [int(s) for s in sids.split(",") if s.strip()]
    except ValueError:
        raise HTTPException(status_code=400,
                            detail="sids must be comma-separated integers")
    if not sid_list:
        raise HTTPException(status_code=400, detail="no sids provided")
    if len(sid_list) > _MAX_SIDS_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"max {_MAX_SIDS_PER_REQUEST} sids per request "
                   f"(got {len(sid_list)}) — batch the fleet")

    rows: List[Dict[str, Any]] = []
    for sid in sid_list:
        key = (sid, lens)
        now_s = time.time()
        with _parity_lock:
            entry = _parity_cache.get(key)
            if (not refresh and entry
                    and now_s - entry["computed_at"] < _CACHE_TTL_SEC):
                rows.append({**entry["payload"], "cached": True})
                continue
        try:
            t0 = time.time()
            payload = _compute_strategy_parity(sid, lens, user)
            logger.info("[MODEL-PARITY] sid=%s lens=%s computed in %.1fs",
                        sid, lens, time.time() - t0)
            with _parity_lock:
                _parity_cache[key] = {"computed_at": time.time(),
                                      "payload": payload}
            rows.append(payload)
        except HTTPException as e:
            rows.append({"strategy_id": sid, "lens": lens,
                         "error": str(e.detail)})
        except Exception as e:  # noqa: BLE001 — row-level trap by design
            logger.exception("[MODEL-PARITY] sid=%s lens=%s failed: %s",
                             sid, lens, e)
            rows.append({"strategy_id": sid, "lens": lens,
                         "error": str(e)[:300]})

    return {
        "now": datetime.now(timezone.utc).isoformat(),
        "lens": lens,
        "cache_ttl_seconds": _CACHE_TTL_SEC,
        "max_sids_per_request": _MAX_SIDS_PER_REQUEST,
        "rows": rows,
    }
