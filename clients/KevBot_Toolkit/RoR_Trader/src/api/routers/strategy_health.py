"""Strategy Health router — per-strategy freshness + red-flag overview.

Backs the /admin/strategy-health frontend page. Aggregates the signals
the data-worker + recompute paths leave in the DB (snapshot age, KPI
age, latest trade, parity status, discrepancies) into one row per
strategy and computes a `red_flags` list so the UI can sort by
"things that look broken."

Read-only, admin-scoped (relies on the existing logged-in guard —
RoR_Trader is solo SaaS today, no separate role gating).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends

from api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/strategy-health",
                   tags=["admin", "strategy-health"])


def _parse_iso(s: Any) -> datetime | None:
    if not isinstance(s, str) or not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _age_sec(ts: datetime | None, now: datetime) -> int | None:
    if ts is None:
        return None
    return int((now - ts).total_seconds())


# How stale before we light up a red flag, by signal.
_STALE_SNAPSHOT_SEC = 60 * 60          # 1h — data-worker should flush 5-min
_STALE_KPIS_SEC = 24 * 60 * 60         # 24h — recompute cron is daily-ish
_STALE_DATA_REFRESH_SEC = 24 * 60 * 60
_NO_RECENT_TRADE_SEC = 7 * 24 * 60 * 60  # 7 days without an entry while
                                         # forward-testing = suspicious


@router.get("")
def get_strategy_health(user=Depends(get_current_user)):
    """One row per strategy with all the health signals + red flags.

    Response:
        {
          "now": "...",
          "rows": [{
            "strategy_id": 137, "user_id": "...",
            "name": "...", "symbol": "TSLA", "timeframe": "5Min",
            "direction": "LONG", "strategy_origin": "manual",
            "forward_testing": true, "forward_test_start": "...",
            "data_source": "backtest_rest_hifi",
            "snapshot_at": "...", "snapshot_age_sec": N,
            "last_recompute_until_ts": "...",
            "kpis_computed_at": "...", "kpis_age_sec": N,
            "kpis_stale_since": "..." | null,
            "data_refreshed_at": "...", "data_refreshed_age_sec": N,
            "last_entry_ts": "...", "last_entry_age_sec": N,
            "last_exit_ts": "...", "last_exit_age_sec": N,
            "trade_count_backtest": N,
            "parity_status": {...} | null,
            "discrepancies_count": N,
            "red_flags": ["snapshot_stale", ...]
          }, ...]
        }
    """
    from db import get_admin_client

    c = get_admin_client()
    now = datetime.now(timezone.utc)

    # Pull every strategy with the columns we need. `config` is JSONB so
    # we reach into it for snapshot/recompute timestamps; the rest are
    # dedicated columns (Phase 40 + Strategy Health Badge migrations).
    strat_resp = c.table("strategies").select(
        "id,user_id,name,symbol,direction,timeframe,strategy_origin,"
        "forward_testing,forward_test_start,"
        "data_source,data_refreshed_at,"
        "kpis_computed_at,kpis_stale_since,"
        "parity_status,discrepancies,discrepancies_dismissed_at,"
        "config,updated_at,created_at"
    ).order("id").execute()
    strats: List[Dict[str, Any]] = strat_resp.data or []

    # Aggregate trade-recency in a single pass: pull entry_fill_ts +
    # exit_fill_ts for every backtest_% trade, group by strategy_id.
    # For ~30 strategies × thousands of trades this is ~MB of data — keep
    # the projection tight (3 fields) so we don't OOM. If it ever becomes
    # hot, move to a Postgres aggregate function.
    trade_resp = c.table("trades").select(
        "strategy_id,entry_fill_ts,exit_fill_ts,data_source"
    ).like("data_source", "backtest_%").execute()
    trades: List[Dict[str, Any]] = trade_resp.data or []

    # Reduce: per-strategy { count, latest_entry, latest_exit }.
    per_sid: Dict[int, Dict[str, Any]] = {}
    for t in trades:
        sid = t.get("strategy_id")
        if sid is None:
            continue
        agg = per_sid.setdefault(sid, {"count": 0, "latest_entry": None,
                                      "latest_exit": None})
        agg["count"] += 1
        ek, xk = t.get("entry_fill_ts"), t.get("exit_fill_ts")
        if ek and (agg["latest_entry"] is None or ek > agg["latest_entry"]):
            agg["latest_entry"] = ek
        if xk and (agg["latest_exit"] is None or xk > agg["latest_exit"]):
            agg["latest_exit"] = xk

    out_rows: List[Dict[str, Any]] = []
    for s in strats:
        sid = s.get("id")
        cfg = s.get("config") or {}
        if not isinstance(cfg, dict):
            cfg = {}

        snapshot_at = _parse_iso(cfg.get("engine_snapshot_at"))
        last_recompute_until = _parse_iso(cfg.get("last_recompute_until_ts"))
        kpis_at = _parse_iso(s.get("kpis_computed_at"))
        data_refreshed_at = _parse_iso(s.get("data_refreshed_at"))

        agg = per_sid.get(sid, {"count": 0, "latest_entry": None,
                                "latest_exit": None})
        last_entry_dt = _parse_iso(agg["latest_entry"])
        last_exit_dt = _parse_iso(agg["latest_exit"])

        discrepancies = s.get("discrepancies") or []
        if not isinstance(discrepancies, list):
            discrepancies = []
        # A dismissed discrepancy doesn't count as a red flag.
        dismissed_at = _parse_iso(s.get("discrepancies_dismissed_at"))
        active_discrepancies = (0 if dismissed_at and not discrepancies
                                else len(discrepancies))

        parity = s.get("parity_status") or None
        parity_verdict = (parity.get("verdict")
                          if isinstance(parity, dict) else None)

        forward_testing = bool(s.get("forward_testing"))
        is_streaming_eligible = ("entry_trigger_confluence_id" in cfg)

        snapshot_age = _age_sec(snapshot_at, now)
        kpis_age = _age_sec(kpis_at, now)
        data_refresh_age = _age_sec(data_refreshed_at, now)
        last_entry_age = _age_sec(last_entry_dt, now)
        last_exit_age = _age_sec(last_exit_dt, now)

        # ── Red flags — only ones that mean *action needed*.
        red_flags: List[str] = []
        if not is_streaming_eligible:
            red_flags.append("legacy_no_confluence_id")
        elif agg["count"] == 0:
            red_flags.append("no_baseline")
        else:
            # Active strategies only — flag staleness against the data-
            # worker / recompute cadences.
            if snapshot_at is None:
                red_flags.append("snapshot_missing")
            elif snapshot_age is not None and snapshot_age > _STALE_SNAPSHOT_SEC:
                red_flags.append("snapshot_stale")

            if kpis_age is not None and kpis_age > _STALE_KPIS_SEC:
                red_flags.append("kpis_stale")
            if s.get("kpis_stale_since"):
                red_flags.append("kpis_marked_stale")

            if (data_refresh_age is not None
                    and data_refresh_age > _STALE_DATA_REFRESH_SEC):
                red_flags.append("data_refresh_stale")

            if (forward_testing and last_entry_age is not None
                    and last_entry_age > _NO_RECENT_TRADE_SEC):
                red_flags.append("no_recent_trades")

        if parity_verdict == "fail":
            red_flags.append("parity_fail")
        if active_discrepancies > 0:
            red_flags.append("has_discrepancies")

        out_rows.append({
            "strategy_id": sid,
            "user_id": s.get("user_id"),
            "name": s.get("name"),
            "symbol": s.get("symbol"),
            "timeframe": s.get("timeframe"),
            "direction": s.get("direction"),
            "strategy_origin": s.get("strategy_origin"),
            "forward_testing": forward_testing,
            "forward_test_start": s.get("forward_test_start"),
            "streaming_eligible": is_streaming_eligible,
            "data_source": s.get("data_source") or cfg.get("data_source"),
            "backtest_model": (cfg.get("backtest_model")
                               or s.get("backtest_model")
                               or "rest_hifi"),
            "snapshot_at": snapshot_at.isoformat() if snapshot_at else None,
            "snapshot_age_sec": snapshot_age,
            "last_recompute_until_ts": (last_recompute_until.isoformat()
                                        if last_recompute_until else None),
            "kpis_computed_at": kpis_at.isoformat() if kpis_at else None,
            "kpis_age_sec": kpis_age,
            "kpis_stale_since": s.get("kpis_stale_since"),
            "data_refreshed_at": (data_refreshed_at.isoformat()
                                  if data_refreshed_at else None),
            "data_refreshed_age_sec": data_refresh_age,
            "last_entry_ts": last_entry_dt.isoformat() if last_entry_dt else None,
            "last_entry_age_sec": last_entry_age,
            "last_exit_ts": last_exit_dt.isoformat() if last_exit_dt else None,
            "last_exit_age_sec": last_exit_age,
            "trade_count_backtest": agg["count"],
            "parity_status": parity if isinstance(parity, dict) else None,
            "parity_verdict": parity_verdict,
            "discrepancies_count": active_discrepancies,
            "red_flags": red_flags,
            "updated_at": s.get("updated_at"),
            "created_at": s.get("created_at"),
        })

    # Sort: most-red-flagged first, then by symbol/timeframe so the UI
    # has a stable secondary order.
    out_rows.sort(key=lambda r: (-len(r["red_flags"]),
                                  r["symbol"] or "",
                                  r["timeframe"] or "",
                                  r["strategy_id"] or 0))

    return {
        "now": now.isoformat(),
        "rows": out_rows,
    }
