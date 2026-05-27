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
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query

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

# Backtest ↔ Alert divergence matching tolerance.
# Kevin's defs: phantom = alert-only, missed = backtest-trade-only.
# Match purely by (strategy_id, fill_ts) ±tolerance so we don't get
# tangled in the mixed event_type schema (legacy 'entry_signal' vs
# Phase 39 'fill'/'entry'). A trade edge (entry_fill_ts OR exit_fill_ts)
# within tolerance of an alert.fill_ts counts as paired. The window
# itself is caller-supplied (window_hours query param, default 24h).
_DIVERGENCE_TOLERANCE_SEC = 60.0            # ±60s
_WINDOW_HOURS_DEFAULT = 24
_WINDOW_HOURS_MIN = 1
_WINDOW_HOURS_MAX = 168                     # 7 days max


def _parse_iso_or_unix(s) -> datetime | None:
    """Accept ISO 8601 ('2026-05-26T22:00:00Z') or Unix-sec string.
    Mirrors data_health.py:48."""
    if not s:
        return None
    s = str(s).strip()
    try:
        return datetime.fromtimestamp(int(s), tz=timezone.utc)
    except (ValueError, TypeError):
        pass
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


@router.get("")
def get_strategy_health(
    user=Depends(get_current_user),
    window_hours: int = Query(
        _WINDOW_HOURS_DEFAULT,
        ge=_WINDOW_HOURS_MIN,
        le=_WINDOW_HOURS_MAX,
        description="Divergence window in hours for phantom/missed counts. "
                    "Clamped to [1, 168]; default 24. Ignored when both "
                    "`start` and `end` are provided.",
    ),
    start: str | None = Query(
        None,
        description="Custom window start (ISO 8601 or Unix sec). When set "
                    "with `end`, replaces window_hours with a fixed range. "
                    "Mirrors /admin/data-health's custom mode.",
    ),
    end: str | None = Query(
        None,
        description="Custom window end (ISO 8601 or Unix sec).",
    ),
):
    """One row per strategy with all the health signals + red flags.

    Response:
        {
          "now": "...",
          "window_hours": N,           # echoes the query param
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
    # exit_fill_ts + created_at for every backtest_% trade, group by
    # strategy_id. For ~30 strategies × thousands of trades this is ~MB
    # of data — keep the projection tight so we don't OOM. If it ever
    # becomes hot, move to a Postgres aggregate function.
    # `created_at` is the freshness signal Kevin asked for 2026-05-27 —
    # "when was the latest backtest trade written" vs entry_fill_ts which
    # is "when did the trade execute" (could be days ago for backfills).
    trade_resp = c.table("trades").select(
        "strategy_id,entry_fill_ts,exit_fill_ts,data_source,created_at"
    ).like("data_source", "backtest_%").execute()
    trades: List[Dict[str, Any]] = trade_resp.data or []

    # Fetch the most-recent alert per strategy (for the "last alert
    # received" column and the upside-down-timestamps filter). Bound by
    # last 30 days so we don't OOM on stale strategies — anything older
    # is effectively "never" for our purposes. Note this is UN-windowed
    # vs the recent-alerts pull below (which uses [window_start, window_end]
    # for phantom/missed pairing). 2026-05-27.
    last_alert_per_sid: Dict[int, str] = {}
    try:
        thirty_d_ago = (now - timedelta(days=30)).isoformat()
        last_alert_resp = c.table("alerts").select(
            "strategy_id,timestamp"
        ).gte("timestamp", thirty_d_ago).order(
            "timestamp", desc=True).execute()
        for a in (last_alert_resp.data or []):
            sid_a = a.get("strategy_id")
            ts_a = a.get("timestamp")
            if sid_a is None or not ts_a:
                continue
            # First occurrence per sid wins because list is sorted desc.
            if sid_a not in last_alert_per_sid:
                last_alert_per_sid[sid_a] = ts_a
    except Exception as e:
        logger.warning(
            "[health] last-alert-per-sid load failed: %s — column will be null", e)

    # Resolve the divergence window. Custom mode (both `start` and `end`
    # provided) overrides `window_hours` with an explicit fixed range
    # (mirrors /admin/data-health). When start/end are partial or
    # invalid, fall back to the rolling window_hours mode.
    custom_start = _parse_iso_or_unix(start)
    custom_end = _parse_iso_or_unix(end)
    is_custom = (custom_start is not None and custom_end is not None
                 and custom_start < custom_end)

    if is_custom:
        window_start_dt = custom_start
        window_end_dt = custom_end
    else:
        window_start_dt = now - timedelta(hours=window_hours)
        window_end_dt = now

    # Reduce: per-strategy { count, latest_entry, latest_exit,
    #                       recent_edges } where recent_edges are the
    # entry+exit timestamps within the divergence window, used below
    # for phantom/missed matching against alerts.
    window_cutoff_iso = window_start_dt.isoformat()
    window_end_iso = window_end_dt.isoformat()
    per_sid: Dict[int, Dict[str, Any]] = {}
    for t in trades:
        sid = t.get("strategy_id")
        if sid is None:
            continue
        agg = per_sid.setdefault(sid, {"count": 0, "latest_entry": None,
                                      "latest_exit": None,
                                      "latest_created": None,
                                      "recent_edges": []})
        agg["count"] += 1
        ek, xk = t.get("entry_fill_ts"), t.get("exit_fill_ts")
        ca = t.get("created_at")
        if ek and (agg["latest_entry"] is None or ek > agg["latest_entry"]):
            agg["latest_entry"] = ek
        if xk and (agg["latest_exit"] is None or xk > agg["latest_exit"]):
            agg["latest_exit"] = xk
        if ca and (agg["latest_created"] is None or ca > agg["latest_created"]):
            agg["latest_created"] = ca
        # Collect every trade edge that falls inside the divergence
        # window — entries AND exits, since alerts can correspond to
        # either side of a trade. In custom mode, also bound by end.
        if ek and window_cutoff_iso <= ek <= window_end_iso:
            agg["recent_edges"].append(ek)
        if xk and window_cutoff_iso <= xk <= window_end_iso:
            agg["recent_edges"].append(xk)

    # Recent alerts per strategy. Keep the projection tight. Bound by
    # [window_start, window_end] — for custom mode end < now, otherwise
    # end == now and the upper bound is a no-op.
    alert_resp = c.table("alerts").select(
        "strategy_id,fill_ts,trigger_ts,event_type"
    ).gte("timestamp", window_cutoff_iso).lte(
        "timestamp", window_end_iso).execute()
    alerts: List[Dict[str, Any]] = alert_resp.data or []
    alerts_per_sid: Dict[int, List[float]] = {}
    for a in alerts:
        sid = a.get("strategy_id")
        if sid is None:
            continue
        # Prefer fill_ts; fall back to trigger_ts. Both are ISO strings.
        ts_iso = a.get("fill_ts") or a.get("trigger_ts")
        dt = _parse_iso(ts_iso)
        if dt is None:
            continue
        alerts_per_sid.setdefault(sid, []).append(dt.timestamp())

    def _pair_phantom_missed(edge_isos: List[str], alert_unix: List[float],
                              upper_bound_unix: Optional[float] = None):
        """Greedy pairing within ±tolerance. Returns (phantom, missed, paired).

        phantom = unpaired alerts (alert-only — Kevin's 'phantom').
        missed  = unpaired trade edges (backtest-only — Kevin's 'missed').
        paired  = successful (alert, edge) matches within ±tolerance.
                  paired_alerts == paired_edges by construction (1:1 pairing).

        2026-05-27: `upper_bound_unix` is an optional cap. Events newer
        than this timestamp are dropped before pairing. Used to compute
        an "apples-to-apples" count that excludes the lag-tail — alerts
        and trade edges that don't yet have a corresponding entry on the
        OTHER source because that source hasn't caught up. Without this,
        the trailing lag inflates phantom (alerts fired but backtest
        not yet recorded) and missed (backtest recorded but alert not
        yet captured by a finished query) counts.
        """
        # Convert edges → unix once, sort both lists.
        edges: List[float] = []
        for s in edge_isos:
            dt = _parse_iso(s)
            if dt is not None:
                edges.append(dt.timestamp())
        edges.sort()
        alerts_sorted = sorted(alert_unix)
        if upper_bound_unix is not None:
            edges = [e for e in edges if e <= upper_bound_unix]
            alerts_sorted = [a for a in alerts_sorted if a <= upper_bound_unix]
        # Two-pointer greedy pair: walk both lists, pair when |diff| ≤ tol.
        i = j = 0
        paired_edges = 0
        paired_alerts = 0
        while i < len(edges) and j < len(alerts_sorted):
            diff = alerts_sorted[j] - edges[i]
            if abs(diff) <= _DIVERGENCE_TOLERANCE_SEC:
                paired_edges += 1
                paired_alerts += 1
                i += 1
                j += 1
            elif diff < 0:
                j += 1   # alert too early — it's a phantom, skip ahead
            else:
                i += 1   # edge too early — it's missed, skip ahead
        phantom = len(alerts_sorted) - paired_alerts
        missed = len(edges) - paired_edges
        return phantom, missed, paired_alerts

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
                                "latest_exit": None, "recent_edges": []})
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

        # 2026-05-27: freshness signals Kevin asked for. Computed BEFORE
        # the pairing so we can derive fair_cutoff from them.
        # `last_backtest_created_at` = the most-recent created_at across all
        # backtest_% trade rows (i.e., when the trades table last got a
        # new backtest entry). Distinct from `last_entry_ts` which is the
        # latest entry_fill_ts (could be days old for backfills).
        # `last_alert_at` = the most-recent alert across the whole alerts
        # table (un-windowed, capped to last 30d).
        last_bt_created = _parse_iso(agg.get("latest_created"))
        last_alert_at = _parse_iso(last_alert_per_sid.get(sid))

        # Phantom (alert-only), missed (backtest-trade-only), and paired
        # (successful match) counts in the window. Symmetric matching:
        # each trade has two edges (entry + exit) that can each be paired
        # against an alert, so paired_count is in edges-per-window, not
        # trades-per-window (could be up to ~2× trade_count_backtest).
        edge_isos = agg["recent_edges"]
        alert_unix_list = alerts_per_sid.get(sid, [])
        phantom_count, missed_count, paired_count = _pair_phantom_missed(
            edge_isos, alert_unix_list)

        # 2026-05-27: apples-to-apples counts. Cap event timestamps to
        # min(last_alert_at, last_backtest_created_at). Events newer than
        # this cap haven't had a chance to be matched on the slower side,
        # so excluding them removes the lag-tail false positives Kevin
        # described: "phantom says 291 but the last 20 were fired in the
        # last 5 min and the backtest data is 10 min stale — so 271 is
        # the honest count."
        # When one source is missing entirely, fair_cutoff is None and
        # fair counts mirror raw counts (nothing to cap against).
        fair_cutoff_dt: Optional[datetime] = None
        if last_bt_created is not None and last_alert_at is not None:
            fair_cutoff_dt = min(last_bt_created, last_alert_at)
        elif last_bt_created is not None:
            fair_cutoff_dt = last_bt_created
        elif last_alert_at is not None:
            fair_cutoff_dt = last_alert_at
        if fair_cutoff_dt is not None:
            phantom_count_fair, missed_count_fair, paired_count_fair = (
                _pair_phantom_missed(
                    edge_isos, alert_unix_list,
                    upper_bound_unix=fair_cutoff_dt.timestamp()))
        else:
            phantom_count_fair = phantom_count
            missed_count_fair = missed_count
            paired_count_fair = paired_count

        forward_testing = bool(s.get("forward_testing"))
        is_streaming_eligible = ("entry_trigger_confluence_id" in cfg)

        snapshot_age = _age_sec(snapshot_at, now)
        kpis_age = _age_sec(kpis_at, now)
        data_refresh_age = _age_sec(data_refreshed_at, now)
        last_entry_age = _age_sec(last_entry_dt, now)
        last_exit_age = _age_sec(last_exit_dt, now)
        last_bt_created_age = _age_sec(last_bt_created, now)
        last_alert_age = _age_sec(last_alert_at, now)

        # Upside-down filter signal: when last alert and last backtest are
        # both present but materially out of sync (>1h apart in either
        # direction), phantom/missed numbers aren't meaningful — one source
        # is stale relative to the other. Surfaces as a top-level boolean
        # so the UI filter can hide these rows in one expression. The
        # threshold mirrors `_STALE_SNAPSHOT_SEC` for consistency.
        timestamps_upside_down = False
        upside_down_delta_sec: Optional[int] = None
        if last_bt_created is not None and last_alert_at is not None:
            delta_sec = abs((last_alert_at - last_bt_created).total_seconds())
            upside_down_delta_sec = int(delta_sec)
            if delta_sec > _STALE_SNAPSHOT_SEC:
                timestamps_upside_down = True

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
        if phantom_count > 0:
            red_flags.append("phantom_alerts")
        if missed_count > 0:
            red_flags.append("missed_alerts")

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
            # 2026-05-27 — freshness signals + upside-down filter source.
            "last_backtest_created_at": (last_bt_created.isoformat()
                                         if last_bt_created else None),
            "last_backtest_created_age_sec": last_bt_created_age,
            "last_alert_at": (last_alert_at.isoformat()
                              if last_alert_at else None),
            "last_alert_age_sec": last_alert_age,
            "timestamps_upside_down": timestamps_upside_down,
            "upside_down_delta_sec": upside_down_delta_sec,
            # 2026-05-27 — per-strategy auto-snapshot opt-out flag.
            # True = data-worker maintains snapshot; False = data-worker
            # skips this strategy entirely. Default True.
            "snapshot_subscribe_enabled": bool(
                cfg.get("snapshot_subscribe_enabled", True)),
            "trade_count_backtest": agg["count"],
            "parity_status": parity if isinstance(parity, dict) else None,
            "parity_verdict": parity_verdict,
            "discrepancies_count": active_discrepancies,
            "phantom_count": phantom_count,
            "missed_count": missed_count,
            "paired_count": paired_count,
            # 2026-05-27 — apples-to-apples counts (lag-tail excluded).
            # Same pairing, but skips events newer than min(last_alert,
            # last_backtest_created) so the trailing lag doesn't inflate
            # counts. Use these when "apples-to-apples" mode is on.
            "phantom_count_fair": phantom_count_fair,
            "missed_count_fair": missed_count_fair,
            "paired_count_fair": paired_count_fair,
            "fair_cutoff_ts": (fair_cutoff_dt.isoformat()
                               if fair_cutoff_dt else None),
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
        "window_hours": window_hours,   # echo request, for UI bookkeeping
        "window_start": window_cutoff_iso,
        "window_end": window_end_iso,
        "mode": "custom" if is_custom else "rolling",
        "rows": out_rows,
    }


# =============================================================================
# Backlog endpoint — trade-level divergence with auto-classification
# =============================================================================

# Auto-classification reasons. The frontend's "needs investigation" filter
# hides rows whose reason is anything other than `needs_investigation`.
# Add new reasons here as we identify new bug-classes.
_CLASS_NEEDS_INVESTIGATION = "needs_investigation"
_CLASS_TIMESTAMPS_OUT_OF_SYNC = "timestamps_out_of_sync"
_CLASS_PHASE2_SIGNAL_EXIT = "phase2_signal_exit"
_CLASS_NON_FILL_EVENT = "non_fill_event"
_CLASS_LEGACY_STRATEGY = "legacy_strategy"


@router.get("/backlog")
def get_strategy_health_backlog(
    user=Depends(get_current_user),
    window_hours: int = Query(
        _WINDOW_HOURS_DEFAULT,
        ge=_WINDOW_HOURS_MIN,
        le=_WINDOW_HOURS_MAX,
        description="Divergence lookback in hours. Ignored when both "
                    "`start` and `end` are provided.",
    ),
    start: str | None = Query(
        None, description="Custom window start (ISO 8601 or Unix-sec)."),
    end: str | None = Query(
        None, description="Custom window end (ISO 8601 or Unix-sec)."),
    only_needs_investigation: bool = Query(
        False,
        description="If True, return ONLY events classified as "
                    "'needs_investigation' — the mysteries worth chasing. "
                    "Default False returns all events with classifications."),
    apples_to_apples: bool = Query(
        True,
        description="If True (default), drop events newer than "
                    "min(last_alert_at, last_backtest_created_at) per "
                    "strategy. Matches the V1 'apples-to-apples' filter — "
                    "excludes events that haven't had a chance to be "
                    "matched yet (tail-lag false positives)."),
    max_rows: int = Query(
        500, ge=10, le=5000,
        description="Hard cap on returned rows. Backlog runs heavy when "
                    "every strategy has hundreds of unpaired events. UI "
                    "warns when truncation kicks in. 2026-05-27."),
):
    """Per-event divergence backlog.

    Same window logic as `GET /api/admin/strategy-health` but returns one
    row per UNPAIRED event (phantom alert or missed backtest edge) instead
    of one row per strategy. Each row carries an auto-classification so
    the UI can show known-cause divergence muted and surface the real
    mysteries.

    The pairing is the same 2-way (alerts vs backtest) used in the
    overview endpoint. Algo lane stays out per the 2026-05-27 product
    decision — the goal is "live alerts match backtest", and algo is a
    diagnostic detour for that.

    Response: ``{rows: [{strategy_id, name, ..., event_type, classification,
    classification_reason, ...}]}``. UI sorts/filters client-side.
    """
    from db import get_admin_client
    c = get_admin_client()
    now = datetime.now(timezone.utc)

    # Resolve window — duplicates the overview endpoint's logic on purpose.
    # When we share more later we'll factor out a helper.
    custom_start = _parse_iso_or_unix(start)
    custom_end = _parse_iso_or_unix(end)
    is_custom = (custom_start is not None and custom_end is not None
                 and custom_start < custom_end)
    if is_custom:
        window_start_dt = custom_start
        window_end_dt = custom_end
    else:
        window_start_dt = now - timedelta(hours=window_hours)
        window_end_dt = now
    window_start_iso = window_start_dt.isoformat()
    window_end_iso = window_end_dt.isoformat()

    # Pull strategies once for name/symbol/tf metadata + config.
    strat_resp = c.table("strategies").select(
        "id,name,symbol,timeframe,direction,config,data_refreshed_at"
    ).execute()
    strats_by_id: Dict[int, Dict[str, Any]] = {}
    for s in (strat_resp.data or []):
        strats_by_id[s.get("id")] = s

    # Per-strategy "last backtest created" + "last alert" for the apples-
    # to-apples cap. Cheap: 2 queries (sorted desc, take first per sid).
    last_bt_per_sid: Dict[int, datetime] = {}
    if apples_to_apples:
        try:
            bt_resp = c.table("trades").select(
                "strategy_id,created_at"
            ).like("data_source", "backtest_%").order(
                "created_at", desc=True).execute()
            for t in (bt_resp.data or []):
                sid_t = t.get("strategy_id")
                ts_t = t.get("created_at")
                if sid_t is None or not ts_t:
                    continue
                if sid_t not in last_bt_per_sid:
                    dt = _parse_iso(ts_t)
                    if dt is not None:
                        last_bt_per_sid[sid_t] = dt
        except Exception as e:
            logger.warning("[backlog] last_bt_per_sid load failed: %s", e)

    last_alert_per_sid: Dict[int, datetime] = {}
    if apples_to_apples:
        try:
            thirty_d_ago = (now - timedelta(days=30)).isoformat()
            la_resp = c.table("alerts").select(
                "strategy_id,timestamp"
            ).gte("timestamp", thirty_d_ago).order(
                "timestamp", desc=True).execute()
            for a in (la_resp.data or []):
                sid_a = a.get("strategy_id")
                ts_a = a.get("timestamp")
                if sid_a is None or not ts_a:
                    continue
                if sid_a not in last_alert_per_sid:
                    dt = _parse_iso(ts_a)
                    if dt is not None:
                        last_alert_per_sid[sid_a] = dt
        except Exception as e:
            logger.warning("[backlog] last_alert_per_sid load failed: %s", e)

    # Pull trades within window for the missed-edge side. Window-scoped
    # here for efficiency — the overview pulls all backtest trades but
    # the backlog only needs edges in the divergence window.
    # NOTE: `exit_trigger` is NOT a top-level column on the trades table
    # — it lives inside the `data` JSONB. _classify_missed reads it
    # from data when needed.
    trade_resp = c.table("trades").select(
        "id,strategy_id,entry_fill_ts,exit_fill_ts,exit_reason,"
        "data,data_source"
    ).like("data_source", "backtest_%").gte(
        "entry_fill_ts", window_start_iso).execute()
    trades_in_window: List[Dict[str, Any]] = trade_resp.data or []

    # Pull alerts within window.
    alert_resp = c.table("alerts").select(
        "id,strategy_id,fill_ts,trigger_ts,event_type"
    ).gte("timestamp", window_start_iso).lte(
        "timestamp", window_end_iso).execute()
    alerts_in_window: List[Dict[str, Any]] = alert_resp.data or []

    # Group by strategy.
    trades_per_sid: Dict[int, List[Dict[str, Any]]] = {}
    for t in trades_in_window:
        sid = t.get("strategy_id")
        if sid is None:
            continue
        trades_per_sid.setdefault(sid, []).append(t)

    alerts_per_sid: Dict[int, List[Dict[str, Any]]] = {}
    for a in alerts_in_window:
        sid = a.get("strategy_id")
        if sid is None:
            continue
        alerts_per_sid.setdefault(sid, []).append(a)

    # Build per-event backlog rows by replicating the 2-pointer pairing
    # logic from `_pair_phantom_missed` but preserving identities. Each
    # unpaired edge/alert becomes a row.
    def _classify_phantom(alert: Dict[str, Any], strat: Dict[str, Any]) -> tuple[str, str]:
        """Return (classification, human-readable reason)."""
        cfg_ = strat.get("config") or {}
        if not isinstance(cfg_, dict):
            cfg_ = {}
        if "entry_trigger_confluence_id" not in cfg_:
            return _CLASS_LEGACY_STRATEGY, "Strategy pre-dates confluence-ID schema"
        ev = (alert.get("event_type") or "").lower()
        if ev and ev not in ("entry", "exit", "fill",
                              "entry_signal", "exit_signal"):
            return _CLASS_NON_FILL_EVENT, f"event_type={ev!r} (not a fill)"
        return _CLASS_NEEDS_INVESTIGATION, "Alert fired but no matching backtest edge"

    def _classify_missed(trade: Dict[str, Any], edge: str,
                         strat: Dict[str, Any]) -> tuple[str, str]:
        """Classify a missed entry or exit edge."""
        cfg_ = strat.get("config") or {}
        if not isinstance(cfg_, dict):
            cfg_ = {}
        if "entry_trigger_confluence_id" not in cfg_:
            return _CLASS_LEGACY_STRATEGY, "Strategy pre-dates confluence-ID schema"
        # Signal-exit triggers without _ib suffix have no L-type walker
        # (Phase 2 gap — indicator-vs-indicator crosses). Common case for
        # macd_line_v2 cross_bear / cross_bull.
        exit_trigger = (trade.get("exit_trigger")
                        or (trade.get("data") or {}).get("exit_trigger") or "")
        if edge == "exit" and isinstance(exit_trigger, str) and exit_trigger:
            if not exit_trigger.endswith("_ib") and trade.get("exit_reason") not in (
                    "stop_loss", "stop", "target"):
                return _CLASS_PHASE2_SIGNAL_EXIT, (
                    f"exit_trigger={exit_trigger!r} has no L-type spec — "
                    "Phase 2 indicator-vs-indicator gap")
        return _CLASS_NEEDS_INVESTIGATION, (
            f"Backtest {edge} edge with no matching alert within ±60s")

    out_rows: List[Dict[str, Any]] = []
    truncated = False
    for sid, strat in strats_by_id.items():
        # Apples-to-apples cap: drop events newer than min(last_alert,
        # last_backtest_created). Mirrors V1 honest-counts logic.
        fair_cutoff_unix: Optional[float] = None
        if apples_to_apples:
            lb = last_bt_per_sid.get(sid)
            la = last_alert_per_sid.get(sid)
            if lb is not None and la is not None:
                fair_cutoff_unix = min(lb, la).timestamp()
            elif lb is not None:
                fair_cutoff_unix = lb.timestamp()
            elif la is not None:
                fair_cutoff_unix = la.timestamp()

        # Build edge list: each trade contributes (entry_ts, 'entry', trade)
        # and (exit_ts, 'exit', trade). Sort by ts.
        edges: List[tuple[float, str, Dict[str, Any]]] = []
        for t in trades_per_sid.get(sid, []):
            for col, edge in (("entry_fill_ts", "entry"), ("exit_fill_ts", "exit")):
                ek = t.get(col)
                if not ek:
                    continue
                dt = _parse_iso(ek)
                if dt is None:
                    continue
                if fair_cutoff_unix is not None and dt.timestamp() > fair_cutoff_unix:
                    continue   # apples-to-apples: drop lag-tail
                edges.append((dt.timestamp(), edge, t))
        edges.sort(key=lambda x: x[0])

        alerts_sorted = sorted(
            ((_parse_iso(a.get("fill_ts") or a.get("trigger_ts")), a)
             for a in alerts_per_sid.get(sid, [])
             if (a.get("fill_ts") or a.get("trigger_ts"))),
            key=lambda x: x[0].timestamp() if x[0] is not None else 0,
        )
        alerts_sorted = [(dt.timestamp(), a) for dt, a in alerts_sorted
                          if dt is not None and (
                              fair_cutoff_unix is None
                              or dt.timestamp() <= fair_cutoff_unix)]

        # 2-pointer pair.
        i = j = 0
        paired_edge_idx = set()
        paired_alert_idx = set()
        while i < len(edges) and j < len(alerts_sorted):
            diff = alerts_sorted[j][0] - edges[i][0]
            if abs(diff) <= _DIVERGENCE_TOLERANCE_SEC:
                paired_edge_idx.add(i)
                paired_alert_idx.add(j)
                i += 1
                j += 1
            elif diff < 0:
                j += 1
            else:
                i += 1

        # Emit unpaired events.
        for idx, (ts, edge, trade) in enumerate(edges):
            if idx in paired_edge_idx:
                continue
            cls, reason = _classify_missed(trade, edge, strat)
            if only_needs_investigation and cls != _CLASS_NEEDS_INVESTIGATION:
                continue
            out_rows.append({
                "event_type": "missed",
                "strategy_id": sid,
                "strategy_name": strat.get("name"),
                "symbol": strat.get("symbol"),
                "timeframe": strat.get("timeframe"),
                "direction": strat.get("direction"),
                "edge": edge,
                "timestamp": datetime.fromtimestamp(
                    ts, tz=timezone.utc).isoformat(),
                "trade_id": trade.get("id"),
                "alert_id": None,
                "exit_reason": trade.get("exit_reason"),
                "exit_trigger": (trade.get("exit_trigger")
                                 or (trade.get("data") or {}).get("exit_trigger")),
                "classification": cls,
                "classification_reason": reason,
            })
        for idx, (ts, alert) in enumerate(alerts_sorted):
            if idx in paired_alert_idx:
                continue
            cls, reason = _classify_phantom(alert, strat)
            if only_needs_investigation and cls != _CLASS_NEEDS_INVESTIGATION:
                continue
            out_rows.append({
                "event_type": "phantom",
                "strategy_id": sid,
                "strategy_name": strat.get("name"),
                "symbol": strat.get("symbol"),
                "timeframe": strat.get("timeframe"),
                "direction": strat.get("direction"),
                "edge": None,
                "timestamp": datetime.fromtimestamp(
                    ts, tz=timezone.utc).isoformat(),
                "trade_id": None,
                "alert_id": alert.get("id"),
                "exit_reason": None,
                "exit_trigger": None,
                "classification": cls,
                "classification_reason": reason,
            })

    # Sort: needs_investigation first (action item), then by timestamp desc.
    out_rows.sort(key=lambda r: (
        0 if r["classification"] == _CLASS_NEEDS_INVESTIGATION else 1,
        -(_parse_iso(r["timestamp"]).timestamp()
          if _parse_iso(r["timestamp"]) else 0),
    ))

    total_before_truncation = len(out_rows)
    if total_before_truncation > max_rows:
        out_rows = out_rows[:max_rows]
        truncated = True

    return {
        "now": now.isoformat(),
        "window_hours": window_hours,
        "window_start": window_start_iso,
        "window_end": window_end_iso,
        "mode": "custom" if is_custom else "rolling",
        "apples_to_apples": apples_to_apples,
        "row_count": len(out_rows),
        "total_count": total_before_truncation,
        "truncated": truncated,
        "max_rows": max_rows,
        "rows": out_rows,
    }
