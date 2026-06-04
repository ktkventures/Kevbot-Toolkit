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

# Supabase REST returns up to ~50k rows per request when no .range() is set.
# The fleet has > 50k backtest trades total; an unpaginated select silently
# truncates and drops whichever strategies happen to fall past the cap (which
# strategies it drops is non-deterministic without an explicit .order). Always
# use _paginated_fetch for the per-strategy aggregation queries below.
# (Bug discovered 2026-06-02 — strategy-health reported 100% phantom for new
# canaries sids 271/272/275 because their trades sat past the silent cap.)
_PAGE_SIZE = 5000


def _paginated_fetch(query_builder) -> List[Dict[str, Any]]:
    """Iterate `.range(offset, offset+_PAGE_SIZE-1)` on a PostgREST query
    builder until a partial page comes back (signal: < _PAGE_SIZE rows).
    Caller is responsible for setting `.order(...)` on the builder so
    pagination is deterministic.
    """
    rows: List[Dict[str, Any]] = []
    offset = 0
    while True:
        chunk = query_builder.range(
            offset, offset + _PAGE_SIZE - 1).execute().data or []
        rows.extend(chunk)
        if len(chunk) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return rows


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
    # Sort + paginate so the result is deterministic and complete. Without
    # this, PostgREST silently caps at ~50k rows; strategies whose trades
    # land past the cap get an empty `recent_edges` and are misreported as
    # 100% phantom downstream (see _PAGE_SIZE docstring).
    trade_query = c.table("trades").select(
        "strategy_id,entry_fill_ts,exit_fill_ts,data_source,created_at"
    ).like("data_source", "backtest_%").order("strategy_id").order("id")
    trades: List[Dict[str, Any]] = _paginated_fetch(trade_query)

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

    # 2026-05-29: GLOBAL fair_cutoff — used for cross-strategy comparisons.
    # The per-strategy `fair_cutoff_dt = min(last_bt_processed, last_alert_at)`
    # gives each strategy its own cutoff, which can drift by minutes between
    # otherwise-identical strategies (different batch timing). When you're
    # comparing strategies, that asymmetry inflates one strategy's phantom
    # count vs another's even when the underlying alerts/trades are
    # identical. The GLOBAL cutoff is the EARLIEST per-strategy fair_cutoff
    # across the ACTIVE fleet — every strategy is then evaluated against
    # the same absolute time. Important constraints:
    #   - Only strategies with at least one event in the current window
    #     contribute (otherwise a strategy that's been dormant for months
    #     drags the global cutoff back to its last activity).
    #   - The global cutoff is clamped to window_start so it can never
    #     pre-date the window itself.
    _global_fair_cutoff_dt: Optional[datetime] = None
    for s in strats:
        sid = s.get("id")
        agg = per_sid.get(sid)
        if not agg:
            continue
        # Skip strategies that don't have BOTH lanes active in the
        # window. A strategy with alerts but no recent backtest trades
        # would otherwise drag the global cutoff back to its last
        # backtest-processed time — which is exactly the phantom condition
        # we want to NOT have control the cutoff.
        _le = _parse_iso(agg.get("latest_entry"))
        _lx = _parse_iso(agg.get("latest_exit"))
        _la = _parse_iso(last_alert_per_sid.get(sid))
        _bt = None
        for cand in (_le, _lx):
            if cand is not None and (_bt is None or cand > _bt):
                _bt = cand
        if _bt is None or _la is None:
            continue
        # Require last_bt_processed AND last_alert_at to both be within the
        # current window. A strategy that hasn't fired in this window
        # would otherwise contribute a stale value.
        if _bt < window_start_dt or _la < window_start_dt:
            continue
        _strat_cutoff = min(_bt, _la)
        if (_global_fair_cutoff_dt is None
                or _strat_cutoff < _global_fair_cutoff_dt):
            _global_fair_cutoff_dt = _strat_cutoff

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

        # 2026-05-27 (revised): apples-to-apples cutoff = "the latest bar
        # both sources have actually processed". Use MAX(entry_fill_ts,
        # exit_fill_ts) for backtest — that's the latest bar the backtest
        # has produced output for. Do NOT use created_at: the data-worker
        # writes trades for OLD bars in real-time (15-min LAG_MINUTES
        # convention), so created_at can be fresh-now while the strategy's
        # processed-through point is 15+ min old. Used the wrong field on
        # first cut → fair filter didn't actually filter anything.
        last_bt_processed: Optional[datetime] = None
        for candidate in (last_entry_dt, last_exit_dt):
            if candidate is not None and (
                    last_bt_processed is None
                    or candidate > last_bt_processed):
                last_bt_processed = candidate

        fair_cutoff_dt: Optional[datetime] = None
        if last_bt_processed is not None and last_alert_at is not None:
            fair_cutoff_dt = min(last_bt_processed, last_alert_at)
        elif last_bt_processed is not None:
            fair_cutoff_dt = last_bt_processed
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

        # 2026-05-29: GLOBAL fair counts (cross-strategy apples-to-apples).
        # Use the fleet-wide minimum cutoff so every strategy is filtered
        # against the same time. Eliminates the per-strategy batch-timing
        # asymmetry — when two strategies fire identical alerts and have
        # identical trades, their global_fair counts will match. When
        # global cutoff differs from per-strategy cutoff, the global is
        # always earlier (or equal), so global counts ≤ per-strategy
        # counts.
        if _global_fair_cutoff_dt is not None:
            (phantom_count_global_fair,
             missed_count_global_fair,
             paired_count_global_fair) = _pair_phantom_missed(
                edge_isos, alert_unix_list,
                upper_bound_unix=_global_fair_cutoff_dt.timestamp())
        else:
            phantom_count_global_fair = phantom_count_fair
            missed_count_global_fair = missed_count_fair
            paired_count_global_fair = paired_count_fair

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
            # 2026-05-29 — GLOBAL fair counts. Use these for cross-strategy
            # comparison; the per-strategy `*_fair` counts are still useful
            # when looking at one strategy in isolation. Both share the
            # same underlying pairing logic, only the upper-bound cutoff
            # differs.
            "phantom_count_global_fair": phantom_count_global_fair,
            "missed_count_global_fair": missed_count_global_fair,
            "paired_count_global_fair": paired_count_global_fair,
            "global_fair_cutoff_ts": (
                _global_fair_cutoff_dt.isoformat()
                if _global_fair_cutoff_dt else None),
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
_CLASS_CROSS_EXEC_TYPE_MISMATCH = "cross_exec_type_mismatch"

# Below this, a within-window pair is treated as the "same event" — even
# a 1s delta on bar-close exits is normal alert dispatch latency. Above
# this, a pair where the live alert and the backtest trade disagree on
# exec_type (C vs L) is flagged as cross_exec_type_mismatch instead of
# being silently filtered as "paired = OK". See SOP §"Proposed new
# buckets" — promoted to the auto-classifier 2026-06-04.
_PAIR_TIGHT_THRESHOLD_SEC = 2.0


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
    strategy_id: Optional[int] = Query(
        None,
        description="If set, only include events for this strategy ID. "
                    "Lets the UI (and an investigation script) drill into "
                    "one sid at a time. 2026-05-27."),
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
    # When strategy_id filter is set, scope to just that strategy — saves
    # work on the trade/alert pulls below.
    strat_query = c.table("strategies").select(
        "id,name,symbol,timeframe,direction,config,data_refreshed_at"
    )
    if strategy_id is not None:
        strat_query = strat_query.eq("id", strategy_id)
    strat_resp = strat_query.execute()
    strats_by_id: Dict[int, Dict[str, Any]] = {}
    for s in (strat_resp.data or []):
        strats_by_id[s.get("id")] = s

    # Per-strategy "latest bar backtest has processed" — MAX(entry/exit
    # fill_ts) — and "last alert" for the apples-to-apples cap. NOT
    # created_at: the data-worker writes lagged trades in real-time, so
    # created_at is misleadingly fresh. See the matching fix in the
    # overview endpoint above for full context.
    last_bt_per_sid: Dict[int, datetime] = {}
    if apples_to_apples:
        try:
            bt_resp = c.table("trades").select(
                "strategy_id,entry_fill_ts,exit_fill_ts"
            ).like("data_source", "backtest_%").execute()
            for t in (bt_resp.data or []):
                sid_t = t.get("strategy_id")
                if sid_t is None:
                    continue
                for col in ("entry_fill_ts", "exit_fill_ts"):
                    ts_t = t.get(col)
                    if not ts_t:
                        continue
                    dt = _parse_iso(ts_t)
                    if dt is None:
                        continue
                    cur = last_bt_per_sid.get(sid_t)
                    if cur is None or dt > cur:
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
    trade_query = c.table("trades").select(
        "id,strategy_id,entry_fill_ts,exit_fill_ts,exit_reason,"
        "data,data_source"
    ).like("data_source", "backtest_%").gte(
        "entry_fill_ts", window_start_iso).order("strategy_id").order("id")
    if strategy_id is not None:
        trade_query = trade_query.eq("strategy_id", strategy_id)
    # Window-scoped but still paginate — the divergence backlog can pull
    # multi-day windows; a 7d window on a busy fleet exceeds the silent
    # 50k cap (see _paginated_fetch).
    trades_in_window: List[Dict[str, Any]] = _paginated_fetch(trade_query)

    # Pull alerts within window.
    alert_query = c.table("alerts").select(
        "id,strategy_id,fill_ts,trigger_ts,event_type"
    ).gte("timestamp", window_start_iso).lte(
        "timestamp", window_end_iso)
    if strategy_id is not None:
        alert_query = alert_query.eq("strategy_id", strategy_id)
    alert_resp = alert_query.execute()
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

    def _detect_pair_mismatch(
        alert: Dict[str, Any],
        trade: Dict[str, Any],
        edge: str,
        delta_s: float,
    ) -> tuple[bool, str]:
        """Decide whether a paired (alert, trade) tuple is actually a
        cross_exec_type_mismatch and should be surfaced rather than
        silently filtered as "paired = OK".

        Catches two directions of the bug:
          (a) live exit alert is C-type signal (utv4_bear_flip etc.) at
              bar close, but backtest exited intra-bar via stop_loss /
              target (L-type). The pair times match within the 60s
              window but represent different events.
          (b) live exit alert is L-type stop_loss, but the backtest exit
              was a signal trigger (non-_ib C-type) — the inverse case.

        Same logic on the entry side: alert.exec_type vs the trade's
        entry_exec_type.

        Returns (is_mismatch, reason).
        """
        if delta_s <= _PAIR_TIGHT_THRESHOLD_SEC:
            # Within 2s is normal alert dispatch latency on a same-event pair.
            return False, ""

        alert_ex = (alert.get("exec_type") or "").upper()
        if not alert_ex:
            return False, ""

        if edge == "exit":
            # Trade-side: stop_loss / target are L-type (intra-bar level
            # cross). Anything else is a C-type signal exit.
            er = (trade.get("exit_reason") or "").lower()
            trade_implied = "L" if er in ("stop_loss", "stop", "target") else "C"
            alert_trig = (alert.get("trigger_id") or "")
            # If the alert came in as L-type stop, the implied "alert" type
            # also resolves to L regardless of exec_type spelling.
            if alert_trig in ("stop_loss", "stop", "target"):
                alert_implied = "L"
            else:
                alert_implied = alert_ex
            if alert_implied != trade_implied:
                return True, (
                    f"exit pair Δ={delta_s:.1f}s — live={alert_implied} "
                    f"({alert_trig or alert_ex}) vs "
                    f"backtest={trade_implied} ({er or 'signal'})")
        elif edge == "entry":
            # Trade-side: read stored entry_exec_type. Default to C
            # (most strategies are C-on-entry).
            trade_data = trade.get("data") or {}
            if not isinstance(trade_data, dict):
                trade_data = {}
            trade_entry_ex = (
                trade.get("entry_exec_type")
                or trade_data.get("entry_exec_type")
                or "C"
            ).upper()
            if alert_ex != trade_entry_ex:
                return True, (
                    f"entry pair Δ={delta_s:.1f}s — live={alert_ex} vs "
                    f"backtest={trade_entry_ex}")
        return False, ""

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
        # `paired_tuples` keeps the (alert_idx, edge_idx, delta_s) for
        # post-pairing mismatch detection. Without it the pair scan loses
        # the delta and we'd have to recompute.
        i = j = 0
        paired_edge_idx = set()
        paired_alert_idx = set()
        paired_tuples: List[tuple[int, int, float]] = []
        while i < len(edges) and j < len(alerts_sorted):
            diff = alerts_sorted[j][0] - edges[i][0]
            if abs(diff) <= _DIVERGENCE_TOLERANCE_SEC:
                paired_edge_idx.add(i)
                paired_alert_idx.add(j)
                paired_tuples.append((j, i, abs(diff)))
                i += 1
                j += 1
            elif diff < 0:
                j += 1
            else:
                i += 1

        # cross_exec_type_mismatch detection (2026-06-04). For each
        # paired (alert, edge) tuple, check whether the live exec_type
        # disagrees with the backtest's implied exec_type. These were
        # previously filtered out as "paired = OK" but they're real
        # divergences (different exit triggers ~5-30s apart inflate
        # Layer 3 fill_ts delta without showing up as phantom/missed).
        for (al_idx, edge_idx, delta_s) in paired_tuples:
            ts, edge, trade = edges[edge_idx]
            _, alert = alerts_sorted[al_idx]
            is_mm, mm_reason = _detect_pair_mismatch(
                alert, trade, edge, delta_s)
            if not is_mm:
                continue
            cls = _CLASS_CROSS_EXEC_TYPE_MISMATCH
            if only_needs_investigation and cls != _CLASS_NEEDS_INVESTIGATION:
                continue
            out_rows.append({
                "event_type": "mismatch",
                "strategy_id": sid,
                "strategy_name": strat.get("name"),
                "symbol": strat.get("symbol"),
                "timeframe": strat.get("timeframe"),
                "direction": strat.get("direction"),
                "edge": edge,
                "timestamp": datetime.fromtimestamp(
                    ts, tz=timezone.utc).isoformat(),
                "trade_id": trade.get("id"),
                "alert_id": alert.get("id"),
                "exit_reason": trade.get("exit_reason"),
                "exit_trigger": (trade.get("exit_trigger")
                                 or (trade.get("data") or {}).get("exit_trigger")),
                "classification": cls,
                "classification_reason": mm_reason,
            })

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
