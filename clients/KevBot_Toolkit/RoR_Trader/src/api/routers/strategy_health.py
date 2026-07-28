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
import os
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


def _hb_supersedes_snapshot() -> bool:
    """#157 kill switch: when ON, a FRESH shadow heartbeat suppresses the
    `snapshot_stale` / `snapshot_missing` red flag.

    `config.engine_snapshot_at` is written ONLY by the legacy data_worker
    Phase-2 flush (data_worker.py:_flush_all_snapshots). When the fleet moved
    to the M-RS5a resident/shadow engine (~2026-07-21) that flush stopped
    advancing for resident strategies, so `snapshot_stale` tripped fleet-wide
    ("0 healthy") even though the engine is provably current — it just
    publishes freshness via `shadow_heartbeats` now (W2-0). A fresh heartbeat
    (hb_fresh) IS that proof, so it supersedes the deprecated snapshot signal.
    Genuine stalls still fire: if the heartbeat is ALSO stale/absent the
    snapshot flag stands. The staleness threshold is deliberately NOT widened.

    Per-call env read so the flag flips on Railway without a redeploy. Default
    OFF preserves the pre-fix behaviour (kill switch).
    """
    return os.environ.get(
        "RORT_HEALTH_HB_SUPERSEDES_SNAPSHOT", "").strip().lower() in (
        "1", "true", "yes", "on")


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
# Appends write backtest trades only up to now − this lag (settle window),
# but stamp last_recompute_until_ts at wall-clock "now". So the TBD coverage
# cutoff = last_recompute_until − this, i.e. the real BT data edge. Mirrors
# forward_test_service._ALGO_HISTORY_LAG_MINUTES (15 min).
_COVERAGE_LAG_SEC = 15 * 60
# W2-0 shadow heartbeat (2026-07-06): the resident shadow engine publishes its
# TRUE settled coverage boundary per strategy to `shadow_heartbeats`
# (shadow_manager.HEARTBEAT). A heartbeat whose updated_at is within this
# freshness window is authoritative coverage — its current_through IS the
# settled data edge (no lag subtraction needed), honest even when the model is
# quiet and Last BT ages. Stale/missing → legacy recompute-stamp fallback.
_HEARTBEAT_FRESH_SEC = 30 * 60
_WINDOW_HOURS_DEFAULT = 24
_WINDOW_HOURS_MIN = 1
_WINDOW_HOURS_MAX = 168                     # 7 days max


def _snapshot_red_flag(snapshot_at: datetime | None,
                       snapshot_age: int | None,
                       hb_fresh: bool,
                       hb_supersedes: bool) -> str | None:
    """The engine-freshness red flag for one strategy, or None if fresh.

    #157: when `hb_supersedes` is on, a FRESH shadow heartbeat (`hb_fresh`)
    proves the resident engine is current and supersedes the legacy
    `engine_snapshot_at` signal — which stopped advancing for the resident
    fleet ~2026-07-21 (see `_hb_supersedes_snapshot`). Otherwise fall back to
    the snapshot age: missing → 'snapshot_missing', older than
    `_STALE_SNAPSHOT_SEC` → 'snapshot_stale'. A genuine stall (heartbeat ALSO
    stale/absent) still fires. The threshold is deliberately NOT widened.
    """
    if hb_supersedes and hb_fresh:
        return None
    if snapshot_at is None:
        return "snapshot_missing"
    if snapshot_age is not None and snapshot_age > _STALE_SNAPSHOT_SEC:
        return "snapshot_stale"
    return None


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


# ── OFFLINE HARNESS HOOK (2026-07-15, Plan_Measurement_Trust Phase A) ──────────
# When set to {strategy_id: [fill_unix, ...]}, get_strategy_health scores those
# fills IN PLACE OF the real alerts for those sids — using the SAME backtest
# edges, coverage cutoff, pairing and TBD-exclusion. This lets the replay harness
# compute a REPLAY ceiling `combined_pct` that is byte-comparable to the dashboard
# LIVE number (same code path), instead of a divergent re-implementation.
# ALWAYS None in the deployed API (never touched by any request path) → the live
# dashboard and HTTP route are byte-identical. Set/cleared only by an offline
# caller (replay_harness). Not thread-safe by design — offline single-thread use.
_REPLAY_FILLS_OVERRIDE: "Optional[Dict[int, List[float]]]" = None
# Companion offline hook: scope the (otherwise fleet-wide) strategies/trades/
# alerts fetches to just these sids. The fleet-wide queries time out under load
# when called repeatedly offline (the harness); scoping to a handful makes them
# light. ALWAYS None in the deployed API → the page fetches the whole fleet as
# before. Offline-only.
_ONLY_SIDS: "Optional[List[int]]" = None


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
    tolerance_seconds: float = Query(
        _DIVERGENCE_TOLERANCE_SEC,
        ge=0.0,
        le=300.0,
        description="Alert↔backtest match tolerance (seconds) for "
                    "phantom/missed pairing. Default 60 (no behavior change). "
                    "Use 5 or 10 for a stricter fidelity view.",
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
            "bt_current_through": "..." | null,   # W2-0 shadow heartbeat
            "bt_heartbeat_at": "..." | null,
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
    _strat_q = c.table("strategies").select(
        "id,user_id,name,symbol,direction,timeframe,strategy_origin,"
        "forward_testing,forward_test_start,"
        "data_source,data_refreshed_at,"
        "kpis_computed_at,kpis_stale_since,"
        "parity_status,discrepancies,discrepancies_dismissed_at,"
        "config,updated_at,created_at"
    ).order("id")
    if _ONLY_SIDS is not None:
        _strat_q = _strat_q.in_("id", _ONLY_SIDS)
    strat_resp = _strat_q.execute()
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
    if _ONLY_SIDS is not None:
        trade_query = trade_query.in_("strategy_id", _ONLY_SIDS)
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
        _la_q = c.table("alerts").select(
            "strategy_id,timestamp"
        ).gte("timestamp", thirty_d_ago).order("timestamp", desc=True)
        if _ONLY_SIDS is not None:
            _la_q = _la_q.in_("strategy_id", _ONLY_SIDS)
        last_alert_resp = _la_q.execute()
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

    # W2-0 (2026-07-06): shadow-engine heartbeats — one fetch per request
    # (~70 rows). sid → (current_through, updated_at) as datetimes. A FRESH
    # heartbeat supersedes the recompute-stamp coverage derivation below;
    # see _HEARTBEAT_FRESH_SEC.
    heartbeats: Dict[int, tuple] = {}
    try:
        hb_resp = c.table("shadow_heartbeats").select("*").execute()
        for hb in (hb_resp.data or []):
            hb_sid = hb.get("strategy_id")
            hb_ct = _parse_iso(hb.get("current_through"))
            hb_ua = _parse_iso(hb.get("updated_at"))
            if hb_sid is None or hb_ct is None:
                continue
            heartbeats[hb_sid] = (hb_ct, hb_ua)
    except Exception as e:
        logger.warning("[health] shadow_heartbeats load failed: %s — "
                       "coverage falls back to recompute stamps", e)

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
    _al_q = c.table("alerts").select(
        "strategy_id,fill_ts,trigger_ts,event_type"
    ).gte("timestamp", window_cutoff_iso).lte("timestamp", window_end_iso)
    if _ONLY_SIDS is not None:
        _al_q = _al_q.in_("strategy_id", _ONLY_SIDS)
    alert_resp = _al_q.execute()
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
                              upper_bound_unix: Optional[float] = None,
                              coverage_unix: Optional[float] = None):
        """Greedy pairing within ±tolerance. Returns (phantom, missed, paired, tbd).

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
        # Track the timestamps of UNPAIRED alerts so we can split them into
        # phantom (backtest had a chance — alert ≤ coverage) vs TBD (backtest
        # lane hasn't reached it yet — alert > coverage), per By-Hour.
        i = j = 0
        paired_edges = 0
        paired_alerts = 0
        unpaired_alert_ts: List[float] = []
        while i < len(edges) and j < len(alerts_sorted):
            diff = alerts_sorted[j] - edges[i]
            if abs(diff) <= tolerance_seconds:
                paired_edges += 1
                paired_alerts += 1
                i += 1
                j += 1
            elif diff < 0:
                unpaired_alert_ts.append(alerts_sorted[j])
                j += 1   # alert too early — it's a phantom/TBD, skip ahead
            else:
                i += 1   # edge too early — it's missed, skip ahead
        unpaired_alert_ts.extend(alerts_sorted[j:])  # trailing unmatched alerts
        phantom = len(alerts_sorted) - paired_alerts
        missed = len(edges) - paired_edges
        tbd = 0
        if coverage_unix is not None:
            # Unpaired alerts NEWER than the backtest-lane coverage aren't
            # phantoms — the reference doesn't exist yet (convert on next
            # append). Split them out of phantom into TBD. (Per-strategy
            # coverage — NOT a fleet-wide min — so one quiet strategy can't
            # over-filter the rest. This is what makes TBD apples-to-apples
            # naturally and supersedes the old global-fair cutoff.)
            tbd = sum(1 for a in unpaired_alert_ts if a > coverage_unix)
            phantom -= tbd
        return phantom, missed, paired_alerts, tbd

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

    # #157: read the heartbeat-supersedes-snapshot kill switch once per request.
    _hb_supersedes = _hb_supersedes_snapshot()

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
        # Harness hook (Phase A): score injected replay fills in place of the
        # real alerts for this sid, when the offline override is active. Prod =
        # None → always the real alerts. Same edges/coverage/pairing either way.
        alert_unix_list = (
            _REPLAY_FILLS_OVERRIDE.get(sid, [])
            if _REPLAY_FILLS_OVERRIDE is not None
            else alerts_per_sid.get(sid, []))
        phantom_count, missed_count, paired_count, _ = _pair_phantom_missed(
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
            phantom_count_fair, missed_count_fair, paired_count_fair, _ = (
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
             paired_count_global_fair, _) = _pair_phantom_missed(
                edge_isos, alert_unix_list,
                upper_bound_unix=_global_fair_cutoff_dt.timestamp())
        else:
            phantom_count_global_fair = phantom_count_fair
            missed_count_global_fair = missed_count_fair
            paired_count_global_fair = paired_count_fair

        # ── Coverage-classified counts + Combined% + TBD (2026-06-16) ─────
        # The PRIMARY per-strategy numbers, matching By-Hour/By-Deploy:
        # pair alerts↔BT edges, then split UNPAIRED alerts by this strategy's
        # OWN backtest coverage — alert newer than coverage → TBD (lane hasn't
        # caught up, converts next append), else phantom. Combined% =
        # paired/(paired+phantom+missed), TBD excluded.
        #
        # Coverage = last_recompute_until_ts (when the lane last recomputed),
        # falling back to last_bt_processed. PER-STRATEGY — NOT the old
        # fleet-wide global-fair MIN, which one quiet strategy (e.g. 305
        # stopped firing at midday) dragged back far enough to over-filter
        # every other strategy's recent events into TBD (blank Combined%).
        # TBD makes the comparison apples-to-apples naturally, so the
        # global-fair cutoff is retired from the display (2026-06-16).
        # Coverage = the backtest-lane DATA EDGE, not the wall-clock recompute
        # stamp. last_recompute_until_ts is stamped at append "now", but the
        # append only WRITES trades up to now − LAG (15-min settle), so the
        # last ~15 min of alerts have no BT reference yet → they're TBD, not
        # phantoms. Subtract the lag so the trailing lag window classifies as
        # TBD (matches the append cutoff). Fallback to last_bt_processed.
        #
        # W2-0 (2026-07-06): PRECEDENCE — a FRESH shadow-engine heartbeat wins.
        # current_through is the engine's true settled boundary (already the
        # data edge — no lag subtraction), so a quiet-but-current strategy
        # reads as covered instead of stale. Stale/missing heartbeat → the
        # legacy derivation above, unchanged.
        hb_row = heartbeats.get(sid)
        hb_current_through = hb_row[0] if hb_row else None
        hb_updated_at = hb_row[1] if hb_row else None
        hb_fresh = (
            hb_updated_at is not None
            and (now - hb_updated_at).total_seconds() <= _HEARTBEAT_FRESH_SEC)
        if hb_fresh and hb_current_through is not None:
            _cov_dt = hb_current_through
        elif last_recompute_until is not None:
            _cov_dt = last_recompute_until - timedelta(
                seconds=_COVERAGE_LAG_SEC)
            if last_bt_processed is not None and last_bt_processed > _cov_dt:
                _cov_dt = last_bt_processed  # data edge is actually later
        else:
            _cov_dt = last_bt_processed
        _cov_unix = _cov_dt.timestamp() if _cov_dt is not None else None
        (phantom_count_cov, missed_count_cov,
         paired_count_cov, tbd_count) = _pair_phantom_missed(
            edge_isos, alert_unix_list, coverage_unix=_cov_unix)
        _denom_cov = paired_count_cov + phantom_count_cov + missed_count_cov
        combined_pct = (round(100.0 * paired_count_cov / _denom_cov, 1)
                        if _denom_cov > 0 else None)

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
            #
            # #157: a FRESH shadow heartbeat proves the resident engine is
            # current, so it supersedes the legacy engine_snapshot_at signal
            # (deprecated for the resident fleet since ~2026-07-21 — see
            # _snapshot_red_flag / _hb_supersedes_snapshot).
            _snap_flag = _snapshot_red_flag(
                snapshot_at, snapshot_age, hb_fresh, _hb_supersedes)
            if _snap_flag is not None:
                red_flags.append(_snap_flag)

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
            # W2-0 (2026-07-06) — shadow-engine heartbeat: the engine's true
            # settled coverage boundary + when it last reported. Null = no
            # heartbeat on file (engine not resident for this strategy).
            "bt_current_through": (hb_current_through.isoformat()
                                   if hb_current_through else None),
            "bt_heartbeat_at": (hb_updated_at.isoformat()
                                if hb_updated_at else None),
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
            # 2026-06-16 — PRIMARY per-strategy coverage-classified counts
            # (matches By-Hour/By-Deploy). paired/phantom/missed split by this
            # strategy's own coverage; TBD = unpaired alerts newer than
            # coverage; combined% = paired/(paired+phantom+missed). The V1
            # table reads THESE (the raw/fair/global_fair fields above are
            # retained for back-compat but no longer drive the display).
            "combined_pct": combined_pct,
            "tbd_count": tbd_count,
            "paired_count_cov": paired_count_cov,
            "phantom_count_cov": phantom_count_cov,
            "missed_count_cov": missed_count_cov,
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
_CLASS_PHANTOM_PRE_CORRECTION = "phantom_pre_correction"

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

    # Pull alerts within window. `would_fire_post_correction` (#57H) is
    # set asynchronously by ralph_engine.apply_rest_correction when a
    # REST splice fires on the alert's bar — used by _classify_phantom
    # to surface the phantom_pre_correction bucket separately from
    # needs_investigation.
    alert_query = c.table("alerts").select(
        "id,strategy_id,fill_ts,trigger_ts,event_type,"
        "would_fire_post_correction"
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
        # phantom_pre_correction (#57H): the alert's bar had a REST splice
        # and the post-correction indicator state would NOT have fired the
        # trigger. This is structural WS-vs-REST noise, not a real bug.
        # NULL = no splice fired on this bar (classification unchanged).
        # TRUE = trigger still fires post-correction (real divergence;
        # falls through to needs_investigation below).
        wf = alert.get("would_fire_post_correction")
        if wf is False:
            return _CLASS_PHANTOM_PRE_CORRECTION, (
                "Alert fired on WS-derived state; REST-corrected state "
                "would NOT have fired the same trigger (structural WS "
                "noise, not a backtest gap)")
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


# ─────────────────────────────────────────────────────────────────────
# By-Hour Pair-Rate Endpoint (2026-06-06)
# Backs the new Strategy Health "By Hour" tab. Mirrors the
# `_fleet_hourly_may28_jun5.py` script: pairs alerts vs BT trade
# events at ±5s, bins per hour per strategy, returns the cross-hour
# KPI summary + per-hour drill-down data.
# ─────────────────────────────────────────────────────────────────────

import bisect as _bisect
import re as _re


# Map trigger ID prefix → user-friendly pack code (the short form
# Kevin uses in conversation: utv4 / eppv3 / mhv2 / bb / etc.).
# These are extracted by stripping the trigger-specific suffix from
# the entry_trigger string. The list is exhaustive for the known
# packs; unknown prefixes fall through to whatever the entry_trigger
# string itself contains before the first '_'.
_TRIGGER_PREFIX_PACKS = {
    "utv4": "ut_bot_v4",
    "eppv3": "ema_pp_v3",
    "eppv4": "ema_pp_v4",
    "mhv2": "macd_histogram_v2",
    "mlv2": "macd_line_v2",
    "rv2": "rvol_v2",
    "bb": "bollinger_bands",
    "esv2": "ema_stack_v2",
    "sto": "stochastic_oscillator",
    "src": "sr_channels",
    "strat": "strat_assistant",
    "st": "supertrend",
    "sw123": "swing_1_2_3",
    "vwapv2": "vwap_v2",
    "rsi2": "rsi_zones_2",
    "rsi": "rsi_zones",
    "macd": "macd",
    "ema": "ema",
}


def _trigger_to_pack(trigger_id: str | None) -> str | None:
    """Map an entry/exit trigger id to its pack code."""
    if not trigger_id:
        return None
    # Try longest prefix first so 'utv4' matches before 'ut'
    for prefix in sorted(_TRIGGER_PREFIX_PACKS.keys(),
                          key=len, reverse=True):
        if trigger_id.startswith(prefix + "_") or trigger_id == prefix:
            return _TRIGGER_PREFIX_PACKS[prefix]
    # Fallback: use the substring before the first '_'
    head = trigger_id.split("_")[0]
    return head if head else None


# Confluence records are formatted "{TF}-{INTERPRETER}-{state}"
# e.g. "1M-EMA_STACK-SML". Map interpreter codes to pack codes.
_INTERPRETER_PACKS = {
    "EMA_STACK": "ema_stack_v2",
    "EMA_PP_V3": "ema_pp_v3",
    "EMA_PP_V4": "ema_pp_v4",
    "EPP_V3": "ema_pp_v3",
    "EPP_V4": "ema_pp_v4",
    "EPPV3": "ema_pp_v3",
    "EPPV4": "ema_pp_v4",
    "MACD_HISTOGRAM_V2": "macd_histogram_v2",
    "MACD_LINE_V2": "macd_line_v2",
    "MHV2": "macd_histogram_v2",
    "MLV2": "macd_line_v2",
    "RVOL_V2": "rvol_v2",
    "RV2": "rvol_v2",
    "BOLLINGER": "bollinger_bands",
    "BB": "bollinger_bands",
    "STOCHASTIC": "stochastic_oscillator",
    "STO": "stochastic_oscillator",
    "SR_CHANNELS": "sr_channels",
    "SRC": "sr_channels",
    "STRAT_ASSISTANT": "strat_assistant",
    "STRAT": "strat_assistant",
    "SUPERTREND": "supertrend",
    "ST": "supertrend",
    "SWING_1_2_3": "swing_1_2_3",
    "SW123": "swing_1_2_3",
    "VWAP_V2": "vwap_v2",
    "VWAPV2": "vwap_v2",
    "RSI_ZONES": "rsi_zones",
    "RSI_ZONES_2": "rsi_zones_2",
    "UTBOT_V4": "ut_bot_v4",
    "UT_BOT_V4": "ut_bot_v4",
    "UTV4": "ut_bot_v4",
}


def _confluence_to_packs(confluence: list | None) -> list[str]:
    """Map confluence records → unique list of pack codes."""
    if not confluence:
        return []
    found = set()
    for rec in confluence:
        if not isinstance(rec, str):
            continue
        parts = rec.split("-", 2)
        if len(parts) < 2:
            continue
        # Skip general-pack records (TF prefix 'GEN')
        if parts[0] == "GEN":
            continue
        interp = parts[1].strip().upper()
        if interp in _INTERPRETER_PACKS:
            found.add(_INTERPRETER_PACKS[interp])
        # Some records use lowercase / hybrid keys — try variants
        elif interp.replace("_", "").upper() in _INTERPRETER_PACKS:
            found.add(_INTERPRETER_PACKS[interp.replace("_", "").upper()])
    return sorted(found)


def _build_strategy_metadata(c, cohort: list[int]) -> dict:
    """For each sid in cohort, return:
      {sid: {name, symbol, timeframe, session, origin, trigger_pack,
             gate_packs[], strategy_type}}
    strategy_type: 'trigger' if entry_trigger pack and no other gate
                   packs in confluence; 'gate' if confluence has at
                   least one pack distinct from the entry pack;
                   'both' otherwise.
    timeframe and strategy_origin live as top-level columns on the
    strategies table — NOT inside config.
    """
    if not cohort:
        return {}
    meta = {}
    # Batch in groups of 500
    for i in range(0, len(cohort), 500):
        batch = cohort[i:i + 500]
        r = (c.table("strategies")
             .select("id,name,symbol,timeframe,direction,strategy_origin,config")
             .in_("id", batch).execute())
        for row in r.data or []:
            sid = row["id"]
            cfg = row.get("config") or {}
            if isinstance(cfg, str):
                try:
                    import json as _json2
                    cfg = _json2.loads(cfg)
                except (TypeError, ValueError):
                    cfg = {}
            entry_trig = (cfg.get("entry_trigger")
                          or cfg.get("entry_trigger_confluence_id"))
            trigger_pack = _trigger_to_pack(entry_trig)
            gate_packs = _confluence_to_packs(cfg.get("confluence") or [])
            # Exclude trigger pack from gate set
            gate_packs = [g for g in gate_packs if g != trigger_pack]
            # Classify
            name = row.get("name") or ""
            name_lc = name.lower()
            # PACKTEST naming pattern is reliable; fall back to confluence
            if "· gate" in name_lc or "- gate" in name_lc:
                stype = "gate"
            elif "· trigger" in name_lc or "- trigger" in name_lc:
                stype = "trigger"
            elif gate_packs:
                stype = "gate"
            else:
                stype = "trigger"
            # Prefer top-level columns; fall back to config for legacy
            tf = row.get("timeframe") or cfg.get("timeframe")
            meta[sid] = {
                "strategy_id": sid,
                "name": name or None,
                "symbol": row.get("symbol"),
                "timeframe": tf,
                "session": cfg.get("trading_session"),
                "direction": row.get("direction"),
                "origin": row.get("strategy_origin"),
                "trigger_pack": trigger_pack,
                "gate_packs": gate_packs,
                "strategy_type": stype,
            }
    return meta


def _pair_events_5s(a_ts: list[float],
                    b_ts: list[float],
                    window_s: float = 5.0) -> tuple[list[bool], list[bool]]:
    """Symmetric ±window pairing of two sorted timestamp arrays.
    Same algorithm as src/_fleet_hourly_may28_jun5.py."""
    a_paired = [False] * len(a_ts)
    b_paired = [False] * len(b_ts)
    if not a_ts or not b_ts:
        return a_paired, b_paired
    for i, t in enumerate(a_ts):
        idx = _bisect.bisect_left(b_ts, t)
        best = None
        best_delta = window_s + 1
        for j in (idx - 1, idx):
            if 0 <= j < len(b_ts):
                d = abs(b_ts[j] - t)
                if d < best_delta:
                    best_delta = d
                    best = j
        if best is not None and best_delta <= window_s:
            a_paired[i] = True
            b_paired[best] = True
    for j, t in enumerate(b_ts):
        if b_paired[j]:
            continue
        idx = _bisect.bisect_left(a_ts, t)
        for i in (idx - 1, idx):
            if 0 <= i < len(a_ts):
                d = abs(a_ts[i] - t)
                if d <= window_s:
                    b_paired[j] = True
                    a_paired[i] = True
                    break
    return a_paired, b_paired


@router.get("/by-hour")
def get_strategy_health_by_hour(
    since: Optional[str] = Query(
        None,
        description="ISO timestamp (inclusive) for the start of the "
                    "window. Default: 7 days ago."),
    until: Optional[str] = Query(
        None,
        description="ISO timestamp (exclusive) for the end of the "
                    "window. Default: now."),
    pair_window_s: float = Query(
        5.0,
        description="Pair window in seconds for alert↔BT event matching. "
                    "Default 5s (tight). The earlier strategy_health "
                    "endpoint uses ±60s; this is intentionally stricter."),
    stability_tail_minutes: float = Query(
        15.0,
        description="Clip the upper bound to (now - stability_tail_minutes) "
                    "to exclude the BT-lane lag tail. The BT engine's "
                    "lag_minutes setting holds it back from the most-recent "
                    "bars; events in this window have artificially high "
                    "phantom counts. Set to 0 to disable (shows partial "
                    "current hour with low-confidence flag instead)."),
    user=Depends(get_current_user),  # noqa: ARG001 — admin guard
) -> dict:
    """Hour-by-hour combined pair-rate aggregation across the cohort.

    Returns:
      - hours: list of per-hour KPI rows (one per hour with activity)
      - strategies_by_hour: nested dict {hour: [per-strategy rows...]}
      - cohort: list of strategies covered

    Computation mirrors src/_fleet_hourly_may28_jun5.py:
      - For each (strategy, hour): count alerts, BT trade events, paired,
        phantom, missed via ±5s pairing
      - Combined % = paired / (paired + phantom + missed)
      - Per-hour KPIs: avg / min / max combined % across "ranked"
        strategies (alerts ≥ 1 AND bt_events ≥ 1)
      - Per-hour ranks: cohort-wide rank by combined % within that hour
        (rank 1 = cleanest)
      - Cross-hour ranks: each hour ranked across all hours by
        avg / max / min combined % (rank 1 = best)
    """
    from db import USE_DB, get_admin_client
    from collections import defaultdict
    if not USE_DB:
        return {"error": "DB not configured"}
    c = get_admin_client()

    now = datetime.now(timezone.utc)
    if since:
        since_dt = _parse_iso(since)
        if since_dt is None:
            return {"error": f"could not parse since={since!r}"}
    else:
        since_dt = now - timedelta(days=7)
    if until:
        until_dt = _parse_iso(until)
        if until_dt is None:
            return {"error": f"could not parse until={until!r}"}
    else:
        until_dt = now
    # Apply stability tail — clip upper bound to (now - stability_tail_minutes)
    # to exclude the BT-lane lag region where alerts have no possible BT pair
    # yet (artificially inflates phantom counts).
    if stability_tail_minutes > 0:
        stable_until_dt = now - timedelta(minutes=stability_tail_minutes)
        if until_dt > stable_until_dt:
            until_dt = stable_until_dt
    since_iso = since_dt.isoformat()
    until_iso = until_dt.isoformat()

    # Cohort = any strategy with at least one alert OR BT trade in window
    sids_set = set()
    s = 0
    while True:
        r = (c.table("alerts").select("strategy_id")
             .gte("fill_ts", since_iso).lt("fill_ts", until_iso)
             .range(s, s + 999).execute())
        chunk = r.data or []
        sids_set.update(a["strategy_id"] for a in chunk
                        if a.get("strategy_id"))
        if len(chunk) < 1000:
            break
        s += 1000
    s = 0
    while True:
        r = (c.table("trades").select("strategy_id")
             .like("data_source", "backtest_%")
             .gte("entry_fill_ts", since_iso)
             .lt("entry_fill_ts", until_iso)
             .range(s, s + 999).execute())
        chunk = r.data or []
        sids_set.update(t["strategy_id"] for t in chunk
                        if t.get("strategy_id"))
        if len(chunk) < 1000:
            break
        s += 1000
    cohort = sorted(sids_set)

    # Pull strategy names
    names = {}
    if cohort:
        # Batch in chunks of 500 sids
        for i in range(0, len(cohort), 500):
            batch = cohort[i:i + 500]
            r = (c.table("strategies").select("id,name,symbol,timeframe")
                 .in_("id", batch).execute())
            for row in r.data or []:
                names[row["id"]] = {
                    "name": row.get("name"),
                    "symbol": row.get("symbol"),
                    "timeframe": row.get("timeframe"),
                }

    # Per-strategy: pull alerts + trades, do pairing, hour-bin
    # hour_data[hour_str][sid] = {alerts, bt_events, paired, phantom, missed}
    hour_data: dict = defaultdict(
        lambda: defaultdict(
            lambda: {"alerts": 0, "bt_events": 0, "paired": 0,
                     "phantom": 0, "missed": 0, "tbd": 0}))

    # TBD classification (2026-06-12, Kevin's ask): an unpaired alert
    # NEWER than the strategy's backtest-lane coverage isn't a phantom —
    # the reference doesn't exist yet. Classify TBD, exclude from
    # combined%, converts automatically on the next append (computed at
    # query time). Coverage = config.last_recompute_until_ts, fallback
    # to the newest in-window BT edge.
    # TODO(W2-0): prefer a fresh shadow_heartbeats.current_through here, like
    # the overview endpoint does (scope was overview-only on 2026-07-06).
    _cov_end: dict = {}
    try:
        _cfg_rows = (c.table("strategies").select("id,config")
                     .in_("id", cohort).execute().data or [])
        for _r in _cfg_rows:
            _ts = (_r.get("config") or {}).get("last_recompute_until_ts")
            _dt = _parse_iso(_ts) if _ts else None
            if _dt is not None:
                _cov_end[_r["id"]] = _dt.timestamp()
    except Exception:
        pass

    def _parse_ts(s):
        if not s:
            return None
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    for sid in cohort:
        # Alerts
        alerts = []
        s_ = 0
        while True:
            r = (c.table("alerts").select("fill_ts")
                 .eq("strategy_id", sid)
                 .gte("fill_ts", since_iso).lt("fill_ts", until_iso)
                 .order("fill_ts").range(s_, s_ + 999).execute())
            chunk = r.data or []
            alerts.extend(chunk)
            if len(chunk) < 1000:
                break
            s_ += 1000
        # BT trades
        trades = []
        s_ = 0
        while True:
            r = (c.table("trades").select("entry_fill_ts,exit_fill_ts")
                 .eq("strategy_id", sid).like("data_source", "backtest_%")
                 .gte("entry_fill_ts", since_iso)
                 .lt("entry_fill_ts", until_iso)
                 .order("entry_fill_ts").range(s_, s_ + 999).execute())
            chunk = r.data or []
            trades.extend(chunk)
            if len(chunk) < 1000:
                break
            s_ += 1000

        # Build timestamp arrays
        a_dts = []
        a_orig = []
        for a in alerts:
            ts = _parse_ts(a.get("fill_ts"))
            if ts is None:
                continue
            a_dts.append(ts.timestamp())
            a_orig.append(a)
        b_events = []
        for t in trades:
            ets = _parse_ts(t.get("entry_fill_ts"))
            if ets is not None:
                b_events.append((ets.timestamp(), "entry"))
            xts = _parse_ts(t.get("exit_fill_ts"))
            if xts is not None:
                b_events.append((xts.timestamp(), "exit"))

        # Sort, pair, map back
        a_sorted_idx = sorted(range(len(a_dts)), key=lambda k: a_dts[k])
        a_sorted_ts = [a_dts[k] for k in a_sorted_idx]
        b_sorted = sorted(b_events, key=lambda x: x[0])
        b_sorted_ts = [x[0] for x in b_sorted]
        a_paired_s, b_paired_s = _pair_events_5s(
            a_sorted_ts, b_sorted_ts, pair_window_s)
        a_paired = [False] * len(a_dts)
        for k, si in enumerate(a_sorted_idx):
            a_paired[si] = a_paired_s[k]

        # Hour-bin
        for i, a in enumerate(a_orig):
            ts = _parse_ts(a.get("fill_ts"))
            if ts is None:
                continue
            h = ts.strftime("%Y-%m-%dT%H")
            bucket = hour_data[h][sid]
            bucket["alerts"] += 1
            if a_paired[i]:
                bucket["paired"] += 1
            else:
                _cov = _cov_end.get(sid)
                if b_sorted_ts:
                    _cov = max(_cov or 0, b_sorted_ts[-1])
                if _cov is None or ts.timestamp() > _cov:
                    bucket["tbd"] += 1
                else:
                    bucket["phantom"] += 1
        for k, (ts_sec, _kind) in enumerate(b_sorted):
            ts = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
            h = ts.strftime("%Y-%m-%dT%H")
            bucket = hour_data[h][sid]
            bucket["bt_events"] += 1
            if not b_paired_s[k]:
                bucket["missed"] += 1

    # Per-hour KPI aggregation + cross-hour ranking
    hours_sorted = sorted(hour_data.keys())
    per_hour_kpis = {}
    for h in hours_sorted:
        cpcts = []
        for sid, b in hour_data[h].items():
            if b["alerts"] >= 1 and b["bt_events"] >= 1:
                denom = b["paired"] + b["phantom"] + b["missed"]
                cpct = (100 * b["paired"] / denom) if denom > 0 else 0.0
                cpcts.append(cpct)
        if cpcts:
            per_hour_kpis[h] = {
                "n_ranked": len(cpcts),
                "n_active": len(hour_data[h]),
                "avg": sum(cpcts) / len(cpcts),
                "max": max(cpcts),
                "min": min(cpcts),
            }
        else:
            per_hour_kpis[h] = {
                "n_ranked": 0, "n_active": len(hour_data[h]),
                "avg": None, "max": None, "min": None,
            }

    # Cross-hour ranks
    with_data = [h for h in hours_sorted
                 if per_hour_kpis[h]["avg"] is not None]
    n_with = len(with_data)
    avg_rank = {h: i + 1 for i, h in enumerate(
        sorted(with_data, key=lambda h: -per_hour_kpis[h]["avg"]))}
    max_rank = {h: i + 1 for i, h in enumerate(
        sorted(with_data, key=lambda h: -per_hour_kpis[h]["max"]))}
    min_rank = {h: i + 1 for i, h in enumerate(
        sorted(with_data, key=lambda h: -per_hour_kpis[h]["min"]))}

    # Build response shape
    hours_out = []
    for h in hours_sorted:
        k = per_hour_kpis[h]
        hours_out.append({
            "hour_utc": h,
            "n_active": k["n_active"],
            "n_ranked": k["n_ranked"],
            "avg_combined_pct": (round(k["avg"], 1)
                                  if k["avg"] is not None else None),
            "max_combined_pct": (round(k["max"], 1)
                                  if k["max"] is not None else None),
            "min_combined_pct": (round(k["min"], 1)
                                  if k["min"] is not None else None),
            "avg_rank": avg_rank.get(h),
            "max_rank": max_rank.get(h),
            "min_rank": min_rank.get(h),
            "ranked_total": n_with,
        })

    # Per-hour, per-strategy rows (for drill-down)
    strategies_by_hour: dict = {}
    for h in hours_sorted:
        rows = []
        ranking_pool = []
        for sid, b in hour_data[h].items():
            denom = b["paired"] + b["phantom"] + b["missed"]
            cpct = (100 * b["paired"] / denom) if denom > 0 else 0.0
            a_denom = b["paired"] + b["phantom"]
            apr = (100 * b["paired"] / a_denom) if a_denom > 0 else 0.0
            rankable = (b["alerts"] >= 1 and b["bt_events"] >= 1)
            row = {
                "strategy_id": sid,
                "name": names.get(sid, {}).get("name"),
                "symbol": names.get(sid, {}).get("symbol"),
                "timeframe": names.get(sid, {}).get("timeframe"),
                "alerts": b["alerts"],
                "bt_events": b["bt_events"],
                "paired": b["paired"],
                "phantom": b["phantom"],
                "missed": b["missed"],
                "tbd": b.get("tbd", 0),
                "combined_pct": round(cpct, 1),
                "alert_pair_pct": round(apr, 1),
                "rankable": rankable,
            }
            rows.append(row)
            if rankable:
                ranking_pool.append((sid, cpct, b["alerts"]))
        # Rank within hour
        ranking_pool.sort(key=lambda t: (-t[1], -t[2]))
        rank_within = {sid: i + 1 for i, (sid, _, _) in
                       enumerate(ranking_pool)}
        for r in rows:
            r["rank"] = rank_within.get(r["strategy_id"])
            r["rank_total"] = len(ranking_pool)
        rows.sort(key=lambda x: x["strategy_id"])
        strategies_by_hour[h] = rows

    # Strategy metadata (packs, type, session, etc.) for client-side filtering
    strategy_meta = _build_strategy_metadata(c, cohort)

    return {
        "now": now.isoformat(),
        "since": since_iso,
        "until": until_iso,
        "pair_window_s": pair_window_s,
        "stability_tail_minutes": stability_tail_minutes,
        "cohort_size": len(cohort),
        "hours_count": len(hours_sorted),
        "hours_with_data": n_with,
        "hours": hours_out,
        "strategies_by_hour": strategies_by_hour,
        "cohort": [
            {"strategy_id": sid,
             "name": names.get(sid, {}).get("name"),
             "symbol": names.get(sid, {}).get("symbol"),
             "timeframe": names.get(sid, {}).get("timeframe")}
            for sid in cohort
        ],
        "strategy_metadata": [strategy_meta[sid] for sid in cohort
                              if sid in strategy_meta],
    }


# ─────────────────────────────────────────────────────────────────────
# By-Deploy Pair-Rate Endpoint (2026-06-06)
# Bucket pair-rate metrics by deploy window. Each push to dev triggers
# Railway redeploys (including worker → Ralph restart), so the window
# [commit_n.timestamp, commit_{n-1}.timestamp] is the period during
# which commit_n's code was the active build. The active window of
# the latest commit extends from its timestamp to now.
# ─────────────────────────────────────────────────────────────────────

import json as _json
import os as _os


def _load_deploy_history() -> list[dict]:
    """Read src/deploy_history.json (generated by _generate_deploy_history.py).

    Returns list of commits, newest first. Empty list if file missing
    or unreadable — the endpoint then degrades gracefully to "no
    deploys known."
    """
    path = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.dirname(
            _os.path.abspath(__file__)))),
        "deploy_history.json")
    try:
        with open(path) as f:
            payload = _json.load(f)
        return payload.get("commits") or []
    except (FileNotFoundError, _json.JSONDecodeError, OSError) as e:
        logger.warning("could not load deploy_history.json: %s", e)
        return []


@router.get("/by-deploy")
def get_strategy_health_by_deploy(
    since: Optional[str] = Query(
        None,
        description="ISO timestamp (inclusive) for the start of the "
                    "analysis window. Default: 2026-05-28."),
    until: Optional[str] = Query(
        None,
        description="ISO timestamp (exclusive) for end. Default: now."),
    pair_window_s: float = Query(
        5.0, description="Pair window in seconds. Default 5s."),
    stability_tail_minutes: float = Query(
        15.0,
        description="Clip each deploy window's upper bound to "
                    "(end - stability_tail_minutes) to exclude BT-lane lag. "
                    "Applied to the newest deploy (whose end is 'now'). Set "
                    "to 0 to disable."),
    warmup_minutes: float = Query(
        30.0,
        description="Clip each deploy window's lower bound to "
                    "(deploy_ts + warmup_minutes) to exclude worker warmup "
                    "+ snapshot resume settling. Stateful stops (swing/ATR) "
                    "need ~30 bars to populate buffers before generating "
                    "trades. Set to 0 to disable."),
    user=Depends(get_current_user),  # noqa: ARG001
) -> dict:
    """Pair-rate aggregation bucketed by deploy window.

    For each known commit (read from src/deploy_history.json), the
    "active window" is [commit.timestamp, next_newer_commit.timestamp].
    For the newest commit, it's [commit.timestamp, now].

    Returns one row per deploy window with:
      - sha, subject, timestamp
      - window_start / window_end
      - alerts / bt_events / paired / phantom / missed
      - combined_pct (paired / (paired + phantom + missed))
      - n_active_strategies (those with any alert OR BT event)
      - n_ranked_strategies (those with both alerts >= 1 AND BT >= 1)
      - avg_combined_pct across ranked strategies (cohort average)
    """
    from db import USE_DB, get_admin_client
    from collections import defaultdict
    if not USE_DB:
        return {"error": "DB not configured"}
    c = get_admin_client()

    now = datetime.now(timezone.utc)
    if since:
        since_dt = _parse_iso(since)
        if since_dt is None:
            return {"error": f"could not parse since={since!r}"}
    else:
        since_dt = datetime(2026, 5, 28, tzinfo=timezone.utc)
    if until:
        until_dt = _parse_iso(until)
        if until_dt is None:
            return {"error": f"could not parse until={until!r}"}
    else:
        until_dt = now

    commits = _load_deploy_history()
    # Sort ascending so windows are well-defined: window n is
    # [commits[n].timestamp, commits[n+1].timestamp) — for the last
    # commit, the end is "until_dt".
    commits_asc = sorted(commits, key=lambda c_: c_["timestamp_iso"])

    # Filter commits to those whose timestamp falls within the
    # requested window. We also include the commit immediately
    # BEFORE since_dt (if any) so we know what was deployed at
    # since_dt — that commit's window straddles the start.
    in_window = []
    for c_ in commits_asc:
        ts = _parse_iso(c_["timestamp_iso"])
        if ts is None:
            continue
        if ts >= since_dt and ts < until_dt:
            in_window.append(c_)

    # Build deploy windows
    windows = []
    for i, c_ in enumerate(in_window):
        deploy_dt = _parse_iso(c_["timestamp_iso"])
        if i + 1 < len(in_window):
            raw_w_end = _parse_iso(in_window[i + 1]["timestamp_iso"])
        else:
            raw_w_end = until_dt
        if deploy_dt is None or raw_w_end is None or raw_w_end <= deploy_dt:
            continue
        # Apply warmup clipping: skip first warmup_minutes after deploy
        # (worker warmup + snapshot resume + stateful-stop buffer settle)
        w_start = deploy_dt + timedelta(minutes=warmup_minutes)
        # Apply stability tail clipping to the latest window only (where
        # raw_w_end ~= now). Older windows ended at the next deploy, which
        # is a hard boundary that BT had time to catch up to.
        if i + 1 < len(in_window):
            w_end = raw_w_end
        else:
            if stability_tail_minutes > 0:
                stable_until = now - timedelta(minutes=stability_tail_minutes)
                w_end = min(raw_w_end, stable_until)
            else:
                w_end = raw_w_end
        if w_end <= w_start:
            # Warmup consumed the whole window — record with zero stable
            # duration so the UI can flag "no stable data" rather than
            # silently dropping the deploy.
            windows.append({
                "sha": c_["sha"],
                "subject": c_["subject"],
                "timestamp_iso": c_["timestamp_iso"],
                "window_start": w_start.isoformat(),
                "window_end": w_start.isoformat(),
                "duration_seconds": 0.0,
                "stable_duration_seconds": 0.0,
                "raw_duration_seconds": (raw_w_end - deploy_dt).total_seconds(),
                "low_confidence": True,
            })
            continue
        stable_duration = (w_end - w_start).total_seconds()
        windows.append({
            "sha": c_["sha"],
            "subject": c_["subject"],
            "timestamp_iso": c_["timestamp_iso"],
            "window_start": w_start.isoformat(),
            "window_end": w_end.isoformat(),
            "duration_seconds": stable_duration,
            "stable_duration_seconds": stable_duration,
            "raw_duration_seconds": (raw_w_end - deploy_dt).total_seconds(),
            # Flag low-confidence when < 2 hours of stable data
            "low_confidence": stable_duration < 2 * 3600,
        })

    if not windows:
        return {
            "now": now.isoformat(),
            "since": since_dt.isoformat(),
            "until": until_dt.isoformat(),
            "pair_window_s": pair_window_s,
            "stability_tail_minutes": stability_tail_minutes,
            "warmup_minutes": warmup_minutes,
            "deploy_count": 0,
            "deploys": [],
            "note": "no deploys recorded in window (deploy_history.json "
                    "may be missing or out of date)",
        }

    # For each window, compute aggregate pair-rate stats. We do this
    # by fetching alerts + BT events for the entire range, then
    # bucketing into windows in Python — much faster than per-window
    # DB queries when there are dozens of windows.
    range_start_iso = windows[0]["window_start"]
    range_end_iso = windows[-1]["window_end"]

    # Build cohort = any strategy with activity in range
    sids_set = set()
    s = 0
    while True:
        r = (c.table("alerts").select("strategy_id,fill_ts")
             .gte("fill_ts", range_start_iso)
             .lt("fill_ts", range_end_iso)
             .range(s, s + 999).execute())
        chunk = r.data or []
        for a in chunk:
            if a.get("strategy_id"):
                sids_set.add(a["strategy_id"])
        if len(chunk) < 1000:
            break
        s += 1000
    s = 0
    while True:
        r = (c.table("trades").select("strategy_id,entry_fill_ts")
             .like("data_source", "backtest_%")
             .gte("entry_fill_ts", range_start_iso)
             .lt("entry_fill_ts", range_end_iso)
             .range(s, s + 999).execute())
        chunk = r.data or []
        for t in chunk:
            if t.get("strategy_id"):
                sids_set.add(t["strategy_id"])
        if len(chunk) < 1000:
            break
        s += 1000
    cohort = sorted(sids_set)

    # Per-strategy: pull data, do pairing, bucket events into windows
    def _parse_ts_local(s_):
        if not s_:
            return None
        try:
            return datetime.fromisoformat(s_.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    # window_data[window_idx][sid] = {alerts, bt_events, paired, phantom, missed}
    window_data: dict = defaultdict(
        lambda: defaultdict(lambda: {
            "alerts": 0, "bt_events": 0, "paired": 0,
            "phantom": 0, "missed": 0, "tbd": 0,
        }))

    # TBD classification (2026-06-12, mirrors by-hour): an unpaired alert
    # NEWER than the strategy's backtest-lane coverage isn't a phantom —
    # the reference doesn't exist yet. Excluded from combined%; converts
    # on the next append (computed at query time).
    # TODO(W2-0): prefer a fresh shadow_heartbeats.current_through here, like
    # the overview endpoint does (scope was overview-only on 2026-07-06).
    _cov_end: dict = {}
    try:
        _cfg_rows = (c.table("strategies").select("id,config")
                     .in_("id", cohort).execute().data or [])
        for _r in _cfg_rows:
            _ts = (_r.get("config") or {}).get("last_recompute_until_ts")
            _dt = _parse_iso(_ts) if _ts else None
            if _dt is not None:
                _cov_end[_r["id"]] = _dt.timestamp()
    except Exception:
        pass

    # Build window boundary arrays for fast bucket lookup
    w_starts = [_parse_iso(w["window_start"]).timestamp()
                for w in windows]

    def _which_window(ts_sec: float) -> int | None:
        idx = _bisect.bisect_right(w_starts, ts_sec) - 1
        if idx < 0 or idx >= len(windows):
            return None
        # Check it's actually within this window (timestamp could be after
        # the last window's end if range_end shifted)
        w_end_sec = _parse_iso(windows[idx]["window_end"]).timestamp()
        if ts_sec >= w_end_sec:
            return None
        return idx

    for sid in cohort:
        alerts = []
        s_ = 0
        while True:
            r = (c.table("alerts").select("fill_ts")
                 .eq("strategy_id", sid)
                 .gte("fill_ts", range_start_iso)
                 .lt("fill_ts", range_end_iso)
                 .order("fill_ts").range(s_, s_ + 999).execute())
            chunk = r.data or []
            alerts.extend(chunk)
            if len(chunk) < 1000:
                break
            s_ += 1000
        trades = []
        s_ = 0
        while True:
            r = (c.table("trades").select("entry_fill_ts,exit_fill_ts")
                 .eq("strategy_id", sid).like("data_source", "backtest_%")
                 .gte("entry_fill_ts", range_start_iso)
                 .lt("entry_fill_ts", range_end_iso)
                 .order("entry_fill_ts").range(s_, s_ + 999).execute())
            chunk = r.data or []
            trades.extend(chunk)
            if len(chunk) < 1000:
                break
            s_ += 1000

        # Pair globally over the range
        a_dts = []
        a_orig = []
        for a in alerts:
            ts = _parse_ts_local(a.get("fill_ts"))
            if ts is None:
                continue
            a_dts.append(ts.timestamp())
            a_orig.append(a)
        b_events = []
        for t in trades:
            ets = _parse_ts_local(t.get("entry_fill_ts"))
            if ets is not None:
                b_events.append((ets.timestamp(), "entry"))
            xts = _parse_ts_local(t.get("exit_fill_ts"))
            if xts is not None:
                b_events.append((xts.timestamp(), "exit"))

        a_sorted_idx = sorted(range(len(a_dts)), key=lambda k: a_dts[k])
        a_sorted_ts = [a_dts[k] for k in a_sorted_idx]
        b_sorted = sorted(b_events, key=lambda x: x[0])
        b_sorted_ts = [x[0] for x in b_sorted]
        a_paired_s, b_paired_s = _pair_events_5s(
            a_sorted_ts, b_sorted_ts, pair_window_s)
        a_paired = [False] * len(a_dts)
        for k, si in enumerate(a_sorted_idx):
            a_paired[si] = a_paired_s[k]

        # Bucket into windows
        for i, a in enumerate(a_orig):
            ts = _parse_ts_local(a.get("fill_ts"))
            if ts is None:
                continue
            w = _which_window(ts.timestamp())
            if w is None:
                continue
            bucket = window_data[w][sid]
            bucket["alerts"] += 1
            if a_paired[i]:
                bucket["paired"] += 1
            else:
                _cov = _cov_end.get(sid)
                if b_sorted_ts:
                    _cov = max(_cov or 0, b_sorted_ts[-1])
                if _cov is None or ts.timestamp() > _cov:
                    bucket["tbd"] += 1
                else:
                    bucket["phantom"] += 1
        for k, (ts_sec, _kind) in enumerate(b_sorted):
            w = _which_window(ts_sec)
            if w is None:
                continue
            bucket = window_data[w][sid]
            bucket["bt_events"] += 1
            if not b_paired_s[k]:
                bucket["missed"] += 1

    # Build response rows
    deploys_out = []
    for i, w in enumerate(windows):
        by_sid = window_data.get(i, {})
        tot_alerts = sum(b["alerts"] for b in by_sid.values())
        tot_bt = sum(b["bt_events"] for b in by_sid.values())
        tot_paired = sum(b["paired"] for b in by_sid.values())
        tot_phantom = sum(b["phantom"] for b in by_sid.values())
        tot_missed = sum(b["missed"] for b in by_sid.values())
        tot_tbd = sum(b.get("tbd", 0) for b in by_sid.values())
        denom = tot_paired + tot_phantom + tot_missed
        combined_pct = (100 * tot_paired / denom) if denom > 0 else None
        # Per-strategy combined % for the avg-across-strategies KPI
        ranked_cpcts = []
        for sid, b in by_sid.items():
            if b["alerts"] >= 1 and b["bt_events"] >= 1:
                d = b["paired"] + b["phantom"] + b["missed"]
                cp = (100 * b["paired"] / d) if d > 0 else 0.0
                ranked_cpcts.append(cp)
        avg_cpct = (sum(ranked_cpcts) / len(ranked_cpcts)
                    if ranked_cpcts else None)
        n_active = len(by_sid)
        n_ranked = len(ranked_cpcts)
        deploys_out.append({
            "sha": w["sha"],
            "subject": w["subject"],
            "timestamp_iso": w["timestamp_iso"],
            "window_start": w["window_start"],
            "window_end": w["window_end"],
            "duration_seconds": w["duration_seconds"],
            "stable_duration_seconds": w.get("stable_duration_seconds",
                                              w["duration_seconds"]),
            "raw_duration_seconds": w.get("raw_duration_seconds",
                                           w["duration_seconds"]),
            "low_confidence": w.get("low_confidence", False),
            "alerts": tot_alerts,
            "bt_events": tot_bt,
            "paired": tot_paired,
            "phantom": tot_phantom,
            "missed": tot_missed,
            "tbd": tot_tbd,
            "combined_pct": (round(combined_pct, 1)
                              if combined_pct is not None else None),
            "avg_strategy_combined_pct": (round(avg_cpct, 1)
                                           if avg_cpct is not None else None),
            "n_active_strategies": n_active,
            "n_ranked_strategies": n_ranked,
        })

    # Per-strategy per-deploy data for client-side filtering
    strategies_by_deploy: dict = {}
    for i, w in enumerate(windows):
        sha = w["sha"]
        by_sid = window_data.get(i, {})
        rows = []
        for sid, b in by_sid.items():
            denom = b["paired"] + b["phantom"] + b["missed"]
            cpct = (100 * b["paired"] / denom) if denom > 0 else 0.0
            rows.append({
                "strategy_id": sid,
                "alerts": b["alerts"],
                "bt_events": b["bt_events"],
                "paired": b["paired"],
                "phantom": b["phantom"],
                "missed": b["missed"],
                "tbd": b.get("tbd", 0),
                "combined_pct": round(cpct, 1),
                "rankable": b["alerts"] >= 1 and b["bt_events"] >= 1,
            })
        rows.sort(key=lambda r: r["strategy_id"])
        strategies_by_deploy[sha] = rows

    # Reverse so newest first for UI
    deploys_out.reverse()

    strategy_meta = _build_strategy_metadata(c, cohort)

    return {
        "now": now.isoformat(),
        "since": since_dt.isoformat(),
        "until": until_dt.isoformat(),
        "pair_window_s": pair_window_s,
        "stability_tail_minutes": stability_tail_minutes,
        "warmup_minutes": warmup_minutes,
        "deploy_count": len(deploys_out),
        "deploys": deploys_out,
        "strategies_by_deploy": strategies_by_deploy,
        "cohort_size": len(cohort),
        "strategy_metadata": [strategy_meta[sid] for sid in cohort
                              if sid in strategy_meta],
    }


# ─────────────────────────────────────────────────────────────────────
# Bar Parity % Endpoint (W2-6a, 2026-07-07)
# Per-(symbol, timeframe) field-level match rate between the LIVE BARS
# cache (live_bars — what the WS-fed engine saw at decision time) and
# REST truth (bar_cache 1Sec/1Min resampled to the strategy TF), over
# a recent window. Surfaces messy-vs-clean live-data strategies at a
# glance on the Strategy Health page.
#
# DEFINITION (documented for the V1 column tooltip + PR):
#   - Window: anchored at the newest SETTLED live bar for the pair
#     (bar closed ≥ _BP_SETTLE_MARGIN_SEC ago, so REST has had time to
#     settle), extending back min(240 bars, 2 trading hours) — capped
#     at 1 hour for sub-minute TFs so the 1Sec source pull stays ≤3600
#     rows. Anchoring at the last live bar (not wall-clock now) keeps
#     the column meaningful after hours / weekends: it grades the tail
#     of the most recent session.
#   - Sources: live side = live_bars rows with a WS-origin source
#     ('ws', 'ws_agg'). REST-derived rows (rest_backfill / rest_insert /
#     warmup_seed) are EXCLUDED from the comparison — they are REST
#     bytes and would trivially match, masking WS messiness. Their
#     count is reported as bars_backfilled. REST side = bar_cache
#     '1Sec' (TF < 60s) or '1Min' (TF ≥ 60s, per CLAUDE.md's
#     resample-from-1Min rule — native coarse Polygon bars have split
#     issues), bucketed to the strategy TF in pure Python
#     (epoch-aligned, first/max/min/last — same as pandas resample
#     label='left').
#   - Match: per field |live − rest| ≤ max($0.01, 0.02% · |rest|).
#   - Reported % = MIN of the four per-field match rates (O/H/L/C)
#     over bars present on BOTH sides. Worst-field is deliberately
#     conservative: a strategy whose closes match but whose highs are
#     WS-mangled is messy — stops/targets key off H/L. Per-field rates
#     + coverage gaps (live-only / rest-only bar counts) ship in the
#     payload for the tooltip but do NOT dilute the headline number.
#
# PERFORMANCE: computed per unique (symbol, timeframe) pair — many
# strategies share a pair, so the fleet page maps ~70 rows onto ~15-25
# computations. Each pair = 1 indexed live_bars query (LIMIT ≤ 240) +
# 1 indexed bar_cache range query (≤ 3600 rows). The whole result is
# cached module-level for _BP_CACHE_TTL_SEC (5 min) behind a lock, so
# the 30s fleet-page poll never triggers recomputation storms; only
# the first request after TTL pays the cost.
# ─────────────────────────────────────────────────────────────────────

import re as _bp_re
import threading as _bp_threading

_BP_CACHE_TTL_SEC = 300              # serve cached fleet result for 5 min
_BP_MAX_BARS = 240                   # per-pair bar cap
_BP_WINDOW_SEC = 2 * 3600            # ~2 trading hours (TF ≥ 60s)
_BP_SUBMIN_WINDOW_SEC = 3600         # sub-minute TFs: cap 1Sec pull at 1h
_BP_SETTLE_MARGIN_SEC = 300          # skip bars closed <5 min ago — REST
                                     # (and the settle sweeper) still revising
_BP_ABS_TOL = 0.01                   # $0.01 …
_BP_REL_TOL = 0.0002                 # … or 0.02% of the REST value, whichever
                                     # is larger, per field
_BP_MAX_TF_SEC = 3600                # TFs coarser than 1h can't fit ≥2 bars
                                     # in the window — reported unsupported
_BP_WS_SOURCE_PREFIX = "ws"          # live_bars.source values counted as WS

_bar_parity_cache: Dict[str, Any] = {}   # {"computed_at": dt, "payload": {..}}
_bar_parity_lock = _bp_threading.Lock()


def _bp_tf_seconds(timeframe: Any) -> Optional[int]:
    """'30Sec'→30, '1Min'→60, '1Hour'→3600. None for unparseable — the
    pair is then skipped VISIBLY (reason=bad_timeframe) instead of
    silently defaulting (no-silent-defaults house rule)."""
    m = _bp_re.match(r"^\s*(\d+)\s*(Sec|Min|Hour|Day|Week)",
                     str(timeframe or ""), _bp_re.I)
    if not m:
        return None
    return int(m.group(1)) * {"sec": 1, "min": 60, "hour": 3600,
                              "day": 86400, "week": 604800}[m.group(2).lower()]


def _bp_field_match(live_v: Any, rest_v: Any) -> bool:
    try:
        lv, rv = float(live_v), float(rest_v)
    except (TypeError, ValueError):
        return False
    return abs(lv - rv) <= max(_BP_ABS_TOL, _BP_REL_TOL * abs(rv))


def _bp_pair_stub(symbol: str, tf_str: str, tf_sec: Optional[int],
                  reason: str) -> Dict[str, Any]:
    return {
        "symbol": symbol, "timeframe": tf_str, "timeframe_seconds": tf_sec,
        "source_timeframe": None, "parity_pct": None, "field_pcts": None,
        "bars_compared": 0, "bars_live_only": 0, "bars_rest_only": 0,
        "bars_backfilled": 0, "window_start": None, "window_end": None,
        "reason": reason,
    }


def _compute_pair_parity(c, symbol: str, tf_str: str,
                         now: datetime) -> Dict[str, Any]:
    """Compute Bar Parity for one (symbol, timeframe) pair. Small,
    per-pair indexed queries only — see section header for definition."""
    tf_sec = _bp_tf_seconds(tf_str)
    if tf_sec is None:
        return _bp_pair_stub(symbol, tf_str, None, "bad_timeframe")
    if tf_sec > _BP_MAX_TF_SEC:
        return _bp_pair_stub(symbol, tf_str, tf_sec, "timeframe_too_coarse")

    # Newest bar eligible for comparison: closed ≥ settle-margin ago.
    settle_cutoff = now - timedelta(seconds=_BP_SETTLE_MARGIN_SEC + tf_sec)

    # 1 indexed query on the live_bars PK (symbol, tf, bar_start), LIMIT-
    # bounded. Newest-first; the first row anchors the window.
    live_rows = (c.table("live_bars")
                 .select("bar_start,open,high,low,close,source")
                 .eq("symbol", symbol)
                 .eq("timeframe_seconds", tf_sec)
                 .lte("bar_start", settle_cutoff.isoformat())
                 .order("bar_start", desc=True)
                 .limit(_BP_MAX_BARS)
                 .execute().data or [])
    if not live_rows:
        return _bp_pair_stub(symbol, tf_str, tf_sec, "no_live_bars")

    anchor = _parse_iso(live_rows[0].get("bar_start"))
    if anchor is None:
        return _bp_pair_stub(symbol, tf_str, tf_sec, "no_live_bars")
    window_sec = min(_BP_MAX_BARS * tf_sec,
                     _BP_SUBMIN_WINDOW_SEC if tf_sec < 60 else _BP_WINDOW_SEC)
    # Inclusive lower bound such that the window holds window_sec/tf slots.
    window_start = anchor - timedelta(seconds=window_sec - tf_sec)

    ws_bars: Dict[int, Dict[str, Any]] = {}    # aligned epoch → row
    backfilled_epochs: set = set()
    for r in live_rows:
        dt = _parse_iso(r.get("bar_start"))
        if dt is None or dt < window_start:
            continue
        src = str(r.get("source") or "")
        if not src.startswith(_BP_WS_SOURCE_PREFIX):
            # REST-derived hole-fill — excluded from comparison (see hdr)
            backfilled_epochs.add(int(dt.timestamp()))
            continue
        ws_bars[int(dt.timestamp())] = r
    backfilled = len(backfilled_epochs)
    if not ws_bars:
        return {**_bp_pair_stub(symbol, tf_str, tf_sec, "no_ws_bars"),
                "bars_backfilled": backfilled,
                "window_start": window_start.isoformat(),
                "window_end": anchor.isoformat()}

    # REST truth: 1Sec for sub-minute TFs; 1Min for TF ≥ 60s (CLAUDE.md —
    # never native coarse bars). Odd TFs not divisible by 60 fall back to
    # 1Sec so buckets stay exact. Row count bounded by window_sec (≤3600
    # for 1Sec) / window_sec/60 (≤120 for 1Min) — ≤1 page either way.
    src_tf = "1Sec" if (tf_sec < 60 or tf_sec % 60 != 0) else "1Min"
    src_sec = 1 if src_tf == "1Sec" else 60
    rest_end = anchor + timedelta(seconds=tf_sec - src_sec)
    rest_query = (c.table("bar_cache")
                  .select("ts,open,high,low,close")
                  .eq("symbol", symbol)
                  .eq("timeframe", src_tf)
                  .gte("ts", window_start.isoformat())
                  .lte("ts", rest_end.isoformat())
                  .order("ts"))
    rest_rows = _paginated_fetch(rest_query)
    if not rest_rows:
        return {**_bp_pair_stub(symbol, tf_str, tf_sec, "no_rest_bars"),
                "source_timeframe": src_tf, "bars_backfilled": backfilled,
                "window_start": window_start.isoformat(),
                "window_end": anchor.isoformat()}

    # Bucket REST rows to the strategy TF: epoch-aligned buckets,
    # open=first / high=max / low=min / close=last (rows arrive ts-asc).
    rest_buckets: Dict[int, Dict[str, float]] = {}
    for r in rest_rows:
        dt = _parse_iso(r.get("ts"))
        if dt is None:
            continue
        try:
            o, h = float(r["open"]), float(r["high"])
            lo, cl = float(r["low"]), float(r["close"])
        except (KeyError, TypeError, ValueError):
            continue
        epoch = int(dt.timestamp())
        b = epoch - (epoch % tf_sec)
        agg = rest_buckets.get(b)
        if agg is None:
            rest_buckets[b] = {"open": o, "high": h, "low": lo, "close": cl}
        else:
            agg["high"] = max(agg["high"], h)
            agg["low"] = min(agg["low"], lo)
            agg["close"] = cl

    fields = ("open", "high", "low", "close")
    matches = {f: 0 for f in fields}
    compared = 0
    for epoch, live in ws_bars.items():
        rest = rest_buckets.get(epoch)
        if rest is None:
            continue
        compared += 1
        for f in fields:
            if _bp_field_match(live.get(f), rest.get(f)):
                matches[f] += 1
    live_only = len(ws_bars) - compared
    anchor_epoch = int(anchor.timestamp())
    # Backfilled live rows count as "covered" for the rest_only gap stat —
    # REST having a bar the WS lane only got via backfill is already
    # surfaced by bars_backfilled.
    rest_only = sum(1 for b in rest_buckets
                    if b <= anchor_epoch and b not in ws_bars
                    and b not in backfilled_epochs)

    if compared == 0:
        return {**_bp_pair_stub(symbol, tf_str, tf_sec, "no_overlap"),
                "source_timeframe": src_tf, "bars_live_only": live_only,
                "bars_rest_only": rest_only, "bars_backfilled": backfilled,
                "window_start": window_start.isoformat(),
                "window_end": anchor.isoformat()}

    field_pcts = {f: round(100.0 * matches[f] / compared, 1) for f in fields}
    return {
        "symbol": symbol,
        "timeframe": tf_str,
        "timeframe_seconds": tf_sec,
        "source_timeframe": src_tf,
        # Headline = worst field (see definition in the section header).
        "parity_pct": min(field_pcts.values()),
        "field_pcts": field_pcts,
        "bars_compared": compared,
        "bars_live_only": live_only,
        "bars_rest_only": rest_only,
        "bars_backfilled": backfilled,
        "window_start": window_start.isoformat(),
        "window_end": anchor.isoformat(),
        "reason": None,
    }


@router.get("/bar-parity")
def get_bar_parity(
    user=Depends(get_current_user),  # noqa: ARG001 — admin guard
    refresh: bool = Query(
        False,
        description="Force recomputation, bypassing the 5-min server-side "
                    "cache. Debug/manual use — the UI never sets this."),
) -> dict:
    """Per-(symbol, timeframe) Bar Parity % for the fleet.

    Response:
        {
          "now": "...", "computed_at": "...", "cached": bool,
          "cache_ttl_seconds": 300,
          "tolerance": "max($0.01, 0.02%)",
          "pairs": [{
            "symbol": "TSLA", "timeframe": "30Sec", "timeframe_seconds": 30,
            "source_timeframe": "1Sec",
            "parity_pct": 96.7,                    # min of field_pcts
            "field_pcts": {"open": .., "high": .., "low": .., "close": ..},
            "bars_compared": N, "bars_live_only": N, "bars_rest_only": N,
            "bars_backfilled": N,
            "window_start": "...", "window_end": "...",
            "reason": null | "no_live_bars" | "no_rest_bars" | "no_ws_bars"
                    | "no_overlap" | "timeframe_too_coarse"
                    | "bad_timeframe" | "compute_error"
          }, ...]
        }

    The frontend joins rows to pairs by (symbol, timeframe) — one
    computation serves every strategy sharing the pair.
    """
    from db import get_admin_client
    now = datetime.now(timezone.utc)

    # Lock covers both the freshness check and the recompute so concurrent
    # stale hits can't stampede — the second caller blocks briefly, then
    # serves the freshly cached payload.
    with _bar_parity_lock:
        cached_payload = _bar_parity_cache.get("payload")
        computed_at = _bar_parity_cache.get("computed_at")
        if (not refresh and cached_payload is not None
                and computed_at is not None
                and (now - computed_at).total_seconds() < _BP_CACHE_TTL_SEC):
            return {**cached_payload, "now": now.isoformat(), "cached": True}

        c = get_admin_client()

        # Unique (symbol, timeframe) pairs across the fleet — the dedupe
        # that keeps this endpoint cheap (~70 strategies → ~15-25 pairs).
        pairs: set = set()
        try:
            srows = (c.table("strategies").select("symbol,timeframe")
                     .execute().data or [])
            for s in srows:
                sym, tf = s.get("symbol"), s.get("timeframe")
                if sym and tf:
                    pairs.add((sym, tf))
        except Exception as e:
            logger.warning("[bar-parity] strategies load failed: %s", e)

        pairs_out: List[Dict[str, Any]] = []
        for sym, tf in sorted(pairs):
            try:
                pairs_out.append(_compute_pair_parity(c, sym, tf, now))
            except Exception as e:
                logger.warning("[bar-parity] compute failed for %s/%s: %s",
                               sym, tf, e)
                pairs_out.append(_bp_pair_stub(
                    sym, tf, _bp_tf_seconds(tf), "compute_error"))

        payload = {
            "computed_at": now.isoformat(),
            "cache_ttl_seconds": _BP_CACHE_TTL_SEC,
            "tolerance": "max($0.01, 0.02%)",
            "pairs": pairs_out,
        }
        _bar_parity_cache["payload"] = payload
        _bar_parity_cache["computed_at"] = now
        return {**payload, "now": now.isoformat(), "cached": False}
