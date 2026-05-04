"""Data Health router — per-(symbol, timeframe) coverage of live_bars cache.

M8.7 (2026-05-04): backs the /admin/data-health frontend page.  Computes
expected vs actual bar counts over rolling windows so we can spot AM-stream
loss patterns (current known issue: ~60% loss on 1Min for high-volume
symbols like AAPL/AMD/SPY).  Also returns subscriber counts so we can
correlate coverage to demand.

This is read-only and admin-scoped (relies on the existing logged-in
guard — RoR_Trader is solo SaaS today, no separate role gating).
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Query

from api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/data-health", tags=["admin", "data-health"])


# Windows we report on, in seconds.  Order matters — table renders columns
# left-to-right in this order.
WINDOWS = [
    ("1h",   3600),
    ("4h",   4 * 3600),
    ("rth",  None),       # 09:30 ET → now (special-cased below)
    ("24h",  24 * 3600),
]


def _rth_start(now_utc: datetime) -> datetime:
    """09:30 ET on the date of `now_utc`. Returns midnight UTC if pre-RTH."""
    # Crude: 09:30 ET ≈ 13:30 UTC during EDT (March-Nov), 14:30 UTC during EST.
    # Use 13:30 UTC year-round as approximation — accurate enough for the
    # coverage calc.  If now is before RTH, anchor to today's RTH start
    # anyway (window will just have 0 expected).
    return datetime(now_utc.year, now_utc.month, now_utc.day,
                    13, 30, 0, tzinfo=timezone.utc)


def _parse_iso_or_unix(s: str | None) -> datetime | None:
    """Accept ISO 8601 ('2026-05-04T18:30:00Z') or Unix-sec string."""
    if not s:
        return None
    s = s.strip()
    # Try Unix integer first
    try:
        return datetime.fromtimestamp(int(s), tz=timezone.utc)
    except (ValueError, TypeError):
        pass
    # ISO — accept trailing Z
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


@router.get("")
def get_data_health(
    user=Depends(get_current_user),
    minutes: int = Query(1440, description="History window for full scan (default 24h)"),
    start: str | None = Query(None, description="Custom window start (ISO 8601 or Unix sec). When set with `end`, replaces rolling windows with single 'custom' window."),
    end: str | None = Query(None, description="Custom window end (ISO 8601 or Unix sec)."),
):
    """Return per-(symbol, tf) coverage metrics.

    Default mode: rolling windows {1h, 4h, rth, 24h}.

    Custom mode: when `start` AND `end` are both provided, the response
    contains a single 'custom' window scoped to that range.  Useful for
    "show me coverage since the last deploy" without rolling-window noise.

    Response shape:
        {
          "now": "...",
          "scan_minutes": 1440,
          "mode": "rolling" | "custom",
          "window_start": "..." (custom mode only),
          "window_end":   "..." (custom mode only),
          "rows": [
            {
              "symbol": "SPY", "timeframe_seconds": 60,
              "subscribers": 9,
              "windows": {
                <label>: {"expected": N, "actual": M, "coverage": M/N,
                          "ws": x, "rest_backfill": y, "other": z}
              },
              "latest_bar": "...",
              "latest_bar_age_sec": N,
              "gap_events_4h": N,
              "bars_missing_4h": N
            }, ...
          ]
        }

    Note: gap_events / bars_missing always reflect the last 4h window even
    in custom mode — they're a separate diagnostic, not gated by the
    rolling/custom toggle.
    """
    from db import get_admin_client

    c = get_admin_client()
    now = datetime.now(timezone.utc)

    # Resolve custom window if both start + end provided
    custom_start = _parse_iso_or_unix(start)
    custom_end = _parse_iso_or_unix(end)
    is_custom = (custom_start is not None and custom_end is not None
                 and custom_start < custom_end)

    # Scan must cover whichever is wider — the rolling windows or the
    # explicit custom range.
    if is_custom:
        # Pull from min(custom_start, now-4h) so the gap-analysis 4h window
        # is still computable
        scan_start = min(custom_start, now - timedelta(hours=4))
    else:
        scan_start = now - timedelta(minutes=minutes)

    # Pull a single page of rows per (symbol, tf) is hard via PostgREST without
    # group_by, so just paginate through the recent rows.  At ~9k rows/day for
    # the current tracked symbols this is cheap.  Lift to a SQL view if it
    # ever becomes hot.
    rows: List[Dict[str, Any]] = []
    offset = 0
    batch = 1000
    while True:
        r = c.table("live_bars").select(
            "symbol,timeframe_seconds,bar_start,source"
        ).gte("bar_start", scan_start.isoformat()) \
         .order("bar_start") \
         .range(offset, offset + batch - 1).execute()
        page = r.data or []
        rows.extend(page)
        if len(page) < batch:
            break
        offset += batch

    # Group by (symbol, tf) for window scans
    by_key: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(row["symbol"], row["timeframe_seconds"])].append(row)

    # Subscriber counts (active strategies, current + by-tf)
    strats = c.table("strategies").select("id,symbol,timeframe").execute().data or []
    sub_counts: Dict[tuple, int] = Counter()
    for s in strats:
        tf_label = s.get("timeframe", "")
        # Map TF label → seconds for matching
        tf_secs_map = {
            "5Sec": 5, "10Sec": 10, "30Sec": 30, "1Min": 60, "5Min": 300,
            "15Min": 900, "30Min": 1800, "1Hour": 3600, "1Day": 86400,
        }
        tf_secs = tf_secs_map.get(tf_label)
        if tf_secs is not None and s.get("symbol"):
            sub_counts[(s["symbol"], tf_secs)] += 1

    # Compute per-key metrics.
    # In rolling mode: use the predefined WINDOWS list.
    # In custom mode: a single 'custom' window scoped to [start, end].
    out_rows: List[Dict[str, Any]] = []
    rth_anchor = _rth_start(now)

    if is_custom:
        active_windows: list[tuple[str, datetime, float]] = [
            ("custom", custom_start, max(1, (custom_end - custom_start).total_seconds()))
        ]
    else:
        active_windows = []
        for label, secs in WINDOWS:
            if label == "rth":
                active_windows.append((label, rth_anchor,
                                       max(1, (now - rth_anchor).total_seconds())))
            else:
                active_windows.append((label, now - timedelta(seconds=secs), float(secs)))

    for (sym, tf), key_rows in sorted(by_key.items()):
        windows_out: Dict[str, Dict[str, Any]] = {}

        for label, window_start, window_secs in active_windows:
            # In custom mode, also bound by `end` (rolling mode end is now)
            window_end_dt = custom_end if is_custom else now
            window_rows = [
                r for r in key_rows
                if window_start <= datetime.fromisoformat(
                    r["bar_start"].replace("Z", "+00:00")) <= window_end_dt
            ]
            actual = len(window_rows)
            expected = max(1, int(window_secs // tf))
            coverage = actual / expected if expected > 0 else 0.0

            src = Counter(r["source"] for r in window_rows)
            windows_out[label] = {
                "expected": expected,
                "actual": actual,
                "coverage": round(coverage, 4),
                "ws": src.get("ws", 0),
                "rest_backfill": src.get("rest_backfill", 0),
                "other": sum(v for k, v in src.items()
                             if k not in ("ws", "rest_backfill")),
            }

        # Latest bar freshness
        latest_iso = None
        latest_age = None
        if key_rows:
            latest = max(key_rows,
                         key=lambda r: datetime.fromisoformat(
                             r["bar_start"].replace("Z", "+00:00")))
            latest_iso = latest["bar_start"]
            latest_dt = datetime.fromisoformat(latest_iso.replace("Z", "+00:00"))
            latest_age = int((now - latest_dt).total_seconds())

        # Gap analysis over the last 4h
        four_h_start = now - timedelta(hours=4)
        recent = sorted(
            [r for r in key_rows
             if datetime.fromisoformat(r["bar_start"].replace("Z", "+00:00")) >= four_h_start],
            key=lambda r: r["bar_start"],
        )
        gap_events = 0
        bars_missing = 0
        for i in range(1, len(recent)):
            t0 = datetime.fromisoformat(recent[i-1]["bar_start"].replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(recent[i]["bar_start"].replace("Z", "+00:00"))
            delta_units = round((t1 - t0).total_seconds() / tf)
            if delta_units > 1:
                gap_events += 1
                bars_missing += int(delta_units - 1)

        out_rows.append({
            "symbol": sym,
            "timeframe_seconds": tf,
            "subscribers": sub_counts.get((sym, tf), 0),
            "windows": windows_out,
            "latest_bar": latest_iso,
            "latest_bar_age_sec": latest_age,
            "gap_events_4h": gap_events,
            "bars_missing_4h": bars_missing,
        })

    # Tracked-but-empty entries: any (symbol, tf) the worker is supposed to be
    # subscribed to but has zero rows in the scan window.  Worth surfacing
    # even though they have no data.
    seen = set(by_key.keys())
    for key, n_subs in sub_counts.items():
        if key in seen or n_subs == 0:
            continue
        sym, tf = key
        out_rows.append({
            "symbol": sym,
            "timeframe_seconds": tf,
            "subscribers": n_subs,
            "windows": {label: {"expected": 0, "actual": 0, "coverage": 0.0,
                                "ws": 0, "rest_backfill": 0, "other": 0}
                        for label, _, _ in active_windows},
            "latest_bar": None,
            "latest_bar_age_sec": None,
            "gap_events_4h": 0,
            "bars_missing_4h": 0,
        })

    out_rows.sort(key=lambda r: (r["symbol"], r["timeframe_seconds"]))

    response: Dict[str, Any] = {
        "now": now.isoformat(),
        "scan_minutes": minutes,
        "mode": "custom" if is_custom else "rolling",
        "rows": out_rows,
    }
    if is_custom:
        response["window_start"] = custom_start.isoformat()
        response["window_end"] = custom_end.isoformat()
    return response
