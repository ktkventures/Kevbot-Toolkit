"""3-way divergence: pair REST-backtest, cache-backtest, and live-alert
trade events for a single strategy and surface drift metrics.

Lives behind GET /api/strategies/{id}/divergence-data. Pure functions —
input is the three lane payloads, output is a normalized rows list +
KPI summary. No DB access here; the router does the loading.

Match strategy:
  - Pivot on (direction, entry_fill_ts) — entries are the trust anchor.
  - Greedy: walk REST first, attach closest cache + closest live within
    direction match. Then unmatched cache → live. Then unmatched live
    → phantom rows.
  - No tolerance gate — every event lands in some row. Drift colors
    in the UI separate "matched within 2s" vs "matched within 60s"
    vs "structurally divergent."
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterable

logger = logging.getLogger(__name__)


def _parse_ts(ts) -> datetime | None:
    """Best-effort timestamp parsing. Returns timezone-aware UTC or None."""
    if ts is None or ts == '' or ts == '--':
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    if isinstance(ts, str):
        try:
            # Common Postgres ISO format with 'Z' or +00:00
            if ts.endswith('Z'):
                ts = ts[:-1] + '+00:00'
            return datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            return None
    return None


def _trade_entry_ts(t: dict) -> datetime | None:
    """Normalize various trade-shape entry-time fields to a datetime."""
    for k in ('entry_fill_ts', 'entryFillTime', 'entry_time', 'entryTime'):
        v = t.get(k)
        ts = _parse_ts(v)
        if ts is not None:
            return ts
    return None


def _trade_exit_ts(t: dict) -> datetime | None:
    for k in ('exit_fill_ts', 'exitFillTime', 'exit_time', 'exitTime'):
        v = t.get(k)
        ts = _parse_ts(v)
        if ts is not None:
            return ts
    return None


def _trade_direction(t: dict | None) -> str | None:
    if not t:
        return None
    d = t.get('direction')
    if not d:
        return None
    d = str(d).upper().strip()
    return d if d in ('LONG', 'SHORT') else None


def _alert_anchor_ts(a: dict) -> datetime | None:
    """Engine-fill moment for an alert. Prefer entry_fill_ts/exit_fill_ts
    (when the engine nominally filled) over `timestamp` (when the
    alert webhook fired — typically 1-2 bars later)."""
    atype = (a.get('type') or '').lower()
    if 'entry' in atype:
        ts = _parse_ts(a.get('entry_fill_ts')) or _parse_ts(a.get('fill_ts'))
    else:
        ts = _parse_ts(a.get('exit_fill_ts')) or _parse_ts(a.get('fill_ts'))
    if ts is None:
        ts = _parse_ts(a.get('timestamp'))
    return ts


def _pair_alerts_to_trades(alerts: list[dict]) -> list[dict]:
    """Walk alerts in chronological order, pair entry_signal + exit_signal
    into trade-shaped dicts. Open positions (entry without matching exit)
    surface as exit_fill_ts=None.

    Pairing rule: FIFO per (strategy, direction). When an exit_signal
    arrives, match the oldest open entry of the same direction.

    Anchor timestamp = entry_fill_ts / exit_fill_ts (engine-fill moments).
    `timestamp` (webhook send time) is 1-2 bars later and would inflate
    apparent drift against backtest trades.
    """
    if not alerts:
        return []

    sorted_alerts = sorted(
        alerts,
        key=lambda a: _alert_anchor_ts(a) or datetime.min.replace(tzinfo=timezone.utc),
    )

    open_entries_by_dir: dict[str, list[dict]] = {'LONG': [], 'SHORT': []}
    paired: list[dict] = []

    for a in sorted_alerts:
        atype = (a.get('type') or '').lower()
        adir = _trade_direction(a) or 'LONG'
        if adir not in open_entries_by_dir:
            adir = 'LONG'
        ats = _alert_anchor_ts(a)
        data = a.get('data') or {}
        # Most alert fields are now top-level (not nested under data) per
        # the Trade Timestamps Spec — read both with top-level priority.
        price = a.get('price') or data.get('price') or data.get('fill_price')

        if 'entry' in atype:
            open_entries_by_dir[adir].append({
                'entry_fill_ts': ats.isoformat() if ats else None,
                'direction': adir,
                'entry_price': price,
                'entry_alert_id': a.get('id'),
                'live_model': a.get('live_model'),
            })
        elif 'exit' in atype:
            queue = open_entries_by_dir.get(adir, [])
            if queue:
                entry = queue.pop(0)
                paired.append({
                    'entry_fill_ts': entry['entry_fill_ts'],
                    'exit_fill_ts': ats.isoformat() if ats else None,
                    'direction': adir,
                    'entry_price': entry['entry_price'],
                    'exit_price': price,
                    'exit_reason': a.get('trigger') or data.get('exit_reason'),
                    'entry_alert_id': entry.get('entry_alert_id'),
                    'exit_alert_id': a.get('id'),
                    '_lane': 'live',
                    # Live model labeling (2026-05-07). Take ENTRY's live_model
                    # since that's when the position was opened. Exit's model
                    # would technically be different if user switched mid-trade,
                    # but that's vanishingly rare and we're keeping it simple.
                    'live_model': entry.get('live_model') or a.get('live_model'),
                })

    for direction, queue in open_entries_by_dir.items():
        for entry in queue:
            paired.append({
                'entry_fill_ts': entry['entry_fill_ts'],
                'exit_fill_ts': None,
                'direction': entry['direction'],
                'entry_price': entry['entry_price'],
                'exit_price': None,
                'exit_reason': None,
                'entry_alert_id': entry.get('entry_alert_id'),
                'exit_alert_id': None,
                '_lane': 'live',
                '_open': True,
                'live_model': entry.get('live_model'),
            })

    return paired


def _within_forward_test(t: dict, forward_test_start: datetime | None) -> bool:
    if forward_test_start is None:
        return True
    ets = _trade_entry_ts(t)
    if ets is None:
        return False
    return ets >= forward_test_start


def _find_closest_match(
    target_ts: datetime,
    target_dir: str,
    candidates: list[dict],
    used: list[bool],
    max_distance_s: float = 3600.0,
) -> int | None:
    """Return index of closest unused same-direction candidate within
    `max_distance_s`, or None.

    The distance gate prevents spurious cross-period matches when one
    lane covers, say, last week and another covers today (e.g. stale
    REST backtest paired with fresh live alerts would otherwise match
    across days). Default 1 hour: matches across normal bar drift +
    operator delay, rejects stale-data pairings cleanly.
    """
    best_idx = None
    best_dist = float('inf')
    for i, cand in enumerate(candidates):
        if used[i]:
            continue
        if _trade_direction(cand) != target_dir:
            continue
        cts = _trade_entry_ts(cand)
        if cts is None:
            continue
        dist = abs((cts - target_ts).total_seconds())
        if dist > max_distance_s:
            continue
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def _drift_seconds(a: dict | None, b: dict | None, kind: str = 'entry') -> float | None:
    if not a or not b:
        return None
    a_ts = _trade_entry_ts(a) if kind == 'entry' else _trade_exit_ts(a)
    b_ts = _trade_entry_ts(b) if kind == 'entry' else _trade_exit_ts(b)
    if a_ts is None or b_ts is None:
        return None
    return abs((a_ts - b_ts).total_seconds())


def _shape_lane_trade(t: dict | None) -> dict | None:
    """Strip to the fields the frontend renders."""
    if not t:
        return None
    return {
        'entry_fill_ts': t.get('entry_fill_ts') or t.get('entryFillTime') or t.get('entryTime'),
        'exit_fill_ts': t.get('exit_fill_ts') or t.get('exitFillTime') or t.get('exitTime'),
        'entry_price': t.get('entry_price') or t.get('entryPrice'),
        'exit_price': t.get('exit_price') or t.get('exitPrice'),
        'direction': _trade_direction(t),
        'exit_reason': t.get('exit_reason') or t.get('exitReason'),
        'data_source': t.get('data_source'),
        'live_model': t.get('live_model'),
    }


def _classify_lane_composition(rest, cache, live) -> str:
    has = (bool(rest), bool(cache), bool(live))
    if all(has):
        return '3way'
    if has[0] and has[2]:
        return 'rest_live'
    if has[1] and has[2]:
        return 'cache_live'
    if has[0] and has[1]:
        return 'rest_cache'
    if has[0]:
        return 'rest_only'
    if has[1]:
        return 'cache_only'
    if has[2]:
        return 'live_only'
    return 'empty'


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round(pct * (len(s) - 1)))))
    return s[idx]


def _compute_kpis(rows: list[dict]) -> dict:
    counts = {
        'matched_3way': 0,
        'matched_rest_live': 0,
        'matched_cache_live': 0,
        'matched_rest_cache': 0,
        'rest_only': 0,
        'cache_only': 0,
        'live_only': 0,
    }

    drift_rl_entries: list[float] = []
    drift_cl_entries: list[float] = []
    drift_rc_entries: list[float] = []
    drift_rl_exits: list[float] = []
    drift_cl_exits: list[float] = []

    for r in rows:
        comp = r.get('lane_composition', '')
        if comp == '3way':
            counts['matched_3way'] += 1
        elif comp == 'rest_live':
            counts['matched_rest_live'] += 1
        elif comp == 'cache_live':
            counts['matched_cache_live'] += 1
        elif comp == 'rest_cache':
            counts['matched_rest_cache'] += 1
        elif comp == 'rest_only':
            counts['rest_only'] += 1
        elif comp == 'cache_only':
            counts['cache_only'] += 1
        elif comp == 'live_only':
            counts['live_only'] += 1

        d_rl = r.get('drift_rest_live_entry_s')
        if d_rl is not None:
            drift_rl_entries.append(d_rl)
        d_cl = r.get('drift_cache_live_entry_s')
        if d_cl is not None:
            drift_cl_entries.append(d_cl)
        d_rc = r.get('drift_rest_cache_entry_s')
        if d_rc is not None:
            drift_rc_entries.append(d_rc)
        d_rl_x = r.get('drift_rest_live_exit_s')
        if d_rl_x is not None:
            drift_rl_exits.append(d_rl_x)
        d_cl_x = r.get('drift_cache_live_exit_s')
        if d_cl_x is not None:
            drift_cl_exits.append(d_cl_x)

    def _stats(values):
        if not values:
            return {'count': 0, 'median_s': None, 'p95_s': None, 'max_s': None}
        return {
            'count': len(values),
            'median_s': _percentile(values, 0.5),
            'p95_s': _percentile(values, 0.95),
            'max_s': max(values),
        }

    return {
        **counts,
        'drift_rest_live_entry': _stats(drift_rl_entries),
        'drift_cache_live_entry': _stats(drift_cl_entries),
        'drift_rest_cache_entry': _stats(drift_rc_entries),
        'drift_rest_live_exit': _stats(drift_rl_exits),
        'drift_cache_live_exit': _stats(drift_cl_exits),
    }


def compute_three_way_divergence(
    rest_trades: list[dict],
    cache_trades: list[dict],
    alerts: list[dict],
    *,
    forward_test_start: str | None = None,
    direction_filter: str | None = None,
    default_direction: str | None = None,
    max_match_seconds: float = 300.0,
) -> dict:
    """Pair REST + cache + live trades on entry timestamp + direction.

    `default_direction`: stored_trades on `strategies.stored_trades`
    JSONB don't carry a per-trade `direction` field — they inherit it
    from the strategy. The endpoint passes the strategy direction so
    the joiner can match across lanes.

    Returns:
        {
          'rows': [DivergenceRow, ...],   # one row per trade-event-cluster
          'kpis': {...},                  # counts + drift stats
          'lane_counts': {                # input sizes pre-filter
            'rest': N, 'cache': N, 'live': N,
          },
        }
    """
    fts = _parse_ts(forward_test_start) if forward_test_start else None

    def _ensure_direction(trades: list[dict]) -> list[dict]:
        if not default_direction:
            return trades
        out = []
        for t in trades:
            if not _trade_direction(t):
                # Don't mutate caller's data
                t = {**t, 'direction': default_direction}
            out.append(t)
        return out

    rest_trades = _ensure_direction(rest_trades)
    cache_trades = _ensure_direction(cache_trades)

    live_trades = _pair_alerts_to_trades(alerts)

    # Filter all 3 by forward_test_start + direction.
    def _ok(t):
        if fts and not _within_forward_test(t, fts):
            return False
        if direction_filter:
            if _trade_direction(t) != direction_filter.upper():
                return False
        return True

    rest_q = [t for t in rest_trades if _ok(t)]
    cache_q = [t for t in cache_trades if _ok(t)]
    live_q = [t for t in live_trades if _ok(t)]

    # Sort by entry_fill_ts for deterministic greedy iteration.
    def _sort_key(t):
        ts = _trade_entry_ts(t)
        return ts or datetime.min.replace(tzinfo=timezone.utc)

    rest_q.sort(key=_sort_key)
    cache_q.sort(key=_sort_key)
    live_q.sort(key=_sort_key)

    used_cache = [False] * len(cache_q)
    used_live = [False] * len(live_q)

    rows: list[dict] = []

    # Pass 1: walk REST, attach closest cache + closest live (same dir).
    for r in rest_q:
        r_ts = _trade_entry_ts(r)
        r_dir = _trade_direction(r)
        c_idx = _find_closest_match(r_ts, r_dir, cache_q, used_cache, max_match_seconds) if r_ts and r_dir else None
        l_idx = _find_closest_match(r_ts, r_dir, live_q, used_live, max_match_seconds) if r_ts and r_dir else None

        cache_match = cache_q[c_idx] if c_idx is not None else None
        live_match = live_q[l_idx] if l_idx is not None else None

        rows.append(_build_row(rest=r, cache=cache_match, live=live_match))

        if c_idx is not None:
            used_cache[c_idx] = True
        if l_idx is not None:
            used_live[l_idx] = True

    # Pass 2: unmatched cache → attach closest live.
    for i, c in enumerate(cache_q):
        if used_cache[i]:
            continue
        c_ts = _trade_entry_ts(c)
        c_dir = _trade_direction(c)
        l_idx = _find_closest_match(c_ts, c_dir, live_q, used_live, max_match_seconds) if c_ts and c_dir else None
        live_match = live_q[l_idx] if l_idx is not None else None
        rows.append(_build_row(rest=None, cache=c, live=live_match))
        used_cache[i] = True
        if l_idx is not None:
            used_live[l_idx] = True

    # Pass 3: orphan live (phantom alerts).
    for i, l in enumerate(live_q):
        if used_live[i]:
            continue
        rows.append(_build_row(rest=None, cache=None, live=l))
        used_live[i] = True

    # Sort rows by anchor timestamp DESCENDING (most-recent first) for
    # table display. 2026-05-12: flipped from ascending — Kevin's
    # iteration loop is "look at the most recent trade and verify it,"
    # so paginating from the OLDEST end means clicking through hundreds
    # of pages of historical rest_only rows to reach today's data.
    def _row_sort_key(row):
        for lane_key in ('rest', 'cache', 'live'):
            t = row.get(lane_key)
            if t and t.get('entry_fill_ts'):
                ts = _parse_ts(t['entry_fill_ts'])
                if ts:
                    return ts
        return datetime.min.replace(tzinfo=timezone.utc)

    rows.sort(key=_row_sort_key, reverse=True)

    # Assign sequential row_ids for the table.
    for i, row in enumerate(rows):
        row['row_id'] = i + 1

    return {
        'rows': rows,
        'kpis': _compute_kpis(rows),
        'lane_counts': {
            'rest': len(rest_q),
            'cache': len(cache_q),
            'live': len(live_q),
        },
    }


def _build_row(rest=None, cache=None, live=None) -> dict:
    """Build a single comparison row given up to 3 lane references."""
    rest_shape = _shape_lane_trade(rest)
    cache_shape = _shape_lane_trade(cache)
    live_shape = _shape_lane_trade(live)

    return {
        'rest': rest_shape,
        'cache': cache_shape,
        'live': live_shape,
        'lane_composition': _classify_lane_composition(rest, cache, live),
        # Anchor timestamp = whichever lane has the earliest entry_fill_ts;
        # used for table sort + display.
        'anchor_entry_fill_ts': _earliest_entry_ts(rest_shape, cache_shape, live_shape),
        'direction': (
            _trade_direction(rest)
            or _trade_direction(cache)
            or _trade_direction(live)
        ),
        'drift_rest_live_entry_s': _drift_seconds(rest, live, 'entry'),
        'drift_cache_live_entry_s': _drift_seconds(cache, live, 'entry'),
        'drift_rest_cache_entry_s': _drift_seconds(rest, cache, 'entry'),
        'drift_rest_live_exit_s': _drift_seconds(rest, live, 'exit'),
        'drift_cache_live_exit_s': _drift_seconds(cache, live, 'exit'),
    }


def _earliest_entry_ts(*lanes) -> str | None:
    candidates = [
        _parse_ts((l or {}).get('entry_fill_ts'))
        for l in lanes if l
    ]
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        return None
    return min(candidates).isoformat()
