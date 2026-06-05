"""Gap detection — find time windows where alerts exist but BT trades are missing.

Output of `detect_gaps_for_strategy()` is a list of (start, end) datetime tuples
identifying ranges where the backtest lane has a coverage gap relative to live
alert volume. Used by the hourly gap-detection cron (to-be-wired in worker.py)
and by `_gap_detector_probe.py` for interactive testing.

Algorithm:
1. Pull alerts + backtest_* trades for strategy in `lookback_hours` window
2. Bucket alerts and trades into `bucket_minutes` windows
3. For each bucket: flag as gap if alerts >= `min_alerts` AND
   bt_trades < alerts * `coverage_threshold`
4. Merge adjacent flagged buckets into contiguous gap windows
5. Drop gaps fully inside the most-recent `recent_skip_minutes` (BT-lane
   batch lag is normal there, not a true gap)

Defaults are chosen for a 10-second TF strategy firing 20-30 alerts/hour:
- bucket = 5 min
- min_alerts/bucket = 2
- coverage_threshold = 0.33 (BT must produce trades for >= 1/3 of alerts)
- recent_skip = 10 min (BT writes can lag by this much normally)

Tune via kwargs for other TFs or volumes.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional


@dataclass
class GapBucket:
    """One 5-minute bucket flagged as having insufficient BT coverage."""
    bucket_start: datetime
    bucket_end: datetime
    alerts: int
    bt_trades: int
    coverage: float  # bt_trades / alerts


@dataclass
class Gap:
    """A contiguous gap window built from one or more adjacent GapBuckets."""
    start: datetime
    end: datetime
    bucket_count: int
    total_alerts: int
    total_bt_trades: int

    @property
    def duration_minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60

    @property
    def coverage_pct(self) -> float:
        return 100 * self.total_bt_trades / max(1, self.total_alerts)


def detect_gaps_for_strategy(
    strategy_id: int,
    *,
    lookback_hours: float = 2.0,
    bucket_minutes: int = 5,
    min_alerts_per_bucket: int = 2,
    coverage_threshold: float = 0.33,
    recent_skip_minutes: int = 10,
    now: Optional[datetime] = None,
) -> list[Gap]:
    """Detect BT-lane gaps for a single strategy.

    Returns list of Gap objects describing contiguous windows where
    the backtest lane is missing trades relative to live alert volume.
    Empty list = no gaps detected (BT lane is healthy for this strategy).

    Caller is responsible for the response action (typically: enqueue a
    targeted Update New Data over each gap's window via update_jobs).
    """
    if now is None:
        now = datetime.now(timezone.utc)

    from db import USE_DB
    if not USE_DB:
        return []

    from db import get_admin_client
    client = get_admin_client()

    window_start = now - timedelta(hours=lookback_hours)
    window_start_iso = window_start.isoformat()
    skip_after = now - timedelta(minutes=recent_skip_minutes)

    # Pull alerts in window (all that matter for pairing)
    a_rows = []
    start = 0
    while True:
        r = (client.table('alerts')
             .select('fill_ts,side')
             .eq('strategy_id', strategy_id)
             .gte('fill_ts', window_start_iso)
             .order('fill_ts')
             .range(start, start + 999)
             .execute())
        chunk = r.data or []
        a_rows.extend(chunk)
        if len(chunk) < 1000:
            break
        start += 1000

    # Pull backtest_* trades in window
    t_rows = []
    start = 0
    while True:
        r = (client.table('trades')
             .select('entry_fill_ts,exit_fill_ts')
             .eq('strategy_id', strategy_id)
             .like('data_source', 'backtest_%')
             .gte('entry_fill_ts', window_start_iso)
             .order('entry_fill_ts')
             .range(start, start + 999)
             .execute())
        chunk = r.data or []
        t_rows.extend(chunk)
        if len(chunk) < 1000:
            break
        start += 1000

    # Helper: parse + truncate to bucket boundary
    def _parse(s):
        if not s:
            return None
        try:
            return datetime.fromisoformat(s.replace('Z', '+00:00'))
        except Exception:
            return None

    def _bucket(ts):
        m = (ts.minute // bucket_minutes) * bucket_minutes
        return ts.replace(minute=m, second=0, microsecond=0)

    # Count alerts per bucket (entries + exits both count as "activity")
    alerts_by_bucket: dict = {}
    for a in a_rows:
        ts = _parse(a.get('fill_ts'))
        if not ts:
            continue
        b = _bucket(ts)
        alerts_by_bucket[b] = alerts_by_bucket.get(b, 0) + 1

    # Count BT trades per bucket (count BOTH entry and exit fills — each is an
    # event that should pair with an alert)
    trades_by_bucket: dict = {}
    for t in t_rows:
        for col in ('entry_fill_ts', 'exit_fill_ts'):
            ts = _parse(t.get(col))
            if not ts:
                continue
            b = _bucket(ts)
            trades_by_bucket[b] = trades_by_bucket.get(b, 0) + 1

    # Flag gap buckets
    flagged: list[GapBucket] = []
    all_buckets = sorted(set(alerts_by_bucket.keys()) | set(trades_by_bucket.keys()))
    for b in all_buckets:
        # Skip buckets fully inside the recent-skip window
        b_end = b + timedelta(minutes=bucket_minutes)
        if b_end > skip_after:
            continue
        n_a = alerts_by_bucket.get(b, 0)
        n_t = trades_by_bucket.get(b, 0)
        if n_a < min_alerts_per_bucket:
            continue
        coverage = n_t / n_a
        if coverage < coverage_threshold:
            flagged.append(GapBucket(
                bucket_start=b,
                bucket_end=b_end,
                alerts=n_a,
                bt_trades=n_t,
                coverage=coverage,
            ))

    # Merge adjacent flagged buckets into contiguous gaps
    if not flagged:
        return []
    flagged.sort(key=lambda g: g.bucket_start)
    gaps: list[Gap] = []
    cur_start = flagged[0].bucket_start
    cur_end = flagged[0].bucket_end
    cur_alerts = flagged[0].alerts
    cur_trades = flagged[0].bt_trades
    cur_count = 1
    for gb in flagged[1:]:
        if gb.bucket_start == cur_end:
            # Adjacent — extend current
            cur_end = gb.bucket_end
            cur_alerts += gb.alerts
            cur_trades += gb.bt_trades
            cur_count += 1
        else:
            # New gap
            gaps.append(Gap(cur_start, cur_end, cur_count, cur_alerts, cur_trades))
            cur_start = gb.bucket_start
            cur_end = gb.bucket_end
            cur_alerts = gb.alerts
            cur_trades = gb.bt_trades
            cur_count = 1
    gaps.append(Gap(cur_start, cur_end, cur_count, cur_alerts, cur_trades))

    return gaps


def detect_gaps_for_fleet(
    strategy_ids: list[int],
    **kwargs,
) -> dict[int, list[Gap]]:
    """Run detection for every strategy in the list. Returns {sid: [gaps]}.

    Strategies with no gaps are still represented (empty list value) so the
    caller can tell the difference between "not checked" and "checked, healthy."
    """
    out = {}
    for sid in strategy_ids:
        try:
            out[sid] = detect_gaps_for_strategy(sid, **kwargs)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "detect_gaps_for_strategy(%s) failed: %s", sid, e)
            out[sid] = []
    return out
