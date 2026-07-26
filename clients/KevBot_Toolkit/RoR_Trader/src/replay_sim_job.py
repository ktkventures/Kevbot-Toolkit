"""Replay-simulated last-10 basis — the PRODUCER job (board #120, V1.16).

Design: docs/_active/Design_Replay_Sim_Basis.md (job+schema = E's half; F renders).

Runs the REAL live engine over a strategy's recorded decision-time bars via
`replay_harness.replay()` (non-circular — the same StrategyMonitor / SymbolHub /
_ShadowIndicatorEngine live uses, under the ARMED prod flag stack), sizes a
BOUNDED window to cover the strategy's last-10 backtest trades (warmup stays
sacred — §6), records the would-have-fired edges + the divergence ledger (§8) +
budget telemetry, and writes ONE `replay_sim_basis` cache row. The health page
pairs those cached edges vs the current last-10 bt edges cheaply at read time —
this job is the expensive part and NEVER runs on page load.

TWO callers, one job body:
  * Phase 1 (on-demand button)  — one sid per click, off-hours, requested_by set.
  * Phase 2 (nightly opted-in)  — the poller enqueues opted-in sids; guardrails
                                  (off-hours / serialized / span-cap / skip-fresh)
                                  live in the poller, not here.

IMPORT-SAFE: `replay_harness` (pack-registry load + admin context + chdir + prod
DB reachability) is imported LAZILY inside run_sim_job so the pure helpers
(size_window / build_divergence_ledger / measure_coverage / classify_status /
apply_staleness) can be unit-tested offline. See test_replay_sim.py.

Prod-DB load rules apply (feedback_local_analysis_starves_live_worker): the job
issues exactly the reads the harness needs plus two light metadata queries; it
runs off-hours only when driven by the poller's nightly sweep.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("replay_sim_job")

# RTH open, in ET, for session-floor. Mirrors data_loader.SESSION_HOURS["RTH"].
_RTH_OPEN_HH, _RTH_OPEN_MM = 9, 30
_RTH_SECONDS = int(6.5 * 3600)          # 23400 — RTH bars/day denominator basis
_UNTIL_PAD_SEC = 120                    # ensure the last trade's exit lands (§6)
DEFAULT_MAX_SPAN_TRADING_DAYS = 5       # §6/§7 span cap; poller passes SIM_MAX_SPAN
_PRIMARY_WARMUP_DAYS = 7                # warmup_asof default (the sacred preamble)


# ─────────────────────────── pure helpers (offline-testable) ─────────────────

def _parse_iso(s: Any) -> Optional[datetime]:
    """Tolerant ISO→aware-UTC parse (accepts trailing Z). None on failure."""
    if s is None:
        return None
    if isinstance(s, datetime):
        return s if s.tzinfo else s.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _to_et(dt_utc: datetime):
    """Convert an aware UTC datetime to America/New_York (DST-correct)."""
    import pytz
    return dt_utc.astimezone(pytz.timezone("America/New_York"))


def session_floor_utc(dt_utc: datetime) -> datetime:
    """RTH open (09:30 ET) of dt's ET calendar date, returned as aware UTC.

    §6: `since` is floored to a SESSION boundary — never mid-bar — so the
    bounded window starts where a real trading day starts (bounds D5/D11)."""
    import pytz
    et = pytz.timezone("America/New_York")
    et_dt = dt_utc.astimezone(et)
    floor_et = et.localize(datetime(et_dt.year, et_dt.month, et_dt.day,
                                    _RTH_OPEN_HH, _RTH_OPEN_MM))
    return floor_et.astimezone(timezone.utc)


def _trading_day_span(since: datetime, until: datetime) -> int:
    """Weekday (Mon–Fri) ET dates in [since, until] inclusive — the cost proxy
    the span cap bounds. Calendar-free (no holiday calendar): a small
    overcount on a holiday week is a conservative cap, never an undercount."""
    d0 = _to_et(since).date()
    d1 = _to_et(until).date()
    if d1 < d0:
        return 0
    n = 0
    d = d0
    while d <= d1:
        if d.weekday() < 5:
            n += 1
        d += timedelta(days=1)
    return n


def _edge(trade: Dict[str, Any], kind: str) -> Optional[datetime]:
    return _parse_iso(trade.get(f"{kind}_fill_ts"))


def size_window(bt_trades: List[Dict[str, Any]],
                max_span_trading_days: int = DEFAULT_MAX_SPAN_TRADING_DAYS,
                pad_seconds: int = _UNTIL_PAD_SEC) -> Dict[str, Any]:
    """Bound the replay window to cover the last-10 bt trades (§6).

    bt_trades: completed backtest trades OLDEST→NEWEST (entry/exit ISO strings).
    Returns since/until (aware UTC), the covered subset, span metadata, and
    `capped` (True ⇒ the full span exceeded the cap and the OLDEST trades were
    dropped → a partial basis, denom shrinks, `window-capped` divergence). Never
    reduces warmup — this bounds only the decision/trade span. Empty input ⇒
    {'covered': []}."""
    covered = [t for t in bt_trades
               if _edge(t, "entry") is not None or _edge(t, "exit") is not None]
    if not covered:
        return {"since": None, "until": None, "covered": [], "capped": False,
                "bt_span_start": None, "bt_span_end": None, "bt_trade_count": 0,
                "trading_day_span": 0}

    def _bounds(trades: List[Dict[str, Any]]) -> Tuple[datetime, datetime]:
        first = trades[0]
        first_edge = _edge(first, "entry") or _edge(first, "exit")
        last_edges = [e for t in trades
                      for e in (_edge(t, "exit"), _edge(t, "entry"))
                      if e is not None]
        since = session_floor_utc(first_edge)
        until = max(last_edges) + timedelta(seconds=pad_seconds)
        return since, until

    capped = False
    while len(covered) > 1:
        since, until = _bounds(covered)
        if _trading_day_span(since, until) <= max_span_trading_days:
            break
        covered = covered[1:]          # drop the oldest, retry
        capped = True
    since, until = _bounds(covered)

    entries = [_edge(t, "entry") for t in covered if _edge(t, "entry")]
    exits = [_edge(t, "exit") for t in covered if _edge(t, "exit")]
    return {
        "since": since, "until": until, "covered": covered, "capped": capped,
        "bt_span_start": min(entries) if entries else since,
        "bt_span_end": max(exits) if exits else until,
        "bt_trade_count": len(covered),
        "trading_day_span": _trading_day_span(since, until),
    }


def _expected_rth_closes(since: datetime, until: datetime, tf_s: int) -> int:
    """Estimate of primary bar closes in an RTH window — coverage denominator.

    Approximate on purpose (no holiday/half-day calendar): weekday count ×
    RTH-bars-per-day. Drives a provenance chip only ("decision-time bars: N% of
    closes"), never a score. The zero case is what MUST be exact and is handled
    by the caller (n_closes==0 ⇒ coverage 0.0 ⇒ unres)."""
    days = _trading_day_span(since, until)
    if days <= 0 or tf_s <= 0:
        return 0
    if tf_s <= _RTH_SECONDS:
        per_day = max(1, _RTH_SECONDS // tf_s)
    else:
        per_day = 1
    return days * per_day


def measure_coverage(since: datetime, until: datetime, tf_s: int,
                     n_closes: int) -> float:
    """Fraction of expected primary closes the replay actually saw (§8 D4).

    numerator = n_closes (primary decision-time bars the engine iterated — each
    corresponds to a live_bars row that closed in-window, so this needs NO extra
    prod-DB read). denominator = _expected_rth_closes. Clamped to [0,1]. Zero
    decision-time bars ⇒ 0.0 (the caller turns that into a fail-loud unres,
    never a silent 0-score)."""
    if n_closes <= 0:
        return 0.0
    expected = _expected_rth_closes(since, until, tf_s)
    if expected <= 0:
        return 1.0
    return max(0.0, min(1.0, n_closes / expected))


# The canonical divergence ledger (§8). `applies` decides membership per run;
# footnotes are appended for run-specific detail (unres keys, drift tokens,
# window-cap, coverage%). New optimizations MUST add a row here with a test.
def _base_ledger() -> List[Dict[str, str]]:
    return [
        {"id": "D1", "title": "Bar-replay, not provider-event replay",
         "bias": "SIM can't see construction bugs → may score a trade live "
                 "actually mis-constructed",
         "surface": "bar replay — not provider-event; construction defects "
                    "invisible"},
        {"id": "D2", "title": "No live close/publish topology (single-monitor)",
         "bias": "SIM assumes ideal scheduling",
         "surface": "single-monitor — no fleet publish contention"},
        {"id": "D3", "title": "No operational loss modeled (SIM ≥ live)",
         "bias": "logic-would-fire evidence, NEVER delivery evidence",
         "surface": "SIM — would-fire, not delivery"},
        {"id": "D5", "title": "Session-open / warmup-seed convergence",
         "bias": "first-of-window entries can be late (~1 trade/day floor)",
         "surface": "state seeded as-of window start (characterized ~1 trade/day)"},
        {"id": "D6", "title": "Rapid re-entry clusters",
         "bias": "the 2nd of a 60–90s double-entry may be taken (~1 trade/day)",
         "surface": "rapid re-entry residual (characterized ~1 trade/day)"},
        {"id": "D9", "title": "Flag-stack drift",
         "bias": "a stale cache reflects a stale engine",
         "surface": "engine config as of compute time (flags fp)"},
        {"id": "D10", "title": "120s refresher is a MODEL of the live re-true",
         "bias": "re-true timing differs from live's",
         "surface": "120s refresher simulated (armed model)"},
        {"id": "D11", "title": "Bounded window re-seeds at window start",
         "bias": "amplifies D5; mitigated by session-floored `since` + full warmup",
         "surface": "window bounded to last-10 span (+session-floored warmup)"},
    ]


def build_divergence_ledger(diag: Dict[str, Any], coverage: Optional[float],
                            capped: bool, refresh_s: int, flags_fp: str,
                            computed_at_iso: str, corrected: bool) -> List[Dict[str, str]]:
    """Applicable ledger subset + run-specific footnotes (§8). Rendered next to
    the score; stored per-run in replay_sim_basis.divergences."""
    out: List[Dict[str, str]] = []
    for e in _base_ledger():
        if e["id"] == "D10" and refresh_s <= 0:
            continue                    # refresher retired → not simulated
        entry = dict(e)
        if e["id"] == "D9":
            entry["surface"] = (f"engine config as of {computed_at_iso} "
                                f"(flags fp {flags_fp})")
        if e["id"] == "D11" and capped:
            entry["surface"] += " · window-capped (last K of 10)"
        out.append(entry)

    # D4 coverage — always surfaced, value inline.
    cov_pct = "n/a" if coverage is None else f"{coverage * 100:.0f}%"
    out.append({
        "id": "D4", "title": "Decision-time live_bars coverage",
        "bias": "missing/partial, not wrong",
        "surface": f"decision-time bars: {cov_pct} of expected closes"})

    # D7 UNRES — fail-loud footnotes (never a silent 0).
    for u in (diag.get("unresolved") or []):
        out.append({"id": "D7", "title": "Unresolved ribbon (SIM unavailable)",
                    "bias": "no score — not a 0",
                    "surface": f"SIM unavailable — unresolved ribbon {u}"})
    # D8 LABEL-DRIFT — suspected live-engine bug, spun out separately.
    for w in (diag.get("label_drift") or []):
        out.append({"id": "D8", "title": "Label drift (suspected live-engine bug)",
                    "bias": "a live bug suspect, not a SIM artifact",
                    "surface": f"label-drift: {w}"})

    if corrected:
        out.append({"id": "corrected", "title": "REST-corrected OHLC (not decision-time)",
                    "bias": "convergent with backtest — not the honest ceiling",
                    "surface": "corrected OHLC — not decision-time"})
    return out


def classify_status(diag: Dict[str, Any], covered_trade_count: int,
                    capped: bool, coverage: Optional[float]) -> str:
    """ok | partial | unres. unres dominates (fail-loud); partial when the basis
    is incomplete (window-capped or coverage<1); ok otherwise. A low trade
    COUNT is not partial — the denominator just shrinks, like the alert basis."""
    if diag.get("unresolved"):
        return "unres"
    if covered_trade_count <= 0 or (coverage is not None and coverage <= 0.0):
        return "unres"
    if capped or (coverage is not None and coverage < 1.0):
        return "partial"
    return "ok"


def flags_fingerprint(armed_flags: Dict[str, str]) -> str:
    """Stable short fingerprint of the armed flag stack (§8 D9). The read path
    warns/invalidates if the LIVE armed fingerprint differs from a cache row."""
    blob = json.dumps(armed_flags, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def apply_staleness(bt_trades: List[Dict[str, Any]],
                    window_since: Optional[datetime],
                    window_until: Optional[datetime]) -> Dict[str, Any]:
    """Read-path staleness rule (§4.3). Given the CURRENT last-10 bt trades and a
    cache row's window, return the subset the cache actually covers + a `stale`
    flag. A trade whose exit lands past `window_until` is NOT covered — score
    only the covered subset (denom shrinks), never a full-denom score against an
    uncovered window."""
    if window_since is None or window_until is None:
        return {"covered": [], "stale": True, "n_covered": 0,
                "n_total": len(bt_trades)}
    covered = []
    stale = False
    for t in bt_trades:
        ex = _edge(t, "exit")
        en = _edge(t, "entry")
        latest = max([e for e in (ex, en) if e is not None], default=None)
        earliest = min([e for e in (en, ex) if e is not None], default=None)
        if latest is None or earliest is None:
            continue
        if earliest >= window_since and latest <= window_until:
            covered.append(t)
        else:
            stale = True
    return {"covered": covered, "stale": stale, "n_covered": len(covered),
            "n_total": len(bt_trades)}


def engine_sha() -> Optional[str]:
    """Best-effort short git sha of the engine code (provenance). None if git
    is unavailable (e.g. a detached tarball deploy)."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        sha = out.stdout.strip()
        return sha or None
    except Exception:  # noqa: BLE001
        return None


# ─────────────────────────── DB / harness orchestration ─────────────────────

def _stamp_armed_fp(sb, flags_fp: str) -> None:
    """Best-effort: record the armed-flag fingerprint of THIS run in
    system_settings so the API read path can detect drift cheaply (§8 D9)."""
    try:
        sb.table("system_settings").upsert({
            "key": "replay_sim_armed_fp",
            "value": {"fp": flags_fp,
                      "at": datetime.now(timezone.utc).isoformat()},
            "description": "Armed-flag fingerprint of the last replay-sim job "
                           "(board #120 D9 drift check)."},
            on_conflict="key").execute()
    except Exception as e:  # noqa: BLE001
        logger.warning("[sim_job] armed-fp stamp failed (non-fatal): %s", e)


def _last10_bt_trades(sb, sid: int) -> List[Dict[str, Any]]:
    """The last 10 completed BACKTEST-lane trades, oldest→newest — the SAME
    reference set health_last10._score_sid pairs against (like backtest_%,
    exit_fill_ts NOT NULL, newest-10, then reversed for display order)."""
    tr = (sb.table("trades")
          .select("id,entry_fill_ts,exit_fill_ts")
          .eq("strategy_id", sid).like("data_source", "backtest_%")
          .not_.is_("exit_fill_ts", "null")
          .order("exit_fill_ts", desc=True).limit(10).execute().data or [])
    return list(reversed(tr))


def run_sim_job(sid: int, sb=None, requested_by: Optional[str] = None,
                max_span_trading_days: int = DEFAULT_MAX_SPAN_TRADING_DAYS,
                corrected: bool = False, save: bool = True) -> Dict[str, Any]:
    """Produce (and optionally persist) one SIM basis for `sid`.

    Returns a result dict: {'status', 'result_id', 'row', 'skipped'?, 'reason'?}.
    'skipped' (no basis row written) when the strategy has no completed backtest
    trades to size a window to — that's not a SIM failure, there's just nothing
    to simulate against. Every other outcome writes a row (including the
    fail-loud unres case), so a 0 is never silent."""
    import time
    from db import get_admin_client
    from api.routers.strategy_health import _bp_tf_seconds

    sb = sb or get_admin_client()

    bt_trades = _last10_bt_trades(sb, sid)
    if not bt_trades:
        return {"status": "skipped", "skipped": True,
                "reason": "no completed backtest trades to size a window",
                "result_id": None, "row": None}

    # Symbol/timeframe from the service client (no admin-uid guesswork).
    srow = (sb.table("strategies").select("symbol,timeframe")
            .eq("id", sid).execute().data or [])
    if not srow:
        return {"status": "skipped", "skipped": True,
                "reason": f"strategy {sid} not found", "result_id": None,
                "row": None}
    symbol = srow[0]["symbol"]
    tf_s = _bp_tf_seconds(srow[0]["timeframe"])

    win = size_window(bt_trades, max_span_trading_days=max_span_trading_days)

    # Lazy import — pulls the ARMED flag stack, pack registry, admin context,
    # chdir to src/. Kept out of module import so the pure helpers test offline.
    import replay_harness

    t0 = time.time()
    r = replay_harness.replay(sid, win["since"], win["until"], sb,
                              corrected=corrected)
    compute_secs = round(time.time() - t0, 2)

    entries = [t.isoformat() for t in (r.get("entries") or [])]
    exits = [t.isoformat() for t in (r.get("exits") or [])]
    diag = r.get("diag") or {}
    n_closes = int(diag.get("n_closes") or 0)

    coverage = measure_coverage(win["since"], win["until"], tf_s, n_closes)
    capped = win["capped"]
    status = classify_status(diag, win["bt_trade_count"], capped, coverage)
    flags_fp = flags_fingerprint(replay_harness.ARMED_FLAGS)
    computed_at = datetime.now(timezone.utc).isoformat()
    refresh_s = getattr(replay_harness, "REFRESH_S", 0)
    ledger = build_divergence_ledger(diag, coverage, capped, refresh_s,
                                     flags_fp, computed_at, corrected)

    row = {
        "strategy_id": sid, "symbol": symbol, "timeframe_secs": tf_s,
        "window_since": win["since"].isoformat(),
        "window_until": win["until"].isoformat(),
        "warmup_days": _PRIMARY_WARMUP_DAYS,
        "bt_span_start": win["bt_span_start"].isoformat() if win["bt_span_start"] else None,
        "bt_span_end": win["bt_span_end"].isoformat() if win["bt_span_end"] else None,
        "bt_trade_count": win["bt_trade_count"],
        "replay_entries": entries, "replay_exits": exits,
        "corrected": corrected, "coverage": coverage,
        "divergences": ledger, "flags_fp": flags_fp, "engine_sha": engine_sha(),
        "status": status, "compute_secs": compute_secs,
        "computed_at": computed_at, "requested_by": requested_by,
    }

    # Stamp the current armed fingerprint so the API read path can flag rows
    # that predate a flag change (§8 D9) WITHOUT importing the heavy harness.
    _stamp_armed_fp(sb, flags_fp)

    result_id = None
    if save:
        res = sb.table("replay_sim_basis").insert(row).execute()
        result_id = (res.data[0]["id"] if res.data else None)
        logger.info("[sim_job] sid=%s status=%s span=%dd covered=%d/%d "
                    "entries=%d exits=%d coverage=%.2f secs=%.1f id=%s",
                    sid, status, win["trading_day_span"], win["bt_trade_count"],
                    len(bt_trades), len(entries), len(exits), coverage,
                    compute_secs, result_id)
    return {"status": status, "result_id": result_id, "row": row,
            "skipped": False}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Produce a replay-sim basis row.")
    ap.add_argument("--sid", type=int, required=True)
    ap.add_argument("--requested-by", default=None)
    ap.add_argument("--max-span-days", type=int,
                    default=int(os.environ.get("SIM_MAX_SPAN_TRADING_DAYS",
                                               DEFAULT_MAX_SPAN_TRADING_DAYS)))
    ap.add_argument("--corrected", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute but do NOT write the cache row")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    out = run_sim_job(args.sid, requested_by=args.requested_by,
                      max_span_trading_days=args.max_span_days,
                      corrected=args.corrected, save=not args.dry_run)
    print(json.dumps({k: v for k, v in out.items() if k != "row"}, default=str))
    if out.get("row"):
        print(f"status={out['status']} result_id={out['result_id']} "
              f"entries={len(out['row']['replay_entries'])} "
              f"exits={len(out['row']['replay_exits'])} "
              f"coverage={out['row']['coverage']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
