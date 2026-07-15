"""REPLAY HARNESS v1 — what would LIVE have traded, with no stalls and no lost alerts?

Replays the RECORDED DECISION-TIME bars (`live_bars.first_*` — the exact values the
WS delivered at the moment the engine decided) back through the REAL live engine
classes (StrategyMonitor / SymbolHub / _ShadowIndicatorEngine — never a
re-implementation), then pairs the resulting trades against the backtest lane at
5s / 10s / 60s.

Three numbers come out, and the gaps between them are the whole point:

    REPLAY vs BACKTEST   = the ACHIEVABLE fidelity ceiling (engine logic + the data
                           live actually had). A gap here is a REAL remaining bug.
    LIVE   vs BACKTEST   = what actually happened (the Strategy Health number).
    LIVE   vs REPLAY     = the OPERATIONAL cost — stalls, starved bar closes, lost
                           alert saves. Not a logic problem; an ops problem.

Born 2026-07-14, the day a starved engine (PRIMARY-RESYNC holding the GIL for ~9
min at a time) silently ate live entries and delayed alerts by 5-10 minutes,
contaminating a whole session of fidelity data before anyone noticed. A replay that
can answer "would we have taken that trade?" in minutes — instead of staking out
another market day — is the fix for that whole class of waiting.

FIDELITY RULES BAKED IN (learned the hard way — do not "simplify" these away):
 * Decision-time values: live_bars.first_* when present, else the healed values.
   (The `first (decision-time)` toggle on the Gate Parity panel is the same idea.)
 * NO LOOK-AHEAD: warmup is built to END at the replay start, never `now`-anchored
   (`_load_warmup_df` is now-anchored — using it here would leak the future).
 * Ordering invariant (test_ralph_gap_heal Bug-B lock): at coinciding closes the
   PRIMARY monitor pipeline runs BEFORE the secondary fan-out, so a closing primary
   bar gates on the PREVIOUS secondary bar. Reversing this reintroduces look-ahead.
 * Prod flag mirror: the replay runs the ARMED stack (PB defer, fail-closed,
   prev-chain, fine-authority, force-full re-trues). A replay under different flags
   answers a question nobody asked (local-update hard rule 0, same trap).
 * The 120s MTF refresher is simulated with force-full re-trues from canonical bars
   available AS OF each tick — it is part of the armed stack and materially moves
   gate states.
 * Standalone-engine env SOP: admin context + pack registry + USE_DB, or packs load
   as 0 and every gate silently fails.

Usage (run from src/, outside RTH per feedback_local_analysis_starves_live_worker):
    ../.venv/bin/python replay_harness.py --sids 327,328,333 \
        --since 2026-07-14T13:30:00Z --until 2026-07-14T20:00:00Z
    RH_DEBUG=1 ... → per-primary-bar gate/eff/prev trace

⚠️ STATUS (2026-07-14): the GATE lane is faithful — replayed secondary states match the
canonical series bar-for-bar (verified 13:30→14:00). The PRIMARY lane is NOT yet faithful:
the replay's primary trigger sequence does not reproduce live's (live took 11 entries on sid
328; the replay takes 0 — at 13:41:30 the gate was correctly open but the primary
`sw123_bull_c2` trigger never fired). **Do NOT trust the combined-% columns until the primary
lane reproduces live.** Next step: align the primary lineage — live's monitor warms once at
boot and evolves all day, while this replay warms as-of `since`; and confirm the 30Sec
decision-time bars in live_bars are exactly what the monitor consumed (vs. bars the engine
built from per-second data).

WHAT IT HAS ALREADY PROVEN (worth keeping even at v1):
 * The engine's fan-out 5Min bar is BYTE-IDENTICAL to the canonical resample-from-1Min and
   to REST — the "fan-out bars are noisy" theory is DEAD for these bars.
 * SWING_123 is anchor-INSENSITIVE (2d…20d warmup all agree) and the incremental
   implementation == the vectorized one. So a wrong live gate state cannot be blamed on
   warmup depth or on incremental-vs-vectorized drift — it was the re-true fast-path bug
   (#65), which is consistent with the gate lane replaying correctly under force-full.
 * A refresh must never run BEFORE the incremental close of a bar its canonical df already
   contains (double-apply → corrupted state). Live is immune (its re-true RESETS after the
   close); a replay must enforce it explicitly.
"""
import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv('.env')
os.environ['USE_DB'] = 'true'
os.environ['BAR_CACHE_ENABLED'] = '1'

# ── Mirror the ARMED prod Worker stack (see docs/Deploy_Log.md) ──────────────
ARMED_FLAGS = {
    'RORT_MTF_PB_DEFER': '1',
    'RORT_GATE_FAIL_CLOSED': '1',
    'RORT_WARMUP_PREV_CHAIN': '1',
    'RORT_MTF_FINE_INCREMENTAL_AUTHORITY': '1',
    'RORT_SHADOW_RETRUE_FORCE_FULL': '1',
    'RORT_MTF_SESSION_SHADOWS': '1',
    'RORT_INTERP_AWARE_SHADOWS': '1',
    'RORT_RESAMPLED_STORE_READ': '1',
    'RORT_RESAMPLED_STORE_SERVE_LIVE': '1',
    'RORT_MTF_STATE_REFRESH_S': '120',
    'RORT_ENFORCE_1MIN_GATE': '1',
    'RORT_COARSE_SECONDARY_FROM_1MIN': '1',
    'RORT_RIGHTSIZE_WARMUP': '1',
}
for _k, _v in ARMED_FLAGS.items():
    os.environ[_k] = _v

from db import set_admin_user_context, get_admin_client, get_strategy_by_id_admin  # noqa: E402
ADMIN = "19d47e46-f718-49a6-af32-5f5407f5b170"
set_admin_user_context(ADMIN)
import pack_registry  # noqa: E402
pack_registry.scan_and_load_all()
assert pack_registry.get_registered_packs(), "pack registry empty — gates would silently fail"

import ralph_engine as RE  # noqa: E402
from ralph_engine import StrategyMonitor, SymbolHub, SECONDS_TO_TIMEFRAME  # noqa: E402
from data_loader import load_market_data, resample_to_timeframe  # noqa: E402
# Phase A (Plan_Measurement_Trust): score BOTH lanes through the dashboard's own
# get_strategy_health, so the REPLAY ceiling `combined_pct` is byte-comparable to
# the LIVE dashboard number (identical edges/coverage/pairing/TBD-exclusion).
from api.routers import strategy_health as SH  # noqa: E402

REFRESH_S = int(ARMED_FLAGS['RORT_MTF_STATE_REFRESH_S'])
TOLERANCES = (5, 10, 60)
OHLC = ['open', 'high', 'low', 'close', 'volume']


# ─────────────────────────────── data ────────────────────────────────────────

def warmup_asof(sym: str, tf_s: int, session: str, end: datetime,
                days: int = 7) -> pd.DataFrame:
    """Warmup bars ENDING at `end` — never now-anchored (no look-ahead)."""
    if tf_s <= 60:
        df = load_market_data(sym, start_date=end - timedelta(days=days),
                              end_date=end, timeframe=SECONDS_TO_TIMEFRAME.get(tf_s, '1Min'),
                              feed='sip', session=session)
        return df[OHLC] if df is not None and len(df) else pd.DataFrame()
    df1 = load_market_data(sym, start_date=end - timedelta(days=days), end_date=end,
                           timeframe='1Min', feed='sip', session=session)
    if df1 is None or len(df1) == 0:
        return pd.DataFrame()
    out = resample_to_timeframe(df1[OHLC].copy(), SECONDS_TO_TIMEFRAME[tf_s])
    return out[out.index + pd.Timedelta(seconds=tf_s) <= pd.Timestamp(end)]


def decision_time_bars(sb, sym: str, tf_s: int, since: datetime,
                       until: datetime, corrected: bool = False) -> pd.DataFrame:
    """The bars the engine SAW.

    corrected=False (default): live_bars first-write values (`first_*`) — the
      decision-time WS view. The HONEST ceiling: what live could achieve with
      the real-time data it actually had (WS-vs-REST tip included).
    corrected=True: the LATEST/healed values (main OHLC columns) — REST-corrected,
      convergent with the backtest's REST hi-fi bars. The floor-vs-logic
      SPLITTER: a corrected ceiling that is STILL low ⇒ a real logic bug; a
      corrected ceiling ≈ 100% while decision-time is lower ⇒ the gap is just
      the irreducible WS/REST data-timing floor, not a bug.
    """
    rows = (sb.table('live_bars')
            .select('bar_start,open,high,low,close,volume,'
                    'first_open,first_high,first_low,first_close,first_volume')
            .eq('symbol', sym).eq('timeframe_seconds', tf_s)
            .gte('bar_start', since.isoformat()).lt('bar_start', until.isoformat())
            .order('bar_start').execute()).data
    if not rows:
        return pd.DataFrame()
    recs = []
    for r in rows:
        rec = {'timestamp': pd.Timestamp(r['bar_start'])}
        for c in ('open', 'high', 'low', 'close', 'volume'):
            if corrected:
                rec[c] = float(r[c])
            else:
                fv = r.get('first_' + c)
                rec[c] = float(fv if fv is not None else r[c])
        recs.append(rec)
    return pd.DataFrame(recs).set_index('timestamp').sort_index()


def canonical_bars_asof(sym: str, tf_s: int, session: str,
                        asof: datetime, days: int = 3) -> pd.DataFrame:
    """What the refresher's clean reload would have seen at `asof`."""
    return warmup_asof(sym, tf_s, session, asof, days=days)


# ─────────────────────────────── replay ──────────────────────────────────────

def replay(sid: int, since: datetime, until: datetime, sb,
           canonical_edge: bool = False, corrected: bool = False) -> list:
    """canonical_edge=False → replays the engine AS ARMED TODAY: fine-TF gate
    shadows advance on the FAN-OUT bar the live engine builds.

    canonical_edge=True → replays the PROPOSED fix: at each fine-TF close the
    shadow's state is rebuilt from the canonical resample of the engine's own
    1Min bars (the same construction the backtest uses), instead of the fan-out
    aggregation. No DB, no settle lag — 1Min bars the engine already holds.

    Why this A/B exists: measured 2026-07-14, the fan-out 5Min bar disagrees
    with the canonical resample at ~20% of closes (SWING pivot counts flip on
    tiny OHLC differences) — e.g. 14:35 canonical BULL_C2 vs fan-out NEUTRAL,
    which is exactly the state live held when it missed the 14:39 entries.
    """
    strat = get_strategy_by_id_admin(sid, ADMIN)
    sym = strat['symbol']
    monitor = StrategyMonitor(strat, None, general_packs=[])
    tf_s = monitor.tf_seconds
    session = monitor.session
    hub = SymbolHub(sym)
    hub.add_monitor(monitor)
    hub.finalize_shadow_engines()

    # Warmup (as-of `since`) — monitor + every shadow, then seed the gate dict.
    wdf = warmup_asof(sym, tf_s, session, since)
    if len(wdf) == 0:
        return []
    monitor.warmup(wdf)
    for (sec_tf, sec_sess), shadow in list(hub._shadow_engines.items()):
        sdf = warmup_asof(sym, sec_tf, sec_sess, since,
                          days=7 if sec_tf < 3600 else 60)
        if len(sdf):
            shadow.warmup(sdf)
            hub.seed_mtf_from_shadow(sec_tf, sec_sess)

    # Event stream: every bar CLOSE in the window, primary + secondaries.
    events = []   # (close_ts, kind, key, bar_dict)
    prim = decision_time_bars(sb, sym, tf_s, since, until, corrected=corrected)
    for ts, row in prim.iterrows():
        events.append((ts + pd.Timedelta(seconds=tf_s), 1, None,
                       {'timestamp': ts.isoformat(), **{c: float(row[c]) for c in OHLC}}))
    for (sec_tf, sec_sess), _sh in list(hub._shadow_engines.items()):
        sdf = decision_time_bars(sb, sym, sec_tf, since, until, corrected=corrected)
        for ts, row in sdf.iterrows():
            events.append((ts + pd.Timedelta(seconds=sec_tf), 2, (sec_tf, sec_sess),
                           {'timestamp': ts.isoformat(), **{c: float(row[c]) for c in OHLC}}))
    # ORDERING INVARIANT: at coinciding closes the PRIMARY runs BEFORE the
    # secondary fan-out (kind 1 < 2) — a closing primary gates on the PREVIOUS
    # secondary bar (test_ralph_gap_heal Bug-B lock). Do not "fix" this.
    events.sort(key=lambda e: (e[0], e[1]))

    # The engine's own 1Min series (warmup + decision-time), used by the
    # refresher sim and by canonical_edge. Loaded ONCE — a per-tick DB load
    # would be both glacial and a load risk (see
    # feedback_local_analysis_starves_live_worker).
    m1_warm = warmup_asof(sym, 60, session, since)
    m1_live = decision_time_bars(sb, sym, 60, since, until, corrected=corrected)
    m1 = pd.concat([m1_warm, m1_live])[OHLC]
    m1 = m1[~m1.index.duplicated(keep='last')].sort_index()

    def canonical_upto(sec_tf: int, close_ts) -> pd.DataFrame:
        """Canonical secondary bars (resample-from-1Min) fully closed by close_ts."""
        src = m1[m1.index + pd.Timedelta(minutes=1) <= close_ts]
        if len(src) == 0:
            return pd.DataFrame()
        out = resample_to_timeframe(src.copy(), SECONDS_TO_TIMEFRAME[sec_tf])
        return out[out.index + pd.Timedelta(seconds=sec_tf) <= close_ts]

    # Count entries and exits INDEPENDENTLY — exactly like the alert lane does.
    # (v1 bug: recording only completed round-trips made every unpaired entry
    # vanish, which read as "the replay took no trades".)
    # INTRA-BAR TICKS: live checks stops/targets every second (on_tick), so its
    # exits land MID-BAR (backtest exits at 13:44:38, 14:41:25, 16:01:56 … are
    # not on bar boundaries). A close-only replay defers every stop to the next
    # boundary and reads as an exit-timing divergence that isn't real. Feed the
    # 1Sec series through on_tick exactly as on_second_bar does.
    sec1 = load_market_data(sym, start_date=since, end_date=until,
                            timeframe='1Sec', feed='sip', session=session)
    sec1 = sec1[OHLC] if sec1 is not None and len(sec1) else pd.DataFrame()

    entries: list = []
    exits: list = []
    bar_n = len(wdf)          # continue the warmup's bar count, like the builder
    next_refresh = since + timedelta(seconds=REFRESH_S)

    for close_ts, kind, key, bar in events:
        # 120s force-full re-true (the armed RORT_SHADOW_RETRUE_FORCE_FULL path).
        # STRICTLY `>`: a refresh must never run BEFORE the incremental close of a
        # bar its own canonical df already contains — the re-true would seed the
        # state through bar B and the close would then apply B a SECOND time
        # (double-count → corrupted state; cost hours on 2026-07-14). Live is
        # immune because its re-true RESETS state after the close, never before.
        while close_ts > next_refresh:
            for (sec_tf, sec_sess), shadow in list(hub._shadow_engines.items()):
                cdf = canonical_upto(sec_tf, next_refresh)
                if len(cdf):
                    hub._publish_mtf((sec_tf, sec_sess),
                                     shadow.recompute_confluence(cdf),
                                     closed_bar_start_ts=cdf.index[-1])
            next_refresh += timedelta(seconds=REFRESH_S)

        if kind == 2:
            shadow = hub._shadow_engines.get(key)
            if shadow is None:
                continue
            sec_tf = key[0]
            if canonical_edge and 60 <= sec_tf < RE.COARSE_SECONDARY_SECONDS:
                cdf = canonical_upto(sec_tf, close_ts)
                if len(cdf):
                    hub._publish_mtf(key, shadow.recompute_confluence(cdf),
                                     closed_bar_start_ts=cdf.index[-1])
                    continue
            hub._publish_mtf(key, shadow.on_bar_close(bar),
                             closed_bar_start_ts=bar['timestamp'])
            continue

        if os.getenv('RH_DEBUG') == '1':
            _k = next(iter(hub._shadow_engines), None)
            print(f"    [dbg] primary close {str(close_ts)[11:19]} "
                  f"gate={sorted(hub._mtf_confluence.get(_k, []))} "
                  f"eff={hub._mtf_confluence_effective_from.get(_k)} "
                  f"prev={sorted(hub._mtf_confluence_prev.get(_k, []))}")

        # Intra-bar ticks for THIS primary bar, then its close (live's order:
        # per-second on_tick checks stops/targets, then on_bar_close).
        if len(sec1):
            _b0 = pd.Timestamp(bar['timestamp'])
            ticks = sec1[(sec1.index >= _b0) & (sec1.index < close_ts)]
            for _tts, _trow in ticks.iterrows():
                tsig = monitor.on_tick(float(_trow['close']),
                                       _tts.isoformat(), bar_n)
                for s in tsig or []:
                    k = (s.get('type') or s.get('side') or '').lower()
                    if 'entry' in k:
                        entries.append(_tts.to_pydatetime())
                    elif 'exit' in k:
                        exits.append(_tts.to_pydatetime())

        # bar_count must INCREMENT exactly as the live builder's does — the
        # position/execution machinery reads it (reference bars, bars-in-trade).
        # Feeding a constant 0 makes check_entry return None even when the
        # trigger fired AND the gate passed: the trade silently never happens.
        bar_n += 1
        signals, _ = monitor.on_bar_close(
            bar, bar_n,
            mtf_confluence=hub._mtf_confluence,
            mtf_prev=hub._mtf_confluence_prev,
            mtf_effective_from=hub._mtf_confluence_effective_from)
        if os.getenv('RH_DEBUG') == '1' and signals:
            print(f"    [dbg] SIGNALS at {str(close_ts)[11:19]}: "
                  f"{[(s.get('type'), s.get('trigger') or s.get('reason')) for s in signals]}")
        if os.getenv('RH_DEBUG') == '1':
            _pos = getattr(monitor.position, 'state', None)
            _st = getattr(_pos, 'status', '?') if _pos else '?'
            if _st != 'FLAT':
                print(f"    [dbg] position at {str(close_ts)[11:19]}: {_st}")

        for sig in signals or []:
            kind_s = (sig.get('type') or sig.get('side') or '').lower()
            fill = close_ts.to_pydatetime()  # C-type: fill at the deciding bar's close
            if 'entry' in kind_s:
                entries.append(fill)
            elif 'exit' in kind_s:
                exits.append(fill)
    return {'entries': entries, 'exits': exits}


# ─────────────────────────────── pairing ─────────────────────────────────────

def pair(a: list, b: list, tol: float) -> int:
    """Greedy nearest-first 1:1 match within tol seconds."""
    b = sorted(b)
    n = 0
    for x in sorted(a):
        best, bd = None, None
        for y in b:
            d = abs((x - y).total_seconds())
            if bd is None or d < bd:
                best, bd = y, d
        if best is not None and bd <= tol:
            n += 1
            b.remove(best)
    return n


def combined_pct(a: list, b: list, tol: float) -> float:
    if not a and not b:
        return float('nan')
    return 200.0 * pair(a, b, tol) / (len(a) + len(b))


def lanes(sb, sid: int, since: datetime, until: datetime):
    bt = (sb.table('trades').select('entry_fill_ts,exit_fill_ts')
          .eq('strategy_id', sid).eq('data_source', 'backtest_rest_hifi')
          .gte('entry_fill_ts', since.isoformat())
          .lte('entry_fill_ts', until.isoformat()).execute()).data
    al = (sb.table('alerts').select('side,fill_ts')
          .eq('strategy_id', sid)
          .gte('timestamp', (since - timedelta(minutes=30)).isoformat())
          .lte('timestamp', (until + timedelta(minutes=30)).isoformat())
          .execute()).data
    P = lambda s: datetime.fromisoformat(s.replace('Z', '+00:00'))  # noqa: E731
    bt_e = [P(t['entry_fill_ts']) for t in bt]
    bt_x = [P(t['exit_fill_ts']) for t in bt if t.get('exit_fill_ts')]
    live_e = [P(a['fill_ts']) for a in al
              if a['side'] == 'entry' and a['fill_ts']
              and since <= P(a['fill_ts']) <= until]
    live_x = [P(a['fill_ts']) for a in al
              if a['side'] == 'exit' and a['fill_ts']
              and since <= P(a['fill_ts']) <= until]
    return bt_e, bt_x, live_e, live_x


def _health_scores(since: datetime, until: datetime, tol: float,
                   replay_fills: dict | None) -> dict:
    """Call the dashboard's own get_strategy_health at one tolerance, either
    normally (LIVE) or with replay fills injected (REPLAY ceiling). Returns
    {sid: combined_pct}. Byte-comparable to the Strategy Health page."""
    SH._REPLAY_FILLS_OVERRIDE = replay_fills   # None → real alerts (LIVE)
    try:
        r = SH.get_strategy_health(
            user=None, window_hours=3,
            start=since.isoformat(), end=until.isoformat(),
            tolerance_seconds=tol)
    finally:
        SH._REPLAY_FILLS_OVERRIDE = None
    return {row['strategy_id']: row.get('combined_pct') for row in r['rows']}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sids', required=True)
    ap.add_argument('--since', required=True)
    ap.add_argument('--until', required=True)
    ap.add_argument('--corrected', action='store_true',
                    help="ALSO compute the REST-corrected ceiling (floor-vs-logic "
                         "splitter: corrected still low ⇒ real logic bug; "
                         "corrected≈100% but decision-time lower ⇒ WS/REST floor). "
                         "~2× the replay work.")
    args = ap.parse_args()
    since = datetime.fromisoformat(args.since.replace('Z', '+00:00'))
    until = datetime.fromisoformat(args.until.replace('Z', '+00:00'))
    sb = get_admin_client()
    sids = [int(s) for s in args.sids.split(',')]

    print("REPLAY HARNESS v2 — dashboard-scored (Plan_Measurement_Trust)")
    print(f"window {since:%Y-%m-%d %H:%M} → {until:%H:%M}Z")
    print("DASHBOARD-live and every REPLAY ceiling are scored by the SAME "
          "get_strategy_health (identical pairing/coverage/TBD-exclusion).\n")

    # 1) Replay each sid (decision-time; and corrected if requested) → fills.
    replay_fills: dict = {}
    corr_fills: dict = {}
    raw: dict = {}
    for sid in sids:
        try:
            r = replay(sid, since, until, sb)
        except Exception as e:  # noqa: BLE001
            print(f"{sid:>4} REPLAY FAILED: {str(e)[:80]}")
            continue
        replay_fills[sid] = ([t.timestamp() for t in r['entries']]
                             + [t.timestamp() for t in r['exits']])
        raw[sid] = r
        if args.corrected:
            try:
                rc = replay(sid, since, until, sb, corrected=True)
                corr_fills[sid] = ([t.timestamp() for t in rc['entries']]
                                   + [t.timestamp() for t in rc['exits']])
            except Exception as e:  # noqa: BLE001
                print(f"{sid:>4} CORRECTED REPLAY FAILED: {str(e)[:80]}")

    # 2) Score every lane via the dashboard code, at each tolerance.
    live = {t: _health_scores(since, until, t, None) for t in TOLERANCES}
    rep = {t: _health_scores(since, until, t, replay_fills) for t in TOLERANCES}
    corr = ({t: _health_scores(since, until, t, corr_fills) for t in TOLERANCES}
            if args.corrected else None)

    # 3) Report.
    def cell(v):
        return f"{v:5.1f}%" if isinstance(v, (int, float)) else "  n/a"
    cols = f"{'sid':>4}  {'DASHBOARD-live':>13}   {'CEILING(dtime)':>13}"
    subs = f"{'':>4}  {'@10s':>6} {'@60s':>6}   {'@10s':>6} {'@60s':>6}"
    if args.corrected:
        cols += f"   {'CEILING(corr)':>13}"
        subs += f"   {'@10s':>6} {'@60s':>6}"
    cols += f"   {'self':>5}"
    subs += f"   {'r≈l':>5}"
    print(cols)
    print(subs)
    print('-' * len(subs))
    for sid in sids:
        if sid not in raw:
            continue
        r = raw[sid]
        bt_e, bt_x, live_e, live_x = lanes(sb, sid, since, until)
        vals = [v for v in (combined_pct(r['entries'], live_e, 10),
                            combined_pct(r['exits'], live_x, 10)) if v == v]
        selfchk = f"{(sum(vals)/len(vals)):4.0f}%" if vals else " n/a"
        line = (f"{sid:>4}  {cell(live[10].get(sid))} {cell(live[60].get(sid))}   "
                f"{cell(rep[10].get(sid))} {cell(rep[60].get(sid))}")
        if args.corrected:
            line += f"   {cell(corr[10].get(sid))} {cell(corr[60].get(sid))}"
        line += f"   {selfchk:>5}"
        print(line)
    print("\nRead: live < CEILING(dtime) ⇒ operational loss (recoverable). "
          "CEILING(corr) high but CEILING(dtime) lower ⇒ WS/REST data-timing "
          "floor. CEILING(corr) ALSO low ⇒ a REAL logic bug. self<90% ⇒ replay "
          "didn't reproduce live (ops-contaminated window).")
    return 0


if __name__ == '__main__':
    sys.exit(main())
