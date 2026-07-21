"""REPLAY HARNESS v2 — what would LIVE have traded, with no stalls and no lost alerts?

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

✅ STATUS (2026-07-21, Plan_Harness_Secondary_TF_Gap executed): the secondary-TF gap is
CLOSED for coarse gates and fail-loud everywhere else. GEN- gates now evaluate (worker-mirror
pack load); coarse (>=1h) refresher re-trues use a 60-day 1Min series (the 7-day one fed a 4h
stoch ~11 bars -> NaN -> clobbered a good seed: sid 314 read 0 while live fired — now
92.3%@10s vs its settled backtest); sub-minute re-trues use the shadow's own series (the 1Min
upsample was garbage); and a ceiling the replay could not resolve prints UNRES (never a silent
0) while a resolved-but-unmatchable ribbon prints a LABEL-DRIFT warning (sub-minute '10Sec-'
vs required '10S-' — a LIVE-ENGINE bug suspect, e.g. 339, spun out; do not fix here).
Validation canaries post-bulk-delete (327/328/333 are GONE from the DB): 314 coarse, 267
fine (100%@10s, 91 bt entries 07-16), 340 short-fine (100%), 325 honest real-closed n/a.

✅ STATUS (2026-07-15): the primary lane is now FAITHFUL WITHIN ITS SCOPE and validated.
`bar_count` increments like the live builder, intra-bar 1Sec ticks drive stops/targets, and the
replay reproduces LIVE within 1-2% on clean days (self-check 271=98 / 267=91 / 308=96) AND the
backtest lane trade-for-trade on settled days (328: 10-16 of 11-17 entries second-identical;
the residual is CHARACTERIZED — one trade/day of session-open convergence + rapid re-entry —
not mysterious). The combined-% columns ARE trustworthy: they are scored by the dashboard's own
`get_strategy_health` (identical pairing / coverage / TBD-exclusion), NOT the local helpers.

⚠️ SCOPE LIMIT (Brandon audit 2026-07-15 — read it honestly): this is a BAR/ENGINE replay, NOT
a provider-event replay. It consumes the ALREADY-AGGREGATED `live_bars.first_*` rows and calls
the monitor/shadow classes directly, so it BYPASSES WS parsing, WsAggMinute/BarBuilder
construction, reconnect/dup/correction ordering, the flush-vs-fan-out race, and grace firing.
It therefore certifies "the decision LOGIC is faithful GIVEN the recorded bars" — it does NOT
certify the bars were CONSTRUCTED / ROUTED / grace-gated correctly. The live-path construction
defects (P0 `>=60s` partial force-close, P1 default-model routing gap, P1 grace suppression)
live BELOW this harness; catching them needs the provider-event replay upgrade. See
docs/audits/Current_Dev_Candle_Parity_Audit_2026-07-15.md.

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
    'RORT_MTF_STATE_REFRESH_S': '120',     # refresher RE-ENABLED 2026-07-17: 5b's
    #   canonical fine-TF derive does NOT cover slow-close gates (15Min UT_BOT) — its
    #   staleness drove 333's 6 morning over-fires + 340's 14:01. Re-armed to 120 until
    #   a proper slow-gate replacement lands. Mirrors prod Worker.
    'RORT_CANONICAL_FINE_TF_STATE': '1',   # M-RS5b: canonical fine-TF gate state (kept)
    'RORT_ENFORCE_1MIN_GATE': '1',
    'RORT_COARSE_SECONDARY_FROM_1MIN': '1',
    'RORT_RIGHTSIZE_WARMUP': '1',
    'RORT_TF_LABEL_SEC_FIX': '1',  # armed 2026-07-21 14:52Z (PR #68): sub-minute
    #   confluence label drift — Sec→S/Week→W canonical emit at both record sites.
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
           canonical_edge: bool = True, corrected: bool = False) -> list:
    """DEFAULT canonical_edge=True (2026-07-16): matches TODAY's live stack —
    M-RS5b canonical fine-TF gate state, refresher retired. Pass
    canonical_edge=False for the pre-5b incremental+refresher semantics (A/B).

    canonical_edge=False → replays the engine AS ARMED TODAY: fine-TF gate
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
    if not strat:
        raise ValueError(f"strategy {sid} not found (deleted?)")
    sym = strat['symbol']
    # GEN- gates: mirror the worker's startup admin-path pack load (the
    # ralph_engine 2026-06-22 fix — same loader, same uid). general_packs=[]
    # made every GEN- gated strategy's gate unsatisfiable IN THE REPLAY:
    # confluence_set = confluence | general_confluences, and on_bar_close
    # step 3b only emits GEN- records for packs present in this list — so
    # e.g. 339 (GEN-TIME_OF_DAY) replayed as a silent 0 regardless of its
    # secondary TFs. (Found in the 2026-07-20 secondary-TF-gap diagnosis.)
    from db import load_general_packs_admin
    from general_packs import _parse_pack_list, get_enabled_general_packs
    gen_packs = get_enabled_general_packs(
        _parse_pack_list(load_general_packs_admin(ADMIN)))
    monitor = StrategyMonitor(strat, None, general_packs=gen_packs)
    tf_s = monitor.tf_seconds
    session = monitor.session
    hub = SymbolHub(sym)
    hub.add_monitor(monitor)
    hub.finalize_shadow_engines()

    DBG = os.getenv('RH_DEBUG') == '1'
    need = monitor.position.confluence_set or set()

    # ── Step-0 diagnostics (Plan_Harness_Secondary_TF_Gap): track every
    # _publish_mtf through a counting wrapper so the summary can say, per
    # shadow key, how it advanced (seed / close events / refresher) and the
    # UNION of every record it ever held — the evidence that separates
    # "ribbon never resolved" (empty warmup / NaN indicator / label drift)
    # from "resolved and legitimately closed". Instance-attribute patch:
    # internal hub calls (seed_mtf_from_shadow) route through it too.
    pub_n: dict = {}
    pub_union: dict = {}
    _last_pub: dict = {}
    _now_ref = [since]           # sim-time for debug lines
    _orig_publish = hub._publish_mtf

    def _tracked_publish(key, records, closed_bar_start_ts=None):
        pub_n[key] = pub_n.get(key, 0) + 1
        pub_union.setdefault(key, set()).update(records or ())
        if DBG and _last_pub.get(key) != set(records or ()):
            print(f"    [diag] {_now_ref[0]} publish {key}: "
                  f"{sorted(records or ())}")
            _last_pub[key] = set(records or ())
        return _orig_publish(key, records,
                             closed_bar_start_ts=closed_bar_start_ts)
    hub._publish_mtf = _tracked_publish

    if DBG:
        print(f"    [diag] sid={sid} required gate: {sorted(need)}")
        print(f"    [diag] gen_packs loaded: {len(gen_packs)}")

    # Warmup (as-of `since`) — monitor + every shadow, then seed the gate dict.
    wdf = warmup_asof(sym, tf_s, session, since)
    if len(wdf) == 0:
        return {'entries': [], 'exits': [],
                'diag': {'required': sorted(need), 'gen_packs': len(gen_packs),
                         'n_closes': 0, 'gate_open_closes': 0, 'tok_hits': {},
                         'keys': {}, 'unresolved': ['PRIMARY-WARMUP-EMPTY']}}
    monitor.warmup(wdf)
    shadow_warm_rows: dict = {}
    subminute_src: dict = {}     # key → seed-warmup df for sub-minute re-trues
    for (sec_tf, sec_sess), shadow in list(hub._shadow_engines.items()):
        sdf = warmup_asof(sym, sec_tf, sec_sess, since,
                          days=7 if sec_tf < 3600 else 60)
        shadow_warm_rows[(sec_tf, sec_sess)] = len(sdf)
        if len(sdf):
            shadow.warmup(sdf)
            hub.seed_mtf_from_shadow(sec_tf, sec_sess)
            if sec_tf < 60:
                subminute_src[(sec_tf, sec_sess)] = sdf
        if DBG:
            span = (f"{sdf.index[0]} .. {sdf.index[-1]}"
                    if len(sdf) else "EMPTY")
            print(f"    [diag] shadow tf={sec_tf}s sess={sec_sess}: "
                  f"warmup_rows={len(sdf)} span={span}")
    seed_pubs = dict(pub_n)      # publishes so far = warmup seeds

    # Event stream: every bar CLOSE in the window, primary + secondaries.
    events = []   # (close_ts, kind, key, bar_dict)
    prim = decision_time_bars(sb, sym, tf_s, since, until, corrected=corrected)
    for ts, row in prim.iterrows():
        events.append((ts + pd.Timedelta(seconds=tf_s), 1, None,
                       {'timestamp': ts.isoformat(), **{c: float(row[c]) for c in OHLC}}))
    for (sec_tf, sec_sess), _sh in list(hub._shadow_engines.items()):
        sdf = decision_time_bars(sb, sym, sec_tf, since, until, corrected=corrected)
        if sec_tf < 60 and len(sdf):
            # Sub-minute re-true source = seed warmup + decision-time bars
            # (the exact series the shadow's close events consume).
            _prev = subminute_src.get((sec_tf, sec_sess))
            _cat = (pd.concat([_prev, sdf[OHLC]])
                    if _prev is not None else sdf[OHLC])
            subminute_src[(sec_tf, sec_sess)] = (
                _cat[~_cat.index.duplicated(keep='last')].sort_index())
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

    # COARSE re-true source (Plan_Harness_Secondary_TF_Gap Step 2b, sid-314
    # lying-0): a 4h stochastic re-trued from the 7-day m1 gets ~11 bars →
    # NaN → the 120s refresher clobbers a GOOD 60-day seed with an empty
    # record set at its first tick, and the coarse gate stays shut all day.
    # Live's refresher reloads per-TF rightsized depth; the as-of-safe
    # equivalent is a deep 1Min series matching warmup_asof's coarse depth.
    # Kept SEPARATE from m1 so the fine-TF (60s..<3600s) canonical path
    # stays byte-identical to the validated 327/328 behavior.
    coarse_keys = any(tf >= RE.COARSE_SECONDARY_SECONDS
                      for tf, _s in hub._shadow_engines)
    if coarse_keys:
        m1_deep = pd.concat([warmup_asof(sym, 60, session, since, days=60),
                             m1_live])[OHLC]
        m1_deep = m1_deep[~m1_deep.index.duplicated(keep='last')].sort_index()
    else:
        m1_deep = m1

    def canonical_upto(sec_tf: int, close_ts,
                       key: tuple | None = None) -> pd.DataFrame:
        """Canonical secondary bars fully closed by close_ts.

        tf >= 60: resample-from-1Min (deep series for coarse TFs).
        tf < 60: the 1Min resample would be an UPSAMPLE (minute OHLC
        relabeled as sub-minute bars — garbage the live refresher, which
        reloads native sub-minute REST bars, never sees). Serve the
        shadow's own seed-warmup + decision-time series instead.
        """
        if sec_tf < 60:
            src = subminute_src.get(key)
            if src is None or len(src) == 0:
                return pd.DataFrame()
            return src[src.index + pd.Timedelta(seconds=sec_tf) <= close_ts]
        base = (m1_deep if sec_tf >= RE.COARSE_SECONDARY_SECONDS else m1)
        src = base[base.index + pd.Timedelta(minutes=1) <= close_ts]
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

    # Step-0 per-token gate telemetry: at each primary close, was each
    # required record present in the monitor's MERGED confluence view (the
    # exact set check_entry gated on)? tok_hits==0 for a token across the
    # whole window + its interpreter absent from the shadow's publish union
    # ⇒ the ribbon never RESOLVED (warmup/NaN/label drift) — a lying 0.
    # tok_hits>0 for every token but never simultaneous ⇒ a REAL closed gate.
    tok_hits = {t: 0 for t in need}
    n_closes = 0
    gate_open_closes = 0
    _gate_state = [False]
    refresh_rows: dict = {}      # key → rows in the LAST refresher rebuild

    for close_ts, kind, key, bar in events:
        _now_ref[0] = close_ts
        # 120s force-full re-true (the armed RORT_SHADOW_RETRUE_FORCE_FULL path).
        # STRICTLY `>`: a refresh must never run BEFORE the incremental close of a
        # bar its own canonical df already contains — the re-true would seed the
        # state through bar B and the close would then apply B a SECOND time
        # (double-count → corrupted state; cost hours on 2026-07-14). Live is
        # immune because its re-true RESETS state after the close, never before.
        while REFRESH_S > 0 and close_ts > next_refresh:  # REFRESH_S=0 → retired
            for (sec_tf, sec_sess), shadow in list(hub._shadow_engines.items()):
                cdf = canonical_upto(sec_tf, next_refresh,
                                     key=(sec_tf, sec_sess))
                refresh_rows[(sec_tf, sec_sess)] = len(cdf)
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
                cdf = canonical_upto(sec_tf, close_ts, key=key)
                if len(cdf):
                    hub._publish_mtf(key, shadow.recompute_confluence(cdf),
                                     closed_bar_start_ts=cdf.index[-1])
                    continue
            hub._publish_mtf(key, shadow.on_bar_close(bar),
                             closed_bar_start_ts=bar['timestamp'])
            continue

        # (the old first-shadow-only per-close dump is superseded by the
        # publish wrapper's change-only prints + the per-token summary below)

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
        n_closes += 1
        if need:
            cur = monitor._current_confluence
            for t in need:
                if t in cur:
                    tok_hits[t] += 1
            _open = need.issubset(cur)
            gate_open_closes += 1 if _open else 0
            if DBG and _open != _gate_state[0]:
                print(f"    [diag] GATE {'OPEN' if _open else 'CLOSED'} at "
                      f"{close_ts} missing={sorted(need - cur)}")
                _gate_state[0] = _open
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

    # ── Fail-loud classification (Step 1, Plan_Harness_Secondary_TF_Gap):
    # a reported 0 must mean "the engine genuinely took no trades".
    # UNRESOLVED = something prevented a required ribbon from resolving at
    # all (empty warmup / all-NaN / missing shadow / missing GEN pack) —
    # the ceiling is a lie and must not be printed as a number.
    # LABEL-DRIFT = the ribbon RESOLVED but emits a tf-label namespace that
    # can never intersect the required token (e.g. shadow '10Sec-…' vs
    # required '10S-…'). The replay is faithfully reproducing LIVE's
    # behavior (same classes), so the number stands as the live ceiling —
    # but it flags a live-ENGINE bug suspect, never a quiet real-0.
    own_sess = session if RE.MTF_SESSION_SHADOWS else 'RTH'
    unresolved: list = []
    label_drift: list = []
    for t in sorted(need):
        if t.startswith('GEN-'):
            if not monitor._general_packs:
                unresolved.append(f"{t}: GEN-PACK-MISSING")
            continue
        lbl, _, rest = t.partition('-')
        ikey = rest.split('-', 1)[0]
        tf_sec = (RE._LABEL_TO_TF_SECONDS.get(lbl)
                  or RE._LABEL_TO_TF_SECONDS.get(lbl.lower()))
        if not tf_sec:
            unresolved.append(f"{t}: UNPARSEABLE-TF-LABEL")
            continue
        if tf_sec == tf_s:
            continue                 # own-TF record, not a shadow ribbon
        skey = (tf_sec, own_sess)
        if skey not in hub._shadow_engines:
            unresolved.append(f"{t}: NO-SHADOW-ENGINE")
            continue
        union = pub_union.get(skey, set())
        same_interp = [r for r in union
                       if len(r.split('-', 2)) == 3
                       and r.split('-', 2)[1] == ikey]
        if not same_interp:
            why = ('EMPTY-WARMUP-NO-EVENTS'
                   if (shadow_warm_rows.get(skey, 0) == 0
                       and pub_n.get(skey, 0) == seed_pubs.get(skey, 0))
                   else 'RIBBON-NEVER-RESOLVED')
            unresolved.append(f"{t}: {why}")
            continue
        if not any(r.split('-', 2)[0].upper() == lbl.upper()
                   for r in same_interp):
            emitted = sorted({r.split('-', 2)[0] for r in same_interp})
            label_drift.append(
                f"{t}: ribbon resolved but shadow emits prefix "
                f"{emitted} — required '{lbl}' can NEVER match "
                f"(live-engine label drift suspect)")

    diag = {
        'required': sorted(need),
        'gen_packs': len(gen_packs),
        'n_closes': n_closes,
        'gate_open_closes': gate_open_closes,
        'tok_hits': dict(tok_hits),
        'unresolved': unresolved,
        'label_drift': label_drift,
        'keys': {},
    }
    for key in hub._shadow_engines:
        diag['keys'][key] = {
            'seed_pubs': seed_pubs.get(key, 0),
            'pubs': pub_n.get(key, 0),
            'union': sorted(pub_union.get(key, set())),
            'final': sorted(hub._mtf_confluence.get(key, set())),
            'refresh_rows_last': refresh_rows.get(key),
        }
    if DBG:
        print(f"    [diag] ===== sid={sid} summary =====")
        print(f"    [diag] primary closes={n_closes} "
              f"gate_open_closes={gate_open_closes} "
              f"entries={len(entries)} exits={len(exits)}")
        for u in unresolved:
            print(f"    [diag] UNRESOLVED {u}")
        for w in label_drift:
            print(f"    [diag] LABEL-DRIFT {w}")
        for t in sorted(need):
            print(f"    [diag] token {t!r}: active at "
                  f"{tok_hits[t]}/{n_closes} primary closes")
        for key, d in diag['keys'].items():
            print(f"    [diag] key {key}: pubs={d['pubs']} "
                  f"(seed {d['seed_pubs']}) "
                  f"refresh_rows_last={d['refresh_rows_last']}")
            print(f"    [diag]   union: {d['union']}")
        for key, sh in hub._shadow_engines.items():
            try:
                vals = sh.indicators.get_values()
                nans = sorted(k for k, v in vals.items()
                              if v is None or (isinstance(v, float)
                                               and v != v))
                print(f"    [diag] key {key} NaN-at-end: "
                      f"{nans if nans else 'none'}")
            except Exception as e:  # noqa: BLE001
                print(f"    [diag] key {key} NaN census failed: {e}")
    return {'entries': entries, 'exits': exits, 'diag': diag}


# ─────────────────────────────── pairing ─────────────────────────────────────

def pair(a: list, b: list, tol: float) -> int:
    """Maximum 1:1 matches within tol seconds — optimal two-pointer over sorted
    events, mirroring the dashboard's `_pair_phantom_missed`. NOT greedy
    nearest-first, which UNDERSTATES parity (Brandon audit 07-15: a=[4,8],
    b=[0,5], tol=5 admits a 2-match assignment; nearest-first finds only 1)."""
    aa, bb = sorted(a), sorted(b)
    i = j = n = 0
    while i < len(aa) and j < len(bb):
        diff = (bb[j] - aa[i]).total_seconds()
        if abs(diff) <= tol:
            n += 1
            i += 1
            j += 1
        elif diff < 0:   # bb[j] earlier than aa[i] and out of tol → unmatched
            j += 1
        else:            # aa[i] earlier than bb[j] and out of tol → unmatched
            i += 1
    return n


def combined_pct(a: list, b: list, tol: float) -> float:
    if not a and not b:
        return float('nan')
    return 200.0 * pair(a, b, tol) / (len(a) + len(b))


def lanes(sb, sid: int, since: datetime, until: datetime):
    def _bt(col):
        # Backtest rows windowed INDEPENDENTLY by `col`. Brandon audit 07-15:
        # exits must be filtered by exit_fill_ts, NOT inherited from the
        # entry-windowed rows — else an exit whose ENTRY preceded the window is
        # wrongly dropped, and an exit AFTER the window is wrongly kept.
        return (sb.table('trades').select('entry_fill_ts,exit_fill_ts')
                .eq('strategy_id', sid).eq('data_source', 'backtest_rest_hifi')
                .gte(col, since.isoformat())
                .lte(col, until.isoformat()).execute()).data
    P = lambda s: datetime.fromisoformat(s.replace('Z', '+00:00'))  # noqa: E731
    bt_e = [P(t['entry_fill_ts']) for t in _bt('entry_fill_ts')
            if t.get('entry_fill_ts')]
    bt_x = [P(t['exit_fill_ts']) for t in _bt('exit_fill_ts')
            if t.get('exit_fill_ts')]
    al = (sb.table('alerts').select('side,fill_ts')
          .eq('strategy_id', sid)
          .gte('timestamp', (since - timedelta(minutes=30)).isoformat())
          .lte('timestamp', (until + timedelta(minutes=30)).isoformat())
          .execute()).data
    live_e = [P(a['fill_ts']) for a in al
              if a['side'] == 'entry' and a['fill_ts']
              and since <= P(a['fill_ts']) <= until]
    live_x = [P(a['fill_ts']) for a in al
              if a['side'] == 'exit' and a['fill_ts']
              and since <= P(a['fill_ts']) <= until]
    return bt_e, bt_x, live_e, live_x


def _health_scores(since: datetime, until: datetime, tol: float,
                   replay_fills: dict | None, only_sids: list) -> dict:
    """Call the dashboard's own get_strategy_health at one tolerance, either
    normally (LIVE) or with replay fills injected (REPLAY ceiling). Returns
    {sid: combined_pct}. Byte-comparable to the Strategy Health page."""
    SH._REPLAY_FILLS_OVERRIDE = replay_fills   # None → real alerts (LIVE)
    # Scope the fleet-wide fetches to just our targets, or get_strategy_health
    # times out under load (the recurring Supabase statement-timeout).
    SH._ONLY_SIDS = only_sids
    try:
        r = SH.get_strategy_health(
            user=None, window_hours=3,
            start=since.isoformat(), end=until.isoformat(),
            tolerance_seconds=tol)
    finally:
        SH._REPLAY_FILLS_OVERRIDE = None
        SH._ONLY_SIDS = None
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
    live = {t: _health_scores(since, until, t, None, sids) for t in TOLERANCES}
    rep = {t: _health_scores(since, until, t, replay_fills, sids)
           for t in TOLERANCES}
    corr = ({t: _health_scores(since, until, t, corr_fills, sids)
             for t in TOLERANCES} if args.corrected else None)

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
    footnotes: list = []
    for sid in sids:
        if sid not in raw:
            continue
        r = raw[sid]
        d = r.get('diag') or {}
        unres = d.get('unresolved') or []
        drift = d.get('label_drift') or []
        bt_e, bt_x, live_e, live_x = lanes(sb, sid, since, until)
        vals = [v for v in (combined_pct(r['entries'], live_e, 10),
                            combined_pct(r['exits'], live_x, 10)) if v == v]
        selfchk = f"{(sum(vals)/len(vals)):4.0f}%" if vals else " n/a"
        if unres:
            # Step 1 fail-loud: never print a ceiling number the replay
            # couldn't actually resolve — a silent 0 here is a lie.
            ceil_cells = f"{'UNRES':>6} {'UNRES':>6}"
            for u in unres:
                footnotes.append(f"  {sid} UNRESOLVED — {u}")
        else:
            ceil_cells = f"{cell(rep[10].get(sid))} {cell(rep[60].get(sid))}"
        for w in drift:
            footnotes.append(f"  {sid} ⚠ LABEL-DRIFT — {w}")
        line = (f"{sid:>4}  {cell(live[10].get(sid))} {cell(live[60].get(sid))}   "
                f"{ceil_cells}")
        if args.corrected:
            line += f"   {cell(corr[10].get(sid))} {cell(corr[60].get(sid))}"
        line += f"   {selfchk:>5}"
        print(line)
    if footnotes:
        print()
        for f in footnotes:
            print(f)
    print("\nRead: live < CEILING(dtime) ⇒ operational loss (recoverable). "
          "CEILING(corr) high but CEILING(dtime) lower ⇒ WS/REST data-timing "
          "floor. CEILING(corr) ALSO low ⇒ a REAL logic bug. self<90% ⇒ replay "
          "didn't reproduce live (ops-contaminated window). UNRES ⇒ the replay "
          "could not resolve a required ribbon — fix the harness input, don't "
          "trust a 0. LABEL-DRIFT ⇒ ribbon resolved but its record namespace "
          "can never match the gate — live-engine bug suspect, spin out.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
