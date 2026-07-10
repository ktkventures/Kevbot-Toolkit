"""Gate Parity Harness — diagnose live↔backtest cross-TF gate divergence.

Phase 2 / H1. For a gated strategy, classify each LIVE entry by whether its
cross-TF (e.g. 2m) confluence gate is satisfied under:

    PB (Previous Bar)  — the previous *closed* secondary bar's state.
                         This is what BACKTEST uses (services.py shifts the
                         secondary TF forward one period + ffill).
    CB (Current Bar)   — the secondary bar the entry falls inside, at its
                         close. PB and CB differ by exactly one secondary bar.

Then pairs each live entry against the BACKTEST lane (`trades` table,
data_source LIKE 'backtest%') at ±tolerance and flags phantoms (live entry
with no backtest match). The decisive signal: do phantoms cluster on
"CB-pass / PB-fail"? That is the off-by-one fingerprint — live behaving as
if CB while backtest gates on PB.

This is the data layer behind the "Gate Parity" detail-page tab. Read-only.

Usage:
    python gate_parity_harness.py --sid 303 --window-hours 8
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
load_dotenv(override=True)

import pack_registry
pack_registry.scan_and_load_all()

from db import get_admin_client, _row_to_strategy
from ralph_engine import StrategyMonitor, SymbolHub, _LABEL_TO_TF_SECONDS
from data_loader import load_market_data, resample_to_timeframe, get_tf_label

TF_LABEL = {60: '1Min', 120: '2Min', 300: '5Min', 900: '15Min',
            3600: '1Hour', 14400: '4Hour', 86400: '1Day'}


def _parse_iso(s) -> Optional[datetime]:
    if not s:
        return None
    if isinstance(s, datetime):
        return s if s.tzinfo else s.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(s).replace('Z', '+00:00'))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def load_context(sid: int) -> dict:
    """Resolve the strategy's gate condition, entry trigger, and the
    secondary-TF shadow that computes the gate's interpreter."""
    c = get_admin_client()
    row = c.table('strategies').select('*').eq('id', sid).execute().data[0]
    strat = _row_to_strategy(row)
    conf = list(strat.get('confluence') or [])
    if not conf:
        raise SystemExit(f"sid {sid} has no confluence gate — nothing to diagnose")
    gate = conf[0]                       # e.g. '2m-UT_BOT_V4-BULL_TREND'
    tf_label = gate.split('-', 1)[0]     # '2m'
    sec_tf = _LABEL_TO_TF_SECONDS.get(tf_label, 0)
    interp = gate.split('-', 2)[1]       # 'UT_BOT_V4'
    state = gate.split('-', 2)[2]        # 'BULL_TREND'

    mon = StrategyMonitor(strat, None, general_packs=[])
    hub = SymbolHub(strat['symbol']); hub.add_monitor(mon); hub.finalize_shadow_engines()
    shadow = next((sh for (_tf, _s), sh in hub._shadow_engines.items()
                   if _tf == sec_tf), None)
    if shadow is None:
        raise SystemExit(f"sid {sid}: no shadow engine for {sec_tf}s — gate TF not resolved")
    # Resolve the BASE trigger id (matches alerts.trigger_id). Modern strategies
    # store only entry_trigger_confluence_id; see _resolve_entry_trigger_base.
    entry_trigger, _cid = _resolve_entry_trigger_base(strat, row)
    return {
        'sid': sid, 'symbol': strat['symbol'], 'user_id': row['user_id'],
        'gate': gate, 'sec_tf': sec_tf, 'interp': interp, 'state': state,
        'entry_trigger': entry_trigger,
        'live_model': strat.get('live_model'), 'backtest_model': strat.get('backtest_model'),
        'shadow': shadow,
    }


def gate_state_series(ctx: dict, days: int = 1) -> pd.Series:
    """Closed secondary-bar state of the gate interpreter, indexed by 2m
    bar-start. Built from 10Sec base → resampled to the secondary TF, mirroring
    how the live shadow aggregates and how backtest resamples."""
    sec_tf = ctx['sec_tf']; shadow = ctx['shadow']; interp = ctx['interp']
    df1 = load_market_data(ctx['symbol'], days=days, timeframe='10Sec')
    sec = resample_to_timeframe(df1, TF_LABEL[sec_tf]).dropna(subset=['open'])
    shadow.warmup(sec.iloc[:20])
    out = {}
    for ts, r in sec.iloc[20:].iterrows():
        recs = shadow.on_bar_close({
            'open': float(r.open), 'high': float(r.high), 'low': float(r.low),
            'close': float(r.close), 'volume': float(r.volume),
            'timestamp': ts.isoformat()})
        st = next((x.split('-', 2)[2] for x in recs if f'-{interp}-' in x), None)
        out[pd.Timestamp(ts)] = st
    return pd.Series(out).sort_index()


def classify(ctx: dict, window_hours: float, tol_sec: float = 5.0) -> dict:
    c = get_admin_client()
    now = datetime.now(timezone.utc)
    lo = now - timedelta(hours=window_hours)
    lo_iso = lo.isoformat()

    states = gate_state_series(ctx)
    buckets = states.index
    want = ctx['state']

    def pb_cb(ts: datetime):
        """Return (PB_state, CB_state) for an entry at ts."""
        bs = pd.Timestamp(ts).floor(f'{ctx["sec_tf"]}s')
        # CB = bucket containing ts; PB = the bucket before it.
        cb = states.get(bs, None)
        prev = buckets[buckets < bs]
        pb = states.get(prev[-1], None) if len(prev) else None
        return pb, cb

    # live entries = alerts on the entry trigger
    al = c.table('alerts').select('timestamp,fill_ts,trigger_id').eq('strategy_id', ctx['sid'])\
        .eq('trigger_id', ctx['entry_trigger']).gte('timestamp', lo_iso).limit(5000).execute().data
    live = sorted(_parse_iso(a.get('fill_ts') or a.get('timestamp')) for a in al if _parse_iso(a.get('fill_ts') or a.get('timestamp')))

    # backtest entries (trades table, backtest lane)
    tr = c.table('trades').select('entry_fill_ts,entry_trigger_ts,data_source').eq('strategy_id', ctx['sid'])\
        .like('data_source', 'backtest%').gte('entry_fill_ts', lo_iso).limit(5000).execute().data
    bt = sorted(_parse_iso(t.get('entry_fill_ts') or t.get('entry_trigger_ts')) for t in tr if _parse_iso(t.get('entry_fill_ts') or t.get('entry_trigger_ts')))

    # two-pointer pairing live↔bt within ±tol
    bt_ts = [b.timestamp() for b in bt]
    rows = []
    for e in live:
        et = e.timestamp()
        paired = any(abs(b - et) <= tol_sec for b in bt_ts)
        pb, cb = pb_cb(e)
        rows.append({'ts': e, 'pb': pb, 'cb': cb,
                     'pb_pass': pb == want, 'cb_pass': cb == want, 'paired': paired})
    return {'rows': rows, 'live_n': len(live), 'bt_n': len(bt), 'want': want}


# ── Live-model splice reconstruction ────────────────────────────────────────
# Carbon-copy of ws_rest_spliced: at replay time `scrub_ts`, every closed bar
# shows REST (latest, post-correction) values EXCEPT the WS tip — the forming
# bar plus any bar whose close is within `grace_sec` of scrub_ts (not yet
# REST-verified). Grace defaults are the live model's ACTUAL values:
#   primary <60s   -> 4s alert-grace (rest_verifier.py:744, TF-aware; 10Sec=4s)
#                     with a 5s per-bar backstop (ralph_engine.py:2516)
#   secondary >=60s -> 30s per-bar drift grace (ralph_engine.py:2516)
# Per-strategy `config.grace_seconds` overrides the primary alert-grace.
LIVE_GRACE_DEFAULT = {'primary_10s': 4.0, 'primary_bar_backstop': 5.0, 'secondary': 30.0}


def splice_alert_lens(symbol: str, tf_seconds: int, scrub_ts: datetime,
                      grace_sec: float, day: Optional[str] = None) -> list:
    """Reconstruct the live engine's bar history at replay time `scrub_ts`.

    Pulls live_bars (WS first-write `first_*` + REST latest), and for each bar
    with bar_start <= scrub_ts picks WS values if it's the WS tip (forming, or
    closed < grace_sec ago) else REST. Returns dicts with a 'provenance' tag.
    """
    c = get_admin_client()
    # Fetch the window ENDING at scrub_ts (most-recent bars first, then reverse)
    # so the WS tip near scrub_ts is always included regardless of row caps.
    rows = c.table('live_bars').select(
        'bar_start,first_open,first_high,first_low,first_close,first_volume,'
        'open,high,low,close,volume,source'
    ).eq('symbol', symbol).eq('timeframe_seconds', tf_seconds)\
     .lte('bar_start', scrub_ts.isoformat())\
     .order('bar_start', desc=True).limit(500).execute().data
    rows = list(reversed(rows))
    out = []
    for r in rows:
        bs = _parse_iso(r['bar_start'])
        if bs is None or bs > scrub_ts:
            continue
        bar_close = bs + timedelta(seconds=tf_seconds)
        forming = bar_close > scrub_ts
        is_ws = forming or (scrub_ts - bar_close).total_seconds() < grace_sec
        pre = 'first_' if is_ws else ''
        def g(k):
            v = r.get(pre + k)
            return float(v) if v is not None else (float(r.get(k)) if r.get(k) is not None else 0.0)
        out.append({'bar_start': bs, 'open': g('open'), 'high': g('high'),
                    'low': g('low'), 'close': g('close'), 'volume': g('volume'),
                    'provenance': ('WS-forming' if forming else 'WS-tip') if is_ws else 'REST',
                    'ws_close': r.get('first_close'), 'rest_close': r.get('close')})
    return out


def demo_splice(sid: int):
    """Prove the splice on a real strategy: show the tip vs body at a scrub
    time, and flag bars where WS != REST."""
    ctx = load_context(sid)
    sym = ctx['symbol']
    # pick a scrub time: the latest live_bar for the primary TF today, minus a bit
    c = get_admin_client()
    from ralph_engine import _LABEL_TO_TF_SECONDS
    prim = _LABEL_TO_TF_SECONDS.get(_row_to_strategy(
        c.table('strategies').select('*').eq('id', sid).execute().data[0])['timeframe'].lower(), 10)
    last = c.table('live_bars').select('bar_start').eq('symbol', sym)\
        .eq('timeframe_seconds', prim).order('bar_start', desc=True).limit(1).execute().data
    if not last:
        print('no live_bars for', sym, prim); return
    scrub = _parse_iso(last[0]['bar_start']) + timedelta(seconds=prim - 2)  # mid-forming
    for tf, grace, lbl in [(prim, LIVE_GRACE_DEFAULT['primary_10s'], 'PRIMARY'),
                           (ctx['sec_tf'], LIVE_GRACE_DEFAULT['secondary'], 'SECONDARY/gate')]:
        bars = splice_alert_lens(sym, tf, scrub, grace)
        ws = [b for b in bars if b['provenance'] != 'REST']
        differ = [b for b in bars if b['ws_close'] is not None and b['rest_close'] is not None
                  and abs(float(b['ws_close']) - float(b['rest_close'])) > 1e-9]
        print(f"\n=== {lbl} {tf}s | scrub={scrub:%H:%M:%S} grace={grace}s ===")
        print(f"  {len(bars)} bars <= scrub | WS-tip bars: {len(ws)} | bars where WS!=REST: {len(differ)}")
        print("  last 6 bars (provenance / chosen close):")
        for b in bars[-6:]:
            tag = b['provenance']
            print(f"    {b['bar_start']:%H:%M:%S}  {tag:<11} close={b['close']:.2f}"
                  f"  (ws={b['ws_close']} rest={b['rest_close']})")


def _resolve_entry_trigger_base(strat: dict, row: Optional[dict] = None) -> tuple:
    """Resolve the BASE trigger id that the live `alerts.trigger_id` column and
    the enriched df's ``trig_<id>`` column actually use.

    Modern strategies leave the top-level ``entry_trigger`` unset and carry the
    confluence id in ``entry_trigger_confluence_id`` (e.g.
    ``swing_123_default_bull_c2``), while ``alerts.trigger_id`` stores the base
    id (``sw123_bull_c2``). Legacy strategies already store the base id in
    ``entry_trigger``. We therefore prefer deriving the base id from the
    confluence id (canonical — same path strategy_factory uses to set
    ``entry_trigger``) and fall back to the stored ``entry_trigger``.

    Returns ``(base_trigger_id, entry_trigger_confluence_id)``.
    """
    cfg = (row.get('config') if row else None) or {}
    entry_cid = (strat.get('entry_trigger_confluence_id')
                 or cfg.get('entry_trigger_confluence_id'))
    base = strat.get('entry_trigger') or cfg.get('entry_trigger')
    if entry_cid:
        try:
            from alerts import _get_base_trigger_id
            resolved = _get_base_trigger_id(entry_cid)
            if resolved:
                base = resolved
        except Exception:
            pass
    return base, entry_cid


def _parse_gate(gate: str) -> Optional[dict]:
    """Parse a confluence gate string like '4h-STRAT_ASSISTANT-INSIDE' into its
    parts plus the enriched-df ribbon column names.

    The ribbon column suffix is derived the SAME way the engine's
    `prepare_data_with_indicators` derives it — ``get_tf_label(canonical_tf)``
    (e.g. '1Min' -> '1m') — NOT from the raw confluence TF token, whose case can
    differ ('1M' gate token vs '1m' column). This is what previously made
    modern gates raise 'expected ribbon column ... not in enriched df'.
    """
    parts = gate.split('-', 2)
    if len(parts) < 3:
        return None
    tf_part, interp, want = parts[0], parts[1], parts[2]
    sec_tf = _LABEL_TO_TF_SECONDS.get(tf_part, 0)
    sec_label = TF_LABEL.get(sec_tf)               # canonical, e.g. '1Min'
    df_tf = get_tf_label(sec_label) if sec_label else tf_part.lower()
    return {'gate': gate, 'tf': tf_part, 'interp': interp, 'want': want,
            'sec_tf': sec_tf, 'sec_label': sec_label,
            'pb_col': f'{interp}__{df_tf}', 'cb_col': f'_spec_{interp}__{df_tf}'}


def _state_str(v):
    """JSON-safe gate state: NaN/None -> None, else str."""
    try:
        if v is None or pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return str(v)


def build_gate_parity_view(sid: int, hours: float = 4.0,
                           session: Optional[str] = None) -> dict:
    """Theoretical, truth-reliable Gate Parity data for the detail-page tab.

    Runs the REAL engine pipeline once (`prepare_data_with_indicators` +
    `unified_trades`) for the backtest lens, extracting the engine's own
    PB ribbon (`<interp>__<tf>`) and CB ribbon (`_spec_<interp>__<tf>`) for
    EVERY confluence gate, the theoretical BT entries, and the actual live
    entries. Per live entry it reports each gate's state under PB/CB (from the
    engine columns at that bar) and whether it pairs with a theoretical BT
    entry. JSON-serializable; the FastAPI route is a thin pass-through and the
    frontend only renders.

    Backward-compatible: the top-level ``meta.gate`` / ``ribbon`` /
    ``live_rows[].{pb,cb,pb_pass,cb_pass}`` fields still describe the FIRST gate
    (conf[0]); the full multi-gate breakdown is added under ``per_gate`` and
    ``live_rows[].gates``. Ungated strategies are handled gracefully (no gate
    ribbon, still reports trigger + entries) instead of raising.
    """
    import services as svc
    from db import set_admin_user_context
    c = get_admin_client()
    row = c.table('strategies').select('*').eq('id', sid).execute().data[0]
    set_admin_user_context(row['user_id'])
    strat = _row_to_strategy(row)
    conf = list(strat.get('confluence') or [])
    primary_tf = strat.get('timeframe')

    # Resolve the base trigger id that live alerts + the df trig_ column use.
    entry_trigger, entry_cid = _resolve_entry_trigger_base(strat, row)

    # Parse every gate up-front (skips malformed records).
    gates = [g for g in (_parse_gate(x) for x in conf) if g]

    # Enrich against the UNION of every gate's secondary TF (canonical labels).
    sec_labels = []
    for gd in gates:
        if gd['sec_label'] and gd['sec_label'] not in sec_labels:
            sec_labels.append(gd['sec_label'])

    # Use the strategy's ACTUAL session — not a hardcoded guess — so the
    # theoretical backtest fires in the same window the live engine does.
    session = session or strat.get('trading_session') or 'RTH'
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    df = svc.prepare_data_with_indicators(
        strat['symbol'], timeframe=primary_tf, secondary_tfs=tuple(sec_labels),
        start_date=start, end_date=end, session=session, strat=strat)
    # The cache-backed path can return the full cached range regardless of
    # start/end — clip to the requested window so counts are window-accurate.
    if len(df):
        _idx = df.index.tz_convert('UTC') if df.index.tz is not None else df.index.tz_localize('UTC')
        df = df[(_idx >= pd.Timestamp(start)) & (_idx <= pd.Timestamp(end))]

    # Mark which gates actually resolved to df columns (defensive — a missing
    # ribbon column now degrades gracefully instead of raising).
    for gd in gates:
        gd['resolved'] = gd['pb_col'] in df.columns and gd['cb_col'] in df.columns

    # Theoretical BT entries (fresh, current logic) — the engine's own trades.
    # unified_trades enforces ALL gates jointly, so this count is strategy-level.
    trades_bt = svc.unified_trades(df, strat)
    bt_entry_ts = sorted(_parse_iso(t) for t in
                         (trades_bt['entry_time'] if 'entry_time' in trades_bt.columns
                          else trades_bt.get('entry_fill_ts', [])) if _parse_iso(t)) \
        if trades_bt is not None and len(trades_bt) else []

    # Actual live entries (carry historical logic — labeled as such).
    live_ts = []
    if entry_trigger:
        al = c.table('alerts').select('timestamp,fill_ts').eq('strategy_id', sid)\
            .eq('trigger_id', entry_trigger).gte('timestamp', start.isoformat())\
            .limit(5000).execute().data
        live_ts = sorted(_parse_iso(a.get('fill_ts') or a.get('timestamp')) for a in al
                         if _parse_iso(a.get('fill_ts') or a.get('timestamp')))

    # Gate state at a timestamp from a gate's PB/CB columns (asof).
    idx = df.index
    def gate_at(ts, pb_col, cb_col):
        pos = idx.searchsorted(pd.Timestamp(ts), side='right') - 1
        if pos < 0:
            return None, None
        r = df.iloc[pos]
        return r.get(pb_col), r.get(cb_col)

    def paired(ts, pool, tol=5.0):
        t = ts.timestamp()
        return any(abs(p.timestamp() - t) <= tol for p in pool)

    # Per-live-entry breakdown across ALL gates.
    rows = []
    for ts in live_ts:
        gate_states = {}
        for gd in gates:
            pb_raw, cb_raw = gate_at(ts, gd['pb_col'], gd['cb_col'])
            gate_states[gd['gate']] = {
                'pb': _state_str(pb_raw), 'cb': _state_str(cb_raw),
                'pb_pass': pb_raw == gd['want'], 'cb_pass': cb_raw == gd['want']}
        pbt = paired(ts, bt_entry_ts)
        # Top-level fields describe the first gate (backward compat).
        first = gate_states.get(gates[0]['gate']) if gates else None
        rows.append({
            'ts': ts.isoformat(),
            'pb': first['pb'] if first else None,
            'cb': first['cb'] if first else None,
            'pb_pass': bool(first['pb_pass']) if first else False,
            'cb_pass': bool(first['cb_pass']) if first else False,
            'paired_bt': pbt,
            'gates': gate_states,
        })

    # Per-gate summary. open% denominator = non-null states (mirrors frontend).
    per_gate = []
    for gd in gates:
        want = gd['want']
        if gd['resolved']:
            pb_dist = {str(k): int(v) for k, v in df[gd['pb_col']].value_counts().items()}
            cb_dist = {str(k): int(v) for k, v in df[gd['cb_col']].value_counts().items()}
        else:
            pb_dist, cb_dist = {}, {}
        pb_total = sum(pb_dist.values()) or 1
        cb_total = sum(cb_dist.values()) or 1
        live_pb_pass = sum(1 for r in rows if r['gates'].get(gd['gate'], {}).get('pb_pass'))
        live_cb_pass = sum(1 for r in rows if r['gates'].get(gd['gate'], {}).get('cb_pass'))
        # Phantom live entries (no paired theoretical-BT entry) this gate would
        # have BLOCKED — the fingerprint of the divergent gate. Two lenses:
        #   phantom_pb_fail: gate CLOSED under PB (what backtest gates on). NOTE
        #     coarse-gate PB (1Day/4Hour) is NaN-shallow over short windows —
        #     see the secondary-load follow-up — so PB counts under-report.
        #   phantom_cb_fail: gate CLOSED under CB (current bar). CB is reliably
        #     loaded, so this is the trustworthy "gate was actually closed at
        #     the live entry" signal used to pick the divergent gate.
        phantom_pb_fail = sum(1 for r in rows if not r['paired_bt']
                              and not r['gates'].get(gd['gate'], {}).get('pb_pass'))
        phantom_cb_fail = sum(1 for r in rows if not r['paired_bt']
                              and not r['gates'].get(gd['gate'], {}).get('cb_pass'))
        per_gate.append({
            'gate': gd['gate'], 'tf': gd['tf'], 'interp': gd['interp'],
            'want_state': want, 'resolved': gd['resolved'],
            'pb_dist': pb_dist, 'cb_dist': cb_dist,
            'pb_open_pct': round(100.0 * pb_dist.get(want, 0) / pb_total, 1),
            'cb_open_pct': round(100.0 * cb_dist.get(want, 0) / cb_total, 1),
            'live_n': len(rows), 'live_pb_pass': live_pb_pass, 'live_cb_pass': live_cb_pass,
            'phantom_pb_fail': phantom_pb_fail, 'phantom_cb_fail': phantom_cb_fail,
        })

    # Which gate most explains the phantoms (the divergent one), for the UI.
    # Use the CB signal — reliably loaded, unlike coarse-gate PB — and only
    # count RESOLVED gates so an unresolvable ribbon can't masquerade as closed.
    divergent_gate = None
    if per_gate:
        top = max(per_gate, key=lambda g: (g['phantom_cb_fail'] if g['resolved'] else -1))
        if top['resolved'] and top['phantom_cb_fail'] > 0:
            divergent_gate = top['gate']

    # Top-level ribbon = first gate (backward compat).
    g0 = gates[0] if gates else None
    if g0 and g0['resolved']:
        top_ribbon = {'pb_dist': {str(k): int(v) for k, v in df[g0['pb_col']].value_counts().items()},
                      'cb_dist': {str(k): int(v) for k, v in df[g0['cb_col']].value_counts().items()}}
    else:
        top_ribbon = {'pb_dist': {}, 'cb_dist': {}}

    return {
        'meta': {'sid': sid, 'symbol': strat['symbol'], 'primary_tf': primary_tf,
                 'gate': g0['gate'] if g0 else None,
                 'want_state': g0['want'] if g0 else None,
                 'entry_trigger': entry_trigger,
                 'entry_trigger_confluence_id': entry_cid,
                 'gates': [gd['gate'] for gd in gates],
                 'ungated': len(gates) == 0,
                 'divergent_gate': divergent_gate,
                 'live_model': strat.get('live_model'), 'backtest_model': strat.get('backtest_model'),
                 'session': session,
                 'window': [start.isoformat(), end.isoformat()], 'bars': len(df)},
        'ribbon': top_ribbon,
        'per_gate': per_gate,
        'entries': {'theoretical_bt': len(bt_entry_ts), 'live_actual': len(live_ts)},
        'live_rows': rows,
    }


def print_view(sid: int, hours: float):
    v = build_gate_parity_view(sid, hours)
    m = v['meta']
    print(f"sid {m['sid']} {m['symbol']} {m['primary_tf']} | gates={m['gates']} "
          f"| trigger={m['entry_trigger']} (cid={m['entry_trigger_confluence_id']}) "
          f"| live={m['live_model']} bt={m['backtest_model']}")
    print(f"window {m['window'][0][11:19]}–{m['window'][1][11:19]} | {m['bars']} bars "
          f"| ungated={m['ungated']} divergent_gate={m['divergent_gate']}")
    print(f"\nEntries: theoretical-BT={v['entries']['theoretical_bt']}  "
          f"live-actual={v['entries']['live_actual']}")
    rows = v['live_rows']
    ph = [r for r in rows if not r['paired_bt']]
    print(f"Live entries vs theoretical BT: paired={len(rows)-len(ph)} phantom={len(ph)}")
    if not v['per_gate']:
        print("  (ungated — no confluence gates to break down)")
    for gd in v['per_gate']:
        print(f"\n  [gate] {gd['gate']}  want={gd['want_state']}  resolved={gd['resolved']}")
        print(f"         PB open {gd['pb_open_pct']}% {gd['pb_dist']}")
        print(f"         CB open {gd['cb_open_pct']}% {gd['cb_dist']}")
        print(f"         of {gd['live_n']} live entries: PB-pass={gd['live_pb_pass']} "
              f"CB-pass={gd['live_cb_pass']} | phantoms this gate blocks: "
              f"CB-fail={gd['phantom_cb_fail']} (PB-fail={gd['phantom_pb_fail']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sid', type=int, required=True)
    ap.add_argument('--window-hours', type=float, default=8.0)
    ap.add_argument('--tolerance', type=float, default=5.0)
    ap.add_argument('--demo-splice', action='store_true',
                    help='Demonstrate the ws_rest_spliced reconstruction on this strategy')
    ap.add_argument('--view', action='store_true',
                    help='Build + print the theoretical Gate Parity view (endpoint core)')
    args = ap.parse_args()

    if args.demo_splice:
        demo_splice(args.sid)
        return
    if args.view:
        print_view(args.sid, args.window_hours)
        return

    ctx = load_context(args.sid)
    print(f"sid {ctx['sid']} {ctx['symbol']} | gate={ctx['gate']} | "
          f"entry={ctx['entry_trigger']} | live={ctx['live_model']} bt={ctx['backtest_model']}")
    res = classify(ctx, args.window_hours, args.tolerance)
    rows = res['rows']
    print(f"live entries={res['live_n']}  backtest entries={res['bt_n']}  "
          f"(want gate state={res['want']})\n")

    phantom = [r for r in rows if not r['paired']]
    paired = [r for r in rows if r['paired']]
    def frac(rs, key): return f"{sum(1 for r in rs if r[key])}/{len(rs)}" if rs else "0/0"

    print(f"{'entries':<10}{'n':>5}{'PB-pass':>10}{'CB-pass':>10}{'CBpass&PBfail':>15}")
    for label, rs in [('paired', paired), ('phantom', phantom), ('ALL', rows)]:
        cbpb = sum(1 for r in rs if r['cb_pass'] and not r['pb_pass'])
        print(f"{label:<10}{len(rs):>5}{frac(rs,'pb_pass'):>10}{frac(rs,'cb_pass'):>10}{cbpb:>15}")

    print("\nOff-by-one signal: of phantom entries, "
          f"{sum(1 for r in phantom if r['cb_pass'] and not r['pb_pass'])}/{len(phantom)} "
          "pass under CB but fail under PB (=> live behaving like CB while BT gates on PB).")
    print("\nsample phantom rows (ts, PB, CB):")
    for r in phantom[:10]:
        print(f"   {r['ts']:%H:%M:%S}  PB={str(r['pb']):<12} CB={str(r['cb']):<12} "
              f"pb_pass={int(r['pb_pass'])} cb_pass={int(r['cb_pass'])}")


if __name__ == '__main__':
    main()
