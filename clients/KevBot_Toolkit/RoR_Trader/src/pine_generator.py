"""Pine Script generator — RoR Trader strategy → TradingView Pine v6 strategy().

Modular, mirrors our pack system. Each user pack gets a PineEmitter that
hand-reimplements its indicator math + triggers/states in Pine, faithfully
matching our engine's seeding (validated for ema_pp_v3 + ut_bot_v4 against
TradingView: RTH trade parity 98.6%, indicator math exact). Adding a pack =
adding one emitter; every strategy using it then ports for free.

The orchestrator `generate_pine(strat)` assembles:
  header → helper lib → primary-pack indicators + entry/exit triggers →
  confluence gates (per-pack state fn via request.security, lookahead_off) →
  stop/target → orders → visuals (EMAs, gate bgcolor, heatmap table).

Fidelity rules baked in (see docs/tradingview/TradingView_Export_Spec.md):
  • EMAs seed at first close (hand-rolled rorEma, NOT ta.ema's SMA seed).
  • Stop ATR = EMA(span) of TR seeded at first TR (NOT Wilder); UT Bot ATR IS
    Wilder (alpha=1/period).
  • UT Bot BULL_TREND = close>stop AND not bullFlip (flip bar is BULL_FLIP).
  • Gate = last CLOSED secondary bar: request.security(lookahead_off), no [1].

Pure code generation — no engine state, no I/O beyond reading pack params.
"""
from __future__ import annotations

import json
import os
from typing import Optional

_PACKS_DIR = os.path.join(os.path.dirname(__file__), '..', 'user_packs')

# Map our timeframe labels to Pine timeframe strings (minutes, or seconds "Ns").
_TF_TO_PINE = {
    '10sec': '10S', '15sec': '15S', '30sec': '30S', '5sec': '5S',
    '1min': '1', '2min': '2', '3min': '3', '5min': '5', '15min': '15',
    '30min': '30', '1hour': '60', '2hour': '120', '4hour': '240',
    '1day': 'D', '1week': 'W',
}


def _pine_tf(label: str) -> str:
    return _TF_TO_PINE.get((label or '').lower().replace(' ', ''), '1')


def _gate_tf_to_pine(tf_lbl: str) -> str:
    """Confluence-record TF label ('2m','15m','1H','1D','10sec') → Pine tf."""
    import re
    m = re.match(r'(\d+)\s*(sec|s|min|m|hour|h|day|d|week|w)',
                 (tf_lbl or '').strip().lower())
    if not m:
        return '1'
    n, unit = m.group(1), m.group(2)
    if unit in ('sec', 's'):
        return f'{n}S'
    if unit in ('min', 'm'):
        return n
    if unit in ('hour', 'h'):
        return str(int(n) * 60)
    if unit in ('day', 'd'):
        return 'D' if n == '1' else f'{n}D'
    if unit in ('week', 'w'):
        return 'W'
    return '1'


def _pack_params(slug: str) -> dict:
    """Default params from a pack manifest's parameters_schema (best effort)."""
    path = os.path.join(_PACKS_DIR, slug, 'manifest.json')
    try:
        with open(path) as f:
            man = json.load(f)
    except Exception:
        return {}
    out = {}
    schema = man.get('parameters_schema') or man.get('parameters') or {}
    if isinstance(schema, dict):
        for k, v in schema.items():
            if isinstance(v, dict) and 'default' in v:
                out[k] = v['default']
            else:
                out[k] = v
    elif isinstance(schema, list):
        for item in schema:
            if isinstance(item, dict) and 'key' in item:
                out[item['key']] = item.get('default')
    return out


# ───────────────────────────────────────────────────────────────────────────
# Pine helper library (emitted once at top when required)
# ───────────────────────────────────────────────────────────────────────────
_HELPERS = {
    'rorEma': (
        "rorEma(float src, int period) =>\n"
        "    var float e = na\n"
        "    a = 2.0 / (period + 1)\n"
        "    e := na(e) ? src : a * src + (1.0 - a) * e\n"
        "    e"
    ),
    'rorTrueRange': (
        "rorTrueRange() =>\n"
        "    na(close[1]) ? (high - low)\n"
        "         : math.max(high - low, math.abs(high - close[1]),"
        " math.abs(low - close[1]))"
    ),
}


# ───────────────────────────────────────────────────────────────────────────
# Per-pack emitters
# ───────────────────────────────────────────────────────────────────────────
class PineEmitter:
    """Base: a pack's Pine reimplementation. Subclasses override the bits
    they support (primary triggers and/or gate states)."""
    slug = ''
    helpers: tuple = ()

    def __init__(self, params: Optional[dict] = None):
        self.params = {**_pack_params(self.slug), **(params or {})}

    # Inline setup for PRIMARY-TF use; returns Pine code lines (str).
    def emit_primary_setup(self) -> str:
        return ''

    # Bool Pine expr for an entry/exit trigger base (e.g. 'cross_short_up').
    def emit_trigger(self, base: str) -> Optional[str]:
        return None

    # A Pine function (code, fn_name) that returns the bool for an interpreter
    # state (e.g. 'BULL_TREND'), suitable for request.security wrapping.
    def emit_gate_function(self, state: str, fn_name: str) -> Optional[str]:
        return None


class EmaPpV3Emitter(PineEmitter):
    slug = 'ema_pp_v3'
    helpers = ('rorEma',)

    def _periods(self):
        p = self.params
        return (int(p.get('short_period', 8)), int(p.get('mid_period', 21)),
                int(p.get('long_period', 50)))

    def emit_primary_setup(self) -> str:
        s, m, l = self._periods()
        return (f"emaShort = rorEma(close, {s})\n"
                f"emaMid   = rorEma(close, {m})\n"
                f"emaLong  = rorEma(close, {l})")

    def emit_trigger(self, base: str) -> Optional[str]:
        # base like 'eppv3_cross_short_up' or 'cross_short_up'
        b = base.replace('eppv3_', '')
        m = {
            'cross_short_up':   "close > emaShort and close[1] <= emaShort[1]",
            'cross_short_down': "close < emaShort and close[1] >= emaShort[1]",
            'cross_mid_up':     "close > emaMid and close[1] <= emaMid[1]",
            'cross_mid_down':   "close < emaMid and close[1] >= emaMid[1]",
        }
        return m.get(b)


class UtBotV4Emitter(PineEmitter):
    slug = 'ut_bot_v4'
    helpers = ()

    def _params(self):
        p = self.params
        return int(p.get('atr_period', 10)), float(p.get('key_value', 1.0))

    def emit_gate_function(self, state: str, fn_name: str) -> Optional[str]:
        atr_p, key = self._params()
        st = state.upper()
        # The function computes the UT Bot trailing stop + flips, then returns
        # the requested interpreter state. Verbatim translation of
        # UtBotV4Incremental (Wilder ATR, 4-case ratchet).
        if st == 'BULL_TREND':
            ret = "(close > stop) and not bullFlip"
        elif st == 'BEAR_TREND':
            ret = "(close < stop) and not bearFlip"
        elif st == 'BULL_FLIP':
            ret = "bullFlip"
        elif st == 'BEAR_FLIP':
            ret = "bearFlip"
        else:
            return None
        return (
            f"{fn_name}() =>\n"
            f"    var float atr  = na\n"
            f"    var float stop = na\n"
            f"    var float pc   = na\n"
            f"    bool bullFlip = false\n"
            f"    bool bearFlip = false\n"
            f"    if na(stop)\n"
            f"        tr = high - low\n"
            f"        atr  := tr\n"
            f"        stop := close - {key} * tr\n"
            f"        pc   := close\n"
            f"    else\n"
            f"        tr = math.max(high - low, math.abs(high - pc),"
            f" math.abs(low - pc))\n"
            f"        atr := atr + (1.0 / {atr_p}) * (tr - atr)\n"
            f"        nLoss = {key} * atr\n"
            f"        ps = stop\n"
            f"        float ns = na\n"
            f"        if close > ps and pc > ps\n"
            f"            ns := math.max(ps, close - nLoss)\n"
            f"        else if close < ps and pc < ps\n"
            f"            ns := math.min(ps, close + nLoss)\n"
            f"        else if close > ps\n"
            f"            ns := close - nLoss\n"
            f"        else\n"
            f"            ns := close + nLoss\n"
            f"        bullFlip := pc <= ps and close > ns\n"
            f"        bearFlip := pc >= ps and close < ns\n"
            f"        stop := ns\n"
            f"        pc   := close\n"
            f"    {ret}"
        )


EMITTERS = {e.slug: e for e in (EmaPpV3Emitter, UtBotV4Emitter)}


def _emitter_for_trigger(trigger: str) -> Optional[PineEmitter]:
    if trigger.startswith('eppv3_'):
        return EmaPpV3Emitter()
    return None


def _emitter_for_interp(interp: str) -> Optional[PineEmitter]:
    cls = EMITTERS.get(interp.lower())
    return cls() if cls else None


# ───────────────────────────────────────────────────────────────────────────
# Stop emitter
# ───────────────────────────────────────────────────────────────────────────
def _emit_stop(stop_config: dict) -> tuple:
    """Return (setup_lines, stop_level_expr, helpers_needed). stop_level_expr
    uses `close` (entry fill) — snapshotted at entry by the orchestrator."""
    method = (stop_config or {}).get('method', 'atr')
    if method == 'atr':
        mult = float(stop_config.get('atr_mult', 1.5))
        period = int(stop_config.get('atr_period', 14))
        setup = (f"stopAtrLen = {period}\n"
                 f"stopMult   = {mult}\n"
                 f"atrStopSeries = rorEma(rorTrueRange(), stopAtrLen)")
        return setup, "close - stopMult * atrStopSeries", ('rorEma', 'rorTrueRange')
    if method == 'fixed_dollar':
        amt = float(stop_config.get('dollar_amount', 1.0))
        return f"stopDist = {amt}", "close - stopDist", ()
    if method == 'percentage':
        pct = float(stop_config.get('percentage', 0.5))
        return f"stopPct = {pct}", f"close - close * (stopPct / 100.0)", ()
    # swing / unknown → ATR fallback (matches our engine's fallback)
    setup = ("stopAtrLen = 14\nstopMult = 1.5\n"
             "atrStopSeries = rorEma(rorTrueRange(), stopAtrLen)")
    return setup, "close - stopMult * atrStopSeries", ('rorEma', 'rorTrueRange')


# ───────────────────────────────────────────────────────────────────────────
# Orchestrator
# ───────────────────────────────────────────────────────────────────────────
def generate_pine(strat: dict) -> str:
    """strat: the strategy dict (with .config). Returns Pine v6 source.

    Raises ValueError with a clear message if a required pack has no emitter
    yet (so the UI can say exactly which pack to add)."""
    cfg = strat.get('config', strat) or {}
    name = strat.get('name') or f"sid {strat.get('id', '?')}"
    symbol = strat.get('symbol') or cfg.get('symbol') or 'SPY'
    primary_tf = cfg.get('timeframe') or strat.get('timeframe') or '1Min'
    entry_t = cfg.get('entry_trigger')
    exit_t = cfg.get('exit_trigger')
    gates = list(cfg.get('confluence') or [])
    stop_cfg = cfg.get('stop_config') or {}
    session = cfg.get('trading_session') or cfg.get('session') or 'RTH'

    if not entry_t:
        raise ValueError("strategy has no entry_trigger")

    prim = _emitter_for_trigger(entry_t)
    if prim is None:
        raise ValueError(
            f"no Pine emitter for entry trigger '{entry_t}'. "
            f"Add an emitter for its pack to pine_generator.EMITTERS.")
    entry_expr = prim.emit_trigger(entry_t)
    if entry_expr is None:
        raise ValueError(f"emitter {prim.slug} can't emit trigger '{entry_t}'")
    exit_expr = prim.emit_trigger(exit_t) if exit_t else None

    helpers_needed = set(prim.helpers)
    lines = []
    L = lines.append

    # ── header ──
    L("//@version=6")
    L(f"// Auto-generated by RoR Trader → TradingView export. Strategy: {name}")
    L(f"// Symbol {symbol} · primary {primary_tf} · session {session}")
    L("// Faithful port: hand-rolled indicator seeding to match our engine.")
    L("// Seconds charts hold only a few days of history on TradingView;")
    L("// extended-hours tape is sparse (use RTH for the cleanest parity).")
    L(f'strategy("{name} [RoR port]", overlay=true, '
      "default_qty_type=strategy.fixed, default_qty_value=1, "
      "process_orders_on_close=true, calc_on_every_tick=false, pyramiding=0)")
    L("")

    # ── gates: build functions + security calls ──
    gate_fns = []
    gate_calls = []
    gate_meta = []  # (label, varname) for the heatmap
    for i, rec in enumerate(gates):
        rec_clean = rec.replace('[CB]', '').replace('[PB]', '')
        parts = rec_clean.split('-', 2)
        if len(parts) != 3:
            continue
        tf_lbl, interp, state = parts
        em = _emitter_for_interp(interp)
        if em is None:
            raise ValueError(
                f"no Pine emitter for gate interpreter '{interp}' "
                f"(record '{rec}'). Add an emitter to pine_generator.EMITTERS.")
        fn_name = f"f_gate_{i}"
        fn_code = em.emit_gate_function(state, fn_name)
        if fn_code is None:
            raise ValueError(
                f"emitter {em.slug} can't emit state '{state}' (record '{rec}')")
        helpers_needed.update(em.helpers)
        gate_fns.append(fn_code)
        pine_tf = _gate_tf_to_pine(tf_lbl)
        gate_calls.append(
            f'gate_{i} = request.security(syminfo.tickerid, "{pine_tf}", '
            f'{fn_name}(), lookahead=barmerge.lookahead_off)')
        gate_meta.append((rec_clean, f"gate_{i}"))

    # ── stop ──
    stop_setup, stop_expr, stop_helpers = _emit_stop(stop_cfg)
    helpers_needed.update(stop_helpers)

    # ── helper lib ──
    for h in ('rorEma', 'rorTrueRange'):
        if h in helpers_needed:
            L(_HELPERS[h])
            L("")

    # ── primary indicators + triggers ──
    L("// ── primary pack: " + prim.slug + " ──")
    L(prim.emit_primary_setup())
    L(f"entrySig = {entry_expr}")
    if exit_expr:
        L(f"exitSig  = {exit_expr}")
    L("")

    # ── gates ──
    if gate_fns:
        L("// ── confluence gates (last CLOSED secondary bar, non-repaint) ──")
        for fn in gate_fns:
            L(fn)
        L("")
        for call in gate_calls:
            L(call)
        gate_all = " and ".join(g[1] for g in gate_meta)
        L(f"gateAll = {gate_all}")
        L("")
    else:
        L("gateAll = true")
        L("")

    # ── stop setup ──
    if stop_cfg:
        L("// ── stop ──")
        L(stop_setup)
        L("")

    # ── orders ──
    L("// ── orders ──")
    L("entryGated = entrySig and gateAll")
    L("var float stopLevel = na")
    L("if entryGated and strategy.position_size == 0")
    L('    strategy.entry("L", strategy.long)')
    if stop_cfg:
        L(f"    stopLevel := {stop_expr}")
    if stop_cfg:
        L("if strategy.position_size > 0 and not na(stopLevel)")
        L('    strategy.exit("stop", from_entry="L", stop=stopLevel, '
          'comment="stop_loss")')
    if exit_expr:
        L("if exitSig and strategy.position_size > 0")
        L(f'    strategy.close("L", comment="{exit_t}")')
    L("")

    # ── visuals ──
    L("// ── visuals ──")
    if prim.slug == 'ema_pp_v3':
        L('plot(emaShort, "EMA short", color.new(color.aqua, 0), 1)')
        L('plot(emaMid,   "EMA mid",   color.new(color.orange, 0), 1)')
    if stop_cfg:
        L('plot(strategy.position_size > 0 ? stopLevel : na, "stop", '
          'color.new(color.red, 0), 1, plot.style_linebr)')
    if gate_meta:
        L('bgcolor(gateAll ? color.new(color.green, 90) : na, title="gates open")')
    L('plotshape(entryGated and strategy.position_size == 0, "entry", '
      'shape.triangleup, location.belowbar, color.new(color.green, 0), '
      'size=size.tiny)')

    # ── heatmap table (per-gate state) ──
    if gate_meta:
        L("")
        L(f"var table gt = table.new(position.top_right, 1, {len(gate_meta)})")
        L("if barstate.islast")
        for r, (label, var) in enumerate(gate_meta):
            L(f'    table.cell(gt, 0, {r}, ({var} ? "✓ " : "✗ ") + "{label}", '
              f'bgcolor=({var} ? color.new(color.green, 60) : '
              f'color.new(color.red, 60)), text_color=color.white, '
              f'text_size=size.small)')

    return "\n".join(lines) + "\n"
