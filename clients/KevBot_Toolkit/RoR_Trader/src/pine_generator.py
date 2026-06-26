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


def _ordering_expr(state: str, varmap: dict) -> Optional[str]:
    """A permutation state like 'PSML' / 'SML' → strict-descending chain
    (matches the interpreters' `a>b and b>c ...` classification)."""
    try:
        vars_ = [varmap[ch] for ch in state]
    except KeyError:
        return None
    return " and ".join(f"{vars_[i]} > {vars_[i+1]}"
                        for i in range(len(vars_) - 1))


class EmaPpEmitter(PineEmitter):
    """EMA Price-Position v3 AND v4 (identical math; differ only by prefix).
    States = ordering of P(rice)/S(hort)/M(id)/L(ong) EMAs."""
    helpers = ('rorEma',)

    def __init__(self, prefix='eppv3', params=None):
        self.prefix = prefix
        self.slug = 'ema_pp_v3' if prefix == 'eppv3' else 'ema_pp_v4'
        super().__init__(params)

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
        b = base.replace(self.prefix + '_', '')
        m = {
            'cross_short_up':   "close > emaShort and close[1] <= emaShort[1]",
            'cross_short_down': "close < emaShort and close[1] >= emaShort[1]",
            'cross_mid_up':     "close > emaMid and close[1] <= emaMid[1]",
            'cross_mid_down':   "close < emaMid and close[1] >= emaMid[1]",
        }
        return m.get(b)

    def emit_gate_function(self, state: str, fn_name: str) -> Optional[str]:
        s, m, l = self._periods()
        chain = _ordering_expr(state.upper(),
                               {'P': 'c', 'S': 's', 'M': 'm', 'L': 'l'})
        if chain is None:
            return None
        return (f"{fn_name}() =>\n"
                f"    c = close\n"
                f"    s = rorEma(close, {s})\n"
                f"    m = rorEma(close, {m})\n"
                f"    l = rorEma(close, {l})\n"
                f"    {chain}")


class UtBotV4Emitter(PineEmitter):
    slug = 'ut_bot_v4'
    helpers = ()

    def _params(self):
        p = self.params
        return int(p.get('atr_period', 10)), float(p.get('key_value', 1.0))

    def _recursion(self, indent: str) -> str:
        # UT Bot trailing stop + flips. Verbatim translation of
        # UtBotV4Incremental (Wilder ATR α=1/period, 4-case ratchet). Reused at
        # indent='' (primary-TF setup) and indent='    ' (inside a gate fn).
        # var floats persist across bars; bullFlip/bearFlip are per-bar events.
        atr_p, key = self._params()
        i = indent
        return (
            f"{i}var float atr  = na\n"
            f"{i}var float stop = na\n"
            f"{i}var float pc   = na\n"
            f"{i}bool bullFlip = false\n"
            f"{i}bool bearFlip = false\n"
            f"{i}if na(stop)\n"
            f"{i}    tr = high - low\n"
            f"{i}    atr  := tr\n"
            f"{i}    stop := close - {key} * tr\n"
            f"{i}    pc   := close\n"
            f"{i}else\n"
            f"{i}    tr = math.max(high - low, math.abs(high - pc),"
            f" math.abs(low - pc))\n"
            f"{i}    atr := atr + (1.0 / {atr_p}) * (tr - atr)\n"
            f"{i}    nLoss = {key} * atr\n"
            f"{i}    ps = stop\n"
            f"{i}    float ns = na\n"
            f"{i}    if close > ps and pc > ps\n"
            f"{i}        ns := math.max(ps, close - nLoss)\n"
            f"{i}    else if close < ps and pc < ps\n"
            f"{i}        ns := math.min(ps, close + nLoss)\n"
            f"{i}    else if close > ps\n"
            f"{i}        ns := close - nLoss\n"
            f"{i}    else\n"
            f"{i}        ns := close + nLoss\n"
            f"{i}    bullFlip := pc <= ps and close > ns\n"
            f"{i}    bearFlip := pc >= ps and close < ns\n"
            f"{i}    stop := ns\n"
            f"{i}    pc   := close"
        )

    def emit_primary_setup(self) -> str:
        return self._recursion('')

    def emit_trigger(self, base: str) -> Optional[str]:
        # Entry/exit triggers are the flip events (edge-triggered), mirroring
        # SupertrendEmitter. The trend STATES are gate-only (emit_gate_function).
        b = base.replace('utv4_', '')
        return {
            'bull_flip': "bullFlip",
            'bear_flip': "bearFlip",
        }.get(b)

    def emit_gate_function(self, state: str, fn_name: str) -> Optional[str]:
        st = state.upper()
        ret = {
            'BULL_TREND': "(close > stop) and not bullFlip",
            'BEAR_TREND': "(close < stop) and not bearFlip",
            'BULL_FLIP':  "bullFlip",
            'BEAR_FLIP':  "bearFlip",
        }.get(st)
        if ret is None:
            return None
        return f"{fn_name}() =>\n{self._recursion('    ')}\n    {ret}"


class EmaStackV2Emitter(PineEmitter):
    """EMA stack ordering (short/mid/long). States = SML/SLM/MSL/MLS/LSM/LMS."""
    slug = 'ema_stack_v2'
    helpers = ('rorEma',)

    def _periods(self):
        p = self.params
        return (int(p.get('short_period', 8)), int(p.get('mid_period', 21)),
                int(p.get('long_period', 50)))

    def emit_primary_setup(self) -> str:
        s, m, l = self._periods()
        return (f"esShort = rorEma(close, {s})\n"
                f"esMid   = rorEma(close, {m})\n"
                f"esLong  = rorEma(close, {l})")

    def emit_trigger(self, base: str) -> Optional[str]:
        b = base.replace('esv2_', '')
        m = {
            'cross_bull':     "esShort > esMid and esShort[1] <= esMid[1]",
            'cross_bear':     "esShort < esMid and esShort[1] >= esMid[1]",
            'mid_cross_bull': "esMid > esLong and esMid[1] <= esLong[1]",
            'mid_cross_bear': "esMid < esLong and esMid[1] >= esLong[1]",
        }
        return m.get(b)

    def emit_gate_function(self, state: str, fn_name: str) -> Optional[str]:
        s, m, l = self._periods()
        chain = _ordering_expr(state.upper(), {'S': 's', 'M': 'm', 'L': 'l'})
        if chain is None:
            return None
        return (f"{fn_name}() =>\n"
                f"    s = rorEma(close, {s})\n"
                f"    m = rorEma(close, {m})\n"
                f"    l = rorEma(close, {l})\n"
                f"    {chain}")


class _MacdBase(PineEmitter):
    helpers = ('rorEma',)
    _pfx = 'mlv2'

    def _periods(self):
        p = self.params
        return (int(p.get('fast_period', 12)), int(p.get('slow_period', 26)),
                int(p.get('signal_period', 9)))

    def _macd_lines(self, indent='') -> str:
        f, s, sig = self._periods()
        return (f"{indent}mFast = rorEma(close, {f})\n"
                f"{indent}mSlow = rorEma(close, {s})\n"
                f"{indent}mLine = mFast - mSlow\n"
                f"{indent}mSignal = rorEma(mLine, {sig})\n"
                f"{indent}mHist = mLine - mSignal")

    def emit_primary_setup(self) -> str:
        return self._macd_lines()


class MacdLineV2Emitter(_MacdBase):
    slug = 'macd_line_v2'
    _pfx = 'mlv2'

    def emit_trigger(self, base: str) -> Optional[str]:
        b = base.replace('mlv2_', '')
        m = {
            'cross_bull':      "mLine > mSignal and mLine[1] <= mSignal[1]",
            'cross_bear':      "mLine < mSignal and mLine[1] >= mSignal[1]",
            'zero_cross_up':   "mLine > 0 and mLine[1] <= 0",
            'zero_cross_down': "mLine < 0 and mLine[1] >= 0",
        }
        return m.get(b)

    def emit_gate_function(self, state: str, fn_name: str) -> Optional[str]:
        # states: M>S+ / M>S- / M<S+ / M<S-  (line vs signal, sign of line)
        st = state.strip()
        expr = {
            'M>S+': "mLine > mSignal and mLine > 0",
            'M>S-': "mLine > mSignal and mLine <= 0",
            'M<S-': "mLine <= mSignal and mLine <= 0",
            'M<S+': "mLine <= mSignal and mLine > 0",
        }.get(st)
        if expr is None:
            return None
        return (f"{fn_name}() =>\n{self._macd_lines('    ')}\n    {expr}")


class MacdHistV2Emitter(_MacdBase):
    slug = 'macd_histogram_v2'
    _pfx = 'mhv2'

    def emit_trigger(self, base: str) -> Optional[str]:
        b = base.replace('mhv2_', '')
        m = {
            'flip_pos':            "mHist > 0 and mHist[1] <= 0",
            'flip_neg':            "mHist < 0 and mHist[1] >= 0",
            'momentum_shift_up':   "mHist > mHist[1] and mHist[1] < mHist[2]",
            'momentum_shift_down': "mHist < mHist[1] and mHist[1] > mHist[2]",
        }
        return m.get(b)

    def emit_gate_function(self, state: str, fn_name: str) -> Optional[str]:
        # states: H+up / H+dn / H-up / H-dn  (sign + direction vs prev bar)
        expr = {
            'H+up': "mHist > 0 and mHist > mHist[1]",
            'H+dn': "mHist > 0 and mHist <= mHist[1]",
            'H-up': "mHist <= 0 and mHist > mHist[1]",
            'H-dn': "mHist <= 0 and mHist <= mHist[1]",
        }.get(state.strip())
        if expr is None:
            return None
        return (f"{fn_name}() =>\n{self._macd_lines('    ')}\n    {expr}")


class SupertrendEmitter(PineEmitter):
    """Supertrend (Wilder ATR + ratcheting hl2 bands). Hand-rolled recursion
    to match the engine exactly (Pine's ta.supertrend differs in seeding)."""
    slug = 'supertrend'
    helpers = ()

    def _params(self):
        p = self.params
        return int(p.get('atr_period', 10)), float(p.get('atr_multiplier', 3.0))

    def _recursion(self, indent: str) -> str:
        atrp, mult = self._params()
        i = indent
        return (
            f"{i}var float stAtr = na\n"
            f"{i}var float stUp = na\n"
            f"{i}var float stDn = na\n"
            f"{i}var int   stTrend = na\n"
            f"{i}var float stLine = na\n"
            f"{i}var float stPc = na\n"
            f"{i}if na(stAtr)\n"
            f"{i}    tr0 = high - low\n"
            f"{i}    stAtr := tr0\n"
            f"{i}    src0 = (high + low) / 2.0\n"
            f"{i}    stUp := src0 - {mult} * tr0\n"
            f"{i}    stDn := src0 + {mult} * tr0\n"
            f"{i}    stTrend := 1\n"
            f"{i}    stLine := stUp\n"
            f"{i}    stPc := close\n"
            f"{i}else\n"
            f"{i}    tr = math.max(high - low, math.abs(high - stPc),"
            f" math.abs(low - stPc))\n"
            f"{i}    stAtr := stAtr + (1.0 / {atrp}) * (tr - stAtr)\n"
            f"{i}    src = (high + low) / 2.0\n"
            f"{i}    basicUp = src - {mult} * stAtr\n"
            f"{i}    basicDn = src + {mult} * stAtr\n"
            f"{i}    newUp = stPc > stUp ? math.max(basicUp, stUp) : basicUp\n"
            f"{i}    newDn = stPc < stDn ? math.min(basicDn, stDn) : basicDn\n"
            f"{i}    newTrend = stTrend == -1 and close > stDn ? 1 :"
            f" (stTrend == 1 and close < stUp ? -1 : stTrend)\n"
            f"{i}    stUp := newUp\n"
            f"{i}    stDn := newDn\n"
            f"{i}    stTrend := newTrend\n"
            f"{i}    stLine := newTrend == 1 ? newUp : newDn\n"
            f"{i}    stPc := close"
        )

    def emit_primary_setup(self) -> str:
        return self._recursion('')

    def emit_trigger(self, base: str) -> Optional[str]:
        b = base.replace('st_', '')
        return {
            'bull_flip': "stTrend == 1 and stTrend[1] == -1",
            'bear_flip': "stTrend == -1 and stTrend[1] == 1",
        }.get(b)

    def emit_gate_function(self, state: str, fn_name: str) -> Optional[str]:
        st = state.upper()
        expr = {
            'BULL_TRENDING':  "stTrend == 1 and (close - stLine) > 0.5 * stAtr",
            'BULL_NEAR_STOP': "stTrend == 1 and (close - stLine) <= 0.5 * stAtr",
            'BEAR_TRENDING':  "stTrend == -1 and (stLine - close) > 0.5 * stAtr",
            'BEAR_NEAR_STOP': "stTrend == -1 and (stLine - close) <= 0.5 * stAtr",
        }.get(st)
        if expr is None:
            return None
        return f"{fn_name}() =>\n{self._recursion('    ')}\n    {expr}"


class RvolV2Emitter(PineEmitter):
    """Relative volume = volume / SMA(volume, N). NOTE: volume itself diverges
    WS-vs-REST, so RVOL gates are inherently noisier than price gates."""
    slug = 'rvol_v2'
    helpers = ()

    def _period(self):
        return int(self.params.get('vol_sma_period', 20))

    def emit_primary_setup(self) -> str:
        n = self._period()
        return (f"rvSma = ta.sma(volume, {n})\n"
                f"rvRvol = rvSma > 0 ? volume / rvSma : 0.0")

    def emit_trigger(self, base: str) -> Optional[str]:
        b = base.replace('rv2_', '')
        return {
            'spike':   "rvRvol > 1.5 and rvRvol[1] <= 1.5",
            'extreme': "rvRvol > 3.0 and rvRvol[1] <= 3.0",
            'fade':    "rvRvol < 1.0 and rvRvol[1] >= 1.0",
        }.get(b)

    def emit_gate_function(self, state: str, fn_name: str) -> Optional[str]:
        n = self._period()
        expr = {
            'EXTREME': "r > 3.0",
            'HIGH':    "r > 1.5 and r <= 3.0",
            'NORMAL':  "r > 0.75 and r <= 1.5",
            'LOW':     "r > 0.5 and r <= 0.75",
            'MINIMAL': "r > 0.0 and r <= 0.5",
        }.get(state.upper())
        if expr is None:
            return None
        return (f"{fn_name}() =>\n"
                f"    sma = ta.sma(volume, {n})\n"
                f"    r = sma > 0 ? volume / sma : 0.0\n"
                f"    {expr}")


class RsiZones2Emitter(PineEmitter):
    """RSI zones. Wilder RSI seeded at first delta (NOT ta.rsi's SMA seed)."""
    slug = 'rsi_zones_2'
    helpers = ()

    def _params(self):
        p = self.params
        return (int(p.get('rsi_period', 14)), float(p.get('overbought_level', 70)),
                float(p.get('oversold_level', 30)),
                float(p.get('extreme_overbought_level', 80)),
                float(p.get('extreme_oversold_level', 20)),
                float(p.get('midline', 50)))

    def _rsi(self, indent: str) -> str:
        period = self._params()[0]
        i = indent
        return (
            f"{i}var float rsiAg = na\n"
            f"{i}var float rsiAl = na\n"
            f"{i}var float rsiPc = na\n"
            f"{i}float rsiVal = na\n"
            f"{i}if not na(rsiPc)\n"
            f"{i}    d = close - rsiPc\n"
            f"{i}    g = d > 0 ? d : 0.0\n"
            f"{i}    l = d < 0 ? -d : 0.0\n"
            f"{i}    rsiAg := na(rsiAg) ? g : (1 - 1.0/{period})*rsiAg + (1.0/{period})*g\n"
            f"{i}    rsiAl := na(rsiAl) ? l : (1 - 1.0/{period})*rsiAl + (1.0/{period})*l\n"
            f"{i}    rsiVal := rsiAl == 0 ? 100.0 : 100.0 - 100.0/(1.0 + rsiAg/rsiAl)\n"
            f"{i}rsiPc := close"
        )

    def emit_primary_setup(self) -> str:
        return self._rsi('')

    def emit_trigger(self, base: str) -> Optional[str]:
        _, ob, os_, eob, eos, mid = self._params()
        b = base.replace('rsi2_', '')
        return {
            'cross_into_overbought': f"rsiVal >= {ob} and rsiVal[1] < {ob}",
            'cross_into_extreme_overbought': f"rsiVal >= {eob} and rsiVal[1] < {eob}",
            'cross_into_oversold': f"rsiVal <= {os_} and rsiVal[1] > {os_}",
            'cross_into_extreme_oversold': f"rsiVal <= {eos} and rsiVal[1] > {eos}",
            'exit_overbought_zone': f"rsiVal < {ob} and rsiVal[1] >= {ob}",
            'exit_oversold_zone': f"rsiVal > {os_} and rsiVal[1] <= {os_}",
            'cross_above_midline': f"rsiVal > {mid} and rsiVal[1] <= {mid}",
            'cross_below_midline': f"rsiVal < {mid} and rsiVal[1] >= {mid}",
            'exit_long_overbought': f"rsiVal < {ob} and rsiVal[1] >= {ob}",
            'exit_long_midline_cross_down': f"rsiVal < {mid} and rsiVal[1] >= {mid}",
            'exit_short_oversold': f"rsiVal > {os_} and rsiVal[1] <= {os_}",
            'exit_short_midline_cross_up': f"rsiVal > {mid} and rsiVal[1] <= {mid}",
        }.get(b)

    def emit_gate_function(self, state: str, fn_name: str) -> Optional[str]:
        _, ob, os_, eob, eos, mid = self._params()
        st = state.upper()
        expr = {
            'EXTREME_OVERBOUGHT': f"rsiVal >= {eob}",
            'OVERBOUGHT': f"rsiVal >= {ob} and rsiVal < {eob}",
            'NEUTRAL_BULLISH': f"rsiVal >= {mid} and rsiVal < {ob}",
            'NEUTRAL_BEARISH': f"rsiVal > {os_} and rsiVal < {mid}",
            'OVERSOLD': f"rsiVal > {eos} and rsiVal <= {os_}",
            'EXTREME_OVERSOLD': f"rsiVal <= {eos}",
        }.get(st)
        if expr is None:
            return None
        return f"{fn_name}() =>\n{self._rsi('    ')}\n    {expr}"


class StochasticEmitter(PineEmitter):
    slug = 'stochastic_oscillator'
    helpers = ()

    def _params(self):
        p = self.params
        return (int(p.get('k_period', 14)), int(p.get('k_smooth', 3)),
                int(p.get('d_period', 3)), float(p.get('overbought', 80)),
                float(p.get('oversold', 20)), float(p.get('midline', 50)))

    def _setup(self, indent: str) -> str:
        k, ks, d, _, _, _ = self._params()
        i = indent
        return (
            f"{i}ll = ta.lowest(low, {k})\n"
            f"{i}hh = ta.highest(high, {k})\n"
            f"{i}rawK = hh == ll ? 50.0 : (close - ll) / (hh - ll) * 100.0\n"
            f"{i}stoK = ta.sma(rawK, {ks})\n"
            f"{i}stoD = ta.sma(stoK, {d})"
        )

    def emit_primary_setup(self) -> str:
        return self._setup('')

    def emit_trigger(self, base: str) -> Optional[str]:
        _, _, _, ob, os_, mid = self._params()
        b = base.replace('sto_', '')
        kx_up = "stoK > stoD and stoK[1] <= stoD[1]"
        kx_dn = "stoK < stoD and stoK[1] >= stoD[1]"
        return {
            'k_cross_above_d_oversold':   f"({kx_up}) and stoK <= {os_}",
            'k_cross_below_d_overbought': f"({kx_dn}) and stoK >= {ob}",
            'k_cross_above_d_midrange':   f"({kx_up}) and stoK > {os_} and stoK < {ob}",
            'k_cross_below_d_midrange':   f"({kx_dn}) and stoK > {os_} and stoK < {ob}",
            'enter_overbought': f"stoK >= {ob} and stoK[1] < {ob}",
            'exit_overbought':  f"stoK < {ob} and stoK[1] >= {ob}",
            'enter_oversold':   f"stoK <= {os_} and stoK[1] > {os_}",
            'exit_oversold':    f"stoK > {os_} and stoK[1] <= {os_}",
            'k_cross_above_midline': f"stoK > {mid} and stoK[1] <= {mid}",
            'k_cross_below_midline': f"stoK < {mid} and stoK[1] >= {mid}",
        }.get(b)

    def emit_gate_function(self, state: str, fn_name: str) -> Optional[str]:
        _, _, _, ob, os_, mid = self._params()
        st = state.upper()
        expr = {
            'OVERBOUGHT_BEARISH': f"stoK >= {ob} and stoK < stoD",
            'OVERBOUGHT_BULLISH': f"stoK >= {ob} and stoK >= stoD",
            'OVERSOLD_BULLISH': f"stoK <= {os_} and stoK > stoD",
            'OVERSOLD_BEARISH': f"stoK <= {os_} and stoK <= stoD",
            'BULLISH_MIDRANGE': f"stoK > {os_} and stoK < {ob} and stoK >= {mid}",
            'BEARISH_MIDRANGE': f"stoK > {os_} and stoK < {ob} and stoK < {mid}",
        }.get(st)
        if expr is None:
            return None
        return f"{fn_name}() =>\n{self._setup('    ')}\n    {expr}"


class BollingerEmitter(PineEmitter):
    """Bollinger Bands. Uses SAMPLE stdev (ddof=1) like the engine —
    converted from Pine's population ta.stdev via ×√(n/(n-1))."""
    slug = 'bollinger_bands'
    helpers = ()

    def _params(self):
        p = self.params
        return (int(p.get('bb_period', 20)), float(p.get('bb_mult', 2.0)),
                float(p.get('squeeze_threshold', 0.04)))

    def _setup(self, indent: str) -> str:
        n, mult, _ = self._params()
        i = indent
        return (
            f"{i}bbBasis = ta.sma(close, {n})\n"
            f"{i}bbStd = ta.stdev(close, {n}) * math.sqrt({n} / ({n} - 1.0))\n"
            f"{i}bbUpper = bbBasis + {mult} * bbStd\n"
            f"{i}bbLower = bbBasis - {mult} * bbStd\n"
            f"{i}bbBw = bbBasis > 0 ? (bbUpper - bbLower) / bbBasis : na"
        )

    def emit_primary_setup(self) -> str:
        return self._setup('')

    def emit_trigger(self, base: str) -> Optional[str]:
        _, _, sq = self._params()
        b = base.replace('bb_', '')
        return {
            'cross_upper': "close > bbUpper and close[1] <= bbUpper[1]",
            'cross_lower': "close < bbLower and close[1] >= bbLower[1]",
            'cross_basis_up': "close > bbBasis and close[1] <= bbBasis[1]",
            'cross_basis_down': "close < bbBasis and close[1] >= bbBasis[1]",
            'squeeze_on': f"bbBw < {sq} and bbBw[1] >= {sq}",
            'squeeze_off': f"bbBw > {sq} and bbBw[1] <= {sq}",
        }.get(b)

    def emit_gate_function(self, state: str, fn_name: str) -> Optional[str]:
        _, _, sq = self._params()
        st = state.upper()
        # mid_zone_half = (upper-lower)*0.25 from basis; squeeze = bw < threshold
        cond = {
            'SQUEEZE_UPPER': f"close >= bbBasis + mzh and bbBw < {sq}",
            'UPPER_ZONE':    f"close >= bbBasis + mzh and bbBw >= {sq}",
            'SQUEEZE_LOWER': f"close <= bbBasis - mzh and bbBw < {sq}",
            'LOWER_ZONE':    f"close <= bbBasis - mzh and bbBw >= {sq}",
            'SQUEEZE_MID':   f"close < bbBasis + mzh and close > bbBasis - mzh and bbBw < {sq}",
            'MID_ZONE':      f"close < bbBasis + mzh and close > bbBasis - mzh and bbBw >= {sq}",
        }.get(st)
        if cond is None:
            return None
        return (f"{fn_name}() =>\n{self._setup('    ')}\n"
                f"    mzh = (bbUpper - bbLower) * 0.25\n"
                f"    {cond}")


class VwapV2Emitter(PineEmitter):
    """Session-anchored VWAP + volume-weighted stdev bands. HIGHEST divergence
    risk: depends on VOLUME (diverges WS-vs-REST), is cumulative (early diffs
    compound), and anchoring must match. Uses a daily anchor; validate."""
    slug = 'vwap_v2'
    helpers = ()

    def _params(self):
        p = self.params
        return float(p.get('sd1_mult', 1.0)), float(p.get('sd2_mult', 2.0))

    def _setup(self, indent: str) -> str:
        sd1, sd2 = self._params()
        i = indent
        return (
            f"{i}vwAnchor = timeframe.change(\"1D\")\n"
            f"{i}[vwVal, vwU1, vwL1] = ta.vwap(hlc3, vwAnchor, {sd1})\n"
            f"{i}vwStd = {sd1} != 0 ? (vwU1 - vwVal) / {sd1} : 0.0\n"
            f"{i}vwU2 = vwVal + {sd2} * vwStd\n"
            f"{i}vwL2 = vwVal - {sd2} * vwStd\n"
            f"{i}vwHalf = (vwU1 - vwVal) * 0.5"
        )

    def emit_primary_setup(self) -> str:
        return self._setup('')

    def emit_trigger(self, base: str) -> Optional[str]:
        b = base.replace('vwapv2_', '')
        return {
            'cross_above': "close > vwVal and close[1] <= vwVal[1]",
            'cross_below': "close < vwVal and close[1] >= vwVal[1]",
            'enter_upper_extreme': "close > vwU2 and close[1] <= vwU2[1]",
            'enter_lower_extreme': "close < vwL2 and close[1] >= vwL2[1]",
        }.get(b)

    def emit_gate_function(self, state: str, fn_name: str) -> Optional[str]:
        st = state.strip()
        expr = {
            '>+2σ': "close > vwU2",
            '>+1σ': "close > vwU1 and close <= vwU2",
            '>V':   "close > vwVal + vwHalf and close <= vwU1",
            '@V':   "close >= vwVal - vwHalf and close <= vwVal + vwHalf",
            '<V':   "close >= vwL1 and close < vwVal - vwHalf",
            '<-1σ': "close >= vwL2 and close < vwL1",
            '<-2σ': "close < vwL2",
        }.get(st)
        if expr is None:
            return None
        return f"{fn_name}() =>\n{self._setup('    ')}\n    {expr}"


class Swing123Emitter(PineEmitter):
    """Swing 1-2-3 (C2/C3 price-action patterns). Pure OHLC, no params."""
    slug = 'swing_123'
    helpers = ()

    def _recursion(self, indent: str) -> str:
        i = indent
        return (
            f"{i}var float swH = na\n"
            f"{i}var float swL = na\n"
            f"{i}var float swC = na\n"
            f"{i}var int swPrevPat = 0\n"
            f"{i}int swPat = 0\n"
            f"{i}if not na(swC)\n"
            f"{i}    bullC2 = low < swL and close > swC\n"
            f"{i}    bearC2 = high > swH and close < swC\n"
            f"{i}    bullC3 = swPrevPat == 1 and close > swH\n"
            f"{i}    bearC3 = swPrevPat == -1 and close < swL\n"
            f"{i}    swPat := bullC3 ? 2 : (bearC3 ? -2 : (bullC2 ? 1 :"
            f" (bearC2 ? -1 : 0)))\n"
            f"{i}swH := high\n"
            f"{i}swL := low\n"
            f"{i}swC := close\n"
            f"{i}swPrevPat := swPat"
        )

    def emit_primary_setup(self) -> str:
        return self._recursion('')

    def emit_trigger(self, base: str) -> Optional[str]:
        b = base.replace('sw123_', '')
        return {
            'bull_c2': "swPat == 1", 'bull_c3': "swPat == 2",
            'bear_c2': "swPat == -1", 'bear_c3': "swPat == -2",
        }.get(b)

    def emit_gate_function(self, state: str, fn_name: str) -> Optional[str]:
        expr = {
            'BULL_C3': "swPat == 2", 'BULL_C2': "swPat == 1",
            'BEAR_C3': "swPat == -2", 'BEAR_C2': "swPat == -1",
            'NEUTRAL': "swPat == 0",
        }.get(state.upper())
        if expr is None:
            return None
        return f"{fn_name}() =>\n{self._recursion('    ')}\n    {expr}"


class StratAssistantEmitter(PineEmitter):
    """TheStrat bar types (Inside/2-up/2-down/Outside) + wick shooter/hammer."""
    slug = 'strat_assistant'
    helpers = ()

    def _wick(self):
        return float(self.params.get('wick_pct', 0.75))

    def _recursion(self, indent: str) -> str:
        i = indent
        return (
            f"{i}var float saH = na\n"
            f"{i}var float saL = na\n"
            f"{i}int saBt = 0\n"
            f"{i}if not na(saH)\n"
            f"{i}    saBt := (high <= saH and low >= saL) ? 1 :"
            f" ((high > saH and low < saL) ? 3 :"
            f" ((high > saH and not (low < saL)) ? 2 :"
            f" ((low < saL and not (high > saH)) ? -2 : 0)))\n"
            f"{i}saH := high\n"
            f"{i}saL := low"
        )

    def emit_primary_setup(self) -> str:
        w = self._wick()
        return (self._recursion('') + "\n"
                f"saRange = high - low\n"
                f"saWick = saRange * {w}\n"
                f"saShoot = high - saWick\n"
                f"saHammer = low + saWick")

    def emit_trigger(self, base: str) -> Optional[str]:
        b = base.replace('strat_', '')
        return {
            'bull_c2': "saBt == 2",
            'bear_c2': "saBt == -2",
            'outside_bar': "saBt == 3",
            'inside_bar': "saBt == 1",
            'shooter': "saBt != 1 and open < saShoot and close < saShoot",
            'hammer': "saBt != 1 and open > saHammer and close > saHammer",
        }.get(b)

    def emit_gate_function(self, state: str, fn_name: str) -> Optional[str]:
        expr = {
            'INSIDE': "saBt == 1", 'TWO_UP': "saBt == 2",
            'TWO_DOWN': "saBt == -2", 'OUTSIDE': "saBt == 3",
        }.get(state.upper())
        if expr is None:
            return None
        return f"{fn_name}() =>\n{self._recursion('    ')}\n    {expr}"


# Registry: interpreter-name (gate) → emitter class
EMITTERS = {
    'ema_pp_v3': lambda: EmaPpEmitter('eppv3'),
    'ema_pp_v4': lambda: EmaPpEmitter('eppv4'),
    'ema_stack_v2': EmaStackV2Emitter,
    'macd_line_v2': MacdLineV2Emitter,
    'macd_histogram_v2': MacdHistV2Emitter,
    'ut_bot_v4': UtBotV4Emitter,
    'supertrend': SupertrendEmitter,
    'rvol_v2': RvolV2Emitter,
    'rsi_zones_2': RsiZones2Emitter,
    'stochastic_oscillator': StochasticEmitter,
    'bollinger_bands': BollingerEmitter,
    'vwap_v2': VwapV2Emitter,
    'swing_123': Swing123Emitter,
    'strat_assistant': StratAssistantEmitter,
    # sr_channels: NOT YET PORTED — LuxAlgo-style pivot-clustering (pivot
    # detection → proximity clustering → strength ranking → nearest channel).
    # A faithful port is a focused mini-project; left unported so the endpoint
    # cleanly reports it rather than emitting a misleading approximation.
}

# Trigger-prefix → emitter factory (for the primary entry/exit pack)
_TRIGGER_PREFIX = {
    'eppv3_': lambda: EmaPpEmitter('eppv3'),
    'eppv4_': lambda: EmaPpEmitter('eppv4'),
    'esv2_': EmaStackV2Emitter,
    'mlv2_': MacdLineV2Emitter,
    'mhv2_': MacdHistV2Emitter,
    'utv4_': UtBotV4Emitter,
    'st_': SupertrendEmitter,
    'rv2_': RvolV2Emitter,
    'rsi2_': RsiZones2Emitter,
    'sto_': StochasticEmitter,
    'bb_': BollingerEmitter,
    'vwapv2_': VwapV2Emitter,
    'sw123_': Swing123Emitter,
    'strat_': StratAssistantEmitter,
}


def _emitter_for_trigger(trigger: str) -> Optional[PineEmitter]:
    for pfx, factory in _TRIGGER_PREFIX.items():
        if trigger.startswith(pfx):
            return factory()
    return None


def _emitter_for_interp(interp: str) -> Optional[PineEmitter]:
    factory = EMITTERS.get(interp.lower())
    return factory() if factory else None


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
    if prim.slug in ('ema_pp_v3', 'ema_pp_v4'):
        L('plot(emaShort, "EMA short", color.new(color.aqua, 0), 1)')
        L('plot(emaMid,   "EMA mid",   color.new(color.orange, 0), 1)')
    elif prim.slug == 'ema_stack_v2':
        L('plot(esShort, "EMA short", color.new(color.aqua, 0), 1)')
        L('plot(esMid,   "EMA mid",   color.new(color.orange, 0), 1)')
        L('plot(esLong,  "EMA long",  color.new(color.purple, 0), 1)')
    # (MACD-primary: lines live near 0, not plotted on the price overlay —
    #  the entry shapes + gate background carry the signal.)
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
