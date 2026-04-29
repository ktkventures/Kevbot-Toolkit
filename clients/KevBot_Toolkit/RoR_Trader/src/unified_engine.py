#!/usr/bin/env python3
"""
Unified Chart Engine — single bar-by-bar state machine for backtest and live.

Processes OHLCV bars sequentially through the same computation path regardless
of whether data is historical (backtest) or live (streaming).  Eliminates
parity bugs between batch and incremental pipelines by construction.

Phase 30A: Backtest-only.  No WebSocket, no live ticks, no alert dispatch.

Usage:
    from unified_engine import run_unified_backtest

    trades_df, enriched_df = run_unified_backtest(raw_df, strategy)
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger("unified_engine")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_HISTORY = 25_000

TIMEFRAME_SECONDS = {
    "5Sec": 5, "10Sec": 10, "15Sec": 15, "30Sec": 30,
    "1Min": 60, "2Min": 120, "3Min": 180, "5Min": 300,
    "10Min": 600, "15Min": 900, "30Min": 1800,
    "1Hour": 3600, "2Hour": 7200, "4Hour": 14400,
    "1Day": 86400, "1Week": 604800, "1Month": 2592000,
}
SECONDS_TO_TIMEFRAME = {v: k for k, v in TIMEFRAME_SECONDS.items()}

VWAP_SESSION_GAP_SECONDS = 30 * 60  # 30 minutes

# Template key -> required indicator set and interpreter key
TEMPLATE_REQUIREMENTS = {
    'ema_stack': ({'ema'}, 'EMA_STACK'),
    'ema_price_position': ({'ema'}, 'EMA_PRICE_POSITION'),
    'ema_price_position_v2': ({'ema'}, 'EMA_PRICE_POSITION_V2'),
    'macd_line': ({'macd'}, 'MACD_LINE'),
    'macd_histogram': ({'macd'}, 'MACD_HISTOGRAM'),
    'vwap': ({'vwap'}, 'VWAP'),
    'rvol': ({'rvol'}, 'RVOL'),
    'utbot': ({'utbot'}, 'UTBOT'),
    'utbot_v2': ({'utbot'}, 'UTBOT_V2'),
    'bar_count': (set(), None),
}

# Trigger prefix -> template key
TRIGGER_PREFIX_TO_TEMPLATE = {
    'ema': 'ema_stack',
    'ema_pp': 'ema_price_position',
    'ema_pp_v2': 'ema_price_position_v2',
    'macd': 'macd_line',
    'macd_hist': 'macd_histogram',
    'vwap': 'vwap',
    'rvol': 'rvol',
    'utbot': 'utbot',
    'utbot_v2': 'utbot_v2',
    'bar_count': 'bar_count',
}

# Intra-bar level map: base trigger -> level column + cross direction
INTRABAR_LEVEL_MAP: Dict[str, Dict[str, str]] = {
    "vwap_cross_above":          {"column": "vwap",            "cross": "above"},
    "vwap_cross_below":          {"column": "vwap",            "cross": "below"},
    "vwap_enter_upper_extreme":  {"column": "vwap_sd2_upper",  "cross": "above"},
    "vwap_enter_lower_extreme":  {"column": "vwap_sd2_lower",  "cross": "below"},
    "utbot_buy":                 {"column": "utbot_stop",      "cross": "above"},
    "utbot_sell":                {"column": "utbot_stop",      "cross": "below"},
    "utbot_v2_buy":              {"column": "utbot_stop_prev", "cross": "above"},
    "utbot_v2_sell":             {"column": "utbot_stop_prev", "cross": "below"},
    "ema_pp_cross_short_up":     {"column": "ema_9",  "cross": "above", "param_key": "short_period"},
    "ema_pp_cross_short_down":   {"column": "ema_9",  "cross": "below", "param_key": "short_period"},
    "ema_pp_cross_mid_up":       {"column": "ema_21", "cross": "above", "param_key": "mid_period"},
    "ema_pp_cross_mid_down":     {"column": "ema_21", "cross": "below", "param_key": "mid_period"},
    "ema_pp_v2_cross_short_up":  {"column": "ema_9_prev",  "cross": "above", "param_key": "short_period"},
    "ema_pp_v2_cross_short_down": {"column": "ema_9_prev",  "cross": "below", "param_key": "short_period"},
    "ema_pp_v2_cross_mid_up":    {"column": "ema_21_prev", "cross": "above", "param_key": "mid_period"},
    "ema_pp_v2_cross_mid_down":  {"column": "ema_21_prev", "cross": "below", "param_key": "mid_period"},
}

# L-type triggers: prev-bar-close-opposite-side gating
# (as opposed to H-type gating which requires bar-close C-type trigger to fire)
# Mutable set: user packs can register additional L-type triggers at runtime
# via pack_registry._inject_intrabar_level_map().
_IB_L_TYPE_TRIGGERS: set = {
    'vwap_cross_above', 'vwap_cross_below',
    'vwap_enter_upper_extreme', 'vwap_enter_lower_extreme',
    'ema_pp_cross_short_up', 'ema_pp_cross_short_down',
    'ema_pp_cross_mid_up', 'ema_pp_cross_mid_down',
    'ema_pp_v2_cross_short_up', 'ema_pp_v2_cross_short_down',
    'ema_pp_v2_cross_mid_up', 'ema_pp_v2_cross_mid_down',
    'utbot_v2_buy', 'utbot_v2_sell',
}

# Trigger execution type classification
# C  = Bar Close
# L0 = Level Cross, current bar's indicator level
# L1 = Level Cross, previous bar's indicator level (fixed reference)
# HM = Hybrid Market (L1 cross + bar-close confirmation → market order)  [legacy]
# HL = Hybrid Limit  (L1 cross + bar-close confirmation → limit order)   [legacy]
# LC = Level-Close (L cross + configurable confirmation → configurable bail)
# CC = Close-Close (C entry + next-bar confirmation → configurable bail)
_L_TYPES = frozenset({'L0', 'L1'})
_LC_CC_TYPES = frozenset({'LC', 'CC'})


def get_trigger_exec_type(trigger_id: str) -> str:
    """Get execution type for a trigger.

    Handles both template-prefixed IDs (e.g. ``utbot_v2_buy_ib``) and
    group-prefixed IDs (e.g. ``utbot_v2_default_buy_ib``).

    Returns: 'C' (bar close), 'L0' (level cross, current bar),
             'L1' (level cross, previous bar), 'HM' (hybrid market),
             'HL' (hybrid limit), 'LC' (level-close), or 'CC' (close-close).
    """
    if trigger_id.endswith('_hm'):
        return 'HM'
    if trigger_id.endswith('_hl'):
        return 'HL'
    if trigger_id.endswith('_lc'):
        return 'LC'
    if trigger_id.endswith('_cc'):
        return 'CC'
    if trigger_id.endswith('_ib'):
        base = trigger_id[:-3]
        level_spec = INTRABAR_LEVEL_MAP.get(base)
        if level_spec is None:
            # Group-prefixed ID: find the template prefix in the trigger_id,
            # then match the base trigger suffix against INTRABAR_LEVEL_MAP.
            _sorted_prefixes = sorted(
                TRIGGER_PREFIX_TO_TEMPLATE, key=len, reverse=True)
            for tp in _sorted_prefixes:
                if base.startswith(tp + '_'):
                    base_suffix = base[len(tp) + 1:]  # e.g. "default_buy"
                    for mk, spec in INTRABAR_LEVEL_MAP.items():
                        if mk.startswith(tp + '_'):
                            mk_suffix = mk[len(tp) + 1:]  # e.g. "buy"
                            if base_suffix.endswith(mk_suffix):
                                level_spec = spec
                                break
                    break
        col = level_spec.get('column', '') if level_spec else ''
        return 'L1' if col.endswith('_prev') else 'L0'
    return 'C'


def _strip_exec_suffix(trigger_id: str) -> str:
    """Strip execution suffix (_ib/_hm/_hl/_lc/_cc) to get base trigger."""
    for suffix in ('_ib', '_hm', '_hl', '_lc', '_cc'):
        if trigger_id.endswith(suffix):
            return trigger_id[:-len(suffix)]
    return trigger_id


# ═══════════════════════════════════════════════════════════════════════════
# GENERAL PACK SCALAR EVALUATORS
# ═══════════════════════════════════════════════════════════════════════════

_GP_SESSION_WINDOWS = {
    "pre_market": (4, 0, 9, 30),
    "regular": (9, 30, 16, 0),
    "after_hours": (16, 0, 20, 0),
    "extended": (4, 0, 20, 0),
}


def _eval_gp_scalar(pack, ts: datetime) -> Optional[str]:
    """Evaluate a GeneralPack against a single timestamp. Returns state label."""
    try:
        from general_packs import TEMPLATES
    except ImportError:
        return None
    logic = TEMPLATES.get(pack.base_template, {}).get("condition_logic")
    params = pack.parameters
    if logic == "time_window":
        start = params.get("start_hour", 9) * 60 + params.get("start_minute", 30)
        end = params.get("end_hour", 12) * 60 + params.get("end_minute", 0)
        bar_min = ts.hour * 60 + ts.minute
        return "IN_WINDOW" if start <= bar_min < end else "OUT_OF_WINDOW"
    elif logic == "session_filter":
        session = params.get("session", "regular")
        sh, sm, eh, em = _GP_SESSION_WINDOWS.get(
            session, _GP_SESSION_WINDOWS["regular"])
        start = sh * 60 + sm
        end = eh * 60 + em
        bar_min = ts.hour * 60 + ts.minute
        return "IN_SESSION" if start <= bar_min < end else "OUT_OF_SESSION"
    elif logic == "day_filter":
        day_map = {0: "monday", 1: "tuesday", 2: "wednesday",
                   3: "thursday", 4: "friday"}
        day_name = day_map.get(ts.weekday())
        if day_name is None:
            return "BLOCKED_DAY"
        return "ALLOWED_DAY" if params.get(day_name, True) else "BLOCKED_DAY"
    elif logic == "calendar_filter":
        if params.get("avoid_fomc", True) and ts.weekday() == 2 and ts.day <= 7:
            return "BLOCKED"
        if params.get("avoid_nfp", True) and ts.weekday() == 4 and ts.day <= 7:
            return "BLOCKED"
        if params.get("avoid_opex", False) and ts.weekday() == 4 and 15 <= ts.day <= 21:
            return "BLOCKED"
        return "CLEAR"
    return None


def _load_enabled_general_packs() -> list:
    """Load and return enabled GeneralPack objects."""
    try:
        from general_packs import load_general_packs, get_enabled_general_packs
        return get_enabled_general_packs(load_general_packs())
    except Exception:
        return []


def _evaluate_general_packs(packs: list, ts: datetime) -> Set[str]:
    """Evaluate all packs against a single timestamp, return GEN- records."""
    records = set()
    for pack in packs:
        state = _eval_gp_scalar(pack, ts)
        if state:
            records.add(f"GEN-{pack.id.upper()}-{state}")
    return records


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY RESOLVER
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_trigger_id(strategy: dict, key: str) -> str:
    """Resolve a strategy's trigger ID using confluence mapping."""
    confluence_key = key.replace('trigger', 'trigger_confluence_id')
    cid = strategy.get(confluence_key)
    if cid:
        try:
            from alerts import _get_base_trigger_id
            return _get_base_trigger_id(cid)
        except Exception:
            pass
    return strategy.get(key, '')


def _resolve_trigger_ids(strategy: dict, key: str) -> List[str]:
    """Resolve multiple trigger IDs (exit_triggers)."""
    confluence_key = key.replace('triggers', 'trigger_confluence_ids')
    cids = strategy.get(confluence_key, [])
    if cids:
        try:
            from alerts import _get_base_trigger_id
            return [_get_base_trigger_id(c) for c in cids if c]
        except Exception:
            pass
    return [t for t in strategy.get(key, []) if t]


# ═══════════════════════════════════════════════════════════════════════════
# INDICATOR PARAMETER CONTRACT
# ═══════════════════════════════════════════════════════════════════════════
# Single source of truth: the params dict returned by
# resolve_strategy_requirements MUST contain every key listed here for every
# indicator in the required set. The engine constructors below perform strict
# params[key] lookups — a missing key raises KeyError immediately rather than
# silently substituting a hardcoded default.
#
# Adding a new indicator pack? Add it here, read its params from the matching
# group, and the engine will pick it up automatically. Do NOT add
# `params.get(key, default)` fallbacks in engine constructors — defaults
# belong here where they are visible as part of the contract.

class _GroupResolutionError(Exception):
    """Raised when no matching confluence group can be found."""


# Maps indicator slug → resolution spec.
#   'params': list of (engine_param_key, group_param_key) pairs
#   'templates': tuple of confluence template slugs whose groups supply params
#   'interpreters': tuple of interpreter keys this indicator powers
#   'defaults_only': True if indicator has no user-configurable group
#                    (e.g., ATR for stop-loss — period is a convention)
_INDICATOR_PARAM_SPEC: Dict[str, Dict[str, Any]] = {
    'ema': {
        'params': [('ema_periods', '__ema_periods__')],
        'templates': ('ema_stack', 'ema_price_position',
                      'ema_price_position_v2'),
        'interpreters': ('EMA_STACK', 'EMA_PRICE_POSITION',
                         'EMA_PRICE_POSITION_V2'),
    },
    'macd': {
        'params': [
            ('macd_fast_period', 'fast_period'),
            ('macd_slow_period', 'slow_period'),
            ('macd_signal_period', 'signal_period'),
        ],
        'templates': ('macd_line', 'macd_histogram'),
        'interpreters': ('MACD_LINE', 'MACD_HISTOGRAM'),
    },
    'vwap': {
        'params': [
            ('vwap_sd1_mult', 'sd1_mult'),
            ('vwap_sd2_mult', 'sd2_mult'),
        ],
        'templates': ('vwap',),
        'interpreters': ('VWAP',),
    },
    'utbot': {
        'params': [
            ('utbot_atr_period', 'atr_period'),
            ('utbot_atr_mult', 'atr_multiplier'),
        ],
        'templates': ('utbot', 'utbot_v2'),
        'interpreters': ('UTBOT', 'UTBOT_V2'),
    },
    'rvol': {
        'params': [('vol_sma_period', 'sma_period')],
        'templates': ('rvol',),
        'interpreters': ('RVOL',),
    },
    # ATR is used for stop-loss distance; period is a convention (14), not
    # user-configurable via a confluence group.
    'atr': {
        'params': [('atr_period', 14)],
        'defaults_only': True,
    },
}


def _find_group_for_indicator(groups, strategy: dict, spec: Dict[str, Any]):
    """Find the enabled confluence group supplying params for an indicator.

    Resolution order:
    1. Group whose trigger ID matches strategy's entry / exit trigger.
    2. Group whose interpreter appears in the strategy's confluence records.
    3. Any enabled group matching one of spec['templates'] (same indicator
       family shares column names, so variant params are compatible).

    Raises _GroupResolutionError if no candidate exists.
    """
    from confluence_groups import get_all_triggers, TEMPLATES

    candidates = [g for g in groups
                  if g.base_template in spec['templates']]
    if not candidates:
        raise _GroupResolutionError(
            f"no enabled confluence group with base_template in "
            f"{spec['templates']!r}"
        )

    # 1. Match via entry/exit trigger IDs
    all_triggers = get_all_triggers(groups)
    cid = strategy.get('entry_trigger_confluence_id') or ''
    exit_cids = strategy.get('exit_trigger_confluence_ids', []) or []
    for c in ([cid] if cid else []) + list(exit_cids):
        if not c or c not in all_triggers:
            continue
        base_t = all_triggers[c].base_trigger
        for g in candidates:
            if g.get_trigger_id(base_t) == c:
                return g

    # 2. Match via confluence record interpreter keys
    wanted_interps = set(spec.get('interpreters', ()))
    if wanted_interps:
        conf_interps = set()
        for rec in (list(strategy.get('confluence', []))
                    + list(strategy.get('general_confluences', []))):
            parts = rec.split('-', 2)
            if len(parts) >= 2 and parts[1] in wanted_interps:
                conf_interps.add(parts[1])
        if conf_interps:
            for g in candidates:
                tpl = TEMPLATES.get(g.base_template, {})
                if any(ik in conf_interps
                       for ik in tpl.get('interpreters', [])):
                    return g

    # 3. Variant fallback — same family, shared indicator column names
    return candidates[0]


def _load_enabled_groups_for_strategy(strategy: dict):
    """Load enabled confluence groups for a strategy.

    Uses thread-local user context if set (Streamlit / API request paths).
    Falls back to admin-client lookup keyed by strategy['user_id'] when no
    thread context is available (worker path — monitors run in threads that
    never authenticate as the owning user).

    Raises if neither a thread context nor a stamped user_id is available.
    """
    from confluence_groups import (
        get_enabled_groups, _parse_group_list, create_default_groups,
    )
    try:
        from db import USE_DB, get_current_user_id
    except ImportError:
        return get_enabled_groups()

    if not USE_DB:
        return get_enabled_groups()

    if get_current_user_id():
        return get_enabled_groups()

    strat_user_id = strategy.get('user_id')
    if not strat_user_id:
        raise RuntimeError(
            f"Cannot load confluence groups for strategy "
            f"{strategy.get('id')}: no thread-local user context and "
            f"strategy has no user_id stamped. Worker path must stamp "
            f"user_id on every strategy dict before engine init."
        )

    from db import load_confluence_groups_admin
    raw = load_confluence_groups_admin(strat_user_id)
    groups = _parse_group_list(raw) if raw else create_default_groups()
    return [g for g in groups if g.enabled]


def resolve_strategy_requirements(strategy: dict) -> Tuple[
        Set[str], Set[str], Set[str], Dict[str, Any]]:
    """Resolve what indicators, interpreters, and triggers a strategy needs.

    Returns:
        (required_indicators, required_interpreters,
         required_triggers, indicator_params)
    """
    indicators: Set[str] = set()
    interpreters: Set[str] = set()
    triggers: Set[str] = set()
    params: Dict[str, Any] = {}

    # Always need ATR for stop calculations
    indicators.add('atr')

    def _process_trigger_id(trigger_id: str):
        if not trigger_id:
            return
        base = _strip_exec_suffix(trigger_id)
        triggers.add(trigger_id)
        if base != trigger_id:
            triggers.add(base)

        matched_template = None
        for prefix in sorted(TRIGGER_PREFIX_TO_TEMPLATE.keys(),
                             key=len, reverse=True):
            if base.startswith(prefix + '_') or base == prefix:
                matched_template = TRIGGER_PREFIX_TO_TEMPLATE[prefix]
                break

        if matched_template and matched_template in TEMPLATE_REQUIREMENTS:
            ind_set, interp_key = TEMPLATE_REQUIREMENTS[matched_template]
            indicators.update(ind_set)
            if interp_key:
                interpreters.add(interp_key)

    def _process_confluence_record(record: str):
        parts = record.split('-', 2)
        if len(parts) < 3:
            return
        tf_label, interp_key, _state = parts
        if tf_label == 'GEN':
            return
        interpreters.add(interp_key)
        for tpl_key, (ind_set, ikey) in TEMPLATE_REQUIREMENTS.items():
            if ikey == interp_key:
                indicators.update(ind_set)
                break

    entry = _resolve_trigger_id(strategy, 'entry_trigger')
    _process_trigger_id(entry)

    exit_triggers = _resolve_trigger_ids(strategy, 'exit_triggers')
    for et in exit_triggers:
        _process_trigger_id(et)
    if not exit_triggers:
        exit_single = _resolve_trigger_id(strategy, 'exit_trigger')
        if exit_single and exit_single != 'opposite_signal':
            _process_trigger_id(exit_single)
        elif exit_single == 'opposite_signal' or \
                strategy.get('exit_trigger') == 'opposite_signal':
            try:
                from triggers import get_opposite_trigger
                opp = get_opposite_trigger(entry)
                if opp:
                    _process_trigger_id(opp)
            except ImportError:
                pass

    for conf in strategy.get('confluence', []):
        _process_confluence_record(conf)
    for conf in strategy.get('general_confluences', []):
        _process_confluence_record(conf)

    # Resolve indicator params from the user's confluence groups.
    # Every group-parameterized indicator is resolved here; no downstream
    # engine constructor may substitute hardcoded defaults. See
    # _INDICATOR_PARAM_SPEC below for the full contract.
    groups_cache = None

    def _groups():
        nonlocal groups_cache
        if groups_cache is None:
            groups_cache = _load_enabled_groups_for_strategy(strategy)
        return groups_cache

    for ind in sorted(indicators):
        spec = _INDICATOR_PARAM_SPEC.get(ind)
        if spec is None:
            # No parameterized state — user packs, _user_pack_* slugs, etc.
            continue
        if spec.get('defaults_only'):
            for engine_key, default_val in spec['params']:
                params[engine_key] = default_val
            continue
        try:
            group = _find_group_for_indicator(_groups(), strategy, spec)
        except _GroupResolutionError as e:
            raise RuntimeError(
                f"Cannot resolve params for indicator {ind!r} on strategy "
                f"{strategy.get('id')} ({strategy.get('name')!r}): {e}"
            ) from e
        gp = group.parameters
        for engine_key, group_key in spec['params']:
            if engine_key == 'ema_periods':
                params[engine_key] = [
                    gp['short_period'],
                    gp['mid_period'],
                    gp.get('long_period', gp['mid_period'] * 10),
                ]
                continue
            if group_key not in gp:
                raise RuntimeError(
                    f"Confluence group {group.id!r} is missing required "
                    f"parameter {group_key!r} needed by indicator {ind!r} "
                    f"for strategy {strategy.get('id')}. Fix the group in "
                    f"Confluence Packs or delete it."
                )
            params[engine_key] = gp[group_key]
        logger.debug(
            "resolved params for %s via group %s on strategy %s",
            ind, group.id, strategy.get('id'),
        )

    # Resolve user pack requirements.
    #
    # A pack is "in play" for a strategy if EITHER (a) any required trigger
    # uses its `trigger_prefix`, OR (b) any required interpreter is in the
    # pack's manifest `interpreters` list (the latter happens when a
    # strategy uses a user-pack interpreter purely as a confluence gate,
    # without firing any trigger from that pack).
    #
    # In both cases the pack's `_user_pack_<slug>` indicator marker is
    # added so IncrementalIndicatorEngine instantiates the live
    # incremental_class (or the batch pipeline runs the indicator function
    # via run_indicators_for_group).
    try:
        import pack_registry
        from confluence_groups import TEMPLATES
        registered = pack_registry.get_registered_packs()
        for slug, pack in registered.items():
            manifest = pack.manifest
            tp = manifest.get('trigger_prefix', '')
            pack_interpreters = set(manifest.get('interpreters', []))
            pack_triggers_used = bool(tp) and any(
                t.startswith(tp + '_') or t == tp
                for t in triggers
            )
            pack_interps_used = bool(pack_interpreters & interpreters)
            if not pack_triggers_used and not pack_interps_used:
                continue
            # Pull pack's interpreter keys into the required set so the
            # live dispatch and the backtest user_pack_data merge both
            # surface its state. (No-op when only triggers triggered the
            # match — interpreters set already had whatever was needed.)
            for ik in pack_interpreters:
                interpreters.add(ik)
            indicators.add(f'_user_pack_{slug}')
    except Exception as e:
        # Surface — user packs that fail to register would silently produce
        # strategies with missing interpreters/columns at engine init.
        logger.warning(
            "pack_registry load failed while resolving strategy %s: %s",
            strategy.get('id'), e,
        )

    # TriggerEvaluator always consumes ema_periods (for L-type cross
    # registration) regardless of whether 'ema' is in the required set.
    # When absent, empty list means "no EMA triggers to register" — this is
    # a structural requirement of the API, not a silent parameter fallback.
    params.setdefault('ema_periods', [])

    # Contract check: every parameterized indicator in the required set has
    # all its engine-level params populated. Any gap here is a bug in the
    # resolver above or in _INDICATOR_PARAM_SPEC — never paper over it.
    missing = []
    for ind in indicators:
        spec = _INDICATOR_PARAM_SPEC.get(ind)
        if spec is None:
            continue
        for engine_key, _ in spec['params']:
            if engine_key not in params:
                missing.append((ind, engine_key))
    if missing:
        raise RuntimeError(
            f"resolve_strategy_requirements: params contract violated for "
            f"strategy {strategy.get('id')} ({strategy.get('name')!r}): "
            f"missing {missing!r}. This is a bug in the resolver — do NOT "
            f"add hardcoded defaults in the engine to paper over it."
        )

    return indicators, interpreters, triggers, params


# ═══════════════════════════════════════════════════════════════════════════
# INCREMENTAL INDICATOR ENGINE — O(1) updates per bar close
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class IndicatorState:
    """Cached state for O(1) incremental indicator computation."""

    ema: Dict[int, float] = field(default_factory=dict)

    macd_ema_fast: float = 0.0
    macd_ema_slow: float = 0.0
    macd_signal_ema: float = 0.0
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9

    atr_value: float = 0.0
    atr_period: int = 14
    prev_close: float = 0.0

    vwap_cum_pv: float = 0.0
    vwap_cum_vol: float = 0.0
    vwap_cum_sq_dev_vol: float = 0.0
    vwap_value: float = 0.0
    vwap_std: float = 0.0
    vwap_sd1_mult: float = 1.0
    vwap_sd2_mult: float = 2.0
    vwap_prev_ts: Optional[datetime] = None

    utbot_atr: float = 0.0
    utbot_atr_period: int = 10
    utbot_atr_mult: float = 1.0
    utbot_trail_stop: float = 0.0
    utbot_direction: int = 0
    utbot_prev_close: float = 0.0

    vol_buffer: deque = field(default_factory=lambda: deque(maxlen=20))
    vol_sum: float = 0.0
    vol_count: int = 0
    vol_sma_period: int = 20

    prev_values: Dict[str, float] = field(default_factory=dict)
    current: Dict[str, float] = field(default_factory=dict)

    prev_macd_hist: float = 0.0
    prev2_macd_hist: float = 0.0


class IncrementalIndicatorEngine:
    """Strategy-scoped incremental indicator computation.

    Only computes indicators that the strategy actually needs.
    """

    def __init__(self, required_indicators: Set[str], params: Dict[str, Any]):
        # params must be produced by resolve_strategy_requirements.
        # Strict lookups — missing key = bug in the resolver. Do NOT add
        # .get(key, default) here; defaults belong in _INDICATOR_PARAM_SPEC.
        self.required = required_indicators
        self.params = params
        self.state = IndicatorState()
        self._initialized = False

        if 'ema' in self.required:
            for p in params['ema_periods']:
                self.state.ema[p] = 0.0

        if 'macd' in self.required:
            self.state.macd_fast_period = params['macd_fast_period']
            self.state.macd_slow_period = params['macd_slow_period']
            self.state.macd_signal_period = params['macd_signal_period']

        if 'atr' in self.required:
            self.state.atr_period = params['atr_period']

        if 'vwap' in self.required:
            self.state.vwap_sd1_mult = params['vwap_sd1_mult']
            self.state.vwap_sd2_mult = params['vwap_sd2_mult']

        if 'utbot' in self.required:
            self.state.utbot_atr_period = params['utbot_atr_period']
            self.state.utbot_atr_mult = params['utbot_atr_mult']

        if 'rvol' in self.required:
            period = params['vol_sma_period']
            self.state.vol_sma_period = period
            self.state.vol_buffer = deque(maxlen=period)

        # User packs: instantiate the incremental class for any required
        # indicator marker `_user_pack_<slug>` whose pack ships an
        # incremental_class. Packs without one are batch-only and stay
        # silent in live mode (FAIL_SILENT in the parity simulator) —
        # that's by design, see pack_builder_context.md.
        self._user_pack_engines: dict = {}
        user_pack_markers = [
            i for i in self.required if i.startswith('_user_pack_')
        ]
        if user_pack_markers:
            try:
                import pack_registry
                # Resolve pack-specific params from the user's enabled
                # confluence group for that pack, falling back to the
                # manifest's parameters_schema defaults. This matches
                # what the batch path (run_indicators_for_group) does.
                from confluence_groups import load_confluence_groups
                try:
                    groups = load_confluence_groups()
                except Exception:
                    groups = []
                groups_by_template = {g.base_template: g for g in groups}
                for marker in user_pack_markers:
                    slug = marker[len('_user_pack_'):]
                    pack = pack_registry.get_pack(slug)
                    if pack is None or pack.incremental_class is None:
                        continue  # batch-only — no live wiring
                    pack_params = {}
                    g = groups_by_template.get(slug)
                    if g and isinstance(g.parameters, dict):
                        pack_params = dict(g.parameters)
                    # Fill in any missing keys from manifest defaults
                    for key, spec in pack.manifest.get(
                        'parameters_schema', {}
                    ).items():
                        pack_params.setdefault(key, spec.get('default'))
                    try:
                        self._user_pack_engines[slug] = pack.incremental_class(
                            **pack_params
                        )
                        logger.info(
                            "user pack instantiated: slug=%s params=%s",
                            slug, pack_params)
                    except Exception as e:
                        logger.warning(
                            "could not instantiate incremental_class for "
                            "user pack %r: %s", slug, e)
            except Exception as e:
                logger.warning(
                    "user pack incremental load failed: %s", e)
        if user_pack_markers and not self._user_pack_engines:
            logger.warning(
                "User-pack markers present in required set %s but "
                "_user_pack_engines is empty after init — live triggers "
                "will not fire.", sorted(user_pack_markers))

    def warmup(self, df: pd.DataFrame):
        """Initialize state from historical bars."""
        if len(df) < 2:
            return

        for i in range(len(df)):
            row = df.iloc[i]
            bar = {
                'open': float(row['open']), 'high': float(row['high']),
                'low': float(row['low']), 'close': float(row['close']),
                'volume': float(row.get('volume', 0)),
                'timestamp': df.index[i],
            }
            self._update_indicators(bar, is_first=(i == 0))

        self._initialized = True

    def update_bar(self, bar: dict) -> Dict[str, float]:
        """Incremental O(1) update for a new completed bar.

        Auto-detects first bar (no warmup needed for backtest mode).
        Returns dict of current indicator values.
        """
        self.state.prev2_macd_hist = self.state.prev_macd_hist
        self.state.prev_macd_hist = self.state.current.get('macd_hist', 0.0)
        self.state.prev_values = dict(self.state.current)

        is_first = not self._initialized
        self._update_indicators(bar, is_first=is_first)
        if not self._initialized:
            self._initialized = True
        return dict(self.state.current)

    def get_values(self) -> Dict[str, float]:
        return dict(self.state.current)

    def get_prev_values(self) -> Dict[str, float]:
        return dict(self.state.prev_values)

    def _update_indicators(self, bar: dict, is_first: bool = False):
        close = bar['close']
        high = bar['high']
        low = bar['low']
        volume = bar['volume']
        timestamp = bar['timestamp']

        vals: Dict[str, float] = {
            'close': close, 'high': high, 'low': low,
            'open': bar['open'], 'volume': volume,
        }

        # ── EMA ──
        if 'ema' in self.required:
            for period, prev_ema in self.state.ema.items():
                alpha = 2.0 / (period + 1)
                if is_first:
                    new_ema = close
                else:
                    new_ema = alpha * close + (1.0 - alpha) * prev_ema
                self.state.ema[period] = new_ema
                vals[f'ema_{period}'] = new_ema

        # ── ATR (ewm-based) ──
        if 'atr' in self.required:
            if is_first:
                tr = high - low
                self.state.atr_value = tr
                self.state.prev_close = close
            else:
                tr1 = high - low
                tr2 = abs(high - self.state.prev_close)
                tr3 = abs(low - self.state.prev_close)
                tr = max(tr1, tr2, tr3)
                alpha = 2.0 / (self.state.atr_period + 1)
                self.state.atr_value = (
                    alpha * tr + (1.0 - alpha) * self.state.atr_value)
                self.state.prev_close = close
            vals['atr'] = self.state.atr_value

        # ── MACD ──
        if 'macd' in self.required:
            fp = self.state.macd_fast_period
            sp = self.state.macd_slow_period
            sig_p = self.state.macd_signal_period
            af = 2.0 / (fp + 1)
            a_s = 2.0 / (sp + 1)
            a_sig = 2.0 / (sig_p + 1)

            if is_first:
                self.state.macd_ema_fast = close
                self.state.macd_ema_slow = close
                self.state.macd_signal_ema = 0.0
            else:
                self.state.macd_ema_fast = (
                    af * close + (1.0 - af) * self.state.macd_ema_fast)
                self.state.macd_ema_slow = (
                    a_s * close + (1.0 - a_s) * self.state.macd_ema_slow)

            macd_line = self.state.macd_ema_fast - self.state.macd_ema_slow

            if is_first:
                self.state.macd_signal_ema = macd_line
            else:
                self.state.macd_signal_ema = (
                    a_sig * macd_line
                    + (1.0 - a_sig) * self.state.macd_signal_ema)

            macd_hist = macd_line - self.state.macd_signal_ema
            vals['macd_line'] = macd_line
            vals['macd_signal'] = self.state.macd_signal_ema
            vals['macd_hist'] = macd_hist

        # ── VWAP ──
        if 'vwap' in self.required:
            tp = (high + low + close) / 3.0

            if self.state.vwap_prev_ts is not None:
                if hasattr(timestamp, 'timestamp'):
                    ts_epoch = timestamp.timestamp()
                else:
                    ts_epoch = pd.Timestamp(timestamp).timestamp()
                prev_epoch = (self.state.vwap_prev_ts.timestamp()
                              if hasattr(self.state.vwap_prev_ts, 'timestamp')
                              else pd.Timestamp(
                                  self.state.vwap_prev_ts).timestamp())
                if ts_epoch - prev_epoch > VWAP_SESSION_GAP_SECONDS:
                    self.state.vwap_cum_pv = 0.0
                    self.state.vwap_cum_vol = 0.0
                    self.state.vwap_cum_sq_dev_vol = 0.0

            if is_first:
                self.state.vwap_cum_pv = 0.0
                self.state.vwap_cum_vol = 0.0
                self.state.vwap_cum_sq_dev_vol = 0.0

            self.state.vwap_cum_pv += tp * volume
            self.state.vwap_cum_vol += volume

            if self.state.vwap_cum_vol > 0:
                vwap = self.state.vwap_cum_pv / self.state.vwap_cum_vol
                sq_dev = volume * (tp - vwap) ** 2
                self.state.vwap_cum_sq_dev_vol += sq_dev
                std = math.sqrt(
                    self.state.vwap_cum_sq_dev_vol / self.state.vwap_cum_vol)
            else:
                vwap = close
                std = 0.0

            self.state.vwap_value = vwap
            self.state.vwap_std = std
            self.state.vwap_prev_ts = timestamp

            vals['vwap'] = vwap
            m1 = self.state.vwap_sd1_mult
            m2 = self.state.vwap_sd2_mult
            vals['vwap_sd1_upper'] = vwap + m1 * std
            vals['vwap_sd1_lower'] = vwap - m1 * std
            vals['vwap_sd2_upper'] = vwap + m2 * std
            vals['vwap_sd2_lower'] = vwap - m2 * std

        # ── UT Bot (Wilder ATR) ──
        if 'utbot' in self.required:
            period = self.state.utbot_atr_period
            mult = self.state.utbot_atr_mult
            alpha_w = 1.0 / period  # Wilder smoothing

            if is_first:
                tr = high - low
                self.state.utbot_atr = tr
                self.state.utbot_trail_stop = close - mult * tr
                self.state.utbot_direction = 0
                self.state.utbot_prev_close = close
            else:
                prev_c = self.state.utbot_prev_close
                tr1 = high - low
                tr2 = abs(high - prev_c)
                tr3 = abs(low - prev_c)
                tr = max(tr1, tr2, tr3)

                self.state.utbot_atr = (
                    self.state.utbot_atr + alpha_w
                    * (tr - self.state.utbot_atr))

                n_loss = mult * self.state.utbot_atr
                prev_stop = self.state.utbot_trail_stop
                prev_dir = self.state.utbot_direction

                if close > prev_stop and prev_c > prev_stop:
                    new_stop = max(prev_stop, close - n_loss)
                elif close < prev_stop and prev_c < prev_stop:
                    new_stop = min(prev_stop, close + n_loss)
                elif close > prev_stop:
                    new_stop = close - n_loss
                else:
                    new_stop = close + n_loss

                if prev_c < prev_stop and close > prev_stop:
                    direction = 1
                elif prev_c > prev_stop and close < prev_stop:
                    direction = -1
                else:
                    direction = prev_dir

                self.state.utbot_trail_stop = new_stop
                self.state.utbot_direction = direction
                self.state.utbot_prev_close = close

            vals['utbot_stop'] = self.state.utbot_trail_stop
            vals['utbot_direction'] = self.state.utbot_direction
            vals['utbot_atr'] = self.state.utbot_atr
            prev_stop_val = self.state.prev_values.get(
                'utbot_stop', self.state.utbot_trail_stop)
            vals['utbot_stop_prev'] = prev_stop_val

        # ── Volume SMA / RVOL ──
        if 'rvol' in self.required:
            buf = self.state.vol_buffer
            if len(buf) == buf.maxlen:
                self.state.vol_sum -= buf[0]
            buf.append(volume)
            self.state.vol_sum += volume
            self.state.vol_count = min(
                self.state.vol_count + 1, self.state.vol_sma_period)

            if self.state.vol_count >= 5:
                vol_sma = self.state.vol_sum / len(buf)
                rvol = volume / vol_sma if vol_sma > 0 else 0.0
            else:
                vol_sma = 0.0
                rvol = 0.0
            vals['vol_sma'] = vol_sma
            vals['rvol'] = rvol

        # ── Store previous EMA values for v2 triggers ──
        if 'ema' in self.required:
            for period in self.state.ema:
                prev_key = f'ema_{period}_prev'
                vals[prev_key] = self.state.prev_values.get(
                    f'ema_{period}', vals.get(f'ema_{period}', 0.0))

        # ── User pack incremental classes ──
        # Each registered pack updates its own state and returns a dict
        # of column values + trigger booleans (under their full
        # `{trigger_prefix}_{base}` keys). Merge into `vals` so the
        # TriggerEvaluator picks them up alongside built-in indicators.
        for slug, engine in getattr(self, '_user_pack_engines', {}).items():
            try:
                out = engine.update_bar(bar)
                if isinstance(out, dict):
                    vals.update(out)
            except Exception as e:
                logger.warning(
                    "user pack %r incremental update failed: %s", slug, e)

        self.state.current = vals


# ═══════════════════════════════════════════════════════════════════════════
# TRIGGER EVALUATOR — single-bar interpreter + trigger evaluation
# ═══════════════════════════════════════════════════════════════════════════

class TriggerEvaluator:
    """Evaluates interpreters and triggers on a single bar's data.

    For backtest mode, also evaluates L-type triggers via reachability.
    """

    def __init__(self, required_interpreters: Set[str],
                 required_triggers: Set[str],
                 ema_periods: List[int]):
        # ema_periods must come from resolve_strategy_requirements — pass []
        # for strategies with no EMA indicator. No internal defaulting.
        if ema_periods is None:
            raise TypeError(
                "TriggerEvaluator requires ema_periods (list). Use [] when "
                "the strategy has no EMA indicator. Pass "
                "resolve_strategy_requirements()['ema_periods'] directly."
            )
        self.required_interpreters = required_interpreters
        self.required_triggers = required_triggers
        self.ema_periods = ema_periods

        # Intra-bar state
        self._ib_fired: Dict[str, bool] = {}
        self._cached_levels: Dict[str, float] = {}
        self._bar_close_triggers: Dict[str, bool] = {}
        self._ib_gate_open: Dict[str, bool] = {}

    def evaluate_bar_close(self, current: Dict[str, float],
                           prev: Dict[str, float],
                           prev2_macd_hist: float = 0.0,
                           ) -> Tuple[Dict[str, str], Dict[str, bool]]:
        """Evaluate interpreters and C-type triggers on bar close.

        Returns:
            (interpreter_states, trigger_booleans)
        """
        interps: Dict[str, str] = {}
        triggers: Dict[str, bool] = {}

        # ── Interpreters ──
        if 'EMA_STACK' in self.required_interpreters:
            s, m, l_ = self._get_ema_triple(current)
            if s is not None:
                interps['EMA_STACK'] = self._classify_ema_stack(s, m, l_)

        if 'EMA_PRICE_POSITION' in self.required_interpreters:
            interps['EMA_PRICE_POSITION'] = self._classify_ema_price_pos(
                current)

        if 'EMA_PRICE_POSITION_V2' in self.required_interpreters:
            interps['EMA_PRICE_POSITION_V2'] = self._classify_ema_price_pos(
                current)

        if 'MACD_LINE' in self.required_interpreters:
            ml = current.get('macd_line', 0)
            ms = current.get('macd_signal', 0)
            if ml > ms:
                interps['MACD_LINE'] = 'M>S+' if ml > 0 else 'M>S-'
            else:
                interps['MACD_LINE'] = 'M<S-' if ml <= 0 else 'M<S+'

        if 'MACD_HISTOGRAM' in self.required_interpreters:
            mh = current.get('macd_hist', 0)
            pmh = prev.get('macd_hist', 0)
            if mh > 0:
                interps['MACD_HISTOGRAM'] = (
                    'H+up' if mh > pmh else 'H+dn')
            else:
                interps['MACD_HISTOGRAM'] = (
                    'H-dn' if mh <= pmh else 'H-up')

        if 'VWAP' in self.required_interpreters:
            interps['VWAP'] = self._classify_vwap(current)

        if 'RVOL' in self.required_interpreters:
            rv = current.get('rvol', 0)
            if rv > 3.0:
                interps['RVOL'] = 'EXTREME'
            elif rv > 1.5:
                interps['RVOL'] = 'HIGH'
            elif rv > 0.75:
                interps['RVOL'] = 'NORMAL'
            elif rv > 0.5:
                interps['RVOL'] = 'LOW'
            else:
                interps['RVOL'] = 'MINIMAL'

        if 'UTBOT' in self.required_interpreters:
            d = current.get('utbot_direction', 0)
            if d == 1:
                interps['UTBOT'] = 'BULL'
            elif d == -1:
                interps['UTBOT'] = 'BEAR'

        if 'UTBOT_V2' in self.required_interpreters:
            d = current.get('utbot_direction', 0)
            if d == 1:
                interps['UTBOT_V2'] = 'BULL'
            elif d == -1:
                interps['UTBOT_V2'] = 'BEAR'

        # ── User-pack interpreters (generic dispatch) ──
        # Built-in interpreters above are inlined for speed. Any required
        # interpreter NOT handled above is presumed to come from a user
        # pack: look up the pack via pack_registry, build a 1-row
        # DataFrame from `current`, and call the pack's interpreter
        # function. This restores live confluence-record emission for
        # user-pack interpreters — without it, strategies that gate on a
        # `<TF>-<USER_PACK_INTERPRETER>-<state>` confluence record never
        # see the gate satisfied in live mode (backtest path uses
        # evaluate_bar_for_backtest which merges pre-computed states).
        unhandled = self.required_interpreters - set(interps)
        if unhandled:
            try:
                import pack_registry
                # Build a 2-row DataFrame [prev, current] (or 1-row if prev
                # is empty/missing). User-pack interpreters often compute
                # state from row vs row-1 via `df.shift(1)` — e.g.
                # MACD_HISTOGRAM_V2 classifies H+up/H+dn by whether hist
                # rose or fell vs the previous bar. A single-row DataFrame
                # makes shift(1) → NaN and the interpreter emits None for
                # every bar, silently dropping the cross-TF confluence
                # record. Confirmed broken on sid 137 (TSLA/5Min) where
                # the 1Day MACD_HISTOGRAM_V2 shadow emitted 0 records
                # across 33 bars fed, blocking every entry. Found via
                # `_probe_parity_divergence.py` 2026-04-29.
                row_df = None
                for pack in pack_registry.get_registered_packs().values():
                    pack_interps = set(
                        pack.manifest.get('interpreters', []))
                    overlap = pack_interps & unhandled
                    if not overlap or pack.interpreter_func is None:
                        continue
                    if row_df is None:
                        if prev:
                            row_df = pd.DataFrame([prev, current])
                        else:
                            row_df = pd.DataFrame([current])
                    try:
                        states = pack.interpreter_func(row_df)
                        if states is not None and len(states) > 0:
                            latest = states.iloc[-1]
                            for ikey in overlap:
                                if latest is not None and not (
                                    isinstance(latest, float)
                                    and pd.isna(latest)
                                ):
                                    interps[ikey] = latest
                    except Exception as e:
                        logger.warning(
                            "user-pack interpreter %s failed live "
                            "dispatch: %s", overlap, e)
            except Exception as e:
                logger.warning(
                    "pack_registry lookup failed in evaluate_bar_close: "
                    "%s", e)

        # ── C-type Triggers (crossover detection: current vs prev) ──

        # EMA Stack triggers
        if 'ema_cross_bull' in self.required_triggers:
            triggers['ema_cross_bull'] = (
                current.get('ema_8', 0) > current.get('ema_21', 0)
                and prev.get('ema_8', 0) <= prev.get('ema_21', 0))
        if 'ema_cross_bear' in self.required_triggers:
            triggers['ema_cross_bear'] = (
                current.get('ema_8', 0) < current.get('ema_21', 0)
                and prev.get('ema_8', 0) >= prev.get('ema_21', 0))
        if 'ema_mid_cross_bull' in self.required_triggers:
            triggers['ema_mid_cross_bull'] = (
                current.get('ema_21', 0) > current.get('ema_50', 0)
                and prev.get('ema_21', 0) <= prev.get('ema_50', 0))
        if 'ema_mid_cross_bear' in self.required_triggers:
            triggers['ema_mid_cross_bear'] = (
                current.get('ema_21', 0) < current.get('ema_50', 0)
                and prev.get('ema_21', 0) >= prev.get('ema_50', 0))

        # EMA Price Position triggers (bar-close booleans)
        for prefix in ('ema_pp', 'ema_pp_v2'):
            sp = self.ema_periods[0] if len(self.ema_periods) > 0 else 8
            mp = self.ema_periods[1] if len(self.ema_periods) > 1 else 21
            s_key = f'ema_{sp}'
            m_key = f'ema_{mp}'

            t = f'{prefix}_cross_short_up'
            if t in self.required_triggers:
                triggers[t] = (
                    current.get('close', 0) > current.get(s_key, 0)
                    and prev.get('close', 0) <= prev.get(s_key, 0))
            t = f'{prefix}_cross_short_down'
            if t in self.required_triggers:
                triggers[t] = (
                    current.get('close', 0) < current.get(s_key, 0)
                    and prev.get('close', 0) >= prev.get(s_key, 0))
            t = f'{prefix}_cross_mid_up'
            if t in self.required_triggers:
                triggers[t] = (
                    current.get('close', 0) > current.get(m_key, 0)
                    and prev.get('close', 0) <= prev.get(m_key, 0))
            t = f'{prefix}_cross_mid_down'
            if t in self.required_triggers:
                triggers[t] = (
                    current.get('close', 0) < current.get(m_key, 0)
                    and prev.get('close', 0) >= prev.get(m_key, 0))

        # MACD triggers
        if 'macd_cross_bull' in self.required_triggers:
            triggers['macd_cross_bull'] = (
                current.get('macd_line', 0) > current.get('macd_signal', 0)
                and prev.get('macd_line', 0) <= prev.get('macd_signal', 0))
        if 'macd_cross_bear' in self.required_triggers:
            triggers['macd_cross_bear'] = (
                current.get('macd_line', 0) < current.get('macd_signal', 0)
                and prev.get('macd_line', 0) >= prev.get('macd_signal', 0))
        if 'macd_zero_cross_up' in self.required_triggers:
            triggers['macd_zero_cross_up'] = (
                current.get('macd_line', 0) > 0
                and prev.get('macd_line', 0) <= 0)
        if 'macd_zero_cross_down' in self.required_triggers:
            triggers['macd_zero_cross_down'] = (
                current.get('macd_line', 0) < 0
                and prev.get('macd_line', 0) >= 0)

        # MACD Histogram triggers
        if 'macd_hist_flip_pos' in self.required_triggers:
            triggers['macd_hist_flip_pos'] = (
                current.get('macd_hist', 0) > 0
                and prev.get('macd_hist', 0) <= 0)
        if 'macd_hist_flip_neg' in self.required_triggers:
            triggers['macd_hist_flip_neg'] = (
                current.get('macd_hist', 0) < 0
                and prev.get('macd_hist', 0) >= 0)
        if 'macd_hist_momentum_shift_up' in self.required_triggers:
            mh = current.get('macd_hist', 0)
            pmh = prev.get('macd_hist', 0)
            triggers['macd_hist_momentum_shift_up'] = (
                mh > pmh and pmh < prev2_macd_hist)
        if 'macd_hist_momentum_shift_down' in self.required_triggers:
            mh = current.get('macd_hist', 0)
            pmh = prev.get('macd_hist', 0)
            triggers['macd_hist_momentum_shift_down'] = (
                mh < pmh and pmh > prev2_macd_hist)

        # VWAP triggers
        if 'vwap_cross_above' in self.required_triggers:
            triggers['vwap_cross_above'] = (
                current.get('close', 0) > current.get('vwap', 0)
                and prev.get('close', 0) <= prev.get('vwap', 0))
        if 'vwap_cross_below' in self.required_triggers:
            triggers['vwap_cross_below'] = (
                current.get('close', 0) < current.get('vwap', 0)
                and prev.get('close', 0) >= prev.get('vwap', 0))
        if 'vwap_enter_upper_extreme' in self.required_triggers:
            triggers['vwap_enter_upper_extreme'] = (
                current.get('close', 0) > current.get('vwap_sd2_upper', 0)
                and prev.get('close', 0)
                <= prev.get('vwap_sd2_upper', 0))
        if 'vwap_enter_lower_extreme' in self.required_triggers:
            triggers['vwap_enter_lower_extreme'] = (
                current.get('close', 0) < current.get('vwap_sd2_lower', 0)
                and prev.get('close', 0)
                >= prev.get('vwap_sd2_lower', 0))
        if 'vwap_return_to_vwap' in self.required_triggers:
            pc = prev.get('close', 0)
            pv1u = prev.get('vwap_sd1_upper', 0)
            pv1l = prev.get('vwap_sd1_lower', 0)
            was_extreme = pc > pv1u or pc < pv1l
            v = current.get('vwap', 0)
            v1u = current.get('vwap_sd1_upper', 0)
            half_sd = (v1u - v) * 0.5 if v1u > v else v * 0.001
            c = current.get('close', 0)
            now_at_vwap = (v - half_sd) <= c <= (v + half_sd)
            triggers['vwap_return_to_vwap'] = was_extreme and now_at_vwap

        # RVOL triggers
        if 'rvol_spike' in self.required_triggers:
            triggers['rvol_spike'] = (
                current.get('rvol', 0) > 1.5
                and prev.get('rvol', 0) <= 1.5)
        if 'rvol_extreme' in self.required_triggers:
            triggers['rvol_extreme'] = (
                current.get('rvol', 0) > 3.0
                and prev.get('rvol', 0) <= 3.0)
        if 'rvol_fade' in self.required_triggers:
            triggers['rvol_fade'] = (
                current.get('rvol', 0) < 1.0
                and prev.get('rvol', 0) >= 1.0)

        # UT Bot triggers
        if 'utbot_buy' in self.required_triggers:
            triggers['utbot_buy'] = (
                current.get('utbot_direction', 0) == 1
                and prev.get('utbot_direction', 0) != 1)
        if 'utbot_sell' in self.required_triggers:
            triggers['utbot_sell'] = (
                current.get('utbot_direction', 0) == -1
                and prev.get('utbot_direction', 0) != -1)
        if 'utbot_v2_buy' in self.required_triggers:
            triggers['utbot_v2_buy'] = (
                current.get('utbot_direction', 0) == 1
                and prev.get('utbot_direction', 0) != 1)
        if 'utbot_v2_sell' in self.required_triggers:
            triggers['utbot_v2_sell'] = (
                current.get('utbot_direction', 0) == -1
                and prev.get('utbot_direction', 0) != -1)

        # ── User pack triggers ──
        # Any required trigger that the built-in branches above didn't
        # populate may have been written into `current` by a user
        # pack's incremental class (under `{trigger_prefix}_{base}`).
        # Pick those up by direct lookup. Suffixed runtime variants
        # (`_ib`, `_lc`, `_cc`) are handled by their respective
        # execution_types modules off the base trigger; we only need to
        # source the C-variant boolean here.
        for trig_id in self.required_triggers:
            if trig_id in triggers:
                continue  # already produced by a built-in branch above
            val = current.get(trig_id)
            if isinstance(val, bool):
                triggers[trig_id] = val

        # Store for gating and caching
        self._bar_close_triggers = dict(triggers)
        self._update_cached_levels(current)
        self._compute_ib_gates(current)
        self._ib_fired.clear()

        return interps, triggers

    def evaluate_bar_for_backtest(self, current: Dict[str, float],
                                  prev: Dict[str, float],
                                  prev2_macd_hist: float = 0.0,
                                  ) -> Tuple[Dict[str, str],
                                             Dict[str, bool],
                                             Dict[str, float]]:
        """Evaluate all trigger types for one bar in backtest mode.

        Calls evaluate_bar_close() for C-type triggers and interpreters,
        then simulates L-type triggers via gate check + reachability.

        Timing: In live mode, gates are set at bar N-1's close and
        crossings are checked during bar N (different bars).  To match
        this, the backtest saves bar N-1's gate and cached levels and
        uses them for bar N's reachability checks.  H-type triggers
        (which require a C-type trigger on the same bar) still use the
        current bar's bar_close_triggers.

        Returns:
            (interpreter_states, c_type_triggers, l_type_fills)
            l_type_fills: {trigger_id: fill_price} for L0/L1-type triggers
                that passed both gate and reachability checks.
        """
        # Save previous bar's L-type state before evaluate_bar_close
        # overwrites it.  This matches live timing where the gate is
        # set at bar N-1's close and crossings happen during bar N.
        prev_gates = dict(self._ib_gate_open)
        prev_cached = dict(self._cached_levels)

        interps, c_triggers = self.evaluate_bar_close(
            current, prev, prev2_macd_hist)

        l_fills: Dict[str, float] = {}

        # Evaluate L-type triggers via gate + reachability
        high = current.get('high', 0)
        low = current.get('low', 0)
        bar_open = current.get('open', 0)

        # Temporarily swap to previous bar's cached levels for crossing
        # checks (matches live mode where levels are cached at bar N-1
        # close and used during bar N).
        current_cached = self._cached_levels
        self._cached_levels = prev_cached

        for trigger_id, (level, direction) in self._get_ib_checks():
            base_trigger = _strip_exec_suffix(trigger_id)
            exec_type = get_trigger_exec_type(trigger_id)

            # L-type gate: use PREVIOUS bar's gate (matches live timing)
            if exec_type in ('HM', 'HL', 'LC') or base_trigger in _IB_L_TYPE_TRIGGERS:
                if not prev_gates.get(trigger_id, False):
                    continue
            else:
                # H-type (legacy UT Bot V1 _ib): bar-close trigger on
                # THIS bar must have fired
                if not self._bar_close_triggers.get(base_trigger, False):
                    continue

            # Reachability check
            # For user pack triggers, require true cross (bar touched both sides of the level).
            # This prevents false fills when a bar gaps open above/below the level without
            # actually crossing it intra-bar.
            is_user_pack = base_trigger in INTRABAR_LEVEL_MAP and base_trigger not in (
                'vwap_cross_above', 'vwap_cross_below', 'vwap_enter_upper_extreme', 'vwap_enter_lower_extreme',
                'utbot_buy', 'utbot_sell', 'utbot_v2_buy', 'utbot_v2_sell',
                'ema_pp_cross_short_up', 'ema_pp_cross_short_down', 'ema_pp_cross_mid_up', 'ema_pp_cross_mid_down',
                'ema_pp_v2_cross_short_up', 'ema_pp_v2_cross_short_down', 'ema_pp_v2_cross_mid_up', 'ema_pp_v2_cross_mid_down',
            )
            if is_user_pack:
                # Strict cross check: bar must have touched both sides of the level
                if direction == 'above' and low < level <= high:
                    l_fills[trigger_id] = level
                elif direction == 'below' and high > level >= low:
                    l_fills[trigger_id] = level
            else:
                # Built-in gap-through semantic: when the previous bar's level
                # sits outside the current bar's range (common for v2 _prev
                # triggers on illiquid / degenerate 10-sec bars), price never
                # traded at `level` on this bar. Fill at bar_open as the
                # first realistic touchable price on the bar. Guarantees
                # entry_price ∈ [bar_low, bar_high]. For normal crosses
                # (level inside the bar), max/min reduces to `level`.
                if direction == 'above' and high >= level:
                    l_fills[trigger_id] = max(level, bar_open)
                elif direction == 'below' and low <= level:
                    l_fills[trigger_id] = min(level, bar_open)

        # Restore current bar's cached levels for live-mode compatibility
        self._cached_levels = current_cached

        return interps, c_triggers, l_fills

    def check_intrabar(self, price: float) -> Optional[Tuple[str, float]]:
        """Check if tick price crosses any cached level (O(1) per level).

        Returns (trigger_id, fill_price) if a crossing is detected,
        or None.  Each trigger fires at most once per bar.

        Gate logic:
        - L-type (_ib) + HM/HL with base in _IB_L_TYPE_TRIGGERS: use _ib_gate_open
          (prev-close-opposite-side check).  Includes VWAP, EMA, EMA V2, UT Bot V2.
        - HM/HL with base NOT in _IB_L_TYPE_TRIGGERS: use _ib_gate_open
          (HM/HL always use L-type gate, per Phase 30B design)
        - H-type (legacy UT Bot V1 _ib only): use _bar_close_triggers
        """
        for trigger_id, (level, direction) in self._get_ib_checks():
            if self._ib_fired.get(trigger_id, False):
                continue
            base_trigger = _strip_exec_suffix(trigger_id)
            exec_type = get_trigger_exec_type(trigger_id)

            # Gate check
            if exec_type in ('HM', 'HL', 'LC') or base_trigger in _IB_L_TYPE_TRIGGERS:
                if not self._ib_gate_open.get(trigger_id, False):
                    continue
            else:
                # H-type (legacy UT Bot V1 _ib): bar-close trigger must have fired
                if not self._bar_close_triggers.get(base_trigger, False):
                    continue

            # Fill at the tick price that actually triggered the cross, not
            # at the static `level`. `price` is by construction on the correct
            # side of `level` (above for LONG, below for SHORT) and always
            # within the forming bar's range, so entry_price ∈ [bar_low,
            # bar_high] is guaranteed when the bar closes. This matches the
            # backtest max/min(level, bar_open) semantic and captures realistic
            # slippage on fast moves / gap-throughs.
            if direction == 'above' and price > level:
                self._ib_fired[trigger_id] = True
                return (trigger_id, price)
            elif direction == 'below' and price < level:
                self._ib_fired[trigger_id] = True
                return (trigger_id, price)
        return None

    def _get_ib_checks(self) -> List[Tuple[str, Tuple[float, str]]]:
        """Return list of (trigger_id, (level, direction)) for intra-bar."""
        checks = []
        IB_MAP = {
            'vwap_cross_above': ('vwap', 'above'),
            'vwap_cross_below': ('vwap', 'below'),
            'vwap_enter_upper_extreme': ('vwap_sd2_upper', 'above'),
            'vwap_enter_lower_extreme': ('vwap_sd2_lower', 'below'),
            'utbot_buy': ('utbot_stop', 'above'),
            'utbot_sell': ('utbot_stop', 'below'),
            'utbot_v2_buy': ('utbot_stop_prev', 'above'),
            'utbot_v2_sell': ('utbot_stop_prev', 'below'),
        }
        if self.ema_periods:
            sp = self.ema_periods[0]
            mp = self.ema_periods[1] if len(self.ema_periods) > 1 else 21
            IB_MAP.update({
                'ema_pp_cross_short_up': (f'ema_{sp}', 'above'),
                'ema_pp_cross_short_down': (f'ema_{sp}', 'below'),
                'ema_pp_cross_mid_up': (f'ema_{mp}', 'above'),
                'ema_pp_cross_mid_down': (f'ema_{mp}', 'below'),
                'ema_pp_v2_cross_short_up': (f'ema_{sp}_prev', 'above'),
                'ema_pp_v2_cross_short_down': (f'ema_{sp}_prev', 'below'),
                'ema_pp_v2_cross_mid_up': (f'ema_{mp}_prev', 'above'),
                'ema_pp_v2_cross_mid_down': (f'ema_{mp}_prev', 'below'),
            })

        for base_trigger, (level_key, direction) in IB_MAP.items():
            for suffix in ('_ib', '_hm', '_hl', '_lc'):
                trigger_id = f'{base_trigger}{suffix}'
                if trigger_id in self.required_triggers:
                    level = self._cached_levels.get(level_key)
                    if level is not None and level > 0:
                        checks.append((trigger_id, (level, direction)))

        # User pack triggers from INTRABAR_LEVEL_MAP
        for base_trigger, level_spec in INTRABAR_LEVEL_MAP.items():
            if base_trigger in IB_MAP:
                continue  # already handled above
            level_col = level_spec.get('column', '')
            direction = level_spec.get('cross', 'above')
            for suffix in ('_ib', '_hm', '_hl', '_lc'):
                trigger_id = f'{base_trigger}{suffix}'
                if trigger_id in self.required_triggers:
                    level = self._cached_levels.get(level_col)
                    if level is not None and level > 0:
                        checks.append((trigger_id, (level, direction)))
        return checks

    def _update_cached_levels(self, current: Dict[str, float]):
        """Cache indicator levels for intra-bar crossing detection.

        For _prev keys, cache the CURRENT bar's non-_prev value.
        Matches backtest .shift(1) semantics.
        """
        for key in ('vwap', 'vwap_sd2_upper', 'vwap_sd2_lower',
                    'utbot_stop'):
            if key in current:
                self._cached_levels[key] = current[key]
        if 'utbot_stop' in current:
            self._cached_levels['utbot_stop_prev'] = current['utbot_stop']
        for period in self.ema_periods:
            key = f'ema_{period}'
            if key in current:
                self._cached_levels[key] = current[key]
                self._cached_levels[f'ema_{period}_prev'] = current[key]
        # User pack level columns: cache any columns referenced in INTRABAR_LEVEL_MAP
        for spec in INTRABAR_LEVEL_MAP.values():
            level_col = spec.get('column', '')
            if level_col and level_col in current:
                val = current[level_col]
                if val is not None and not (isinstance(val, float) and val != val):  # not NaN
                    self._cached_levels[level_col] = val

    def _compute_ib_gates(self, current: Dict[str, float]):
        """Compute L-type intra-bar gate states from the just-closed bar."""
        self._ib_gate_open.clear()
        close = current.get('close', 0)

        sp = self.ema_periods[0] if self.ema_periods else 8
        mp = self.ema_periods[1] if len(self.ema_periods) > 1 else 21

        gate_map = {
            'vwap_cross_above':          ('vwap', 'above'),
            'vwap_cross_below':          ('vwap', 'below'),
            'vwap_enter_upper_extreme':  ('vwap_sd2_upper', 'above'),
            'vwap_enter_lower_extreme':  ('vwap_sd2_lower', 'below'),
            'utbot_buy':                 ('utbot_stop', 'above'),
            'utbot_sell':                ('utbot_stop', 'below'),
            'utbot_v2_buy':              ('utbot_stop', 'above'),
            'utbot_v2_sell':             ('utbot_stop', 'below'),
            'ema_pp_cross_short_up':     (f'ema_{sp}', 'above'),
            'ema_pp_cross_short_down':   (f'ema_{sp}', 'below'),
            'ema_pp_cross_mid_up':       (f'ema_{mp}', 'above'),
            'ema_pp_cross_mid_down':     (f'ema_{mp}', 'below'),
            'ema_pp_v2_cross_short_up':  (f'ema_{sp}', 'above'),
            'ema_pp_v2_cross_short_down':(f'ema_{sp}', 'below'),
            'ema_pp_v2_cross_mid_up':    (f'ema_{mp}', 'above'),
            'ema_pp_v2_cross_mid_down':  (f'ema_{mp}', 'below'),
        }

        for base_trigger, (level_key, direction) in gate_map.items():
            for suffix in ('_ib', '_hm', '_hl', '_lc'):
                trigger_id = f'{base_trigger}{suffix}'
                if trigger_id not in self.required_triggers:
                    continue
                level = current.get(level_key, 0)
                if level <= 0:
                    continue
                if direction == 'above':
                    self._ib_gate_open[trigger_id] = (close <= level)
                else:
                    self._ib_gate_open[trigger_id] = (close >= level)

        # User pack triggers: open gates for any L-type triggers registered
        # in INTRABAR_LEVEL_MAP that aren't already in the hardcoded gate_map
        for base_trigger, level_spec in INTRABAR_LEVEL_MAP.items():
            if base_trigger in gate_map:
                continue  # already handled above
            level_col = level_spec.get('column', '')
            direction = level_spec.get('cross', 'above')
            for suffix in ('_ib', '_hm', '_hl', '_lc'):
                trigger_id = f'{base_trigger}{suffix}'
                if trigger_id not in self.required_triggers:
                    continue
                level = current.get(level_col, 0)
                if level is None or (isinstance(level, float) and (level != level)):  # NaN check
                    continue
                if level <= 0:
                    continue
                if direction == 'above':
                    self._ib_gate_open[trigger_id] = (close <= level)
                else:
                    self._ib_gate_open[trigger_id] = (close >= level)

    # ── Confirmation check (HM/HL) ──

    def check_confirmation(self, base_trigger: str,
                           current: Dict[str, float]) -> bool:
        """Check if bar close confirms the entry's cross direction.

        For HM/HL entries: after entering intra-bar at the indicator level,
        confirm that the bar closed on the same side as the cross direction.
        Returns True if confirmed (trade continues normally).
        """
        entry_info = INTRABAR_LEVEL_MAP.get(base_trigger)
        if not entry_info:
            return True  # Unknown trigger, assume confirmed

        col = entry_info['column']
        cross = entry_info['cross']

        # For _prev columns (V2 triggers), use current indicator value
        if col.endswith('_prev'):
            col = col[:-5]

        # Resolve parameterized EMA columns
        if 'param_key' in entry_info:
            param_key = entry_info['param_key']
            if 'short' in param_key and self.ema_periods:
                col = f'ema_{self.ema_periods[0]}'
            elif 'mid' in param_key and len(self.ema_periods) > 1:
                col = f'ema_{self.ema_periods[1]}'

        level = current.get(col, 0)
        close = current.get('close', 0)

        if level <= 0:
            return True  # Can't determine, assume confirmed

        if cross == 'above':
            return close > level
        else:
            return close < level

    # ── Interpreter helpers ──

    def _get_ema_triple(self, vals):
        if len(self.ema_periods) >= 3:
            s = vals.get(f'ema_{self.ema_periods[0]}')
            m = vals.get(f'ema_{self.ema_periods[1]}')
            l_ = vals.get(f'ema_{self.ema_periods[2]}')
            return s, m, l_
        return None, None, None

    @staticmethod
    def _classify_ema_stack(s, m, l_) -> str:
        if s > m > l_:
            return 'SML'
        elif s > l_ > m:
            return 'SLM'
        elif m > s > l_:
            return 'MSL'
        elif m > l_ > s:
            return 'MLS'
        elif l_ > s > m:
            return 'LSM'
        elif l_ > m > s:
            return 'LMS'
        return 'SML'

    def _classify_ema_price_pos(self, vals) -> str:
        s, m, l_ = self._get_ema_triple(vals)
        c = vals.get('close', 0)
        if s is None:
            return 'PSML'
        items = [('P', c), ('S', s), ('M', m), ('L', l_)]
        items.sort(key=lambda x: -x[1])
        return ''.join(x[0] for x in items)

    @staticmethod
    def _classify_vwap(vals) -> str:
        c = vals.get('close', 0)
        v = vals.get('vwap', 0)
        v2u = vals.get('vwap_sd2_upper', v)
        v1u = vals.get('vwap_sd1_upper', v)
        v1l = vals.get('vwap_sd1_lower', v)
        v2l = vals.get('vwap_sd2_lower', v)
        half_sd = (v1u - v) * 0.5 if v1u > v else v * 0.001

        if c > v2u:
            return '>+2sigma'
        elif c > v1u:
            return '>+1sigma'
        elif c > v + half_sd:
            return '>V'
        elif c >= v - half_sd:
            return '@V'
        elif c >= v1l:
            return '<V'
        elif c >= v2l:
            return '<-1sigma'
        else:
            return '<-2sigma'


# ═══════════════════════════════════════════════════════════════════════════
# POSITION STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PositionState:
    """Persistent position state for one strategy."""
    status: str = 'FLAT'
    entry_price: float = 0.0
    entry_time: Optional[str] = None
    stop_price: float = 0.0
    initial_stop_price: float = 0.0
    target_price: Optional[float] = None
    entry_bar_count: int = 0
    last_exit_bar_count: int = 0
    direction: str = 'LONG'
    entry_trigger: str = ''
    confluence_records: Optional[set] = None
    exec_type: str = 'C'            # C, L0, L1, HM, HL, LC, CC
    pending_hm_exit: bool = False    # Exit at next bar open (HM unconfirmed)
    pending_hl_limit: bool = False   # Limit exit at entry_price (HL unconfirmed)
    # LC/CC confirmation fields
    pending_confirm_bar: int = -1    # Bar count when confirmation is due (-1 = none)
    bail_action: str = 'exit_market' # exit_market | exit_limit | exit_limit_breakeven
    hold_seconds: int = 0            # Minimum hold time before exit (L/LC)
    # CC-type stop/target confirmation tracking (M5.5)
    pending_stop_confirm_bar: Optional[int] = None
    pending_target_confirm_bar: Optional[int] = None
    # Trade_Timestamps_Spec (2026-04-17): 4 timestamps per trade, + metadata.
    # entry_trigger_ts = when the entry condition first became true.
    # entry_fill_ts    = when the position actually opened (per exec manifest).
    # exit_trigger_ts  = when the exit condition first became true.
    # exit_fill_ts     = when the position actually closed.
    # hold_duration_s  = configured hold from exec manifest (descriptive).
    # behavior         = 'A' (wait-then-fill) or 'B' (fill-then-validate).
    # `entry_time` remains as a transitional alias (= entry_fill_ts) until
    # Step 6 of the spec execution plan drops legacy aliases.
    entry_trigger_ts: Optional[str] = None
    entry_fill_ts: Optional[str] = None
    exit_trigger_ts: Optional[str] = None
    exit_fill_ts: Optional[str] = None
    hold_duration_s: int = 0
    behavior: str = 'B'

    def to_dict(self) -> dict:
        return {
            'status': self.status,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            'stop_price': self.stop_price,
            'initial_stop_price': self.initial_stop_price,
            'target_price': self.target_price,
            'entry_bar_count': self.entry_bar_count,
            'last_exit_bar_count': self.last_exit_bar_count,
            'direction': self.direction,
            'entry_trigger': self.entry_trigger,
            'exec_type': self.exec_type,
            'pending_hm_exit': self.pending_hm_exit,
            'pending_hl_limit': self.pending_hl_limit,
            'pending_confirm_bar': self.pending_confirm_bar,
            'bail_action': self.bail_action,
            'hold_seconds': self.hold_seconds,
            'pending_stop_confirm_bar': self.pending_stop_confirm_bar,
            'pending_target_confirm_bar': self.pending_target_confirm_bar,
            'entry_trigger_ts': self.entry_trigger_ts,
            'entry_fill_ts': self.entry_fill_ts,
            'exit_trigger_ts': self.exit_trigger_ts,
            'exit_fill_ts': self.exit_fill_ts,
            'hold_duration_s': self.hold_duration_s,
            'behavior': self.behavior,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'PositionState':
        return cls(
            status=d.get('status', 'FLAT'),
            entry_price=d.get('entry_price', 0.0),
            entry_time=d.get('entry_time'),
            stop_price=d.get('stop_price', 0.0),
            initial_stop_price=d.get('initial_stop_price', 0.0),
            target_price=d.get('target_price'),
            entry_bar_count=d.get('entry_bar_count', 0),
            last_exit_bar_count=d.get('last_exit_bar_count', 0),
            direction=d.get('direction', 'LONG'),
            entry_trigger=d.get('entry_trigger', ''),
            exec_type=d.get('exec_type', 'C'),
            pending_hm_exit=d.get('pending_hm_exit', False),
            pending_hl_limit=d.get('pending_hl_limit', False),
            pending_confirm_bar=d.get('pending_confirm_bar', -1),
            bail_action=d.get('bail_action', 'exit_market'),
            hold_seconds=d.get('hold_seconds', 0),
            pending_stop_confirm_bar=d.get('pending_stop_confirm_bar'),
            pending_target_confirm_bar=d.get('pending_target_confirm_bar'),
            entry_trigger_ts=d.get('entry_trigger_ts'),
            entry_fill_ts=d.get('entry_fill_ts'),
            exit_trigger_ts=d.get('exit_trigger_ts'),
            exit_fill_ts=d.get('exit_fill_ts'),
            hold_duration_s=d.get('hold_duration_s', 0),
            behavior=d.get('behavior', 'B'),
        )


class PositionStateMachine:
    """Manages FLAT <-> IN_POSITION transitions for one strategy."""

    def __init__(self, strategy: dict, state: Optional[PositionState] = None,
                 resolved_entry: str = '', resolved_exits: List[str] = None):
        self.strategy = strategy
        self.state = state or PositionState(
            direction=strategy.get('direction', 'LONG'))
        self.strat_id = strategy.get('id', '')

        self.entry_trigger = resolved_entry or strategy.get('entry_trigger', '')
        self.exit_triggers: Set[str] = set()
        if resolved_exits:
            self.exit_triggers = set(resolved_exits)
        elif strategy.get('exit_triggers'):
            self.exit_triggers = set(strategy['exit_triggers'])
        elif strategy.get('exit_trigger'):
            et = strategy['exit_trigger']
            if et == 'opposite_signal':
                try:
                    from triggers import get_opposite_trigger
                    opp = get_opposite_trigger(self.entry_trigger)
                    if opp:
                        self.exit_triggers = {opp}
                except ImportError:
                    pass
            elif et:
                self.exit_triggers = {et}

        self.bar_count_exit = strategy.get('bar_count_exit')
        self.time_exit_config = strategy.get('time_exit_config')
        self.stop_config = strategy.get('stop_config') or {
            'method': 'atr', 'atr_mult': strategy.get('stop_atr_mult', 1.5)}
        self.target_config = strategy.get('target_config')
        self.confluence_set = set(strategy.get('confluence', [])) | set(
            strategy.get('general_confluences', []))
        self.confluence_set = self.confluence_set or None

        # Rolling buffer for swing stop/target lookback
        self._high_low_buffer: deque = deque(maxlen=50)

        # Trade_Timestamps_Spec: bar duration in seconds, used by
        # execution_types.compute_fill_ts to derive fill_ts from trigger_ts.
        # Resolved once here so per-bar calls don't repeat the lookup.
        self.tf_seconds = TIMEFRAME_SECONDS.get(
            strategy.get('timeframe', '1Min'), 60)

    def update_high_low(self, high: float, low: float):
        """Append a (high, low) pair to the rolling buffer for swing lookback."""
        self._high_low_buffer.append((high, low))

    def _get_exec_variant_param(self, trigger_id: str, variant_key: str,
                                param_name: str, default=None):
        """Look up an exec_variant parameter from the confluence group template.

        Searches TEMPLATES for the trigger's template and returns the
        specified parameter from exec_variants[variant_key] on the matching
        trigger.  Falls back to *default* if not found.
        """
        try:
            from confluence_groups import TEMPLATES, TRIGGER_PREFIX_TO_TEMPLATE
            base = _strip_exec_suffix(trigger_id)
            # Find which template this trigger belongs to
            for tp, tmpl_slug in TRIGGER_PREFIX_TO_TEMPLATE.items():
                if base.startswith(tp + '_') or base == tp:
                    tmpl = TEMPLATES.get(tmpl_slug)
                    if not tmpl:
                        continue
                    # Match the trigger's base in the template's triggers list
                    suffix = base[len(tp) + 1:] if base.startswith(tp + '_') else base
                    for trig in tmpl.get('triggers', []):
                        if suffix.endswith(trig['base']) or trig['base'] == suffix:
                            ev = trig.get('exec_variants', {}).get(variant_key, {})
                            return ev.get(param_name, default)
        except Exception:
            pass
        return default

    def check_entry(self, trigger_booleans: Dict[str, bool],
                    current_values: Dict[str, float],
                    bar_count: int, bar_time: str,
                    confluence_records: Set[str] = None,
                    l_type_fills: Dict[str, float] = None,
                    prev_values: Dict[str, float] = None,
                    ) -> Optional[dict]:
        """Check for entry signal (C-type or L-type).

        For L-type triggers, checks l_type_fills dict.
        For C-type triggers, checks trigger_booleans dict.
        prev_values: Previous bar's indicator values.  Used for L-type
            stop/target computation to match live behaviour (intra-bar
            entries don't have the current bar's indicators yet).

        Returns signal dict or None.
        """
        if self.state.status != 'FLAT':
            return None

        # 1-bar cooldown: don't re-enter on the same bar we exited
        if self.state.last_exit_bar_count >= bar_count:
            return None

        trigger_id = self.entry_trigger
        exec_type = get_trigger_exec_type(trigger_id)

        # Delegate entry signal detection to execution type module
        from execution_types import get_module
        module = get_module(exec_type)
        entry_result = module.check_entry_signal(
            trigger_id, exec_type, trigger_booleans,
            l_type_fills or {}, current_values, _strip_exec_suffix)

        if not entry_result.fired:
            return None

        fill_price = entry_result.fill_price
        is_ltype = entry_result.is_ltype

        # Confluence check
        if self.confluence_set and confluence_records:
            if not self.confluence_set.issubset(confluence_records):
                return None

        # For L-type entries, use previous bar's indicator values for
        # stop/target computation — matches live engine behaviour where
        # the current bar hasn't closed yet at time of intra-bar entry.
        vals_for_stop = (prev_values if is_ltype and prev_values
                         else current_values)
        atr = vals_for_stop.get('atr', fill_price * 0.01)
        if not atr or atr <= 0:
            atr = fill_price * 0.01

        stop_price = self._compute_stop(fill_price, atr, vals_for_stop)

        # Stop-validity guard: reject entry if stop is on the wrong side
        if self.state.direction == 'LONG' and stop_price >= fill_price:
            return None
        if self.state.direction == 'SHORT' and stop_price <= fill_price:
            return None

        target_price = self._compute_target(fill_price, stop_price, atr,
                                            vals_for_stop)

        self.state.status = 'IN_POSITION'
        self.state.entry_price = fill_price
        self.state.entry_time = bar_time
        self.state.stop_price = stop_price
        self.state.initial_stop_price = stop_price
        self.state.target_price = target_price
        self.state.entry_bar_count = bar_count
        self.state.entry_trigger = trigger_id
        self.state.confluence_records = confluence_records
        self.state.exec_type = exec_type

        # Trade_Timestamps_Spec (revised 2026-04-20): for shipped exec
        # types (C/L/LC/CC, all Behavior B), trigger_ts == fill_ts — they
        # represent the same moment. For C-type, that moment is bar close
        # (= bar_time + bar_duration), which compute_fill_ts emits via
        # manifest.fill_offset='next_bar_open'. For L-type via this
        # bar-centric path, fill_offset='immediate' so fill_ts == bar_time.
        # Behavior A exec types (wait-then-fill) would split these apart
        # once shipped — spec supports it via the dataclass.
        import execution_types as _et
        manifest = _et.get_manifest(exec_type)
        self.state.entry_fill_ts = _et.compute_fill_ts(
            bar_time, self.tf_seconds, manifest)
        self.state.entry_trigger_ts = self.state.entry_fill_ts
        self.state.hold_duration_s = (
            manifest.hold_duration_seconds if manifest else 0)
        self.state.behavior = manifest.behavior if manifest else 'B'
        # entry_time is an internal alias that downstream services.py
        # bridges to = entry_fill_ts. No longer emitted externally.
        self.state.entry_time = self.state.entry_fill_ts or bar_time

        # Schedule confirmation via execution type module
        confirm_config = module.get_confirmation_config(
            trigger_id, exec_type, bar_count, self._get_exec_variant_param)
        if confirm_config.needs_confirm:
            self.state.pending_confirm_bar = bar_count + confirm_config.confirm_bar_offset
            self.state.bail_action = confirm_config.bail_action
            if confirm_config.hold_seconds:
                self.state.hold_seconds = confirm_config.hold_seconds

        return {
            'type': 'entry_signal',
            'trigger': trigger_id,
            'price': fill_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'bar_time': bar_time,
            'atr': atr,
            'exec_type': exec_type,
            'entry_trigger_ts': self.state.entry_trigger_ts,
            'entry_fill_ts': self.state.entry_fill_ts,
            'hold_duration_s': self.state.hold_duration_s,
            'behavior': self.state.behavior,
        }

    def check_exit(self, trigger_booleans: Dict[str, bool],
                   current_values: Dict[str, float],
                   bar_count: int, bar_time: str,
                   l_type_fills: Dict[str, float] = None,
                   suppress_bar_count: bool = False,
                   ) -> Optional[dict]:
        """Check for exit on bar close. Returns signal dict or None.

        Priority: stop > target > signal exit (C-type + L-type) > bar count.
        suppress_bar_count: If True, skip bar_count_exit (partial bar).
        """
        if self.state.status != 'IN_POSITION':
            return None

        close = current_values.get('close', 0)
        high = current_values.get('high', close)
        low = current_values.get('low', close)
        bar_open = current_values.get('open', close)
        direction = self.state.direction

        # For L-type entries on the same bar, skip OHLC-based stop/target
        # checks.  The bar's low/high includes price action from before
        # the entry, which would create false triggers.  In live, tick-
        # level checks handle intra-bar stops correctly.
        from execution_types import get_module
        _exit_module = get_module(self.state.exec_type)
        same_bar_ltype = (
            self.state.entry_bar_count == bar_count and
            _exit_module.skip_stop_target_on_entry_bar(self.state.exec_type))

        # Update trailing/breakeven stop before checking
        self._update_stop(current_values)

        # Priority 1: Stop loss (gap-aware, exec_type-dispatched)
        # Skipped on entry bar for L-type (pre-entry OHLC is unreliable)
        if self.state.stop_price and not same_bar_ltype:
            stop_fill = self._check_stop_hit(
                bar_open, high, low, close, bar_count)
            if stop_fill is not None:
                self.state.last_exit_bar_count = bar_count
                return self._exit('stop_loss', stop_fill, bar_time, bar_count)

        # Priority 2: Time exit (mechanical time-based risk control)
        # Fires before bail/target/signal — "be flat by X" overrides chart signals
        if self.time_exit_config:
            from time_exit_packs import check_time_exit
            bars_held = bar_count - self.state.entry_bar_count
            time_reason = check_time_exit(
                self.time_exit_config, bar_time, bars_held,
                self.strategy.get('trading_session', 'RTH'))
            if time_reason:
                self.state.last_exit_bar_count = bar_count
                # Clear pending LC/CC confirmations — time exit overrides
                self.state.pending_stop_confirm_bar = None
                self.state.pending_target_confirm_bar = None
                return self._exit(time_reason, close, bar_time, bar_count)

        # Priority 3/3b: Execution-type-specific bail exits (HL limit, LC/CC bail)
        bail_result = _exit_module.check_bail_exit(
            self.state.exec_type, self.state, bar_count,
            direction, high, low)
        if bail_result and bail_result.should_bail:
            self.state.last_exit_bar_count = bar_count
            return self._exit(bail_result.reason, bail_result.fill_price,
                              bar_time, bar_count)

        # Priority 4: Target (also skipped on entry bar for L-type, exec_type-dispatched)
        if self.state.target_price and not same_bar_ltype:
            target_fill = self._check_target_hit(
                bar_open, high, low, close, bar_count)
            if target_fill is not None:
                self.state.last_exit_bar_count = bar_count
                return self._exit('target', target_fill, bar_time, bar_count)

        # Priority 5: Signal exit (C-type booleans + L-type fills)
        for et in self.exit_triggers:
            base_et = _strip_exec_suffix(et)
            # L-type exit
            if l_type_fills and et in l_type_fills:
                self.state.last_exit_bar_count = bar_count
                return self._exit(et, l_type_fills[et], bar_time, bar_count)
            # C-type exit
            if trigger_booleans.get(et, False) or \
               trigger_booleans.get(base_et, False):
                self.state.last_exit_bar_count = bar_count
                return self._exit(et, close, bar_time, bar_count)

        # Priority 6: Bar count exit (legacy — suppressed on partial bars)
        if self.bar_count_exit is not None and not suppress_bar_count:
            bars_held = bar_count - self.state.entry_bar_count
            if bars_held >= self.bar_count_exit:
                self.state.last_exit_bar_count = bar_count
                return self._exit('bar_count_exit', close, bar_time, bar_count)

        return None

    def get_trade_record(self, exit_price: float, exit_time,
                         exit_reason: str, exit_trigger: str = None,
                         bar_count: int = None,
                         ) -> dict:
        """Build a trade record matching generate_trades() schema."""
        entry_price = self.state.entry_price
        direction = self.state.direction
        initial_stop = self.state.initial_stop_price

        if direction == 'LONG':
            pnl = exit_price - entry_price
        else:
            pnl = entry_price - exit_price

        risk = abs(entry_price - initial_stop) if initial_stop else abs(entry_price * 0.01)
        if risk <= 0:
            risk = entry_price * 0.01

        # Compute hold time
        bars_held = None
        hold_time_seconds = None
        if bar_count is not None and self.state.entry_bar_count is not None:
            bars_held = bar_count - self.state.entry_bar_count
        if self.state.entry_time and exit_time:
            try:
                from datetime import datetime as _dt
                entry_dt = _dt.fromisoformat(str(self.state.entry_time).replace('Z', '+00:00')) if isinstance(self.state.entry_time, str) else self.state.entry_time
                exit_dt = _dt.fromisoformat(str(exit_time).replace('Z', '+00:00')) if isinstance(exit_time, str) else exit_time
                hold_time_seconds = (exit_dt - entry_dt).total_seconds()
            except Exception:
                pass

        from stop_target_methods import get_exec_type
        _stop_et = get_exec_type(self.stop_config)
        return {
            # Trade_Timestamps_Spec: 4-timestamp model is the contract.
            # Legacy entry_time / exit_time aliases dropped per locked
            # decision #5 — readers must use *_fill_ts / *_trigger_ts.
            'entry_trigger_ts': self.state.entry_trigger_ts,
            'entry_fill_ts': self.state.entry_fill_ts,
            'exit_trigger_ts': self.state.exit_trigger_ts,
            'exit_fill_ts': self.state.exit_fill_ts,
            'hold_duration_s': self.state.hold_duration_s,
            'behavior': self.state.behavior,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'stop_price': self.state.stop_price,
            'initial_stop_price': initial_stop,
            'target_price': self.state.target_price,
            'pnl': pnl,
            'risk': risk,
            'r_multiple': pnl / risk,
            'win': pnl > 0,
            'exit_reason': exit_reason,
            'entry_trigger': self.state.entry_trigger,
            'exit_trigger': exit_trigger,
            'exec_type': self.state.exec_type or 'C',
            'stop_exec_type': _stop_et,
            'target_exec_type': get_exec_type(self.target_config),
            'confluence_records': self.state.confluence_records or set(),
            'bars_held': bars_held,
            'hold_time_seconds': hold_time_seconds,
        }

    def _reset_position(self):
        """Reset all position state fields to FLAT defaults.

        Preserves last_exit_bar_count for 1-bar cooldown enforcement.
        """
        self.state.status = 'FLAT'
        self.state.entry_price = 0.0
        self.state.entry_time = None
        self.state.stop_price = 0.0
        self.state.initial_stop_price = 0.0
        self.state.target_price = None
        self.state.confluence_records = None
        self.state.exec_type = 'C'
        self.state.pending_hm_exit = False
        self.state.pending_hl_limit = False
        self.state.pending_confirm_bar = -1
        self.state.bail_action = 'exit_market'
        self.state.hold_seconds = 0
        # Trade_Timestamps_Spec: clear 4-timestamp fields on reset.
        self.state.entry_trigger_ts = None
        self.state.entry_fill_ts = None
        self.state.exit_trigger_ts = None
        self.state.exit_fill_ts = None
        self.state.hold_duration_s = 0
        self.state.behavior = 'B'
        # last_exit_bar_count intentionally preserved for cooldown

    def _resolve_exit_exec_type(self, reason: str) -> str:
        """Determine the exec_type that governs fill_ts for this exit event.

        Trade_Timestamps_Spec (2026-04-20 bugfix): stops and targets have
        their own exec_type (from stop_config / target_config) — a C-type
        entry can have an L-type stop that fires intra-bar. Using the
        entry's exec_type for the exit's fill_ts calculation pushes the
        fill_ts into the future for those L-type exits.

        Resolution rules:
          - Intra-bar HM/HL unconfirmed exits → always L
          - stop_loss → stop_config's exec_type
          - target → target_config's exec_type
          - time_exit / bar_count / max_hold / eod / opposite → C (bar close)
          - signal_exit triggers (e.g. ema_cross_bear, and *_ib variants) →
            look up via get_trigger_exec_type
          - bail reasons from exec-type modules → entry's exec_type

        Returns 'C' | 'L' | 'LC' | 'CC' (defaults to 'C' if unknown).
        """
        from stop_target_methods import get_exec_type as _stop_target_exec
        # Intra-bar L-type safety net
        if reason in ('unconfirmed_hm', 'unconfirmed_hl'):
            return 'L'
        if reason == 'stop_loss':
            return _stop_target_exec(self.stop_config) or 'C'
        if reason == 'target':
            return _stop_target_exec(self.target_config) or 'C'
        # Bar-close evaluated time/count exits
        _BAR_CLOSE_EXITS = {
            'bar_count_exit', 'max_hold_bars', 'max_hold_seconds',
            'eod_exit', 'time_of_day_exit', 'session_exit',
            'opposite_signal',
        }
        if reason in _BAR_CLOSE_EXITS:
            return 'C'
        # Signal-exit trigger ids (e.g. 'ema_cross_bear', 'ema_cross_bear_ib')
        try:
            trig_exec = get_trigger_exec_type(reason)
            if trig_exec:
                return trig_exec
        except Exception:
            pass
        # Bail reasons from exec-type modules follow the entry's exec type
        return self.state.exec_type or 'C'

    def _exit(self, reason: str, price: float, bar_time, bar_count: int = None) -> dict:
        """Execute exit transition and return trade record (backtest path)."""
        # Trade_Timestamps_Spec (2026-04-20 bugfix): exit exec_type depends
        # on the EXIT event itself, not the entry's. A stop_loss firing
        # intra-bar is L-type regardless of the entry's C-type. Resolve
        # per-exit-reason via _resolve_exit_exec_type, then compute fill_ts
        # with that manifest.
        import execution_types as _et
        exit_exec = self._resolve_exit_exec_type(reason)
        manifest = _et.get_manifest(exit_exec)
        self.state.exit_fill_ts = _et.compute_fill_ts(
            bar_time, self.tf_seconds, manifest)
        self.state.exit_trigger_ts = self.state.exit_fill_ts
        record = self.get_trade_record(price, bar_time, reason, reason, bar_count=bar_count)
        self._reset_position()
        return record

    def _signal_exit(self, reason: str, price: float,
                     timestamp: str, bar_count: int = None) -> dict:
        """Build signal dict for live alert dispatch and reset state.

        M8.5 B+: also embeds the full `trade_record` dict (identical to what
        backtest produces via get_trade_record) so downstream alert handlers
        can persist it to the strategy's stored_trades without re-running
        the engine. This restores the Streamlit-era behavior where algo
        trades accumulated automatically as live bars closed.
        """
        if bar_count is not None:
            self.state.last_exit_bar_count = bar_count
        # Trade_Timestamps_Spec (2026-04-20 bugfix): exit exec_type depends
        # on the EXIT event, not the entry's. See _resolve_exit_exec_type.
        import execution_types as _et
        exit_exec = self._resolve_exit_exec_type(reason)
        manifest = _et.get_manifest(exit_exec)
        self.state.exit_fill_ts = _et.compute_fill_ts(
            timestamp, self.tf_seconds, manifest)
        self.state.exit_trigger_ts = self.state.exit_fill_ts
        # Build trade record BEFORE _reset_position() clears state.
        trade_record = self.get_trade_record(
            exit_price=price,
            exit_time=timestamp,
            exit_reason=reason,
            exit_trigger=reason,
            bar_count=bar_count,
        )
        sig = {
            'type': 'exit_signal',
            'trigger': reason,
            'price': price,
            'bar_time': timestamp,
            'entry_price': self.state.entry_price,
            'entry_stop_price': self.state.stop_price,
            'direction': self.state.direction,
            'exec_type': self.state.exec_type or 'C',
            'exit_reason': reason,
            'atr': 0,  # filled by caller
            # Trade_Timestamps_Spec: 4-timestamp model is the contract.
            # Legacy entry_time alias dropped per locked decision #5.
            'entry_trigger_ts': self.state.entry_trigger_ts,
            'entry_fill_ts': self.state.entry_fill_ts,
            'exit_trigger_ts': self.state.exit_trigger_ts,
            'exit_fill_ts': self.state.exit_fill_ts,
            'hold_duration_s': self.state.hold_duration_s,
            'behavior': self.state.behavior,
            # Full trade-record dict for auto-persistence into stored_trades.
            'trade_record': trade_record,
        }
        self._reset_position()
        return sig

    # ── Live-tick methods (signal dict returns) ──

    def check_entry_intrabar(self, trigger_id: str, fill_price: float,
                             current_values: Dict[str, float],
                             bar_count: int, timestamp: str,
                             confluence_records: Set[str] = None,
                             ) -> Optional[dict]:
        """Check intra-bar entry from tick-level cross. Returns signal dict."""
        if self.state.status != 'FLAT':
            return None
        if trigger_id != self.entry_trigger:
            return None

        # 1-bar cooldown: don't re-enter on the same bar we exited
        if self.state.last_exit_bar_count >= bar_count:
            return None

        # Confluence check
        if self.confluence_set and confluence_records:
            if not self.confluence_set.issubset(confluence_records):
                return None

        atr = current_values.get('atr', fill_price * 0.01)
        if not atr or atr <= 0:
            atr = fill_price * 0.01

        stop_price = self._compute_stop(fill_price, atr, current_values)

        # Stop-validity guard: reject entry if stop is on the wrong side
        if self.state.direction == 'LONG' and stop_price >= fill_price:
            return None
        if self.state.direction == 'SHORT' and stop_price <= fill_price:
            return None

        target_price = self._compute_target(fill_price, stop_price, atr,
                                            current_values)

        exec_type = get_trigger_exec_type(trigger_id)

        self.state.status = 'IN_POSITION'
        self.state.entry_price = fill_price
        self.state.entry_time = timestamp
        self.state.stop_price = stop_price
        self.state.initial_stop_price = stop_price
        self.state.target_price = target_price
        # Use bar_count + 1 so that entry_bar_count matches the bar_count
        # value that on_bar_close will receive when the entry bar closes.
        # In the live path, intra-bar entries set bar_count = builder._bar_count
        # (incremented when the *previous* bar closed).  The entry bar hasn't
        # closed yet, so its on_bar_close will see bar_count = current + 1.
        # This aligns with the backtest, where entry and exit checks happen
        # inside the same process_bar() call (entry_bar_count == bar_count).
        self.state.entry_bar_count = bar_count + 1
        self.state.entry_trigger = trigger_id
        self.state.exec_type = exec_type

        # Trade_Timestamps_Spec: trigger_ts == fill_ts for shipped exec
        # types. For intra-bar L-type entries, both happen at the cross
        # moment (fill_offset='immediate' → compute_fill_ts returns the
        # input unchanged).
        import execution_types as _et
        manifest = _et.get_manifest(exec_type)
        self.state.entry_fill_ts = _et.compute_fill_ts(
            timestamp, self.tf_seconds, manifest)
        self.state.entry_trigger_ts = self.state.entry_fill_ts
        self.state.hold_duration_s = (
            manifest.hold_duration_seconds if manifest else 0)
        self.state.behavior = manifest.behavior if manifest else 'B'
        self.state.entry_time = self.state.entry_fill_ts or timestamp

        return {
            'type': 'entry_signal',
            'trigger': trigger_id,
            'price': fill_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'bar_time': timestamp,
            'atr': atr,
            'exec_type': exec_type,
            'entry_trigger_ts': self.state.entry_trigger_ts,
            'entry_fill_ts': self.state.entry_fill_ts,
            'hold_duration_s': self.state.hold_duration_s,
            'behavior': self.state.behavior,
        }

    def check_exit_tick(self, price: float,
                        timestamp: str) -> Optional[dict]:
        """Check exit on each tick (O(1)). Handles HM/HL pending exits.

        Stops/targets only fire on tick when their exec_type is 'L' (intra-bar).
        For C/LC/CC, exit confirmation happens in check_exit_bar_close.
        """
        if self.state.status != 'IN_POSITION':
            return None

        direction = self.state.direction

        # Priority 1: Pending HM exit (unconfirmed — exit at market)
        if self.state.pending_hm_exit:
            return self._signal_exit('unconfirmed_hm', price, timestamp)

        # Priority 2: Pending HL limit (exit at entry price)
        if self.state.pending_hl_limit:
            ep = self.state.entry_price
            if direction == 'LONG' and price >= ep:
                return self._signal_exit('unconfirmed_hl', ep, timestamp)
            elif direction == 'SHORT' and price <= ep:
                return self._signal_exit('unconfirmed_hl', ep, timestamp)

        from stop_target_methods import get_exec_type

        # Priority 3: Stop loss (L-type only — C/LC/CC defer to bar close)
        if self.state.stop_price and get_exec_type(self.stop_config) == 'L':
            if direction == 'LONG' and price <= self.state.stop_price:
                return self._signal_exit(
                    'stop_loss', self.state.stop_price, timestamp)
            elif direction == 'SHORT' and price >= self.state.stop_price:
                return self._signal_exit(
                    'stop_loss', self.state.stop_price, timestamp)

        # Priority 4: Target (L-type only — C/LC/CC defer to bar close)
        if self.state.target_price and get_exec_type(self.target_config) == 'L':
            if direction == 'LONG' and price >= self.state.target_price:
                return self._signal_exit(
                    'target', self.state.target_price, timestamp)
            elif direction == 'SHORT' and price <= self.state.target_price:
                return self._signal_exit(
                    'target', self.state.target_price, timestamp)

        return None

    def check_exit_intrabar(self, trigger_id: str, fill_price: float,
                            timestamp: str) -> Optional[dict]:
        """Check signal exit from intra-bar level cross."""
        if self.state.status != 'IN_POSITION':
            return None
        base_id = _strip_exec_suffix(trigger_id)
        if base_id in self.exit_triggers or trigger_id in self.exit_triggers:
            return self._signal_exit(trigger_id, fill_price, timestamp)
        return None

    def check_exit_bar_close(self, trigger_booleans: Dict[str, bool],
                             current_values: Dict[str, float],
                             bar_count: int,
                             bar_time: str) -> Optional[dict]:
        """Check for exit on bar close (live path). Returns signal dict."""
        if self.state.status != 'IN_POSITION':
            return None

        close = current_values.get('close', 0)
        high = current_values.get('high', close)
        low = current_values.get('low', close)
        bar_open = current_values.get('open', close)
        direction = self.state.direction

        # For L-type entries on the same bar, skip OHLC-based stop/target
        # checks.  The bar's low/high includes price action from before
        # the entry, which would create false triggers.
        from execution_types import get_module as _get_live_module
        _live_mod = _get_live_module(self.state.exec_type)
        same_bar_ltype = (
            self.state.entry_bar_count == bar_count and
            _live_mod.skip_stop_target_on_entry_bar(self.state.exec_type))

        self._update_stop(current_values)

        # Priority 1: Stop loss (gap-aware, exec_type-dispatched)
        if self.state.stop_price and not same_bar_ltype:
            stop_fill = self._check_stop_hit(
                bar_open, high, low, close, bar_count)
            if stop_fill is not None:
                return self._signal_exit(
                    'stop_loss', stop_fill, bar_time, bar_count)

        # Priority 2: Time exit (mechanical time-based risk control)
        if self.time_exit_config:
            from time_exit_packs import check_time_exit
            bars_held = bar_count - self.state.entry_bar_count
            time_reason = check_time_exit(
                self.time_exit_config, bar_time, bars_held,
                self.strategy.get('trading_session', 'RTH'))
            if time_reason:
                # Clear pending LC/CC confirmations — time exit overrides
                self.state.pending_stop_confirm_bar = None
                self.state.pending_target_confirm_bar = None
                return self._signal_exit(
                    time_reason, close, bar_time, bar_count)

        # Priority 3/3b: Execution-type-specific bail exits
        bail_result = _live_mod.check_bail_exit(
            self.state.exec_type, self.state, bar_count,
            direction, high, low)
        if bail_result and bail_result.should_bail:
            return self._signal_exit(
                bail_result.reason, bail_result.fill_price, bar_time, bar_count)

        # Priority 4: Target (also skipped on entry bar for L-type, exec_type-dispatched)
        if self.state.target_price and not same_bar_ltype:
            target_fill = self._check_target_hit(
                bar_open, high, low, close, bar_count)
            if target_fill is not None:
                return self._signal_exit(
                    'target', target_fill, bar_time, bar_count)

        # Priority 5: Signal exit
        for et in self.exit_triggers:
            base_et = _strip_exec_suffix(et)
            if trigger_booleans.get(et, False) or \
               trigger_booleans.get(base_et, False):
                return self._signal_exit(et, close, bar_time, bar_count)

        # Priority 6: Bar count exit (legacy)
        if self.bar_count_exit is not None:
            bars_held = bar_count - self.state.entry_bar_count
            if bars_held >= self.bar_count_exit:
                return self._signal_exit(
                    'bar_count_exit', close, bar_time, bar_count)

        return None

    def _update_stop(self, current_values: Dict[str, float]):
        """Update stop price for trailing and/or breakeven stops."""
        if self.state.status != 'IN_POSITION':
            return

        entry_price = self.state.entry_price
        current_stop = self.state.stop_price
        direction = self.state.direction
        close = current_values.get('close', 0)

        risk = abs(entry_price - current_stop)
        if risk <= 0:
            risk = entry_price * 0.01

        new_stop = current_stop

        # Breakeven activation
        be = self.stop_config.get("breakeven")
        if be and be.get("enabled"):
            if direction == "LONG":
                unrealized_r = (close - entry_price) / risk
            else:
                unrealized_r = (entry_price - close) / risk
            if unrealized_r >= be.get("activation_r", 1.0):
                be_offset = be.get("offset", 0.0)
                if direction == "LONG":
                    be_level = entry_price + be_offset
                    new_stop = max(new_stop, be_level)
                else:
                    be_level = entry_price - be_offset
                    new_stop = min(new_stop, be_level)

        # Trailing stop
        trail = self.stop_config.get("trailing")
        if trail and trail.get("enabled"):
            if direction == "LONG":
                unrealized_r = (close - entry_price) / risk
            else:
                unrealized_r = (entry_price - close) / risk

            if unrealized_r >= trail.get("activation_r", 0.0):
                trail_method = trail.get("method", "atr")
                atr = current_values.get('atr', entry_price * 0.01)
                if not atr or atr <= 0:
                    atr = entry_price * 0.01

                if trail_method == "atr":
                    trail_distance = atr * trail.get("atr_mult", 1.0)
                elif trail_method == "fixed_dollar":
                    trail_distance = trail.get("dollar_amount", 1.0)
                elif trail_method == "percentage":
                    trail_distance = entry_price * (trail.get("percentage", 0.5) / 100.0)
                else:
                    trail_distance = None

                if trail_distance is not None:
                    if direction == "LONG":
                        trail_level = close - trail_distance
                        new_stop = max(new_stop, trail_level)
                    else:
                        trail_level = close + trail_distance
                        new_stop = min(new_stop, trail_level)

        self.state.stop_price = new_stop

    # ── Stop/target hit detection (exec_type aware) ──

    def _check_stop_hit(self, bar_open: float, high: float, low: float,
                        close: float, bar_count: int) -> Optional[float]:
        """Check if the stop was hit on this bar based on the stop's exec_type.

        Returns the fill price if hit, None otherwise.

        - L: intra-bar touch fires (current behavior, default).
             Fill is gap-aware: min(stop, open) for LONG, max(stop, open) for SHORT.
        - C: only fires when bar closes past the stop (filters wicks). Fill = close.
        - LC: touch intra-bar AND bar close confirms past. Fill = close.
        - CC: bar close past + next bar close also past (state-tracked). Fill = close.
        """
        if not self.state.stop_price:
            return None
        from stop_target_methods import get_exec_type
        exec_type = get_exec_type(self.stop_config)
        direction = self.state.direction
        stop_price = self.state.stop_price

        touched = (direction == 'LONG' and low <= stop_price) or \
                  (direction == 'SHORT' and high >= stop_price)
        closed_past = (direction == 'LONG' and close <= stop_price) or \
                      (direction == 'SHORT' and close >= stop_price)

        if exec_type == 'L':
            if touched:
                # Gap-aware fill (preserves legacy behavior)
                if direction == 'LONG':
                    return min(stop_price, bar_open)
                return max(stop_price, bar_open)
            return None
        if exec_type == 'C':
            return close if closed_past else None
        if exec_type == 'LC':
            return close if (touched and closed_past) else None
        if exec_type == 'CC':
            # Two-bar close confirmation tracked via state attribute
            pending_bar = getattr(self.state, 'pending_stop_confirm_bar', None)
            if pending_bar is None:
                # No pending — first close past arms confirmation
                if closed_past:
                    self.state.pending_stop_confirm_bar = bar_count
                return None
            if bar_count == pending_bar:
                # Same bar (re-check) — already armed, do nothing
                return None
            if bar_count == pending_bar + 1:
                if closed_past:
                    # Two consecutive bars closed past — confirm
                    self.state.pending_stop_confirm_bar = None
                    return close
                # Confirmation failed — reset
                self.state.pending_stop_confirm_bar = None
                return None
            # Stale pending state — reset, optionally re-arm
            self.state.pending_stop_confirm_bar = bar_count if closed_past else None
            return None
        # Unknown exec_type → fall back to L-type
        if touched:
            if direction == 'LONG':
                return min(stop_price, bar_open)
            return max(stop_price, bar_open)
        return None

    def _check_target_hit(self, bar_open: float, high: float, low: float,
                          close: float, bar_count: int) -> Optional[float]:
        """Check if the target was hit on this bar based on the target's exec_type.
        Mirror of _check_stop_hit but for take profit (direction inverted)."""
        if not self.state.target_price:
            return None
        from stop_target_methods import get_exec_type
        exec_type = get_exec_type(self.target_config)
        direction = self.state.direction
        target_price = self.state.target_price

        touched = (direction == 'LONG' and high >= target_price) or \
                  (direction == 'SHORT' and low <= target_price)
        closed_past = (direction == 'LONG' and close >= target_price) or \
                      (direction == 'SHORT' and close <= target_price)

        if exec_type == 'L':
            if touched:
                # Gap-aware fill (preserves legacy behavior)
                if direction == 'LONG':
                    return max(target_price, bar_open)
                return min(target_price, bar_open)
            return None
        if exec_type == 'C':
            return close if closed_past else None
        if exec_type == 'LC':
            return close if (touched and closed_past) else None
        if exec_type == 'CC':
            pending_bar = getattr(self.state, 'pending_target_confirm_bar', None)
            if pending_bar is None:
                if closed_past:
                    self.state.pending_target_confirm_bar = bar_count
                return None
            if bar_count == pending_bar:
                return None
            if bar_count == pending_bar + 1:
                if closed_past:
                    self.state.pending_target_confirm_bar = None
                    return close
                self.state.pending_target_confirm_bar = None
                return None
            self.state.pending_target_confirm_bar = bar_count if closed_past else None
            return None
        # Unknown exec_type → fall back to L-type
        if touched:
            if direction == 'LONG':
                return max(target_price, bar_open)
            return min(target_price, bar_open)
        return None

    def _compute_stop(self, entry_price: float, atr: float,
                      vals: Dict[str, float]) -> float:
        """Compute the initial stop price via the pluggable method registry.

        Replaces hardcoded if/elif branches. Add a new stop method by
        registering a class in stop_target_methods.STOP_METHODS — no engine
        changes needed.
        """
        from stop_target_methods import StopContext, compute_stop
        ctx = StopContext(
            entry_price=entry_price,
            direction=self.state.direction,
            atr=atr,
            high_low_buffer=self._high_low_buffer,
            config=self.stop_config or {},
        )
        return compute_stop(ctx)

    def _compute_target(self, entry_price: float, stop_price: float,
                        atr: float, vals: Dict[str, float]) -> Optional[float]:
        """Compute the take profit target price via the pluggable method registry.

        Returns None if no target_config or method is unknown. Replaces
        hardcoded if/elif branches.
        """
        if self.target_config is None:
            return None
        from stop_target_methods import TargetContext, compute_target
        ctx = TargetContext(
            entry_price=entry_price,
            stop_price=stop_price,
            direction=self.state.direction,
            atr=atr,
            high_low_buffer=self._high_low_buffer,
            config=self.target_config,
        )
        return compute_target(ctx)


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED STRATEGY — single-strategy coordinator
# ═══════════════════════════════════════════════════════════════════════════

class UnifiedStrategy:
    """Single-strategy coordinator for the unified engine.

    Wraps IncrementalIndicatorEngine + TriggerEvaluator + PositionStateMachine.
    Processes one bar at a time, returns trade records.
    """

    def __init__(self, strategy: dict, general_packs=None):
        self.strategy = strategy
        self.direction = strategy.get('direction', 'LONG')
        self.tf_label = self._get_tf_label(strategy.get('timeframe', '1Min'))

        # Resolve requirements
        req_ind, req_interp, req_trig, params = resolve_strategy_requirements(
            strategy)

        # Resolve entry/exit triggers
        self.entry_trigger = _resolve_trigger_id(strategy, 'entry_trigger')
        self.exit_triggers = _resolve_trigger_ids(strategy, 'exit_triggers')
        if not self.exit_triggers:
            single = _resolve_trigger_id(strategy, 'exit_trigger')
            if single:
                self.exit_triggers = [single]

        # Create components
        self.indicators = IncrementalIndicatorEngine(req_ind, params)
        self.trigger_eval = TriggerEvaluator(
            req_interp, req_trig,
            ema_periods=params['ema_periods'])
        self.position = PositionStateMachine(
            strategy,
            resolved_entry=self.entry_trigger,
            resolved_exits=self.exit_triggers)

        self.general_packs = general_packs or []
        self._bar_count = 0

        # Interpreter keys needed for confluence records
        self._interpreter_keys = list(req_interp)

    def process_bar(self, bar: dict, mtf_records: Set[str] = None,
                    partial: bool = False,
                    user_pack_data: dict = None,
                    ) -> Tuple[List[dict], Dict[str, float],
                               Dict[str, str], Dict[str, bool]]:
        """Process one completed bar.

        Args:
            bar: {'open', 'high', 'low', 'close', 'volume', 'timestamp'}
            mtf_records: Optional set of multi-timeframe confluence records
                from pre-computed columns (e.g. '5m-EMA_STACK-FULL_BULL_STACK').
            partial: If True, this bar is still forming (live chart).
                Only L-type (intra-bar) signals are evaluated; bar-close
                entries/exits (C-type, bar_count_exit) are suppressed.
            user_pack_data: Optional dict with pre-computed user pack data:
                {'interps': {key: state}, 'triggers': {key: bool}}
                Merged into interpreter/trigger results after built-in evaluation.

        Returns:
            (trade_records, indicator_values, interpreter_states, trigger_bools)
            trade_records: List of 0-2 trade dicts (exit then entry)
            indicator_values: Current indicator values for enriched DataFrame
            interpreter_states: Current interpreter states
            trigger_bools: C-type trigger booleans for this bar
        """
        self._bar_count += 1
        trades = []

        # 1. Update indicators (O(1))
        current = self.indicators.update_bar(bar)
        prev = self.indicators.get_prev_values()

        # 1a. Merge user pack indicator values into current so the engine can
        # cache levels and check intra-bar crosses for user pack L-type triggers.
        if user_pack_data and user_pack_data.get('indicators'):
            current.update(user_pack_data['indicators'])

        # 1b. Feed high/low into position buffer (for swing stops/targets)
        self.position.update_high_low(bar['high'], bar['low'])

        # 2. Evaluate triggers (C-type + L-type)
        interps, c_triggers, l_fills = self.trigger_eval.evaluate_bar_for_backtest(
            current, prev, self.indicators.state.prev2_macd_hist)

        # 2b. Merge pre-computed user pack interpreter states and trigger booleans.
        # These come from the batch pipeline (DataFrame columns) and are passed
        # through for user packs whose indicators aren't computed incrementally.
        if user_pack_data:
            for ik, state_val in user_pack_data.get('interps', {}).items():
                if ik not in interps:  # Don't override built-in
                    interps[ik] = state_val
            for tk, fired in user_pack_data.get('triggers', {}).items():
                if tk not in c_triggers:  # Don't override built-in
                    c_triggers[tk] = fired

        # 3. Build confluence records
        bar_time = bar['timestamp']
        confluence_records = set()
        for interp_key, state_val in interps.items():
            confluence_records.add(
                f"{self.tf_label}-{interp_key}-{state_val}")

        # MTF confluence records (from pre-computed columns)
        if mtf_records:
            confluence_records |= mtf_records

        # General packs
        if self.general_packs:
            ts = bar_time
            if hasattr(ts, 'to_pydatetime'):
                ts = ts.to_pydatetime()
            elif not isinstance(ts, datetime):
                ts = pd.Timestamp(ts).to_pydatetime()
            gp_records = _evaluate_general_packs(self.general_packs, ts)
            confluence_records.update(gp_records)

        # On partial (still-forming) bars, only L-type (intra-bar) signals
        # fire.  C-type signals (bar-close triggers, bar_count_exit) are
        # suppressed because the close price is not final.  Stops and
        # targets still fire since they are intra-bar events.
        c_trigs_for_eval = {} if partial else c_triggers

        # 4. Handle pending bail exits (unconfirmed on previous bar)
        # Delegated to execution type module
        if not partial and self.position.state.status == 'IN_POSITION':
            ps = self.position.state
            from execution_types import get_module as _get_exec_module
            _bail_module = _get_exec_module(ps.exec_type)
            bail_result = _bail_module.check_bail_at_market(
                ps.exec_type, ps, self._bar_count, bar['open'])
            if bail_result and bail_result.should_bail:
                exit_record = self.position._exit(
                    bail_result.reason, bail_result.fill_price, bar_time)
                trades.append(exit_record)

        # 5. Check exit (includes HL limit check, stop, target, signal, bar count)
        if self.position.state.status == 'IN_POSITION':
            exit_record = self.position.check_exit(
                c_trigs_for_eval, current, self._bar_count,
                bar_time, l_type_fills=l_fills,
                suppress_bar_count=partial)
            if exit_record:
                trades.append(exit_record)

        # 6. Check entry (only if FLAT after exit check)
        # Entry signals are internal state changes, not trade records.
        # The trade record is produced on exit.
        self.position.check_entry(
            c_trigs_for_eval, current, self._bar_count,
            bar_time, confluence_records=confluence_records,
            l_type_fills=l_fills,
            prev_values=prev)

        # 7. Confirmation check — delegated to execution type module
        ps = self.position.state
        if ps.status == 'IN_POSITION':
            from execution_types import get_module as _get_confirm_module
            _confirm_mod = _get_confirm_module(ps.exec_type)
            if _confirm_mod.should_check_confirmation(
                    ps.exec_type, ps.entry_bar_count, self._bar_count,
                    ps.pending_confirm_bar):
                base_trigger = _strip_exec_suffix(ps.entry_trigger)
                if not self.trigger_eval.check_confirmation(base_trigger, current):
                    _confirm_mod.on_confirmation_failed(ps.exec_type, ps)
                else:
                    _confirm_mod.on_confirmation_passed(ps.exec_type, ps)

        return trades, current, interps, c_triggers

    @staticmethod
    def _get_tf_label(timeframe: str) -> str:
        """Convert timeframe string to label format."""
        tf_map = {
            '1Min': '1M', '2Min': '2M', '3Min': '3M', '5Min': '5M',
            '10Min': '10M', '15Min': '15M', '30Min': '30M',
            '1Hour': '1H', '2Hour': '2H', '4Hour': '4H',
            '1Day': '1D', '1Week': '1W',
        }
        return tf_map.get(timeframe, '1M')


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API — run_unified_backtest
# ═══════════════════════════════════════════════════════════════════════════

def run_unified_backtest(
    df: pd.DataFrame,
    strategy: dict,
    general_packs: list = None,
    secondary_tf_map: dict = None,
    include_open_position: bool = False,
    last_bar_partial: bool = False,
    progress_cb=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run unified backtest on historical OHLCV data.

    Args:
        df: DataFrame (timestamp-indexed). Must contain OHLCV columns.
            May also contain pre-computed MTF columns (e.g. EMA_STACK__5m).
        strategy: Strategy config dict (same format as strategies.json)
        general_packs: Optional list of GeneralPack objects
        secondary_tf_map: Optional {tf_label: [suffixed_col_names]} for MTF
            confluence.  Column names have the form ``{INTERP}__{tf_label}``.
        include_open_position: If True, append a synthetic row for any
            open position at the end of the data (exit_time/exit_price=None).
            Useful for chart rendering to show entry markers immediately.
        last_bar_partial: If True, treat the last bar as still forming.
            Indicators are updated but entry/exit signals are suppressed
            on that bar, preventing premature markers on live charts.

    Returns:
        (trades_df, enriched_df)
        trades_df: DataFrame matching generate_trades() output format
        enriched_df: DataFrame with indicator + interpreter + trigger columns
    """
    if len(df) < 2:
        return pd.DataFrame(), df.copy()

    strat = UnifiedStrategy(strategy, general_packs)

    # Identify user pack columns in the DataFrame for pre-computed fallback.
    # Built-in interpreters are computed incrementally by the engine; user pack
    # interpreters and triggers exist as pre-computed columns in the DataFrame.
    _BUILTIN_INTERPS = {
        'EMA_STACK', 'EMA_PRICE_POSITION', 'EMA_PRICE_POSITION_V2',
        'MACD_LINE', 'MACD_HISTOGRAM', 'VWAP', 'RVOL', 'UTBOT', 'UTBOT_V2',
    }
    _user_interp_cols = [
        ik for ik in strat.trigger_eval.required_interpreters
        if ik not in _BUILTIN_INTERPS and ik in df.columns
    ]
    _user_trig_cols = [
        col for col in df.columns
        if col.startswith('trig_') and col not in {
            f'trig_{t}' for t in strat.trigger_eval.required_triggers
            if any(t.startswith(p + '_') for p in TRIGGER_PREFIX_TO_TEMPLATE)
        }
        and any(col == f'trig_{t}' for t in strat.trigger_eval.required_triggers)
    ]
    # Simpler approach: collect all trig_ columns that match required triggers
    _required_trig_set = {f'trig_{t}' for t in strat.trigger_eval.required_triggers}
    _user_trig_cols = [col for col in df.columns if col in _required_trig_set and col.startswith('trig_')]

    # Collect user pack indicator columns referenced by L-type triggers in INTRABAR_LEVEL_MAP.
    # These need to be passed into `current` so the engine can cache levels and check crosses.
    _user_indicator_cols = set()
    for base_trigger in strat.trigger_eval.required_triggers:
        # Strip _ib/_lc/_cc/_hm/_hl suffix to find the base in INTRABAR_LEVEL_MAP
        for suffix in ('_ib', '_lc', '_cc', '_hm', '_hl'):
            if base_trigger.endswith(suffix):
                base = base_trigger[:-len(suffix)]
                if base in INTRABAR_LEVEL_MAP:
                    level_col = INTRABAR_LEVEL_MAP[base].get('column', '')
                    if level_col and level_col in df.columns:
                        _user_indicator_cols.add(level_col)
                break

    trades = []
    indicator_rows = []
    interp_rows = []
    trigger_rows = []

    # Progress reporting: emit every 1000 bars OR every 250ms, whichever first.
    # No-op when progress_cb is None (typical chart/detail view callers).
    # Exceptions from progress_cb propagate — Mass Builder uses this path to
    # signal cancellation via a custom exception, which must bubble up and
    # stop the bar loop.
    _total_bars = len(df)
    _last_cb_time = 0.0
    _last_cb_bar = 0
    if progress_cb is not None:
        import time as _time
        _last_cb_time = _time.monotonic()
        progress_cb(0, _total_bars)

    for i in range(len(df)):
        if progress_cb is not None and i > 0:
            _dbar = i - _last_cb_bar
            _dt = _time.monotonic() - _last_cb_time
            if _dbar >= 1000 or _dt >= 0.25:
                progress_cb(i, _total_bars)
                _last_cb_time = _time.monotonic()
                _last_cb_bar = i
        row = df.iloc[i]
        bar = {
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row.get('volume', 0)),
            'timestamp': df.index[i],
        }

        # Build MTF confluence records from pre-computed columns
        mtf_records = None
        if secondary_tf_map:
            mtf_set = set()
            for tf_label, cols in secondary_tf_map.items():
                for col in cols:
                    val = row.get(col)
                    if val is not None and pd.notna(val):
                        base_interp = col.rsplit('__', 1)[0]
                        mtf_set.add(f"{tf_label}-{base_interp}-{val}")
            if mtf_set:
                mtf_records = mtf_set

        # Read pre-computed user pack interpreter states and trigger booleans
        user_pack_data = None
        if _user_interp_cols or _user_trig_cols or _user_indicator_cols:
            up_interps = {}
            up_triggers = {}
            up_indicators = {}
            for col in _user_interp_cols:
                val = row.get(col)
                if val is not None and pd.notna(val):
                    up_interps[col] = str(val)
            for col in _user_trig_cols:
                val = row.get(col)
                # Strip 'trig_' prefix to get the trigger key
                trig_key = col[5:]  # len('trig_') == 5
                up_triggers[trig_key] = bool(val) if pd.notna(val) else False
            for col in _user_indicator_cols:
                val = row.get(col)
                if val is not None and pd.notna(val):
                    up_indicators[col] = float(val)
            if up_interps or up_triggers or up_indicators:
                user_pack_data = {'interps': up_interps, 'triggers': up_triggers, 'indicators': up_indicators}

        is_partial = last_bar_partial and (i == len(df) - 1)
        bar_trades, ind_vals, interp_states, trig_bools = strat.process_bar(
            bar, mtf_records=mtf_records, partial=is_partial,
            user_pack_data=user_pack_data)
        trades.extend(bar_trades)
        indicator_rows.append(ind_vals)
        interp_rows.append(interp_states)
        trigger_rows.append(trig_bools)

    # If position is still open, append a synthetic open-trade row so charts
    # can plot the entry marker immediately (before the trade closes).
    if include_open_position and strat.position.state.status == 'IN_POSITION':
        pos = strat.position.state
        last_close = float(df.iloc[-1]['close'])
        direction = pos.direction
        entry_price = pos.entry_price
        initial_stop = pos.initial_stop_price
        risk = abs(entry_price - initial_stop) if initial_stop else abs(entry_price * 0.01)
        if risk <= 0:
            risk = entry_price * 0.01
        unrealized_pnl = (last_close - entry_price) if direction == 'LONG' else (entry_price - last_close)
        trades.append({
            # Trade_Timestamps_Spec: 4-field model on open trades (exit fields null).
            'entry_trigger_ts': pos.entry_trigger_ts,
            'entry_fill_ts': pos.entry_fill_ts,
            'exit_trigger_ts': None,
            'exit_fill_ts': None,
            'hold_duration_s': pos.hold_duration_s,
            'behavior': pos.behavior,
            'entry_price': entry_price,
            'exit_price': None,
            'stop_price': pos.stop_price,
            'initial_stop_price': initial_stop,
            'target_price': pos.target_price,
            'pnl': unrealized_pnl,
            'risk': risk,
            'r_multiple': unrealized_pnl / risk,
            'win': unrealized_pnl > 0,
            'exit_reason': 'open',
            'entry_trigger': pos.entry_trigger,
            'exec_type': pos.exec_type,
            'confluence_records': pos.confluence_records or set(),
        })

    # Build trades DataFrame
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    # Build enriched DataFrame
    enriched_df = df.copy()

    # Add indicator columns
    if indicator_rows:
        ind_df = pd.DataFrame(indicator_rows, index=df.index)
        # Only add columns not already in df (avoid overwriting OHLCV)
        for col in ind_df.columns:
            if col not in ('open', 'high', 'low', 'close', 'volume'):
                enriched_df[col] = ind_df[col]

    # Add interpreter state columns
    if interp_rows:
        interp_df = pd.DataFrame(interp_rows, index=df.index)
        for col in interp_df.columns:
            enriched_df[col] = interp_df[col]

    # Add trigger boolean columns (prefixed with trig_)
    if trigger_rows:
        trig_df = pd.DataFrame(trigger_rows, index=df.index)
        for col in trig_df.columns:
            enriched_df[f'trig_{col}'] = trig_df[col]

    return trades_df, enriched_df


# ═══════════════════════════════════════════════════════════════════════════
# FAST PATH — precompute + replay for Strategy Builder analyzers
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CachedBarState:
    """Pre-computed per-bar state for fast trade replay."""
    bar: dict           # {open, high, low, close, volume, timestamp}
    bar_count: int
    current_values: dict
    prev_values: dict
    c_triggers: dict    # {trigger_id: bool}
    l_fills: dict       # {trigger_id: fill_price}
    confluence_records: set


def precompute_bar_cache(
    df: pd.DataFrame,
    strategy: dict,
    general_packs: list = None,
    secondary_tf_map: dict = None,
) -> Tuple[List[CachedBarState], dict]:
    """Run indicator/trigger pipeline once and cache per-bar state.

    Returns (cache, metadata).  The cache stores all intermediate values
    needed to replay trade generation with different strategy configs
    (entry trigger, exit triggers, stop, target).  Indicators and triggers
    are NOT recomputed during replay — only PositionStateMachine logic runs.

    Typical speedup: 10-50x for analyzer functions that test many configs.
    """
    if len(df) < 2:
        return [], {}

    strat = UnifiedStrategy(strategy, general_packs)
    cache: List[CachedBarState] = []

    for i in range(len(df)):
        row = df.iloc[i]
        bar = {
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row.get('volume', 0)),
            'timestamp': df.index[i],
        }

        strat._bar_count += 1

        # 1. Update indicators (O(1))
        current = strat.indicators.update_bar(bar)
        prev = strat.indicators.get_prev_values()

        # 2. Evaluate triggers
        interps, c_triggers, l_fills = strat.trigger_eval.evaluate_bar_for_backtest(
            current, prev, strat.indicators.state.prev2_macd_hist)

        # 3. Build confluence records
        confluence_records = set()
        for interp_key, state_val in interps.items():
            confluence_records.add(
                f"{strat.tf_label}-{interp_key}-{state_val}")

        # MTF confluence records
        if secondary_tf_map:
            for tf_label, cols in secondary_tf_map.items():
                for col in cols:
                    val = row.get(col)
                    if val is not None and pd.notna(val):
                        base_interp = col.rsplit('__', 1)[0]
                        confluence_records.add(
                            f"{tf_label}-{base_interp}-{val}")

        # General packs
        if strat.general_packs:
            ts = bar['timestamp']
            if hasattr(ts, 'to_pydatetime'):
                ts = ts.to_pydatetime()
            elif not isinstance(ts, datetime):
                ts = pd.Timestamp(ts).to_pydatetime()
            gp_records = _evaluate_general_packs(strat.general_packs, ts)
            confluence_records.update(gp_records)

        cache.append(CachedBarState(
            bar=bar,
            bar_count=strat._bar_count,
            current_values=current,
            prev_values=prev,
            c_triggers=c_triggers,
            l_fills=l_fills,
            confluence_records=confluence_records,
        ))

    metadata = {
        'ema_periods': list(strat.trigger_eval.ema_periods),
        'available_triggers': set(strat.trigger_eval.required_triggers),
    }
    return cache, metadata


def run_trades_from_cache(
    cache: List[CachedBarState],
    strategy: dict,
    metadata: dict,
    include_open_position: bool = False,
) -> pd.DataFrame:
    """Generate trades by replaying only PositionStateMachine logic.

    Uses pre-computed indicator values and trigger booleans from the cache.
    No indicator or trigger computation — ~10-50x faster than full backtest.
    """
    if not cache:
        return pd.DataFrame()

    # Resolve entry/exit triggers
    entry_trigger = _resolve_trigger_id(strategy, 'entry_trigger')
    exit_triggers = _resolve_trigger_ids(strategy, 'exit_triggers')
    if not exit_triggers:
        single = _resolve_trigger_id(strategy, 'exit_trigger')
        if single:
            exit_triggers = [single]

    # Check if the requested triggers exist in the cache.  The cache only
    # contains trigger booleans for the pack it was built with.  If the
    # analyzer is testing a trigger from a different pack, we must raise so
    # the caller falls back to a full backtest.
    available = metadata.get('available_triggers', set())
    if available:
        # Strip execution-type suffixes (e.g. _hm, _hl, _l0, _l1) for base match
        _base_entry = _strip_exec_suffix(entry_trigger) if entry_trigger else ''
        _base_avail = {_strip_exec_suffix(t) for t in available}
        if _base_entry and _base_entry not in _base_avail:
            raise ValueError(
                f"Trigger '{entry_trigger}' not in cache "
                f"(available: {available})")

    psm = PositionStateMachine(
        strategy,
        resolved_entry=entry_trigger,
        resolved_exits=exit_triggers)

    # Lightweight trigger evaluator for HM/HL confirmation only
    trigger_eval = TriggerEvaluator(
        set(), set(),
        ema_periods=metadata['ema_periods'])

    trades = []

    for cb in cache:
        psm.update_high_low(cb.bar['high'], cb.bar['low'])

        # Handle pending bail exits — delegated to execution type module
        if psm.state.status == 'IN_POSITION':
            ps = psm.state
            from execution_types import get_module as _get_cache_module
            _cache_mod = _get_cache_module(ps.exec_type)
            bail_result = _cache_mod.check_bail_at_market(
                ps.exec_type, ps, cb.bar_count, cb.bar['open'])
            if bail_result and bail_result.should_bail:
                exit_record = psm._exit(
                    bail_result.reason, bail_result.fill_price, cb.bar['timestamp'])
                trades.append(exit_record)

        # Check exit
        if psm.state.status == 'IN_POSITION':
            exit_record = psm.check_exit(
                cb.c_triggers, cb.current_values, cb.bar_count,
                cb.bar['timestamp'], l_type_fills=cb.l_fills)
            if exit_record:
                trades.append(exit_record)

        # Check entry (only if FLAT)
        psm.check_entry(
            cb.c_triggers, cb.current_values, cb.bar_count,
            cb.bar['timestamp'], confluence_records=cb.confluence_records,
            l_type_fills=cb.l_fills,
            prev_values=cb.prev_values)

        # Confirmation check — delegated to execution type module
        if psm.state.status == 'IN_POSITION':
            ps = psm.state
            from execution_types import get_module as _get_cache_confirm
            _cache_confirm_mod = _get_cache_confirm(ps.exec_type)
            if _cache_confirm_mod.should_check_confirmation(
                    ps.exec_type, ps.entry_bar_count, cb.bar_count,
                    ps.pending_confirm_bar):
                base_trigger = _strip_exec_suffix(ps.entry_trigger)
                if not trigger_eval.check_confirmation(
                        base_trigger, cb.current_values):
                    _cache_confirm_mod.on_confirmation_failed(ps.exec_type, ps)
                else:
                    _cache_confirm_mod.on_confirmation_passed(ps.exec_type, ps)

    # Open position synthetic row
    if include_open_position and psm.state.status == 'IN_POSITION':
        pos = psm.state
        last_close = cache[-1].bar['close']
        entry_price = pos.entry_price
        initial_stop = pos.initial_stop_price
        risk = abs(entry_price - initial_stop) if initial_stop else abs(
            entry_price * 0.01)
        if risk <= 0:
            risk = entry_price * 0.01
        unrealized = ((last_close - entry_price) if pos.direction == 'LONG'
                      else (entry_price - last_close))
        trades.append({
            # Trade_Timestamps_Spec: 4-field model on open trades (exit fields null).
            'entry_trigger_ts': pos.entry_trigger_ts,
            'entry_fill_ts': pos.entry_fill_ts,
            'exit_trigger_ts': None,
            'exit_fill_ts': None,
            'hold_duration_s': pos.hold_duration_s,
            'behavior': pos.behavior,
            'entry_price': entry_price,
            'exit_price': None,
            'stop_price': pos.stop_price,
            'initial_stop_price': initial_stop,
            'target_price': pos.target_price,
            'pnl': unrealized,
            'risk': risk,
            'r_multiple': unrealized / risk,
            'win': unrealized > 0,
            'exit_reason': 'open',
            'entry_trigger': pos.entry_trigger,
            'exec_type': pos.exec_type,
            'confluence_records': pos.confluence_records or set(),
        })

    return pd.DataFrame(trades) if trades else pd.DataFrame()
