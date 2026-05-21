"""Pack Live Test — canary strategy creator + status.

Spec: docs/Spec_Pack_Live_Test.md

Maintains up to two permanent "canary" strategies per user pack so
the User Packs page can confirm a pack actually fires in the REAL
deployed live worker — the gap that let SWING_123_TEST sit broken as
a cross-TF gate for 8+ days (Signal Validation only checks the
backtest path; Parity Simulator runs a simulated replay; neither
exercises the live worker's secondary-TF shadow dispatch).

  - trigger canary: the pack used as the entry trigger
  - gate canary:    the pack used as a 2Min cross-TF confluence gate,
                    with a known-good borrowed entry trigger

Canaries are created through the canonical create path
(`api.routers.strategies.create_strategy`) so they're indistinguishable
from a user-built strategy — a canary built via a bespoke insert
couldn't be trusted as a diagnostic. They are tagged `pack-canary`,
NOT auto-deleted (always-on sentinels), and creation is idempotent
(2 per pack, max).
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Fixed canary parameters (Spec §3.2) — keep canaries comparable ──
CANARY_TAG = 'pack-canary'
CANARY_SYMBOL = 'SPY'
CANARY_PRIMARY_TF = '10Sec'
CANARY_DIRECTION = 'LONG'
CANARY_SESSION = 'Extended Hours'
CANARY_GATE_TF_PREFIX = '2m'          # confluence-record TF prefix
CANARY_ORIGIN = 'standard'
# Gate canary borrows a known-good, high-frequency entry trigger so the
# pack-under-test is the only variable. ut_bot_v4 bull flip by default;
# when the pack under test IS ut_bot_v4, fall back to a different pack.
_BORROWED_TRIGGER = 'ut_bot_v4_default_bull_flip'
_BORROWED_EXIT = 'ut_bot_v4_default_bear_flip'
_BORROWED_TRIGGER_ALT = 'ema_pp_v3_default_cross_short_up'
_BORROWED_EXIT_ALT = 'ema_pp_v3_default_cross_mid_down'

_BULL_HINTS = ('bull', 'long', 'up', 'above', 'cross_up')
_BEAR_HINTS = ('bear', 'short', 'down', 'below', 'cross_down')


def _canary_name(pack_name: str, role: str) -> str:
    """role: 'trigger' | 'gate'."""
    return f'PACKTEST · {pack_name} · {role}'


def _pick_trigger_base(manifest: dict, bullish: bool) -> Optional[str]:
    """Pick a trigger `base` from the pack manifest, preferring the
    bullish (entry) or bearish (exit) direction. Falls back to the
    first/last trigger when no directional hint matches."""
    trigs = manifest.get('triggers') or []
    if not trigs:
        return None
    hints = _BULL_HINTS if bullish else _BEAR_HINTS
    for t in trigs:
        base = str(t.get('base') or '').lower()
        if any(h in base for h in hints):
            return t.get('base')
    # No directional match — first trigger for entry, a different one
    # (or the same) for exit.
    if bullish:
        return trigs[0].get('base')
    return trigs[-1].get('base')


def _pick_gate_state(manifest: dict) -> Optional[str]:
    """Most-permissive interpreter output for the gate canary — prefer
    NEUTRAL (holds most of the time → the strategy still fires often,
    so a silent canary clearly indicts the gate, not rarity)."""
    outs = manifest.get('outputs') or []
    if not outs:
        return None
    if 'NEUTRAL' in outs:
        return 'NEUTRAL'
    return outs[0]


def _base_canary_config(name: str, user_id: str) -> dict:
    """Shared canary strategy config — modelled on a real 10Sec
    strategy (verified against sid 170). Role-specific fields
    (triggers, confluence) are layered on by the caller."""
    return {
        'name': name,
        'user_id': user_id,
        'symbol': CANARY_SYMBOL,
        'timeframe': CANARY_PRIMARY_TF,
        'direction': CANARY_DIRECTION,
        'strategy_origin': CANARY_ORIGIN,
        'trading_session': CANARY_SESSION,
        'tags': [CANARY_TAG],
        # Monitored so the live worker actually watches it — the whole
        # point of a canary.
        'alert_tracking_enabled': True,
        # Self-contained ATR stop — no dependency on a stop pack
        # existing. Profitability is irrelevant; this just lets trades
        # close so the canary cycles.
        'stop_config': {'method': 'atr', 'atr_mult': 1.5, 'exec_type': 'L'},
        'target_config': None,
        'risk_per_trade': 100.0,
        'starting_balance': 10000.0,
        'data_days': 2,
        'data_seed': 42,
        'lookback_mode': 'Days',
    }


def _build_trigger_canary(pack, user_id: str) -> Optional[dict]:
    """Pack as the ENTRY TRIGGER, no confluence gate. None if the
    pack declares no triggers (gate-only packs)."""
    m = pack.manifest or {}
    slug = m.get('slug')
    bull = _pick_trigger_base(m, bullish=True)
    if not bull:
        return None
    bear = _pick_trigger_base(m, bullish=False)
    cfg = _base_canary_config(
        _canary_name(m.get('name', slug), 'trigger'), user_id)
    cfg['entry_trigger_confluence_id'] = f'{slug}_default_{bull}'
    # Exit on the pack's bearish trigger when it has one, else
    # opposite-signal so the canary still cycles.
    cfg['exit_trigger_confluence_ids'] = (
        [f'{slug}_default_{bear}'] if bear and bear != bull
        else ['opposite_signal'])
    cfg['confluence'] = []
    return cfg


def _build_gate_canary(pack, user_id: str) -> Optional[dict]:
    """Pack as a 2Min cross-TF CONFLUENCE GATE, with a borrowed
    known-good entry trigger. None if the pack declares no interpreter
    outputs (trigger-only packs)."""
    m = pack.manifest or {}
    slug = m.get('slug')
    interps = m.get('interpreters') or []
    state = _pick_gate_state(m)
    if not interps or not state:
        return None
    interp = interps[0]
    # Borrow a high-frequency trigger from a DIFFERENT pack so the
    # gate is the only variable.
    if slug == 'ut_bot_v4':
        entry, exit_ = _BORROWED_TRIGGER_ALT, _BORROWED_EXIT_ALT
    else:
        entry, exit_ = _BORROWED_TRIGGER, _BORROWED_EXIT
    cfg = _base_canary_config(
        _canary_name(m.get('name', slug), 'gate'), user_id)
    cfg['entry_trigger_confluence_id'] = entry
    cfg['exit_trigger_confluence_ids'] = [exit_]
    cfg['confluence'] = [f'{CANARY_GATE_TF_PREFIX}-{interp}-{state}']
    return cfg


def _existing_canaries(slug: str, pack_name: str, user_id: str) -> dict:
    """Find this pack's already-created canaries (idempotency).
    Returns {'trigger': strat|None, 'gate': strat|None}."""
    from db import get_admin_client
    want = {
        'trigger': _canary_name(pack_name, 'trigger'),
        'gate': _canary_name(pack_name, 'gate'),
    }
    out = {'trigger': None, 'gate': None}
    try:
        rows = get_admin_client().table('strategies') \
            .select('id,name,config,alert_tracking_enabled,created_at') \
            .eq('user_id', user_id).execute().data or []
    except Exception as e:
        logger.warning("canary lookup failed for %s: %s", slug, e)
        return out
    for r in rows:
        nm = r.get('name')
        cfg = r.get('config') or {}
        tags = cfg.get('tags') or []
        if CANARY_TAG not in tags:
            continue
        for role, wanted in want.items():
            if nm == wanted:
                out[role] = r
    return out


def _canary_liveness(strategy_id: int, user_id: str, role: str,
                     created_at: Optional[str]) -> dict:
    """Per-canary live firing state for the Live Test status panel.

    Returns {last_alert, alerts_today, alerts_7d, status, verdict}.
    `status`: 'firing' | 'waiting' | 'silent'
      - waiting: created < 30 min ago and nothing yet — give it time
      - firing:  fired a live alert in the last 7 days
      - silent:  old enough to have fired but hasn't → flag it
    """
    from db import get_admin_client
    from datetime import datetime, timezone, timedelta
    c = get_admin_client()
    now = datetime.now(timezone.utc)
    today = now.strftime('%Y-%m-%d')
    since_7d = (now - timedelta(days=7)).isoformat()
    try:
        last = c.table('alerts').select('timestamp') \
            .eq('strategy_id', strategy_id).eq('user_id', user_id) \
            .order('timestamp', desc=True).limit(1).execute().data or []
        last_alert = last[0]['timestamp'] if last else None
        today_n = c.table('alerts').select('id', count='exact', head=True) \
            .eq('strategy_id', strategy_id).gte('timestamp', today) \
            .execute().count or 0
        week_n = c.table('alerts').select('id', count='exact', head=True) \
            .eq('strategy_id', strategy_id).gte('timestamp', since_7d) \
            .execute().count or 0
    except Exception as e:
        logger.warning("canary liveness query failed sid=%s: %s",
                        strategy_id, e)
        return {'last_alert': None, 'alerts_today': 0, 'alerts_7d': 0,
                'status': 'unknown', 'verdict': 'liveness query failed'}

    fresh = False
    if created_at:
        try:
            age_min = (now - datetime.fromisoformat(created_at)).total_seconds() / 60
            fresh = age_min < 30
        except Exception:
            pass

    if week_n > 0:
        status = 'firing'
        verb = 'fires as a trigger' if role == 'trigger' else 'gates correctly'
        verdict = f'✓ Pack {verb} — {week_n} alert(s) in 7d'
    elif fresh:
        status = 'waiting'
        verdict = 'Just created — give it a few minutes of RTH to fire'
    else:
        status = 'silent'
        if role == 'gate':
            verdict = ('⚠ No alerts — the borrowed entry trigger should '
                       'fire often, so a silent gate canary points at a '
                       'cross-TF dispatch issue for this pack')
        else:
            verdict = '⚠ No alerts in 7d — pack trigger may not be firing'
    return {'last_alert': last_alert, 'alerts_today': today_n,
            'alerts_7d': week_n, 'status': status, 'verdict': verdict}


def create_canaries_for_pack(slug: str, user_id: str) -> dict:
    """Create the (up to) two canaries for a pack. Idempotent — any
    canary that already exists is returned untouched.

    Returns {'trigger': {...}|None, 'gate': {...}|None, 'created': [..]}.
    A None role means the pack can't fill it (no triggers / no outputs).
    """
    import pack_registry
    pack = pack_registry.get_pack(slug)
    if pack is None:
        return {'error': f'pack {slug} not found',
                'trigger': None, 'gate': None, 'created': []}
    pack_name = (pack.manifest or {}).get('name', slug)

    existing = _existing_canaries(slug, pack_name, user_id)
    builders = {'trigger': _build_trigger_canary,
                'gate': _build_gate_canary}
    result = {'trigger': existing['trigger'], 'gate': existing['gate'],
              'created': []}

    # The canonical create path (save_strategy_db) uses the user-JWT
    # client → RLS. In a real API request the JWT context is already
    # set so the insert passes as the user; from a worker/cron/test
    # context it isn't — set admin context so get_client() returns the
    # admin client (RLS bypassed) while user_id still scopes the row.
    # Same pattern forward_test_service uses. No-op when context
    # already matches (real request).
    from db import (
        set_admin_user_context, get_current_user_id, clear_current_user,
    )
    from api.routers.strategies import create_strategy
    _need_ctx = get_current_user_id() != user_id
    if _need_ctx:
        set_admin_user_context(user_id)
    try:
        for role, build in builders.items():
            if existing[role] is not None:
                continue  # already exists — idempotent
            cfg = build(pack, user_id)
            if cfg is None:
                continue  # pack can't fill this role
            try:
                saved = create_strategy(cfg, user={'id': user_id})
                result[role] = saved
                result['created'].append(role)
                logger.info("Pack canary created: %s (%s) sid=%s",
                            slug, role,
                            saved.get('id') if isinstance(saved, dict) else '?')
            except Exception as e:
                logger.error("canary create failed %s/%s: %s", slug, role, e,
                              exc_info=True)
                result[role] = {'error': str(e)}
    finally:
        if _need_ctx:
            clear_current_user()
    return result


def get_canaries_for_pack(slug: str, user_id: str) -> dict:
    """Return the pack's current canaries (no creation).

    {'trigger': strat|None, 'gate': strat|None,
     'trigger_possible': bool, 'gate_possible': bool}
    The *_possible flags tell the UI whether a missing canary is
    'not created yet' vs 'this pack can't fill that role'.
    """
    import pack_registry
    pack = pack_registry.get_pack(slug)
    if pack is None:
        return {'error': f'pack {slug} not found',
                'trigger': None, 'gate': None,
                'trigger_possible': False, 'gate_possible': False}
    m = pack.manifest or {}
    pack_name = m.get('name', slug)
    existing = _existing_canaries(slug, pack_name, user_id)
    # Attach per-canary liveness for the Live Test status panel.
    for role in ('trigger', 'gate'):
        strat = existing[role]
        if strat and strat.get('id'):
            strat['liveness'] = _canary_liveness(
                strat['id'], user_id, role, strat.get('created_at'))
    return {
        'trigger': existing['trigger'],
        'gate': existing['gate'],
        'trigger_possible': bool(m.get('triggers')),
        'gate_possible': bool(m.get('interpreters') and m.get('outputs')),
    }
