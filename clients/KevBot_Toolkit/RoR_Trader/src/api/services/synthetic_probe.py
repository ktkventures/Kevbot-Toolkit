"""Synthetic-strategy fire test for user packs (Phase C).

Primary question this answers: **will a strategy built with this pack
actually fire alerts when conditions meet?** Not "do live and backtest
match exactly" (fidelity — Q3) but "is something fundamentally broken
that would prevent firing?" (Q1). The replay-match used internally as
the verdict signal is a proxy: if a fresh backtest fires N entries and
a fresh live replay fires roughly N entries on overlapping bars, the
engine integration is alive.

The 4Q simulator (Q1+Q2+Q3+Q4) tests indicator/interpreter math in
isolation but doesn't exercise the full integration path
(TriggerEvaluator dispatch, parity-service matching, position state
machine, cross-TF shadow engines). Every backtest↔live drift bug
fixed during the Phase B/B+ drill cycle (see
docs/Parity_Trust_Roadmap_2026-04-29.md) lived in those layers, not
in pack math. This probe catches them at pack-creation time.

This module spins up a sandboxed strategy that uses the candidate
pack's triggers as entry/exit, optionally adds a known-good cross-TF
gate from another pack, runs `recompute_and_persist_stored_trades` to
materialize backtest output, then runs `run_strategy_parity` and
returns a structured verdict.

**Naming note:** the underlying functions are still called
`run_strategy_parity` etc. — the module/DB-schema rename to "fire test"
is deferred. The verdict semantics are clearer than the names: PASS =
"engine fires properly," PARTIAL = "fires but with drift," FAIL = "does
not fire at all" (engine bug).

Wire points:
  - `api/routers/ai_builder.py` install handler — call `run_pack_probe`
    after the existing 4Q gate, surface verdict to the wizard.
  - Frontend `PackBuilderPage.tsx` — display verdict in the
    "Validation" panel alongside the 4Q result.

Design constraints:
  - Probe strategy is temporary — created and torn down per probe run.
    Uses a synthetic flag (`is_synthetic_probe=True`) so it never
    appears in My Strategies.
  - Strategy is owned by the calling user's id (we want their group
    parameters / general packs to apply, so the probe matches the
    environment a real strategy would run in).
  - Probe failure does NOT block install on its own — current policy
    is "PASS or warn, don't block" until the probe has been validated
    against real packs. PASS-only is a future tightening.
"""
from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


# Map of "known-good" cross-TF gates we can pair with a candidate pack
# for cross-TF integration testing. Picked because they're parity-clean
# at the pack-level 4Q test (per docs/User_Pack_Parity_Baseline_2026-04-29.md).
# The probe picks one from a different family than the candidate so the
# test exercises actual cross-pack dispatch, not just the candidate's
# own state machine.
_KNOWN_GOOD_CROSS_TF_GATES: List[str] = [
    '1d-MACD_LINE_V2-M>S+',  # batch parity-clean, simple state
    '15m-EMA_STACK_V2-LMS',  # batch parity-clean, three-period state
]


def build_synthetic_strategy(
    pack_slug: str,
    user_id: str,
    *,
    symbol: str = 'SPY',
    timeframe: str = '1Min',
    days: int = 7,
    add_cross_tf_gate: bool = True,
) -> Optional[Dict[str, Any]]:
    """Build an in-memory strategy dict for the candidate pack.

    Picks the pack's first bull/long entry trigger and its opposite as
    exit, optionally pairs with a cross-TF gate from a different pack
    family. Returns None if the pack can't be probed (no incremental
    class, no tradeable triggers, etc.) — caller should fall back to
    4Q-only verdict.
    """
    import pack_registry
    pack = pack_registry.get_pack(pack_slug)
    if pack is None:
        logger.warning("[synth-probe] pack %r not registered", pack_slug)
        return None
    if pack.incremental_class is None:
        logger.warning(
            "[synth-probe] pack %r has no incremental_class — skipping",
            pack_slug)
        return None

    manifest = pack.manifest
    trigger_prefix = manifest.get('trigger_prefix', '')
    if not trigger_prefix:
        logger.warning("[synth-probe] pack %r has no trigger_prefix", pack_slug)
        return None

    # Pick entry/exit base triggers. Conventions across the v2 packs:
    # bull/up triggers and bear/down triggers come in pairs. The
    # manifest's `triggers` block enumerates them. Modular packs use
    # `trigger_levels` for the L-type pairs and a separate set for C.
    triggers = manifest.get('triggers', [])
    if isinstance(triggers, dict):
        # Modular schema — flatten to list of base names
        triggers = list(triggers.keys())
    elif isinstance(triggers, list):
        triggers = [t.get('base') if isinstance(t, dict) else t
                    for t in triggers]
    triggers = [t for t in triggers if t]
    if not triggers:
        logger.warning(
            "[synth-probe] pack %r exposes no triggers", pack_slug)
        return None

    # Heuristic entry/exit pairing
    bull_words = ('bull', 'up', 'cross_above', 'enter_oversold', 'long')
    bear_words = ('bear', 'down', 'cross_below', 'enter_overbought', 'short')
    entry_base = next(
        (t for t in triggers if any(w in t.lower() for w in bull_words)),
        None) or triggers[0]
    exit_base = next(
        (t for t in triggers if any(w in t.lower() for w in bear_words)),
        None)
    # Don't allow entry == exit (degenerate strategy)
    if exit_base == entry_base or exit_base is None:
        # fall back to last trigger that's not entry
        exit_base = next(
            (t for t in triggers if t != entry_base),
            None)
    if exit_base is None:
        logger.warning(
            "[synth-probe] pack %r has only one trigger — can't form pair",
            pack_slug)
        return None

    entry_id = f'{trigger_prefix}_default_{entry_base}'
    exit_id = f'{trigger_prefix}_default_{exit_base}'

    confluence: List[str] = []
    if add_cross_tf_gate:
        # Pick a cross-pack gate that doesn't reference the candidate pack
        cand_interp_keys = set(manifest.get('interpreters', []))
        for gate in _KNOWN_GOOD_CROSS_TF_GATES:
            parts = gate.split('-', 2)
            if len(parts) < 3:
                continue
            _tf, interp, _state = parts
            if interp not in cand_interp_keys:
                confluence.append(gate)
                break

    return {
        'name': f'[probe] {pack_slug}',
        'symbol': symbol,
        'direction': 'LONG',
        'timeframe': timeframe,
        'user_id': user_id,
        'entry_trigger_confluence_id': entry_id,
        'exit_trigger_confluence_ids': [exit_id],
        'confluence': confluence,
        'general_confluences': [],
        'data_days': days,
        'risk_per_trade': 100.0,
        'starting_balance': 10000.0,
        'trading_session': 'RTH',
        'stop_config': {'method': 'swing', 'lookback': 5, 'padding': 0.03},
        'target_config': None,
        'is_synthetic_probe': True,
    }


def run_pack_probe(
    pack_slug: str,
    user_id: str,
    *,
    symbol: str = 'SPY',
    timeframe: str = '1Min',
    days: int = 7,
    add_cross_tf_gate: bool = True,
    cleanup: bool = True,
) -> Dict[str, Any]:
    """Persist a synthetic strategy, run backtest + parity, return verdict.

    Returns a dict with keys:
        - status: 'pass' | 'partial' | 'fail' | 'error' | 'skipped'
        - verdict: parity verdict string
        - score: float in [0, 1] or None
        - matched/stored/replay_only counts
        - most_common_failing_gate
        - failing_gates: list of {required, replay_actual, count}
        - notes: list of human-readable observations

    Cleanup behavior: if cleanup=True (default), the synthetic strategy
    + its stored_trades are deleted before return. Set cleanup=False
    when debugging — the row stays in DB for manual inspection.
    """
    strategy = build_synthetic_strategy(
        pack_slug, user_id,
        symbol=symbol, timeframe=timeframe, days=days,
        add_cross_tf_gate=add_cross_tf_gate)
    if strategy is None:
        return {
            'status': 'skipped',
            'verdict': None,
            'notes': [f'pack {pack_slug!r} not eligible for synthetic probe'],
        }

    # Persist the synthetic strategy. The probe runs under admin context
    # (caller sets set_admin_user_context(user_id) before invoking) so
    # save_strategy_db uses the admin client. Strategy gets a real DB
    # row + assigned sid; we tear it down in the finally block.
    from db import (
        save_strategy_db,
        delete_strategy_db,
        get_admin_client,
        set_admin_user_context,
    )
    from api.services.forward_test_service import (
        recompute_and_persist_stored_trades,
    )
    from api.services.parity_service import run_strategy_parity

    set_admin_user_context(user_id)

    sid = None
    notes: List[str] = []
    try:
        saved = save_strategy_db(strategy)
        sid = saved.get('id') if isinstance(saved, dict) else None
        if sid is None:
            return {
                'status': 'error',
                'verdict': None,
                'notes': ['save_strategy_db returned no id'],
            }
        notes.append(f'synthetic strategy created: sid={sid}')

        # Materialize backtest stored_trades (no parity yet)
        bt = recompute_and_persist_stored_trades(
            sid, user_id, compute_parity=False)
        notes.append(
            f'backtest: status={bt.get("status")} trades={bt.get("trades")}')
        if not bt.get('trades'):
            return {
                'status': 'fail',
                'verdict': 'NO_BACKTEST_TRADES',
                'score': 0.0,
                'notes': notes + [
                    'backtest produced 0 trades — pack likely has no '
                    'firing condition on this symbol/window, or the '
                    'entry/exit pairing is degenerate'],
            }

        # Run parity
        report = run_strategy_parity(sid, user_id, last_n=None,
                                     forward_test_only=False)
        verdict = report.get('verdict', 'UNKNOWN')
        score = report.get('parity_score')
        failing_gates = []
        gate_counter: Dict[tuple, int] = {}
        for u in (report.get('stored_only') or []):
            for fg in (u.get('failing_gates') or []):
                key = (fg.get('required'), fg.get('replay_actual'))
                gate_counter[key] = gate_counter.get(key, 0) + 1
        failing_gates = [
            {'required': req, 'replay_actual': actual, 'count': n}
            for (req, actual), n in sorted(
                gate_counter.items(), key=lambda kv: -kv[1])
        ]

        if verdict == 'PASS':
            status = 'pass'
        elif verdict in ('PARTIAL',):
            status = 'partial'
        else:
            status = 'fail'

        return {
            'status': status,
            'verdict': verdict,
            'score': score,
            'matched_count': report.get('matched_count'),
            'stored_count': report.get('stored_count'),
            'replay_only_count': report.get('replay_only_count'),
            'most_common_failing_gate': report.get('most_common_failing_gate'),
            'failing_gates': failing_gates,
            'notes': notes,
        }
    except Exception as e:
        logger.exception("[synth-probe] sid=%s pack=%s failed", sid, pack_slug)
        return {
            'status': 'error',
            'verdict': None,
            'notes': notes + [f'exception: {type(e).__name__}: {e}'],
        }
    finally:
        if cleanup and sid is not None:
            try:
                delete_strategy_db(sid)
                logger.info("[synth-probe] cleaned up synthetic sid=%s", sid)
            except Exception as e:
                logger.warning(
                    "[synth-probe] cleanup failed for sid=%s: %s", sid, e)
