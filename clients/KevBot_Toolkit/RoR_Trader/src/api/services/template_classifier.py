"""Template classification — shared between the audit script and the
strategies list endpoint's `template_class` attachment.

A strategy is classified by which templates it references via:
  - entry_trigger_confluence_id  (e.g. 'macd_line_v2_default_cross_bull')
  - exit_trigger_confluence_ids
  - confluence records           (e.g. '1Min-MACD_LINE-S>M')

Classes:
  - 'user_pack'  — every reference resolves to a user pack
  - 'legacy'     — every reference resolves to a built-in template that
                   has a user-pack mirror available (i.e. should be
                   migrated)
  - 'mixed'      — references both user-pack and legacy templates
                   (partial migration in flight, like sid 144 was
                   before yesterday's bar_count_exit/eppv4 swap)
  - 'no_mirror'  — references built-ins with no available mirror; can't
                   be cleanly migrated without new pack work
  - 'unknown'    — couldn't classify (deleted pack, custom template)
  - 'empty'      — no references at all (webhook_inbound, etc.)

Used by the My Strategies list endpoint so the frontend can offer a
'Source' filter (All / User-pack / Legacy) without re-running the audit
script per request.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional


# Mirror map mirrors `src/_audit_strategy_migration.py:64-74`. Keep in
# sync — both files reference the same logical mapping.
MIRROR_MAP = {
    'ema_stack': ['ema_stack_v2'],
    'ema_price_position': ['ema_pp_v3', 'ema_pp_v4'],
    'ema_price_position_v2': ['ema_pp_v3', 'ema_pp_v4'],
    'macd_line': ['macd_line_v2'],
    'macd_histogram': ['macd_histogram_v2'],
    'rvol': ['rvol_v2'],
    'vwap': ['vwap_v2'],
    'utbot': ['ut_bot_v4'],
    'utbot_v2': ['ut_bot_v4'],
}


# Cached lazy-builders so the API process loads templates + user packs
# once. Cache invalidates on process restart, which is the correct
# granularity (template/pack registries are loaded at boot anyway).

@lru_cache(maxsize=1)
def _builtin_template_index() -> tuple[set, dict, dict]:
    """Returns (template_slugs, interpreter_to_slug, trigger_prefix_to_slug)."""
    from confluence_groups import TEMPLATES
    slugs = set(TEMPLATES.keys())
    interps: dict[str, str] = {}
    prefixes: dict[str, str] = {}
    for slug, t in TEMPLATES.items():
        interp_field = t.get('interpreters', [])
        if isinstance(interp_field, list):
            for i in interp_field:
                interps[i] = slug
        elif isinstance(interp_field, str):
            interps[interp_field] = slug
        tp = t.get('trigger_prefix', '')
        if tp:
            prefixes[tp] = slug
    return slugs, interps, prefixes


@lru_cache(maxsize=1)
def _user_pack_index() -> tuple[set, dict, dict]:
    """Returns (pack_slugs, interpreter_to_slug, trigger_prefix_to_slug)."""
    user_packs_dir = Path(__file__).resolve().parent.parent.parent.parent / "user_packs"
    slugs: set = set()
    interps: dict[str, str] = {}
    prefixes: dict[str, str] = {}
    if not user_packs_dir.exists():
        return slugs, interps, prefixes
    for manifest_path in user_packs_dir.glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            continue
        slug = manifest_path.parent.name
        slugs.add(slug)
        for i in manifest.get('interpreters', []):
            interps[i] = slug
        tp = manifest.get('trigger_prefix', '')
        if tp:
            prefixes[tp] = slug
    return slugs, interps, prefixes


def _classify_template(slug: Optional[str]) -> str:
    """Map a template slug to its source class. Returns one of:
    'user_pack' | 'legacy_mirrorable' | 'legacy_no_mirror' | 'unknown'."""
    if slug is None:
        return 'unknown'
    user_packs, _, _ = _user_pack_index()
    if slug in user_packs:
        return 'user_pack'
    builtin, _, _ = _builtin_template_index()
    if slug in builtin:
        return 'legacy_mirrorable' if MIRROR_MAP.get(slug) else 'legacy_no_mirror'
    return 'unknown'


def _template_for_interpreter(interp: str) -> Optional[str]:
    _, user_interps, _ = _user_pack_index()
    if interp in user_interps:
        return user_interps[interp]
    _, builtin_interps, _ = _builtin_template_index()
    return builtin_interps.get(interp)


def _template_for_trigger_id(trig_id: str) -> Optional[str]:
    """Match a trigger ID like 'macd_line_v2_default_cross_bull' to its
    source template by longest matching prefix."""
    if not trig_id:
        return None
    user_slugs, _, user_pfx = _user_pack_index()
    builtin_slugs, _, builtin_pfx = _builtin_template_index()
    all_prefixes = sorted(
        list(user_pfx.keys()) + list(builtin_pfx.keys())
        + list(user_slugs) + list(builtin_slugs),
        key=len, reverse=True)
    for p in all_prefixes:
        if trig_id.startswith(p + '_') or trig_id == p:
            return (user_pfx.get(p) or builtin_pfx.get(p)
                    or (p if p in builtin_slugs or p in user_slugs else None))
    return None


def _parse_confluence_record(rec: str) -> Optional[tuple[str, str, str]]:
    """'1h-UT_BOT_V4-BULL_TREND' -> ('1h', 'UT_BOT_V4', 'BULL_TREND').
    Returns None for general-pack records ('GEN-...') or malformed."""
    if not rec or rec.startswith('GEN-'):
        return None
    # Strip [CB] / [PB] fidelity suffixes the engine may attach.
    clean = rec.replace('[CB]', '').replace('[PB]', '')
    parts = clean.split('-', 2)
    if len(parts) < 3:
        return None
    return (parts[0], parts[1], parts[2])


def classify_strategy(strat: dict) -> dict:
    """Classify a single strategy. Returns a dict with:
        - template_class: 'user_pack' | 'legacy' | 'mixed' | 'no_mirror' | 'unknown' | 'empty'
        - referenced_templates: list of distinct template slugs the strategy references

    Pure function; safe to call from request handlers. Uses lru_cache'd
    template/pack indexes so cost is dominated by the strategy's own
    confluence list size.
    """
    cfg = strat.get('config') or {}
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg)
        except Exception:
            cfg = {}

    referenced_templates: set[str] = set()

    # Entry trigger
    eid = cfg.get('entry_trigger_confluence_id') or strat.get('entry_trigger_confluence_id')
    if eid:
        tpl = _template_for_trigger_id(eid)
        if tpl:
            referenced_templates.add(tpl)

    # Exit triggers (config or top-level)
    exits = cfg.get('exit_trigger_confluence_ids') or strat.get('exit_trigger_confluence_ids') or []
    if isinstance(exits, str):
        try:
            exits = json.loads(exits)
        except Exception:
            exits = []
    for x in exits or []:
        tpl = _template_for_trigger_id(x)
        if tpl:
            referenced_templates.add(tpl)

    # Confluence records — cross-TF interpreter states
    confs = cfg.get('confluence') or strat.get('confluence') or []
    if isinstance(confs, str):
        try:
            confs = json.loads(confs)
        except Exception:
            confs = []
    for rec in confs or []:
        parsed = _parse_confluence_record(rec)
        if parsed:
            _tf, interp, _state = parsed
            tpl = _template_for_interpreter(interp)
            if tpl:
                referenced_templates.add(tpl)

    if not referenced_templates:
        return {
            'template_class': 'empty',
            'referenced_templates': [],
        }

    classes = [_classify_template(t) for t in referenced_templates]
    distinct = set(classes)

    if distinct == {'user_pack'}:
        template_class = 'user_pack'
    elif 'user_pack' in distinct and ('legacy_mirrorable' in distinct or 'legacy_no_mirror' in distinct):
        template_class = 'mixed'
    elif distinct == {'legacy_mirrorable'}:
        template_class = 'legacy'
    elif 'legacy_no_mirror' in distinct and 'user_pack' not in distinct:
        # All-legacy with at least one no-mirror template
        template_class = 'no_mirror'
    elif distinct == {'unknown'}:
        template_class = 'unknown'
    else:
        # Mixed legacy-mirrorable + unknown, or other combos
        template_class = 'legacy' if 'legacy_mirrorable' in distinct else 'unknown'

    return {
        'template_class': template_class,
        'referenced_templates': sorted(referenced_templates),
    }


def attach_template_class(strategies: list) -> None:
    """Mutate a list of strategy dicts to add 'template_class' and
    'referenced_templates' fields. Used by list_strategies endpoint."""
    for s in strategies:
        try:
            classification = classify_strategy(s)
            s['template_class'] = classification['template_class']
            s['referenced_templates'] = classification['referenced_templates']
        except Exception:
            # Don't let classification failure break the whole list
            s['template_class'] = 'unknown'
            s['referenced_templates'] = []
