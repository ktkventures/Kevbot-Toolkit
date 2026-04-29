"""Complete the v2 migration on the 10 Tier-A mirror strategies.

Three independent changes, all idempotent:

1. Re-enable swing_123_default confluence group.
   7 mirrors (137,138,139,140,141,142,143) reference swing_123 triggers.
   Group was disabled → next worker restart would crash per
   feedback_disabled_groups_kill_worker.md.

2. Migrate bar_count_exit=4 → time_exit_config={max_hold_bars: 4} on
   sids 135, 136, 144. Modernizes to the time-exit-packs path; the
   legacy bar_count_exit field is no longer the canonical source.

3. Swap sid 144's entry trigger + 1d confluence from
   ema_price_position_v2 → ema_pp_v4 (Tier B). v4 chosen over v3 for
   its L1 semantic (1-bar lag, closer to the original C-type bar-close
   behavior than v3's L0 intra-bar semantic).

Run from src/:
    ../.venv/bin/python _migrate_mirrors_to_v2.py [--apply]

Without --apply: dry-run, prints what would change.
With --apply: executes the writes via admin client.
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

import json
import sys
from typing import Any

from db import get_admin_client, set_admin_user_context

USER = "19d47e46-f718-49a6-af32-5f5407f5b170"
APPLY = '--apply' in sys.argv

set_admin_user_context(USER)
client = get_admin_client()

print(f"\n=== v2 Mirror Migration {'[APPLY]' if APPLY else '[DRY RUN]'} ===\n")


# ---------------------------------------------------------------------------
# 1. Re-enable swing_123_default
# ---------------------------------------------------------------------------

print("[1] Confluence-group enable check")
r = client.table('confluence_groups').select('groups').eq('user_id', USER).limit(1).execute()
groups = (r.data or [{}])[0].get('groups') or []
if isinstance(groups, str):
    groups = json.loads(groups)

changed = False
for g in groups:
    if g.get('id') == 'swing_123_default' and not g.get('enabled', True):
        print(f"   - {g['id']}  enabled: False → True")
        g['enabled'] = True
        changed = True

if not changed:
    print("   - swing_123_default already enabled, no change")

if changed and APPLY:
    client.table('confluence_groups').update({'groups': groups}).eq('user_id', USER).execute()
    print("   ✓ written")


# ---------------------------------------------------------------------------
# 2 + 3. Per-strategy config migrations
# ---------------------------------------------------------------------------

# Each tuple: (sid, description, mutator)
# mutator(cfg) returns True if it modified cfg, False otherwise.

def m_drop_bar_count_exit(cfg: dict) -> bool:
    """Drop bar_count_exit; add time_exit_config if not already present."""
    bc = cfg.get('bar_count_exit')
    if bc is None:
        return False
    cfg.pop('bar_count_exit', None)
    if not cfg.get('time_exit_config'):
        cfg['time_exit_config'] = {
            'method': 'max_hold_bars',
            'max_bars': int(bc),
        }
        print(f"      bar_count_exit={bc} → time_exit_config(max_hold_bars={bc})")
    else:
        print(f"      bar_count_exit={bc} dropped (time_exit_config already set: {cfg['time_exit_config']})")
    return True


def m_swap_sid_144_entry(cfg: dict) -> bool:
    """Sid 144 only: swap entry_trigger + 1d confluence to ema_pp_v4."""
    changed = False
    eid = cfg.get('entry_trigger_confluence_id')
    if eid == 'ema_price_position_v2_default_cross_short_up':
        cfg['entry_trigger_confluence_id'] = 'eppv4_default_cross_short_up'
        print(f"      entry_trigger: {eid!r} → 'eppv4_default_cross_short_up'")
        changed = True

    confs = cfg.get('confluence') or []
    if isinstance(confs, str):
        try: confs = json.loads(confs)
        except Exception: confs = []
    new_confs = []
    for rec in confs:
        if rec == '1d-EMA_PRICE_POSITION-PSML':
            new_confs.append('1d-EMA_PP_V4-PSML')
            print(f"      confluence: {rec!r} → '1d-EMA_PP_V4-PSML'")
            changed = True
        else:
            new_confs.append(rec)
    if changed and new_confs != confs:
        cfg['confluence'] = new_confs
    return changed


PLAN = [
    (135, [m_drop_bar_count_exit]),
    (136, [m_drop_bar_count_exit]),
    (144, [m_drop_bar_count_exit, m_swap_sid_144_entry]),
]


def update_strategy_config(sid: int, new_cfg: dict) -> None:
    """Write the full config blob back. Cannot do partial JSONB merges
    via PostgREST — caller already merged."""
    client.table('strategies').update({'config': new_cfg}).eq('id', sid).eq('user_id', USER).execute()


print("\n[2+3] Per-strategy migrations\n")
for sid, mutators in PLAN:
    r = client.table('strategies').select('id,name,config').eq('id', sid).eq('user_id', USER).limit(1).execute()
    if not r.data:
        print(f"   sid {sid}: NOT FOUND, skipping")
        continue
    row = r.data[0]
    cfg = row['config']
    if isinstance(cfg, str):
        cfg = json.loads(cfg)
    cfg = dict(cfg)  # don't mutate the supabase result

    print(f"   sid {sid}: {row['name']}")
    any_changed = False
    for mutator in mutators:
        if mutator(cfg):
            any_changed = True

    if not any_changed:
        print("      (no changes needed)")
        continue

    if APPLY:
        update_strategy_config(sid, cfg)
        print("      ✓ written")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n=== Done. {'Run with --apply to write.' if not APPLY else 'Changes committed.'} ===\n")
