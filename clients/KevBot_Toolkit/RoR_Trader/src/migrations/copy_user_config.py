#!/usr/bin/env python3
"""
Copy user configuration data between accounts.

Copies confluence_groups, general_packs, and risk_management_packs
from one user to another. Optionally copies strategies, portfolios,
and alert_config.

Usage:
    python -m migrations.copy_user_config <source_email> <target_email> [--all]

    --all    Also copy strategies, portfolios, alerts, alert_config, requirements
    --dry-run  Show what would be copied without writing

Requires: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY environment variables.
"""
import os
import sys
import argparse
from pathlib import Path

# Ensure src/ is on path and USE_DB is set
os.environ["USE_DB"] = "true"
_SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from dotenv import load_dotenv
load_dotenv(_SCRIPT_DIR / '.env', override=True)

from supabase import create_client


def get_client():
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
        sys.exit(1)
    return create_client(url, key)


def find_user_by_email(client, email: str) -> dict:
    """Look up a user by email via Supabase Admin API."""
    users = client.auth.admin.list_users()
    for u in users:
        if u.email == email:
            return {'id': str(u.id), 'email': u.email}
    return None


def copy_jsonb_table(client, table: str, column: str,
                     source_id: str, target_id: str, dry_run: bool):
    """Copy a single-row JSONB config table from source to target user."""
    result = client.table(table) \
        .select(column) \
        .eq('user_id', source_id) \
        .maybe_single() \
        .execute()

    if not result or not result.data:
        print(f"  {table}: source has no data — skipping")
        return

    data = result.data[column]
    count = len(data) if isinstance(data, list) else 'object'
    print(f"  {table}: {count} entries")

    if not dry_run:
        from datetime import datetime, timezone
        client.table(table).upsert({
            'user_id': target_id,
            column: data,
            'updated_at': datetime.now(timezone.utc).isoformat(),
        }, on_conflict='user_id').execute()
        print(f"    -> copied to target")


def copy_strategies(client, source_id: str, target_id: str, dry_run: bool):
    """Copy all strategies from source to target user."""
    result = client.table('strategies') \
        .select('*') \
        .eq('user_id', source_id) \
        .order('id') \
        .execute()

    strategies = result.data
    print(f"  strategies: {len(strategies)} rows")

    if not dry_run:
        id_map = {}  # old_id -> new_id
        for s in strategies:
            old_id = s['id']
            s.pop('id')
            s['user_id'] = target_id
            res = client.table('strategies').insert(s).execute()
            new_id = res.data[0]['id']
            id_map[old_id] = new_id
            print(f"    strategy {old_id} -> {new_id}: {s.get('name', '?')}")
        return id_map
    return {}


def copy_portfolios(client, source_id: str, target_id: str,
                    strategy_id_map: dict, dry_run: bool):
    """Copy portfolios, remapping strategy IDs."""
    result = client.table('portfolios') \
        .select('*') \
        .eq('user_id', source_id) \
        .order('id') \
        .execute()

    portfolios = result.data
    print(f"  portfolios: {len(portfolios)} rows")

    if not dry_run:
        for p in portfolios:
            old_id = p.pop('id')
            p['user_id'] = target_id
            # Remap strategy references
            strats = p.get('strategies', [])
            remapped = []
            for alloc in strats:
                old_sid = alloc.get('strategy_id')
                new_sid = strategy_id_map.get(old_sid)
                if new_sid:
                    alloc['strategy_id'] = new_sid
                    remapped.append(alloc)
            p['strategies'] = remapped
            res = client.table('portfolios').insert(p).execute()
            new_id = res.data[0]['id']
            print(f"    portfolio {old_id} -> {new_id}: {p.get('name', '?')} "
                  f"({len(remapped)} strategies)")


def main():
    parser = argparse.ArgumentParser(
        description="Copy user config between RoR Trader accounts")
    parser.add_argument("source_email", help="Source user email")
    parser.add_argument("target_email", help="Target user email")
    parser.add_argument("--all", action="store_true",
                        help="Also copy strategies, portfolios, alert_config")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be copied without writing")
    args = parser.parse_args()

    client = get_client()

    # Resolve users
    source = find_user_by_email(client, args.source_email)
    if not source:
        print(f"ERROR: No user found with email: {args.source_email}")
        sys.exit(1)
    print(f"Source: {source['email']} ({source['id']})")

    target = find_user_by_email(client, args.target_email)
    if not target:
        print(f"ERROR: No user found with email: {args.target_email}")
        sys.exit(1)
    print(f"Target: {target['email']} ({target['id']})")

    if source['id'] == target['id']:
        print("ERROR: Source and target are the same user")
        sys.exit(1)

    if args.dry_run:
        print("\n--- DRY RUN (no changes will be made) ---\n")
    else:
        print()

    # Always copy config tables
    print("Copying configuration packs:")
    copy_jsonb_table(client, 'confluence_groups', 'groups',
                     source['id'], target['id'], args.dry_run)
    copy_jsonb_table(client, 'general_packs', 'packs',
                     source['id'], target['id'], args.dry_run)
    copy_jsonb_table(client, 'risk_management_packs', 'packs',
                     source['id'], target['id'], args.dry_run)
    copy_jsonb_table(client, 'user_settings', 'settings',
                     source['id'], target['id'], args.dry_run)

    if args.all:
        print("\nCopying strategies & portfolios:")
        strategy_id_map = copy_strategies(
            client, source['id'], target['id'], args.dry_run)
        copy_portfolios(
            client, source['id'], target['id'], strategy_id_map, args.dry_run)

        print("\nCopying alert config:")
        copy_jsonb_table(client, 'alert_config', 'config',
                         source['id'], target['id'], args.dry_run)

    print("\nDone!")


if __name__ == "__main__":
    main()
