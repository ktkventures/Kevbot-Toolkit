#!/usr/bin/env python3
"""
JSON-to-Supabase migration script for RoR Trader.

Reads local JSON files and inserts them into Supabase PostgreSQL tables,
remapping all cross-referenced IDs (strategies, portfolios, requirement sets).

Usage (from src/ directory):
    python -m migrations.migrate_json_to_db <user_id> [--dry-run] [--force] [--skip-alerts]

Requires .env with: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
"""
import argparse
import json
import os
import sys
import copy
from pathlib import Path
from datetime import datetime, timezone

# Ensure USE_DB is true so db.py initializes Supabase support
os.environ["USE_DB"] = "true"

from dotenv import load_dotenv
load_dotenv(override=True)

from db import (
    _strategy_to_row, _alert_to_row,
    SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY,
)

SRC_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = SRC_DIR.parent / "config"


# ============================================================
# JSON Loading
# ============================================================

def load_json(path: Path, default=None):
    """Load a JSON file, returning default if missing."""
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return default if default is not None else []
    with open(path) as f:
        return json.load(f)


def load_all_json():
    """Load all JSON source files."""
    data = {}
    data['strategies'] = load_json(SRC_DIR / "strategies.json", [])
    data['portfolios'] = load_json(SRC_DIR / "portfolios.json", [])
    data['requirements'] = load_json(SRC_DIR / "requirements.json", [])
    data['alerts'] = load_json(SRC_DIR / "alerts.json", [])
    data['alert_config'] = load_json(SRC_DIR / "alert_config.json", {})
    data['webhook_templates'] = load_json(SRC_DIR / "webhook_templates.json", [])
    data['monitor_status'] = load_json(SRC_DIR / "monitor_status.json", {})
    data['engine_state'] = load_json(SRC_DIR / "engine_state.json", {})
    data['confluence_groups'] = load_json(CONFIG_DIR / "confluence_groups.json", {})
    data['general_packs'] = load_json(CONFIG_DIR / "general_packs.json", {})
    data['risk_management_packs'] = load_json(CONFIG_DIR / "risk_management_packs.json", {})
    data['settings'] = load_json(CONFIG_DIR / "settings.json", {})
    return data


# ============================================================
# Migration Phases
# ============================================================

def migrate_requirement_sets(client, requirements, user_id, dry_run=False):
    """Insert requirement sets. Returns old_id → new_id map."""
    req_id_map = {}
    stats = {'found': len(requirements), 'inserted': 0, 'skipped': 0}

    # Check for existing built-in sets
    existing = client.table('requirement_sets') \
        .select('id, name') \
        .is_('user_id', 'null') \
        .execute()
    existing_names = {r['name'] for r in existing.data}

    for req in requirements:
        old_id = req['id']
        row = copy.deepcopy(req)
        row.pop('id', None)

        if req.get('built_in'):
            # Built-in: user_id IS NULL
            if req['name'] in existing_names:
                # Already seeded — find the DB id
                db_match = next(r for r in existing.data if r['name'] == req['name'])
                req_id_map[old_id] = db_match['id']
                stats['skipped'] += 1
                print(f"  [skip] Built-in requirement '{req['name']}' already exists (db_id={db_match['id']})")
                continue
            row.pop('user_id', None)
            row['built_in'] = True
        else:
            row['user_id'] = user_id
            row['built_in'] = False

        if dry_run:
            req_id_map[old_id] = old_id
            stats['inserted'] += 1
            continue

        result = client.table('requirement_sets').insert(row).execute()
        new_id = result.data[0]['id']
        req_id_map[old_id] = new_id
        stats['inserted'] += 1
        print(f"  Requirement '{req['name']}': {old_id} → {new_id}")

    return req_id_map, stats


def migrate_strategies(client, strategies, user_id, dry_run=False):
    """Insert strategies. Returns old_id → new_id map."""
    strategy_id_map = {}
    stats = {'found': len(strategies), 'inserted': 0, 'skipped': 0}

    for strat in strategies:
        old_id = strat['id']
        row_data = copy.deepcopy(strat)
        row_data.pop('id', None)
        row_data['user_id'] = user_id

        row = _strategy_to_row(row_data)
        row.pop('id', None)

        if dry_run:
            strategy_id_map[old_id] = old_id
            stats['inserted'] += 1
            continue

        result = client.table('strategies').insert(row).execute()
        new_id = result.data[0]['id']
        strategy_id_map[old_id] = new_id
        stats['inserted'] += 1
        print(f"  Strategy '{strat['name']}': {old_id} → {new_id}")

    return strategy_id_map, stats


def migrate_portfolios(client, portfolios, user_id, strategy_id_map, req_id_map, dry_run=False):
    """Insert portfolios with remapped strategy and requirement IDs."""
    portfolio_id_map = {}
    stats = {'found': len(portfolios), 'inserted': 0, 'skipped': 0}
    warnings = []

    for port in portfolios:
        old_id = port['id']
        row = copy.deepcopy(port)
        row.pop('id', None)
        row['user_id'] = user_id

        # Remap strategy references
        if 'strategies' in row and isinstance(row['strategies'], list):
            remapped = []
            for s in row['strategies']:
                old_sid = s.get('strategy_id')
                if old_sid in strategy_id_map:
                    s_copy = copy.deepcopy(s)
                    s_copy['strategy_id'] = strategy_id_map[old_sid]
                    remapped.append(s_copy)
                else:
                    warnings.append(f"Portfolio '{port['name']}' (old_id={old_id}): dropped orphan strategy_id={old_sid}")
            row['strategies'] = remapped

        # Remap requirement_set_id
        old_req_id = row.get('requirement_set_id')
        if old_req_id and old_req_id in req_id_map:
            row['requirement_set_id'] = req_id_map[old_req_id]
        elif old_req_id:
            warnings.append(f"Portfolio '{port['name']}': dropped orphan requirement_set_id={old_req_id}")
            row['requirement_set_id'] = None

        if dry_run:
            portfolio_id_map[old_id] = old_id
            stats['inserted'] += 1
            continue

        result = client.table('portfolios').insert(row).execute()
        new_id = result.data[0]['id']
        portfolio_id_map[old_id] = new_id
        stats['inserted'] += 1
        print(f"  Portfolio '{port['name']}': {old_id} → {new_id}")

    return portfolio_id_map, stats, warnings


def migrate_webhook_templates(client, templates, dry_run=False):
    """Insert default webhook templates (user_id IS NULL)."""
    stats = {'found': len(templates), 'inserted': 0, 'skipped': 0}

    # Check existing
    existing = client.table('webhook_templates') \
        .select('id') \
        .is_('user_id', 'null') \
        .execute()
    existing_ids = {r['id'] for r in existing.data}

    for tpl in templates:
        if tpl['id'] in existing_ids:
            stats['skipped'] += 1
            print(f"  [skip] Template '{tpl['name']}' already exists")
            continue

        row = copy.deepcopy(tpl)
        row.pop('user_id', None)  # Ensure NULL for built-in

        if dry_run:
            stats['inserted'] += 1
            continue

        client.table('webhook_templates').insert(row).execute()
        stats['inserted'] += 1
        print(f"  Template '{tpl['name']}': inserted")

    return stats


def migrate_alerts(client, alerts, user_id, strategy_id_map, portfolio_id_map, dry_run=False):
    """Insert alerts with remapped IDs. Batch inserts for performance."""
    stats = {'found': len(alerts), 'inserted': 0, 'skipped': 0}
    warnings = []
    batch = []

    for alert in alerts:
        old_sid = alert.get('strategy_id')
        if old_sid and old_sid not in strategy_id_map:
            stats['skipped'] += 1
            warnings.append(f"Dropped alert (type={alert.get('type')}, strategy_id={old_sid}): orphan strategy")
            continue

        row_data = copy.deepcopy(alert)
        row_data.pop('id', None)
        row_data['user_id'] = user_id

        # Remap strategy_id
        if old_sid:
            row_data['strategy_id'] = strategy_id_map[old_sid]

        # Remap portfolio_context
        if 'portfolio_context' in row_data and isinstance(row_data['portfolio_context'], list):
            remapped_pc = []
            for pc in row_data['portfolio_context']:
                old_pid = pc.get('portfolio_id')
                if old_pid and old_pid in portfolio_id_map:
                    pc_copy = copy.deepcopy(pc)
                    pc_copy['portfolio_id'] = portfolio_id_map[old_pid]
                    remapped_pc.append(pc_copy)
                # Drop orphan portfolio references silently (very common)
            row_data['portfolio_context'] = remapped_pc

        # Remap webhook_deliveries
        if 'webhook_deliveries' in row_data and isinstance(row_data['webhook_deliveries'], list):
            for wd in row_data['webhook_deliveries']:
                old_pid = wd.get('portfolio_id')
                if old_pid and old_pid in portfolio_id_map:
                    wd['portfolio_id'] = portfolio_id_map[old_pid]

        row = _alert_to_row(row_data)
        row.pop('id', None)
        batch.append(row)
        stats['inserted'] += 1

    if dry_run or not batch:
        return stats, warnings

    # Insert in chunks of 50
    CHUNK = 50
    for i in range(0, len(batch), CHUNK):
        chunk = batch[i:i+CHUNK]
        client.table('alerts').insert(chunk).execute()
        print(f"  Inserted alerts {i+1}–{min(i+CHUNK, len(batch))} of {len(batch)}")

    return stats, warnings


def migrate_alert_config(client, config, user_id, strategy_id_map, portfolio_id_map, dry_run=False):
    """Insert alert config with remapped strategy/portfolio keys."""
    if not config:
        return {'found': 0, 'inserted': 0, 'skipped': 0}, []

    warnings = []
    remapped = copy.deepcopy(config)

    # Remap strategies keys
    if 'strategies' in remapped:
        new_strats = {}
        for old_key, val in remapped['strategies'].items():
            old_id = int(old_key)
            if old_id in strategy_id_map:
                new_strats[str(strategy_id_map[old_id])] = val
            else:
                warnings.append(f"Alert config: dropped orphan strategies key={old_key}")
        remapped['strategies'] = new_strats

    # Remap portfolios keys
    if 'portfolios' in remapped:
        new_ports = {}
        for old_key, val in remapped['portfolios'].items():
            old_id = int(old_key)
            if old_id in portfolio_id_map:
                new_ports[str(portfolio_id_map[old_id])] = val
            else:
                warnings.append(f"Alert config: dropped orphan portfolios key={old_key}")
        remapped['portfolios'] = new_ports

    if dry_run:
        return {'found': 1, 'inserted': 1, 'skipped': 0}, warnings

    now = datetime.now(timezone.utc).isoformat()
    client.table('alert_config').upsert({
        'user_id': user_id,
        'config': remapped,
        'updated_at': now,
    }, on_conflict='user_id').execute()
    print(f"  Alert config: upserted ({len(remapped.get('strategies', {}))} strategies, {len(remapped.get('portfolios', {}))} portfolios)")

    return {'found': 1, 'inserted': 1, 'skipped': 0}, warnings


def migrate_user_configs(client, data, user_id, dry_run=False):
    """Insert user settings, confluence groups, general packs, risk mgmt packs."""
    now = datetime.now(timezone.utc).isoformat()
    configs = [
        ('user_settings', 'settings', data.get('settings', {})),
        ('confluence_groups', 'groups', data.get('confluence_groups', {}).get('groups', [])),
        ('general_packs', 'packs', data.get('general_packs', {}).get('packs', [])),
        ('risk_management_packs', 'packs', data.get('risk_management_packs', {}).get('packs', [])),
    ]

    stats = {}
    for table, col, values in configs:
        if not values:
            print(f"  [skip] {table}: no data")
            stats[table] = {'found': 0, 'inserted': 0, 'skipped': 0}
            continue

        if dry_run:
            stats[table] = {'found': 1, 'inserted': 1, 'skipped': 0}
            continue

        client.table(table).upsert({
            'user_id': user_id,
            col: values,
            'updated_at': now,
        }, on_conflict='user_id').execute()
        count = len(values) if isinstance(values, list) else len(values.keys())
        print(f"  {table}: upserted ({count} items)")
        stats[table] = {'found': 1, 'inserted': 1, 'skipped': 0}

    return stats


def migrate_monitor_state(client, monitor_status, engine_state, user_id, strategy_id_map, dry_run=False):
    """Insert monitor status and engine state with remapped position keys."""
    now = datetime.now(timezone.utc).isoformat()
    warnings = []

    # Remap engine state position keys
    positions = engine_state.get('positions', {})
    new_positions = {}
    for old_key, pos in positions.items():
        try:
            old_id = int(old_key)
            if old_id in strategy_id_map:
                new_positions[str(strategy_id_map[old_id])] = pos
            else:
                warnings.append(f"Engine state: dropped orphan position key={old_key}")
        except ValueError:
            new_positions[old_key] = pos  # Keep non-integer keys as-is

    remapped_engine_state = {'positions': new_positions}
    if 'saved_at' in engine_state:
        remapped_engine_state['saved_at'] = engine_state['saved_at']

    # Clean up monitor status (remove runtime fields)
    status = copy.deepcopy(monitor_status)
    status.pop('pid', None)  # PID is stale after migration
    status.pop('running', None)  # Will be false after migration
    desired_state = status.pop('desired_state', 'stopped')

    if dry_run:
        return {'found': 1, 'inserted': 1, 'skipped': 0}, warnings

    client.table('monitor_status').upsert({
        'user_id': user_id,
        'desired_state': 'stopped',  # Always stopped after migration
        'status': status,
        'engine_state': remapped_engine_state,
        'updated_at': now,
    }, on_conflict='user_id').execute()
    print(f"  Monitor status: upserted (desired_state=stopped, {len(new_positions)} positions)")

    return {'found': 1, 'inserted': 1, 'skipped': 0}, warnings


# ============================================================
# Summary Report
# ============================================================

def print_summary(all_stats, id_maps, all_warnings):
    """Print a formatted migration summary."""
    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)

    print(f"\n{'Entity':<30} {'Found':>6} {'Inserted':>9} {'Skipped':>8}")
    print("-" * 60)
    for name, stats in all_stats.items():
        print(f"{name:<30} {stats['found']:>6} {stats['inserted']:>9} {stats['skipped']:>8}")

    if id_maps.get('strategies'):
        print("\nStrategy ID mapping:")
        for old, new in sorted(id_maps['strategies'].items()):
            print(f"  {old} → {new}")

    if id_maps.get('portfolios'):
        print("\nPortfolio ID mapping:")
        for old, new in sorted(id_maps['portfolios'].items()):
            print(f"  {old} → {new}")

    if id_maps.get('requirements'):
        print("\nRequirement ID mapping:")
        for old, new in sorted(id_maps['requirements'].items()):
            print(f"  {old} → {new}")

    if all_warnings:
        print(f"\nWarnings ({len(all_warnings)}):")
        for w in all_warnings:
            print(f"  - {w}")

    print("\n" + "=" * 60)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Migrate RoR Trader JSON data to Supabase")
    parser.add_argument("user_id", help="Supabase auth user UUID to assign data to")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without inserting")
    parser.add_argument("--force", action="store_true", help="Proceed even if data already exists for this user")
    parser.add_argument("--skip-alerts", action="store_true", help="Skip migrating alerts (large dataset)")
    args = parser.parse_args()

    user_id = args.user_id

    # Validate environment
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env")
        sys.exit(1)

    # Create admin client
    from supabase import create_client
    client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

    # Validate user exists
    try:
        user = client.auth.admin.get_user_by_id(user_id)
        print(f"User: {user.user.email} ({user_id})")
    except Exception as e:
        print(f"ERROR: User {user_id} not found: {e}")
        sys.exit(1)

    # Idempotency check
    existing = client.table('strategies').select('id', count='exact') \
        .eq('user_id', user_id).execute()
    if existing.data and not args.force:
        print(f"ERROR: User already has {len(existing.data)} strategies in the database.")
        print("Use --force to proceed anyway (will add data, not replace).")
        sys.exit(1)
    elif existing.data:
        print(f"WARNING: User already has {len(existing.data)} strategies. Proceeding with --force.")

    # Load JSON files
    print("\nLoading JSON files...")
    data = load_all_json()

    if args.dry_run:
        print("\n*** DRY RUN — no data will be inserted ***\n")

    all_stats = {}
    all_warnings = []
    id_maps = {}

    # Phase 1: Requirement sets
    print("\n--- Requirement Sets ---")
    req_id_map, stats = migrate_requirement_sets(client, data['requirements'], user_id, args.dry_run)
    all_stats['Requirement Sets'] = stats
    id_maps['requirements'] = req_id_map

    # Phase 2: Strategies
    print("\n--- Strategies ---")
    strategy_id_map, stats = migrate_strategies(client, data['strategies'], user_id, args.dry_run)
    all_stats['Strategies'] = stats
    id_maps['strategies'] = strategy_id_map

    # Phase 3: Portfolios
    print("\n--- Portfolios ---")
    portfolio_id_map, stats, warnings = migrate_portfolios(
        client, data['portfolios'], user_id, strategy_id_map, req_id_map, args.dry_run)
    all_stats['Portfolios'] = stats
    id_maps['portfolios'] = portfolio_id_map
    all_warnings.extend(warnings)

    # Phase 4: Webhook templates
    print("\n--- Webhook Templates ---")
    stats = migrate_webhook_templates(client, data['webhook_templates'], args.dry_run)
    all_stats['Webhook Templates'] = stats

    # Phase 5: Alerts
    if args.skip_alerts:
        print("\n--- Alerts [SKIPPED] ---")
        all_stats['Alerts'] = {'found': len(data['alerts']), 'inserted': 0, 'skipped': len(data['alerts'])}
    else:
        print("\n--- Alerts ---")
        stats, warnings = migrate_alerts(
            client, data['alerts'], user_id, strategy_id_map, portfolio_id_map, args.dry_run)
        all_stats['Alerts'] = stats
        all_warnings.extend(warnings[:10])  # Cap verbose orphan warnings
        if len(warnings) > 10:
            all_warnings.append(f"  ... and {len(warnings) - 10} more alert warnings")

    # Phase 6: Alert config
    print("\n--- Alert Config ---")
    stats, warnings = migrate_alert_config(
        client, data['alert_config'], user_id, strategy_id_map, portfolio_id_map, args.dry_run)
    all_stats['Alert Config'] = stats
    all_warnings.extend(warnings)

    # Phase 7: User configs
    print("\n--- User Configs ---")
    config_stats = migrate_user_configs(client, data, user_id, args.dry_run)
    all_stats.update({f"  {k}": v for k, v in config_stats.items()})

    # Phase 8: Monitor status + engine state
    print("\n--- Monitor Status & Engine State ---")
    stats, warnings = migrate_monitor_state(
        client, data['monitor_status'], data['engine_state'], user_id, strategy_id_map, args.dry_run)
    all_stats['Monitor/Engine State'] = stats
    all_warnings.extend(warnings)

    # Summary
    print_summary(all_stats, id_maps, all_warnings)

    if args.dry_run:
        print("*** DRY RUN complete — no data was inserted ***")
    else:
        print("Migration complete!")


if __name__ == "__main__":
    main()
