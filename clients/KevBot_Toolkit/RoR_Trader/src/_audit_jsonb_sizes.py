"""JSONB size audit — survey strategy config blob sizes across the DB.

Surfaces:
  - engine_snapshot_b64 size distribution (median/p95/max)
  - position_carryovers list length distribution
  - Top 5 largest strategy configs (by config JSONB byte size)
  - Any strategies with snapshots > 64KB (threshold for considering
    migration to a sibling table)

Read-only — never writes to DB. Uses admin client to scan all users.

Run: cd src && .venv/bin/python _audit_jsonb_sizes.py
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from dotenv import load_dotenv
load_dotenv(override=True)

from db import get_admin_client


def _bytes(s) -> int:
    if s is None:
        return 0
    if isinstance(s, str):
        return len(s.encode('utf-8'))
    return len(json.dumps(s, default=str).encode('utf-8'))


def _human(n: int) -> str:
    for unit in ['B', 'KB', 'MB']:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def _pct(values: list, p: float):
    if not values:
        return 0
    s = sorted(values)
    idx = min(int(len(s) * p), len(s) - 1)
    return s[idx]


def main():
    print("\n=== JSONB size audit — 2026-05-20 ===\n")
    client = get_admin_client()
    res = client.table('strategies') \
        .select('id, user_id, name, symbol, timeframe, config') \
        .execute()
    rows = res.data or []
    print(f"Scanned {len(rows)} strategies\n")

    snapshot_sizes: list = []
    config_sizes: list = []
    carryover_counts: list = []
    snapshot_present = 0

    largest_configs = []  # (size, id, name)
    largest_snapshots = []  # (size, id, name)

    for r in rows:
        cfg = r.get('config') or {}
        if isinstance(cfg, str):
            cfg = json.loads(cfg)

        config_size = _bytes(cfg)
        config_sizes.append(config_size)
        largest_configs.append((config_size, r['id'], r.get('name', '')))

        snap = cfg.get('engine_snapshot_b64')
        if snap:
            snapshot_present += 1
            s = _bytes(snap)
            snapshot_sizes.append(s)
            largest_snapshots.append((s, r['id'], r.get('name', '')))

        carry = cfg.get('position_carryovers') or []
        carryover_counts.append(len(carry))

    # === Snapshot stats ===
    print(f"Snapshots present:  {snapshot_present}/{len(rows)} "
          f"({100*snapshot_present/max(len(rows),1):.1f}%)")
    if snapshot_sizes:
        print(f"Snapshot sizes (b64, encoded):")
        print(f"   median = {_human(_pct(snapshot_sizes, 0.5))}")
        print(f"   p95    = {_human(_pct(snapshot_sizes, 0.95))}")
        print(f"   max    = {_human(max(snapshot_sizes))}")
        print(f"   total  = {_human(sum(snapshot_sizes))}")
        # Flag any snapshots > 64KB
        bloated = [s for s in snapshot_sizes if s > 64 * 1024]
        if bloated:
            print(f"   ⚠️  {len(bloated)} snapshot(s) > 64KB — consider "
                  f"migration to sibling table")
        else:
            print(f"   ✓ all under 64KB threshold")
    else:
        print(f"Snapshot sizes: NONE recorded yet "
              f"(LEF 2c snapshots populate as strategies refresh)")
    print()

    # === Carryover stats ===
    print(f"Position carryovers:")
    if any(c > 0 for c in carryover_counts):
        with_carry = [c for c in carryover_counts if c > 0]
        print(f"   strategies with carryovers: {len(with_carry)}")
        print(f"   median count: {_pct(with_carry, 0.5)}")
        print(f"   max count:    {max(with_carry)}")
        if max(with_carry) >= 50:
            print(f"   ⚠️  some strategies hit the 50-entry FIFO cap")
        else:
            print(f"   ✓ all under the 50-entry FIFO cap")
    else:
        print(f"   none — no Tier 3 carryovers recorded yet "
              f"(expected on first deploy)")
    print()

    # === Config blob sizes ===
    print(f"Config JSONB sizes:")
    print(f"   median = {_human(_pct(config_sizes, 0.5))}")
    print(f"   p95    = {_human(_pct(config_sizes, 0.95))}")
    print(f"   max    = {_human(max(config_sizes) if config_sizes else 0)}")
    print(f"   total  = {_human(sum(config_sizes))}")
    print()

    # === Top 5 largest configs ===
    print(f"Top 5 largest configs:")
    for size, sid, name in sorted(largest_configs, reverse=True)[:5]:
        print(f"   {_human(size):>10}  sid={sid:>4}  {name[:50]}")
    print()

    # === Top 5 largest snapshots ===
    if largest_snapshots:
        print(f"Top 5 largest snapshots:")
        for size, sid, name in sorted(largest_snapshots, reverse=True)[:5]:
            print(f"   {_human(size):>10}  sid={sid:>4}  {name[:50]}")
        print()

    # === Decision ===
    print("=== Recommendation ===")
    if not snapshot_sizes:
        print("⏳  Insufficient data — snapshots haven't populated yet.")
        print("    Re-run this audit after 24-48h of normal refresh activity.")
    elif max(snapshot_sizes) > 256 * 1024:
        print("🔴  At least one snapshot > 256KB. Strongly recommend")
        print("    migrating engine_snapshot_b64 to a sibling table to")
        print("    keep config JSONB lean for hot-path strategy loads.")
    elif _pct(snapshot_sizes, 0.95) > 64 * 1024:
        print("🟡  p95 snapshot > 64KB. Consider sibling-table migration")
        print("    if Snapshot persistence churn becomes a perf concern.")
    else:
        print("✓  All snapshot sizes within reasonable JSONB-column limits.")
        print("    No immediate action needed.")


if __name__ == '__main__':
    main()
