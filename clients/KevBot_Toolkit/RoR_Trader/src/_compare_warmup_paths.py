"""Diagnostic: trade-count + KPI parity across data-loading paths
after the unified warmup helper rollout.

Given a strategy ID, runs:
  - get_strategy_trades (path #1 — used by Strategy Detail page)
  - the prepare_forward_test_data flow indirectly via path #1
  - a Mass-Builder synth (path #3 — what the search would see)
  - the new load_strategy_data helper directly (oracle)

Reports trade counts, KPI deltas, and the resolved visible window
each path used.

Also supports `--warmup-multiplier N` to test the WARMUP_MULTIPLIER
constant — useful for validating that we can drop the `*2` floor
later without trade-count drift.

Usage:
    .venv/bin/python src/_compare_warmup_paths.py --strategy-id 248
    .venv/bin/python src/_compare_warmup_paths.py --strategy-id 248 --warmup-multiplier 1
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv(override=True)

import pack_registry
pack_registry.scan_and_load_all()

import pandas as pd

import strategy_data
from db import get_admin_client, set_admin_user_context
from strategy_data import (
    load_strategy_data,
    trim_trades_to_visible,
    resolve_visible_window,
)


def _load_strategy(sid: int) -> dict:
    c = get_admin_client()
    r = c.table("strategies").select("*").eq("id", sid).single().execute()
    row = r.data or {}
    cfg = row.get("config") or {}
    if isinstance(cfg, str):
        cfg = json.loads(cfg)
    return {**row, **cfg}


def _summary(trades: pd.DataFrame) -> dict:
    if trades is None or len(trades) == 0:
        return {"count": 0}
    # Total R + win rate are quick checks; not authoritative KPIs.
    out = {"count": len(trades)}
    if "r_multiple" in trades.columns:
        r = pd.to_numeric(trades["r_multiple"], errors="coerce").dropna()
        out["total_r"] = round(float(r.sum()), 2)
        wins = (r > 0).sum()
        out["win_rate"] = round(100.0 * wins / len(r), 1) if len(r) else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy-id", type=int, required=True)
    ap.add_argument("--warmup-multiplier", type=float, default=None,
                    help="Override WARMUP_MULTIPLIER (default: module constant)")
    args = ap.parse_args()

    if args.warmup_multiplier is not None:
        strategy_data.WARMUP_MULTIPLIER = args.warmup_multiplier
        print(f"[override] WARMUP_MULTIPLIER = {args.warmup_multiplier}")

    strat = _load_strategy(args.strategy_id)
    if not strat:
        print(f"strategy {args.strategy_id} not found")
        return 1

    set_admin_user_context(strat["user_id"])

    print(f"\n=== sid {args.strategy_id} — {strat.get('symbol')} "
          f"{strat.get('timeframe')} {strat.get('direction')} ===")
    print(f"backtest_start_date = {strat.get('backtest_start_date')}")
    print(f"forward_test_start  = {strat.get('forward_test_start')}")
    print(f"data_days           = {strat.get('data_days')}")
    vs, ve, src = resolve_visible_window(strat)
    print(f"resolved visible    = [{vs.isoformat()}, {ve.isoformat()}]")
    print(f"anchor_source       = {src}")
    print(f"WARMUP_MULTIPLIER   = {strategy_data.WARMUP_MULTIPLIER}")

    # ── Oracle: helper-direct load → engine → trim ───────────────────
    print("\n--- Oracle: load_strategy_data + run_unified_backtest + trim ---")
    from unified_engine import run_unified_backtest
    import general_packs as gp_module
    bundle = load_strategy_data(strat)
    print(f"  bundle df bars: {len(bundle.df)} "
          f"(warmup_start={bundle.warmup_start.isoformat()})")
    enabled_gen = gp_module.get_enabled_general_packs(
        gp_module.load_general_packs())
    trades_full, _ = run_unified_backtest(
        bundle.df, strat,
        general_packs=enabled_gen,
        secondary_tf_map=bundle.secondary_tf_map or None,
    )
    trades_full = trim_trades_to_visible(trades_full, bundle.visible_start)
    print(f"  oracle (helper → engine → trim): {_summary(trades_full)}")

    # ── Path #1: services.get_strategy_trades ─────────────────────────
    # This is what the Strategy Detail page / KPI table calls today.
    from services import get_strategy_trades
    print("\n--- Path #1: services.get_strategy_trades (Strategy Detail) ---")
    trades_path1 = get_strategy_trades(strat)
    # Refactor target: path #1 will also trim post-migration.
    print(f"  raw (pre-migration): {_summary(trades_path1)}")
    if trades_path1 is not None and len(trades_path1) > 0:
        path1_trimmed = trim_trades_to_visible(trades_path1, bundle.visible_start)
        print(f"  trimmed to visible : {_summary(path1_trimmed)}")

    # ── Path #3: Mass-Builder-style synth ─────────────────────────────
    # Simulate what Mass Builder would produce with the new helper —
    # synth strat with backtest_start_date = now - data_days.
    print("\n--- Path #3 simulation: Mass Builder synth ---")
    synth = dict(strat)
    days = int(strat.get("data_days") or 90)
    synth["backtest_start_date"] = (pd.Timestamp.now(tz="UTC")
                                     - pd.Timedelta(days=days)).isoformat()
    synth.pop("forward_test_start", None)
    synth_bundle = load_strategy_data(synth)
    synth_trades, _ = run_unified_backtest(
        synth_bundle.df, synth,
        general_packs=enabled_gen,
        secondary_tf_map=synth_bundle.secondary_tf_map or None,
    )
    synth_trades = trim_trades_to_visible(synth_trades, synth_bundle.visible_start)
    print(f"  search-window synth: visible=[{synth_bundle.visible_start.isoformat()}, "
          f"{synth_bundle.visible_end.isoformat()}]")
    print(f"  result             : {_summary(synth_trades)}")

    # ── Verdict ───────────────────────────────────────────────────────
    print("\n=== Verdict ===")
    oracle_n = len(trades_full)
    p1_n = len(trim_trades_to_visible(trades_path1, bundle.visible_start)) \
        if trades_path1 is not None else 0
    if oracle_n == p1_n:
        print(f"  Path #1 (trimmed) matches oracle: {oracle_n} trades. ✓")
    else:
        print(f"  Path #1 (trimmed) = {p1_n}, oracle = {oracle_n}. DIFF: {p1_n - oracle_n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
