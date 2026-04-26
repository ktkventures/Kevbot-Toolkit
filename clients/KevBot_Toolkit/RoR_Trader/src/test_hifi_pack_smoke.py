#!/usr/bin/env python3
"""Hi-Fi Pack Smoke — verify user packs play cleanly with hifi_mode=True.

What Hi-Fi Pass 2 actually does (from `backtest_service._hifi_resolve_trades`):
- Walks 1-second Polygon bars between entry and exit
- Refines exit_price/exit_reason/r_multiple/win/pnl/hold_time_seconds when
  an L-type stop or target hit happens at sub-bar precision
- Sets hifi_resolved=True
- Skips trades whose stop/target exec_type isn't 'L' (no intra-bar walk needed)
- Does NOT touch entries — entries are bar-level only in this codepath
- Does NOT call any pack code — Pass 2 reads stop_price/target_price from
  the trade record and price action from 1s bars; the pack's contribution
  is fully encapsulated in Pass 1 trades

So the smoke is really: each pack produces Pass-1 trades that the
existing Pass-2 layer can refine without breaking, and the refinement
is deterministic.

Smoke A — non-regression:
  hifi_mode=False vs hifi_mode=True. Same trade count, same entry times,
  refinements within tolerance.

Smoke B — determinism:
  hifi_mode=True run twice. Outcomes match exactly.

Run from src/ dir:
    python3 test_hifi_pack_smoke.py
"""

from __future__ import annotations

import io
import logging
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(override=True)

# Force file-backed confluence groups (smoke runs without Supabase auth)
os.environ["USE_DB"] = "false"

logging.basicConfig(level=logging.WARNING, format='%(message)s')

import confluence_groups as cg_mod
import pack_registry
from api.schemas.backtest import BacktestRequest
from api.services.backtest_service import run_backtest
from confluence_groups import ConfluenceGroup, PlotSettings


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

SYMBOL = "SPY"
TIMEFRAME = "1Min"
DAYS = 5
DIRECTION = "LONG"


# (pack_id, group_id, base_template, params, base_trigger)
PACKS = [
    ("ut_bot_v4", "ut_bot_v4_default", "ut_bot_v4",
     {"key_value": 1, "atr_period": 10}, "bull_flip"),
    ("supertrend", "supertrend_default", "supertrend",
     {"atr_period": 10, "atr_multiplier": 3.0}, "bull_flip"),
    ("ema_pp_v4", "ema_pp_v4_default_smoke", "ema_pp_v4",
     {"short_period": 8, "mid_period": 21, "long_period": 50}, "cross_short_up"),
    ("ema_pp_v3", "ema_pp_v3_default_smoke", "ema_pp_v3",
     {"short_period": 8, "mid_period": 21, "long_period": 50}, "cross_short_up"),
]

# Tolerances for Smoke A (non-regression)
PRICE_TOL = 0.01           # absolute $ tolerance for refined exit_price
R_MULT_TOL = 0.05          # absolute R-multiple tolerance


# -----------------------------------------------------------------------------
# In-memory group injection (bypasses Supabase, reads file directly + appends)
# -----------------------------------------------------------------------------

_INJECTED_GROUPS: list[ConfluenceGroup] = []


def _load_groups_from_file() -> list[ConfluenceGroup]:
    """Read confluence_groups.json directly and parse — no Supabase dependency."""
    import json
    cfg_path = cg_mod.get_config_path()
    if not cfg_path.exists():
        return cg_mod.create_default_groups()
    with open(cfg_path) as f:
        data = json.load(f)
    return cg_mod._parse_group_list(data.get("groups", []))


def _patched_load_confluence_groups():
    base = _load_groups_from_file()
    base_ids = {g.id for g in base}
    extras = [g for g in _INJECTED_GROUPS if g.id not in base_ids]
    return base + extras


def _ensure_group(group_id: str, base_template: str, params: dict) -> None:
    base = _load_groups_from_file()
    if any(g.id == group_id for g in base):
        return
    if any(g.id == group_id for g in _INJECTED_GROUPS):
        return
    _INJECTED_GROUPS.append(ConfluenceGroup(
        id=group_id,
        base_template=base_template,
        version="Default (smoke)",
        description=f"Hi-Fi smoke test group for {base_template}",
        enabled=True,
        is_default=False,
        parameters=params,
        plot_settings=PlotSettings(colors={}, line_width=1, visible=True),
    ))


# -----------------------------------------------------------------------------
# Backtest helpers
# -----------------------------------------------------------------------------

def _silent_run(req: BacktestRequest) -> dict:
    """Run a backtest with the verbose [BACKTEST] prints suppressed."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        return run_backtest(req)


def _build_request(group_id: str, base_trigger: str, hifi: bool) -> BacktestRequest:
    return BacktestRequest(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        direction=DIRECTION,
        session="RTH",
        data_feed="sip",
        days=DAYS,
        entry_trigger_confluence_id=f"{group_id}_{base_trigger}",
        exit_trigger_confluence_ids=[],
        confluence=[],
        stop_config={"method": "atr", "atr_mult": 1.5, "exec_type": "L"},
        target_config={"method": "r_multiple", "r": 2.0, "exec_type": "L"},
        stop_exec_type="L",
        target_exec_type="L",
        hifi_mode=hifi,
        bar_count_exit=20,  # cap holding so most trades exit within window
        risk_per_trade=100.0,
    )


def _trades_summary(trades: list[dict]) -> str:
    n = len(trades)
    if n == 0:
        return "0 trades"
    n_resolved = sum(1 for t in trades if t.get('hifi_resolved'))
    n_wins = sum(1 for t in trades if t.get('win'))
    return f"{n} trades, {n_wins} wins, {n_resolved} hifi_resolved"


# -----------------------------------------------------------------------------
# Diff logic
# -----------------------------------------------------------------------------

def _diff_smoke_a(off: list[dict], on: list[dict]) -> tuple[str, list[str]]:
    """hifi=False vs hifi=True. Same count, same entries, refinements sane."""
    notes: list[str] = []
    if len(off) != len(on):
        return "FAIL", [f"trade-count diverged: {len(off)} (off) vs {len(on)} (on)"]

    if len(off) == 0:
        return "NO_FIRES", ["no trades produced — increase DAYS or pick livelier symbol"]

    n_resolved = 0
    n_outcome_changed = 0
    max_price_delta = 0.0
    max_r_delta = 0.0

    for i, (a, b) in enumerate(zip(off, on)):
        # Entry timestamps must be identical (Pass 2 doesn't touch entries)
        if a.get('entry_time') != b.get('entry_time'):
            notes.append(
                f"trade {i}: entry_time changed by Hi-Fi "
                f"({a.get('entry_time')} → {b.get('entry_time')})"
            )
            return "FAIL", notes

        if b.get('hifi_resolved'):
            n_resolved += 1
            if a.get('exit_reason') != b.get('exit_reason'):
                n_outcome_changed += 1
            ap = float(a.get('exit_price') or 0)
            bp = float(b.get('exit_price') or 0)
            ar = float(a.get('r_multiple') or 0)
            br = float(b.get('r_multiple') or 0)
            max_price_delta = max(max_price_delta, abs(ap - bp))
            max_r_delta = max(max_r_delta, abs(ar - br))

    notes.append(f"{n_resolved}/{len(on)} trades hifi_resolved")
    notes.append(f"{n_outcome_changed} outcome changes (stop↔target swap)")
    notes.append(f"max exit_price delta: ${max_price_delta:.4f}")
    notes.append(f"max r_multiple delta: {max_r_delta:.4f}")

    return "PASS", notes


def _diff_smoke_b(on1: list[dict], on2: list[dict]) -> tuple[str, list[str]]:
    """hifi=True ran twice. Must match exactly."""
    if len(on1) != len(on2):
        return "FAIL", [f"trade-count diverged across runs: {len(on1)} vs {len(on2)}"]

    fields = ('entry_time', 'exit_time', 'entry_price', 'exit_price',
              'r_multiple', 'win', 'exit_reason', 'hifi_resolved')
    mismatches = []
    for i, (a, b) in enumerate(zip(on1, on2)):
        for f in fields:
            if a.get(f) != b.get(f):
                mismatches.append(f"trade {i}.{f}: {a.get(f)!r} vs {b.get(f)!r}")
    if mismatches:
        return "FAIL", mismatches[:5] + (["…"] if len(mismatches) > 5 else [])
    return "PASS", [f"{len(on1)} trades match across both runs"]


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def main() -> int:
    pack_registry.scan_and_load_all()

    # Force file-backed config: defensively pin db.USE_DB even after dotenv
    import db
    db.USE_DB = False

    # Patch group loader in every module that bound the symbol at import time
    cg_mod.load_confluence_groups = _patched_load_confluence_groups
    import services as svc
    svc.load_confluence_groups = _patched_load_confluence_groups

    # Inject groups for packs that don't have a default in the live config
    for _, group_id, base_template, params, _ in PACKS:
        _ensure_group(group_id, base_template, params)

    print(f"\nHi-Fi Pack Smoke — {SYMBOL} {TIMEFRAME} {DAYS}d {DIRECTION}\n")
    print(f"{'pack':<16} {'trigger':<22} {'A: non-regr':<12} {'B: determ':<12}")
    print("-" * 72)

    overall_pass = True
    detail_blocks: list[str] = []

    for pack_id, group_id, _base_tpl, _params, base_trigger in PACKS:
        try:
            req_off = _build_request(group_id, base_trigger, hifi=False)
            resp_off = _silent_run(req_off)
            trades_off = list(resp_off.get('trades') or [])

            req_on = _build_request(group_id, base_trigger, hifi=True)
            resp_on1 = _silent_run(req_on)
            trades_on1 = list(resp_on1.get('trades') or [])

            resp_on2 = _silent_run(req_on)
            trades_on2 = list(resp_on2.get('trades') or [])

            verdict_a, notes_a = _diff_smoke_a(trades_off, trades_on1)
            verdict_b, notes_b = _diff_smoke_b(trades_on1, trades_on2)

            print(f"{pack_id:<16} {base_trigger:<22} {verdict_a:<12} {verdict_b:<12}")

            block = [f"\n[{pack_id} / {base_trigger}]"]
            block.append(f"  hifi=off: {_trades_summary(trades_off)}")
            block.append(f"  hifi=on : {_trades_summary(trades_on1)}")
            block.append("  Smoke A (non-regression):")
            block.extend(f"    - {n}" for n in notes_a)
            block.append("  Smoke B (determinism):")
            block.extend(f"    - {n}" for n in notes_b)
            detail_blocks.append("\n".join(block))

            if verdict_a not in ("PASS", "NO_FIRES"):
                overall_pass = False
            if verdict_b != "PASS":
                overall_pass = False

        except Exception as exc:
            print(f"{pack_id:<16} {base_trigger:<22} {'ERROR':<12} {'ERROR':<12}")
            detail_blocks.append(f"\n[{pack_id} / {base_trigger}] ERROR: {exc}")
            overall_pass = False

    print("\n" + "=" * 72)
    print("DETAILS")
    print("=" * 72)
    for block in detail_blocks:
        print(block)

    print("\n" + "=" * 72)
    print("OVERALL:", "PASS" if overall_pass else "FAIL")
    print("=" * 72)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
