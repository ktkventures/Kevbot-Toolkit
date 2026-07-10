"""Synthetic-df test of find_best_combinations required_groups semantics.

Run:  .venv/bin/python src/test_mass_required_confluences.py
No DB / network — pure mask math on a fabricated trades_df.
"""
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from services import find_best_combinations

# 40 trades; records:
#   GEN-WINDOW-IN        on trades 0..19  (the "10:00-11:30" style requirement)
#   15Min-EMA-SML        on trades 0..29
#   1Day-EMA-SML         on trades 10..39 (same state, other TF -> OR alternative)
#   15Min-MACD-BULL      on trades 0..9
#   GEN-RARE-IN          on trades 0..3   (too thin under min_trades=5)
rows = []
for i in range(40):
    recs = set()
    if i < 20:
        recs.add("GEN-WINDOW-IN")
    if i < 30:
        recs.add("15Min-EMA-SML")
    if i >= 10:
        recs.add("1Day-EMA-SML")
    if i < 10:
        recs.add("15Min-MACD-BULL")
    if i < 4:
        recs.add("GEN-RARE-IN")
    rows.append({
        "confluence_records": recs,
        "win": i % 2 == 0,
        "r_multiple": 1.0 if i % 2 == 0 else -0.5,
        "entry_time": pd.Timestamp("2026-06-01") + pd.Timedelta(days=i),
        "exit_time": pd.Timestamp("2026-06-01") + pd.Timedelta(days=i, hours=1),
        "pnl": 100.0 if i % 2 == 0 else -50.0,
    })
df = pd.DataFrame(rows)

fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


# ── 1. No required: unchanged behavior (depth>=1, no 'required' leakage) ──
base = find_best_combinations(df, max_depth=2, min_trades=5, top_n=50)
check("1a no-required: rows exist", len(base) > 0)
check("1b no-required: no depth-0 rows", all(r["depth"] >= 1 for r in base))
check("1c no-required: required field empty", all(r["required"] == [] for r in base))

# ── 2. Required singleton (GEN) ──
req = find_best_combinations(
    df, max_depth=2, min_trades=5, top_n=50,
    required_groups=[{"GEN-WINDOW-IN"}])
check("2a required in every combination",
      all("GEN-WINDOW-IN" in r["combination"] for r in req))
check("2b depth-0 baseline row present",
      any(r["combination"] == ["GEN-WINDOW-IN"] for r in req))
check("2c required field stamped",
      all(r["required"] == ["GEN-WINDOW-IN"] for r in req))
# 1Day-EMA-SML has only 10 trades inside the required window (i in 10..19)
# -> still >= min_trades 5, allowed. GEN-RARE-IN has 4 -> excluded.
check("2d thin-under-baseline record excluded",
      not any("GEN-RARE-IN" in r["combination"] for r in req))

# ── 3. OR alternatives (suffix label matches two TF variants) ──
alt = find_best_combinations(
    df, max_depth=1, min_trades=5, top_n=50,
    required_groups=[{"EMA-SML"}])
combos = [tuple(r["combination"]) for r in alt]
check("3a both TF variants appear as baselines",
      any(r["required"] == ["15Min-EMA-SML"] for r in alt)
      and any(r["required"] == ["1Day-EMA-SML"] for r in alt))
check("3b no combo mixes without both being real",
      all(("15Min-EMA-SML" in c) or ("1Day-EMA-SML" in c) for c in combos))

# ── 4. Required matches nothing -> loud empty ──
none_ = find_best_combinations(
    df, max_depth=2, min_trades=5, top_n=50,
    required_groups=[{"GEN-NEVER-HAPPENED"}])
check("4 unmatched required -> []", none_ == [])

# ── 5. Required-only search (empty optional pool) ──
only = find_best_combinations(
    df, max_depth=2, min_trades=5, top_n=50,
    allowed_labels=set(),  # empty set = no optional pool (NOT explore-all)
    required_groups=[{"GEN-WINDOW-IN"}])
check("5a exactly the depth-0 row", [r["combination"] for r in only] == [["GEN-WINDOW-IN"]])

# ── 6. allowed_labels=None still explores all (legacy) ──
legacy = find_best_combinations(df, max_depth=1, min_trades=5, top_n=50,
                                allowed_labels=None)
check("6 legacy None explore-all unchanged", len(legacy) >= 3)

# ── 7. KPI sanity: depth-0 required row counts only in-window trades ──
d0 = [r for r in req if r["combination"] == ["GEN-WINDOW-IN"]][0]
check("7 depth-0 trade count = 20", d0["total_trades"] == 20)

print("\n%d failures" % len(fails))
sys.exit(1 if fails else 0)
