"""Trade-level write-through fidelity diff — the §6 #3 gate for BAR_CACHE_ENABLED.

Runs a canary strategy's backtest reading NATIVE Polygon (BAR_CACHE_ENABLED off)
vs reading bar_cache / the write-through stream (on), and diffs the resulting
trades FIELD-BY-FIELD (entry/exit fill ts = identity key; r/prices/reason/pnl/
direction/exec_type = value). This is the evidence gate for flipping
BAR_CACHE_ENABLED on the fleet: a backtest reading the unified write-through cache
must produce byte-identical trades to one reading native Polygon.

Unlike the suite's canary check (entry-ts set + count, tol=0), this compares the
FULL trade record so an exit-price / r / reason shift can't hide behind matching
entry timestamps. Mirrors trade_snapshot.py's KEY/VAL field model.

Usage:  cd <RoR_Trader> && PYTHONPATH=src .venv/bin/python src/_writethrough_trade_diff.py [sid ...]
Default canary: 267 (north-star) + 338 (coarse-secondary). Saves JSON evidence.
"""
import os
import sys
import json
import importlib
import warnings

warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv("src/.env", override=True)
os.environ["USE_DB"] = "true"
os.environ.setdefault("BAR_CACHE_PAGE_SIZE", "50000")

ADMIN_USER = "19d47e46-f718-49a6-af32-5f5407f5b170"
from db import set_admin_user_context, get_strategy_by_id_admin  # noqa: E402
set_admin_user_context(ADMIN_USER)
import pack_registry  # noqa: E402
pack_registry.scan_and_load_all()

VAL_FIELDS = ("r_multiple", "entry_price", "exit_price", "exit_reason",
              "dollar_pnl", "direction", "exec_type")


def _capture(sid, cache_on):
    """Backtest sid under a clean prod-like baseline, toggling ONLY BAR_CACHE_ENABLED.
    Returns a keyed dict {entry|exit-ts: {val fields as str}}."""
    os.environ["RORT_RIGHTSIZE_WARMUP"] = "1"  # prod baseline
    if cache_on:
        os.environ["BAR_CACHE_ENABLED"] = "1"
    else:
        os.environ.pop("BAR_CACHE_ENABLED", None)
    import data_loader, services, bar_cache
    importlib.reload(bar_cache)
    importlib.reload(data_loader)
    importlib.reload(services)
    if hasattr(data_loader, "clear_1s_cache"):
        data_loader.clear_1s_cache()
    res = services.get_strategy_trades(get_strategy_by_id_admin(sid, ADMIN_USER))
    df = res[0] if isinstance(res, tuple) else res
    if df is None or len(df) == 0:
        return {}
    import pandas as pd
    kcols = ("entry_fill_ts", "exit_fill_ts") if "entry_fill_ts" in df.columns \
        else ("entry_time", "exit_time")
    exit_col = kcols[1]
    keyed = {}
    for _, row in df.iterrows():
        # Skip OPEN positions (no exit) — their r_multiple is mark-to-market against
        # the current price, so two runs seconds apart during live RTH legitimately
        # differ. The settled-window fidelity gate compares only CLOSED trades.
        if pd.isna(row.get(exit_col)):
            continue
        k = "|".join(str(row.get(c)) for c in kcols)
        keyed[k] = {f: (str(row.get(f)) if f in df.columns else None) for f in VAL_FIELDS}
    return keyed


def _diff(before, after):
    bk, ak = set(before), set(after)
    added = sorted(ak - bk)
    removed = sorted(bk - ak)
    changed = [k for k in (bk & ak) if before[k] != after[k]]
    return added, removed, changed


def main():
    sids = [int(x) for x in sys.argv[1:]] or [267, 338]
    report = {}
    any_divergent = False
    for sid in sids:
        off = _capture(sid, False)   # native Polygon
        on = _capture(sid, True)     # bar_cache / write-through
        added, removed, changed = _diff(off, on)
        identical = not (added or removed or changed)
        any_divergent |= (not identical)
        report[str(sid)] = {
            "n_native": len(off), "n_cache": len(on),
            "added": len(added), "removed": len(removed), "changed": len(changed),
            "identical": identical,
            "sample_changed": [
                {"key": k, "diff": {f: [off[k].get(f), on[k].get(f)]
                                    for f in VAL_FIELDS if off[k].get(f) != on[k].get(f)}}
                for k in changed[:5]],
            "sample_added": added[:5], "sample_removed": removed[:5],
        }
        flag = "✅ IDENTICAL" if identical else "❌ DIVERGENT"
        print(f"sid {sid}: native={len(off)} cache={len(on)} "
              f"added={len(added)} removed={len(removed)} changed={len(changed)}  {flag}")
        for k in changed[:5]:
            d = ", ".join(f"{f}:{off[k].get(f)}→{on[k].get(f)}"
                          for f in VAL_FIELDS if off[k].get(f) != on[k].get(f))
            print(f"   changed {k}: {d}")
        if added[:3]:
            print(f"   cache-only (added): {added[:3]}")
        if removed[:3]:
            print(f"   native-only (removed): {removed[:3]}")

    out = os.path.join(os.environ.get("SCRATCH", "/tmp"), "writethrough_trade_diff_0629.json")
    try:
        json.dump(report, open(out, "w"), indent=2, default=str)
        print(f"\nsaved {out}")
    except Exception as ex:  # noqa: BLE001
        print(f"(could not save report: {ex})")
    print("RESULT:", "❌ DIVERGENT — do NOT flip BAR_CACHE_ENABLED" if any_divergent
          else "✅ trade-level identical — BAR_CACHE_ENABLED safe for these canaries")
    return 1 if any_divergent else 0


if __name__ == "__main__":
    sys.exit(main())
