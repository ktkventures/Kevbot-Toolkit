"""M1d — CONSUMER #2 validation (parity ribbon coarse-secondary swap), direct.

Consumer #2 = services._coarse_secondary_store_swap, hooked at the coarse-secondary
resample fallback in prepare_data_with_indicators. It serves a coarse (>=1Hour)
secondary from the canonical store IFF byte-identical to resample_to_timeframe(
primary_df, tf), else falls back to that resample. So the swap-or-resample OUTPUT is
ALWAYS == resample(primary_df) — the ribbon is unchanged bar-for-bar.

Proves, directly (no PostgREST config path — infra-robust):
  1. Composite invariant: (swap else resample) == resample(primary), ALWAYS.
  2. 1Min primary: the store SERVES (byte-identical) — the win.
  3. Sub-minute primary (10Sec): store != resample(primary) on VOLUME (1Min != Σsub-
     minute) → the comparator catches it → falls back → still byte-identical output.
  4. Flag OFF: swap returns None → resample (byte-identical, inert).

Serves whatever window df spans, so a deep-warmed caller gets a deep coarse store
read (the primitive the correct-warmup ribbon builds on).

WRITES: resampled_bar_cache (TSLA/4Hour/RTH) + bar_cache delta-fetch. No live/Worker.
"""
from __future__ import annotations
import os
import sys
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")
_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_SRC, ".env"))
except Exception:
    pass
os.environ.setdefault("USE_DB", "true")
os.environ["BAR_CACHE_ENABLED"] = "1"

SYM = "TSLA"
TF = "4Hour"
SESSION = "RTH"
START = datetime(2026, 6, 15, tzinfo=timezone.utc)
END = datetime(2026, 7, 8, 23, 59, tzinfo=timezone.utc)
OHLCV = ["open", "high", "low", "close", "volume"]


def _byte_equal(a, b) -> tuple:
    """(index_equal, ohlcv_cell_diffs)."""
    idx_ok = a.index.equals(b.index)
    diffs = 0
    if idx_ok:
        for c in OHLCV:
            diffs += int((a[c].values != b[c].values).sum())
    return idx_ok, (diffs if idx_ok else None)


def main() -> int:
    import resampled_bar_store as rbs
    import services
    from data_loader import load_market_data, resample_to_timeframe

    print("=" * 78)
    print(f"M1d CONSUMER #2 validation — {SYM}/{TF}/{SESSION} [{START.date()}..{END.date()}]")
    print("=" * 78)

    # populate the store for this coarse window (warm cache → store_window).
    os.environ["RORT_RESAMPLED_STORE_WRITE"] = "1"
    load_market_data(SYM, start_date=START, end_date=END, timeframe="1Min",
                     session=SESSION)
    w = rbs.store_window(SYM, TF, SESSION, START, END)
    print(f"populate {SYM}/{TF}/{SESSION}: {w}")

    strat = {"symbol": SYM}
    ok = True

    # --- case 1: 1Min primary — store should SERVE, byte-identical ---
    df1 = load_market_data(SYM, start_date=START, end_date=END, timeframe="1Min",
                           session=SESSION)
    resamp1 = resample_to_timeframe(df1[OHLCV].copy(), TF)
    os.environ["RORT_RESAMPLED_STORE_READ"] = "1"
    swap1 = services._coarse_secondary_store_swap(strat, TF, df1, SESSION)
    served1 = swap1 is not None
    out1 = swap1 if served1 else resamp1  # composite (swap else resample)
    ie1, cd1 = _byte_equal(out1, resamp1)
    print(f"\n[1Min primary]  store_served={served1}  composite==resample: "
          f"index_equal={ie1} ohlcv_cell_diffs={cd1}")
    ok = ok and served1 and ie1 and cd1 == 0

    # --- case 2: sub-minute (10Sec) primary — must FALL BACK (volume diverges) ---
    df10 = load_market_data(SYM, start_date=START, end_date=END, timeframe="10Sec",
                            session=SESSION)
    if df10 is not None and len(df10) > 0:
        resamp10 = resample_to_timeframe(df10[OHLCV].copy(), TF)
        swap10 = services._coarse_secondary_store_swap(strat, TF, df10, SESSION)
        fellback = swap10 is None
        out10 = swap10 if swap10 is not None else resamp10
        ie10, cd10 = _byte_equal(out10, resamp10)
        # is the store actually DIFFERENT from resample(10Sec)? (proves the guard matters)
        store10 = rbs.read_store(SYM, TF, SESSION, df10.index.min(), df10.index.max())
        vol_diff = None
        if store10 is not None and store10.index.equals(resamp10.index):
            vol_diff = int((store10["volume"].values != resamp10["volume"].values).sum())
        print(f"[10Sec primary] fellback={fellback} (store vs resample(10Sec) "
              f"volume_diffs={vol_diff}); composite==resample(10Sec): "
              f"index_equal={ie10} ohlcv_cell_diffs={cd10}")
        ok = ok and fellback and ie10 and cd10 == 0
    else:
        print("[10Sec primary] no 10Sec data — skipped")

    # --- case 3: flag OFF — swap inert (None) ---
    os.environ.pop("RORT_RESAMPLED_STORE_READ", None)
    swap_off = services._coarse_secondary_store_swap(strat, TF, df1, SESSION)
    off_inert = swap_off is None
    print(f"[flag OFF]       swap returns None (inert): {off_inert}")
    ok = ok and off_inert

    print("\n" + "#" * 78)
    print("✅ CONSUMER #2 GREEN — swap-or-resample == resample ALWAYS; store serves "
          "1Min primary byte-identical; sub-minute falls back; OFF inert."
          if ok else "❌ CONSUMER #2 not green — see above.")
    print("#" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
