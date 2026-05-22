"""`BarStoreFacade` — routes timeframe reads between the data-worker's
Tier-1 (`SymbolBarStore`, 90-min 1-second) and Tier-2 (`CoarseBarStore`,
150-day 1-Minute) stores by TF granularity.

A drop-in replacement for `SymbolBarStore` from `build_injected_frames`'
point of view: it exposes the same `get_timeframe(tf)` and `coverage()`
API. The streaming engine never sees which tier served a TF.

Routing (single source of truth):

  TIMEFRAME_SECONDS[tf] <  120s  →  Tier-1 (1s store, resample-on-demand)
  TIMEFRAME_SECONDS[tf] >= 120s  →  Tier-2 (1Min store, resample-on-demand)

i.e. {1Sec, 5Sec, 10Sec, 15Sec, 30Sec, 1Min} → Tier-1; everything from
2Min upward (5/10/15/30Min, 1Hour, 1Day) → Tier-2. The 120-second
boundary cleanly splits "fast primary path" from "long-warmup secondary
path."

`coverage()` reports the **Tier-1** span: the streaming tick's
resume-window store-gap check is about the primary bar series. Tier-2
freshness is exposed separately via the metrics line.
"""
from __future__ import annotations

import pandas as pd


_TIER2_THRESHOLD_SECONDS = 120  # TFs >= 2 minutes route to Tier-2.


def _route_to_tier2(tf: str) -> bool:
    """True if `tf` should be served by Tier-2 (≥ 120s)."""
    try:
        from unified_engine import TIMEFRAME_SECONDS
        secs = int(TIMEFRAME_SECONDS.get(tf, 60))
    except Exception:
        return False
    return secs >= _TIER2_THRESHOLD_SECONDS


class BarStoreFacade:
    """Drop-in replacement for SymbolBarStore that dispatches across tiers."""

    def __init__(self, tier1, tier2):
        # `tier1` and `tier2` are duck-typed: both must expose
        # `get_timeframe(tf)` and `coverage()`. In production they are
        # `SymbolBarStore` and `CoarseBarStore`; tests can inject fakes.
        self.tier1 = tier1
        self.tier2 = tier2
        # Convenience accessor — most callers want the symbol; both tiers
        # carry it, prefer tier1 (always present).
        self.symbol = getattr(tier1, 'symbol',
                              getattr(tier2, 'symbol', None))

    def get_timeframe(self, tf: str) -> pd.DataFrame:
        """Return resampled bars at `tf` from whichever tier serves it."""
        if _route_to_tier2(tf):
            return self.tier2.get_timeframe(tf)
        return self.tier1.get_timeframe(tf)

    def coverage(self) -> dict:
        """Tier-1 coverage — the streaming tick's resume-window gate."""
        return self.tier1.coverage()
