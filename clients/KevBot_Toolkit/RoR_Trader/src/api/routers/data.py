"""Data router — market data bars and data source info."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query

from api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/data", tags=["data"])


@router.get("/bars/{symbol}")
def get_bars(
    symbol: str,
    timeframe: str = Query("1Min"),
    days: int = Query(30),
    session: str = Query("RTH"),
    feed: str = Query("sip"),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    user=Depends(get_current_user),
):
    """Load OHLCV bars for a symbol.

    Returns array of {timestamp, open, high, low, close, volume}.
    """
    from datetime import datetime
    from data_loader import load_market_data

    sd = datetime.fromisoformat(start_date) if start_date else None
    ed = datetime.fromisoformat(end_date) if end_date else None

    df = load_market_data(
        symbol, days=days, timeframe=timeframe,
        start_date=sd, end_date=ed, feed=feed, session=session,
    )

    if len(df) == 0:
        return []

    # Serialize to JSON-friendly list
    records = []
    for ts, row in df.iterrows():
        records.append({
            "timestamp": ts.isoformat(),
            "open": round(float(row["open"]), 4),
            "high": round(float(row["high"]), 4),
            "low": round(float(row["low"]), 4),
            "close": round(float(row["close"]), 4),
            "volume": int(row.get("volume", 0)),
        })
    return records


@router.get("/source")
def get_source(user=Depends(get_current_user)):
    """Get the name of the current data provider."""
    from data_loader import get_data_source
    return {"source": get_data_source()}


@router.get("/timeframes")
def get_timeframes(user=Depends(get_current_user)):
    """Get available timeframes and their labels."""
    from data_loader import TF_LABELS, MTF_AVAILABLE_TIMEFRAMES, SUB_MINUTE_TIMEFRAMES
    return {
        "available": MTF_AVAILABLE_TIMEFRAMES,
        "labels": TF_LABELS,
        "sub_minute": list(SUB_MINUTE_TIMEFRAMES),
    }
