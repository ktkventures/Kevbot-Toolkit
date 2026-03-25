"""Backtest router — run backtests and compute KPIs."""

import logging

from fastapi import APIRouter, Depends

from api.deps import get_current_user
from api.schemas.backtest import BacktestRequest, BacktestResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


@router.post("/run", response_model=BacktestResponse)
def run_backtest(req: BacktestRequest, user=Depends(get_current_user)):
    """Run a full backtest.

    Accepts strategy configuration, loads market data, runs the unified engine,
    computes KPIs, and returns serialized results.

    This is a sync endpoint — FastAPI runs it in a threadpool.
    Typical response time: 2-10 seconds depending on data size.
    """
    from api.services.backtest_service import run_backtest as _run

    try:
        return _run(req)
    except Exception as e:
        logger.exception("Backtest failed for %s", req.symbol)
        # Return empty result on error rather than 500
        return BacktestResponse(
            trades=[], kpis={
                "total_trades": 0, "win_rate": 0, "profit_factor": 0,
                "avg_r": 0, "total_r": 0, "daily_r": 0, "r_squared": 0.0,
                "max_r_drawdown": 0, "final_balance": 10000, "total_pnl": 0,
            },
            secondary_kpis={},
            equity_curve=[],
            total_bars=0,
            total_trading_days=0,
            data_source="Error",
        )


@router.post("/kpis")
def compute_kpis(trades: list[dict], user=Depends(get_current_user)):
    """Calculate KPIs from a list of trade records.

    Useful for recalculating KPIs on stored trades without running
    a full backtest (e.g., after filtering by date range).
    """
    import pandas as pd
    import services as svc

    if not trades:
        return {"kpis": svc.calculate_kpis(pd.DataFrame()), "secondary_kpis": {}}

    df = pd.DataFrame(trades)
    for col in ["entry_time", "exit_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)

    kpis = svc.calculate_kpis(df)
    secondary = svc.calculate_secondary_kpis(df, kpis)
    return {"kpis": kpis, "secondary_kpis": secondary}
