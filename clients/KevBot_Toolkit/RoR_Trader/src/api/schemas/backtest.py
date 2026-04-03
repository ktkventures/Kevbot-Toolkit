"""Backtest request/response schemas."""

from typing import Optional, Literal
from pydantic import BaseModel


class BacktestRequest(BaseModel):
    """Full backtest configuration — mirrors the strategy builder form."""

    # Core
    symbol: str
    timeframe: str = "1Min"
    direction: Literal["LONG", "SHORT"]
    session: str = "RTH"
    data_feed: str = "sip"

    # Lookback
    days: int = 90
    lookback_mode: str = "Days"
    lookback_start_date: Optional[str] = None
    lookback_end_date: Optional[str] = None

    # Triggers
    entry_trigger_confluence_id: str
    exit_trigger_confluence_ids: list[str] = []

    # Confluence conditions
    confluence: list[str] = []

    # Risk management — either pack IDs or inline config
    stop_loss_pack_id: Optional[str] = None
    take_profit_pack_id: Optional[str] = None
    stop_config: Optional[dict] = None
    target_config: Optional[dict] = None
    stop_atr_mult: float = 1.5
    risk_per_trade: float = 100.0

    # Advanced
    bar_count_exit: Optional[int] = None
    secondary_tfs: list[str] = []
    hifi_mode: bool = False  # Enable 1-second resolution for every entry/exit

    # Output control
    include_chart_data: bool = False
    chart_indicators: list[str] = []


class BacktestTradeZoomRequest(BacktestRequest):
    """Extends BacktestRequest with trade-zoom specific fields."""

    trade_idx: int
    side: str = "exit"  # "entry" or "exit"
    padding_seconds: int = 60


class TradeRecord(BaseModel):
    """Single trade from a backtest."""
    entry_time: Optional[str] = None
    exit_time: Optional[str] = None
    direction: Optional[str] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    r_multiple: float = 0.0
    win: bool = False
    exit_reason: Optional[str] = None
    exec_type: Optional[str] = None
    bars_held: Optional[int] = None
    hold_time_seconds: Optional[float] = None
    entry_trigger: Optional[str] = None
    hifi_resolved: bool = False  # True if this trade was resolved with 1-second data


class BacktestResponse(BaseModel):
    """Full backtest result."""
    trades: list[dict]
    kpis: dict
    secondary_kpis: dict
    equity_curve: list[dict]
    total_bars: int
    total_trading_days: int
    data_source: str
    chart_data: Optional[list[dict]] = None
