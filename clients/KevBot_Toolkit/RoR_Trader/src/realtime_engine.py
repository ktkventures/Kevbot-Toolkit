"""
Real-time intra-bar alert engine using Alpaca WebSocket streaming.

Subscribes to real-time trades/quotes for active [I]-trigger strategy symbols.
Builds partial OHLCV bars from tick data and checks trigger conditions.

IMPORTANT: Requires an Alpaca SIP data feed subscription ($99/mo) for
real-time data on all symbols. The free IEX feed has a ~15-minute delay
and is limited to 30 symbols.

This module is scaffolded for integration but requires the user to confirm
their Alpaca SIP subscription before enabling live WebSocket streams.
"""

import asyncio
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Engine singleton
_engine_instance: Optional['RealtimeAlertEngine'] = None


class PartialBar:
    """Represents a partial OHLCV bar being built from tick data."""

    __slots__ = ('open', 'high', 'low', 'close', 'volume', 'bar_start',
                 'bar_duration_seconds', 'tick_count')

    def __init__(self, price: float, timestamp: datetime, bar_duration_seconds: int = 60):
        self.open = price
        self.high = price
        self.low = price
        self.close = price
        self.volume = 0
        self.bar_start = timestamp
        self.bar_duration_seconds = bar_duration_seconds
        self.tick_count = 0

    def update(self, price: float, volume: int = 1):
        """Update the partial bar with a new tick."""
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += volume
        self.tick_count += 1

    def is_complete(self, current_time: datetime) -> bool:
        """Check if the bar period has elapsed."""
        elapsed = (current_time - self.bar_start).total_seconds()
        return elapsed >= self.bar_duration_seconds

    def to_dict(self) -> dict:
        """Convert to OHLCV dict format."""
        return {
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'timestamp': self.bar_start.isoformat(),
        }


class RealtimeAlertEngine:
    """
    WebSocket-based real-time alert engine for intra-bar [I] triggers.

    Subscribes to Alpaca real-time trade data for symbols with active
    [I]-trigger strategies. Builds partial OHLCV bars from tick data
    and checks trigger conditions against live prices.

    Lifecycle:
        engine = RealtimeAlertEngine()
        engine.start(strategies)  # Starts background thread
        ...
        engine.stop()             # Graceful shutdown
    """

    def __init__(self):
        self.active_symbols: Set[str] = set()
        self.partial_bars: Dict[str, PartialBar] = {}
        self.completed_bars: Dict[str, list] = {}  # symbol -> last N completed bars
        self.strategies: List[dict] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._alert_callback = None
        self._bar_duration_seconds = 60  # 1-minute bars by default

    def start(self, strategies: list, alert_callback=None, bar_duration_seconds: int = 60):
        """Start the real-time engine for strategies with [I] triggers.

        Args:
            strategies: List of strategy dicts.
            alert_callback: Function to call when an alert fires:
                            callback(strategy, signal_type, price, timestamp).
            bar_duration_seconds: Duration of each partial bar (default 60 = 1 min).
        """
        self.strategies = [s for s in strategies if self._has_intrabar_triggers(s)]
        if not self.strategies:
            logger.info("No strategies with [I] triggers found — engine not started")
            return

        self.active_symbols = {s['symbol'] for s in self.strategies}
        self._alert_callback = alert_callback
        self._bar_duration_seconds = bar_duration_seconds
        self._running = True

        logger.info("Starting real-time engine for %d strategies, %d symbols",
                     len(self.strategies), len(self.active_symbols))

        self._thread = threading.Thread(target=self._run_loop, daemon=True,
                                         name="realtime-engine")
        self._thread.start()

    def stop(self):
        """Signal the engine to stop."""
        self._running = False
        logger.info("Real-time engine stop requested")

    def is_running(self) -> bool:
        """Check if the engine is currently running."""
        return self._running

    def get_status(self) -> dict:
        """Get current engine status."""
        return {
            'running': self._running,
            'symbols': sorted(self.active_symbols),
            'strategy_count': len(self.strategies),
            'partial_bars': {sym: bar.to_dict() for sym, bar in self.partial_bars.items()},
        }

    @staticmethod
    def _has_intrabar_triggers(strategy: dict) -> bool:
        """Check if strategy uses any [I] execution type triggers.

        [I] triggers fire intra-bar (on live tick data).
        [C] triggers fire at bar close (handled by existing poller).
        """
        # Check entry trigger execution type
        entry = strategy.get('entry_trigger_confluence_id', '')
        # The trigger execution type is stored in the confluence group's trigger definition
        # For now, check if the strategy has an explicit 'has_intrabar_triggers' flag
        # or if the trigger name contains known intra-bar patterns
        if strategy.get('has_intrabar_triggers'):
            return True

        # Webhook-inbound strategies don't use trigger-based alerting
        if strategy.get('strategy_origin') == 'webhook_inbound':
            return False

        return False

    def _run_loop(self):
        """Main event loop — connect to Alpaca WebSocket and process ticks."""
        try:
            asyncio.run(self._stream_data())
        except Exception as e:
            logger.error("Real-time engine loop error: %s", e)
        finally:
            self._running = False
            logger.info("Real-time engine stopped")

    async def _stream_data(self):
        """Subscribe to Alpaca real-time data and process ticks.

        STOP POINT: This method requires an active Alpaca SIP subscription.
        Without SIP, the WebSocket connection will fail or return delayed data.
        """
        try:
            import os
            api_key = os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("ALPACA_SECRET_KEY")
            if not api_key or not secret_key:
                logger.error("Alpaca API keys not configured — cannot start stream")
                return

            # Import Alpaca streaming client
            # Requires: pip install alpaca-py
            from alpaca.data.live import StockDataStream

            # Use SIP feed for real-time data
            stream = StockDataStream(api_key, secret_key, feed='sip')

            async def on_trade(trade):
                if not self._running:
                    return
                self._process_tick(
                    symbol=trade.symbol,
                    price=float(trade.price),
                    volume=int(trade.size) if hasattr(trade, 'size') else 1,
                    timestamp=trade.timestamp,
                )

            # Subscribe to trades for all active symbols
            for symbol in self.active_symbols:
                stream.subscribe_trades(on_trade, symbol)
                logger.info("Subscribed to real-time trades for %s", symbol)

            # Run until stopped
            while self._running:
                try:
                    await asyncio.wait_for(stream._run_forever(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    if self._running:
                        logger.warning("Stream interrupted: %s — reconnecting in 5s", e)
                        await asyncio.sleep(5)

        except ImportError:
            logger.error("alpaca-py not installed. Install with: pip install alpaca-py")
        except Exception as e:
            logger.error("Real-time stream error: %s", e)

    def _process_tick(self, symbol: str, price: float, volume: int, timestamp: datetime):
        """Process a single trade tick — update partial bar and check triggers."""
        bar = self.partial_bars.get(symbol)

        if bar is None or bar.is_complete(timestamp):
            # Archive completed bar
            if bar is not None:
                if symbol not in self.completed_bars:
                    self.completed_bars[symbol] = []
                self.completed_bars[symbol].append(bar.to_dict())
                # Keep last 100 completed bars per symbol
                if len(self.completed_bars[symbol]) > 100:
                    self.completed_bars[symbol] = self.completed_bars[symbol][-100:]

            # Start new bar
            self.partial_bars[symbol] = PartialBar(
                price, timestamp, self._bar_duration_seconds
            )
        else:
            bar.update(price, volume)

        # Check trigger conditions for strategies using this symbol
        for strategy in self.strategies:
            if strategy['symbol'] != symbol:
                continue
            self._check_intrabar_triggers(strategy, symbol, price, timestamp)

    def _check_intrabar_triggers(self, strategy: dict, symbol: str,
                                  current_price: float, timestamp: datetime):
        """Check if any [I] trigger conditions are met at current price.

        This is the core evaluation logic. For [I] triggers:
        - Compare current_price against the trigger level computed from
          the last completed bar (e.g., UT Bot trailing stop, EMA value).
        - If condition is met, fire the alert via the callback.

        TODO: Full implementation requires reading trigger level from the
        last completed bar's indicator/interpreter output. This depends on
        the trigger type (price crossing above/below a computed level).
        """
        # Placeholder — full trigger evaluation to be implemented
        # when Alpaca SIP subscription is confirmed
        pass


# =============================================================================
# MODULE-LEVEL CONVENIENCE API
# =============================================================================

def get_engine() -> RealtimeAlertEngine:
    """Get or create the singleton engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RealtimeAlertEngine()
    return _engine_instance


def start_engine(strategies: list, alert_callback=None, bar_duration_seconds: int = 60):
    """Start the real-time engine (convenience wrapper)."""
    engine = get_engine()
    if engine.is_running():
        logger.info("Engine already running")
        return engine
    engine.start(strategies, alert_callback, bar_duration_seconds)
    return engine


def stop_engine():
    """Stop the real-time engine (convenience wrapper)."""
    engine = get_engine()
    engine.stop()


def engine_status() -> dict:
    """Get engine status (convenience wrapper)."""
    engine = get_engine()
    return engine.get_status()
