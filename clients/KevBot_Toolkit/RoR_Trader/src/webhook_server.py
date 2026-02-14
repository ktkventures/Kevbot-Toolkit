"""
Lightweight inbound webhook receiver for RoR Trader.

Runs in a background daemon thread alongside the Streamlit app.
Receives signals from external sources (TradingView, LuxAlgo, etc.)
and queues them for processing by webhook-inbound strategies.
"""

import json
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

try:
    from flask import Flask, request, jsonify
except ImportError:
    Flask = None

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent.parent / "config"
INBOUND_SIGNALS_FILE = _CONFIG_DIR / "inbound_signals.json"

_server_thread: Optional[threading.Thread] = None
_server_running = False
_webhook_secrets: Dict[int, str] = {}


def _load_inbound_signals() -> list:
    """Load queued inbound signals from file."""
    if INBOUND_SIGNALS_FILE.exists():
        try:
            with open(INBOUND_SIGNALS_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception):
            return []
    return []


def _save_inbound_signals(signals: list):
    """Save inbound signals to file."""
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(INBOUND_SIGNALS_FILE, 'w') as f:
        json.dump(signals, f, indent=2)


def _append_inbound_signal(signal: dict):
    """Append a single signal to the queue."""
    signals = _load_inbound_signals()
    signal['id'] = len(signals) + 1
    signals.append(signal)
    # Keep last 1000 signals
    if len(signals) > 1000:
        signals = signals[-1000:]
    _save_inbound_signals(signals)
    return signal['id']


def get_unprocessed_signals(strategy_id: int) -> list:
    """Get unprocessed inbound signals for a strategy."""
    signals = _load_inbound_signals()
    return [s for s in signals
            if s.get('strategy_id') == strategy_id and not s.get('processed', False)]


def mark_signals_processed(signal_ids: list):
    """Mark signals as processed."""
    signals = _load_inbound_signals()
    id_set = set(signal_ids)
    for s in signals:
        if s.get('id') in id_set:
            s['processed'] = True
            s['processed_at'] = datetime.now().isoformat()
    _save_inbound_signals(signals)


def _create_flask_app():
    """Create the Flask app with webhook endpoint."""
    if Flask is None:
        raise ImportError("Flask is required for the webhook server. Install with: pip install flask")

    app = Flask(__name__)
    app.logger.setLevel(logging.WARNING)

    @app.route("/webhook/inbound/<int:strategy_id>", methods=["POST"])
    def receive_webhook(strategy_id):
        # Validate secret
        secret = request.headers.get("X-Webhook-Secret") or request.args.get("secret")
        expected = _webhook_secrets.get(strategy_id)
        if not expected or secret != expected:
            return jsonify({"error": "Invalid or missing webhook secret"}), 401

        body = request.get_json(force=True, silent=True) or {}
        signal_id = _append_inbound_signal({
            "strategy_id": strategy_id,
            "received_at": datetime.now().isoformat(),
            "payload": body,
            "processed": False,
        })
        logger.info("Received webhook for strategy %d (signal #%d)", strategy_id, signal_id)
        return jsonify({"status": "received", "signal_id": signal_id}), 200

    @app.route("/webhook/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "strategies": list(_webhook_secrets.keys())}), 200

    return app


def register_strategy_secret(strategy_id: int, secret: str):
    """Register or update a webhook secret for a strategy."""
    _webhook_secrets[strategy_id] = secret


def unregister_strategy(strategy_id: int):
    """Remove a strategy's webhook secret."""
    _webhook_secrets.pop(strategy_id, None)


def start_webhook_server(port: int = 8501, secrets: Optional[Dict[int, str]] = None):
    """Start the webhook server in a daemon thread.

    Args:
        port: Port to listen on (default 8501).
        secrets: Dict mapping strategy_id -> webhook_secret.

    Returns:
        The server thread, or None if Flask is not available.
    """
    global _server_thread, _server_running

    if Flask is None:
        logger.warning("Flask not installed — webhook server disabled")
        return None

    if _server_running:
        logger.info("Webhook server already running")
        return _server_thread

    if secrets:
        _webhook_secrets.update(secrets)

    app = _create_flask_app()

    def _run():
        global _server_running
        _server_running = True
        try:
            # Use werkzeug directly for quieter logging
            from werkzeug.serving import make_server
            server = make_server('0.0.0.0', port, app)
            logger.info("Webhook server listening on port %d", port)
            server.serve_forever()
        except Exception as e:
            logger.error("Webhook server error: %s", e)
        finally:
            _server_running = False

    _server_thread = threading.Thread(target=_run, daemon=True, name="webhook-server")
    _server_thread.start()
    return _server_thread


def stop_webhook_server():
    """Signal the webhook server to stop."""
    global _server_running
    _server_running = False


def is_running() -> bool:
    """Check if the webhook server is currently running."""
    return _server_running
