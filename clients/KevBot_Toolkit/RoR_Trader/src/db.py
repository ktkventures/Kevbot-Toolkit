"""
Centralized database access layer for RoR Trader.

Wraps all Supabase/PostgreSQL operations with a USE_DB toggle for
incremental migration from JSON file storage. When USE_DB is false,
all code paths fall through to the existing JSON file operations.

Phase 22A: Foundation (client setup, toggle, transformation helpers)
Phase 22B: CRUD rewiring (strategies, portfolios, alerts, configs)
"""
import os
import json
import time
import threading
import logging
import contextvars
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ============================================================
# Transient-error retry helper (added 2026-05-11)
# ============================================================
# Supabase has been intermittently unavailable through 2026-05-11
# (Cloudflare 522/521/504 returning HTML the supabase-py client wraps
# as "JSON could not be generated"). The previous replace_trades_admin
# (DELETE then INSERT) was vulnerable to partial-write data loss when
# the second call hit a 522. Wrapping the whole operation in retry
# means a transient blip retries the entire DELETE+INSERT idempotently
# (the second DELETE is a no-op, then INSERT runs on the second try).
#
# Only RETRYABLE transient errors are retried. Permanent errors
# (unique violation, FK violation, syntax) bubble immediately.

_TRANSIENT_SIGNALS = (
    '522', '521', '503', '504', '524',  # Cloudflare gateway errors
    'json could not be generated',        # supabase-py's 5xx HTML wrap
    'connection',                          # generic connection refused / reset
    'timed out', 'timeout',
    'eof occurred', 'broken pipe',
)
_PERMANENT_SIGNALS = (
    'duplicate', 'unique constraint', 'unique_violation',
    'foreign key', 'fk_violation',
    'syntax error', 'invalid',
)


def _is_transient_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if any(sig in msg for sig in _PERMANENT_SIGNALS):
        return False
    return any(sig in msg for sig in _TRANSIENT_SIGNALS)


def _execute_with_retry(fn, op_name: str = 'supabase op',
                        max_attempts: int = 4, base_delay: float = 2.0):
    """Run `fn` with exponential backoff on transient errors.

    Backoff schedule with defaults (max_attempts=4, base_delay=2.0):
      attempt 1 → run; on fail wait 2s
      attempt 2 → run; on fail wait 4s
      attempt 3 → run; on fail wait 8s
      attempt 4 → run; final attempt, raise on fail

    Permanent errors (unique violation, FK, syntax) skip retry and raise
    immediately so the caller can handle them.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if not _is_transient_error(exc) or attempt == max_attempts:
                if attempt > 1:
                    logger.error(
                        '%s: gave up after %d attempts: %s',
                        op_name, attempt, exc)
                raise
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                '%s: transient error attempt %d/%d (%s). Retrying in %.1fs.',
                op_name, attempt, max_attempts, exc, delay)
            time.sleep(delay)
    if last_exc is not None:
        raise last_exc

# ============================================================
# Configuration
# ============================================================

# Toggle: "true" = use database, "false" = use JSON files
USE_DB = os.getenv("USE_DB", "false").lower() == "true"

# Supabase credentials from environment
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# ============================================================
# User Context (contextvars)
#
# History: originally `threading.local` — fine for Streamlit where each
# session runs on one thread. Broke under FastAPI + uvicorn, where a
# sync dependency (`get_current_user`) and a sync endpoint run in
# anyio's threadpool and can land on DIFFERENT worker threads under
# load. Thread-local set on T1 was invisible on T2 → `user_id = None`,
# which postgrest stringified to the literal "None" and Supabase
# rejected as an invalid UUID (2026-04-22).
#
# ContextVar is async-safe: anyio's to_thread.run_sync() copies the
# current Context into the worker thread (via contextvars.copy_context),
# so the user_id set in the dependency propagates to the endpoint
# regardless of which threadpool worker runs it. Daemon threads we spawn
# manually (mass_builder worker) don't inherit the parent Context, but
# they always call set_admin_user_context / set_current_user themselves
# before any DB work — same as before.
# ============================================================

_user_id_var: contextvars.ContextVar = contextvars.ContextVar("user_id", default=None)
_access_token_var: contextvars.ContextVar = contextvars.ContextVar("access_token", default=None)
_admin_mode_var: contextvars.ContextVar = contextvars.ContextVar("admin_mode", default=False)


def set_current_user(user_id: str, access_token: str):
    """Set the current user context for this request/task."""
    _user_id_var.set(user_id)
    _access_token_var.set(access_token)
    _admin_mode_var.set(False)


def set_admin_user_context(user_id: str):
    """Set user context in ADMIN mode — for worker / service paths that
    need to invoke user-context-aware helpers (load_confluence_groups,
    load_general_packs, etc.) without a real user JWT. Under admin mode,
    get_client() returns the admin client so queries bypass RLS;
    get_current_user_id() still returns the given user_id so WHERE-clauses
    scope correctly.
    """
    _user_id_var.set(user_id)
    _access_token_var.set(None)
    _admin_mode_var.set(True)


def clear_current_user():
    """Clear user context. Safe to call even if nothing was set."""
    _user_id_var.set(None)
    _access_token_var.set(None)
    _admin_mode_var.set(False)


def get_current_user_id() -> str:
    """Get the current user's UUID. Returns None if not authenticated."""
    return _user_id_var.get()


def get_current_token() -> str:
    """Get the current user's JWT access token."""
    return _access_token_var.get()


def _is_admin_mode() -> bool:
    return _admin_mode_var.get()


# ============================================================
# Supabase Client Factory
# ============================================================

_anon_client = None
_admin_client = None
_client_lock = threading.Lock()


def get_client():
    """Get a Supabase client authenticated as the current user.

    Uses the anon key. The user's JWT is set on the client so that
    Row Level Security policies are enforced server-side.

    When the thread-local context is in admin mode
    (set_admin_user_context), returns the admin client instead so
    worker/service paths can run user-scoped queries without a JWT.
    """
    global _anon_client
    if not USE_DB:
        raise RuntimeError("get_client() called but USE_DB is false")

    if _is_admin_mode():
        return get_admin_client()

    from supabase import create_client

    # Create a fresh client per call to ensure correct JWT context.
    # supabase-py clients are lightweight — this is the recommended pattern
    # for multi-user server apps where different requests have different JWTs.
    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    token = get_current_token()
    if token:
        client.postgrest.auth(token)
    return client


def get_admin_client():
    """Get a Supabase client with service role key (bypasses RLS).

    Used by the worker service only — never in the web app.
    The service role key has full access to all rows regardless of user_id.

    Explicit `postgrest_client_timeout` (2026-06-03) closes the indefinite-
    hang path: when Supabase has a network blip or 522, the underlying
    HTTPX call could previously block forever, wedging the worker's
    reload thread until manual restart. With a 30s timeout, the call
    raises and the reload thread retries on the next 5-min cycle.
    Watchdog phase 2 — pairs with the STALE detection in data_worker's
    metrics line.
    """
    global _admin_client
    if not USE_DB:
        raise RuntimeError("get_admin_client() called but USE_DB is false")

    with _client_lock:
        if _admin_client is None:
            from supabase import create_client
            try:
                from supabase import ClientOptions
                opts = ClientOptions(postgrest_client_timeout=30)
                _admin_client = create_client(
                    SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, options=opts)
            except ImportError:
                # Older supabase-py without ClientOptions — fall back.
                _admin_client = create_client(
                    SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return _admin_client


# ============================================================
# Strategy Transformation Helpers
# ============================================================

# Fields stored as dedicated database columns (not in config JSONB)
STRATEGY_COLUMN_FIELDS = {
    'id', 'user_id', 'name', 'symbol', 'direction', 'timeframe',
    'strategy_origin', 'forward_testing', 'forward_test_start',
    'alert_tracking_enabled', 'alert_tracking_reset_at',
    'kpis', 'stored_trades', 'equity_curve_data',
    'live_executions', 'discrepancies', 'discrepancies_dismissed_at',
    'data_refreshed_at', 'created_at', 'updated_at',
    # Strategy Health Badge (2026-04-27 migration: strategy_health_badge.sql)
    'data_source', 'kpis_stale_since', 'kpis_computed_at',
    # Auto-parity score (2026-04-28 migration: strategy_parity_status.sql)
    'parity_status',
}


def _strategy_to_row(strat: dict) -> dict:
    """Split a flat strategy dict into column + JSONB format for the database.

    The app works with flat dicts where all fields are top-level keys.
    The database stores queryable fields as proper columns and everything
    else (triggers, confluence, risk params, lookback settings) in a
    single 'config' JSONB column.
    """
    row = {}
    config = {}
    for k, v in strat.items():
        if k in STRATEGY_COLUMN_FIELDS:
            row[k] = v
        else:
            config[k] = v
    row['config'] = config if config else {}
    return row


def _row_to_strategy(row: dict, hydrate_trades: bool = True) -> dict:
    """Merge a database row back into a flat strategy dict.

    Unpacks the 'config' JSONB column and merges it with the column fields,
    producing the same flat dict shape that the rest of the app expects.

    Phase 40: when ``USE_TRADES_TABLE`` is ON and ``hydrate_trades=True``,
    populate ``stored_trades`` from the trades table so callers see the
    same DTO as under the JSONB-era. Pass ``hydrate_trades=False`` on the
    lite list path where we deliberately don't want the trades payload.
    """
    strat = {}
    for k, v in row.items():
        if k == 'config':
            if isinstance(v, str):
                v = json.loads(v)
            if v:
                strat.update(v)
        else:
            strat[k] = v
    if hydrate_trades:
        try:
            import trades_store as _trades_store
            if _trades_store._flag_on() and strat.get('id'):
                # 2026-05-12 fix: scope hydration to backtest_%. Without
                # this filter, stored_trades was getting populated with
                # ALL lanes (backtest + cache + legacy NULL), and any
                # caller that subsequently tagged the list with a single
                # data_source (e.g., append_new_backtest_trades_for_strategy
                # at forward_test_service.py:1407 stamping
                # data_source='backtest_<model>') would create intra-batch
                # duplicates at (sid, entry_ts, exit_ts, 'backtest_<model>')
                # — violating the trades_dedupe_idx unique constraint.
                # Historical semantic: stored_trades = backtest output.
                # Algo-specific callers query trades-table directly with
                # their own filter (cache_%).
                strat['stored_trades'] = _trades_store.load_trades_for_strategy(
                    strat['id'], strat.get('user_id'),
                    data_source_filter='backtest_%')
        except Exception:
            # Never let hydration failures break strategy loads — callers
            # that need trades can call hydrate_strategy_trades explicitly.
            pass
    return strat


# ============================================================
# Alert Transformation Helpers
# ============================================================

# Fields stored as dedicated alert columns (not in data JSONB).
# Trade_Timestamps_Spec (2026-04-17): 8 columns added 2026-04-20 in the
# primary migration; 3 more (hold_duration_s, behavior, webhook_deliveries)
# added in the follow-up ALTER, promoting everything out of JSONB drift.
ALERT_COLUMN_FIELDS = {
    'id', 'user_id', 'type', 'strategy_id', 'strategy_name',
    'symbol', 'direction', 'timeframe', 'source',
    'acknowledged', 'webhook_sent', 'timestamp',
    # Trade_Timestamps_Spec top-level columns (4-timestamp model + metadata)
    'event_type', 'side', 'trigger_ts', 'fill_ts',
    'exec_type', 'trigger_id', 'price', 'bar_time',
    'hold_duration_s', 'behavior',
    # actual_price: near-live market price at save moment (Ralph's last
    # per-second close). Gap vs `price` (theoretical fill) = price slippage.
    'actual_price',
    # Webhook delivery tracking — resolves pre-existing PGRST204 drift
    'webhook_deliveries',
    # algo_model split (2026-05-07): which live_model produced this alert.
    # Stamped at fire time so Divergence tab can attribute alerts to model.
    # Legacy alerts (pre-2026-05-07) stay null and render as "unknown".
    'live_model',
    # ws_rest_spliced model verification (2026-05-28): REST verifier
    # populates these columns AFTER the alert fires. Status values:
    # NULL (pre-migration / not opted in) | 'pending' (queued) |
    # 'verified' (REST matched WS within tolerance) | 'corrected'
    # (REST differed; indicator state updated via apply_last_bar_correction)
    # | 'rest_unavailable' (REST never returned within max_wait). See
    # migrations/alerts_add_verification_columns.sql + breezy-dreaming-
    # umbrella plan.
    'verification_status',
    'verification_close_delta',
    'verification_completed_at',
}


def _alert_to_row(alert: dict) -> dict:
    """Split a flat alert dict into column + JSONB format for the database."""
    row = {}
    data = {}
    for k, v in alert.items():
        if k in ALERT_COLUMN_FIELDS:
            row[k] = v
        else:
            data[k] = v
    row['data'] = data if data else {}
    return row


def _row_to_alert(row: dict) -> dict:
    """Merge a database alert row back into a flat alert dict."""
    alert = {}
    for k, v in row.items():
        if k == 'data':
            if isinstance(v, str):
                v = json.loads(v)
            if v:
                alert.update(v)
        else:
            alert[k] = v
    return alert


# ============================================================
# Strategy CRUD (database path)
# ============================================================

def load_strategies_db() -> list:
    """Load all strategies for the current user from the database."""
    user_id = get_current_user_id()
    client = get_client()
    result = client.table('strategies') \
        .select('*') \
        .eq('user_id', user_id) \
        .order('id') \
        .execute()
    return [_row_to_strategy(r) for r in result.data]


# Columns returned by the lean list path. Includes `equity_curve_data` so
# frontend sparklines keep working (~70 KB typical), but EXCLUDES
# `stored_trades` (can hit 2.5 MB per strategy) and `live_executions`.
# Callers that genuinely need those — recompute paths, forward-test splits,
# strategy-detail page — use load_strategies_db() or get_strategy_by_id_db.
_STRATEGY_LIST_COLUMNS = (
    'id,user_id,name,symbol,direction,timeframe,'
    'config,kpis,equity_curve_data,'
    'alert_tracking_enabled,alert_tracking_reset_at,'
    'created_at,updated_at,data_refreshed_at,'
    'discrepancies,discrepancies_dismissed_at,'
    'forward_test_start,forward_testing,strategy_origin,'
    # Strategy Health Badge columns (2026-04-27 strategy_health_badge.sql)
    # Without these the list endpoint can't fully compute the badge —
    # data_source-based issues (rapid_test_data, no_hifi_run, kpis_time_stale)
    # silently skip.
    'data_source,kpis_stale_since,kpis_computed_at,'
    # Auto-parity score (2026-04-28 strategy_parity_status.sql) — used by
    # the strategy card to show a "live-executable" verdict at a glance.
    'parity_status'
)


def load_strategies_db_lite() -> list:
    """Load all strategies for the current user, skipping heavy JSONB.

    Excludes `stored_trades` and `live_executions`. Use for list endpoints,
    dashboard summaries, and any view that doesn't actually iterate trades.

    Reduces strategies-table read traffic by ~35× on fat strategies (one
    with 3000 trades observed at 2.5 MB stored_trades vs ~75 KB lean).

    Phase 40: hydrate_trades=False to skip the per-strategy trades-table
    fetch when flag ON — list views don't need trades and hydrating N
    strategies' worth would defeat the whole lite-loader purpose.
    """
    user_id = get_current_user_id()
    client = get_client()
    result = client.table('strategies') \
        .select(_STRATEGY_LIST_COLUMNS) \
        .eq('user_id', user_id) \
        .order('id') \
        .execute()
    return [_row_to_strategy(r, hydrate_trades=False) for r in result.data]


def get_strategy_by_id_db(strategy_id: int) -> dict | None:
    """Load a single strategy by ID for the current user."""
    client = get_client()
    result = client.table('strategies') \
        .select('*') \
        .eq('id', strategy_id) \
        .maybe_single() \
        .execute()
    if result and result.data:
        return _row_to_strategy(result.data)
    return None


def save_strategy_db(strategy: dict) -> dict:
    """Insert a new strategy for the current user. Returns the saved strategy."""
    user_id = get_current_user_id()
    strategy['user_id'] = user_id

    # Alerts are always on for monitored strategies — there's no UI toggle
    # any more (removed 2026-04-24 after confirming the flag was never
    # actually checked in the worker's alert-save path). Force the column
    # default so downstream display paths stay consistent.
    strategy.setdefault('alert_tracking_enabled', True)
    if strategy.get('alert_tracking_enabled') and not strategy.get('alert_tracking_reset_at'):
        strategy['alert_tracking_reset_at'] = datetime.now(timezone.utc).isoformat()

    # Phase 40: when the trades-table is authoritative, pull stored_trades
    # off the payload so the strategies row stays lean, and re-insert them
    # into the trades table after the strategy row exists (the new row's
    # assigned id is needed as the FK target).
    pending_trades = None
    try:
        import trades_store as _trades_store
        if _trades_store._flag_on():
            pending_trades = strategy.pop('stored_trades', None) or []
    except Exception:
        _trades_store = None

    row = _strategy_to_row(strategy)
    # Remove 'id' so the database assigns the next SERIAL value
    row.pop('id', None)
    client = get_client()
    result = client.table('strategies').insert(row).execute()
    saved = _row_to_strategy(result.data[0])

    if pending_trades and _trades_store is not None and _trades_store._flag_on():
        try:
            _trades_store.replace_trades_for_strategy(
                saved['id'], saved.get('user_id') or user_id, pending_trades)
            # Echo the trades back onto the returned dict so callers see
            # the same shape they would have gotten from the JSONB path.
            saved['stored_trades'] = pending_trades
        except Exception as e:
            import logging as _l
            _l.getLogger(__name__).warning(
                "save_strategy_db: trades-table insert failed for "
                "strategy %s: %s", saved.get('id'), e)

    return saved


def update_strategy_db(strategy_id: int, updated: dict) -> dict | None:
    """Update an existing strategy by ID. Returns the updated strategy."""
    # Phase 40: redirect stored_trades writes to the trades table when flag ON.
    # Only touch the trades table when the caller actually included a
    # stored_trades key — partial updates that don't mention it must NOT
    # wipe trades (mirrors the feedback_jsonb_partial_updates rule).
    pending_trades = None
    try:
        import trades_store as _trades_store
        if _trades_store._flag_on() and 'stored_trades' in updated:
            pending_trades = updated.pop('stored_trades', None) or []
    except Exception:
        _trades_store = None

    row = _strategy_to_row(updated)
    row.pop('user_id', None)  # Don't allow changing user_id
    row.pop('id', None)       # Don't include PK in the SET clause

    # CRITICAL: partial-update safety. _strategy_to_row always emits a
    # `config` key (empty `{}` when no config-bucket fields are present).
    # On an UPDATE that sends `config: {}`, PostgREST overwrites the
    # JSONB column with `{}` and wipes trigger/confluence/stop settings.
    # This is the `feedback_jsonb_partial_updates` rule — strategies 70+71
    # got wiped on 2026-04-14, and strategies 129+130 got wiped on
    # 2026-04-24 via my own set_forward_test_start endpoint. Only include
    # `config` in the payload when the caller actually provided
    # config-bucket fields to change.
    caller_had_config_fields = any(
        k not in STRATEGY_COLUMN_FIELDS for k in updated.keys()
    )
    if not caller_had_config_fields:
        row.pop('config', None)

    if not row:
        return None  # nothing to update

    client = get_client()
    result = client.table('strategies') \
        .update(row) \
        .eq('id', strategy_id) \
        .execute()

    saved = _row_to_strategy(result.data[0]) if result.data else None

    if pending_trades is not None and _trades_store is not None and _trades_store._flag_on():
        try:
            uid = (saved.get('user_id') if saved else None) or get_current_user_id()
            _trades_store.replace_trades_for_strategy(
                strategy_id, uid, pending_trades)
            if saved is not None:
                saved['stored_trades'] = pending_trades
        except Exception as e:
            import logging as _l
            _l.getLogger(__name__).warning(
                "update_strategy_db: trades-table replace failed for "
                "strategy %s: %s", strategy_id, e)

    return saved


def delete_strategy_db(strategy_id: int) -> bool:
    """Delete a strategy by ID. Returns True if deleted."""
    client = get_client()
    result = client.table('strategies') \
        .delete() \
        .eq('id', strategy_id) \
        .execute()
    return bool(result.data)


# ============================================================
# Portfolio CRUD (database path)
# ============================================================

def load_portfolios_db() -> list:
    """Load all portfolios for the current user."""
    user_id = get_current_user_id()
    client = get_client()
    result = client.table('portfolios') \
        .select('*') \
        .eq('user_id', user_id) \
        .order('id') \
        .execute()
    return [_restore_portfolio_from_db(r) for r in result.data]


def get_portfolio_by_id_db(portfolio_id: int) -> dict | None:
    """Load a single portfolio by ID."""
    client = get_client()
    result = client.table('portfolios') \
        .select('*') \
        .eq('id', portfolio_id) \
        .maybe_single() \
        .execute()
    return _restore_portfolio_from_db(result.data) if result and result.data else None


# Fields stored in portfolio dicts but NOT as DB columns.
# These get nested inside the 'account' JSONB column for DB persistence.
_PORTFOLIO_NON_DB_FIELDS = {'change_log', 'journal_entries', 'buying_power_mode'}


def _prepare_portfolio_for_db(d: dict) -> dict:
    """Prepare portfolio dict for DB storage.

    Moves non-DB fields into the 'account' JSONB column so they persist
    without requiring DB schema changes. Uses deep copy to avoid mutating
    the original dict.
    """
    import copy as _copy
    payload = _copy.deepcopy(d)
    account = payload.get('account', {})
    if not isinstance(account, dict):
        account = {}

    # Nest non-DB fields inside account
    for field in _PORTFOLIO_NON_DB_FIELDS:
        if field in payload:
            account[f'_p37_{field}'] = payload.pop(field)

    payload['account'] = account
    return payload


def _restore_portfolio_from_db(row: dict) -> dict:
    """Restore non-DB fields from the account JSONB column back to top level."""
    if not row:
        return row
    account = row.get('account', {})
    # PostgREST may return JSONB as a string
    if isinstance(account, str):
        try:
            account = json.loads(account)
            row['account'] = account
        except (json.JSONDecodeError, TypeError):
            account = {}
    if isinstance(account, dict):
        for field in _PORTFOLIO_NON_DB_FIELDS:
            db_key = f'_p37_{field}'
            if db_key in account:
                row[field] = account.pop(db_key)
    return row


def save_portfolio_db(portfolio: dict) -> dict:
    """Insert a new portfolio. Returns the saved portfolio."""
    payload = _prepare_portfolio_for_db(portfolio)
    payload['user_id'] = get_current_user_id()
    payload.pop('id', None)
    client = get_client()
    result = client.table('portfolios').insert(payload).execute()
    return _restore_portfolio_from_db(result.data[0]) if result.data else {}


def update_portfolio_db(portfolio_id: int, updated: dict) -> dict | None:
    """Update a portfolio by ID."""
    payload = _prepare_portfolio_for_db(updated)
    payload.pop('user_id', None)
    payload.pop('id', None)
    client = get_client()
    result = client.table('portfolios') \
        .update(payload) \
        .eq('id', portfolio_id) \
        .execute()
    return _restore_portfolio_from_db(result.data[0]) if result.data else None


def delete_portfolio_db(portfolio_id: int) -> bool:
    """Delete a portfolio by ID."""
    client = get_client()
    result = client.table('portfolios') \
        .delete() \
        .eq('id', portfolio_id) \
        .execute()
    return bool(result.data)


# ============================================================
# Requirement Sets CRUD (database path)
# ============================================================

def load_requirements_db() -> list:
    """Load all requirement sets (own + built-in) for the current user."""
    client = get_client()
    result = client.table('requirement_sets') \
        .select('*') \
        .order('id') \
        .execute()
    return result.data


def save_requirement_set_db(req_set: dict) -> dict:
    """Insert a new requirement set."""
    req_set['user_id'] = get_current_user_id()
    req_set.pop('id', None)
    client = get_client()
    result = client.table('requirement_sets').insert(req_set).execute()
    return result.data[0]


def update_requirement_set_db(req_id: int, updated: dict) -> dict | None:
    """Update a requirement set by ID."""
    updated.pop('user_id', None)
    updated.pop('id', None)
    client = get_client()
    result = client.table('requirement_sets') \
        .update(updated) \
        .eq('id', req_id) \
        .execute()
    return result.data[0] if result.data else None


def delete_requirement_set_db(req_id: int) -> bool:
    """Delete a requirement set by ID."""
    client = get_client()
    result = client.table('requirement_sets') \
        .delete() \
        .eq('id', req_id) \
        .execute()
    return bool(result.data)


# ============================================================
# Alert CRUD (database path)
# ============================================================

def load_alerts_db(limit: int = 100) -> list:
    """Load recent alerts for the current user."""
    user_id = get_current_user_id()
    client = get_client()
    result = client.table('alerts') \
        .select('*') \
        .eq('user_id', user_id) \
        .order('timestamp', desc=True) \
        .limit(limit) \
        .execute()
    return [_row_to_alert(r) for r in result.data]


def save_alert_db(alert: dict) -> dict:
    """Insert a new alert."""
    alert['user_id'] = get_current_user_id()
    row = _alert_to_row(alert)
    row.pop('id', None)
    client = get_client()
    result = client.table('alerts').insert(row).execute()
    return _row_to_alert(result.data[0])


def update_alert_db(alert_id: int, updates: dict) -> dict | None:
    """Update an alert by ID (e.g., acknowledge)."""
    client = get_client()
    result = client.table('alerts') \
        .update(updates) \
        .eq('id', alert_id) \
        .execute()
    return _row_to_alert(result.data[0]) if result.data else None


def get_alert_by_id_db(alert_id: int) -> dict | None:
    """Load a single alert row by id, scoped to the current user.

    Used by the Portfolio Trade History Details drawer (roadmap 9ad) to
    surface the full alert including its `webhook_deliveries` JSONB array.
    RLS enforces user scoping when called with a user-scoped client, but
    we still filter on user_id explicitly so this is safe under either
    get_client() or get_admin_client().
    """
    user_id = get_current_user_id()
    client = get_client()
    q = client.table('alerts').select('*').eq('id', alert_id)
    if user_id:
        q = q.eq('user_id', user_id)
    result = q.limit(1).execute()
    if not result.data:
        return None
    return _row_to_alert(result.data[0])


def get_alerts_for_strategy_db(strategy_id: int, limit: int = 100) -> list:
    """Load alerts for a specific strategy."""
    client = get_client()
    result = client.table('alerts') \
        .select('*') \
        .eq('strategy_id', strategy_id) \
        .order('timestamp', desc=True) \
        .limit(limit) \
        .execute()
    return [_row_to_alert(r) for r in result.data]


def delete_alerts_for_strategy_db(strategy_id: int) -> bool:
    """Delete all alerts for a strategy."""
    client = get_client()
    result = client.table('alerts') \
        .delete() \
        .eq('strategy_id', strategy_id) \
        .execute()
    return True


def clear_alerts_db() -> bool:
    """Delete all alerts for the current user."""
    client = get_client()
    # RLS ensures only current user's alerts are affected
    client.table('alerts').delete().neq('id', 0).execute()
    return True


# ============================================================
# Alert Config CRUD (database path)
# ============================================================

def load_alert_config_db() -> dict:
    """Load alert config for the current user."""
    user_id = get_current_user_id()
    client = get_client()
    result = client.table('alert_config') \
        .select('config') \
        .eq('user_id', user_id) \
        .maybe_single() \
        .execute()
    if result and result.data:
        config = result.data['config']
        return json.loads(config) if isinstance(config, str) else config
    return {"global": {}, "strategies": {}, "portfolios": {}}


def save_alert_config_db(config: dict):
    """Upsert alert config for the current user."""
    user_id = get_current_user_id()
    client = get_client()
    client.table('alert_config').upsert({
        'user_id': user_id,
        'config': config,
        'updated_at': datetime.now(timezone.utc).isoformat(),
    }, on_conflict='user_id').execute()


# ============================================================
# User Config CRUD (settings, confluence groups, packs)
# These are single-document-per-user tables.
# ============================================================

def _load_user_config(table: str, data_column: str, default):
    """Generic: load a single-row config for the current user."""
    user_id = get_current_user_id()
    client = get_client()
    result = client.table(table) \
        .select(data_column) \
        .eq('user_id', user_id) \
        .maybe_single() \
        .execute()
    if result and result.data:
        val = result.data.get(data_column)
        if val is None:
            return default
        return json.loads(val) if isinstance(val, str) else val
    return default


def _save_user_config(table: str, data_column: str, data):
    """Generic: upsert a single-row config for the current user."""
    user_id = get_current_user_id()
    client = get_client()
    client.table(table).upsert({
        'user_id': user_id,
        data_column: data,
        'updated_at': datetime.now(timezone.utc).isoformat(),
    }, on_conflict='user_id').execute()


def load_settings_db() -> dict:
    return _load_user_config('user_settings', 'settings', {})

def save_settings_db(settings: dict):
    _save_user_config('user_settings', 'settings', settings)

def load_confluence_groups_db() -> list:
    return _load_user_config('confluence_groups', 'groups', [])

def save_confluence_groups_db(groups: list):
    _save_user_config('confluence_groups', 'groups', groups)

def load_general_packs_db() -> list:
    return _load_user_config('general_packs', 'packs', [])

def save_general_packs_db(packs: list):
    _save_user_config('general_packs', 'packs', packs)

def load_risk_management_packs_db() -> list:
    return _load_user_config('risk_management_packs', 'packs', [])

def save_risk_management_packs_db(packs: list):
    _save_user_config('risk_management_packs', 'packs', packs)

def load_time_exit_packs_db() -> list:
    try:
        return _load_user_config('time_exit_packs', 'packs', [])
    except Exception:
        # Table may not exist yet — return empty so defaults are seeded
        return []

def save_time_exit_packs_db(packs: list):
    try:
        _save_user_config('time_exit_packs', 'packs', packs)
    except Exception:
        # Table may not exist yet — silently fail, defaults still work in-memory
        pass


# ============================================================
# Webhook Templates CRUD (database path)
# ============================================================

def load_webhook_templates_db() -> list:
    """Load all webhook templates (own + defaults)."""
    client = get_client()
    result = client.table('webhook_templates') \
        .select('*') \
        .order('created_at') \
        .execute()
    return result.data


def save_webhook_template_db(template: dict) -> dict:
    """Insert a new webhook template."""
    template['user_id'] = get_current_user_id()
    client = get_client()
    result = client.table('webhook_templates').insert(template).execute()
    return result.data[0]


def update_webhook_template_db(template_id: str, updates: dict) -> dict | None:
    """Update a webhook template by ID."""
    updates.pop('user_id', None)
    updates.pop('id', None)
    client = get_client()
    result = client.table('webhook_templates') \
        .update(updates) \
        .eq('id', template_id) \
        .execute()
    return result.data[0] if result.data else None


def delete_webhook_template_db(template_id: str) -> bool:
    """Delete a webhook template by ID."""
    client = get_client()
    result = client.table('webhook_templates') \
        .delete() \
        .eq('id', template_id) \
        .execute()
    return bool(result.data)


# ============================================================
# Webhook Groups CRUD (database path) — Phase 39
# ============================================================
# Table created by migrations/phase39_webhook_groups.sql. RLS filters by
# auth.uid() = user_id, so user-scoped calls need a JWT. Worker calls are
# routed through get_client() which returns the admin client when
# _admin_mode_var is True (see set_admin_user_context).

def load_webhook_groups_db() -> list:
    """Load all webhook groups for the current user."""
    user_id = get_current_user_id()
    client = get_client()
    q = client.table('webhook_groups').select('*').order('created_at')
    if user_id:
        q = q.eq('user_id', user_id)
    result = q.execute()
    return result.data or []


def get_webhook_group_by_id_db(group_id: str) -> dict | None:
    """Load a single webhook group by ID. RLS enforces user scoping."""
    client = get_client()
    result = client.table('webhook_groups') \
        .select('*') \
        .eq('id', group_id) \
        .maybe_single() \
        .execute()
    return result.data if result and result.data else None


def save_webhook_group_db(group: dict) -> dict:
    """Insert a new webhook group. Uses the caller-supplied id."""
    payload = dict(group)
    payload['user_id'] = get_current_user_id()
    client = get_client()
    result = client.table('webhook_groups').insert(payload).execute()
    return result.data[0] if result.data else {}


def update_webhook_group_db(group_id: str, updates: dict) -> dict | None:
    """Update a webhook group by ID."""
    payload = dict(updates)
    payload.pop('id', None)
    payload.pop('user_id', None)
    client = get_client()
    result = client.table('webhook_groups') \
        .update(payload) \
        .eq('id', group_id) \
        .execute()
    return result.data[0] if result.data else None


def delete_webhook_group_db(group_id: str) -> bool:
    """Delete a webhook group by ID."""
    client = get_client()
    result = client.table('webhook_groups') \
        .delete() \
        .eq('id', group_id) \
        .execute()
    return bool(result.data)


# ============================================================
# Trades CRUD (database path) — Phase 40
# ============================================================
# Table created by migrations/phase40_trades_table.sql. RLS policies
# filter by auth.uid() = user_id. Worker + API callers need admin-mode
# context for service-role access; that's set at the request boundary
# the same way as for alerts/strategies.
#
# Hot columns (filtered/joined/aggregated heavily) + a single `data`
# JSONB for the long tail. Mirrors the alerts-table pattern.
TRADE_COLUMN_FIELDS = {
    'id', 'strategy_id', 'user_id',
    'entry_alert_id', 'exit_alert_id',
    'entry_trigger_ts', 'entry_fill_ts',
    'exit_trigger_ts', 'exit_fill_ts',
    'entry_price', 'exit_price',
    'r_multiple', 'dollar_pnl', 'executed_quantity',
    'direction', 'exec_type', 'exit_reason', 'data_source',
    'created_at', 'provisional',
}


def _trade_to_row(trade: dict) -> dict:
    """Split a flat trade dict into column + data JSONB format.

    Accepts legacy ``entry_time`` / ``exit_time`` aliases and maps them to
    the canonical ``entry_fill_ts`` / ``exit_fill_ts`` columns so older
    backfill payloads persist into the right fields. Mirrors the alerts
    _alert_to_row pattern.
    """
    row: dict = {}
    data: dict = {}
    legacy_entry = trade.get('entry_time')
    legacy_exit = trade.get('exit_time')
    for k, v in trade.items():
        if k in ('entry_time', 'exit_time'):
            continue  # handled via the canonical *_fill_ts below
        if k in TRADE_COLUMN_FIELDS:
            row[k] = v
        else:
            data[k] = v
    # Back-fill canonical timestamps from legacy aliases when needed
    if row.get('entry_fill_ts') in (None, '') and legacy_entry:
        row['entry_fill_ts'] = legacy_entry
    if row.get('exit_fill_ts') in (None, '') and legacy_exit:
        row['exit_fill_ts'] = legacy_exit
    # confluence_records can arrive as a set (trade record originals use
    # a set) — JSONB can't serialize sets; cast to a sorted list to keep
    # order deterministic across inserts.
    cr = data.get('confluence_records')
    if isinstance(cr, set):
        data['confluence_records'] = sorted(cr)
    row['data'] = data if data else {}
    return row


def _row_to_trade(row: dict) -> dict:
    """Reconstruct a flat trade dict from a trades table row.

    Merges the `data` JSONB back into the top-level dict so downstream
    consumers see the same shape as legacy stored_trades entries. Also
    emits the legacy ``entry_time`` / ``exit_time`` aliases so older
    consumers (Streamlit app.py, some services helpers) keep working
    without a bulk rename.
    """
    trade: dict = {}
    for k, v in row.items():
        if k == 'data':
            if isinstance(v, str):
                try:
                    v = json.loads(v)
                except Exception:
                    v = {}
            if isinstance(v, dict) and v:
                trade.update(v)
        else:
            trade[k] = v
    # Legacy aliases for consumers that haven't migrated to *_fill_ts.
    if trade.get('entry_fill_ts') and not trade.get('entry_time'):
        trade['entry_time'] = trade['entry_fill_ts']
    if trade.get('exit_fill_ts') and not trade.get('exit_time'):
        trade['exit_time'] = trade['exit_fill_ts']
    return trade


def load_trades_admin(
    strategy_id: int,
    user_id: str | None = None,
    data_source_filter: str | None = None,
    created_at_gte: str | None = None,
) -> list:
    """Load all trades for a strategy (admin client). Ordered by entry_fill_ts.

    user_id filter is optional — with the FK + CASCADE we know every
    trade's user_id matches the strategy's user_id, but we accept the
    parameter so callers can still supply it for defence-in-depth.

    `data_source_filter` (2026-05-11 Phase 41): SQL LIKE pattern.
    Examples:
      - 'backtest_%' → only backtest trades
      - 'cache_%' → only cache-based algo trades
      - None → all rows (Phase 40 legacy behavior)

    `created_at_gte` (M-RS4 Fix 1a, 2026-07-01): optional ISO timestamp lower
    bound. When set, returns only rows with `created_at >= created_at_gte`
    (OR a NULL `created_at`). Lets `run_hifi_pass2`'s incremental pass push
    its `last_hifi_pass_at` watermark into SQL instead of loading every trade
    then filtering in Python — the reload-all was the dominant per-poll cost
    (14k+ rows for a fat strategy). The `is.null` clause mirrors the caller's
    defensive "keep rows with missing created_at" so the loaded set is
    byte-identical to the old load-all-then-filter path. Default None =
    unchanged full-load behavior for every other caller.

    PostgREST/Supabase caps every response at 1000 rows regardless of
    `limit` / `Range` headers (confirmed 2026-04-24). Strategies in this
    app can exceed 20k trades, so the loader paginates via `.range()`
    until a partial page comes back. Stops at a hard ceiling so a runaway
    query can't load the whole table.
    """
    client = get_admin_client()
    PAGE_SIZE = 1000
    MAX_TRADES = 200_000  # hard ceiling — flags a bug if ever hit
    all_rows: list = []
    offset = 0
    while offset < MAX_TRADES:
        q = client.table('trades') \
            .select('*') \
            .eq('strategy_id', strategy_id) \
            .order('entry_fill_ts') \
            .range(offset, offset + PAGE_SIZE - 1)
        if user_id:
            q = q.eq('user_id', user_id)
        if data_source_filter:
            q = q.like('data_source', data_source_filter)
        if created_at_gte:
            q = q.or_(f"created_at.gte.{created_at_gte},created_at.is.null")
        result = q.execute()
        page = result.data or []
        all_rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return [_row_to_trade(r) for r in all_rows]


def load_trades_kpi_fields_admin(
    strategy_id: int,
    user_id: str | None = None,
    data_source_filter: str | None = None,
) -> list:
    """Load ONLY the columns calculate_kpis() needs — admin client.

    Priority 1 perf fix (2026-05-20): the KPI recompute in
    forward_test_service's append lanes was doing `select('*')` —
    pulling every trade's full `data` JSONB blob (confluence_records,
    stop/target metadata, ~2KB/row). For a 5,000+ trade strategy that
    OOM-killed the API. KPIs only need 5 fields; this projects to them.

    Returns dicts shaped for `trades_df_from_stored` → `calculate_kpis`:
      - r_multiple, exit_reason, entry_fill_ts, exit_fill_ts (top-level cols)
      - win (projected out of the `data` JSONB via PostgREST `data->win`)

    Each row is ~120 bytes vs ~2KB for select('*') — a ~15-20x payload
    cut. KPIs computed from these are byte-identical to the full-load
    path (calculate_kpis only ever touches these 5 fields).
    """
    client = get_admin_client()
    PAGE_SIZE = 1000
    MAX_TRADES = 200_000
    all_rows: list = []
    offset = 0
    # PostgREST projection: top-level cols + `data->win` JSON field.
    _PROJECTION = 'r_multiple,exit_reason,entry_fill_ts,exit_fill_ts,data->win'
    while offset < MAX_TRADES:
        q = client.table('trades') \
            .select(_PROJECTION) \
            .eq('strategy_id', strategy_id) \
            .order('entry_fill_ts') \
            .range(offset, offset + PAGE_SIZE - 1)
        if user_id:
            q = q.eq('user_id', user_id)
        if data_source_filter:
            q = q.like('data_source', data_source_filter)
        result = q.execute()
        page = result.data or []
        all_rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return all_rows


def get_max_entry_ts_admin(
    strategy_id: int,
    user_id: str | None = None,
    data_source_filter: str | None = None,
) -> str | None:
    """Return the latest entry_fill_ts for a strategy's trades.

    Used by the BT-APPEND optimization (2026-05-12) — find the append
    anchor without paginating through the entire backtest history.
    Replaces "load all stored_trades, scan in Python" with a single
    ORDER BY DESC LIMIT 1 query.

    `data_source_filter`: SQL LIKE pattern. Same semantics as
    load_trades_admin.

    Returns the ISO timestamp string, or None if no rows match.
    """
    client = get_admin_client()
    q = (client.table('trades')
         .select('entry_fill_ts')
         .eq('strategy_id', strategy_id)
         .order('entry_fill_ts', desc=True)
         .limit(1))
    if user_id:
        q = q.eq('user_id', user_id)
    if data_source_filter:
        q = q.like('data_source', data_source_filter)
    result = q.execute()
    rows = result.data or []
    if not rows:
        return None
    ts = rows[0].get('entry_fill_ts')
    return str(ts) if ts else None


def insert_trade_admin(strategy_id: int, user_id: str, trade: dict) -> dict | None:
    """Insert a single trade row (admin client). Returns the saved row
    reconstructed into legacy flat shape, or None on conflict / failure.

    Wrapped in _execute_with_retry for transient Supabase errors. Unique
    violations short-circuit (not retried) and return None.
    """
    row = _trade_to_row(trade)
    row['strategy_id'] = strategy_id
    row['user_id'] = user_id
    row.pop('id', None)
    row.pop('created_at', None)
    client = get_admin_client()

    def _do_insert():
        try:
            return client.table('trades').insert(row).execute()
        except Exception as e:
            msg = str(e).lower()
            if 'duplicate' in msg or 'unique' in msg or 'conflict' in msg:
                return None
            raise

    result = _execute_with_retry(
        _do_insert, op_name=f'insert_trade_admin sid={strategy_id}')
    if result is None:
        return None
    return _row_to_trade(result.data[0]) if result.data else None


def delete_trades_for_strategy_admin(strategy_id: int) -> int:
    """Delete every trade row for a strategy. Returns deleted row count."""
    client = get_admin_client()
    result = client.table('trades') \
        .delete() \
        .eq('strategy_id', strategy_id) \
        .execute()
    return len(result.data) if result.data else 0


def delete_trade_placeholder_admin(strategy_id: int, entry_fill_ts) -> int:
    """Delete any open-position placeholder row for (strategy, entry_fill_ts).

    forward_test_service may emit a row with exit_fill_ts IS NULL while a
    position is still held. Once the exit fires, the worker wants to
    replace that placeholder with the closed trade rather than leave
    both in place. Returns deleted row count.
    """
    if entry_fill_ts is None or entry_fill_ts == '':
        return 0
    client = get_admin_client()
    result = client.table('trades') \
        .delete() \
        .eq('strategy_id', strategy_id) \
        .eq('entry_fill_ts', entry_fill_ts) \
        .is_('exit_fill_ts', 'null') \
        .execute()
    return len(result.data) if result.data else 0


def replace_trades_admin(
    strategy_id: int,
    user_id: str,
    trades: list[dict],
    data_source_filter: str | None = None,
    chunk_size: int = 250,
) -> int:
    """Replace trades for a strategy (admin client).

    DELETE followed by chunked bulk INSERT. Used by the forward-test
    recompute and Mass Builder backtest-save paths. Returns inserted
    row count.

    **Retry behavior (added 2026-05-11):** The entire DELETE+INSERT
    operation is idempotent on retry — DELETE is a no-op when nothing
    matches, INSERT is wrapped per-chunk with its own retry. Supabase
    has been intermittently dropping requests with 522 errors; this
    wrapper keeps a transient blip from leaving the trades table in a
    partial-write state.

    Chunked INSERT (default 250 rows/chunk) reduces single-call payload
    size, which both speeds up each individual call and keeps Supabase
    happier under load.

    `data_source_filter` (2026-05-11 Phase 41): SQL LIKE pattern to
    scope the DELETE. Examples:
      - 'backtest_%' → only deletes backtest rows; preserves algo
      - 'cache_%' → only deletes cache/algo rows; preserves backtest
      - None → deletes ALL rows for strategy (Phase 40 legacy)
    """
    client = get_admin_client()

    def _do_delete():
        delete_q = client.table('trades').delete().eq('strategy_id', strategy_id)
        if data_source_filter:
            delete_q = delete_q.like('data_source', data_source_filter)
        return delete_q.execute()

    _execute_with_retry(
        _do_delete,
        op_name=f'replace_trades_admin DELETE sid={strategy_id} '
                f'filter={data_source_filter!r}',
    )

    if not trades:
        return 0

    rows: list[dict] = []
    for trade in trades:
        row = _trade_to_row(trade)
        row['strategy_id'] = strategy_id
        row['user_id'] = user_id
        row.pop('id', None)
        row.pop('created_at', None)
        rows.append(row)

    total_inserted = 0
    total_chunks = (len(rows) + chunk_size - 1) // chunk_size
    for idx in range(0, len(rows), chunk_size):
        chunk = rows[idx:idx + chunk_size]
        chunk_num = (idx // chunk_size) + 1

        def _do_insert_chunk(_chunk=chunk):
            return client.table('trades').insert(_chunk).execute()

        result = _execute_with_retry(
            _do_insert_chunk,
            op_name=(
                f'replace_trades_admin INSERT sid={strategy_id} '
                f'chunk {chunk_num}/{total_chunks} '
                f'({len(chunk)} rows)'
            ),
        )
        total_inserted += len(result.data) if result.data else len(chunk)

    return total_inserted


def replace_trades_in_window_admin(
    strategy_id: int,
    user_id: str,
    trades: list[dict],
    data_source_filter: str,
    entry_floor_iso: str,
    entry_ceil_iso: str,
    chunk_size: int = 250,
) -> int:
    """Replace the trades in a bounded entry-time WINDOW for one lane.

    The B2 fix (2026-06-15): "Update New Data" appends under-generate
    trades near the data edge (windowed engine recompute on unsettled
    REST + imperfect resume warmup), and the INSERT-only dedup on
    (strategy_id, entry_fill_ts, exit_fill_ts) means the fresh, settled
    recompute can never REPLACE the stale rows. This primitive lets each
    append REWRITE its trailing edge band: DELETE every `data_source_filter`
    row with entry_fill_ts in [floor, ceil) then INSERT the freshly
    recomputed set.

    Scope is (strategy_id AND data_source LIKE filter AND entry_fill_ts in
    window) so it never touches the OTHER lane (backtest_% vs cache_%) nor
    settled history below the band.

    Concurrency (Kevin's call 2026-06-15): the API and the live worker can
    append the same strategy from SEPARATE processes, so an in-process lock
    can't serialize them, and the Supabase client runs DELETE and INSERT as
    separate HTTP requests (no shared txn) — a naive delete-then-insert can
    briefly expose an empty band that reads as a phantom spike. The atomic
    path is the `replace_trades_in_window` Postgres function
    (migrations/append_edge_band_replace_rpc.sql) which wraps
    pg_advisory_xact_lock + DELETE + INSERT in ONE transaction. If that
    function isn't installed yet, we fall back to client-side
    delete-then-insert (sub-second window; fine for manual one-at-a-time
    verification before the migration is run).

    Returns inserted row count.
    """
    client = get_admin_client()

    # Shape rows once (same as replace_trades_admin / insert path).
    rows: list[dict] = []
    for trade in trades:
        row = _trade_to_row(trade)
        row['strategy_id'] = strategy_id
        row['user_id'] = user_id
        row.pop('id', None)
        row.pop('created_at', None)
        rows.append(row)

    # NO-WIPE GUARD (2026-06-16): never DELETE the window when there's nothing
    # to insert — an empty `trades` (e.g. caller's recompute silently returned
    # 0 on a failed fetch) would otherwise delete-then-insert-nothing and WIPE
    # the band. Callers (_replace_edge_band) already guard, but make the
    # primitive inherently safe too. To intentionally CLEAR a window, delete
    # explicitly elsewhere.
    if not rows:
        logger.info(
            "replace_trades_in_window_admin sid=%s: 0 rows — no-op (skip "
            "delete to avoid wiping the window)", strategy_id)
        return 0

    # ── Preferred path: atomic server-side RPC (advisory-locked txn).
    try:
        def _do_rpc():
            return client.rpc('replace_trades_in_window', {
                'p_strategy_id': strategy_id,
                'p_user_id': user_id,
                'p_ds_filter': data_source_filter,
                'p_entry_floor': entry_floor_iso,
                'p_entry_ceil': entry_ceil_iso,
                'p_rows': rows,
            }).execute()

        result = _execute_with_retry(
            _do_rpc,
            op_name=(f'replace_trades_in_window RPC sid={strategy_id} '
                     f'filter={data_source_filter!r} '
                     f'[{entry_floor_iso}..{entry_ceil_iso})'),
            max_attempts=1,  # don't retry a missing-function error
        )
        # RPC returns the inserted count as a scalar.
        val = result.data
        if isinstance(val, list):
            val = val[0] if val else 0
        return int(val) if val is not None else 0
    except Exception as e:
        msg = str(e).lower()
        # Function not installed yet → fall back. Any other error also
        # falls back (the band replace must not break the append).
        if not ('could not find' in msg or 'does not exist' in msg
                or 'pgrst202' in msg or '404' in msg or 'schema cache' in msg):
            logger.warning(
                "[BAND-REPLACE] sid=%s RPC failed (%s); using client-side "
                "fallback", strategy_id, e)

    # ── Fallback: client-side scoped DELETE then chunked INSERT.
    def _do_delete():
        return (client.table('trades').delete()
                .eq('strategy_id', strategy_id)
                .like('data_source', data_source_filter)
                .gte('entry_fill_ts', entry_floor_iso)
                .lt('entry_fill_ts', entry_ceil_iso)
                .execute())

    _execute_with_retry(
        _do_delete,
        op_name=(f'replace_trades_in_window_admin DELETE sid={strategy_id} '
                 f'filter={data_source_filter!r} '
                 f'[{entry_floor_iso}..{entry_ceil_iso})'),
    )

    if not rows:
        return 0

    total_inserted = 0
    total_chunks = (len(rows) + chunk_size - 1) // chunk_size
    for idx in range(0, len(rows), chunk_size):
        chunk = rows[idx:idx + chunk_size]
        chunk_num = (idx // chunk_size) + 1

        def _do_insert_chunk(_chunk=chunk):
            return client.table('trades').insert(_chunk).execute()

        result = _execute_with_retry(
            _do_insert_chunk,
            op_name=(
                f'replace_trades_in_window_admin INSERT sid={strategy_id} '
                f'chunk {chunk_num}/{total_chunks} ({len(chunk)} rows)'
            ),
        )
        total_inserted += len(result.data) if result.data else len(chunk)

    return total_inserted


# ============================================================
# Monitor Status CRUD (database path)
# ============================================================

def load_monitor_status_db() -> dict:
    """Load monitor status for the current user."""
    user_id = get_current_user_id()
    client = get_client()
    result = client.table('monitor_status') \
        .select('*') \
        .eq('user_id', user_id) \
        .maybe_single() \
        .execute()
    if result and result.data:
        row = result.data
        status = row.get('status', {})
        if isinstance(status, str):
            status = json.loads(status)
        status['desired_state'] = row.get('desired_state', 'stopped')
        return status
    return {'desired_state': 'stopped'}


def save_monitor_status_db(status: dict):
    """Upsert monitor status for the current user."""
    user_id = get_current_user_id()
    desired_state = status.pop('desired_state', None)
    client = get_client()
    row = {
        'user_id': user_id,
        'status': status,
        'updated_at': datetime.now(timezone.utc).isoformat(),
    }
    if desired_state:
        row['desired_state'] = desired_state
    client.table('monitor_status').upsert(
        row, on_conflict='user_id'
    ).execute()


def set_desired_state_db(desired_state: str):
    """Set the desired monitor state (called from web UI)."""
    user_id = get_current_user_id()
    client = get_client()
    client.table('monitor_status').upsert({
        'user_id': user_id,
        'desired_state': desired_state,
        'updated_at': datetime.now(timezone.utc).isoformat(),
    }, on_conflict='user_id').execute()


# ============================================================
# Engine State CRUD (database path — used by Ralph worker)
# ============================================================

def load_engine_state_db(user_id: str = None) -> dict:
    """Load engine state (positions). Worker uses admin client + explicit user_id."""
    if user_id:
        client = get_admin_client()
        result = client.table('monitor_status') \
            .select('engine_state') \
            .eq('user_id', user_id) \
            .maybe_single() \
            .execute()
    else:
        uid = get_current_user_id()
        client = get_client()
        result = client.table('monitor_status') \
            .select('engine_state') \
            .eq('user_id', uid) \
            .maybe_single() \
            .execute()
    if result and result.data:
        state = result.data.get('engine_state', {})
        return json.loads(state) if isinstance(state, str) else state
    return {'positions': {}}


def save_engine_state_db(state: dict, user_id: str = None):
    """Save engine state. Worker uses admin client + explicit user_id."""
    uid = user_id or get_current_user_id()
    client = get_admin_client() if user_id else get_client()
    client.table('monitor_status').upsert({
        'user_id': uid,
        'engine_state': state,
        'last_heartbeat': datetime.now(timezone.utc).isoformat(),
        'updated_at': datetime.now(timezone.utc).isoformat(),
    }, on_conflict='user_id').execute()


# ============================================================
# Engine Audit Log (database path — used by Ralph worker)
# ============================================================

def append_audit_log_db(entry: dict, user_id: str = None):
    """Append a fidelity audit entry. Worker uses admin client."""
    uid = user_id or get_current_user_id()
    client = get_admin_client() if user_id else get_client()
    client.table('engine_audit_log').insert({
        'user_id': uid,
        'ts': entry.get('ts'),
        'symbol': entry.get('symbol'),
        'timeframe': entry.get('tf'),
        'bar_close': entry.get('bar_close'),
        'indicators': entry.get('indicators', {}),
        'triggers': entry.get('triggers', {}),
        'interpreters': entry.get('interpreters', {}),
        'positions': entry.get('positions', {}),
    }).execute()


# ============================================================
# Tier 3 §8.3 (always-start-flat, 2026-05-20) — carryover persistence
# ============================================================

def append_position_carryovers_admin(carryovers: list, user_id: str):
    """Append `position_carryover` entries to each affected strategy's
    config.position_carryovers list. Worker uses admin client to bypass
    RLS — this is fire-and-forget from the engine's perspective.

    Each carryover dict must include a `strategy_id` field. Carryovers
    for the same strategy_id are batched into one strategy update.

    Per the `feedback_jsonb_partial_updates` rule, this loads each
    strategy via admin client → appends to position_carryovers in
    config → writes the FULL config back (no partial dict). Cap at 50
    entries per strategy so a long-running carryover history doesn't
    bloat the JSONB row indefinitely; older entries truncated FIFO.

    Defensive: per-strategy failures are logged but do not raise. The
    engine has already enforced FLAT — losing a UI-banner write is
    a degradation, not a correctness failure.
    """
    if not carryovers:
        return
    client = get_admin_client()
    # Group by strategy_id
    by_sid: dict[int, list] = {}
    for c in carryovers:
        sid = c.get('strategy_id')
        if sid is None:
            continue
        by_sid.setdefault(sid, []).append(c)

    import logging as _l
    log = _l.getLogger(__name__)

    for sid, new_entries in by_sid.items():
        try:
            # Load current strategy
            res = client.table('strategies') \
                .select('config') \
                .eq('id', sid).eq('user_id', user_id) \
                .maybe_single().execute()
            if not (res and res.data):
                log.warning(
                    "Tier 3 carryover: strategy %s not found for user %s",
                    sid, user_id[:8])
                continue
            cfg = res.data.get('config') or {}
            if isinstance(cfg, str):
                import json as _j
                cfg = _j.loads(cfg)
            existing = list(cfg.get('position_carryovers') or [])
            existing.extend(new_entries)
            # Truncate FIFO at 50 — keep most recent
            if len(existing) > 50:
                existing = existing[-50:]
            cfg['position_carryovers'] = existing
            # Direct config JSONB update — full config dict so we don't
            # trip the partial-update wipe guard.
            client.table('strategies') \
                .update({'config': cfg}) \
                .eq('id', sid).eq('user_id', user_id) \
                .execute()
            log.info(
                "Tier 3 carryover persisted for sid=%s (now %d total)",
                sid, len(existing))
        except Exception as e:
            log.error(
                "Tier 3 carryover persist failed sid=%s: %s",
                sid, e, exc_info=True)


# ============================================================
# Admin-only functions (worker service — bypasses RLS)
# ============================================================

def load_all_desired_states() -> list:
    """Load desired_state for all users. Worker polls this to start/stop engines."""
    client = get_admin_client()
    result = client.table('monitor_status') \
        .select('user_id, desired_state, updated_at') \
        .execute()
    return result.data


def load_alert_config_admin(user_id: str) -> dict:
    """Load alert config for a specific user (admin client)."""
    client = get_admin_client()
    result = client.table('alert_config') \
        .select('config, updated_at') \
        .eq('user_id', user_id) \
        .maybe_single() \
        .execute()
    if result and result.data:
        config = result.data['config']
        config = json.loads(config) if isinstance(config, str) else config
        config['_updated_at'] = result.data.get('updated_at', '')
        return config
    return {"global": {}, "strategies": {}, "portfolios": {}}


def load_strategies_admin(user_id: str) -> list:
    """Load all strategies for a specific user (admin client).

    Phase 40: hydrate_trades=False because Worker's "no strategies to
    monitor" diagnostic fallback only reads id/name/symbol-level fields;
    fetching trades per strategy here would spike Supabase load for a
    debug path. Callers that need trades should use
    ``get_strategy_by_id_admin`` for the specific strategy.
    """
    client = get_admin_client()
    result = client.table('strategies') \
        .select('*') \
        .eq('user_id', user_id) \
        .order('id') \
        .execute()
    return [_row_to_strategy(r, hydrate_trades=False) for r in result.data]


# Columns needed for Worker monitoring decisions (triggers, confluence,
# session, timeframe, alert tracking). Explicitly EXCLUDES the three
# JSONB columns that can grow into the megabytes per strategy:
#   - stored_trades         (trade history; single strategy observed at 2.56 MB)
#   - equity_curve_data     (per-trade cumulative-R points; ~70 KB typical)
#   - live_executions       (position-state mirror; grows with trade count)
# Ralph engine doesn't touch any of those three at monitoring time, and the
# one worker path that legitimately needs stored_trades (_persist_algo_trade)
# uses get_strategy_by_id_admin for a single row — not this bulk path.
# Update this list whenever a column is added to the strategies schema.
_STRATEGY_MONITOR_COLUMNS = (
    'id,user_id,name,symbol,direction,timeframe,'
    'config,kpis,'
    'alert_tracking_enabled,alert_tracking_reset_at,'
    'created_at,updated_at,data_refreshed_at,'
    'discrepancies,discrepancies_dismissed_at,'
    'forward_test_start,forward_testing,strategy_origin'
)


def load_strategies_monitoring_admin(user_id: str) -> list:
    """Load all strategies for a user, excluding heavy JSONB columns.

    Use this variant for the Worker's polling / hot-reload path where we
    only need monitoring metadata (triggers, confluence, timeframe). On
    a strategy with ~3000 trades, this is ~2.5 MB / strategy cheaper than
    ``load_strategies_admin`` — a big deal when the Worker re-polls every
    hot-reload cycle and Supabase statement-timeouts are the limiting
    factor (see roadmap 9an).

    If you need trade history, equity_curve_data, or live_executions, use
    ``load_strategies_admin`` or ``get_strategy_by_id_admin`` instead.
    """
    client = get_admin_client()
    result = client.table('strategies') \
        .select(_STRATEGY_MONITOR_COLUMNS) \
        .eq('user_id', user_id) \
        .order('id') \
        .execute()
    # hydrate_trades=False — monitor cycle never reads trades.
    return [_row_to_strategy(r, hydrate_trades=False) for r in result.data]


def get_strategy_by_id_admin(strategy_id: int, user_id: str) -> dict | None:
    """Load a single strategy by ID for a specific user (admin client)."""
    client = get_admin_client()
    result = client.table('strategies') \
        .select('*') \
        .eq('id', strategy_id) \
        .eq('user_id', user_id) \
        .maybe_single() \
        .execute()
    if result and result.data:
        return _row_to_strategy(result.data)
    return None


def update_strategy_admin(strategy_id: int, user_id: str, updates: dict) -> dict | None:
    """Update specific fields on a strategy (admin client, no JWT required).

    Used by the worker to persist algo trade records into stored_trades as
    they fire live. `updates` is a PARTIAL dict — pass only the fields you
    want to change (typically `stored_trades`, `live_executions`, `kpis`).

    CRITICAL: this function MUST NOT use `_strategy_to_row()` directly because
    that helper always emits `config = {}` when no config fields are present,
    which would overwrite the existing JSONB column and wipe trigger/stop/
    target settings. We split fields into column-vs-config buckets manually
    and only include `config` in the update payload when config fields are
    actually being changed (in which case the caller must pass a FULL config
    merge — partial JSONB updates aren't supported by PostgREST without RPC).
    """
    column_updates = {}
    config_updates = {}
    for k, v in updates.items():
        if k in STRATEGY_COLUMN_FIELDS:
            column_updates[k] = v
        else:
            config_updates[k] = v
    column_updates.pop('user_id', None)
    column_updates.pop('id', None)

    # Phase 40: when the trades-table is authoritative, strip stored_trades
    # out of the strategies UPDATE payload and write to the trades table
    # instead. Only acts when 'stored_trades' is explicitly in the incoming
    # updates dict — omitting it means "don't touch trades" (preserves the
    # feedback_jsonb_partial_updates safety rule).
    pending_trades = None
    try:
        import trades_store as _trades_store
        if _trades_store._flag_on() and 'stored_trades' in column_updates:
            pending_trades = column_updates.pop('stored_trades', None) or []
    except Exception:
        _trades_store = None

    payload = dict(column_updates)
    if config_updates:
        # Caller is changing something inside config — they must have already
        # merged with the existing config. We can't do partial JSONB updates
        # via the supabase-py client, so the caller's responsibility.
        payload['config'] = config_updates

    if not payload and pending_trades is None:
        return None  # nothing to update

    saved = None
    if payload:
        client = get_admin_client()
        result = client.table('strategies') \
            .update(payload) \
            .eq('id', strategy_id) \
            .eq('user_id', user_id) \
            .execute()
        if result.data:
            saved = _row_to_strategy(result.data[0])

    if pending_trades is not None and _trades_store is not None and _trades_store._flag_on():
        try:
            _trades_store.replace_trades_for_strategy(
                strategy_id, user_id, pending_trades)
            if saved is not None:
                saved['stored_trades'] = pending_trades
        except Exception as e:
            import logging as _l
            _l.getLogger(__name__).warning(
                "update_strategy_admin: trades-table replace failed for "
                "strategy %s: %s", strategy_id, e)

    return saved


def load_portfolios_admin(user_id: str) -> list:
    """Load all portfolios for a specific user (admin client)."""
    client = get_admin_client()
    result = client.table('portfolios') \
        .select('*') \
        .eq('user_id', user_id) \
        .order('id') \
        .execute()
    # Apply the same JSONB → top-level restoration as the user-scoped path so
    # the worker sees change_log / journal_entries / buying_power_mode at the
    # top level, not buried inside account as `_p37_*` keys.
    return [_restore_portfolio_from_db(r) for r in result.data]


def save_alert_admin(alert: dict, user_id: str) -> dict:
    """Insert an alert for a specific user (admin client)."""
    alert['user_id'] = user_id
    row = _alert_to_row(alert)
    row.pop('id', None)
    client = get_admin_client()
    result = client.table('alerts').insert(row).execute()
    return _row_to_alert(result.data[0])


def save_monitor_status_admin(user_id: str, status: dict):
    """Update monitor status for a specific user (admin client)."""
    now = datetime.now(timezone.utc).isoformat()
    desired_state = status.pop('desired_state', None)
    row = {
        'user_id': user_id,
        'status': status,
        'last_heartbeat': now,
        'updated_at': now,
    }
    if desired_state:
        row['desired_state'] = desired_state
    client = get_admin_client()
    client.table('monitor_status').upsert(
        row, on_conflict='user_id'
    ).execute()


def load_user_settings_admin(user_id: str) -> dict:
    """Load user settings for a specific user (admin client, for Alpaca keys)."""
    client = get_admin_client()
    result = client.table('user_settings') \
        .select('settings') \
        .eq('user_id', user_id) \
        .maybe_single() \
        .execute()
    if result and result.data:
        val = result.data['settings']
        return json.loads(val) if isinstance(val, str) else val
    return {}


def _load_general_packs_admin_uncached(user_id: str) -> list:
    client = get_admin_client()
    result = client.table('general_packs') \
        .select('packs') \
        .eq('user_id', user_id) \
        .maybe_single() \
        .execute()
    if result and result.data:
        val = result.data['packs']
        return json.loads(val) if isinstance(val, str) else val
    return []


def load_general_packs_admin(user_id: str) -> list:
    """Load general packs for a specific user (admin client).

    M-RS4 Fix 1b: memoized per user_id (worker-path fallback used when there is no
    thread-local user context). Default OFF → uncached. Distinct namespace from the
    context-scoped loader because this returns raw JSON, not parsed GeneralPack objs.
    """
    import config_cache
    return config_cache.cached(
        "general_packs_admin_raw",
        lambda: _load_general_packs_admin_uncached(user_id), uid=user_id)


def _load_confluence_groups_admin_uncached(user_id: str) -> list:
    client = get_admin_client()
    result = client.table('confluence_groups') \
        .select('groups') \
        .eq('user_id', user_id) \
        .maybe_single() \
        .execute()
    if result and result.data:
        val = result.data['groups']
        return json.loads(val) if isinstance(val, str) else val
    return []


def load_confluence_groups_admin(user_id: str) -> list:
    """Load confluence groups for a specific user (admin client).

    M-RS4 Fix 1b: memoized per user_id (worker-path fallback used when there is no
    thread-local user context). Default OFF → uncached. Distinct namespace from the
    context-scoped loader because this returns raw JSON, not parsed ConfluenceGroups.
    """
    import config_cache
    return config_cache.cached(
        "confluence_groups_admin_raw",
        lambda: _load_confluence_groups_admin_uncached(user_id), uid=user_id)


def get_monitored_strategies_db(user_id: str) -> list:
    """Resolve which strategies to monitor for a user (admin client).

    M8.5 change: every strategy with the structural prerequisites (i.e. an
    ``entry_trigger_confluence_id``) is monitored by default, regardless of
    portfolio membership or webhook wiring. This lets alerts + forward-test
    data accumulate for every saved strategy, and lets the live chart fire
    broadcasts for any strategy the user opens on Strategy Detail.

    Webhook dispatch still respects portfolio + webhook-group config at alert
    time — monitoring here is strictly about "does the engine watch this?"
    """
    import logging
    _log = logging.getLogger("worker")

    # Use the lite variant (no stored_trades / equity_curve_data /
    # live_executions) — monitoring decisions only need config + trigger
    # fields, and the Worker polls this every hot-reload cycle.
    strategies = load_strategies_monitoring_admin(user_id)
    _log.info("[%s] get_monitored: scanning %d strategies",
              user_id[:8], len(strategies))

    monitored = []
    skipped_no_confluence = 0
    for strat in strategies:
        if 'entry_trigger_confluence_id' not in strat:
            skipped_no_confluence += 1
            continue
        monitored.append(strat)

    if skipped_no_confluence:
        _log.info("[%s]   %d strategies skipped (no entry_trigger_confluence_id — "
                  "legacy/unmigrated)", user_id[:8], skipped_no_confluence)
    _log.info("[%s] get_monitored result: %d strategies", user_id[:8], len(monitored))
    return monitored


# ============================================================
# Mass Search CRUD
# ============================================================

_MASS_SEARCHES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "config", "mass_searches.json")


def _load_mass_searches_file() -> list:
    if os.path.exists(_MASS_SEARCHES_FILE):
        try:
            with open(_MASS_SEARCHES_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception):
            pass
    return []


def _save_mass_searches_file(searches: list):
    os.makedirs(os.path.dirname(_MASS_SEARCHES_FILE), exist_ok=True)
    with open(_MASS_SEARCHES_FILE, 'w') as f:
        json.dump(searches, f, indent=2, default=str)


def load_mass_searches() -> list:
    """Load all mass searches for the current user.

    Merges row-level fields (id, status, created_at, updated_at) with the
    config_data JSONB payload so callers see a unified dict. Row-level status
    is authoritative — config_data's status may lag behind.

    In USE_DB mode we RAISE on DB errors instead of silently falling back to
    the file-mode cache. On Railway there is no file, so the fallback used
    to return [] and the frontend would render "No saved searches yet" —
    indistinguishable from a user who genuinely has no searches. Raising
    lets the API return 500, which React Query treats as an error (keeping
    the previously-fetched data visible) rather than as an empty list.
    """
    if USE_DB:
        client = get_client()
        result = client.table('mass_searches') \
            .select('*') \
            .order('created_at', desc=True) \
            .execute()
        merged = []
        for r in (result.data or []):
            row = dict(r)
            cfg = row.pop('config_data', {}) or {}
            merged.append({**cfg, **row})
        return merged
    return _load_mass_searches_file()


def get_mass_search(search_id) -> dict | None:
    """Get a single mass search by ID."""
    if USE_DB:
        try:
            client = get_client()
            result = client.table('mass_searches') \
                .select('*') \
                .eq('id', search_id) \
                .maybe_single() \
                .execute()
            if result.data:
                # Merge row-level fields with config_data for a unified view
                row = dict(result.data)
                cfg = row.pop('config_data', {}) or {}
                merged = {**cfg, **row}
                return merged
        except Exception as e:
            logger.warning("get_mass_search DB error: %s", e)
    for s in _load_mass_searches_file():
        if s.get('id') == search_id:
            return s
    return None


def save_mass_search(search: dict) -> str:
    """Save (upsert) a mass search. Returns the search ID."""
    search_id = search.get('id') or None
    now = datetime.now(timezone.utc).isoformat()
    search['updated_at'] = now
    if not search.get('created_at'):
        search['created_at'] = now

    if USE_DB:
        try:
            client = get_client()
            # Historical: this used to reference a module-level
            # threading.local() named `_local`. The user-context layer
            # migrated to ContextVar on 2026-04-22 (see top of this
            # module), but this reference was missed — stayed broken
            # silently because the except clause falls back to file
            # storage, which on Railway is the API container's ephemeral
            # filesystem. Mass Builder progress tracking then polls the
            # DB and never finds the search, so the UI hangs. Fix:
            # use get_current_user_id().
            user_id = get_current_user_id()
            row = {
                'user_id': user_id,
                'name': search.get('name', 'Untitled'),
                'status': search.get('status', 'pending'),
                'config_data': json.loads(json.dumps(search, default=str)),
                'created_at': search['created_at'],
                'updated_at': now,
            }
            if search_id:
                row['id'] = search_id
                client.table('mass_searches').upsert(row).execute()
                return search_id
            else:
                # Generate a unique integer ID (timestamp-based)
                import time as _time
                row['id'] = int(_time.time() * 1000) % (2**31)
                result = client.table('mass_searches').insert(row).execute()
                if result.data and len(result.data) > 0:
                    return result.data[0].get('id')
                return row['id']
        except Exception as e:
            logger.warning("save_mass_search DB error (falling back to file): %s", e)

    # Local JSON fallback
    searches = _load_mass_searches_file()
    existing = next((i for i, s in enumerate(searches) if s.get('id') == search_id), None)
    if existing is not None:
        searches[existing] = search
    else:
        searches.insert(0, search)
    _save_mass_searches_file(searches)
    return search_id


def update_mass_search(search_id, updates: dict):
    """Partial update of a mass search row in the DB."""
    if USE_DB:
        try:
            client = get_client()
            row = {'updated_at': datetime.now(timezone.utc).isoformat()}
            if 'status' in updates:
                row['status'] = updates['status']
            # Store results, progress, summary, checkpoint in config_data.
            # `checkpoint` is the resume-from-checkpoint payload (Roadmap 9s):
            # completed_symbol_tfs + partial_results + diagnostics_so_far.
            _jsonb_keys = ('results', 'progress', 'summary', 'checkpoint')
            if any(k in updates for k in _jsonb_keys):
                # Read existing config_data, merge updates into it
                result = client.table('mass_searches').select('config_data').eq('id', search_id).maybe_single().execute()
                cfg = (result.data or {}).get('config_data', {}) if result and result.data else {}
                for key in _jsonb_keys:
                    if key in updates:
                        cfg[key] = updates[key]
                # Sanitize for JSON: convert numpy/pandas types
                import numpy as np
                def _sanitize(obj):
                    if isinstance(obj, (np.integer,)):
                        return int(obj)
                    if isinstance(obj, (np.floating,)):
                        return float(obj) if not np.isnan(obj) and not np.isinf(obj) else 0
                    if isinstance(obj, (np.bool_,)):
                        return bool(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, set):
                        return list(obj)
                    return str(obj)
                row['config_data'] = json.loads(json.dumps(cfg, default=_sanitize))
            client.table('mass_searches').update(row).eq('id', search_id).execute()
            return
        except Exception as e:
            logger.warning("update_mass_search DB error: %s", e)

    # File fallback
    existing = get_mass_search(search_id)
    if existing:
        existing.update(updates)
        save_mass_search(existing)


def delete_mass_search(search_id: str):
    """Delete a mass search by ID."""
    if USE_DB:
        try:
            client = get_client()
            client.table('mass_searches').delete().eq('id', search_id).execute()
            return
        except Exception as e:
            logger.warning("delete_mass_search DB error: %s", e)

    searches = _load_mass_searches_file()
    searches = [s for s in searches if s.get('id') != search_id]
    _save_mass_searches_file(searches)


# ============================================================
# Update Jobs CRUD (async on-demand UAD)
# ============================================================
# Backs the update_jobs table. Mirrors mass_searches pattern:
# background daemon thread updates progress_data as job runs; row
# survives worker restart via cleanup_orphaned_update_jobs on api boot.
# See migrations/update_jobs_table.sql.

def save_update_job(job: dict) -> int | None:
    """Save (upsert) an update job. Returns the job ID.

    `job` shape mirrors mass_searches usage:
      - top-level columns: name, mode, scope, strategy_ids, status
      - JSONB config_data: progress, summary, results, error
    """
    job_id = job.get('id')
    now = datetime.now(timezone.utc).isoformat()
    job['updated_at'] = now
    if not job.get('created_at'):
        job['created_at'] = now

    if not USE_DB:
        logger.warning("save_update_job: USE_DB is false, dropping job")
        return None

    try:
        client = get_client()
        user_id = get_current_user_id()
        row = {
            'user_id': user_id,
            'name': job.get('name', 'Untitled Update'),
            'mode': job.get('mode', 'new'),
            'scope': job.get('scope', 'single'),
            'strategy_ids': job.get('strategy_ids', []),
            'status': job.get('status', 'queued'),
            'config_data': json.loads(json.dumps(
                {k: v for k, v in job.items() if k not in (
                    'id', 'user_id', 'name', 'mode', 'scope',
                    'strategy_ids', 'status', 'created_at', 'updated_at',
                )},
                default=str
            )),
            'created_at': job['created_at'],
            'updated_at': now,
        }
        if job_id:
            row['id'] = job_id
            client.table('update_jobs').upsert(row).execute()
            return job_id
        # Timestamp-based int ID (matches mass_searches pattern)
        import time as _time
        row['id'] = int(_time.time() * 1000) % (2**31)
        result = client.table('update_jobs').insert(row).execute()
        if result.data and len(result.data) > 0:
            return result.data[0].get('id')
        return row['id']
    except Exception as e:
        logger.warning("save_update_job DB error: %s", e)
        return None


def get_update_job(job_id) -> dict | None:
    """Read a single update job row, merging config_data into top-level."""
    if not USE_DB:
        return None
    try:
        client = get_client()
        result = (client.table('update_jobs').select('*')
                  .eq('id', job_id).maybe_single().execute())
        if not result or not result.data:
            return None
        row = result.data
        # Flatten config_data into the top-level dict so callers can
        # access progress / summary / results / error directly.
        cfg = row.pop('config_data', {}) or {}
        if isinstance(cfg, str):
            try:
                cfg = json.loads(cfg)
            except Exception:
                cfg = {}
        row.update(cfg)
        return row
    except Exception as e:
        logger.warning("get_update_job DB error: %s", e)
        return None


def update_update_job(job_id, updates: dict):
    """Partial update of an update_jobs row.

    Top-level columns (status) are written directly. JSONB fields
    (progress, summary, results, error) are merged into config_data
    via read-modify-write to preserve other keys.
    """
    if not USE_DB:
        return
    try:
        client = get_client()
        row = {'updated_at': datetime.now(timezone.utc).isoformat()}
        if 'status' in updates:
            row['status'] = updates['status']
        _jsonb_keys = ('progress', 'summary', 'results', 'error')
        if any(k in updates for k in _jsonb_keys):
            res = (client.table('update_jobs').select('config_data')
                   .eq('id', job_id).maybe_single().execute())
            cfg = (res.data or {}).get('config_data', {}) if res and res.data else {}
            if isinstance(cfg, str):
                try:
                    cfg = json.loads(cfg)
                except Exception:
                    cfg = {}
            for key in _jsonb_keys:
                if key in updates:
                    cfg[key] = updates[key]
            # Sanitize for JSON (mirrors update_mass_search pattern)
            import numpy as _np
            def _sanitize(o):
                if isinstance(o, (_np.integer,)):
                    return int(o)
                if isinstance(o, (_np.floating,)):
                    return float(o) if not _np.isnan(o) and not _np.isinf(o) else 0
                if isinstance(o, (_np.bool_,)):
                    return bool(o)
                if isinstance(o, _np.ndarray):
                    return o.tolist()
                if isinstance(o, set):
                    return list(o)
                return str(o)
            row['config_data'] = json.loads(json.dumps(cfg, default=_sanitize))
        client.table('update_jobs').update(row).eq('id', job_id).execute()
    except Exception as e:
        logger.warning("update_update_job DB error: %s", e)


def load_update_jobs(limit: int = 50) -> list:
    """Return recent update jobs for current user, newest first."""
    if not USE_DB:
        return []
    try:
        client = get_client()
        result = (client.table('update_jobs').select('*')
                  .order('created_at', desc=True).limit(limit).execute())
        rows = result.data or []
        # Flatten config_data so list-view callers see uniform fields.
        out = []
        for row in rows:
            cfg = row.pop('config_data', {}) or {}
            if isinstance(cfg, str):
                try:
                    cfg = json.loads(cfg)
                except Exception:
                    cfg = {}
            row.update(cfg)
            out.append(row)
        return out
    except Exception as e:
        logger.warning("load_update_jobs DB error: %s", e)
        return []


def delete_update_job(job_id) -> bool:
    """Delete a job. Returns True if a row was deleted, False otherwise."""
    if not USE_DB:
        return False
    try:
        client = get_client()
        result = client.table('update_jobs').delete().eq('id', job_id).execute()
        return bool(result.data)
    except Exception as e:
        logger.warning("delete_update_job DB error: %s", e)
        return False


def load_orphaned_update_jobs() -> list:
    """Return jobs marked status='running' but with no in-process executor.

    Called at api startup — finds rows whose worker died before completion
    (e.g., Railway redeploy mid-job) so they can be marked 'orphaned' and
    surfaced in the UI for the user to retry. Admin client (no RLS) so
    we see all users' jobs.
    """
    if not USE_DB:
        return []
    try:
        client = get_admin_client()
        result = (client.table('update_jobs').select('*')
                  .eq('status', 'running').execute())
        return result.data or []
    except Exception as e:
        logger.warning("load_orphaned_update_jobs DB error: %s", e)
        return []


def mark_update_job_orphaned(job_id, error_msg: str = "worker died mid-job") -> None:
    """Mark a stuck 'running' job as 'orphaned' from admin context.

    Used at api startup for cleanup. No RLS (admin client) so it works
    across all users even when invoked from boot without a user context.
    """
    if not USE_DB:
        return
    try:
        client = get_admin_client()
        cfg_q = (client.table('update_jobs').select('config_data')
                 .eq('id', job_id).maybe_single().execute())
        cfg = (cfg_q.data or {}).get('config_data', {}) if cfg_q and cfg_q.data else {}
        if isinstance(cfg, str):
            try:
                cfg = json.loads(cfg)
            except Exception:
                cfg = {}
        cfg['error'] = error_msg
        (client.table('update_jobs')
         .update({
             'status': 'orphaned',
             'config_data': cfg,
             'updated_at': datetime.now(timezone.utc).isoformat(),
         })
         .eq('id', job_id).execute())
    except Exception as e:
        logger.warning("mark_update_job_orphaned DB error: %s", e)


# ===========================================================================
# system_settings — platform-wide runtime flags
# ===========================================================================

def get_system_setting(key: str, default=None):
    """Read a single system_settings value. Returns `default` on miss or error.

    Uses the admin client so the Data Worker (no JWT) can read.
    """
    if not USE_DB:
        return default
    try:
        client = get_admin_client()
        r = (client.table('system_settings').select('value')
             .eq('key', key).maybe_single().execute())
        if not r or not r.data:
            return default
        val = r.data.get('value')
        # JSONB column — Supabase python client deserializes, but if it
        # ever comes back as a string fall through to json.loads.
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except Exception:
                pass
        return val
    except Exception as e:
        logger.warning("get_system_setting(%s) failed: %s", key, e)
        return default


def list_system_settings() -> list:
    """List all system_settings rows. Admin client."""
    if not USE_DB:
        return []
    try:
        client = get_admin_client()
        r = (client.table('system_settings')
             .select('key,value,description,updated_at,updated_by')
             .order('key').execute())
        return r.data or []
    except Exception as e:
        logger.warning("list_system_settings failed: %s", e)
        return []


# ===========================================================================
# bar_diagnostics — per-bar indicator state for live/backtest divergence diag
# ===========================================================================

def save_bar_diagnostics_batch(rows: list[dict]) -> int:
    """Batch-upsert per-bar diagnostic state. Returns rows-written count.

    Each row must contain: strategy_id (int), bar_ts (ISO str), source
    (live|backtest|algo), values (dict). Uses the admin client so engines
    (no JWT) can write. `on_conflict=(strategy_id,bar_ts,source)` so a
    backtest re-run cleanly overwrites a prior recompute for the same
    bar.

    Empty input is a no-op (returns 0). Failures are logged but never
    raised — diagnostic logging is best-effort and must never block the
    engine's hot path.
    """
    if not USE_DB or not rows:
        return 0
    try:
        client = get_admin_client()
        # Normalize rows — drop any with missing required fields.
        norm = []
        for r in rows:
            sid = r.get('strategy_id')
            bts = r.get('bar_ts')
            src = r.get('source')
            vals = r.get('values')
            if sid is None or not bts or not src or not isinstance(vals, dict):
                continue
            norm.append({
                'strategy_id': int(sid),
                'bar_ts': bts if isinstance(bts, str) else bts.isoformat(),
                'source': src,
                'values': vals,
            })
        if not norm:
            return 0
        (client.table('bar_diagnostics')
         .upsert(norm, on_conflict='strategy_id,bar_ts,source')
         .execute())
        return len(norm)
    except Exception as e:
        logger.warning("save_bar_diagnostics_batch failed: %s", e)
        return 0


def load_bar_diagnostics(
    strategy_id: int,
    start_ts: str | None = None,
    end_ts: str | None = None,
    source: str | None = None,
    limit: int = 10000,
) -> list[dict]:
    """Read bar diagnostics for a strategy. Admin client.

    Returns rows sorted by (bar_ts, source). Used by the admin
    comparison UI (57E) and ad-hoc probes.
    """
    if not USE_DB:
        return []
    try:
        client = get_admin_client()
        q = client.table('bar_diagnostics').select(
            'strategy_id,bar_ts,source,values,created_at'
        ).eq('strategy_id', strategy_id)
        if start_ts:
            q = q.gte('bar_ts', start_ts)
        if end_ts:
            q = q.lte('bar_ts', end_ts)
        if source:
            q = q.eq('source', source)
        r = q.order('bar_ts').order('source').limit(limit).execute()
        return r.data or []
    except Exception as e:
        logger.warning("load_bar_diagnostics failed: %s", e)
        return []


def set_system_setting(key: str, value, *, description: str | None = None,
                       updated_by: str | None = None) -> bool:
    """Upsert a system_settings key. Returns True on success.

    `value` is stored as JSONB; pass Python primitives (bool/int/str/list/dict).
    `updated_by` is the user_id who flipped it (for audit). Uses admin client.
    """
    if not USE_DB:
        return False
    try:
        client = get_admin_client()
        payload = {
            'key': key,
            'value': value,
            'updated_at': datetime.now(timezone.utc).isoformat(),
        }
        if description is not None:
            payload['description'] = description
        if updated_by is not None:
            payload['updated_by'] = updated_by
        client.table('system_settings').upsert(payload, on_conflict='key').execute()
        return True
    except Exception as e:
        logger.warning("set_system_setting(%s) failed: %s", key, e)
        return False


# ============================================================
# User Pack Parity Status CRUD
# ============================================================
# Backs the user_pack_parity_status table — one row per
# (user_id, pack_slug) holding the last 4-quadrant parity-test
# result. See migrations/user_pack_parity_status.sql.

def load_pack_parity_status(pack_slug: str, user_id: str) -> dict | None:
    """Return the most recent saved parity-test result for this
    (user, pack) combo, or None if the user has never run the test."""
    try:
        client = get_admin_client()
        r = (client.table('user_pack_parity_status')
             .select('*')
             .eq('user_id', user_id)
             .eq('pack_slug', pack_slug)
             .limit(1)
             .execute())
        rows = r.data or []
        return rows[0] if rows else None
    except Exception as e:
        logger.warning(
            "load_pack_parity_status(%s, %s) failed: %s",
            pack_slug, user_id[:8] if user_id else '?', e)
        return None


def save_pack_parity_status(
    pack_slug: str,
    user_id: str,
    overall_verdict: str,
    summary: str,
    quadrants: dict,
    test_config: dict,
) -> dict | None:
    """Upsert the parity-test result for this (user, pack) combo.

    Idempotent: re-running overwrites the row. Tested_at is set to now
    on every upsert (independent of created_at).
    """
    from datetime import datetime, timezone
    if overall_verdict not in ('PASS', 'WARN', 'FAIL'):
        raise ValueError(
            f"overall_verdict must be PASS|WARN|FAIL, got {overall_verdict!r}")
    payload = {
        'user_id': user_id,
        'pack_slug': pack_slug,
        'overall_verdict': overall_verdict,
        'summary': summary or '',
        'quadrants': quadrants or {},
        'test_config': test_config or {},
        'tested_at': datetime.now(timezone.utc).isoformat(),
    }
    try:
        client = get_admin_client()
        # Upsert by (user_id, pack_slug) unique constraint
        r = (client.table('user_pack_parity_status')
             .upsert(payload, on_conflict='user_id,pack_slug')
             .execute())
        rows = r.data or []
        return rows[0] if rows else None
    except Exception as e:
        logger.warning(
            "save_pack_parity_status(%s, %s) failed: %s",
            pack_slug, user_id[:8] if user_id else '?', e)
        return None


def list_pack_parity_statuses(user_id: str) -> list:
    """Return all parity statuses for the user (one per pack). Used by
    the user_packs LIST endpoint to surface verdicts inline."""
    try:
        client = get_admin_client()
        r = (client.table('user_pack_parity_status')
             .select('*')
             .eq('user_id', user_id)
             .execute())
        return r.data or []
    except Exception as e:
        logger.warning(
            "list_pack_parity_statuses(%s) failed: %s",
            user_id[:8] if user_id else '?', e)
        return []


# ============================================================
# Algo-history cron cycle stats (M8.7 — 2026-05-07)
# ============================================================

def insert_algo_history_cron_cycle(
    user_id: str,
    started_at: str,
    ended_at: str,
    summary: dict,
) -> None:
    """Persist one cron cycle's outcome. Best-effort — failures swallowed."""
    try:
        client = get_admin_client()
        row = {
            'user_id': user_id,
            'started_at': started_at,
            'ended_at': ended_at,
            'processed': int(summary.get('processed') or 0),
            'inserted_total': int(summary.get('inserted_total') or 0),
            'skipped': int(summary.get('skipped') or 0),
            'errors': int(summary.get('errors') or 0),
            'elapsed_s': float(summary.get('elapsed_s') or 0.0),
            'budget_exhausted': bool(summary.get('budget_exhausted') or False),
            'per_strategy': summary.get('detail') or [],
        }
        client.table('algo_history_cron_cycles').insert(row).execute()
    except Exception as e:
        logger.warning(
            "insert_algo_history_cron_cycle(%s) failed: %s",
            user_id[:8] if user_id else '?', e)


def load_algo_history_cron_cycles(user_id: str, limit: int = 50) -> list:
    """Return the user's most recent N cron cycles, newest first."""
    try:
        client = get_admin_client()
        r = (client.table('algo_history_cron_cycles')
             .select('*')
             .eq('user_id', user_id)
             .order('started_at', desc=True)
             .limit(limit)
             .execute())
        return r.data or []
    except Exception as e:
        logger.warning(
            "load_algo_history_cron_cycles(%s) failed: %s",
            user_id[:8] if user_id else '?', e)
        return []


def prune_old_algo_history_cron_cycles(user_id: str, keep_n: int = 200) -> int:
    """Bound the table by deleting rows older than the keep_n-th newest."""
    try:
        client = get_admin_client()
        r = (client.table('algo_history_cron_cycles')
             .select('started_at')
             .eq('user_id', user_id)
             .order('started_at', desc=True)
             .range(keep_n, keep_n)
             .execute())
        if not r.data:
            return 0
        cutoff = r.data[0]['started_at']
        d = (client.table('algo_history_cron_cycles')
             .delete()
             .eq('user_id', user_id)
             .lt('started_at', cutoff)
             .execute())
        return len(d.data or [])
    except Exception as e:
        logger.warning(
            "prune_old_algo_history_cron_cycles(%s) failed: %s",
            user_id[:8] if user_id else '?', e)
        return 0
