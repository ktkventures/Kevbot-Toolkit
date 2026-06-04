"""Unified strategy creation factory.

Single source of truth for the strategy dict shape that every creation path
(Mass Builder, Strategy Builder UI, Pack Canary Recreate-all, AI/SOP) must
produce before calling `save_strategy_db`. Replaces 4 parallel local
builders that had drifted (canary path was missing ~15 fields that the
Mass Builder path wrote).

The key insight from 2026-06-04's investigation: engine code has default
fallbacks for the critical model fields, but those fallbacks live in
multiple places (`data_worker_engine.py:277`, `forward_test_service.py:283`,
`unified_engine.py:1572` post-`b2387ff`). Every time we update the engine's
default resolution, we have to remember to update all sites — and we keep
forgetting. This factory makes the defaults explicit AT CREATION TIME so
they're written to the DB row alongside everything else, eliminating that
class of drift.

Usage:
    from strategy_factory import build_strategy_config
    cfg = build_strategy_config(
        name="My Strategy",
        user_id="...",
        symbol="SPY",
        timeframe="10Sec",
        direction="LONG",
        entry_trigger_confluence_id="ut_bot_v4_default_bull_flip",
        exit_trigger_confluence_ids=["ut_bot_v4_default_bear_flip"],
        stop_config={"method": "atr", "atr_mult": 1.5, "exec_type": "L"},
    )
    # cfg is ready for db.save_strategy_db(cfg)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Field contract
# =============================================================================
# These two sets define the canonical strategy dict shape. Any caller that
# adds a key not in either set is fighting the factory — surface as a
# warning. Removing a key here is a breaking change; add a migration.

# Required for every strategy. Factory raises if missing.
_REQUIRED_FIELDS = (
    "name",
    "user_id",
    "symbol",
    "timeframe",
    "direction",
    "entry_trigger_confluence_id",
)

# Optional with explicit defaults. Factory fills if missing.
_OPTIONAL_FIELDS_WITH_DEFAULTS: dict[str, Any] = {
    # Triggers
    "exit_trigger_confluence_ids": [],
    "confluence": [],
    "general_confluences": [],
    # Risk / sizing
    "stop_config": None,
    "target_config": None,
    "time_exit_config": None,
    "bar_count_exit": None,
    "risk_per_trade": 100.0,
    "starting_balance": 10000.0,
    "trading_session": "RTH",
    # Data window
    "data_days": 30,
    "lookback_mode": "Days",
    "lookback_start_date": None,
    "lookback_end_date": None,
    "backtest_start_date": None,
    "backtest_end_date": None,
    "data_seed": 42,
    # OOS bands
    "in_sample_start": None,
    "in_sample_end": None,
    # Metadata
    "strategy_origin": "standard",
    "tags": [],
    "asset_type": "equity",
    # Engine model selection — None lets the factory fill via registry default
    "backtest_model": None,
    "algo_model": None,
    "live_model": None,
    # Worker controls
    "snapshot_subscribe_enabled": True,
    # Live alert tracking
    "alert_tracking_enabled": True,
}


class StrategyConfigError(ValueError):
    """Raised when build_strategy_config receives invalid inputs."""


# =============================================================================
# Public API
# =============================================================================

def build_strategy_config(
    *,
    # ── Required ────────────────────────────────────────────────────────
    name: str,
    user_id: str,
    symbol: str,
    timeframe: str,
    direction: str,
    entry_trigger_confluence_id: str,
    # ── Triggers ────────────────────────────────────────────────────────
    exit_trigger_confluence_ids: Optional[list[str]] = None,
    confluence: Optional[list[str]] = None,
    general_confluences: Optional[list[str]] = None,
    # ── Risk / sizing ───────────────────────────────────────────────────
    stop_config: Optional[dict] = None,
    target_config: Optional[dict] = None,
    time_exit_config: Optional[dict] = None,
    bar_count_exit: Optional[int] = None,
    risk_per_trade: float = 100.0,
    starting_balance: float = 10000.0,
    trading_session: str = "RTH",
    # ── Data window ─────────────────────────────────────────────────────
    data_days: int = 30,
    lookback_mode: str = "Days",
    lookback_start_date: Optional[str] = None,
    lookback_end_date: Optional[str] = None,
    backtest_start_date: Optional[str] = None,
    backtest_end_date: Optional[str] = None,
    data_seed: int = 42,
    # ── OOS bands ───────────────────────────────────────────────────────
    in_sample_start: Optional[str] = None,
    in_sample_end: Optional[str] = None,
    # ── Metadata ────────────────────────────────────────────────────────
    strategy_origin: str = "standard",
    tags: Optional[list[str]] = None,
    asset_type: str = "equity",
    # ── Engine model selection ──────────────────────────────────────────
    # If None, factory fills from strategy_models registry default. Pass
    # an explicit value to override (e.g. pin a canary to `cache_locked`).
    backtest_model: Optional[str] = None,
    algo_model: Optional[str] = None,
    live_model: Optional[str] = None,
    # ── Worker controls ─────────────────────────────────────────────────
    snapshot_subscribe_enabled: bool = True,
    alert_tracking_enabled: bool = True,
    # ── Forward-test ────────────────────────────────────────────────────
    # If None, factory stamps `now` so forward_testing starts at creation.
    # Pass explicit value to backfill old strategies or anchor to a
    # specific session.
    forward_test_start: Optional[str] = None,
    forward_testing: bool = True,
) -> dict:
    """Return canonical strategy dict ready for `save_strategy_db()`.

    Single source of truth for the strategy shape. Replaces parallel local
    builders in mass_builder, pack_canary_service, and the Strategy Builder
    UI endpoint. Routes all engine-default resolution through one path so
    every creation path produces an identical key set.

    Required kwargs are enforced; missing required raises StrategyConfigError.
    Optional kwargs get safe defaults (see _OPTIONAL_FIELDS_WITH_DEFAULTS).

    Engine model fields (backtest_model, algo_model, live_model) default to
    the registry-resolved default via `strategy_models.get_default_*()`.
    Explicit values override.

    Legacy fields (entry_trigger string, exit_trigger string, exit_triggers
    list) are derived from the modern *_confluence_id fields via
    `alerts._get_base_trigger_id()`. Display names (entry_trigger_name,
    exit_trigger_name, exit_trigger_names) are resolved via confluence_groups
    lookups; if resolution fails, fields are still populated with the
    confluence ID itself so downstream readers don't see None.

    NOTE: caller is responsible for ensuring the entry/exit confluence IDs
    actually exist (i.e., the confluence group has been created and the
    trigger pack registered). This factory does NOT validate that; it only
    validates SHAPE.
    """
    errors: list[str] = []

    # ── Validate required fields ────────────────────────────────────────
    if not name or not isinstance(name, str):
        errors.append("name must be a non-empty string")
    if not user_id or not isinstance(user_id, str):
        errors.append("user_id must be a non-empty string")
    if not symbol or not isinstance(symbol, str):
        errors.append("symbol must be a non-empty string")
    if not timeframe or not isinstance(timeframe, str):
        errors.append("timeframe must be a non-empty string")
    if direction not in ("LONG", "SHORT"):
        errors.append(f"direction must be 'LONG' or 'SHORT', got {direction!r}")
    if not entry_trigger_confluence_id:
        errors.append("entry_trigger_confluence_id is required")
    if errors:
        raise StrategyConfigError(
            "build_strategy_config validation failed: " + "; ".join(errors)
        )

    # ── Resolve engine model defaults ───────────────────────────────────
    if backtest_model is None or algo_model is None or live_model is None:
        try:
            from strategy_models import (
                get_default_backtest_model,
                get_default_algo_model,
                get_default_live_model,
            )
            if backtest_model is None:
                backtest_model = get_default_backtest_model()
            if algo_model is None:
                algo_model = get_default_algo_model()
            if live_model is None:
                live_model = get_default_live_model()
        except Exception as e:
            # Fall through with hard-coded defaults if registry can't load
            # (shouldn't happen in normal operation, but stay safe).
            logger.warning(
                "strategy_models registry unavailable, using hard-coded "
                "defaults: %s", e
            )
            backtest_model = backtest_model or "rest_hifi"
            algo_model = algo_model or "cache_locked"
            live_model = live_model or "ws_rest_spliced"

    # ── Resolve trigger legacy fields ───────────────────────────────────
    # The modern *_confluence_id fields are the source of truth. Legacy
    # `entry_trigger` (base trigger ID) + `entry_trigger_name` (display)
    # are derived for back-compat readers.
    exit_trigger_confluence_ids = exit_trigger_confluence_ids or []
    try:
        from alerts import _get_base_trigger_id
        entry_trigger = _get_base_trigger_id(entry_trigger_confluence_id)
        exit_triggers = [_get_base_trigger_id(c) for c in exit_trigger_confluence_ids]
    except Exception as e:
        logger.warning(
            "trigger base ID resolution failed, falling back to confluence "
            "IDs as-is: %s", e
        )
        entry_trigger = entry_trigger_confluence_id
        exit_triggers = list(exit_trigger_confluence_ids)

    # Display names: try confluence_groups lookup; fall back to the
    # confluence ID itself if name not resolvable. Downstream code reads
    # these for UI display, not engine logic, so a fallback is safe.
    entry_trigger_name = _resolve_trigger_name(
        entry_trigger_confluence_id, fallback=entry_trigger)
    exit_trigger_names = [
        _resolve_trigger_name(c, fallback=t)
        for c, t in zip(exit_trigger_confluence_ids, exit_triggers)
    ]

    # ── Forward-test anchor ─────────────────────────────────────────────
    if forward_test_start is None and forward_testing:
        forward_test_start = datetime.now(timezone.utc).isoformat()

    # ── Assemble canonical dict ─────────────────────────────────────────
    cfg: dict[str, Any] = {
        # Top-level (saved as columns by _strategy_to_row)
        "name": name,
        "user_id": user_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "direction": direction,
        "strategy_origin": strategy_origin,
        "forward_testing": forward_testing,
        "forward_test_start": forward_test_start,
        "alert_tracking_enabled": alert_tracking_enabled,
        # ── config JSONB fields (everything below lands in config) ──
        # Triggers — modern (canonical) + legacy (back-compat)
        "entry_trigger_confluence_id": entry_trigger_confluence_id,
        "entry_trigger": entry_trigger,
        "entry_trigger_name": entry_trigger_name,
        "exit_trigger_confluence_ids": list(exit_trigger_confluence_ids),
        "exit_triggers": exit_triggers,
        "exit_trigger_names": exit_trigger_names,
        # Legacy single-exit fields (first element of list, for back-compat)
        "exit_trigger_confluence_id": (
            exit_trigger_confluence_ids[0] if exit_trigger_confluence_ids else None
        ),
        "exit_trigger": exit_triggers[0] if exit_triggers else None,
        "exit_trigger_name": (
            exit_trigger_names[0] if exit_trigger_names else None
        ),
        # Confluence gates
        "confluence": list(confluence) if confluence else [],
        "general_confluences": (
            list(general_confluences) if general_confluences else []
        ),
        # Risk / sizing
        "stop_config": stop_config,
        "target_config": target_config,
        "time_exit_config": time_exit_config,
        "bar_count_exit": bar_count_exit,
        "risk_per_trade": float(risk_per_trade),
        "starting_balance": float(starting_balance),
        "trading_session": trading_session,
        # Data window
        "data_days": int(data_days),
        "lookback_mode": lookback_mode,
        "lookback_start_date": lookback_start_date,
        "lookback_end_date": lookback_end_date,
        "backtest_start_date": backtest_start_date,
        "backtest_end_date": backtest_end_date,
        "data_seed": int(data_seed),
        # OOS
        "in_sample_start": in_sample_start,
        "in_sample_end": in_sample_end,
        # Metadata
        "tags": list(tags) if tags else [],
        "asset_type": asset_type,
        # Engine models — explicit defaults so downstream paths don't
        # have to do their own resolution.
        "backtest_model": backtest_model,
        "algo_model": algo_model,
        "live_model": live_model,
        # Worker controls
        "snapshot_subscribe_enabled": snapshot_subscribe_enabled,
    }
    return cfg


def validate_strategy_config(cfg: dict) -> list[str]:
    """Return list of issues found in `cfg`. Empty list = valid.

    Use to audit existing DB rows (backfill script) or verify a caller's
    pre-built dict before save. The same checks are run inside
    `build_strategy_config()` but raise instead of return.
    """
    errors: list[str] = []
    for f in _REQUIRED_FIELDS:
        if cfg.get(f) in (None, "", []):
            errors.append(f"missing required field: {f}")
    # direction is constrained
    if cfg.get("direction") and cfg["direction"] not in ("LONG", "SHORT"):
        errors.append(f"direction invalid: {cfg['direction']!r}")
    # exit fields consistency check: lists should be parallel
    elen = len(cfg.get("exit_trigger_confluence_ids") or [])
    tlen = len(cfg.get("exit_triggers") or [])
    nlen = len(cfg.get("exit_trigger_names") or [])
    if elen != tlen or elen != nlen:
        errors.append(
            f"exit fields parallel mismatch: "
            f"confluence_ids={elen}, triggers={tlen}, names={nlen}"
        )
    return errors


# =============================================================================
# Private helpers
# =============================================================================

def _resolve_trigger_name(confluence_id: str, *, fallback: str = "") -> str:
    """Resolve trigger display name via confluence_groups TEMPLATES.

    Look up the group by name, find the trigger's display label. Fall
    back to the provided fallback (typically the base trigger ID) if
    anything goes wrong — downstream UI accepts either.
    """
    if not confluence_id:
        return fallback
    try:
        from confluence_groups import get_all_triggers
        all_triggers = get_all_triggers()
        if confluence_id in all_triggers:
            td = all_triggers[confluence_id]
            return getattr(td, "name", None) or fallback
    except Exception:
        pass
    return fallback or confluence_id
