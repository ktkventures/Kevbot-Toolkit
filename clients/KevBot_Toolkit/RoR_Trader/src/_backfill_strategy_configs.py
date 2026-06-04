"""Backfill missing canonical config fields on existing strategies.

After the strategy_factory refactor (2026-06-04), the canonical config
dict has 42 keys. Existing strategies created via the pre-refactor canary
path are missing ~15 of these fields. This script fills them in
idempotently: only writes fields that are currently missing, never
overwrites a user-set value.

Usage:
    cd src/
    USE_DB=true ../.venv/bin/python _backfill_strategy_configs.py           # dry-run (default)
    USE_DB=true ../.venv/bin/python _backfill_strategy_configs.py --apply   # commit changes
    USE_DB=true ../.venv/bin/python _backfill_strategy_configs.py --sid 270 # single strategy
    USE_DB=true ../.venv/bin/python _backfill_strategy_configs.py --tag pack-canary  # filter by config.tags

Outputs per-strategy report of which fields would be (or were) filled,
plus a summary tally at the end.

Safety:
- Dry-run by default (must explicitly pass --apply)
- Never overwrites a user-set value (only fills missing/None)
- Reads raw config via admin client (bypasses the flattening bug
  documented in feedback_jsonb_partial_updates.md)
- Writes full merged config dict (not partial PATCH) to dodge the
  partial-JSONB-wipe bug class
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [backfill] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Fields the factory always fills with defaults. Backfill targets these
# when missing on existing rows.
_BACKFILL_FIELDS = (
    # Engine model defaults
    "backtest_model",
    "algo_model",
    "live_model",
    # Legacy trigger fields derived from confluence IDs
    "entry_trigger",
    "entry_trigger_name",
    "exit_triggers",
    "exit_trigger_names",
    "exit_trigger",
    "exit_trigger_confluence_id",
    "exit_trigger_name",
    # Defaults
    "general_confluences",
    "asset_type",
    "alert_tracking_enabled",
    "snapshot_subscribe_enabled",
    # Optional but factory-managed
    "tags",
)


def _factory_kwargs_from_row(row: dict) -> Optional[dict]:
    """Extract kwargs needed for build_strategy_config from a DB row.

    Returns None if the row is missing required factory inputs (e.g.
    no entry_trigger_confluence_id) — those rows can't be safely
    backfilled and are reported but skipped.
    """
    cfg = row.get("config") or {}
    entry_cid = cfg.get("entry_trigger_confluence_id")
    if not entry_cid:
        return None

    # Re-derive required from row + config.
    # Row-level fields (columns) take precedence over config (some are mirrored).
    name = row.get("name")
    user_id = row.get("user_id")
    symbol = row.get("symbol")
    timeframe = row.get("timeframe")
    direction = row.get("direction")

    if not all([name, user_id, symbol, timeframe, direction]):
        return None

    return dict(
        name=name,
        user_id=user_id,
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        entry_trigger_confluence_id=entry_cid,
        # Pass through existing config — factory will fill defaults for None/missing.
        exit_trigger_confluence_ids=cfg.get("exit_trigger_confluence_ids") or [],
        confluence=cfg.get("confluence") or [],
        general_confluences=cfg.get("general_confluences") or [],
        stop_config=cfg.get("stop_config"),
        target_config=cfg.get("target_config"),
        time_exit_config=cfg.get("time_exit_config"),
        bar_count_exit=cfg.get("bar_count_exit"),
        risk_per_trade=cfg.get("risk_per_trade", 100.0),
        starting_balance=cfg.get("starting_balance", 10000.0),
        trading_session=cfg.get("trading_session", "RTH"),
        data_days=cfg.get("data_days", 30),
        lookback_mode=cfg.get("lookback_mode", "Days"),
        lookback_start_date=cfg.get("lookback_start_date"),
        lookback_end_date=cfg.get("lookback_end_date"),
        backtest_start_date=cfg.get("backtest_start_date"),
        backtest_end_date=cfg.get("backtest_end_date"),
        data_seed=cfg.get("data_seed", 42),
        in_sample_start=cfg.get("in_sample_start"),
        in_sample_end=cfg.get("in_sample_end"),
        strategy_origin=cfg.get("strategy_origin") or row.get("strategy_origin") or "standard",
        tags=cfg.get("tags") or [],
        asset_type=cfg.get("asset_type", "equity"),
        # Pass-through existing model fields (factory only fills None).
        backtest_model=cfg.get("backtest_model"),
        algo_model=cfg.get("algo_model"),
        live_model=cfg.get("live_model"),
        snapshot_subscribe_enabled=cfg.get("snapshot_subscribe_enabled", True),
        alert_tracking_enabled=(
            row.get("alert_tracking_enabled")
            if row.get("alert_tracking_enabled") is not None
            else True
        ),
        # Preserve existing forward_test_start; factory only stamps if None.
        forward_test_start=row.get("forward_test_start"),
        forward_testing=row.get("forward_testing", True),
    )


def _diff_fill(current_cfg: dict, canonical_cfg: dict) -> dict:
    """Return dict of {field: new_value} for fields missing/None in current
    that the canonical has a value for. Never overwrites a set value.
    """
    fill: dict = {}
    for k, v in canonical_cfg.items():
        cur = current_cfg.get(k)
        # Treat None/empty-list/empty-string as missing.
        # Don't overwrite a set value (truthy or explicitly False/0).
        if cur is None and v is not None:
            fill[k] = v
        elif cur == [] and v not in ([], None):
            fill[k] = v
        elif cur == "" and v not in ("", None):
            fill[k] = v
    return fill


def backfill_one(row: dict, apply: bool) -> dict:
    """Backfill one row. Returns {sid, name, status, filled_fields, error}.

    status: 'ok' | 'skipped_unfixable' | 'no_op' | 'error'
    """
    from strategy_factory import build_strategy_config
    from db import set_admin_user_context, get_current_user_id, clear_current_user

    sid = row.get("id")
    name = row.get("name", "?")
    cfg = row.get("config") or {}

    kwargs = _factory_kwargs_from_row(row)
    if kwargs is None:
        return {
            "sid": sid,
            "name": name,
            "status": "skipped_unfixable",
            "filled_fields": [],
            "error": "missing required factory inputs (likely no entry_trigger_confluence_id)",
        }

    # Set per-row admin user context so the factory's trigger-name +
    # base-trigger-ID resolution (which queries the user's confluence_groups
    # JSONB) succeeds. Without this, the factory falls back to using the
    # confluence ID as both the name AND the base trigger — workable but
    # less accurate.
    _prev_user = get_current_user_id()
    _need_ctx_swap = _prev_user != row["user_id"]
    if _need_ctx_swap:
        set_admin_user_context(row["user_id"])

    try:
        canonical = build_strategy_config(**kwargs)
    except Exception as e:
        return {
            "sid": sid,
            "name": name,
            "status": "error",
            "filled_fields": [],
            "error": str(e),
        }
    finally:
        if _need_ctx_swap:
            if _prev_user is not None:
                set_admin_user_context(_prev_user)
            else:
                clear_current_user()

    # The factory output contains both top-level fields (name, user_id,
    # symbol, ...) and config-bound fields. We want to fill ONLY config
    # fields; top-level columns are managed separately.
    # Top-level fields (columns) — see db._strategy_to_row STRATEGY_COLUMN_FIELDS.
    _TOP_LEVEL = {
        "name", "user_id", "symbol", "direction", "timeframe",
        "strategy_origin", "forward_testing", "forward_test_start",
        "alert_tracking_enabled",
    }
    canonical_cfg_only = {
        k: v for k, v in canonical.items() if k not in _TOP_LEVEL
    }
    fill = _diff_fill(cfg, canonical_cfg_only)

    if not fill:
        return {
            "sid": sid,
            "name": name,
            "status": "no_op",
            "filled_fields": [],
            "error": None,
        }

    if not apply:
        return {
            "sid": sid,
            "name": name,
            "status": "ok",
            "filled_fields": sorted(fill.keys()),
            "error": None,
        }

    # Apply: write the FULL merged config back (not partial PATCH) to
    # dodge the JSONB partial-wipe bug per feedback_jsonb_partial_updates.md
    new_cfg = dict(cfg)  # don't mutate the input
    new_cfg.update(fill)
    try:
        from db import get_admin_client
        client = get_admin_client()
        client.table("strategies").update({"config": new_cfg}) \
            .eq("id", sid).eq("user_id", row["user_id"]).execute()
    except Exception as e:
        return {
            "sid": sid,
            "name": name,
            "status": "error",
            "filled_fields": sorted(fill.keys()),
            "error": f"write failed: {e}",
        }
    return {
        "sid": sid,
        "name": name,
        "status": "ok",
        "filled_fields": sorted(fill.keys()),
        "error": None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true",
                   help="Commit changes (default: dry-run)")
    p.add_argument("--sid", type=int, default=None,
                   help="Backfill a single strategy by ID (otherwise all)")
    p.add_argument("--tag", default=None,
                   help="Only backfill strategies with this tag in config.tags")
    args = p.parse_args()

    # Load user packs so trigger-name resolution finds their TEMPLATES entries
    # (factory uses confluence_groups.get_all_triggers internally).
    try:
        import pack_registry
        pack_registry.scan_and_load_all()
    except Exception as e:
        logger.warning("pack_registry init failed (trigger names will fall back): %s", e)

    from db import get_admin_client
    client = get_admin_client()

    # Pull strategies. Don't filter at DB layer for tag (it's in JSONB) — load all and filter in Python.
    q = client.table("strategies").select(
        "id,name,user_id,symbol,timeframe,direction,strategy_origin,"
        "forward_testing,forward_test_start,alert_tracking_enabled,config"
    )
    if args.sid:
        q = q.eq("id", args.sid)
    rows = q.execute().data or []

    if args.tag:
        rows = [r for r in rows if args.tag in ((r.get("config") or {}).get("tags") or [])]

    if not rows:
        logger.info("No strategies matched filters. Exit.")
        return

    mode = "APPLY" if args.apply else "DRY-RUN"
    logger.info("─" * 80)
    logger.info("Backfill mode: %s — %d strategies to process", mode, len(rows))
    logger.info("─" * 80)

    results = []
    for row in rows:
        res = backfill_one(row, apply=args.apply)
        results.append(res)
        if res["status"] == "ok" and res["filled_fields"]:
            logger.info("sid=%s [%s] %d fields: %s",
                        res["sid"], res["status"],
                        len(res["filled_fields"]),
                        ", ".join(res["filled_fields"]))
        elif res["status"] == "no_op":
            logger.info("sid=%s [no_op] already canonical", res["sid"])
        elif res["status"] in ("skipped_unfixable", "error"):
            logger.warning("sid=%s [%s] %s", res["sid"], res["status"], res["error"])

    # Tally
    by_status = {}
    total_fills = 0
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        total_fills += len(r["filled_fields"])

    logger.info("─" * 80)
    logger.info("Summary (%s):", mode)
    for status, n in sorted(by_status.items()):
        logger.info("  %-22s %d strategies", status, n)
    logger.info("  total fields filled:   %d (sum across strategies)", total_fills)
    if not args.apply and any(r["status"] == "ok" for r in results):
        logger.info("")
        logger.info("Re-run with --apply to commit changes.")


if __name__ == "__main__":
    sys.exit(main())
