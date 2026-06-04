"""Unit tests for `strategy_factory.build_strategy_config`.

Verifies the canonical strategy dict shape is identical regardless of
which creation path's inputs flow through it, and that defaults +
validation behave as advertised.

Run from `src/` dir:
    USE_DB=true ../.venv/bin/python _test_strategy_factory.py
"""

import sys

from strategy_factory import (
    build_strategy_config,
    validate_strategy_config,
    StrategyConfigError,
)


def _canary_inputs() -> dict:
    """Inputs that match what pack_canary_service builds today."""
    return dict(
        name="PACKTEST · UT Bot V4 · trigger",
        user_id="test-user",
        symbol="SPY",
        timeframe="10Sec",
        direction="LONG",
        entry_trigger_confluence_id="ut_bot_v4_default_bull_flip",
        exit_trigger_confluence_ids=["ut_bot_v4_default_bear_flip"],
        trading_session="Extended Hours",
        stop_config={"method": "atr", "atr_mult": 1.5, "exec_type": "L"},
        target_config=None,
        risk_per_trade=100.0,
        starting_balance=10000.0,
        data_days=2,
        data_seed=42,
        lookback_mode="Days",
        tags=["pack-canary"],
    )


def _mass_builder_inputs() -> dict:
    """Inputs that match what mass_builder builds today."""
    return dict(
        name="SPY LONG - Mass Test",
        user_id="test-user",
        symbol="SPY",
        timeframe="1Min",
        direction="LONG",
        entry_trigger_confluence_id="macd_line_v2_default_cross_bull",
        exit_trigger_confluence_ids=[],
        confluence=["15m-MACD_HISTOGRAM_V2-H+dn"],
        stop_config={"method": "swing", "padding": 0.03, "lookback": 5},
        trading_session="RTH",
        data_days=90,
        lookback_mode="Days",
        in_sample_start="2026-04-01T00:00:00Z",
    )


def _ui_inputs() -> dict:
    """Inputs minimal — what Strategy Builder UI sends."""
    return dict(
        name="My Test Strategy",
        user_id="test-user",
        symbol="TSLA",
        timeframe="5Min",
        direction="LONG",
        entry_trigger_confluence_id="ema_stack_default_cross_bull",
    )


def test_all_paths_produce_identical_shape():
    """Three different creation paths' inputs → identical key set."""
    canary = build_strategy_config(**_canary_inputs())
    mass = build_strategy_config(**_mass_builder_inputs())
    ui = build_strategy_config(**_ui_inputs())
    canary_keys = set(canary.keys())
    mass_keys = set(mass.keys())
    ui_keys = set(ui.keys())
    assert canary_keys == mass_keys, (
        f"canary vs mass key drift: "
        f"canary_only={canary_keys - mass_keys}, "
        f"mass_only={mass_keys - canary_keys}"
    )
    assert canary_keys == ui_keys, (
        f"canary vs ui key drift: "
        f"canary_only={canary_keys - ui_keys}, "
        f"ui_only={ui_keys - canary_keys}"
    )
    print(f"✓ all 3 paths produce identical {len(canary_keys)}-key shape")


def test_engine_model_defaults_applied():
    """When models aren't passed, registry defaults are filled."""
    cfg = build_strategy_config(**_canary_inputs())
    assert cfg["backtest_model"] == "rest_hifi", cfg["backtest_model"]
    assert cfg["algo_model"] == "cache_locked", cfg["algo_model"]
    assert cfg["live_model"] == "ws_rest_spliced", cfg["live_model"]
    print("✓ engine model defaults applied (rest_hifi / cache_locked / ws_rest_spliced)")


def test_explicit_models_override_defaults():
    """Explicit values are passed through unchanged."""
    cfg = build_strategy_config(
        **_canary_inputs(),
        backtest_model="cache_locked",
        live_model="ws_agg_locked",
    )
    assert cfg["backtest_model"] == "cache_locked"
    assert cfg["live_model"] == "ws_agg_locked"
    assert cfg["algo_model"] == "cache_locked"  # default still applied
    print("✓ explicit model values override defaults")


def test_legacy_trigger_fields_derived():
    """entry_trigger / exit_triggers / exit_trigger derived from CIDs."""
    cfg = build_strategy_config(**_canary_inputs())
    # entry_trigger is the base ID (alerts._get_base_trigger_id resolves
    # ut_bot_v4_default_bull_flip → utv4_bull_flip if pack registered,
    # else falls back to the confluence ID itself).
    assert cfg["entry_trigger"], "entry_trigger should be non-empty"
    assert cfg["exit_triggers"] == [cfg["exit_trigger"]]
    assert cfg["exit_trigger_confluence_id"] == cfg["exit_trigger_confluence_ids"][0]
    assert cfg["exit_trigger_names"][0] == cfg["exit_trigger_name"]
    print(f"✓ legacy trigger fields derived: entry={cfg['entry_trigger']}, "
          f"exit={cfg['exit_trigger']}")


def test_required_fields_validation():
    """Missing required fields raise StrategyConfigError."""
    try:
        build_strategy_config(
            name="", user_id="u", symbol="SPY", timeframe="1Min",
            direction="LONG", entry_trigger_confluence_id="x")
    except StrategyConfigError as e:
        assert "name" in str(e)
    else:
        raise AssertionError("missing name should raise")

    try:
        build_strategy_config(
            name="x", user_id="u", symbol="SPY", timeframe="1Min",
            direction="SIDEWAYS", entry_trigger_confluence_id="x")
    except StrategyConfigError as e:
        assert "direction" in str(e)
    else:
        raise AssertionError("invalid direction should raise")

    print("✓ required-field validation raises correctly")


def test_optional_defaults_filled():
    """Optional fields with sensible defaults are always present."""
    cfg = build_strategy_config(**_ui_inputs())
    # Spot-check that defaults landed
    assert cfg["risk_per_trade"] == 100.0
    assert cfg["starting_balance"] == 10000.0
    assert cfg["trading_session"] == "RTH"
    assert cfg["data_days"] == 30
    assert cfg["alert_tracking_enabled"] is True
    assert cfg["snapshot_subscribe_enabled"] is True
    assert cfg["asset_type"] == "equity"
    assert cfg["strategy_origin"] == "standard"
    assert cfg["tags"] == []
    assert cfg["confluence"] == []
    assert cfg["general_confluences"] == []
    assert cfg["exit_trigger_confluence_ids"] == []
    print("✓ optional defaults filled (risk, session, days, flags, lists)")


def test_validate_existing_config():
    """validate_strategy_config catches missing required + parallel-list issues."""
    bad = {
        "name": "x",
        # missing user_id
        "symbol": "SPY",
        "timeframe": "1Min",
        "direction": "INVALID",
        "entry_trigger_confluence_id": "x",
        "exit_trigger_confluence_ids": ["a", "b"],
        "exit_triggers": ["a"],  # length mismatch
        "exit_trigger_names": ["a"],
    }
    errors = validate_strategy_config(bad)
    assert any("user_id" in e for e in errors), errors
    assert any("direction" in e for e in errors), errors
    assert any("parallel mismatch" in e for e in errors), errors
    print(f"✓ validate_strategy_config caught {len(errors)} errors as expected")


def test_forward_test_start_stamped():
    """forward_test_start defaults to now() when not provided."""
    cfg = build_strategy_config(**_ui_inputs())
    assert cfg["forward_test_start"] is not None
    assert "T" in cfg["forward_test_start"]  # ISO-8601
    assert cfg["forward_testing"] is True
    print(f"✓ forward_test_start stamped: {cfg['forward_test_start']}")


def main():
    tests = [
        test_all_paths_produce_identical_shape,
        test_engine_model_defaults_applied,
        test_explicit_models_override_defaults,
        test_legacy_trigger_fields_derived,
        test_required_fields_validation,
        test_optional_defaults_filled,
        test_validate_existing_config,
        test_forward_test_start_stamped,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except Exception as e:
            failures.append((t.__name__, e))
            print(f"✗ {t.__name__}: {e}")
    if failures:
        print(f"\n{len(failures)} of {len(tests)} tests failed")
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed ✓")


if __name__ == "__main__":
    main()
