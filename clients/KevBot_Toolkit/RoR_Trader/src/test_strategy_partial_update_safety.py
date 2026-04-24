"""Regression test for update_strategy_admin partial-update safety.

The original bug: update_strategy_admin used _strategy_to_row() to build the
PostgREST update payload. That helper assumes a FULL strategy dict and always
emits `config = {}` when no config-bucket fields are present. PostgREST then
overwrites the `config` JSONB column with `{}`, wiping all of:
entry_trigger_confluence_id, exit_trigger_confluence_ids, bar_count_exit,
stop_config, target_config, confluence, data_days, asset_type, etc.

That bug was triggered by the worker's algo-trade persist path
(DBAlertDispatcher._persist_algo_trade) calling update_strategy_admin with
just `{stored_trades: [...]}`. Strategies 70 and 71 had their configs wiped
within minutes of deployment.

This test simulates the same call pattern and asserts the config column is
preserved. Uses a mocked Supabase client so it can run in CI without DB
access.

Run: cd src && ../.venv/bin/python test_strategy_partial_update_safety.py
"""

from __future__ import annotations

import unittest.mock as mock


def _setup_db_module():
    """Import db module with a stubbed get_admin_client / get_client."""
    # Load env so module-level constants don't blow up on missing keys
    import os
    os.environ.setdefault('SUPABASE_URL', 'http://stub.local')
    os.environ.setdefault('SUPABASE_SERVICE_ROLE_KEY', 'stub-service-role-key')
    os.environ.setdefault('SUPABASE_ANON_KEY', 'stub-anon-key')
    import db as db_mod
    return db_mod


class FakeQuery:
    """Captures the dict passed to .update() so we can assert on it."""
    def __init__(self):
        self.last_update_payload = None
        self.eq_calls = []

    def select(self, *_a, **_k):
        return self

    def update(self, payload):
        self.last_update_payload = payload
        return self

    def eq(self, k, v):
        self.eq_calls.append((k, v))
        return self

    def execute(self):
        # Return a "row" with just the id so update_strategy_admin can finish
        return mock.MagicMock(data=[{'id': 999, 'config': {}}])


class FakeClient:
    def __init__(self):
        self.fake_query = FakeQuery()

    def table(self, _name):
        return self.fake_query


def test_partial_update_with_only_column_field_omits_config():
    """When updates only contain column-bucket fields (e.g. stored_trades),
    the PostgREST UPDATE payload must NOT include `config` at all. Including
    `config: {}` would wipe the JSONB blob."""
    db = _setup_db_module()
    fake_client = FakeClient()
    with mock.patch.object(db, 'get_admin_client', return_value=fake_client):
        # Simulate worker.py:_persist_algo_trade exact call shape
        db.update_strategy_admin(
            strategy_id=999,
            user_id='user-uuid',
            updates={'stored_trades': [{'entry_time': 't1', 'exit_time': 't2'}]},
        )

    payload = fake_client.fake_query.last_update_payload
    assert payload is not None, "update was never called"
    assert 'stored_trades' in payload, f"payload missing stored_trades: {payload}"
    assert 'config' not in payload, (
        f"BUG: payload includes config (would wipe JSONB!) — {payload}"
    )


def test_partial_update_with_only_config_field_emits_only_config():
    """When updates contain only config-bucket fields (e.g. bar_count_exit),
    the payload should include `config` with just those fields. Caller
    is responsible for merging with existing config (PostgREST has no
    partial JSONB merge without RPC)."""
    db = _setup_db_module()
    fake_client = FakeClient()
    with mock.patch.object(db, 'get_admin_client', return_value=fake_client):
        db.update_strategy_admin(
            strategy_id=999,
            user_id='user-uuid',
            updates={'bar_count_exit': 5},
        )

    payload = fake_client.fake_query.last_update_payload
    assert 'config' in payload, "payload missing config bucket"
    assert payload['config'] == {'bar_count_exit': 5}, (
        f"unexpected config payload: {payload['config']}"
    )
    # Should not include any column-bucket field unless caller supplied one
    column_keys = set(payload.keys()) - {'config'}
    assert column_keys == set(), (
        f"unexpected column fields in payload: {column_keys}"
    )


def test_partial_update_with_mixed_fields_splits_correctly():
    """Mixed column + config updates should produce a payload with both
    the column field at top level and the config field nested in `config`."""
    db = _setup_db_module()
    fake_client = FakeClient()
    with mock.patch.object(db, 'get_admin_client', return_value=fake_client):
        db.update_strategy_admin(
            strategy_id=999,
            user_id='user-uuid',
            updates={
                'stored_trades': [{'r': 1.0}],  # column bucket
                'bar_count_exit': 7,            # config bucket
            },
        )

    payload = fake_client.fake_query.last_update_payload
    assert payload.get('stored_trades') == [{'r': 1.0}]
    assert payload.get('config') == {'bar_count_exit': 7}


def test_empty_updates_returns_none_without_calling_db():
    """An empty updates dict should be a no-op (no UPDATE query)."""
    db = _setup_db_module()
    fake_client = FakeClient()
    with mock.patch.object(db, 'get_admin_client', return_value=fake_client):
        result = db.update_strategy_admin(
            strategy_id=999,
            user_id='user-uuid',
            updates={},
        )

    assert result is None
    assert fake_client.fake_query.last_update_payload is None, (
        "update was called with empty payload — should have skipped entirely"
    )


def test_user_id_and_id_stripped_from_column_payload():
    """Even if the caller passes user_id or id, they should not end up in the
    UPDATE payload (immutable + already in the eq() filters)."""
    db = _setup_db_module()
    fake_client = FakeClient()
    with mock.patch.object(db, 'get_admin_client', return_value=fake_client):
        db.update_strategy_admin(
            strategy_id=999,
            user_id='user-uuid',
            updates={
                'id': 12345,                # should be stripped
                'user_id': 'other-user',    # should be stripped
                'stored_trades': [{'r': 1}],
            },
        )

    payload = fake_client.fake_query.last_update_payload
    assert 'id' not in payload, f"id leaked into payload: {payload}"
    assert 'user_id' not in payload, f"user_id leaked into payload: {payload}"
    assert payload.get('stored_trades') == [{'r': 1}]


# =============================================================================
# Phase 40 — trades-table feature flag tests
# =============================================================================

def _with_flag_on(fn):
    """Run fn with USE_TRADES_TABLE=true set in env."""
    import os
    prev = os.environ.get('USE_TRADES_TABLE')
    os.environ['USE_TRADES_TABLE'] = 'true'
    try:
        return fn()
    finally:
        if prev is None:
            os.environ.pop('USE_TRADES_TABLE', None)
        else:
            os.environ['USE_TRADES_TABLE'] = prev


def test_flag_on_strips_stored_trades_from_strategies_update():
    """With flag ON, update_strategy_admin must NOT write stored_trades to the
    strategies table. Trades should be redirected to the trades table via
    trades_store.replace_trades_for_strategy.
    """
    def run():
        db = _setup_db_module()
        fake_client = FakeClient()
        captured_trades_calls = []

        def fake_replace(strategy_id, user_id, trades):
            captured_trades_calls.append(
                {'strategy_id': strategy_id, 'user_id': user_id, 'trades': trades}
            )
            return True

        import trades_store as ts
        with mock.patch.object(db, 'get_admin_client', return_value=fake_client), \
             mock.patch.object(ts, 'replace_trades_for_strategy', side_effect=fake_replace):
            db.update_strategy_admin(
                strategy_id=999,
                user_id='user-uuid',
                updates={'stored_trades': [{'r': 1.0, 'entry_fill_ts': 't1'}]},
            )

        payload = fake_client.fake_query.last_update_payload
        # The strategies UPDATE must NOT include stored_trades when flag ON
        # (either the UPDATE was skipped entirely, or stored_trades got stripped).
        if payload is not None:
            assert 'stored_trades' not in payload, (
                f"flag-on payload leaks stored_trades: {payload}"
            )
        # And the trades-table redirect fired with the extracted list
        assert len(captured_trades_calls) == 1, (
            f"expected 1 replace_trades_for_strategy call, got {len(captured_trades_calls)}"
        )
        assert captured_trades_calls[0]['strategy_id'] == 999
        assert captured_trades_calls[0]['trades'] == [{'r': 1.0, 'entry_fill_ts': 't1'}]

    _with_flag_on(run)


def test_flag_on_mixed_fields_still_writes_config_to_strategies():
    """With flag ON, a mixed payload (stored_trades + a config field) should
    redirect stored_trades to the trades table AND still update the config
    JSONB on strategies (preserving the partial-update safety)."""
    def run():
        db = _setup_db_module()
        fake_client = FakeClient()

        import trades_store as ts
        with mock.patch.object(db, 'get_admin_client', return_value=fake_client), \
             mock.patch.object(ts, 'replace_trades_for_strategy', return_value=True):
            db.update_strategy_admin(
                strategy_id=999,
                user_id='user-uuid',
                updates={
                    'stored_trades': [{'r': 2.0}],
                    'bar_count_exit': 7,
                },
            )

        payload = fake_client.fake_query.last_update_payload
        assert payload is not None, "strategies update should still fire for config fields"
        assert 'stored_trades' not in payload, f"stored_trades leaked: {payload}"
        assert payload.get('config') == {'bar_count_exit': 7}

    _with_flag_on(run)


def test_flag_on_omitted_stored_trades_skips_trades_call():
    """With flag ON, if the updates dict doesn't mention stored_trades at all,
    the trades table must not be touched (preserves partial-update safety)."""
    def run():
        db = _setup_db_module()
        fake_client = FakeClient()
        calls = []

        def fake_replace(*a, **k):
            calls.append((a, k))
            return True

        import trades_store as ts
        with mock.patch.object(db, 'get_admin_client', return_value=fake_client), \
             mock.patch.object(ts, 'replace_trades_for_strategy', side_effect=fake_replace):
            db.update_strategy_admin(
                strategy_id=999,
                user_id='user-uuid',
                updates={'bar_count_exit': 9},  # config-only, no stored_trades
            )

        assert calls == [], (
            f"trades table was touched despite stored_trades not in updates: {calls}"
        )

    _with_flag_on(run)


# =============================================================================
# Self-execution path
# =============================================================================

if __name__ == '__main__':
    tests = [
        test_partial_update_with_only_column_field_omits_config,
        test_partial_update_with_only_config_field_emits_only_config,
        test_partial_update_with_mixed_fields_splits_correctly,
        test_empty_updates_returns_none_without_calling_db,
        test_user_id_and_id_stripped_from_column_payload,
        # Phase 40 — trades-table feature flag
        test_flag_on_strips_stored_trades_from_strategies_update,
        test_flag_on_mixed_fields_still_writes_config_to_strategies,
        test_flag_on_omitted_stored_trades_skips_trades_call,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}\n  {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}\n  {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(failed)
