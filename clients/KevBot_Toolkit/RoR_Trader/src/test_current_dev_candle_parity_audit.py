"""Current-dev candle parity audit reproductions (2026-07-15).

These tests intentionally fail on dev@3cdb29fc79201efda67d2901dff8dda70d6e9532.
They contain no production fixes. Each test locks one externally observable
contract so a future correction can be evaluated against the exact audited
snapshot.

Run:
    cd clients/KevBot_Toolkit/RoR_Trader/src
    python -m pytest -q test_current_dev_candle_parity_audit.py
"""

from __future__ import annotations

import ast
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, ".")

import engine_health  # noqa: E402
import live_bars_writer  # noqa: E402
from ralph_engine import BarBuilder, SymbolHub  # noqa: E402
from strategy_models import get_default_live_model  # noqa: E402


AUDITED_SHA = "3cdb29fc79201efda67d2901dff8dda70d6e9532"
RTH_START = datetime(2026, 7, 15, 14, 30, tzinfo=timezone.utc)


def _second_bar(start: datetime, second: int) -> dict:
    price = 100.0 + second / 100.0
    return {
        "timestamp": (start + timedelta(seconds=second)).isoformat(),
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "volume": 1.0,
    }


class _ColdIndicators:
    _initialized = False


class _RecordingMonitor:
    """Small monitor double accepted by the real SymbolHub close paths."""

    def __init__(self, tf_seconds: int, live_model: str):
        self.strat_id = 9001
        self.tf_seconds = tf_seconds
        self.live_model = live_model
        self.session = "RTH"
        self.user_id = "audit-user"
        self.strategy = {
            "id": self.strat_id,
            "symbol": "SPY",
            "timeframe": "1Min",
        }
        self.indicators = _ColdIndicators()
        self._intrabar_triggers = set()
        self._current_confluence = set()
        self._fired_bucket = None
        self.close_calls: list[dict] = []

    def on_tick(self, *_args, **_kwargs):
        return []

    def compute_tentative_state(self, _bar):
        return {}, {}

    def on_bar_close(self, bar, _bar_count, **_kwargs):
        self.close_calls.append(dict(bar))
        return [], {
            "position_state": "FLAT",
            "trigger_booleans": {},
            "indicator_values": {},
            "interpreter_states": {},
        }


def test_gte_60_chart_partial_must_not_drive_a_strategy_close():
    """A chart-only partial must never commit trading state.

    Current dev documents the >=60s per-second partial as incomplete and
    chart-only, but flush_stale_bars sends it through monitor.on_bar_close.
    The late :55-:59 tail cannot retract a position transition or alert.
    """
    hub = SymbolHub("SPY")
    monitor = _RecordingMonitor(60, "ws_rest_spliced")
    builder = BarBuilder(60)
    hub.monitors = {monitor.strat_id: monitor}
    hub.builders = {60: builder}

    for second in range(55):
        builder.accept_second_bar(
            _second_bar(RTH_START, second),
            close_on_boundary=False,
            skip_volume=True,
        )

    hub.flush_stale_bars(RTH_START + timedelta(seconds=61))

    assert monitor.close_calls == [], (
        "flush_stale_bars passed an incomplete >=60s chart partial to the "
        "strategy; canonical ws_agg/AM completion must own this close"
    )


def test_default_live_model_receives_the_completed_one_minute_ws_bar(
    monkeypatch,
):
    """The registered default model must be eligible at the dispatch gate."""
    monkeypatch.setenv("WS_AGG_SHADOW_ENABLED", "true")
    monkeypatch.setattr(live_bars_writer, "write_bar", lambda *_a, **_k: None)

    hub = SymbolHub("SPY")
    monitor = _RecordingMonitor(60, get_default_live_model())
    hub.monitors = {monitor.strat_id: monitor}
    hub.builders = {60: BarBuilder(60)}

    for second in range(61):
        hub.on_second_bar(_second_bar(RTH_START, second))

    assert len(monitor.close_calls) == 1, (
        f"default live model {monitor.live_model!r} was not eligible for the "
        "1Min ws_agg dispatch path"
    )
    assert monitor.close_calls[0]["close"] == 100.59


def test_no_signal_at_grace_must_leave_final_close_eligible():
    """Evaluating a partial without dispatching is not a fired bucket."""

    class _NoSignalAtGrace:
        tf_seconds = 10
        live_model = "ws_rest_spliced"
        grace_seconds = 4
        _fired_bucket = None
        strat_id = 1

        def fire_on_partial_bucket(self, *_args, **_kwargs):
            return [], {}

    hub = SymbolHub("SPY")
    monitor = _NoSignalAtGrace()
    hub.monitors = {monitor.strat_id: monitor}
    hub.last_tick_time = RTH_START + timedelta(seconds=14)
    builder = SimpleNamespace(
        _bar_count=10,
        _partial=SimpleNamespace(
            bar_start=RTH_START,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=10.0,
        ),
    )

    hub._check_strategy_grace_fires(10, builder, None, {})

    assert monitor._fired_bucket is None, (
        "the bucket was marked fired even though grace evaluation emitted no "
        "signal; a signal formed by the final seconds will be suppressed"
    )


def test_engine_never_started_boot_grace_is_bounded(tmp_path):
    """A live manager heartbeat cannot grant unlimited engine boot grace."""
    engine_file = str(tmp_path / "engine-never-created")
    manager_file = tmp_path / "manager"
    manager_file.write_text("manager is alive")

    after_one_day = 1_000_000.0 + 86_400
    os.utime(manager_file, (after_one_day - 1, after_one_day - 1))
    healthy, reason = engine_health.check(
        now=after_one_day,
        engine_file=engine_file,
        manager_file=str(manager_file),
        max_age_s=300,
    )

    assert not healthy, (
        "engine heartbeat never appeared, but a freshly touched manager file "
        f"kept boot grace healthy after 24h: {reason}"
    )


def _load_replay_function(name: str):
    """Load one pure function without replay_harness import-time DB effects."""
    path = Path(__file__).with_name("replay_harness.py")
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(
        item for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"datetime": datetime, "timedelta": timedelta}
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    return namespace[name]


def test_replay_pairing_maximizes_one_to_one_matches():
    """Nearest-first greedy matching can undercount valid tolerance pairs."""
    pair = _load_replay_function("pair")
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    replay = [origin + timedelta(seconds=s) for s in (4, 8)]
    reference = [origin + timedelta(seconds=s) for s in (0, 5)]

    assert pair(replay, reference, 5) == 2, (
        "both events have a valid one-to-one matching within 5s; greedy "
        "nearest-first selected only one and understates parity"
    )


class _QueryResult:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    def __init__(self, table: str, rows: list[dict]):
        self.table = table
        self.rows = rows
        self.filters: list[tuple[str, str, str]] = []

    def select(self, *_args):
        return self

    def eq(self, *_args):
        return self

    def gte(self, column, value):
        self.filters.append(("gte", column, value))
        return self

    def lte(self, column, value):
        self.filters.append(("lte", column, value))
        return self

    def execute(self):
        rows = list(self.rows)
        for op, column, raw in self.filters:
            bound = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            kept = []
            for row in rows:
                value = row.get(column)
                if value is None:
                    continue
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if (op == "gte" and parsed >= bound) or (
                    op == "lte" and parsed <= bound
                ):
                    kept.append(row)
            rows = kept
        return _QueryResult(rows)


class _FakeSupabase:
    def __init__(self, trades):
        self.trades = trades

    def table(self, name):
        return _FakeQuery(name, self.trades if name == "trades" else [])


def test_replay_backtest_exits_are_windowed_by_exit_timestamp():
    """Entry and exit parity lanes must apply their windows independently."""
    lanes = _load_replay_function("lanes")
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    since = origin + timedelta(seconds=10)
    until = origin + timedelta(seconds=20)

    def iso(second):
        return (origin + timedelta(seconds=second)).isoformat()

    sb = _FakeSupabase([
        # Exit is inside the window, but entry is just before it.
        {"entry_fill_ts": iso(9), "exit_fill_ts": iso(15)},
        # Entry is inside the window, but exit is after it.
        {"entry_fill_ts": iso(15), "exit_fill_ts": iso(25)},
    ])

    _bt_entries, bt_exits, _live_entries, _live_exits = lanes(
        sb, 123, since, until
    )

    assert bt_exits == [origin + timedelta(seconds=15)], (
        "backtest exits were inherited from entry-window rows instead of "
        "being independently filtered by exit_fill_ts"
    )
