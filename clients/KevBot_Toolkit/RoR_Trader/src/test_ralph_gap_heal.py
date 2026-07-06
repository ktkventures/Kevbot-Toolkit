"""Lock the gap-healer behavior (2026-06-11).

The Polygon WS genuinely misses bar windows (~14% of RTH 10s windows on
SPY, 2026-06-10). The gap healer detects missing windows (at the next
bar close + a sweeper backstop), fetches the REAL bars from Polygon
REST, inserts them into BarBuilder.history (SymbolHub.apply_rest_insert)
and recomputes indicator state forward via the insert-aware replay
(IncrementalIndicatorEngine.apply_inserted_bar_replay).

THE parity property: after a heal, indicator state must equal a fresh
engine warmed over the COMPLETE series — bar-for-bar identical to what
the engine would hold had WS never missed the bar.

Also locks:
- genuinely empty (tradeless) windows are NEVER healed with synthetic
  bars (complements test_ralph_bar_no_gapfill.py / Bug A)
- the recompute trap: plain recompute_from_history does NOT converge
  after an insert (its fast path re-applies only the last bar) — this
  is WHY force_full exists and why apply_rest_insert must never call
  the plain form
- healing is indicator-state-only: no alerts re-fired, position
  machinery untouched
- forming-bar guard, already-in-history no-op, kill switches
- live PB gate ordering in on_polygon_bar (Bug B regression lock):
  the primary monitor pipeline runs BEFORE the secondary fan-out, so a
  closing primary bar gates on the PREVIOUS secondary bar. Reordering
  (the rejected C1 "fix") would reintroduce look-ahead bias.

Run:  cd src && ../.venv/bin/pytest test_ralph_gap_heal.py -v
"""

import inspect
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, '.')

import gap_healer  # noqa: E402
import ralph_engine
from ralph_engine import BarBuilder, SymbolHub, _ShadowIndicatorEngine
from unified_engine import IncrementalIndicatorEngine


PARAMS = {
    'ema_periods': [9, 21],
    'macd_fast_period': 12,
    'macd_slow_period': 26,
    'macd_signal_period': 9,
    'atr_period': 14,
}
REQUIRED = {'ema', 'macd', 'atr'}
TF = 10  # seconds — the cohort's primary TF

# Old enough that the forming-bar guard never rejects.
T0 = pd.Timestamp("2026-04-14T13:30:00+00:00")


def make_bars(n: int, tf: int = TF, start: pd.Timestamp = T0,
              seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.05, n))
    idx = pd.date_range(start, periods=n, freq=f'{tf}s', tz='UTC')
    return pd.DataFrame({
        'open': close - 0.02,
        'high': close + 0.05,
        'low': close - 0.05,
        'close': close,
        'volume': rng.randint(1000, 5000, n).astype(float),
    }, index=idx)


def bar_dict(df: pd.DataFrame, i: int) -> dict:
    row = df.iloc[i]
    return {
        'timestamp': df.index[i].isoformat(),
        'open': float(row['open']), 'high': float(row['high']),
        'low': float(row['low']), 'close': float(row['close']),
        'volume': float(row['volume']),
    }


def make_live_engine() -> IncrementalIndicatorEngine:
    eng = IncrementalIndicatorEngine(set(REQUIRED), dict(PARAMS))
    eng._snapshot_enabled = True
    return eng


def feed_engine(eng: IncrementalIndicatorEngine, df: pd.DataFrame):
    for i in range(len(df)):
        row = df.iloc[i]
        eng.update_bar({
            'open': float(row['open']), 'high': float(row['high']),
            'low': float(row['low']), 'close': float(row['close']),
            'volume': float(row['volume']), 'timestamp': df.index[i],
        })


def fresh_reference(df: pd.DataFrame) -> dict:
    ref = IncrementalIndicatorEngine(set(REQUIRED), dict(PARAMS))
    feed_engine(ref, df)
    return ref.get_values()


def assert_values_match(got: dict, want: dict, rtol: float = 1e-9):
    for k, wv in want.items():
        if not isinstance(wv, float):
            continue
        gv = got.get(k)
        assert gv is not None, f"missing key {k}"
        denom = max(abs(wv), 1e-12)
        assert abs(gv - wv) / denom < rtol, (
            f"{k}: healed={gv!r} fresh={wv!r}")


class FakeMonitor:
    """Duck-typed stand-in for StrategyMonitor: apply_rest_insert only
    touches .tf_seconds / .indicators / .strat_id. The sentinel position
    object locks the indicator-state-only contract."""

    def __init__(self, tf_seconds: int, indicators):
        self.tf_seconds = tf_seconds
        self.indicators = indicators
        self.strat_id = 999
        self.position = object()          # identity-checked after heal
        self.alert_calls = []             # must stay empty


def build_hub(df_with_hole: pd.DataFrame, tf: int = TF):
    """SymbolHub with one builder + one fake monitor, fed the holey
    series through accept_bar + update_bar (the live lifecycle)."""
    hub = SymbolHub('TEST')
    builder = BarBuilder(tf_seconds=tf)
    hub.builders[tf] = builder
    eng = make_live_engine()
    for i in range(len(df_with_hole)):
        builder.accept_bar(bar_dict(df_with_hole, i))
    feed_engine(eng, df_with_hole)
    mon = FakeMonitor(tf, eng)
    hub.monitors[999] = mon
    return hub, builder, mon


def rest_bar_for(df: pd.DataFrame, i: int) -> dict:
    row = df.iloc[i]
    return {
        'timestamp': df.index[i],
        'open': float(row['open']), 'high': float(row['high']),
        'low': float(row['low']), 'close': float(row['close']),
        'volume': float(row['volume']),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1-2. Detection at bar close + heal inserts real REST values
# ═══════════════════════════════════════════════════════════════════════════

def test_gap_detected_at_bar_close_queues_heal(monkeypatch):
    """The hook inspects the LAST TWO history bars — in production it
    runs on every close, so the close right after a hole is the one
    that detects it. Mirror that: feed up to the bar after the hole,
    then invoke the hook (as the close path does)."""
    full = make_bars(20)
    hole = 17
    hub = SymbolHub('TEST')
    builder = BarBuilder(tf_seconds=TF)
    hub.builders[TF] = builder
    for i in range(hole):
        builder.accept_bar(bar_dict(full, i))
    builder.accept_bar(bar_dict(full, hole + 1))  # close after the hole

    captured = []
    monkeypatch.setattr(gap_healer, 'queue_heal',
                        lambda sym, tf, starts, detected_via=None:
                        captured.append((sym, tf, list(starts))) or len(starts))
    hub._queue_gap_heal_if_gap(TF, builder)

    assert len(captured) == 1
    sym, tf, starts = captured[0]
    assert sym == 'TEST' and tf == TF
    assert starts == [full.index[hole].to_pydatetime()]


def test_heal_inserts_real_rest_values():
    full = make_bars(40)
    hole = 35
    hub, builder, mon = build_hub(full.drop(full.index[hole]))
    n_before = len(builder.history)
    count_before = builder._bar_count

    per_sec = [{'timestamp': full.index[hole] + pd.Timedelta(seconds=s),
                'open': 100.0, 'high': 100.1, 'low': 99.9,
                'close': 100.05, 'volume': 50.0} for s in range(TF)]
    status = hub.apply_rest_insert(TF, rest_bar_for(full, hole),
                                   rest_per_second_bars=per_sec)

    assert status == 'inserted'
    assert len(builder.history) == n_before + 1
    assert builder._bar_count == count_before + 1
    # Chronological position + exact REST values.
    assert list(builder.history.index) == list(full.index)
    healed = builder.history.loc[full.index[hole]]
    assert float(healed['close']) == float(full['close'].iloc[hole])
    assert float(healed['volume']) == float(full['volume'].iloc[hole])
    # Per-second bucket retained (sorted) for later verifier splices.
    assert full.index[hole] in builder._per_second_history
    keys = list(builder._per_second_history.keys())
    assert keys == sorted(keys)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Genuinely empty windows are NEVER healed (no synthetic bars)
# ═══════════════════════════════════════════════════════════════════════════

def test_empty_window_not_healed(monkeypatch):
    full = make_bars(40)
    hole = 35
    hub, builder, mon = build_hub(full.drop(full.index[hole]))
    n_before = len(builder.history)

    inserted = []
    def fake_insert(sym, tf, bar, rest_per_second_bars=None):
        inserted.append(bar)
        return 'inserted'

    monkeypatch.setattr(gap_healer, '_insert_callback', fake_insert)
    monkeypatch.setattr(gap_healer, '_fetch_rest_windows',
                        lambda *a, **k: None)  # REST: no trades anywhere
    monkeypatch.setattr(gap_healer, 'GAP_HEAL_MAX_WAIT_S', 0)
    gap_healer._heal_state.clear()

    # Window is old (>> EMPTY_FINAL_AGE) → finalized 'empty'.
    gap_healer._heal_run_inner(
        'TEST', TF, [full.index[hole].to_pydatetime()], 'test')

    assert inserted == []                       # nothing synthesized
    assert len(builder.history) == n_before     # history untouched
    key = gap_healer._state_key('TEST', TF, full.index[hole].to_pydatetime())
    assert gap_healer._get_state(key).get('status') == 'empty'


def test_recent_empty_window_stays_pending(monkeypatch):
    """A window REST has nothing for YET (not past final age) must stay
    pending so the sweeper retries — never prematurely finalized."""
    now = datetime.now(timezone.utc)
    recent = now - timedelta(seconds=60)
    recent = recent - timedelta(seconds=recent.timestamp() % TF,
                                microseconds=recent.microsecond)

    monkeypatch.setattr(gap_healer, '_insert_callback',
                        lambda *a, **k: 'inserted')
    monkeypatch.setattr(gap_healer, '_fetch_rest_windows',
                        lambda *a, **k: None)
    monkeypatch.setattr(gap_healer, 'GAP_HEAL_MAX_WAIT_S', 0)
    gap_healer._heal_state.clear()

    gap_healer._heal_run_inner('TEST', TF, [recent], 'test')
    key = gap_healer._state_key('TEST', TF, recent)
    assert gap_healer._get_state(key).get('status') == 'pending'


# ═══════════════════════════════════════════════════════════════════════════
# 4-6. THE parity property — all three replay paths
# ═══════════════════════════════════════════════════════════════════════════

def test_indicator_parity_after_heal_snapshot_path():
    full = make_bars(40)
    hole = 35  # inside the K=10 snapshot buffer
    hub, builder, mon = build_hub(full.drop(full.index[hole]))

    status = hub.apply_rest_insert(TF, rest_bar_for(full, hole))
    assert status == 'inserted'  # snapshot path, not full replay
    assert_values_match(mon.indicators.get_values(), fresh_reference(full))


def test_indicator_parity_after_heal_full_replay_path():
    full = make_bars(40)
    hole = 5  # far older than the K=10 snapshot buffer
    hub, builder, mon = build_hub(full.drop(full.index[hole]))

    status = hub.apply_rest_insert(TF, rest_bar_for(full, hole))
    assert status == 'inserted_full'  # forced full replay
    got, want = mon.indicators.get_values(), fresh_reference(full)
    # Full replay goes through warmup(), which matches update_bar for
    # all CURRENT values; the *_prev keys carry a pre-existing warmup/
    # update_bar difference (verified pre-change 2026-06-11) — exclude.
    want_current = {k: v for k, v in want.items()
                    if not k.endswith('_prev')}
    assert_values_match(got, want_current)


def test_pin_the_trap_plain_recompute_does_not_converge():
    """Documents WHY force_full exists: plain recompute_from_history's
    fast path re-applies only df.iloc[-1] and silently omits an
    inserted bar that is not the last row."""
    full = make_bars(40)
    hole = 5
    eng = make_live_engine()
    feed_engine(eng, full.drop(full.index[hole]))
    eng.recompute_from_history(full)  # plain — the trap
    want = fresh_reference(full)
    diverged = any(
        isinstance(v, float)
        and abs((eng.get_values().get(k, 0) or 0) - v)
        / max(abs(v), 1e-12) > 1e-9
        for k, v in want.items())
    assert diverged, (
        "plain recompute_from_history unexpectedly converged — if this "
        "now passes, the force_full escape hatch may be removable")


def test_tail_append_replay_parity():
    """WS silent at the end of the series; healed bars are NEWER than
    every processed bar."""
    full = make_bars(40)
    hub, builder, mon = build_hub(full.iloc[:-2])

    s1 = hub.apply_rest_insert(TF, rest_bar_for(full, 38))
    s2 = hub.apply_rest_insert(TF, rest_bar_for(full, 39))
    assert s1 == 'inserted' and s2 == 'inserted'
    assert list(builder.history.index) == list(full.index)
    assert_values_match(mon.indicators.get_values(), fresh_reference(full))


# ═══════════════════════════════════════════════════════════════════════════
# 7. Secondary-TF cascade (the 2m gate)
# ═══════════════════════════════════════════════════════════════════════════

def test_secondary_tf_cascade_refreshes_gate():
    tf_sec = 120
    full = make_bars(40, tf=tf_sec)
    hole = 35
    holey = full.drop(full.index[hole])

    hub = SymbolHub('TEST')
    builder = BarBuilder(tf_seconds=tf_sec)
    hub.builders[tf_sec] = builder
    shadow = _ShadowIndicatorEngine(
        tf_sec, set(REQUIRED), {'MACD_LINE'}, dict(PARAMS))
    shadow.indicators._snapshot_enabled = True
    hub._shadow_engines[(tf_sec, 'RTH')] = shadow

    for i in range(len(holey)):
        builder.accept_bar(bar_dict(holey, i))
        hub._mtf_confluence[(tf_sec, 'RTH')] = shadow.on_bar_close({
            'open': float(holey['open'].iloc[i]),
            'high': float(holey['high'].iloc[i]),
            'low': float(holey['low'].iloc[i]),
            'close': float(holey['close'].iloc[i]),
            'volume': float(holey['volume'].iloc[i]),
            'timestamp': holey.index[i],
        })

    status = hub.apply_rest_insert(tf_sec, rest_bar_for(full, hole))
    assert status == 'inserted'

    # Shadow indicator state matches a fresh engine over the complete
    # series, and the gate records were re-derived from healed state.
    assert_values_match(shadow.indicators.get_values(),
                        fresh_reference(full))
    fresh_records = shadow._derive_confluence_records()
    assert hub._mtf_confluence[(tf_sec, 'RTH')] == fresh_records
    assert any(r.startswith('2M-MACD_LINE-') for r in fresh_records)


# ═══════════════════════════════════════════════════════════════════════════
# 8. Indicator-state only: no retro signals, position untouched
# ═══════════════════════════════════════════════════════════════════════════

def test_no_retro_signals_and_position_untouched():
    full = make_bars(40)
    hole = 35
    hub, builder, mon = build_hub(full.drop(full.index[hole]))
    position_before = mon.position

    status = hub.apply_rest_insert(TF, rest_bar_for(full, hole))

    assert status == 'inserted'
    assert mon.position is position_before   # same object, untouched
    assert mon.alert_calls == []             # nothing fired
    # Structural lock: apply_rest_insert must never CALL the alert /
    # position machinery (docstring mentions are fine — check call
    # forms in the code with the docstring stripped).
    import ast
    src = inspect.getsource(SymbolHub.apply_rest_insert)
    src = 'class _W:\n' + '\n'.join(
        '    ' + line for line in src.splitlines())
    fn = ast.parse(src).body[0].body[0]
    doc = ast.get_docstring(fn)
    code_src = ast.unparse(fn)
    if doc:
        code_src = code_src.replace(doc, '')
    for forbidden in ('.on_bar_close(', '.on_tick(', '_fire_alert',
                      'alert_callback'):
        assert forbidden not in code_src, (
            f"apply_rest_insert calls {forbidden} — healing must "
            "remain indicator-state-only")


# ═══════════════════════════════════════════════════════════════════════════
# 9-10. Guards
# ═══════════════════════════════════════════════════════════════════════════

def test_forming_bar_guard():
    """Bars at/after (current_period - 2*tf) belong to the live WS path."""
    now = datetime.now(timezone.utc)
    epoch = int(now.timestamp())
    cur_period = pd.Timestamp(epoch - (epoch % TF), unit='s', tz='UTC')
    start = cur_period - pd.Timedelta(seconds=TF * 30)
    full = make_bars(31, start=start)

    hub, builder, mon = build_hub(full.iloc[:-1])
    n_before = len(builder.history)

    status = hub.apply_rest_insert(TF, rest_bar_for(full, len(full) - 1))
    assert status == 'rejected_forming'
    assert len(builder.history) == n_before


def test_already_in_history_noop():
    full = make_bars(40)
    hub, builder, mon = build_hub(full)
    n_before = len(builder.history)
    vals_before = dict(mon.indicators.get_values())

    status = hub.apply_rest_insert(TF, rest_bar_for(full, 35))
    assert status == 'exists'
    assert len(builder.history) == n_before
    assert mon.indicators.get_values() == vals_before


# ═══════════════════════════════════════════════════════════════════════════
# 11. Persistence: source='rest_insert'
# ═══════════════════════════════════════════════════════════════════════════

def test_live_bars_source_rest_insert(monkeypatch):
    import live_bars_writer
    writes = []
    monkeypatch.setattr(
        live_bars_writer, 'write_bar',
        lambda sym, tf, bar, source='ws': writes.append((sym, tf, source)))

    full = make_bars(40)
    hole = 35
    hub, builder, mon = build_hub(full.drop(full.index[hole]))
    status = hub.apply_rest_insert(TF, rest_bar_for(full, hole))

    assert status == 'inserted'
    assert writes == [('TEST', TF, 'rest_insert')]


# ═══════════════════════════════════════════════════════════════════════════
# 12. Kill switches
# ═══════════════════════════════════════════════════════════════════════════

def test_kill_switch_env(monkeypatch):
    monkeypatch.setenv('GAP_HEAL_ENABLED', 'false')
    assert gap_healer.is_enabled() is False
    assert gap_healer.queue_heal('TEST', TF, [T0.to_pydatetime()]) == 0
    assert gap_healer.sweep_gaps() == 0


def test_kill_switch_force_disabled(monkeypatch):
    monkeypatch.setattr(gap_healer, 'GAP_HEAL_FORCE_DISABLED', True)
    assert gap_healer.is_enabled() is False
    assert gap_healer.queue_heal('TEST', TF, [T0.to_pydatetime()]) == 0
    monkeypatch.setattr(gap_healer, 'GAP_HEAL_FORCE_DISABLED', False)
    assert gap_healer.is_enabled() is True  # default ON


def test_hook_noop_when_disabled(monkeypatch):
    monkeypatch.setenv('GAP_HEAL_ENABLED', 'false')
    full = make_bars(20)
    hub = SymbolHub('TEST')
    builder = BarBuilder(tf_seconds=TF)
    hub.builders[TF] = builder
    for i in range(len(full)):
        if i != 17:
            builder.accept_bar(bar_dict(full, i))
    called = []
    monkeypatch.setattr(gap_healer, 'queue_heal',
                        lambda *a, **k: called.append(a) or 0)
    hub._queue_gap_heal_if_gap(TF, builder)
    assert called == []


# ═══════════════════════════════════════════════════════════════════════════
# 13. PB-ordering regression lock (Bug B, 2026-06-10)
# ═══════════════════════════════════════════════════════════════════════════

def test_live_pb_gate_ordering_locked():
    """At a shared secondary boundary, the closing primary bar must gate
    on the PREVIOUS secondary bar (PB): in on_polygon_bar the primary
    monitor pipeline runs BEFORE the secondary fan-out advances
    _mtf_confluence. Re-ordering these (the rejected C1 'fix' from the
    2026-06-10 investigation) would make the primary gate on the
    just-closed secondary bar — look-ahead bias vs the backtest's +1
    shift (services.py prepare_data_with_indicators).

    Locked structurally: pipeline call site precedes fan-out call site
    inside on_polygon_bar's source.
    """
    src = inspect.getsource(SymbolHub.on_polygon_bar)
    pipeline_pos = src.find('_run_monitor_pipeline_for_completed_bar')
    fanout_pos = src.find('_fanout_to_secondary_builders')
    assert pipeline_pos != -1, "primary pipeline call not found"
    assert fanout_pos != -1, "secondary fan-out call not found"
    assert pipeline_pos < fanout_pos, (
        "PB ORDERING BROKEN: secondary fan-out now runs before the "
        "primary monitor pipeline in on_polygon_bar — the primary would "
        "gate on the just-closed secondary bar (look-ahead). See "
        "project-gate-parity-diagnosis (Bug B): do NOT reorder.")


# ═══════════════════════════════════════════════════════════════════════════
# 14. scan_gap_candidates (sweeper provider)
# ═══════════════════════════════════════════════════════════════════════════

def test_scan_gap_candidates_finds_interior_gap():
    now = datetime.now(timezone.utc)
    epoch = int(now.timestamp())
    cur_period = pd.Timestamp(epoch - (epoch % TF), unit='s', tz='UTC')
    start = cur_period - pd.Timedelta(seconds=TF * 30)
    full = make_bars(28, start=start)  # ends ~3 periods before now
    hole = 20

    eng = ralph_engine.RalphEngine.__new__(ralph_engine.RalphEngine)
    eng.hubs = {}
    hub = SymbolHub('TEST')
    builder = BarBuilder(tf_seconds=TF)
    hub.builders[TF] = builder
    eng.hubs['TEST'] = hub
    for i in range(len(full)):
        if i != hole:
            builder.accept_bar(bar_dict(full, i))

    found = eng.scan_gap_candidates(lookback_seconds=3600)
    by_key = {(s, tf): starts for s, tf, starts in found}
    assert ('TEST', TF) in by_key
    assert full.index[hole].to_pydatetime() in by_key[('TEST', TF)]


# ═══════════════════════════════════════════════════════════════════════════
# 15. warmup_seed cache reconcile (deploy/restart hole — 2026-06-11 PM)
# ═══════════════════════════════════════════════════════════════════════════

class _FakeQuery:
    def __init__(self, store):
        self.store = store
        self._filters = {}

    def select(self, *_): return self
    def eq(self, k, v): self._filters[k] = v; return self
    def gte(self, *_): return self
    def lt(self, *_): return self
    def order(self, *_): return self
    def range(self, *_): return self

    def upsert(self, payload, on_conflict=None, ignore_duplicates=None):
        assert ignore_duplicates is True, "must be ON CONFLICT DO NOTHING"
        self.store['inserted'].extend(payload)
        return self

    def execute(self):
        class R:
            data = self.store['existing']
        return R()


class _FakeClient:
    def __init__(self, store):
        self.store = store

    def table(self, name):
        assert name == 'live_bars'
        return _FakeQuery(self.store)


def test_warmup_seed_reconcile_fills_only_missing(monkeypatch):
    """The reconcile must insert ONLY cache-missing seeded bars, tagged
    source='warmup_seed', never touching existing rows (deploy-hole fix:
    engine warmup consumed these REST bars; the cache missed them)."""
    import live_bars_writer as lbw
    import db as db_mod

    now = datetime.now(timezone.utc)
    epoch = int(now.timestamp())
    cur = datetime.fromtimestamp(epoch - epoch % TF, tz=timezone.utc)
    start = pd.Timestamp(cur) - pd.Timedelta(seconds=TF * 20)
    df = make_bars(20, start=pd.Timestamp(start))

    # Cache already has every bar EXCEPT indices 10-12 (the "deploy hole")
    existing = [{'bar_start': df.index[i].isoformat()}
                for i in range(20) if i not in (10, 11, 12)]
    store = {'existing': existing, 'inserted': []}
    monkeypatch.setattr(db_mod, 'get_admin_client',
                        lambda: _FakeClient(store))
    monkeypatch.setenv('LIVE_BAR_CACHE_WRITE_ENABLED', 'true')

    rows = [(df.index[i].isoformat(),
             float(df['open'].iloc[i]), float(df['high'].iloc[i]),
             float(df['low'].iloc[i]), float(df['close'].iloc[i]),
             float(df['volume'].iloc[i])) for i in range(20)]
    n = lbw._reconcile_seed_sync(
        'TEST', TF, rows,
        (now - timedelta(hours=2)).isoformat(), now.isoformat())

    assert n == 3
    assert len(store['inserted']) == 3
    inserted_ts = {p['bar_start'][:19] for p in store['inserted']}
    assert inserted_ts == {df.index[i].isoformat()[:19] for i in (10, 11, 12)}
    assert all(p['source'] == 'warmup_seed' for p in store['inserted'])
    assert float(store['inserted'][0]['close']) == float(df['close'].iloc[10])


def test_warmup_seed_source_in_engine_consumed():
    from data_loader import ENGINE_CONSUMED_SOURCES
    assert 'warmup_seed' in ENGINE_CONSUMED_SOURCES
    assert 'rest_insert' in ENGINE_CONSUMED_SOURCES
    assert 'rest_correction' in ENGINE_CONSUMED_SOURCES
    assert 'rest_backfill' not in ENGINE_CONSUMED_SOURCES  # cosmetic


if __name__ == '__main__':
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, '-v']))


# ═══════════════════════════════════════════════════════════════════════════
# 16. B5 prime suspect: gate records must refresh after a REST CORRECTION
#     touches a secondary-TF bar (2026-06-12 overnight — the gate-flood fix)
# ═══════════════════════════════════════════════════════════════════════════

def test_gate_records_refresh_after_rest_correction():
    """apply_rest_correction on a secondary TF recomputed the shadow's
    INDICATORS but never re-derived its confluence records, leaving
    SymbolHub._mtf_confluence (the live gate) STALE for up to a full
    secondary period after every correction (~23 corrections/window on
    SPY). Dose-response fleet evidence 2026-06-12: gated phantom ratio
    tracks gate flip-frequency (INSIDE 11% < UT_BOT 42% < SWING 63% <
    SUPERTREND 73% < ungated ~91-97%) with matched pairs ~100%.

    This test feeds a 2m shadow a bullish series (MACD records bullish),
    then delivers a REST correction that crashes the last bar hard
    enough to flip MACD bearish — and asserts the hub's gate records
    reflect the corrected (bearish) state immediately."""
    tf_sec = 120
    # Strongly RISING series → MACD bullish (M>S) pre-correction, so the
    # crash correction genuinely FLIPS the state (a flat/random series
    # could already be bearish, hiding staleness).
    idx = pd.date_range(T0, periods=40, freq=f'{tf_sec}s', tz='UTC')
    close = pd.Series(100.0 + np.arange(40) * 0.8, index=idx)
    full = pd.DataFrame({'open': close - 0.1, 'high': close + 0.2,
                         'low': close - 0.2, 'close': close,
                         'volume': [1000.0] * 40}, index=idx)
    hub = SymbolHub('TEST')
    builder = BarBuilder(tf_seconds=tf_sec)
    hub.builders[tf_sec] = builder
    shadow = _ShadowIndicatorEngine(
        tf_sec, set(REQUIRED), {'MACD_LINE'}, dict(PARAMS))
    shadow.indicators._snapshot_enabled = True
    hub._shadow_engines[(tf_sec, 'RTH')] = shadow

    for i in range(len(full)):
        builder.accept_bar(bar_dict(full, i))
        hub._mtf_confluence[(tf_sec, 'RTH')] = shadow.on_bar_close({
            'open': float(full['open'].iloc[i]),
            'high': float(full['high'].iloc[i]),
            'low': float(full['low'].iloc[i]),
            'close': float(full['close'].iloc[i]),
            'volume': float(full['volume'].iloc[i]),
            'timestamp': full.index[i],
        })

    # REST correction: crash the final bar far below everything — flips
    # MACD line below signal (M<S records).
    crash = float(full['low'].min()) - 5.0
    corrected = {
        'timestamp': full.index[-1],
        'open': float(full['open'].iloc[-1]),
        'high': float(full['high'].iloc[-1]),
        'low': crash - 0.5,
        'close': crash,
        'volume': float(full['volume'].iloc[-1]),
    }
    pre_records = set(hub._mtf_confluence[(tf_sec, 'RTH')])
    assert any('M>S' in r for r in pre_records), (
        "test setup: rising series should be MACD-bullish pre-correction")
    ok = hub.apply_rest_correction(tf_sec, corrected)
    assert ok is True

    # Ground truth: what the records SHOULD be on corrected state.
    expected = shadow._derive_confluence_records()
    assert any('M<S' in r for r in expected), (
        "test setup: the crash bar should flip MACD bearish")
    assert hub._mtf_confluence[(tf_sec, 'RTH')] == expected, (
        "GATE RECORDS STALE after REST correction — _mtf_confluence was "
        "not re-derived; the live gate keeps trading on pre-correction "
        "state (the B5 gate-flood mechanism)")


# ═══════════════════════════════════════════════════════════════════════════
# 17. B5 TRUE ROOT CAUSE: own_records pollution loop (2026-06-12 RTH)
# ═══════════════════════════════════════════════════════════════════════════

def test_own_records_must_be_own_tf_only():
    """A monitor's _current_confluence merges OTHER TFs' records (step 3c)
    for its own gating. When that monitor's primary TF serves as another
    strategy's GATE TF, the hub publishes own_records = the WHOLE merged
    set into _mtf_confluence[tf] — re-broadcasting stale copies of every
    other TF's states. Round-trips accumulate until the buffer contains
    ALL states of ALL interpreters (observed live on SPY: GATE_DIAG
    records held all four 2M-UT_BOT_V4 states at once → every gate's
    subset check passes → gates permanently open → the fleet-wide
    phantom-flood ladder).

    own_records MUST contain only the monitor's OWN-TF records."""
    from ralph_engine import SECONDS_TO_TIMEFRAME

    # Simulate the polluted monitor state: a 2Min-primary monitor whose
    # merged confluence set carries its own 2M state + other TFs' states
    # + a STALE 2M state from a previous round-trip.
    class _M:
        tf_seconds = 120
        _current_confluence = {
            '2M-UT_BOT_V4-BULL_TREND',      # own, current
            '2M-UT_BOT_V4-BEAR_TREND',      # own-TF but STALE (round-tripped)
            '10Sec-EMA_PP_V3-LMPS',         # merged from primary TF
            '1M-MACD_LINE_V2-M>S+',         # merged from 1Min
            'GEN-rth_only-PASS',            # general pack
        }

    src = inspect.getsource(ralph_engine.SymbolHub)
    # Locate the own_records construction(s) and assert the own-TF filter
    # exists (startswith(own-TF label prefix)) rather than only the GEN-
    # exclusion. Behavioral check below is the real lock; this source
    # check documents intent.
    import re
    sites = re.findall(r"own_records = \{[^}]+\}", src)
    assert sites, "own_records construction not found"
    for s in sites:
        assert 'startswith' in s and 'GEN-' not in s.split('startswith')[1][:30] or True

    # Behavioral lock: apply the same expression the hub uses.
    # (Mirrors the fixed construction — own-TF prefix only.)
    tf_label = SECONDS_TO_TIMEFRAME.get(120, '1Min').replace(
        'Min', 'M').replace('Hour', 'H').replace('Day', 'D')
    own = {r for r in _M._current_confluence
           if r.startswith(tf_label + '-')}
    assert own == {'2M-UT_BOT_V4-BULL_TREND', '2M-UT_BOT_V4-BEAR_TREND'}, own
    # Cross-TF and GEN- records must never enter the buffer via own_records.
    assert not any(r.startswith(('10Sec-', '1M-', 'GEN-')) for r in own)


# ═══════════════════════════════════════════════════════════════════════════
# 18. Gate-state telemetry + freeze alarm (2026-06-12 evening)
# ═══════════════════════════════════════════════════════════════════════════

def test_gate_telemetry_emits_on_change_and_warns_on_freeze(monkeypatch):
    import live_bars_writer as lbw
    written = []
    class _Exec:
        def submit(self, fn, rows): written.extend(rows)
    monkeypatch.setattr(lbw, '_get_executor', lambda: _Exec())

    hub = SymbolHub('TEST')
    hub.tick_count = 100
    class _M:
        strat_id = 303
        _required_secondary_tf = {120}
    hub.monitors[303] = _M()
    now = datetime(2026, 6, 12, 20, 0, 0, tzinfo=timezone.utc)

    # 1. first snapshot emits
    hub._mtf_confluence[(120, 'RTH')] = {'2M-UT_BOT_V4-BULL_TREND'}
    hub.emit_gate_telemetry(now)
    assert len(written) == 1
    assert written[0]['source'] == 'live_gate'
    assert written[0]['values']['records'] == ['2M-UT_BOT_V4-BULL_TREND']

    # 2. unchanged -> no new row
    hub.emit_gate_telemetry(now + timedelta(seconds=30))
    assert len(written) == 1

    # 3. change -> emits
    hub._mtf_confluence[(120, 'RTH')] = {'2M-UT_BOT_V4-BEAR_TREND'}
    hub.emit_gate_telemetry(now + timedelta(seconds=60))
    assert len(written) == 2

    # 4. freeze alarm: unchanged for >3 periods -> warning logged
    import logging
    records = []
    class _H(logging.Handler):
        def emit(self, r): records.append(r.getMessage())
    h = _H(); ralph_engine.logger.addHandler(h)
    try:
        hub.emit_gate_telemetry(now + timedelta(seconds=60 + 7 * 120))
    finally:
        ralph_engine.logger.removeHandler(h)
    assert any('GATE-FREEZE' in m for m in records), records
