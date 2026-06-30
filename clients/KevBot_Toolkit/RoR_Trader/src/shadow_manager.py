"""M-RS4 Phase 3 — ResidentEngineManager (Step C integration).

Wires the validated pure-compute core (`shadow_engine.ResidentStrategyEngine`) to live
data: discover strategies on a symbol shard, hold one warmed resident engine per
strategy, poll `bar_cache` (REST Bars) for new SETTLED bars, feed them in incrementally
(never re-window the engine), and write the resulting `backtest_<model>` trades.

Composition of proven primitives:
  - data prep  = the windowed, warmup-bounded `prepare_data_with_indicators` that
    `services.get_strategy_trades_for_window` uses (the trusted append-path column
    source; honors the never-reduce-warmup rule). The engine recomputes BUILT-IN
    indicators itself; only the user-pack + secondary-TF COLUMNS for new bars come
    from the df, and those depend only on a bar's own warmup window — so a windowed
    re-prepare per poll is stable.
  - engine     = ResidentStrategyEngine.feed(new_bars) — proven byte-identical to a
    from-cold full recompute (harness PATH C/D), given the same pre-computed columns.

Combine them and the manager's settled trades equal a from-cold recompute over the same
window — that is `_shadow_manager_validate.py`'s offline gate (and Step F's gate).

SCOPE (v1): SETTLED-ONLY. Each engine processes only bars at/below the commit boundary
`now - LAG`; trades are written non-provisional (the `trades.provisional` column defaults
false). The unsettled-tail provisional emission (feed forming bars to an engine CLONE so
the resident engine stays pinned to the settled boundary) is a follow-up increment — the
column is already in place for it.

This module does the IO (bar_cache reads via prepare_data_with_indicators, trade writes);
the engine core stays pure. `dry_run=True` computes everything but writes nothing — used
by the validation harness and for safe shadow-mode bring-up.
"""
from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional

logger = logging.getLogger("shadow_worker")

# Trade-commit lag — bars within `now - LAG` are still unsettled. Mirrors the cron /
# data-worker (forward_test_service `_ALGO_HISTORY_LAG_MINUTES`) so every lane agrees
# on "settled."
LAG_MINUTES = int(os.getenv("ALGO_HISTORY_LAG_MINUTES", "15"))
WARMUP_BARS = int(os.getenv("RORT_SHADOW_WARMUP_BARS", "300"))
# How far back a cold engine warms+re-emits on bootstrap when it has no DB anchor.
BOOTSTRAP_DAYS = float(os.getenv("RORT_SHADOW_BOOTSTRAP_DAYS", "1"))

# REST-servable backtest models — the shadow lane reads bar_cache (Polygon REST). Mirror
# data_worker_engine.REST_BACKTEST_MODELS; cache_locked/corrected come from live_bars.
REST_BACKTEST_MODELS = {None, "", "rest_only", "rest_hifi"}


def _warmup_days(timeframe: str, sec_tfs: tuple) -> int:
    """Warmup window sized off the LONGEST (binding) TF — mirrors
    get_strategy_trades_for_window (services.py)."""
    from data_loader import BARS_PER_DAY
    all_tfs = [timeframe] + list(sec_tfs)
    bpds = [BARS_PER_DAY.get(tf, 390) for tf in all_tfs]
    binding_bpd = min(bpd for bpd in bpds if bpd > 0) if bpds else 390
    return max(1, math.ceil(WARMUP_BARS / max(binding_bpd, 0.001) * 365 / 252))


def prepare_window(strat: dict, model, since_dt: datetime, until_dt: datetime,
                   timeframe: str, sec_tfs: tuple):
    """Cold (no-snapshot) windowed prepare → fully-warmed df for `(since-warmup, until]`.
    Mirrors get_strategy_trades_for_window's prep so the engine sees identical inputs."""
    import pandas as pd
    from services import prepare_data_with_indicators

    wd = _warmup_days(timeframe, sec_tfs)
    since_naive = since_dt.replace(tzinfo=None) if since_dt.tzinfo else since_dt
    end_naive = until_dt.replace(tzinfo=None) if until_dt.tzinfo else until_dt
    start_date = since_naive - timedelta(days=wd)

    df = prepare_data_with_indicators(
        strat['symbol'], seed=strat.get('data_seed', 42),
        start_date=start_date, end_date=end_naive, timeframe=timeframe,
        data_feed="sip", session=strat.get('trading_session', 'RTH'),
        secondary_tfs=sec_tfs, secondary_tf_dfs=None, strat=strat,
        model_override=model,
    )
    if df is None or len(df) == 0:
        return df
    end_clip = pd.Timestamp(end_naive)
    if df.index.tz is not None and end_clip.tz is None:
        end_clip = end_clip.tz_localize('UTC')
    elif df.index.tz is None and end_clip.tz is not None:
        end_clip = end_clip.tz_localize(None)
    return df[df.index <= end_clip]


class EngineSlot:
    """Per-strategy resident state: the live engine + bookkeeping. Classification
    (eligibility, model, secondary TFs, fingerprint) is derived via the data-worker's
    `classify_strategy` so the two lanes agree on eligibility."""

    def __init__(self, sid: int, uid: str, strat: dict):
        self.sid = sid
        self.uid = uid
        self.strat = strat
        from shadow_engine import ResidentStrategyEngine  # noqa: F401 (type ref)
        self.engine = None                      # ResidentStrategyEngine | None
        self.last_processed_ts = None           # tz-aware Timestamp (engine boundary)
        self.last_entry_written: Optional[str] = None
        self.classify()

    def classify(self) -> None:
        from data_worker_engine import StrategyEngineState, classify_strategy
        st = StrategyEngineState(self.sid, self.uid, self.strat)
        classify_strategy(st)
        old_fp = getattr(self, 'fingerprint', None)
        self.symbol = st.symbol
        self.timeframe = st.timeframe
        self.tf_seconds = st.tf_seconds
        self.bt_model = st.bt_model
        self.data_source = st.data_source
        self.sec_tfs = st.sec_tfs
        self.fingerprint = st.fingerprint
        self.eligible = st.streaming_eligible and st.bt_model in REST_BACKTEST_MODELS
        self.ineligible_reason = st.ineligible_reason
        # A config edit (fingerprint change) invalidates the resident engine.
        if old_fp is not None and self.fingerprint != old_fp:
            self.engine = None
            self.last_processed_ts = None


class ResidentEngineManager:
    """Owns the resident engines for a symbol shard. The shadow-worker constructs one,
    calls discover() periodically, and poll()s each eligible slot on a cadence."""

    def __init__(self, shard_symbols=None, dry_run: bool = False):
        self.shard_symbols = set(shard_symbols) if shard_symbols else None
        self.dry_run = dry_run
        self.slots: dict[int, EngineSlot] = {}
        self._enabled_gen = None

    def _gen_packs(self):
        if self._enabled_gen is None:
            import general_packs as gp
            self._enabled_gen = gp.get_enabled_general_packs(gp.load_general_packs())
        return self._enabled_gen

    def discover(self) -> dict:
        """Refresh the slot set from monitored strategies (data-worker filters), scoped
        to the shard. Adds new, drops removed, re-classifies the rest."""
        from db import load_all_desired_states, load_strategies_monitoring_admin
        desired = load_all_desired_states() or []
        uids = sorted({d.get('user_id') for d in desired if d.get('user_id')})
        seen = set()
        for uid in uids:
            try:
                strategies = load_strategies_monitoring_admin(uid) or []
            except Exception as e:  # noqa: BLE001
                logger.warning("[shadow] load strategies user=%s failed: %s", uid, e)
                continue
            for strat in strategies:
                if not strat.get('symbol'):
                    continue
                if 'entry_trigger_confluence_id' not in strat:
                    continue
                if strat.get('strategy_origin') == 'webhook_inbound':
                    continue
                cfg = strat.get('config') if isinstance(strat.get('config'), dict) else {}
                if cfg.get('snapshot_subscribe_enabled', True) is False:
                    continue
                if self.shard_symbols and strat.get('symbol') not in self.shard_symbols:
                    continue
                sid = strat.get('id')
                if sid is None:
                    continue
                seen.add(sid)
                slot = self.slots.get(sid)
                if slot is None:
                    self.slots[sid] = EngineSlot(sid, uid, strat)
                else:
                    slot.strat = strat
                    slot.classify()
        for sid in list(self.slots):
            if sid not in seen:
                del self.slots[sid]
        return self.slots

    def advance(self, slot: EngineSlot, until_dt: datetime,
                since_override: Optional[datetime] = None) -> List[dict]:
        """Apply new settled bars up to `until_dt` to the slot's resident engine and
        return the NEW closed trade dicts (engine-shaped). Bootstraps a cold engine.

        Never re-windows the engine: `feed` only applies bars after its `last_bar_ts`.
        The windowed re-prepare supplies warm user-pack/secondary columns for those bars.
        """
        import pandas as pd
        from services import get_secondary_tf_map
        from shadow_engine import ResidentStrategyEngine

        from data_worker_engine import _UserContext

        since = since_override or slot.last_processed_ts
        if since is None:
            since = until_dt - timedelta(days=BOOTSTRAP_DAYS)
        if isinstance(since, pd.Timestamp):
            since = since.to_pydatetime()

        # The prep path loads PER-USER config (confluence groups, general packs); without
        # the slot's user context it queries user_id=None and silently drops them.
        with _UserContext(slot.uid):
            df = prepare_window(slot.strat, slot.bt_model, since, until_dt,
                                slot.timeframe, slot.sec_tfs)
            if df is None or len(df) == 0:
                return []
            sec_tf_map = get_secondary_tf_map(df) or None

            if slot.engine is None:
                slot.engine = ResidentStrategyEngine(slot.strat, self._gen_packs())

            new = slot.engine.feed(df, sec_tf_map)
        slot.last_processed_ts = slot.engine.last_bar_ts
        return [t for t in (new or []) if t.get('exit_fill_ts')]

    def _filter_new(self, slot: EngineSlot, closed: List[dict]) -> List[dict]:
        """Closed trades whose entry is strictly newer than the DB anchor (idempotent
        with the (strategy_id, entry_fill_ts, exit_fill_ts) unique index regardless)."""
        if not closed:
            return []
        anchor = slot.last_entry_written
        if not anchor:
            return closed
        return [t for t in closed if str(t.get('entry_fill_ts')) > str(anchor)]

    def commit(self, slot: EngineSlot, closed: List[dict]) -> int:
        """Write settled closed trades. No-op in dry_run. Advances the entry anchor."""
        if not closed:
            return 0
        if self.dry_run:
            mx = slot.last_entry_written
            for t in closed:
                ek = t.get('entry_fill_ts')
                if ek and (mx is None or str(ek) > str(mx)):
                    mx = str(ek)
            slot.last_entry_written = mx
            return len(closed)

        import pandas as pd
        from api.services.forward_test_service import _serialize_trades
        from db import insert_trade_admin

        records = _serialize_trades(pd.DataFrame(closed))
        inserted = 0
        mx = slot.last_entry_written
        for rec in records:
            rec = dict(rec)
            rec['data_source'] = slot.data_source
            rec['provisional'] = False  # v1: settled-only
            if insert_trade_admin(slot.sid, slot.uid, rec) is not None:
                inserted += 1
            ek = rec.get('entry_fill_ts')
            if ek and (mx is None or str(ek) > str(mx)):
                mx = str(ek)
        slot.last_entry_written = mx
        return inserted

    def poll(self, slot: EngineSlot, now: Optional[datetime] = None) -> dict:
        """One settled poll for a slot: advance to `now - LAG`, write new closed trades.

        Cadence gate: when the engine is already warm and no NEW full settled bar has
        formed since `last_processed_ts`, skip — re-preparing a warmup-sized window every
        cycle for zero new bars is the bulk of the steady-state cost. Mirrors
        data_worker_engine.tick_strategy's no_new_bar gate. (A cold engine always runs:
        the first poll must bootstrap.)
        """
        if not slot.eligible:
            return {'status': 'ineligible', 'reason': slot.ineligible_reason}
        now = now or datetime.now(timezone.utc)
        settled_until = now - timedelta(minutes=LAG_MINUTES)
        if slot.engine is not None and slot.last_processed_ts is not None:
            import pandas as pd
            last_dt = pd.Timestamp(slot.last_processed_ts).to_pydatetime()
            if settled_until <= last_dt + timedelta(seconds=slot.tf_seconds):
                return {'status': 'no_new_bar', 'inserted': 0,
                        'last_bar_ts': slot.last_processed_ts}
        closed = self.advance(slot, settled_until)
        new = self._filter_new(slot, closed)
        inserted = self.commit(slot, new)
        return {'status': 'ok', 'inserted': inserted,
                'last_bar_ts': slot.last_processed_ts}
