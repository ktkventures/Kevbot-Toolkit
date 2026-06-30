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

# Provisional unsettled-tail emission (gated, default OFF). When on, each refresh shows
# the trades in the still-forming window (exit > settled boundary, plus any open position)
# marked provisional=true; they flip to settled=false rows when their bars settle. Its own
# (slower) cadence — a from-cold tail recompute is more expensive than a settled poll.
PROVISIONAL_ENABLED = os.getenv("RORT_SHADOW_PROVISIONAL", "0").strip().lower() in (
    "1", "true", "yes", "on")
PROVISIONAL_REFRESH_S = int(os.getenv("RORT_SHADOW_PROVISIONAL_S", "60"))


# Read-only cache access (M-RS4 Phase 3): the shadow-worker reads bar_cache but must
# NOT backfill it from Polygon (the data-worker owns ingest; plan §3). Default ON.
READ_ONLY_BARS = os.getenv("RORT_SHADOW_READ_ONLY_BARS", "1").strip().lower() in (
    "1", "true", "yes", "on")


def prepare_window(strat: dict, model, since_dt: datetime, until_dt: datetime,
                   timeframe: str, sec_tfs: tuple):
    """Fully-warmed df for `(since-warmup, until]` via the shared production prep
    (`services.prepare_strategy_window_df`) — which uses the secondary-TF-snapshot
    fast path (warmup off the PRIMARY, coarse secondary injected from cache) instead
    of reading multi-day 1Sec, and reads the cache READ-ONLY (no Polygon backfill).
    `timeframe`/`sec_tfs` are derived inside the shared prep; kept in the signature
    for call-site clarity."""
    from services import prepare_strategy_window_df
    return prepare_strategy_window_df(
        strat, since_dt, until_dt, warmup_bars=WARMUP_BARS, data_feed="sip",
        model_override=model, no_backfill=READ_ONLY_BARS,
        persist_snapshot=False,   # read-only consumer never writes the snapshot
    )


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
        self.last_provisional_at = None         # monotonic stamp of last prov refresh
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

    def _provisional_tail(self, slot: EngineSlot, now: datetime,
                          settled_until: datetime) -> List[dict]:
        """Trades in the unsettled window: a from-cold tail recompute over
        `(settled_until - warmup, now]` (last bar partial), returning closed trades whose
        EXIT is past the settled boundary plus any OPEN position. Partition by exit (not
        entry) so a trade open across the boundary is never dropped and never collides
        with the settled lane (which owns exit <= settled_until). Engine-clone is avoided
        (user-pack pickle issue) — the from-cold tail is byte-consistent with the resident
        lane at the boundary."""
        import pandas as pd
        from unified_engine import run_unified_backtest
        from services import get_secondary_tf_map
        from data_worker_engine import _UserContext

        with _UserContext(slot.uid):
            df = prepare_window(slot.strat, slot.bt_model, settled_until, now,
                                slot.timeframe, slot.sec_tfs)
            if df is None or len(df) < 2:
                return []
            sec_tf_map = get_secondary_tf_map(df) or None
            trades_df, _ = run_unified_backtest(
                df, slot.strat, general_packs=self._gen_packs(),
                secondary_tf_map=sec_tf_map, include_open_position=True,
                last_bar_partial=True)
        if trades_df is None or len(trades_df) == 0:
            return []
        su = pd.Timestamp(settled_until)
        su = su.tz_localize('UTC') if su.tz is None else su
        out = []
        for t in trades_df.to_dict('records'):
            ex = t.get('exit_fill_ts')
            if ex is None or (not hasattr(ex, 'isoformat') and pd.isna(ex)) or \
                    (hasattr(ex, 'isoformat') and pd.isna(ex)):
                out.append(t)                       # open position
                continue
            exd = pd.to_datetime(ex, utc=True, errors='coerce')
            if pd.notna(exd) and exd > su:
                out.append(t)                       # closed in the unsettled window
        return out

    def _delete_provisional(self, slot: EngineSlot,
                            settling_until: Optional[datetime] = None) -> None:
        """Delete the slot's provisional rows. With `settling_until`, delete ONLY closed
        provisional rows whose exit has now settled (`exit_fill_ts <= settling_until`) —
        the targeted clear that frees a now-settled trade's key before the settled write
        (open rows have NULL exit and are left). Without it, clear ALL provisional rows
        for the slot (the full-refresh path)."""
        from db import get_admin_client
        q = get_admin_client().table('trades').delete() \
            .eq('strategy_id', slot.sid).eq('data_source', slot.data_source) \
            .eq('provisional', True)
        if settling_until is not None:
            su = settling_until.isoformat() if hasattr(settling_until, 'isoformat') \
                else str(settling_until)
            q = q.lte('exit_fill_ts', su)
        q.execute()

    def _write_provisional(self, slot: EngineSlot, tail: List[dict]) -> int:
        """Insert the fresh provisional tail (provisional=true). No-op in dry_run."""
        if not tail:
            return 0
        if self.dry_run:
            return len(tail)
        import pandas as pd
        from api.services.forward_test_service import _serialize_trades
        from db import insert_trade_admin
        records = _serialize_trades(pd.DataFrame(tail))
        n = 0
        for rec in records:
            rec = dict(rec)
            rec['data_source'] = slot.data_source
            rec['provisional'] = True
            if insert_trade_admin(slot.sid, slot.uid, rec) is not None:
                n += 1
        return n

    def refresh_provisional(self, slot: EngineSlot, now: datetime,
                            settled_until: datetime, force: bool = False) -> int:
        """On its own cadence (PROVISIONAL_REFRESH_S): clear ALL the slot's provisional
        rows and re-emit the current unsettled tail (delete-all + recompute + insert as
        one unit, so rows don't vanish between refreshes). Returns inserted count, or -1
        if not due."""
        import time as _time
        if not PROVISIONAL_ENABLED:
            return 0
        nowm = _time.monotonic()
        if (not force and slot.last_provisional_at is not None
                and nowm - slot.last_provisional_at < PROVISIONAL_REFRESH_S):
            return -1   # not due
        slot.last_provisional_at = nowm
        tail = self._provisional_tail(slot, now, settled_until)
        if not self.dry_run:
            try:
                self._delete_provisional(slot)   # clear all, then re-insert fresh tail
            except Exception as e:  # noqa: BLE001
                logger.warning("[shadow] sid=%s provisional delete-all failed: %s",
                               slot.sid, e)
        return self._write_provisional(slot, tail)

    def poll(self, slot: EngineSlot, now: Optional[datetime] = None) -> dict:
        """One poll for a slot: (optionally clear provisional →) advance the resident
        engine to `now - LAG` and write new settled trades → (optionally re-emit the
        provisional tail).

        Cadence gate: when the engine is already warm and no NEW full settled bar has
        formed since `last_processed_ts`, skip the settled advance — re-preparing a
        warmup-sized window for zero new bars is the bulk of the steady-state cost
        (mirrors data_worker_engine.tick_strategy). Provisional refresh runs on its own
        (slower) cadence regardless, since the forming bar changes continuously.
        """
        if not slot.eligible:
            return {'status': 'ineligible', 'reason': slot.ineligible_reason}
        now = now or datetime.now(timezone.utc)
        settled_until = now - timedelta(minutes=LAG_MINUTES)

        import pandas as pd
        new_settled_bar = slot.engine is None or slot.last_processed_ts is None
        if not new_settled_bar:
            last_dt = pd.Timestamp(slot.last_processed_ts).to_pydatetime()
            new_settled_bar = settled_until > last_dt + timedelta(seconds=slot.tf_seconds)

        # Targeted clear BEFORE the settled write: drop provisional rows whose exit has
        # now settled (exit <= settled_until) so the settled insert can't collide with a
        # stale provisional row on the (sid, entry, exit) unique index. Runs only on a new
        # settled bar; open rows (NULL exit) are untouched. Decoupled from the full
        # refresh so provisional rows persist between refresh ticks.
        if new_settled_bar and PROVISIONAL_ENABLED and not self.dry_run:
            try:
                self._delete_provisional(slot, settling_until=settled_until)
            except Exception as e:  # noqa: BLE001
                logger.warning("[shadow] sid=%s settling-provisional delete failed: %s",
                               slot.sid, e)

        inserted = 0
        if new_settled_bar:
            closed = self.advance(slot, settled_until)
            new = self._filter_new(slot, closed)
            inserted = self.commit(slot, new)

        prov = self.refresh_provisional(slot, now, settled_until)
        status = 'ok' if new_settled_bar else 'no_new_bar'
        return {'status': status, 'inserted': inserted,
                'provisional': max(prov, 0), 'last_bar_ts': slot.last_processed_ts}
