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
import threading
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
# Persist the secondary-TF snapshot on a full compute so the NEXT bootstrap of a
# secondary-gated strategy takes the cheap fast path (warmup off the PRIMARY) instead
# of a slow multi-day full resample. This writes only the strategy's OWN config cache
# (secondary_snapshot_b64) — NOT bar_cache/Polygon — so it preserves the read-only-bars
# contract while fixing bootstrap perf for strategies the append lane hasn't warmed.
# Default ON; gate off with RORT_SHADOW_PERSIST_SNAPSHOT=0.
PERSIST_SNAPSHOT = os.getenv("RORT_SHADOW_PERSIST_SNAPSHOT", "1").strip().lower() in (
    "1", "true", "yes", "on")
# Step E: after the shadow writes new trades, the strategy's kpis/equity_curve/Health
# would otherwise lag (the UI hydrates fresh TRADES from the table, but derived metrics
# come from the persisted `kpis`/`equity_curve_data` columns). Recompute them on a
# per-slot debounce (mirrors data_worker_engine's streaming KPI loop) so the whole view
# stays fresh. Default ON; gate off with RORT_SHADOW_RECOMPUTE_KPIS=0.
RECOMPUTE_KPIS = os.getenv("RORT_SHADOW_RECOMPUTE_KPIS", "1").strip().lower() in (
    "1", "true", "yes", "on")
KPI_DEBOUNCE_S = int(os.getenv("RORT_SHADOW_KPI_DEBOUNCE_S", "300"))

# ── M-RS4 Fix 2 — anti-starvation (all default to today's behavior) ──────────────
# Fix 2a: bounded per-poll advance. Cap how far the resident engine advances each
# poll so a deep-gap strategy (e.g. a 9.8k-trade bootstrap) backfills in bounded
# chunks across cycles instead of stalling the single-threaded loop on one giant
# pass. Byte-identical to advancing to now-LAG in one shot (the engine is resident
# and feeds bars in order — proven by _shadow_manager_validate PATH D). 0 = unbounded
# (today's behavior). Steady-state (caught-up) strategies advance their small delta
# in one poll regardless.
MAX_ADVANCE_S = int(os.getenv("RORT_SHADOW_MAX_ADVANCE_S", "0"))
# Gap-skip: when a MAX_ADVANCE_S-capped advance finds NO new bars, move the cursor
# to the capped bound anyway so the engine can walk across bar-free regions (the
# overnight session gap is >1h wide, so a capped engine otherwise re-prepares the
# same empty window forever and never reaches the next session — 2026-07-02 stall).
# Safe: prepare_window delta-fetches the range from Polygon before reading, so an
# empty capped window means Polygon has no (session-eligible) bars there — and a
# capped cursor is already >MAX_ADVANCE_S behind now-LAG, far past REST finality.
# A true ingest outage backfilled later is the nightly full_recompute's job (§G).
GAP_SKIP = os.getenv("RORT_SHADOW_GAP_SKIP", "1").strip().lower() in (
    "1", "true", "yes", "on")
# Fix 2b: run the KPI + equity + Hi-Fi recompute on a SEPARATE worker thread instead
# of inline in poll() — the poll loop then only does advance + write (fast, O(new
# bars)); KPIs/Health/Hi-Fi lag on their own cadence without starving trade freshness.
# Default OFF = today's inline recompute.
# graduated to default-ON 2026-07-03 (flag-graduation; env remains the kill switch)
KPI_ASYNC = os.getenv("RORT_SHADOW_KPI_ASYNC", "1").strip().lower() in (
    "1", "true", "yes", "on")
# Empty-window probe (2026-07-03): before the expensive warmup-window prep, ask the
# source layer ONE indexed question — "does any bar exist in (cursor, bound]?" — and
# skip the whole advance when the answer is no. Gap-walking (nights/weekends/holidays)
# otherwise re-prepares the full warmup window per slot per tf-interval just to learn
# "still no bars": measured 34k window reads / ~889M rows returned in one idle night
# (pg_stat_statements, 07-03) — the top egress driver. Byte-identical: with zero
# source bars in the window, advance() would return [] and leave the cursor unmoved;
# the probe just reaches the same outcome without the read. Session-filtered windows
# (source bars exist but all filtered) still take the full prep + gap-skip path.
# Probe failures fall through to the normal advance (fail-open).
EMPTY_PROBE = os.getenv("RORT_SHADOW_EMPTY_PROBE", "1").strip().lower() in (
    "1", "true", "yes", "on")
# W2-0 shadow heartbeat (2026-07-06, Kevin's trust ask + Fleet_Divergence_Audit
# W2-0): after each poll, publish the slot's TRUE settled coverage boundary
# (last_processed_ts) to `shadow_heartbeats` so the Health page can distinguish
# "engine current, model quiet" from "engine stalled" — Last BT ages on quiet
# strategies and TBD is otherwise computed from stale recompute stamps, so
# trust evaporates. Debounced (~4 min/slot); write failures are swallowed
# (heartbeat must never break the poll). Default ON.
HEARTBEAT = os.getenv("RORT_SHADOW_HEARTBEAT", "1").strip().lower() in (
    "1", "true", "yes", "on")

# ── M-RS5a — RESIDENT DATA WINDOW (cost/scale; default OFF, per-slot kill switch) ──
# advance() today re-preps the full [cursor - WARMUP_BARS, until] window from the DB every
# poll (services.prepare_strategy_window_df) just to feed the resident engine 1-5 new rows —
# ~1000:1 waste for sub-minute TFs (Design_MRS5_Resident_Window §1; forced POLL_S=60 + a
# compute upgrade + spend-cap disabled). With this flag ON, an ELIGIBLE slot keeps a resident
# frame and each poll reads ONLY the delta source bars (frame_tail, bound], resampled through
# the SAME load_market_data path the full prep uses (byte-identical: _filter_session is per-bar
# and resample_to_timeframe is midnight-origin-bucketed — both window-independent), and feeds
# just the new rows. The resident engine already holds all built-in indicator state, so a PLAIN
# slot (no user-pack columns, no secondary TFs) needs only the new OHLCV — Phase 1 scope. User-
# pack / secondary-gated slots stay on the full-prep path (frame_eligible=False, decided from the
# bootstrap prep) until Phase 2 (incremental derived columns). Any error or a broken frame<->engine
# cursor invariant DROPS the frame and falls through to full-prep — self-healing, byte-safe. The
# nightly from-cold recompute remains the truth backstop. OFFLINE byte-identity gate
# (_shadow_manager_validate PATH) must be GREEN before any live arm. Kill switch:
# RORT_SHADOW_RESIDENT_FRAME=0 reverts a slot to full-prep instantly. Default OFF.
RESIDENT_FRAME = os.getenv("RORT_SHADOW_RESIDENT_FRAME", "0").strip().lower() in (
    "1", "true", "yes", "on")


def _source_bars_exist(symbol: str, tf_seconds: int, after_dt, until_dt) -> bool:
    """ONE indexed row: does the symbol's SOURCE layer (1Sec for sub-minute
    primaries, else 1Min) hold any bar in (after_dt, until_dt]? Raises on query
    failure — the caller treats that as 'unknown' and runs the full advance."""
    from db import get_admin_client
    layer = "1Sec" if tf_seconds < 60 else "1Min"
    r = (get_admin_client().table("bar_cache").select("ts")
         .eq("symbol", symbol).eq("timeframe", layer)
         .gt("ts", after_dt.isoformat()).lte("ts", until_dt.isoformat())
         .limit(1).execute())
    return bool(r.data)


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
        persist_snapshot=PERSIST_SNAPSHOT,  # warm the per-strategy snapshot cache for
                                            # a cheap next bootstrap (config cache only)
    )


class ResidentFrame:
    """M-RS5a — a per-slot RESIDENT prepared window: the engine-consumed columns kept in
    memory and extended by the per-poll delta instead of re-prepared from the DB each poll.

    Phase 1 (PLAIN slots): `df` carries OHLCV only — the only columns
    `shadow_engine.ResidentStrategyEngine._build_inputs` reads for a plain strategy (built-in
    indicators are recomputed by the resident engine itself; user-pack + secondary columns
    arrive in Phase 2). `append` adds the new resampled primary bars, `trim` bounds the frame
    to warmup depth, and `tail` (the last bar's ts) MUST equal the resident engine's
    `last_bar_ts` every poll — the lockstep invariant. A mismatch drops the frame and lets the
    full-prep path re-heal (byte-safe fail-back to today's behavior).
    """

    # OHLCV are the only Phase-1 engine-consumed columns (shadow_engine._build_inputs).
    OHLCV = ("open", "high", "low", "close", "volume")

    def __init__(self, df):
        cols = [c for c in self.OHLCV if c in df.columns]
        self.df = df[cols].copy()
        self.tail = self.df.index[-1] if len(self.df) else None

    def append(self, new_rows) -> None:
        import pandas as pd
        if new_rows is None or len(new_rows) == 0:
            return
        cols = [c for c in self.OHLCV if c in new_rows.columns]
        self.df = pd.concat([self.df, new_rows[cols]])
        # Defensive de-dup (keep latest) + sort; the caller only ever appends bars strictly
        # after `tail`, so this is a no-op in the normal path.
        self.df = self.df[~self.df.index.duplicated(keep="last")].sort_index()
        self.tail = self.df.index[-1]

    def trim(self, warmup_bars: int) -> None:
        """Bound memory to warmup depth (head-trim never touches the tail or the resident
        engine, which is already warm — the frame history is for Phase 2's incremental
        columns + the invariant, not for feeding a plain slot)."""
        if warmup_bars and len(self.df) > warmup_bars:
            self.df = self.df.iloc[-warmup_bars:]


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
        self.has_traded_since_kpi = False       # set on a settled write; drives KPI recompute
        self.last_kpi_at = None                 # monotonic stamp of last KPI recompute
        self.last_tick_at = None                # monotonic stamp of last poll (Fix 2a fairness)
        self.last_hb_at = None                  # monotonic stamp of last heartbeat (W2-0)
        self.frame = None                       # ResidentFrame | None (M-RS5a resident window)
        self.frame_eligible = None              # None=undecided; True=plain; False=full-prep only
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
        # A config edit (fingerprint change) invalidates the resident engine AND its frame
        # (M-RS5a): a new fingerprint may add/remove user-pack or secondary columns, so both
        # the warm engine and its resident window must be rebuilt from cold.
        if old_fp is not None and self.fingerprint != old_fp:
            self.engine = None
            self.last_processed_ts = None
            self.frame = None
            self.frame_eligible = None


class ResidentEngineManager:
    """Owns the resident engines for a symbol shard. The shadow-worker constructs one,
    calls discover() periodically, and poll()s each eligible slot on a cadence."""

    def __init__(self, shard_symbols=None, dry_run: bool = False, shard_sids=None):
        self.shard_symbols = set(shard_symbols) if shard_symbols else None
        # Precise per-strategy shard (RORT_SHADOW_SIDS): when set, scope to EXACTLY these
        # strategy ids (overrides the symbol shard). Lets us shadow a specific set — e.g.
        # only the extended-hours strategies — without arming whole symbols (which would
        # re-mix already-reconciled RTH lanes).
        self.shard_sids = set(int(s) for s in shard_sids) if shard_sids else None
        self.dry_run = dry_run
        self.slots: dict[int, EngineSlot] = {}
        self._enabled_gen = None
        # Fix 2b: sids that traded and need a (debounced) KPI/Hi-Fi recompute, drained
        # by the shadow-worker's KPI thread. De-duped set under a lock.
        self.kpi_queue: set[int] = set()
        self._kpi_lock = threading.Lock()

    def enqueue_kpi(self, sid: int) -> None:
        """Mark a slot as needing a KPI/Hi-Fi recompute (Fix 2b async path)."""
        with self._kpi_lock:
            self.kpi_queue.add(sid)

    def drain_kpi(self, max_n: int = 0) -> list:
        """Pop up to `max_n` queued sids (0 = all). Called by the KPI worker thread."""
        with self._kpi_lock:
            if not self.kpi_queue:
                return []
            if max_n and len(self.kpi_queue) > max_n:
                out = [self.kpi_queue.pop() for _ in range(max_n)]
            else:
                out = list(self.kpi_queue)
                self.kpi_queue.clear()
        return out

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
                sid = strat.get('id')
                if sid is None:
                    continue
                if self.shard_sids is not None:
                    if sid not in self.shard_sids:
                        continue
                elif self.shard_symbols and strat.get('symbol') not in self.shard_symbols:
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
        from data_loader import normalize_1min_secondary_gate

        from data_worker_engine import _UserContext

        # ── M-RS5a resident-frame fast path (flag-gated, eligible PLAIN slots only) ──
        # When the slot carries an active resident frame (built at bootstrap for an eligible
        # plain strategy), skip the full warmup-window re-prep entirely: read only the delta
        # source bars (frame.tail, until_dt], resample via the SAME load_market_data path the
        # full prep uses, and feed just the new rows to the already-warm engine. Returns the
        # new closed trades on success; returns None to signal "fall through to full-prep this
        # poll" (invariant broke → self-heal). Only for normal warm polls (since_override is
        # set only by bootstrap/restart, which must take the full-prep path).
        if (RESIDENT_FRAME and slot.frame is not None and slot.engine is not None
                and since_override is None):
            try:
                out = self._advance_resident_frame(slot, until_dt)
                if out is not None:
                    return out
            except Exception as e:  # noqa: BLE001
                logger.warning("[shadow] sid=%s resident-frame advance failed (%s) — "
                               "dropping frame, full-prep re-heal", slot.sid, e)
                slot.frame = None
            # fall through to full-prep (re-heals + rebuilds the frame below)

        since = since_override or slot.last_processed_ts
        if since is None:
            since = until_dt - timedelta(days=BOOTSTRAP_DAYS)
        if isinstance(since, pd.Timestamp):
            since = since.to_pydatetime()

        # 1-minute-secondary gate (RORT_ENFORCE_1MIN_GATE): relabel an overloaded
        # '1M-' gate → lowercase '1m-' on the strat the resident ENGINE sees, so
        # its confluence_set matches the '1m-' records the native 1Min secondary
        # (built by prepare_strategy_window_df) produces — not the sub-minute
        # primary's own '1M-' record, which over-counts. prepare_window normalizes
        # the DATA side independently; this normalizes the GATE side. Flag OFF /
        # 1Min-primary → unchanged (byte-identical). Shallow copy — the slot's
        # strat (backing the fingerprint) is never mutated.
        _eng_strat = slot.strat
        _norm_conf = normalize_1min_secondary_gate(
            slot.strat.get('confluence'), slot.timeframe)
        if _norm_conf is not slot.strat.get('confluence'):
            _eng_strat = {**slot.strat, 'confluence': _norm_conf}

        # The prep path loads PER-USER config (confluence groups, general packs); without
        # the slot's user context it queries user_id=None and silently drops them.
        with _UserContext(slot.uid):
            df = prepare_window(_eng_strat, slot.bt_model, since, until_dt,
                                slot.timeframe, slot.sec_tfs)
            if df is None or len(df) == 0:
                return []
            sec_tf_map = get_secondary_tf_map(df) or None

            if slot.engine is None:
                slot.engine = ResidentStrategyEngine(_eng_strat, self._gen_packs())
                # Bootstrap anchor: hydrate last_entry_written from the DB so the engine's
                # warm-window re-emissions (which re-derive the strategy's existing trades)
                # are filtered out instead of re-attempted one-by-one → flooding 409
                # conflicts and bogging the loop. The resident lane writes FORWARD (entry >
                # anchor); deep/interspersed historical gaps are the nightly full_recompute
                # backstop's job (per Plan §G). In-memory anchor resets on restart, so do
                # this on every (re)bootstrap.
                if slot.last_entry_written is None:
                    try:
                        from db import get_max_entry_ts_admin
                        slot.last_entry_written = get_max_entry_ts_admin(
                            slot.sid, slot.uid, data_source_filter='backtest_%')
                    except Exception as e:  # noqa: BLE001
                        logger.warning("[shadow] sid=%s anchor hydrate failed: %s",
                                       slot.sid, e)

            new = slot.engine.feed(df, sec_tf_map)
            # ── M-RS5a: decide frame-eligibility on this prep + (re)build the resident frame ──
            # PLAIN = the engine reads ONLY OHLCV from the df (no user-pack columns detected on
            # the first feed, no secondary-TF map). Those slots run the resident-frame fast path;
            # user-pack / secondary-gated slots stay on full-prep until Phase 2. This block runs
            # on the bootstrap prep AND on a full-prep re-heal poll (frame is None then) — a plain
            # slot's frame is rebuilt from the freshly-prepped window, anchored to the engine
            # cursor. Warm plain polls never reach here (they return via the fast path above).
            if RESIDENT_FRAME and slot.frame is None:
                eng = slot.engine
                # ── M-RS5a frame-eligibility ──
                # The resident engine computes user-pack indicators+triggers INTERNALLY via each
                # pack's incremental_class (unified_engine._update_indicators:1609), byte-identical
                # to the batch prep on settled bars (probe _frame_userpack_probe.py: 6 pack
                # archetypes GREEN incl. rvol_v2). Built-in triggers/interps are likewise recomputed
                # from OHLCV and their df values ignored ("Don't override built-in", process_bar).
                # So a slot is frame-eligible (OHLCV-only feed, NO derived columns needed) when:
                #   (1) no secondary-TF columns are needed (sec_tf_map empty), AND
                #   (2) every required `_user_pack_*` marker maps to a pack that ships an
                #       incremental_class (engine computes it live — no df needed), AND
                #   (3) no confluence GATE leg references a USER-PACK (non-builtin) interpreter —
                #       a gated user-pack interp STATE is NOT computed internally, so it would need
                #       the df confluence record (fleet has 0 of these today; guard regardless).
                # Secondary-gated slots stay on full-prep until the secondary-column path lands.
                # Any classification error → NOT eligible (fail-safe to full-prep).
                try:
                    from shadow_engine import _BUILTIN_INTERPS as _BI
                    import pack_registry as _pr
                    us = getattr(eng, 'engine', None)
                    _req = getattr(getattr(us, 'indicators', None), 'required', ()) or ()
                    _markers = [i[len('_user_pack_'):] for i in _req
                                if isinstance(i, str) and i.startswith('_user_pack_')]
                    _packs_ok = all(
                        (_pr.get_pack(s) is not None
                         and _pr.get_pack(s).incremental_class is not None) for s in _markers)
                    _nb = [ik for ik in getattr(getattr(us, 'trigger_eval', None),
                                                'required_interpreters', ()) or ()
                           if ik not in _BI]
                    _conf = slot.strat.get('confluence') or []
                    _gate_userpack = any(
                        isinstance(leg, str) and not leg.startswith('GEN-')
                        and any(nb in leg for nb in _nb) for leg in _conf)
                    # Use the AUTHORITATIVE classified secondary TFs (slot.sec_tfs), NOT
                    # get_secondary_tf_map(df): the latter matches any '__' and misreads
                    # '__'-PREFIXED trigger columns (e.g. '__utv4_bull_flip') as bogus
                    # secondaries, wrongly disqualifying no-secondary user-pack slots.
                    plain = (not slot.sec_tfs and _packs_ok and not _gate_userpack)
                except Exception:  # noqa: BLE001
                    plain = False   # unknown → fail-safe to full-prep
                slot.frame_eligible = plain
                if plain and eng.last_bar_ts is not None:
                    try:
                        f = ResidentFrame(df[df.index <= eng.last_bar_ts])
                        f.trim(WARMUP_BARS)
                        # Anchor the frame tail EXACTLY to the engine cursor, else stay full-prep.
                        if f.tail is not None and pd.Timestamp(f.tail) == pd.Timestamp(
                                eng.last_bar_ts):
                            slot.frame = f
                    except Exception as e:  # noqa: BLE001
                        logger.warning("[shadow] sid=%s frame bootstrap failed (%s) — "
                                       "full-prep", slot.sid, e)
                        slot.frame = None
        # High-water mark: never move the cursor backwards. A window can be
        # non-empty yet feed nothing new (warmup context only), leaving the
        # engine's last_bar_ts at an older bar than a cursor a gap-skip already
        # pushed past a bar-free region — clobbering it would oscillate forever.
        eng_ts = slot.engine.last_bar_ts
        cur = slot.last_processed_ts
        if eng_ts is not None and (
                cur is None or pd.Timestamp(eng_ts) > pd.Timestamp(cur)):
            slot.last_processed_ts = eng_ts
        return [t for t in (new or []) if t.get('exit_fill_ts')]

    def _advance_resident_frame(self, slot: EngineSlot,
                                until_dt: datetime) -> Optional[List[dict]]:
        """M-RS5a warm-poll fast path for a PLAIN slot with an active resident frame.

        Reads ONLY the delta primary bars (frame.tail, until_dt] via the trusted
        `load_market_data` path — the SAME source-layer read + `_filter_session` +
        `resample_to_timeframe` the full prep uses — so the new bars are byte-identical to a
        full-window re-prep for every complete bucket past the tail (session filter is per-bar,
        the resample is midnight-origin-bucketed; both are window-independent). Feeds the new
        rows to the resident engine (which already holds all built-in indicator state), extends
        + trims the frame, and asserts the frame<->engine cursor invariant.

        Returns the new closed trades on success, [] when no new settled bar has formed, or
        None to signal the caller to FALL THROUGH to full-prep (invariant broke → self-heal).
        """
        import pandas as pd
        from data_loader import load_market_data
        from data_worker_engine import _UserContext

        frame = slot.frame
        tail = frame.tail
        if tail is None:
            return None                     # no anchor — let full-prep rebuild the frame
        tail_ts = pd.Timestamp(tail)

        # Delta read: only (tail, until] of the PRIMARY TF, read-only cache (no Polygon /
        # no cache write). Starting the read at the tail's bucket means every NEW bucket has
        # its full source range present → byte-identical resample. _UserContext is harmless
        # here (bar_cache is admin-scoped) and kept for parity with the full-prep path.
        start = tail_ts.to_pydatetime()
        with _UserContext(slot.uid):
            delta = load_market_data(
                slot.symbol, start_date=start, end_date=until_dt,
                timeframe=slot.timeframe, feed="sip",
                session=slot.strat.get('trading_session', 'RTH'),
                no_backfill=READ_ONLY_BARS)
        if delta is None or len(delta) == 0:
            return []                       # cache holds no bars past the cursor yet

        # tz-align the delta index to the frame/engine cursor, then keep strictly-after-tail
        # bars (the re-formed tail bucket is dropped — the engine already consumed it).
        if delta.index.tz is None and tail_ts.tz is not None:
            delta.index = delta.index.tz_localize('UTC')
        elif delta.index.tz is not None and tail_ts.tz is None:
            tail_ts = tail_ts.tz_localize('UTC')
        new_rows = delta[delta.index > tail_ts]
        if len(new_rows) == 0:
            return []                       # only the re-formed tail bucket returned

        # Feed the new bars to the resident engine (plain slot → no secondary_tf_map), then
        # extend + bound the frame and assert lockstep with the engine cursor.
        new = slot.engine.feed(new_rows, None)
        frame.append(new_rows)
        frame.trim(WARMUP_BARS)
        eng_ts = slot.engine.last_bar_ts
        if eng_ts is None or frame.tail is None or \
                pd.Timestamp(frame.tail) != pd.Timestamp(eng_ts):
            # Frame<->engine cursor diverged (e.g. the engine skipped a bar it had already
            # seen) — DROP the frame; the caller re-heals via full-prep this poll.
            logger.warning("[shadow] sid=%s resident-frame INVARIANT broke "
                           "(frame_tail=%s engine_ts=%s) — dropping frame",
                           slot.sid, frame.tail, eng_ts)
            slot.frame = None
            return None

        # High-water cursor advance (never backwards — mirror the full-prep path).
        cur = slot.last_processed_ts
        if cur is None or pd.Timestamp(eng_ts) > pd.Timestamp(cur):
            slot.last_processed_ts = eng_ts
        return [t for t in (new or []) if t.get('exit_fill_ts')]

    def _filter_new(self, slot: EngineSlot, closed: List[dict]) -> List[dict]:
        """Closed trades whose entry is strictly newer than the DB anchor (idempotent
        with the (strategy_id, entry_fill_ts, exit_fill_ts) unique index regardless).

        Compare as Timestamps, NOT raw strings: the DB-hydrated anchor is ISO
        (`2026-07-01T00:00:00+00:00`, 'T' separator) while a trade's entry_fill_ts is a
        pandas-Timestamp str (`2026-07-01 20:46:20+00:00`, space separator). Space (0x20)
        < 'T' (0x54), so a naive string `>` filters out EVERY same-day trade → 0 writes
        on every re-arm once the anchor is DB-hydrated (b14657c). See feedback_gengate_
        live_string_ts for the same string-vs-datetime bug class."""
        if not closed:
            return []
        anchor = slot.last_entry_written
        if not anchor:
            return closed
        import pandas as pd
        anchor_ts = pd.Timestamp(anchor)
        return [t for t in closed
                if pd.Timestamp(t.get('entry_fill_ts')) > anchor_ts]

    def commit(self, slot: EngineSlot, closed: List[dict]) -> int:
        """Write settled closed trades. No-op in dry_run. Advances the entry anchor."""
        if not closed:
            return 0
        import pandas as pd
        # Advance the anchor by Timestamp comparison, not raw strings (mixed 'T'/space
        # separators otherwise mis-order — see _filter_new).
        if self.dry_run:
            mx = slot.last_entry_written
            mx_ts = pd.Timestamp(mx) if mx else None
            for t in closed:
                ek = t.get('entry_fill_ts')
                if ek:
                    ek_ts = pd.Timestamp(ek)
                    if mx_ts is None or ek_ts > mx_ts:
                        mx, mx_ts = str(ek), ek_ts
            slot.last_entry_written = mx
            return len(closed)

        from api.services.forward_test_service import _serialize_trades
        from db import insert_trade_admin

        records = _serialize_trades(pd.DataFrame(closed))
        inserted = 0
        mx = slot.last_entry_written
        mx_ts = pd.Timestamp(mx) if mx else None
        for rec in records:
            rec = dict(rec)
            rec['data_source'] = slot.data_source
            rec['provisional'] = False  # v1: settled-only
            if insert_trade_admin(slot.sid, slot.uid, rec) is not None:
                inserted += 1
            ek = rec.get('entry_fill_ts')
            if ek:
                ek_ts = pd.Timestamp(ek)
                if mx_ts is None or ek_ts > mx_ts:
                    mx, mx_ts = str(ek), ek_ts
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
        from data_loader import normalize_1min_secondary_gate

        # 1-minute-secondary gate (RORT_ENFORCE_1MIN_GATE): normalize the gate on
        # the strat the provisional-tail engine sees so the unsettled window
        # enforces the 1Min gate identically to the settled resident lane (no
        # over-count at the settled/unsettled boundary). Flag OFF / 1Min-primary →
        # unchanged (byte-identical). Shallow copy — slot.strat is never mutated.
        _eng_strat = slot.strat
        _norm_conf = normalize_1min_secondary_gate(
            slot.strat.get('confluence'), slot.timeframe)
        if _norm_conf is not slot.strat.get('confluence'):
            _eng_strat = {**slot.strat, 'confluence': _norm_conf}

        with _UserContext(slot.uid):
            df = prepare_window(_eng_strat, slot.bt_model, settled_until, now,
                                slot.timeframe, slot.sec_tfs)
            if df is None or len(df) < 2:
                return []
            sec_tf_map = get_secondary_tf_map(df) or None
            trades_df, _ = run_unified_backtest(
                df, _eng_strat, general_packs=self._gen_packs(),
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

    def maybe_recompute_kpis(self, slot: EngineSlot) -> bool:
        """Recompute the strategy's KPIs + equity curve from its (now-fresh) backtest_%
        lane, on a per-slot debounce — so the UI's derived metrics track the trades the
        shadow just wrote (Step E). Reuses data_worker_engine.recompute_kpis_for_strategy
        (KPIs + equity + Hi-Fi pass). No-op in dry_run / when nothing new traded / not due.
        Returns True if it recomputed."""
        import time as _time
        if not RECOMPUTE_KPIS or self.dry_run or not slot.has_traded_since_kpi:
            return False
        nowm = _time.monotonic()
        if slot.last_kpi_at is not None and nowm - slot.last_kpi_at < KPI_DEBOUNCE_S:
            return False
        slot.last_kpi_at = nowm
        try:
            from data_worker_engine import StrategyEngineState, classify_strategy, \
                recompute_kpis_for_strategy
            st = StrategyEngineState(slot.sid, slot.uid, slot.strat)
            classify_strategy(st)
            st.has_traded_since_kpi = True
            recompute_kpis_for_strategy(st)
            slot.has_traded_since_kpi = False
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("[shadow] sid=%s KPI recompute failed: %s", slot.sid, e)
            return False

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
            # Fix 2a: bound how far a WARM engine advances per poll so a behind slot
            # catches up in bounded chunks across cycles instead of stalling the loop.
            # Byte-identical (resident engine feeds bars in order — PATH D). Cold
            # bootstrap (last_processed None) is left unbounded — it needs its full warm
            # window; the KPI decouple (2b) keeps that poll light. 0 = unbounded (today).
            advance_until = settled_until
            capped = False
            if MAX_ADVANCE_S > 0 and slot.last_processed_ts is not None:
                bound = (pd.Timestamp(slot.last_processed_ts).to_pydatetime()
                         + timedelta(seconds=MAX_ADVANCE_S))
                if bound < advance_until:
                    advance_until, capped = bound, True
            prev_cursor = slot.last_processed_ts
            # Empty-window probe: cheap indexed existence check before the full
            # warmup-window prep. Only for WARM slots (cold bootstrap must prep).
            skip_advance = False
            if EMPTY_PROBE and slot.engine is not None and prev_cursor is not None:
                try:
                    cur_dt = pd.Timestamp(prev_cursor).to_pydatetime()
                    if not _source_bars_exist(slot.symbol, slot.tf_seconds,
                                              cur_dt, advance_until):
                        skip_advance = True
                except Exception as e:  # noqa: BLE001
                    logger.debug("[shadow] sid=%s empty-probe failed (%s) — "
                                 "full advance", slot.sid, e)
            closed = [] if skip_advance else self.advance(slot, advance_until)
            if (GAP_SKIP and capped and prev_cursor is not None
                    and slot.last_processed_ts == prev_cursor):
                # Capped window had no new bars → bar-free (or session-filtered,
                # which every compute path drops identically). Move the cursor to
                # the bound so the next poll's window starts past the dead zone;
                # the engine walks a gap in MAX_ADVANCE_S steps, one per poll.
                # Uncapped no-new-bars keeps today's behavior (window grows with
                # the clock; late REST bars inside LAG can still land).
                slot.last_processed_ts = advance_until
                logger.info("[shadow] sid=%s gap-skip: no bars in capped window, "
                            "cursor %s -> %s", slot.sid, prev_cursor, advance_until)
            new = self._filter_new(slot, closed)
            inserted = self.commit(slot, new)
            if inserted > 0:
                slot.has_traded_since_kpi = True

        prov = self.refresh_provisional(slot, now, settled_until)
        # Fix 2b: KPI/Hi-Fi recompute inline (today) OR hand off to the KPI worker thread
        # (async) so the poll loop only does advance + write and never stalls on the
        # reload-all recompute. maybe_recompute_kpis stays the debounced recompute body.
        if KPI_ASYNC:
            if slot.has_traded_since_kpi:
                self.enqueue_kpi(slot.sid)
            kpis = False
        else:
            kpis = self.maybe_recompute_kpis(slot)   # debounced; refreshes derived metrics
        import time as _time
        slot.last_tick_at = _time.monotonic()
        # Telemetry statuses (2026-07-06 watchdog): distinguish the silent paths so
        # a pass full of zeros is attributable from one log line. 'probe_skip' =
        # empty-window probe skipped the advance; 'ok_empty' = advance ran but fed
        # nothing new (prep empty / all bars ≤ engine cursor).
        if not new_settled_bar:
            status = 'no_new_bar'
        elif skip_advance:
            status = 'probe_skip'
        elif not closed and slot.last_processed_ts == prev_cursor:
            status = 'ok_empty'
        else:
            status = 'ok'
        # W2-0 shadow heartbeat: publish the engine's settled coverage boundary
        # so Health can tell "engine current, model quiet" from "engine stalled".
        # Debounced per slot; NEVER allowed to break the poll.
        if (HEARTBEAT and not self.dry_run and slot.last_processed_ts is not None
                and (slot.last_hb_at is None
                     or _time.monotonic() - slot.last_hb_at > 240)):
            try:
                from db import get_admin_client
                ct = slot.last_processed_ts
                # pd.Timestamp and datetime both expose isoformat(); anything
                # else (already a string) passes through via str().
                ct_iso = ct.isoformat() if hasattr(ct, 'isoformat') else str(ct)
                get_admin_client().table('shadow_heartbeats').upsert({
                    'strategy_id': slot.sid,
                    'current_through': ct_iso,
                    'updated_at': datetime.now(timezone.utc).isoformat(),
                }, on_conflict='strategy_id').execute()
                slot.last_hb_at = _time.monotonic()
            except Exception as e:  # noqa: BLE001
                logger.debug("[shadow] sid=%s heartbeat write failed: %s",
                             slot.sid, e)
        return {'status': status, 'inserted': inserted, 'provisional': max(prov, 0),
                'kpis_recomputed': kpis, 'last_bar_ts': slot.last_processed_ts}
