"""The dev consumer's engine adapter — board #165 Phase B, step 31.

Drives the REAL live engine classes (`StrategyMonitor` / `SymbolHub` /
`_ShadowIndicatorEngine`) from bars pulled out of `live_bars`. Never a
re-implementation: `src/replay_harness.py` established that shape offline and
this is the same call sequence, now-anchored instead of window-anchored.

Kept in its own module because it is the only part of the consumer that needs
pandas, the pack registry and `ralph_engine`. `dev_bars_consumer.py` stays
import-light so its rails can be exercised hermetically, without a market, a
database, or the engine's import cost.

WHAT IT MIRRORS FROM `replay_harness.py`, DELIBERATELY
-----------------------------------------------------
* **Decision-time values** — `live_bars.first_*` when present. Reading the healed
  columns would score the dev fleet against data live never had.
* **`bar_count` increments** like the live builder's. Feeding a constant makes
  `check_entry` return None even when the trigger fired and the gate passed —
  the trade silently never happens.
* **Canonical secondary state** — at each primary close every shadow is
  republished from the canonical resample of the engine's own 1Min series. That
  is today's armed live stack (`RORT_CANONICAL_FINE_TF_STATE` +
  `RORT_COARSE_SECONDARY_FROM_1MIN`), not an improvement on it.
* **Pack registry loaded before any monitor is built.** Skip it and packs load as
  0 and every GEN- gate silently fails — a lying zero
  (feedback_worker_pack_registry_init).
* **Warmup is now-anchored.** The no-look-ahead rule in `replay_harness.
  warmup_asof` exists because a replay's "now" is in the past. For a live
  consumer, now-anchored IS as-of-safe.

TWO LIMITATIONS THAT MUST REACH THE VERDICT, NOT A LOG FILE
-----------------------------------------------------------
1. **No intra-bar ticks.** Live checks stops and targets every second
   (`on_tick`), so its stop exits land MID-BAR. This consumer sees bar closes
   only — a 1Sec `live_bars` poll would be a different egress budget and a
   different design. Every decision row is therefore stamped
   `data.intrabar_ticks = false`: a stop/target exit here is deferred to the next
   1Min boundary and will read as an exit-TIMING divergence that is an artifact
   of the feed, not of logic. Phase B's verdict is "do the two fleets take the
   same trades", and §6 of the design already says it cannot speak to *when* —
   this is a second, independent reason that is true.
2. **Sub-minute secondaries cannot be fed.** The poll is 1Min+; a shadow engine
   below 60s would hold its warmup state forever and silently shut its gate.
   `assert_feedable()` refuses to start such a strategy rather than let it
   produce a confident zero.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dev_bars_consumer import (
    DevConsumerConfigError, MIN_SUPPORTED_TF_S, require_supported_timeframes,
)

logger = logging.getLogger("dev_consumer_engine")

#: Warmup depth, mirroring `replay_harness.warmup_asof` (7d fine, 60d coarse).
_WARMUP_DAYS_FINE = 7
_WARMUP_DAYS_COARSE = 60

_OHLC = ["open", "high", "low", "close", "volume"]


class EngineDecider:
    """One `SymbolHub` per symbol, every configured monitor attached to it.

    Hub-per-symbol rather than the harness's hub-per-strategy: that is how the
    live worker runs, and secondary shadow engines are shared per symbol there.
    """

    def __init__(self, sids: Sequence[int], admin_user_id: str,
                 *, flag_set: Optional[Dict[str, Any]] = None) -> None:
        self.sids = list(sids)
        self.admin = admin_user_id
        self.flag_set = dict(flag_set or {})
        self._strats: List[Dict[str, Any]] = []
        self._monitors: Dict[int, Any] = {}
        self._hubs: Dict[str, Any] = {}
        self._bar_n: Dict[int, int] = {}
        self._m1: Dict[str, Any] = {}      # symbol → rolling 1Min DataFrame
        self._loaded = False

    # ── boot ────────────────────────────────────────────────────────────────

    def load(self, *, now: Optional[datetime] = None) -> None:
        """Build every monitor and warm it as of `now`. Fails loud, never partly."""
        import pandas as pd  # noqa: F401 — imported for its side effect on ralph_engine
        import pack_registry
        from db import get_strategy_by_id_admin, load_general_packs_admin
        from general_packs import _parse_pack_list, get_enabled_general_packs
        from ralph_engine import StrategyMonitor, SymbolHub

        now = now or datetime.now(timezone.utc)

        if not pack_registry.get_registered_packs():
            pack_registry.scan_and_load_all()
        if not pack_registry.get_registered_packs():
            raise DevConsumerConfigError(
                "pack registry empty — refusing to start. Every gate would "
                "silently fail and the fleet would report a confident zero "
                "(feedback_worker_pack_registry_init).")

        gen_packs = get_enabled_general_packs(
            _parse_pack_list(load_general_packs_admin(self.admin)))

        for sid in self.sids:
            strat = get_strategy_by_id_admin(sid, self.admin)
            if not strat:
                raise DevConsumerConfigError(
                    f"strategy {sid} not found. The sid list is explicit "
                    f"configuration; a missing one is a mistake, not a strategy "
                    f"to skip quietly.")
            monitor = StrategyMonitor(strat, None, general_packs=gen_packs)
            symbol = strat["symbol"]
            hub = self._hubs.get(symbol)
            if hub is None:
                hub = self._hubs[symbol] = SymbolHub(symbol)
            hub.add_monitor(monitor)
            self._monitors[sid] = monitor
            self._strats.append({
                "strategy_id": sid,
                "user_id": strat.get("user_id"),
                "strategy_name": strat.get("name"),
                "symbol": symbol,
                "timeframe": strat.get("timeframe"),
                "timeframe_seconds": int(monitor.tf_seconds),
                "session": monitor.session,
            })

        require_supported_timeframes(self._strats)
        for hub in self._hubs.values():
            hub.finalize_shadow_engines()
        self.assert_feedable()

        for sid, monitor in self._monitors.items():
            self._warm(sid, monitor, now)
        for symbol in self._hubs:
            self._reload_m1(symbol, now)
        self._seed_shadows(now)
        self._loaded = True
        logger.warning("[dev-consumer] engine loaded: %d strategies, %d symbols",
                       len(self._strats), len(self._hubs))

    def assert_feedable(self) -> None:
        """Refuse a strategy whose gate needs a bar this consumer cannot supply.

        The poll is 1Min and coarser. A sub-minute shadow engine would never
        advance past its warmup, its ribbon would freeze, and the gate would sit
        shut all session — reported as a real zero. That is the lying-zero class
        `replay_harness`'s UNRES classification exists to refuse, and it is
        refused here at boot instead of discovered in a verdict.
        """
        require_supported_timeframes(self._strats)
        offenders: List[Tuple[str, int]] = []
        for symbol, hub in self._hubs.items():
            for (sec_tf, _sess) in getattr(hub, "_shadow_engines", {}):
                if int(sec_tf) < MIN_SUPPORTED_TF_S:
                    offenders.append((symbol, int(sec_tf)))
        if offenders:
            raise DevConsumerConfigError(
                f"refusing to start: {sorted(set(offenders))} require sub-minute "
                f"secondary bars, which a 1Min+ live_bars poll cannot supply. The "
                f"shadow would hold its warmup state all session and the gate "
                f"would report a confident zero. Drop those strategies from "
                f"RORT_DEV_CONSUMER_SIDS or wait for Phase C.")

    # ── warmup / state ──────────────────────────────────────────────────────

    def _warm(self, sid: int, monitor: Any, now: datetime) -> None:
        df = self._history(monitor.symbol, int(monitor.tf_seconds),
                           monitor.session, now)
        if df is None or not len(df):
            raise DevConsumerConfigError(
                f"strategy {sid}: warmup returned no bars. Starting on an empty "
                f"warmup produces a fleet that cannot fire and reads as "
                f"agreement (feedback_indicator_warmup).")
        monitor.warmup(df)
        self._bar_n[sid] = len(df)

    def _history(self, symbol: str, tf_s: int, session: str, end: datetime,
                 days: Optional[int] = None):
        """Bars ENDING at `end`. Mirrors `replay_harness.warmup_asof`."""
        import pandas as pd
        from data_loader import load_market_data, resample_to_timeframe
        from ralph_engine import SECONDS_TO_TIMEFRAME

        days = days or (_WARMUP_DAYS_FINE if tf_s < 3600 else _WARMUP_DAYS_COARSE)
        if tf_s <= 60:
            df = load_market_data(symbol, start_date=end - timedelta(days=days),
                                  end_date=end,
                                  timeframe=SECONDS_TO_TIMEFRAME.get(tf_s, "1Min"),
                                  feed="sip", session=session)
            return df[_OHLC] if df is not None and len(df) else pd.DataFrame()
        df1 = load_market_data(symbol, start_date=end - timedelta(days=days),
                               end_date=end, timeframe="1Min", feed="sip",
                               session=session)
        if df1 is None or not len(df1):
            return pd.DataFrame()
        out = resample_to_timeframe(df1[_OHLC].copy(), SECONDS_TO_TIMEFRAME[tf_s])
        return out[out.index + pd.Timedelta(seconds=tf_s) <= pd.Timestamp(end)]

    def _reload_m1(self, symbol: str, now: datetime) -> None:
        """The rolling 1Min series every shadow's canonical rebuild reads from.

        Loaded ONCE per (re)seed and then extended in memory by consumed bars — a
        per-close DB reload would be both glacial and a load risk on the shared
        production database (feedback_local_analysis_starves_live_worker).
        """
        session = next((s["session"] for s in self._strats
                        if s["symbol"] == symbol), "RTH")
        deep = any(int(tf) >= 3600
                   for (tf, _s) in getattr(self._hubs[symbol], "_shadow_engines", {}))
        self._m1[symbol] = self._history(
            symbol, 60, session, now,
            days=_WARMUP_DAYS_COARSE if deep else _WARMUP_DAYS_FINE)

    def _seed_shadows(self, now: datetime) -> None:
        for symbol, hub in self._hubs.items():
            for (sec_tf, sec_sess), shadow in list(
                    getattr(hub, "_shadow_engines", {}).items()):
                df = self._history(symbol, int(sec_tf), sec_sess, now)
                if df is not None and len(df):
                    shadow.warmup(df)
                    hub.seed_mtf_from_shadow(sec_tf, sec_sess)
                else:
                    logger.warning(
                        "[dev-consumer] %s shadow tf=%ss sess=%s seeded EMPTY — "
                        "its ribbon cannot resolve and any gate needing it will "
                        "read as closed", symbol, sec_tf, sec_sess)

    def _republish_shadows(self, symbol: str, close_ts) -> None:
        """Canonical secondary state at a primary close (the armed live stack)."""
        import pandas as pd
        from data_loader import resample_to_timeframe
        from ralph_engine import SECONDS_TO_TIMEFRAME

        hub = self._hubs[symbol]
        m1 = self._m1.get(symbol)
        if m1 is None or not len(m1):
            return
        src = m1[m1.index + pd.Timedelta(minutes=1) <= close_ts]
        if not len(src):
            return
        for (sec_tf, sec_sess), shadow in list(
                getattr(hub, "_shadow_engines", {}).items()):
            tf = int(sec_tf)
            if tf < MIN_SUPPORTED_TF_S:
                continue          # refused at boot; belt-and-braces
            out = resample_to_timeframe(src.copy(), SECONDS_TO_TIMEFRAME[tf])
            out = out[out.index + pd.Timedelta(seconds=tf) <= close_ts]
            if len(out):
                hub._publish_mtf(  # noqa: SLF001 — same call replay_harness makes
                    (sec_tf, sec_sess), shadow.recompute_confluence(out),
                    closed_bar_start_ts=out.index[-1])

    # ── the consumer's interface ────────────────────────────────────────────

    def strategies(self) -> List[Dict[str, Any]]:
        return list(self._strats)

    def decide_all(self, strats: Sequence[Dict[str, Any]],
                   bar: Dict[str, Any]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Close `bar` on every strategy at this `(symbol, timeframe)`.

        ORDERING INVARIANT (test_ralph_gap_heal Bug-B lock): the PRIMARY monitor
        pipeline runs BEFORE the secondary republish, so a closing primary gates
        on the PREVIOUS secondary bar. Reversing it reintroduces look-ahead.
        """
        import pandas as pd

        if not strats:
            return []
        symbol = strats[0]["symbol"]
        tf = int(strats[0]["timeframe_seconds"])
        hub = self._hubs[symbol]
        bar_start = pd.Timestamp(bar["timestamp"])
        close_ts = bar_start + pd.Timedelta(seconds=tf)

        out: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for strat in strats:
            sid = strat["strategy_id"]
            monitor = self._monitors[sid]
            self._bar_n[sid] = self._bar_n.get(sid, 0) + 1
            signals, _ = monitor.on_bar_close(
                bar, self._bar_n[sid],
                mtf_confluence=hub._mtf_confluence,            # noqa: SLF001
                mtf_prev=hub._mtf_confluence_prev,             # noqa: SLF001
                mtf_effective_from=hub._mtf_confluence_effective_from)  # noqa: SLF001
            for sig in signals or []:
                kind = (sig.get("type") or sig.get("side") or "").lower()
                side = "entry" if "entry" in kind else ("exit" if "exit" in kind
                                                        else None)
                if side is None:
                    logger.warning("[dev-consumer] sid=%s unrecognised signal %r",
                                   sid, sig)
                    continue
                fill = close_ts.to_pydatetime()
                out.append((strat, {
                    **sig,
                    "side": side,
                    "trigger_ts": fill,
                    "fill_ts": fill,
                    # Stamped on every row: this fleet never saw an intra-bar
                    # tick, so a stop/target exit here is deferred to the bar
                    # boundary. See the module docstring, limitation 1.
                    "intrabar_ticks": False,
                }))

        # 1Min primaries extend the canonical series every shadow rebuild reads.
        if tf == 60:
            m1 = self._m1.get(symbol)
            row = pd.DataFrame(
                [{c: float(bar[c]) for c in _OHLC}], index=[bar_start])
            self._m1[symbol] = (
                pd.concat([m1, row]) if m1 is not None and len(m1) else row)
            self._m1[symbol] = self._m1[symbol][
                ~self._m1[symbol].index.duplicated(keep="last")].sort_index()
        self._republish_shadows(symbol, close_ts)
        return out

    def reseed(self, key: Tuple[str, int], *, at: datetime) -> None:
        """Re-warm as of `at` after the lag ceiling tripped (design §2.5).

        Post-reseed state is NOT the state live holds; it converges over the
        first minutes. Every skipped bar already has a `bar_skipped` row, so the
        contamination is counted in the verdict rather than inferred.
        """
        symbol, tf = key
        logger.warning("[dev-consumer] re-seeding %s/%ss as of %s", symbol, tf, at)
        for strat in self._strats:
            if (strat["symbol"], strat["timeframe_seconds"]) != key:
                continue
            sid = strat["strategy_id"]
            self._warm(sid, self._monitors[sid], at)
        self._reload_m1(symbol, at)
        self._seed_shadows(at)


def armed_flag_set() -> Dict[str, str]:
    """The `RORT_*` stack this process is running, for `dev_alerts.flag_set`.

    Phase B runs a NULL DELTA against live deliberately (design §4.3): with the
    flag stacks identical, a divergence is attributable to the feed path and live
    process state rather than to configuration. Recording the stack per row is
    what makes that claim checkable after the fact instead of remembered.
    """
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith("RORT_")}
