"""The dev-environment `live_bars` consumer — board #165 Phase B, step 31.

Builds what `docs/_active/Design_Live_Bars_Consumer.md` specified and Kevin
stamped (chain step 30). Read that document first; this module implements it and
does not restate its reasoning, only its rules.

WHAT IT IS
----------
A second engine fleet, running in the **dev** Railway environment, that learns a
bar landed by polling `live_bars` and drives the REAL engine classes
(`StrategyMonitor` / `SymbolHub`, never a re-implementation) as bars arrive. It
writes its decisions to `dev_alerts` and to nothing else, ever.

The offline batch version of this is already proven: `src/replay_harness.py`
matched live to the exact second on 66 of 72 decisions (91.7%) on the 1Min slice
of 2026-07-31. What had never been built — and is built here — is the *live*
poll-and-decide loop.

THE SCOPE LIMIT, BEFORE ANYTHING ELSE (design §1)
-------------------------------------------------
A DB-fed fleet is structurally behind a socket-fed one: **median +3.2s, p90
+4.1s, and the deciding bar reached the database AFTER live had already decided
in 1,317 of 1,317 measured cases.** That is 5% of a 1Min bar and 32% of a 10Sec
bar. At 32% the consumer is not "the same decision, later" — it is evaluating a
*different bar boundary*, which is exactly what entry/exit gating keys on.

**1Min and coarser only. 10Sec/15Sec are out of scope until Phase C — including
every 15Sec live-money strategy.** `_require_supported_timeframes()` refuses to
boot rather than letting that limit erode quietly.

THE THREE RAILS (design §4.4) — each has a test that fails without it
--------------------------------------------------------------------
1. **It refuses to boot in production.** This is a worker process, so #286's
   HTTP middleware never sees it; the inverse assertion is made explicitly at
   boot. `RORT_ENV_ROLE` unset means production, so a consumer accidentally
   deployed there dies on startup instead of running.
   → `assert_dev_environment()`

2. **A dev row cannot land in `alerts`.** `DevAlertsWriter` is the module's only
   write surface, it is pinned to `dev_alerts`, and it re-asserts every DB-level
   constraint in Python — table, `dev_` tag, closed outcome vocabulary,
   decision⇔side, and the deliberately-absent columns — so the rail fires in a
   test, not only in production. This module never imports or calls
   `save_alert_db` / `save_alert_admin` and never names `alerts` as a table;
   `test_dev_bars_consumer_165.py` parses this source and asserts so.

3. **It never re-processes a bar.** `ForwardOnlyCursor` admits a bar only if its
   `bar_start` is strictly newer than the last consumed one for that
   `(symbol, timeframe)`. The GRACE overlap re-reads consumed rows on **every**
   poll by design, and `apply_rest_correction` upserts already-consumed rows —
   so without this rail the engine would double-apply bars to shadow state,
   which is the corruption `replay_harness.py:408-414` cost hours to learn.

WHY A ROW IS NOT WRITTEN FOR EVERY BAR CLOSE
--------------------------------------------
`outcome='decision'` means *the consumer evaluated the bar and produced a
signal*, and the DB CHECK `dev_alerts_decision_has_side` enforces that a
decision row carries a side. A bar that arrives and produces no signal writes
**no row** — the same shape as the `alerts` lane it is compared against, which
is what keeps the comparison symmetric.

What IS recorded positively is a bar the consumer could not evaluate:
`bar_absent`, `bar_late`, `bar_skipped`, `consumer_restart`. That is the point of
the `outcome` column and the one thing `pilot_alerts` could not express — absence
of a bar is otherwise byte-identical to "evaluated and declined to fire", and
inflates apparent logic divergence by the 2.8% 1Min blind-spot rate
(feedback_no_silent_defaults).

STRUCTURE — pure core, injected edges
-------------------------------------
`DevBarsConsumer` depends on three small interfaces (`bar_source`, `decider`,
`writer`). The real implementations are `LiveBarsPoller`, `EngineDecider` and
`DevAlertsWriter`; the test drives the identical control flow with recording
stubs, so the rails are exercised without a database, a market, or the engine's
import cost.

FLAGS — all runtime-read, all default-OFF or required (board #165 rails)
-----------------------------------------------------------------------
  RORT_DEV_CONSUMER_ENABLED       master switch, default OFF
  RORT_DEV_CONSUMER_DRY_RUN       default ON — computes, writes nothing
  RORT_DEV_CONSUMER_SIDS          REQUIRED, no default (design §4.5)
  RORT_DEV_CONSUMER_FLEET_TAG     default 'dev_live_bars_1min'
  RORT_DEV_CONSUMER_POLL_S        default 2      (design §2.3)
  RORT_DEV_CONSUMER_GRACE_S       default 180    (design §2.4)
  RORT_DEV_CONSUMER_MAX_LAG_S     default 300    (design §2.5)
  RORT_DEV_CONSUMER_MAX_SKIP_ROWS default 20     — named cap, never silent

⛔ MERGE ORDERING: `dev_alerts` does not exist yet. Its migration
(`src/migrations/dev_alerts_table.sql`, authored at chain step 28) is AUTHORED,
NOT APPLIED. This code must not merge before Kevin authorizes it by name and it
is applied — with `RORT_DEV_CONSUMER_ENABLED` off nothing here runs, but the
ordering rule is the design's (§7) and is not negotiable.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import env_role

logger = logging.getLogger("dev_bars_consumer")


# ══════════════════════════════════════════════════════════════════════════════
# Constants — the parts of the design that are NOT tunable
# ══════════════════════════════════════════════════════════════════════════════

#: The one table this module may write. Not configurable, deliberately: a
#: configurable write target is a configurable way to write `alerts`.
DEV_ALERTS_TABLE = "dev_alerts"

#: `dev_alerts.fleet_tag` CHECK, mirrored in Python so it fails in a test too.
FLEET_TAG_PREFIX = "dev_"

#: The closed outcome vocabulary (migration CONSTRAINT dev_alerts_outcome_known).
OUTCOME_DECISION = "decision"
OUTCOME_BAR_ABSENT = "bar_absent"
OUTCOME_BAR_LATE = "bar_late"
OUTCOME_BAR_SKIPPED = "bar_skipped"
OUTCOME_CONSUMER_RESTART = "consumer_restart"
OUTCOMES = frozenset({
    OUTCOME_DECISION, OUTCOME_BAR_ABSENT, OUTCOME_BAR_LATE,
    OUTCOME_BAR_SKIPPED, OUTCOME_CONSUMER_RESTART,
})

#: Columns `dev_alerts` deliberately does NOT have (migration, "DELIBERATELY
#: ABSENT"). PostgREST would reject these with PGRST204; rejecting them here too
#: means the rail fires in a hermetic test instead of only in production.
FORBIDDEN_COLUMNS = frozenset({
    "acknowledged", "webhook_sent", "webhook_deliveries", "verification_status",
    "verification_close_delta", "verification_completed_at",
    "would_fire_post_correction", "source", "orphaned_at", "orphaned_reason",
})

#: `live_bars.source` values the live engine actually consumed. A LIST, never
#: `eq.ws_agg`: `apply_rest_correction` upserts the row and OVERWRITES `source`
#: (`data_loader.py:1399-1412`), so filtering to `ws_agg` alone would silently
#: drop every corrected bar — feedback_lane_filter_symmetry, on the input side.
#: `rest_backfill` stays out: cosmetic cache patching live never consumed.
LIVE_BARS_SOURCES: Tuple[str, ...] = (
    "ws", "ws_agg", "rest_correction", "rest_insert", "warmup_seed",
)

#: Design §1: the consumer cannot faithfully evaluate a bar shorter than this.
MIN_SUPPORTED_TF_S = 60

_OHLC = ("open", "high", "low", "close", "volume")

#: Files hashed into `code_fingerprint` (shadow_worker.py:52-70, the 2026-07-07
#: lesson: "which code produced this row" must be answerable FROM THE ROW).
_FINGERPRINT_FILES = ("dev_bars_consumer.py", "dev_consumer_engine.py",
                      "dev_consumer_worker.py")


class DevConsumerWriteBlocked(RuntimeError):
    """A write was attempted that would leave the dev lane. Never caught here."""


class DevConsumerConfigError(RuntimeError):
    """Configuration that would make the session meaningless. Fails at boot."""


# ══════════════════════════════════════════════════════════════════════════════
# Flags — read at call time, every time (never at import; see env_role.py's
# constraint 1: an import-time read cannot be flipped and cannot be tested)
# ══════════════════════════════════════════════════════════════════════════════

def _truthy(raw: Optional[str]) -> bool:
    return (raw or "").strip().lower() in ("1", "true", "yes", "on")


def _int_flag(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        # feedback_no_silent_defaults: a typo'd interval must not quietly become
        # the default and produce a session whose cadence nobody can reconstruct.
        raise DevConsumerConfigError(f"{name}={raw!r} is not an integer")


def consumer_enabled() -> bool:
    """Master switch. Default OFF — this ships inert."""
    return _truthy(os.getenv("RORT_DEV_CONSUMER_ENABLED"))


def dry_run() -> bool:
    """Default ON: compute everything, write nothing (shadow_worker mold)."""
    return not _truthy_off(os.getenv("RORT_DEV_CONSUMER_DRY_RUN"))


def _truthy_off(raw: Optional[str]) -> bool:
    """True when the operator explicitly turned dry-run OFF."""
    return (raw or "").strip().lower() in ("0", "false", "no", "off")


def fleet_tag() -> str:
    return (os.getenv("RORT_DEV_CONSUMER_FLEET_TAG") or "dev_live_bars_1min").strip()


def poll_interval_s() -> int:
    return _int_flag("RORT_DEV_CONSUMER_POLL_S", 2)


def grace_s() -> int:
    return _int_flag("RORT_DEV_CONSUMER_GRACE_S", 180)


def max_lag_s() -> int:
    return _int_flag("RORT_DEV_CONSUMER_MAX_LAG_S", 300)


def max_skip_rows() -> int:
    return _int_flag("RORT_DEV_CONSUMER_MAX_SKIP_ROWS", 20)


def configured_sids() -> List[int]:
    """REQUIRED configuration (design §4.5). No default.

    An empty default would silently produce a zero-decision session, and a
    zero-decision session reads as perfect agreement between the fleets.
    """
    raw = (os.getenv("RORT_DEV_CONSUMER_SIDS") or "").strip()
    if not raw:
        raise DevConsumerConfigError(
            "RORT_DEV_CONSUMER_SIDS is unset. The dev fleet's strategy list is "
            "REQUIRED configuration (design §4.5) — an empty default produces a "
            "zero-decision session that reads as agreement.")
    out: List[int] = []
    for part in raw.replace(" ", "").split(","):
        if not part:
            continue
        if not part.lstrip("-").isdigit():
            raise DevConsumerConfigError(
                f"RORT_DEV_CONSUMER_SIDS contains {part!r}, which is not a "
                f"strategy id.")
        out.append(int(part))
    if not out:
        raise DevConsumerConfigError("RORT_DEV_CONSUMER_SIDS parsed to no ids.")
    return out


def code_fingerprint() -> str:
    """sha256/12 over this service's sources, in `_FINGERPRINT_FILES` order.

    Reproducible from a shell:
        cat src/dev_bars_consumer.py src/dev_consumer_engine.py \
            src/dev_consumer_worker.py | sha256sum
    """
    h = hashlib.sha256()
    base = Path(__file__).resolve().parent
    for name in _FINGERPRINT_FILES:
        try:
            h.update((base / name).read_bytes())
        except Exception:  # noqa: BLE001 — a missing sibling must still be visible
            h.update(f"missing:{name}".encode())
    return h.hexdigest()[:12]


# ══════════════════════════════════════════════════════════════════════════════
# RAIL 1 — this process refuses to run in production (design §4.4.1)
# ══════════════════════════════════════════════════════════════════════════════

def assert_dev_environment() -> None:
    """Abort unless `RORT_ENV_ROLE` marks this as a guarded (non-production) env.

    #286 guards the API's fleet-wide HTTP routes. This is a worker: no request,
    no middleware, no guard — so the assertion has to be made here, and it is
    made in the INVERSE direction. `RORT_ENV_ROLE` unset means production
    (env_role.py), so a consumer accidentally deployed to the production
    environment dies on startup rather than running a second fleet beside the
    money path.
    """
    if not env_role.is_write_guarded():
        raise DevConsumerWriteBlocked(
            f"The dev live_bars consumer refuses to start: {env_role.ENV_VAR}="
            f"{env_role.env_role()!r} (production). This process is the dev "
            f"environment's second engine fleet (board #165 Phase B); running it "
            f"in production would put a second fleet beside the money path. "
            f"Set RORT_ENV_ROLE=dev, or do not deploy it here.")


# ══════════════════════════════════════════════════════════════════════════════
# RAIL 2 — the only write surface (design §4.1-§4.4.2)
# ══════════════════════════════════════════════════════════════════════════════

class DevAlertsWriter:
    """Writes `dev_alerts` rows. It is the module's ONLY write path.

    Every DB-level constraint from `src/migrations/dev_alerts_table.sql` is
    re-asserted here in Python. That is not belt-and-braces for its own sake: the
    table does not exist yet, the constraints therefore cannot fire in a test,
    and a rail that only fires in production is a rail nobody has seen work.

    `dry_run=True` (the default) records rows in `written` and sends nothing.
    """

    def __init__(self, client: Any, tag: str, *, fingerprint: str = "",
                 flag_set: Optional[Dict[str, Any]] = None,
                 dry: bool = True, table: Optional[str] = None) -> None:
        # The target table is pinned, and the pin is resolved at CALL time (the
        # env_role.py rule: an import-time binding cannot be tested). A
        # configurable target is a configurable way to write `alerts`; the
        # parameter exists only so a test can prove anything else is refused.
        table = DEV_ALERTS_TABLE if table is None else table
        if table != DEV_ALERTS_TABLE:
            raise DevConsumerWriteBlocked(
                f"DevAlertsWriter may only write {DEV_ALERTS_TABLE!r}; refused "
                f"{table!r}. The dev fleet never writes the live alert lane — "
                f"`alerts` has no lane axis and none of its 33 call sites is "
                f"lane-scoped, so a dev row there would join get_strategy_health "
                f"and the parity gate on day one (board #165 design §4.1).")
        self._client = client
        self._table = table
        self.fleet_tag = (tag or "").strip()
        self._assert_fleet_tag(self.fleet_tag)
        self.fingerprint = fingerprint
        self.flag_set = dict(flag_set or {})
        self.dry = bool(dry)
        self.written: List[Dict[str, Any]] = []
        self.counts: Dict[str, int] = {}

    # ── validation ──────────────────────────────────────────────────────────

    @staticmethod
    def _assert_fleet_tag(tag: str) -> None:
        """Mirror of CONSTRAINT dev_alerts_fleet_tag_is_dev."""
        if not tag.startswith(FLEET_TAG_PREFIX) or len(tag) <= len(FLEET_TAG_PREFIX):
            raise DevConsumerWriteBlocked(
                f"fleet_tag {tag!r} does not name a dev fleet. The tag cannot "
                f"lie: it must start with {FLEET_TAG_PREFIX!r} and carry a "
                f"suffix, so no row here can ever read as a live or backtest "
                f"lane (dev_alerts CHECK, board #165 design §4.3).")

    def validate(self, row: Dict[str, Any]) -> None:
        """Every rail, in one place, so removing one is a one-line diff a test
        can point at."""
        self._assert_fleet_tag(str(row.get("fleet_tag") or ""))

        outcome = row.get("outcome")
        if outcome not in OUTCOMES:
            raise DevConsumerWriteBlocked(
                f"outcome {outcome!r} is not in the closed vocabulary "
                f"{sorted(OUTCOMES)}. A new outcome is a schema change and a "
                f"design conversation, not a string invented at the call site.")

        # Mirror of CONSTRAINT dev_alerts_decision_has_side. A bookkeeping row
        # must never be countable as a trade by a reader that forgot to filter.
        has_side = row.get("side") is not None
        if outcome == OUTCOME_DECISION and not has_side:
            raise DevConsumerWriteBlocked(
                "a 'decision' row must carry a side (entry/exit).")
        if outcome != OUTCOME_DECISION and has_side:
            raise DevConsumerWriteBlocked(
                f"a {outcome!r} row must NOT carry a side — it records that no "
                f"decision was possible, not a decision.")

        for col in ("fleet_tag", "user_id", "strategy_id", "symbol",
                    "timeframe_seconds", "bar_start"):
            if row.get(col) is None:
                raise DevConsumerWriteBlocked(f"{col} is NOT NULL in dev_alerts.")

        stray = FORBIDDEN_COLUMNS & set(row)
        if stray:
            # This is the rail firing, not a bug to fix by adding the column:
            # those columns only mean something on the money path, and a dev row
            # must not even be SHAPED like a delivered live alert.
            raise DevConsumerWriteBlocked(
                f"{sorted(stray)} do not exist on dev_alerts and must not. "
                f"PostgREST would reject them (PGRST204); this refusal is the "
                f"same rail, fired earlier.")

    # ── the write ───────────────────────────────────────────────────────────

    def write(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(row)
        row.setdefault("fleet_tag", self.fleet_tag)
        row.setdefault("code_fingerprint", self.fingerprint)
        row.setdefault("flag_set", self.flag_set)
        row.setdefault("bar_source", "live_bars")
        self.validate(row)

        self.counts[row["outcome"]] = self.counts.get(row["outcome"], 0) + 1
        self.written.append(row)
        if self.dry:
            return row
        # The table name is the module constant, never a variable a caller
        # supplied — see __init__.
        self._client.table(DEV_ALERTS_TABLE).insert(_jsonable(row)).execute()
        return row


def _jsonable(row: Dict[str, Any]) -> Dict[str, Any]:
    """PostgREST wants ISO strings for timestamps."""
    out: Dict[str, Any] = {}
    for k, v in row.items():
        out[k] = v.isoformat() if isinstance(v, datetime) else v
    return out


# ══════════════════════════════════════════════════════════════════════════════
# RAIL 3 — forward-only (design §3.2 duplicate, §3.4 out-of-order)
# ══════════════════════════════════════════════════════════════════════════════

ADMIT_OK = "ok"
ADMIT_DUPLICATE = "duplicate"
ADMIT_LATE = "late"


class ForwardOnlyCursor:
    """Per `(symbol, timeframe_seconds)`, the newest `bar_start` consumed.

    Three verdicts, and only one of them reaches the engine:

      ok         strictly newer than everything consumed → feed it
      duplicate  exactly the bar we last consumed → the GRACE overlap re-reading
                 itself, or `apply_rest_correction` upserting a consumed row.
                 A corrected bar is NEVER re-decided: the decision-time value is
                 `first_*`, the bar is already in shadow state, and re-running it
                 would DOUBLE-APPLY that bar (replay_harness.py:408-414 —
                 "cost hours on 2026-07-14").
      late       older than the newest consumed bar. `rest_insert` and
                 `warmup_seed` land 30-124s after close, so this happens for
                 real. It is a RECORDED LOSS, never a replay: feeding it would
                 re-order the series. Live cannot rewind either, so a consumer
                 that did would be measuring something live can never do.
    """

    def __init__(self) -> None:
        self._last: Dict[Tuple[str, int], datetime] = {}

    def seed(self, key: Tuple[str, int], bar_start: datetime) -> None:
        self._last[key] = bar_start

    def last(self, key: Tuple[str, int]) -> Optional[datetime]:
        return self._last.get(key)

    def admit(self, key: Tuple[str, int], bar_start: datetime) -> str:
        prev = self._last.get(key)
        if prev is None or bar_start > prev:
            return ADMIT_OK
        return ADMIT_DUPLICATE if bar_start == prev else ADMIT_LATE

    def commit(self, key: Tuple[str, int], bar_start: datetime) -> None:
        """Record a bar as consumed. Never moves the cursor backwards."""
        prev = self._last.get(key)
        if prev is None or bar_start > prev:
            self._last[key] = bar_start


# ══════════════════════════════════════════════════════════════════════════════
# The poll (design §2.2) — keyed on the index that provably exists
# ══════════════════════════════════════════════════════════════════════════════

_POLL_COLUMNS = (
    "symbol,timeframe_seconds,bar_start,open,high,low,close,volume,"
    "first_open,first_high,first_low,first_close,first_volume,source,written_at"
)


class LiveBarsPoller:
    """One round trip per tick, for the whole dev fleet.

    `bar_start` is the cursor axis because it is the only indexed one:
    `live_bars` has NO index on `written_at`, so a `written_at > cursor` poll
    would be a sequential scan of a ~42M-row table 11,700×/day **against the
    shared production database during RTH** — DB CPU on the instance the live
    worker depends on (feedback_local_analysis_starves_live_worker). Arrival
    order is recovered by the GRACE overlap instead, which costs 3 extra rows
    per symbol per poll: ~16 rows/poll, ~94 MB/day, 0.19% of the measured
    49.58 GB/day (design §2.3).
    """

    def __init__(self, client: Any, symbols: Sequence[str],
                 timeframes: Sequence[int], *, grace: int = 180,
                 sources: Sequence[str] = LIVE_BARS_SOURCES) -> None:
        self._client = client
        self.symbols = sorted(set(symbols))
        self.timeframes = sorted(set(int(t) for t in timeframes))
        self.grace = int(grace)
        self.sources = list(sources)
        self.polls = 0
        self.rows_seen = 0

    def poll(self, floor: datetime) -> List[Dict[str, Any]]:
        """Rows with `bar_start >= floor - GRACE`, oldest first."""
        since = floor - timedelta(seconds=self.grace)
        q = (self._client.table("live_bars")
             .select(_POLL_COLUMNS)
             .in_("symbol", self.symbols)
             .in_("timeframe_seconds", self.timeframes)
             .in_("source", self.sources)
             .gte("bar_start", since.isoformat())
             .order("bar_start"))
        rows = q.execute().data or []
        self.polls += 1
        self.rows_seen += len(rows)
        return sorted(rows, key=lambda r: (str(r.get("bar_start")),
                                           str(r.get("symbol"))))


def decision_time_ohlc(row: Dict[str, Any]) -> Dict[str, float]:
    """The values the engine SAW: `first_*` when present, else the healed ones.

    Identical rule to `replay_harness.decision_time_bars(corrected=False)`. A
    consumer that read the healed columns would be scoring a fleet against data
    live never had, which flatters it.
    """
    out: Dict[str, float] = {}
    for col in _OHLC:
        first = row.get("first_" + col)
        out[col] = float(first if first is not None else row.get(col) or 0.0)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# The consumer
# ══════════════════════════════════════════════════════════════════════════════

def _as_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class DevBarsConsumer:
    """Poll → admit → decide → record. One tick per `poll_interval_s`.

    `decider` must expose:
        strategies() -> [{'strategy_id', 'user_id', 'symbol',
                          'timeframe_seconds', 'strategy_name', 'timeframe'}]
        decide(strategy, bar) -> [{'side': 'entry'|'exit', ...}]
    `EngineDecider` is the real one; the test injects a recording stub and
    exercises this identical control flow.
    """

    def __init__(self, *, bar_source: Any, decider: Any, writer: DevAlertsWriter,
                 grace: int = 180, max_lag: int = 300,
                 max_skip_rows: int = 20,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
                 ) -> None:
        self.bar_source = bar_source
        self.decider = decider
        self.writer = writer
        self.grace = int(grace)
        self.max_lag = int(max_lag)
        self.max_skip_rows = int(max_skip_rows)
        self.clock = clock
        self.cursor = ForwardOnlyCursor()
        self.stats: Dict[str, int] = {}

        # (symbol, tf) → [strategy, …]. Built once; the fleet is fixed for a
        # session by design (§4.5 — the sid list is required configuration).
        self._by_key: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        for strat in self.decider.strategies():
            key = (strat["symbol"], int(strat["timeframe_seconds"]))
            self._by_key.setdefault(key, []).append(strat)

    # ── bookkeeping helpers ─────────────────────────────────────────────────

    def _bump(self, name: str, n: int = 1) -> None:
        self.stats[name] = self.stats.get(name, 0) + n

    def _bookkeeping_row(self, strat: Dict[str, Any], outcome: str,
                         bar_start: datetime, *, now: datetime,
                         data: Optional[Dict[str, Any]] = None,
                         bar_read_at: Optional[datetime] = None,
                         bar_written_at: Optional[datetime] = None
                         ) -> Dict[str, Any]:
        """A row that records something the consumer could NOT decide.

        `side` is NULL on every one of these — that is what stops a bookkeeping
        row being counted as a trade by a reader that forgot to filter.
        """
        tf = int(strat["timeframe_seconds"])
        return {
            "outcome": outcome,
            "user_id": strat["user_id"],
            "strategy_id": strat["strategy_id"],
            "strategy_name": strat.get("strategy_name"),
            "symbol": strat["symbol"],
            "timeframe": strat.get("timeframe"),
            "timeframe_seconds": tf,
            "bar_start": bar_start,
            "bar_written_at": bar_written_at,
            "bar_read_at": bar_read_at or now,
            "consumer_lag_s": (now - (bar_start + timedelta(seconds=tf))
                               ).total_seconds(),
            "data": dict(data or {}),
            "timestamp": now,
        }

    def _record_for_all(self, key: Tuple[str, int], outcome: str,
                        bar_start: datetime, now: datetime,
                        data: Optional[Dict[str, Any]] = None,
                        **kw: Any) -> int:
        n = 0
        for strat in self._by_key.get(key, ()):
            self.writer.write(self._bookkeeping_row(
                strat, outcome, bar_start, now=now, data=data, **kw))
            n += 1
        self._bump(outcome, n)
        return n

    # ── boot (design §3.3) ──────────────────────────────────────────────────

    def seed_from(self, recovered: Dict[Tuple[str, int], Optional[datetime]],
                  *, now: Optional[datetime] = None) -> None:
        """Recover the cursor and record the restart hole.

        The cursor lives in `dev_alerts`, not in memory, so a restarted consumer
        cannot re-decide a bar it has already recorded and needs no cursor table.

        Stated limitation, recorded rather than hidden: post-restart state is NOT
        the state live holds. Live has continuous state; a re-warmed consumer
        converges over the first minutes — the same session-open convergence
        already characterized in Plan_Measurement_Trust.md. Restarts contaminate
        the comparison and must be COUNTED in the verdict, which is why each one
        gets a row instead of a log line.
        """
        now = now or self.clock()
        for key in sorted(self._by_key):
            prev = recovered.get(key)
            symbol, tf = key
            anchor = _floor_bar(now, tf)
            if prev is not None:
                self.cursor.seed(key, _as_dt(prev))
            self._record_for_all(
                key, OUTCOME_CONSUMER_RESTART, anchor, now,
                data={
                    "recovered_cursor": prev.isoformat() if prev else None,
                    "hole_s": (None if prev is None
                               else (now - _as_dt(prev)).total_seconds()),
                    "warmup": "reseeded as-of boot",
                })

    # ── one tick ────────────────────────────────────────────────────────────

    def tick(self, now: Optional[datetime] = None) -> Dict[str, int]:
        now = now or self.clock()
        before = dict(self.stats)

        floor = self._poll_floor(now)
        rows = self.bar_source.poll(floor)

        for row in rows:
            self._handle_row(row, now)

        # Absence sweep last: a bar that arrived in THIS poll must not be
        # declared absent by the same tick.
        self._sweep_absent(now)

        return {k: v - before.get(k, 0) for k, v in self.stats.items()
                if v - before.get(k, 0)}

    def _poll_floor(self, now: datetime) -> datetime:
        """Oldest cursor across the fleet — one query serves every key."""
        seen = [self.cursor.last(k) for k in self._by_key]
        seen = [s for s in seen if s is not None]
        if not seen:
            # Cold start: only bars from roughly this bar period onward. The
            # consumer never walks history — that is `replay_harness.py`'s job.
            tf = max((k[1] for k in self._by_key), default=MIN_SUPPORTED_TF_S)
            return _floor_bar(now, tf) - timedelta(seconds=tf)
        return min(seen)

    def _handle_row(self, row: Dict[str, Any], now: datetime) -> None:
        symbol = row.get("symbol")
        tf = int(row.get("timeframe_seconds") or 0)
        key = (symbol, tf)
        if key not in self._by_key:
            self._bump("row_not_ours")
            return

        bar_start = _as_dt(row["bar_start"])
        verdict = self.cursor.admit(key, bar_start)

        if verdict == ADMIT_DUPLICATE:
            # By design, every poll re-reads GRACE seconds of already-consumed
            # rows. Not an anomaly, not a row — just counted, so the revision
            # rate stays visible in the tick summary.
            self._bump("duplicate")
            return

        if verdict == ADMIT_LATE:
            self._record_for_all(
                key, OUTCOME_BAR_LATE, bar_start, now,
                bar_written_at=_opt_dt(row.get("written_at")),
                data={
                    "source": row.get("source"),
                    "last_consumed": _iso(self.cursor.last(key)),
                    "arrival_delta_s": _arrival_delta(row, bar_start, tf),
                    "replayed": False,   # strictly forward-only (design §3.4)
                })
            return

        # ── admitted ────────────────────────────────────────────────────────
        # TWO different lags, and conflating them is how a gap gets misread as a
        # stall. `behind` is the design's definition — how far the CONSUMER has
        # fallen behind the clock (§2.5, `now − (last_consumed + tf)`). `lag` is
        # how stale the bar in hand is, which is what a decision row records.
        prev = self.cursor.last(key)
        lag = (now - (bar_start + timedelta(seconds=tf))).total_seconds()
        behind = (lag if prev is None
                  else (now - (prev + timedelta(seconds=tf))).total_seconds())

        if behind > self.max_lag or lag > self.max_lag:
            # Design §2.5 item 3. A fleet ten minutes behind is deciding on a
            # market that no longer exists; continuing would contaminate the
            # comparison with rows that look like logic divergence.
            self._skip_to(key, bar_start, now,
                          reason="lag_ceiling", lag=max(behind, lag))
            return

        # A hole the consumer is about to step over. Forward-only means those
        # bars will never be fed, so the gap is written down NOW rather than
        # left as an absence of rows (feedback_no_silent_defaults). If one of
        # them turns up later it is separately recorded as `bar_late`, so both
        # facts survive and stay distinguishable.
        if prev is not None and bar_start > prev + timedelta(seconds=tf):
            self._fill_hole(key, prev, bar_start, now)

        bar = {
            "timestamp": bar_start.isoformat(),
            **decision_time_ohlc(row),
        }
        signals = self.decider.decide_all(self._by_key[key], bar) or []
        self.cursor.commit(key, bar_start)
        self._bump("bars_consumed")

        for strat, sig in signals:
            side = (sig.get("side") or "").strip().lower() or None
            if side not in ("entry", "exit"):
                # The engine's own vocabulary is entry/exit; anything else means
                # the adapter mis-read a signal, and a guessed side would land a
                # wrong trade in the comparison.
                raise DevConsumerWriteBlocked(
                    f"engine signal has no usable side: {sig!r}")
            self.writer.write({
                "outcome": OUTCOME_DECISION,
                "user_id": strat["user_id"],
                "strategy_id": strat["strategy_id"],
                "strategy_name": strat.get("strategy_name"),
                "symbol": strat["symbol"],
                "timeframe": strat.get("timeframe"),
                "timeframe_seconds": tf,
                "bar_start": bar_start,
                "type": sig.get("type"),
                "side": side,
                "event_type": sig.get("event_type"),
                "direction": sig.get("direction"),
                "exec_type": sig.get("exec_type"),
                "trigger_id": sig.get("trigger_id"),
                "trigger_ts": sig.get("trigger_ts") or bar_start + timedelta(seconds=tf),
                "fill_ts": sig.get("fill_ts") or bar_start + timedelta(seconds=tf),
                "price": sig.get("price"),
                "actual_price": sig.get("actual_price"),
                "bar_time": bar_start,
                "behavior": sig.get("behavior"),
                "live_model": sig.get("live_model"),
                "bar_written_at": _opt_dt(row.get("written_at")),
                "bar_read_at": now,
                "consumer_lag_s": lag,
                "data": {
                    "source": row.get("source"),
                    # A decision produced while the consumer is more than one bar
                    # behind is REAL, but its timing is meaningless. The row says
                    # so rather than leaving a later analyst to work it out.
                    "catchup": lag > tf,
                    "trigger": sig.get("trigger") or sig.get("reason"),
                },
                "timestamp": now,
            })
            self._bump(OUTCOME_DECISION)

    def _fill_hole(self, key: Tuple[str, int], prev: datetime,
                   bar_start: datetime, now: datetime) -> None:
        """`bar_absent` for every bar between `prev` and `bar_start`.

        Capped by `max_skip_rows` and NEVER silently: when the cap bites, every
        row carries `absent_bars_total` and the full span.
        """
        _symbol, tf = key
        step = timedelta(seconds=tf)
        total = max(0, int((bar_start - prev).total_seconds() // tf) - 1)
        if total <= 0:
            return
        n = min(total, self.max_skip_rows)
        for i in range(n):
            ts = bar_start - step * (n - i)
            self._record_for_all(
                key, OUTCOME_BAR_ABSENT, ts, now,
                data={"grace_s": self.grace,
                      "expected_close": _iso(ts + step),
                      "absent_bars_total": total,
                      "absent_from": _iso(prev + step),
                      "absent_to": _iso(bar_start - step),
                      "rows_capped_at": self.max_skip_rows if total > n else None,
                      "detected": "stepped_over"})
        if total > n:
            logger.warning(
                "[dev-consumer] %s/%ss: %d bars never arrived; wrote %d rows, "
                "capped by RORT_DEV_CONSUMER_MAX_SKIP_ROWS — the full span is on "
                "every row as absent_from/absent_to", _symbol, tf, total, n)

    def _skip_to(self, key: Tuple[str, int], bar_start: datetime,
                 now: datetime, *, reason: str, lag: float) -> None:
        """Record the gap, then re-seed the cursor at `bar_start`.

        NAMED CAP, never silent: at most `max_skip_rows` rows are written per
        gap. When the cap bites, every row carries `skipped_bars_total` and the
        span, so a truncated burst still describes the whole hole rather than
        reading as "we skipped 20 bars".
        """
        symbol, tf = key
        prev = self.cursor.last(key)
        if prev is None:
            total = 1
            first_missing = bar_start
        else:
            total = max(1, int((bar_start - prev).total_seconds() // tf))
            first_missing = prev + timedelta(seconds=tf)

        n = min(total, self.max_skip_rows)
        # Report the MOST RECENT n — the tail is what the next decisions depend on.
        start = bar_start - timedelta(seconds=tf * (n - 1))
        for i in range(n):
            ts = start + timedelta(seconds=tf * i)
            self._record_for_all(
                key, OUTCOME_BAR_SKIPPED, ts, now,
                data={
                    "reason": reason,
                    "lag_s": lag,
                    "max_lag_s": self.max_lag,
                    "skipped_bars_total": total,
                    "skipped_from": _iso(first_missing),
                    "skipped_to": _iso(bar_start),
                    "rows_capped_at": self.max_skip_rows if total > n else None,
                    "warmup": "reseeded as-of now",
                })
        if total > n:
            logger.warning(
                "[dev-consumer] %s/%ss: %d bars skipped (lag %.0fs > %ds); wrote "
                "%d rows, capped by RORT_DEV_CONSUMER_MAX_SKIP_ROWS — the full "
                "span is on every row as skipped_from/skipped_to",
                symbol, tf, total, lag, self.max_lag, n)
        self.decider.reseed(key, at=now)
        self.cursor.commit(key, bar_start)

    def _sweep_absent(self, now: datetime) -> None:
        """A bar whose GRACE window has closed with nothing in it (design §3.1).

        Measured at 2.8% of 1Min live decisions. The consumer does NOT synthesize
        the bar — it holds, exactly as live holds when the WS misses one; gap
        healing happens out of band in the live path and must not be
        re-implemented here.
        """
        for key in sorted(self._by_key):
            symbol, tf = key
            prev = self.cursor.last(key)
            if prev is None:
                continue
            guard = 0
            while True:
                guard += 1
                nxt = prev + timedelta(seconds=tf)
                deadline = nxt + timedelta(seconds=tf + self.grace)
                if now < deadline:
                    break
                if (now - (nxt + timedelta(seconds=tf))).total_seconds() > self.max_lag:
                    # Past the ceiling this is a gap, not an absence — §2.5 owns it.
                    self._skip_to(key, _floor_bar(now, tf), now,
                                  reason="absent_beyond_ceiling",
                                  lag=(now - (nxt + timedelta(seconds=tf))
                                       ).total_seconds())
                    break
                self._record_for_all(
                    key, OUTCOME_BAR_ABSENT, nxt, now,
                    data={"grace_s": self.grace,
                          "expected_close": _iso(nxt + timedelta(seconds=tf))})
                self.cursor.commit(key, nxt)
                prev = nxt
                if guard > self.max_skip_rows:
                    break


def _floor_bar(ts: datetime, tf: int) -> datetime:
    epoch = int(ts.timestamp())
    return datetime.fromtimestamp(epoch - (epoch % tf), tz=timezone.utc)


def _iso(value: Any) -> Optional[str]:
    return value.isoformat() if isinstance(value, datetime) else value


def _opt_dt(value: Any) -> Optional[datetime]:
    return _as_dt(value) if value else None


def _arrival_delta(row: Dict[str, Any], bar_start: datetime,
                   tf: int) -> Optional[float]:
    written = row.get("written_at")
    if not written:
        return None
    return (_as_dt(written) - (bar_start + timedelta(seconds=tf))).total_seconds()


# ══════════════════════════════════════════════════════════════════════════════
# Boot-time scope checks (design §1, §4.5)
# ══════════════════════════════════════════════════════════════════════════════

def require_supported_timeframes(strategies: Iterable[Dict[str, Any]]) -> None:
    """Refuse any strategy the consumer cannot faithfully evaluate.

    Sub-minute is not a caveat to annotate, it is a confound: at 32% of a 10Sec
    bar the consumer evaluates a DIFFERENT bar boundary than live did, and bar
    boundaries are what entry/exit gating keys on. `source='ws_agg'` also carries
    no bar shorter than 60s, and the deciding bar is missing from `live_bars` for
    13.9% of 15Sec decisions. Refusing is the only honest option.
    """
    bad = [(s.get("strategy_id"), int(s.get("timeframe_seconds") or 0))
           for s in strategies
           if int(s.get("timeframe_seconds") or 0) < MIN_SUPPORTED_TF_S]
    if bad:
        raise DevConsumerConfigError(
            f"refusing to start: {bad} have a primary timeframe below "
            f"{MIN_SUPPORTED_TF_S}s. Phase B is 1Min and coarser ONLY — a DB-fed "
            f"fleet runs ~3.2s behind, which is 32% of a 10Sec bar, so it would "
            f"be evaluating a different bar boundary than live. Sub-minute waits "
            f"for Phase C (board #165 design §1).")
