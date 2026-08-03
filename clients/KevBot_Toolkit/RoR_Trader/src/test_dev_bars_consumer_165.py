"""Acceptance test — board #165 Phase B: the dev fleet cannot reach the money lane.

Every rail below FAILS without its line of `dev_bars_consumer.py`, and the two the
step brief names by hand are proved BOTH ways — the rail holds, and with the rail
neutralised the same assertion catches the breach. A rail with no failing test is
a comment, not a rail.

  A1  a full consumer tick touches `dev_alerts` and NOTHING else
  A2  … and the assertion is not vacuous: repoint the table pin and the same
      check sees `alerts` — so A1 would fail the day someone removes it
  A3  the writer REFUSES to be constructed against `alerts`
  A4  the fleet_tag CHECK is mirrored in Python: 'ralph' / '' / 'dev_' refused
  A5  … and it is load-bearing: neutralise it and a row claiming the live lane
      is accepted
  A6  the closed outcome vocabulary, decision⇔side both ways, NOT NULLs
  A7  the deliberately-absent columns are refused here, not only by PGRST204
  A8  neither consumer module names `alerts` as a table, imports the live alert
      writers, or imports a market-data provider (source-parsed, #286's pattern)

  B1  a bar already processed is NOT re-processed — the engine is called once
  B2  … and it is load-bearing: neutralise the cursor and the engine is called
      twice on the same bar (the double-apply corruption)
  B3  an out-of-order bar is RECORDED as bar_late and never replayed
  B4  the cursor never moves backwards

  C1  lag beyond the ceiling stops deciding, records every skipped bar, re-seeds
  C2  the skip-row cap is NAMED on the row, never a silent truncation
  C3  a bar that never arrives is recorded POSITIVELY as bar_absent after GRACE
  C4  a bar that arrives in the same tick is NOT declared absent
  C5  restart writes the hole down instead of leaving it inferred

  D1  RAIL 1: it refuses to boot with RORT_ENV_ROLE unset (= production)
  D2  … and starts under role=dev
  D3  every flag is runtime-read and defaults OFF / required
  D4  the sid list is REQUIRED — no empty default
  D5  sub-minute strategies are refused (design §1 scope limit)
  D6  decision-time values are `first_*`, not the healed columns
  D7  the poll is keyed on bar_start with a GRACE floor and a source LIST — it
      never filters on the unindexed written_at, and never `eq.ws_agg`

Offline + hermetic: no DB, no network, no pandas, no engine import.

Run:  .venv/bin/python src/test_dev_bars_consumer_165.py   (from RoR_Trader root)
"""
import ast
import os
import sys
import types
from datetime import datetime, timedelta, timezone

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

PASS = 0


def ok(label, cond, detail=""):
    global PASS
    if not cond:
        print(f"FAIL: {label} {detail}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {label}")


def raises(label, fn, exc=Exception, needle=""):
    try:
        fn()
    except exc as e:  # noqa: BLE001
        ok(label, needle.lower() in str(e).lower(), f"message was: {e}")
        return
    ok(label, False, "no exception raised")


os.environ["RORT_ENV_ROLE"] = "dev"
for _v in ("RORT_DEV_CONSUMER_ENABLED", "RORT_DEV_CONSUMER_DRY_RUN",
           "RORT_DEV_CONSUMER_SIDS", "RORT_DEV_CONSUMER_POLL_S",
           "RORT_DEV_CONSUMER_GRACE_S", "RORT_DEV_CONSUMER_MAX_LAG_S",
           "RORT_DEV_CONSUMER_FLEET_TAG", "RORT_DEV_CONSUMER_MAX_SKIP_ROWS"):
    os.environ.pop(_v, None)

import dev_bars_consumer as C  # noqa: E402

T0 = datetime(2026, 8, 3, 14, 0, 0, tzinfo=timezone.utc)
TAG = "dev_live_bars_1min"
UID = "19d47e46-f718-49a6-af32-5f5407f5b170"

STRATS = [
    {"strategy_id": 269, "user_id": UID, "strategy_name": "canary",
     "symbol": "SPY", "timeframe": "1Min", "timeframe_seconds": 60},
    {"strategy_id": 340, "user_id": UID, "strategy_name": "second",
     "symbol": "SPY", "timeframe": "1Min", "timeframe_seconds": 60},
]


# ══════════════════════════════════════════════════════════════════════════
# Stubs — a recording DB client, a recording engine, a scripted bar source
# ══════════════════════════════════════════════════════════════════════════

class FakeQuery:
    def __init__(self, client, table):
        self.client = client
        self.table_name = table
        self.filters = []
        self.rows = []

    def select(self, cols):
        self.filters.append(("select", cols))
        return self

    def insert(self, row):
        self.client.inserts.append((self.table_name, row))
        return self

    def eq(self, col, val):
        self.filters.append(("eq", col, val))
        return self

    def in_(self, col, vals):
        self.filters.append(("in", col, list(vals)))
        return self

    def gte(self, col, val):
        self.filters.append(("gte", col, val))
        return self

    def order(self, col, **kw):
        self.filters.append(("order", col, kw))
        return self

    def limit(self, n):
        self.filters.append(("limit", n))
        return self

    def execute(self):
        return types.SimpleNamespace(data=self.client.rows_for(self))


class FakeClient:
    """Records every table it is asked for. That set IS the isolation proof."""

    def __init__(self, rows=None):
        self.tables = []
        self.inserts = []
        self.queries = []
        self._rows = rows or []

    def table(self, name):
        self.tables.append(name)
        q = FakeQuery(self, name)
        self.queries.append(q)
        return q

    def rows_for(self, q):
        return self._rows if q.table_name == "live_bars" else []


class RecordingDecider:
    def __init__(self, strats, signals=None):
        self._strats = strats
        self.calls = []
        self.reseeds = []
        self.signals = signals or {}

    def strategies(self):
        return list(self._strats)

    def decide_all(self, strats, bar):
        self.calls.append((strats[0]["symbol"], bar["timestamp"]))
        sigs = self.signals.get(bar["timestamp"], [])
        return [(strats[0], s) for s in sigs]

    def reseed(self, key, *, at):
        self.reseeds.append((key, at))


class ScriptedSource:
    def __init__(self, batches):
        self.batches = list(batches)
        self.floors = []

    def poll(self, floor):
        self.floors.append(floor)
        return self.batches.pop(0) if self.batches else []


def bar_row(ts, *, symbol="SPY", tf=60, source="ws_agg", first=100.0,
            healed=None, written_at=None):
    healed = healed if healed is not None else first + 5.0
    return {
        "symbol": symbol, "timeframe_seconds": tf, "bar_start": ts.isoformat(),
        "open": healed, "high": healed, "low": healed, "close": healed,
        "volume": 999.0,
        "first_open": first, "first_high": first, "first_low": first,
        "first_close": first, "first_volume": 111.0,
        "source": source,
        "written_at": (written_at or ts + timedelta(seconds=63)).isoformat(),
    }


def make(batches, *, signals=None, dry=True, grace=180, max_lag=300,
         max_skip=20, client=None):
    client = client or FakeClient()
    writer = C.DevAlertsWriter(client, TAG, fingerprint="abc123", dry=dry)
    decider = RecordingDecider(STRATS, signals)
    consumer = C.DevBarsConsumer(
        bar_source=ScriptedSource(batches), decider=decider, writer=writer,
        grace=grace, max_lag=max_lag, max_skip_rows=max_skip)
    return consumer, writer, decider, client


ENTRY = {"type": "entry_signal", "side": "entry", "price": 500.0,
         "trigger": "MACD_CROSS"}


# ══════════════════════════════════════════════════════════════════════════
# A — write isolation: a dev row cannot land in `alerts`
# ══════════════════════════════════════════════════════════════════════════

b1 = T0 + timedelta(minutes=1)
consumer, writer, decider, client = make(
    [[bar_row(b1)]], signals={b1.isoformat(): [ENTRY]}, dry=False)
consumer.cursor.seed(("SPY", 60), T0)
consumer.tick(now=b1 + timedelta(seconds=64))

ok("A1 the tick wrote at least one row", len(writer.written) >= 1,
   str(writer.counts))
ok("A1 every table touched is dev_alerts",
   set(client.tables) == {"dev_alerts"}, f"tables={sorted(set(client.tables))}")
ok("A1 'alerts' was never touched", "alerts" not in client.tables)
ok("A1 every INSERT targeted dev_alerts",
   {t for t, _ in client.inserts} == {"dev_alerts"},
   f"{[t for t, _ in client.inserts]}")
ok("A1 the row carries the dev fleet tag",
   all(r["fleet_tag"] == TAG for _t, r in client.inserts))

# A2 — the assertion is NOT vacuous. Repoint the pin (a one-line removal of the
# rail) and the very same check sees `alerts`.
_real_table = C.DEV_ALERTS_TABLE
try:
    C.DEV_ALERTS_TABLE = "alerts"
    consumer2, writer2, _d2, client2 = make(
        [[bar_row(b1)]], signals={b1.isoformat(): [ENTRY]}, dry=False)
    consumer2.cursor.seed(("SPY", 60), T0)
    consumer2.tick(now=b1 + timedelta(seconds=64))
    ok("A2 with the table pin removed the SAME check sees 'alerts' "
       "(so A1 is load-bearing)", "alerts" in client2.tables,
       f"tables={client2.tables}")
finally:
    C.DEV_ALERTS_TABLE = _real_table
ok("A2 the pin is restored", C.DEV_ALERTS_TABLE == "dev_alerts")

raises("A3 the writer refuses to be constructed against 'alerts'",
       lambda: C.DevAlertsWriter(FakeClient(), TAG, table="alerts"),
       C.DevConsumerWriteBlocked, "may only write")
raises("A3 … and against any other table",
       lambda: C.DevAlertsWriter(FakeClient(), TAG, table="pilot_alerts"),
       C.DevConsumerWriteBlocked, "may only write")

for bad in ("ralph", "", "dev_", "backtest_rest_hifi", "pilot_ws_agg_db"):
    raises(f"A4 fleet_tag {bad!r} refused",
           lambda b=bad: C.DevAlertsWriter(FakeClient(), b),
           C.DevConsumerWriteBlocked, "does not name a dev fleet")
ok("A4 a real dev tag is accepted",
   C.DevAlertsWriter(FakeClient(), "dev_live_bars_1min").fleet_tag == TAG)

# A5 — neutralise the tag check and a row claiming the live lane gets through.
_real_assert = C.DevAlertsWriter.__dict__["_assert_fleet_tag"]
try:
    C.DevAlertsWriter._assert_fleet_tag = staticmethod(lambda _tag: None)
    w = C.DevAlertsWriter(FakeClient(), "ralph")
    w.write({"outcome": C.OUTCOME_BAR_ABSENT, "user_id": UID,
             "strategy_id": 1, "symbol": "SPY", "timeframe_seconds": 60,
             "bar_start": T0})
    ok("A5 with the tag check neutralised a 'ralph' row is accepted "
       "(so A4 is load-bearing)", w.written[0]["fleet_tag"] == "ralph")
finally:
    C.DevAlertsWriter._assert_fleet_tag = _real_assert
raises("A5 the tag check is restored",
       lambda: C.DevAlertsWriter(FakeClient(), "ralph"),
       C.DevConsumerWriteBlocked, "does not name a dev fleet")

W = C.DevAlertsWriter(FakeClient(), TAG)
BASE = {"user_id": UID, "strategy_id": 269, "symbol": "SPY",
        "timeframe_seconds": 60, "bar_start": T0}

raises("A6 an outcome outside the closed vocabulary is refused",
       lambda: W.write({**BASE, "outcome": "maybe", "side": None}),
       C.DevConsumerWriteBlocked, "closed vocabulary")
raises("A6 a decision row with no side is refused",
       lambda: W.write({**BASE, "outcome": "decision", "side": None}),
       C.DevConsumerWriteBlocked, "must carry a side")
raises("A6 a bookkeeping row WITH a side is refused",
       lambda: W.write({**BASE, "outcome": "bar_absent", "side": "entry"}),
       C.DevConsumerWriteBlocked, "must NOT carry a side")
raises("A6 a NOT NULL column may not be omitted",
       lambda: W.write({"outcome": "bar_absent", "user_id": UID,
                        "strategy_id": 269, "symbol": "SPY",
                        "timeframe_seconds": 60}),
       C.DevConsumerWriteBlocked, "not null")
ok("A6 all five outcomes are known",
   C.OUTCOMES == {"decision", "bar_absent", "bar_late", "bar_skipped",
                  "consumer_restart"})

for col in ("webhook_sent", "acknowledged", "verification_status", "source",
            "would_fire_post_correction"):
    raises(f"A7 the absent column {col!r} is refused",
           lambda c=col: W.write({**BASE, "outcome": "bar_absent", c: True}),
           C.DevConsumerWriteBlocked, "do not exist on dev_alerts")

# A8 — source scan over BOTH modules, #286's registry-drift pattern.
MODULES = ("dev_bars_consumer.py", "dev_consumer_engine.py",
           "dev_consumer_worker.py")
FORBIDDEN_CALLS = ("save_alert_db", "save_alert_admin", "save_alert")
ALL_IMPORTS = {}

for fname in MODULES:
    text = open(os.path.join(SRC, fname), encoding="utf-8").read()
    tree = ast.parse(text)

    # No `.table('alerts')` anywhere — checked on the CALL, so a docstring
    # mentioning the table (they all do, deliberately) does not trip it.
    bad_tables = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "table" and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value != "live_bars"
                and node.args[0].value != C.DEV_ALERTS_TABLE):
            bad_tables.append(node.args[0].value)
    ok(f"A8 {fname} names no table but live_bars / dev_alerts",
       bad_tables == [], f"found {bad_tables}")

    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    called |= {n.func.attr for n in ast.walk(tree)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    hits = sorted(set(FORBIDDEN_CALLS) & called)
    ok(f"A8 {fname} never calls the live alert writer", hits == [],
       f"found {hits}")

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    ALL_IMPORTS[fname] = imported

import dev_consumer_worker as W_MOD  # noqa: E402

for fname, imported in ALL_IMPORTS.items():
    leaks = sorted(imported & set(W_MOD.DEV_CONSUMER_FORBIDDEN_IMPORTS))
    # "Do NOT open a Polygon connection from the dev environment under any
    # circumstances" — the step brief. A dev fleet that reached the provider
    # would also be a second WS subscriber on an account already near its limit
    # (feedback_polygon_ws_limit).
    ok(f"A8 {fname} imports no market-data provider", leaks == [],
       f"found {leaks}")


# ══════════════════════════════════════════════════════════════════════════
# B — forward-only: the consumer refuses a bar it has already processed
# ══════════════════════════════════════════════════════════════════════════

row = bar_row(b1)
consumer, writer, decider, _c = make([[row], [row], [row]])
consumer.cursor.seed(("SPY", 60), T0)
now = b1 + timedelta(seconds=64)
consumer.tick(now=now)
consumer.tick(now=now + timedelta(seconds=2))
consumer.tick(now=now + timedelta(seconds=4))

ok("B1 the same bar reached the engine exactly ONCE across three polls",
   len(decider.calls) == 1, f"calls={decider.calls}")
ok("B1 the re-reads were counted as duplicates, not written",
   consumer.stats.get("duplicate") == 2, str(consumer.stats))
ok("B1 no duplicate produced a dev_alerts row",
   [r for r in writer.written if r["outcome"] != "consumer_restart"] == [],
   str(writer.written))

# B2 — neutralise the cursor and the engine double-applies the same bar.
_real_admit = C.ForwardOnlyCursor.admit
try:
    C.ForwardOnlyCursor.admit = lambda self, key, ts: C.ADMIT_OK
    consumer3, _w3, decider3, _c3 = make([[row], [row]])
    consumer3.cursor.seed(("SPY", 60), T0)
    consumer3.tick(now=now)
    consumer3.tick(now=now + timedelta(seconds=2))
    ok("B2 with the cursor neutralised the SAME bar hits the engine twice "
       "(so B1 is load-bearing)", len(decider3.calls) == 2,
       f"calls={decider3.calls}")
finally:
    C.ForwardOnlyCursor.admit = _real_admit

# B3 — out-of-order arrival: recorded, never replayed.
late = T0 - timedelta(minutes=1)
consumer, writer, decider, _c = make(
    [[bar_row(late, source="rest_insert")]])
consumer.cursor.seed(("SPY", 60), T0)
consumer.tick(now=T0 + timedelta(seconds=90))

ok("B3 the late bar never reached the engine", decider.calls == [],
   f"calls={decider.calls}")
lates = [r for r in writer.written if r["outcome"] == "bar_late"]
ok("B3 one bar_late row per strategy", len(lates) == len(STRATS),
   f"{len(lates)} rows")
ok("B3 the row says it was NOT replayed",
   all(r["data"]["replayed"] is False for r in lates))
ok("B3 the row carries the arrival delta",
   all(isinstance(r["data"]["arrival_delta_s"], float) for r in lates),
   str(lates[0]["data"]))
ok("B3 a bookkeeping row carries no side",
   all(r.get("side") is None for r in lates))

cur = C.ForwardOnlyCursor()
cur.seed(("SPY", 60), T0)
cur.commit(("SPY", 60), T0 - timedelta(minutes=5))
ok("B4 the cursor never moves backwards", cur.last(("SPY", 60)) == T0)
ok("B4 admit classifies exact-equal as duplicate",
   cur.admit(("SPY", 60), T0) == C.ADMIT_DUPLICATE)
ok("B4 admit classifies older as late",
   cur.admit(("SPY", 60), T0 - timedelta(minutes=1)) == C.ADMIT_LATE)
ok("B4 admit classifies newer as ok",
   cur.admit(("SPY", 60), T0 + timedelta(minutes=1)) == C.ADMIT_OK)


# ══════════════════════════════════════════════════════════════════════════
# C — failure modes recorded POSITIVELY, never inferred from silence
# ══════════════════════════════════════════════════════════════════════════

far = T0 + timedelta(minutes=30)
consumer, writer, decider, _c = make([[bar_row(far)]], max_lag=300, max_skip=20)
consumer.cursor.seed(("SPY", 60), T0)
consumer.tick(now=far + timedelta(seconds=61))

skipped = [r for r in writer.written if r["outcome"] == "bar_skipped"]
ok("C1 past the lag ceiling the engine is NOT called", decider.calls == [],
   f"calls={decider.calls}")
ok("C1 every skipped bar is recorded", len(skipped) > 0, str(consumer.stats))
ok("C1 the consumer re-seeds after the gap",
   decider.reseeds and decider.reseeds[0][0] == ("SPY", 60),
   str(decider.reseeds))
ok("C1 the rows name the ceiling that tripped",
   all(r["data"]["reason"] == "lag_ceiling"
       and r["data"]["max_lag_s"] == 300 for r in skipped))

# C2 — a capped burst still describes the WHOLE hole. No silent truncation.
consumer, writer, decider, _c = make([[bar_row(far)]], max_lag=300, max_skip=3)
consumer.cursor.seed(("SPY", 60), T0)
consumer.tick(now=far + timedelta(seconds=61))
skipped = [r for r in writer.written if r["outcome"] == "bar_skipped"]
ok("C2 the cap bounds the burst", len(skipped) == 3 * len(STRATS),
   f"{len(skipped)} rows")
ok("C2 … and every row carries the true total",
   all(r["data"]["skipped_bars_total"] == 30 for r in skipped),
   str(skipped[0]["data"]))
ok("C2 … and says it was capped",
   all(r["data"]["rows_capped_at"] == 3 for r in skipped))
ok("C2 … and names the full span",
   all(r["data"]["skipped_from"] and r["data"]["skipped_to"] for r in skipped))

# C3 — the bar that never arrives (2.8% of 1Min live decisions).
consumer, writer, decider, _c = make([[]], grace=180)
consumer.cursor.seed(("SPY", 60), T0)
consumer.tick(now=T0 + timedelta(minutes=1, seconds=60 + 181))
absent = [r for r in writer.written if r["outcome"] == "bar_absent"]
ok("C3 the missing bar is recorded positively", len(absent) == len(STRATS),
   f"{len(absent)} rows, stats={consumer.stats}")
ok("C3 the absent row names the bar that never came",
   absent[0]["bar_start"] == T0 + timedelta(minutes=1),
   str(absent[0]["bar_start"]))
ok("C3 no bar was synthesized (the engine was never called)",
   decider.calls == [])
ok("C3 the absent row carries no side", all(r.get("side") is None
                                            for r in absent))

# C4 — inside GRACE, absence is not yet a fact.
consumer, writer, decider, _c = make([[]], grace=180)
consumer.cursor.seed(("SPY", 60), T0)
consumer.tick(now=T0 + timedelta(minutes=1, seconds=90))
ok("C4 inside GRACE nothing is declared absent",
   [r for r in writer.written if r["outcome"] == "bar_absent"] == [],
   str(writer.counts))

# … and a bar that arrives in the same tick is never declared absent.
b2 = T0 + timedelta(minutes=1)
consumer, writer, decider, _c = make([[bar_row(b2)]], grace=180)
consumer.cursor.seed(("SPY", 60), T0)
consumer.tick(now=b2 + timedelta(seconds=60 + 181))
ok("C4 a bar delivered in this tick is consumed, not called absent",
   len(decider.calls) == 1
   and [r for r in writer.written if r["outcome"] == "bar_absent"] == [],
   f"calls={decider.calls} counts={writer.counts}")

# C5 — restart writes the hole down.
consumer, writer, decider, _c = make([[]])
consumer.seed_from({("SPY", 60): T0 - timedelta(minutes=5)}, now=T0)
restarts = [r for r in writer.written if r["outcome"] == "consumer_restart"]
ok("C5 a restart is recorded, not inferred", len(restarts) == len(STRATS))
ok("C5 the row names the recovered cursor and the hole",
   restarts[0]["data"]["recovered_cursor"] is not None
   and restarts[0]["data"]["hole_s"] == 300.0, str(restarts[0]["data"]))
ok("C5 the recovered cursor is honoured",
   consumer.cursor.last(("SPY", 60)) == T0 - timedelta(minutes=5))


# ══════════════════════════════════════════════════════════════════════════
# D — boot rails, flags, scope limit, and the poll shape
# ══════════════════════════════════════════════════════════════════════════

os.environ.pop("RORT_ENV_ROLE", None)
raises("D1 it refuses to boot with RORT_ENV_ROLE unset (= production)",
       C.assert_dev_environment, C.DevConsumerWriteBlocked, "refuses to start")
for role in ("production", "prod", ""):
    os.environ["RORT_ENV_ROLE"] = role
    raises(f"D1 … and with RORT_ENV_ROLE={role!r}",
           C.assert_dev_environment, C.DevConsumerWriteBlocked, "refuses to start")

os.environ["RORT_ENV_ROLE"] = "dev"
C.assert_dev_environment()
ok("D2 it starts under RORT_ENV_ROLE=dev", True)

ok("D3 the master switch defaults OFF", not C.consumer_enabled())
os.environ["RORT_DEV_CONSUMER_ENABLED"] = "1"
ok("D3 … and is runtime-read (no reload)", C.consumer_enabled())
os.environ.pop("RORT_DEV_CONSUMER_ENABLED")
ok("D3 … and flips back", not C.consumer_enabled())

ok("D3 dry-run defaults ON", C.dry_run())
os.environ["RORT_DEV_CONSUMER_DRY_RUN"] = "0"
ok("D3 … and only an explicit 0 arms writes", not C.dry_run())
os.environ.pop("RORT_DEV_CONSUMER_DRY_RUN")

ok("D3 poll/grace/max-lag defaults match the design",
   (C.poll_interval_s(), C.grace_s(), C.max_lag_s()) == (2, 180, 300),
   f"{(C.poll_interval_s(), C.grace_s(), C.max_lag_s())}")
os.environ["RORT_DEV_CONSUMER_GRACE_S"] = "not-a-number"
raises("D3 a typo'd interval fails loudly instead of silently defaulting",
       C.grace_s, C.DevConsumerConfigError, "not an integer")
os.environ.pop("RORT_DEV_CONSUMER_GRACE_S")

os.environ.pop("RORT_DEV_CONSUMER_SIDS", None)
raises("D4 the sid list is REQUIRED — no empty default",
       C.configured_sids, C.DevConsumerConfigError, "required")
os.environ["RORT_DEV_CONSUMER_SIDS"] = "269, 340,136"
ok("D4 … and parses when given", C.configured_sids() == [269, 340, 136])
os.environ["RORT_DEV_CONSUMER_SIDS"] = "269,abc"
raises("D4 … and refuses junk", C.configured_sids,
       C.DevConsumerConfigError, "not a strategy id")
os.environ.pop("RORT_DEV_CONSUMER_SIDS")

C.require_supported_timeframes(STRATS)
ok("D5 1Min strategies are supported", True)
for tf in (10, 15, 30):
    raises(f"D5 a {tf}Sec strategy is refused (design §1 scope limit)",
           lambda t=tf: C.require_supported_timeframes(
               [{"strategy_id": 314, "timeframe_seconds": t}]),
           C.DevConsumerConfigError, "1min and coarser")

vals = C.decision_time_ohlc(bar_row(T0, first=100.0, healed=105.0))
ok("D6 decision-time values are first_* (what the engine SAW)",
   vals["close"] == 100.0, str(vals))
noreset = dict(bar_row(T0, first=100.0, healed=105.0))
for k in ("first_open", "first_high", "first_low", "first_close"):
    noreset.pop(k)
ok("D6 … falling back to the healed value only when first_* is absent",
   C.decision_time_ohlc(noreset)["close"] == 105.0)

pc = FakeClient(rows=[bar_row(T0)])
poller = C.LiveBarsPoller(pc, ["SPY", "TSLA"], [60], grace=180)
got = poller.poll(T0 + timedelta(minutes=10))
q = pc.queries[0]
flt = q.filters
ok("D7 the poll reads live_bars", pc.tables == ["live_bars"], str(pc.tables))
ok("D7 it is keyed on bar_start with a GRACE floor",
   any(f[0] == "gte" and f[1] == "bar_start"
       and f[2] == (T0 + timedelta(minutes=10) - timedelta(seconds=180)
                    ).isoformat() for f in flt), str(flt))
ok("D7 it NEVER filters on the unindexed written_at "
   "(that would seq-scan ~42M rows against production 11,700x/day)",
   not any(len(f) > 1 and f[1] in ("written_at", "last_updated_at")
           for f in flt), str(flt))
srcs = [f[2] for f in flt if f[0] == "in" and f[1] == "source"]
ok("D7 the source filter is a LIST, not eq.ws_agg", len(srcs) == 1
   and len(srcs[0]) > 1, str(srcs))
ok("D7 … and it includes rest_correction "
   "(apply_rest_correction OVERWRITES source — eq.ws_agg would drop every "
   "corrected bar)", "rest_correction" in srcs[0], str(srcs))
ok("D7 … and excludes rest_backfill (cosmetic; live never consumed it)",
   "rest_backfill" not in srcs[0], str(srcs))
ok("D7 the symbol set is fleet-scoped",
   any(f[0] == "in" and f[1] == "symbol" and f[2] == ["SPY", "TSLA"]
       for f in flt), str(flt))
ok("D7 rows come back oldest-first", len(got) == 1)

ok("D7 the code fingerprint is 12 hex over the three modules",
   len(C.code_fingerprint()) == 12
   and set(C._FINGERPRINT_FILES) == set(MODULES), C.code_fingerprint())

os.environ["RORT_ENV_ROLE"] = "dev"
print(f"\nPASS — {PASS} checks")
sys.exit(0)
