"""Acceptance test — board #228: DAILY_CAP is adjustable at RUNTIME.

The defect: `DAILY_CAP` was a module constant read at IMPORT. The comment above
it had claimed "easy to adjust — one integer, read at runtime, no deploy" since
board #219, and that was never true — changing it cost a PR, gates, a merge, a
deploy AND a dispatcher restart. Kevin hit exactly that raising 40 -> 50 on
07-30 and asked why it was not faster.

The fix: `effective_daily_cap()` reads `system_settings.dispatcher_daily_cap`
fresh on EVERY poll, same table and shape as `data_worker_streaming_enabled`.
Change the row, the running loop honours it next poll.

Every rail below is asserted with the behaviour that FAILS without it, because
the thing being defended is a circuit breaker: both "silently stuck at the old
value" and "silently honouring nonsense" are worse than a loud wrong number.

Run:  .venv/bin/python src/test_dispatcher_runtime_cap_228.py   (from RoR_Trader root)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "team_dispatcher"))
import dispatcher as D  # noqa: E402

FAILED = []


def ok(label, cond, detail=""):
    print(f"  {'ok' if cond else 'FAIL'}: {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(label)


class FakeApi:
    """Stands in for api(); records the query so we can assert what was asked."""

    def __init__(self, rows=None, raises=None):
        self.rows, self.raises, self.queries = rows, raises, []

    def __call__(self, method, path, *a, **k):
        self.queries.append((method, path))
        if self.raises:
            raise self.raises
        return self.rows


def with_api(fake, fn):
    real = D.api
    D.api = fake
    try:
        return fn()
    finally:
        D.api = real


print("― rail 1: the live value comes from the settings row, not the constant ―")
f = FakeApi(rows=[{"value": 75}])
cap, src = with_api(f, D.effective_daily_cap)
ok("a settings row overrides the constant", cap == 75, f"got {cap}")
ok("source names the settings row", src == "settings", src)
ok("the constant is NOT what was returned", cap != D.DAILY_CAP)
ok("it READ system_settings by key",
   any("system_settings" in p and D.DAILY_CAP_SETTING in p for _, p in f.queries), str(f.queries))
ok("it only READ — a cap check never writes",
   all(m == "GET" for m, _ in f.queries), str(f.queries))

print("― rail 2: a missing row falls back to the constant, and SAYS so ―")
cap, src = with_api(FakeApi(rows=[]), D.effective_daily_cap)
ok("missing row -> the constant", cap == D.DAILY_CAP, f"got {cap}")
ok("source says the row is missing", "no row" in src, src)

print("― rail 3: an unreadable settings table never silently reverts ―")
cap, src = with_api(FakeApi(raises=RuntimeError("boom")), D.effective_daily_cap)
ok("unreadable -> the constant", cap == D.DAILY_CAP, f"got {cap}")
ok("source names the failure, not just 'constant'",
   "unreadable" in src and "RuntimeError" in src, src)

print("― rail 4: a non-integer value is refused, loudly ―")
for bad in ("not-an-int", None, {"nested": 1}):
    cap, src = with_api(FakeApi(rows=[{"value": bad}]), D.effective_daily_cap)
    ok(f"value {bad!r} -> the constant", cap == D.DAILY_CAP, f"got {cap}")
    ok(f"value {bad!r} is named in the source", "not an int" in src, src)

print("― rail 5: out-of-range is CLAMPED and reported, never honoured ―")
cap, src = with_api(FakeApi(rows=[{"value": 99999}]), D.effective_daily_cap)
ok("absurd high -> clamped to the ceiling", cap == D.DAILY_CAP_MAX, f"got {cap}")
ok("clamp is reported with the original value",
   "CLAMPED" in src and "99999" in src, src)
cap, src = with_api(FakeApi(rows=[{"value": 0}]), D.effective_daily_cap)
ok("zero -> clamped to 1, never 0", cap == 1, f"got {cap}")
ok("a negative cap cannot disable the breaker",
   with_api(FakeApi(rows=[{"value": -5}]), D.effective_daily_cap)[0] == 1)

print("― rail 6: the breaker still exists — a huge row cannot remove it ―")
cap, _ = with_api(FakeApi(rows=[{"value": 10 ** 9}]), D.effective_daily_cap)
ok("no setting can exceed DAILY_CAP_MAX", cap <= D.DAILY_CAP_MAX, f"got {cap}")
ok("DAILY_CAP_MAX is a real ceiling, not None", isinstance(D.DAILY_CAP_MAX, int))

print("― rail 7: a string integer is accepted (JSONB round-trips loosely) ―")
cap, src = with_api(FakeApi(rows=[{"value": "45"}]), D.effective_daily_cap)
ok('"45" is honoured as 45', cap == 45 and src == "settings", f"{cap} {src}")

print("― rail 8: the constant survives as the documented fallback ―")
ok("DAILY_CAP still defined", isinstance(D.DAILY_CAP, int) and D.DAILY_CAP > 0)
ok("the setting key is named once, not scattered",
   D.DAILY_CAP_SETTING == "dispatcher_daily_cap", D.DAILY_CAP_SETTING)

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): " + " · ".join(FAILED))
    sys.exit(1)
print("ALL PASS — board #228 runtime daily cap")
