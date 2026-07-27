"""#125 (V1.17): pure-logic tests for the settled-day candidate walk and the
nightly-retrue window list — no DB, no network. Run:
    python src/_test_settled_day_walk.py
Covers: legacy walk unchanged when RORT_PARITY_SETTLED_MIN_TDAYS is unset;
trading-day-aware walk across a weekend (the #103/#116/#125 false-red class:
'3 calendar days back' is only T+2 trading days after a weekend); retrue
window list always includes the pointer day and keeps the newest days."""
import os
import sys
import datetime as dt

sys.path.insert(0, os.path.dirname(__file__))
import settle_sweeper as ss

FAILURES = []


def check(name, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got} want={want}")
    if not ok:
        FAILURES.append(name)


def d(iso):
    return dt.date.fromisoformat(iso)


def utc(iso):
    x = d(iso)
    return dt.datetime(x.year, x.month, x.day, tzinfo=dt.timezone.utc)


def candidates(today, n=3):
    return [x.isoformat() for i, x in
            enumerate(ss.iter_settled_candidates(d(today))) if i < n]


print("— legacy walk (env unset): byte-identical to the historical range(3,25) —")
os.environ.pop("RORT_PARITY_SETTLED_MIN_TDAYS", None)
check("Sat 07-25 legacy first", candidates("2026-07-25", 1), ["2026-07-22"])
check("Sat 07-25 legacy seq", candidates("2026-07-25", 4),
      ["2026-07-22", "2026-07-21", "2026-07-20", "2026-07-17"])
check("Mon 07-27 legacy first", candidates("2026-07-27", 1), ["2026-07-24"])
check("Thu 07-23 legacy first", candidates("2026-07-23", 1), ["2026-07-20"])

print("— elapsed weekdays —")
check("07-22→Sat 07-25", ss._elapsed_weekdays(d("2026-07-22"), d("2026-07-25")), 2)
check("07-21→Fri 07-24", ss._elapsed_weekdays(d("2026-07-21"), d("2026-07-24")), 2)
check("07-24→Sat 07-25", ss._elapsed_weekdays(d("2026-07-24"), d("2026-07-25")), 0)
check("07-20→Mon 07-27", ss._elapsed_weekdays(d("2026-07-20"), d("2026-07-27")), 4)

print("— armed walk (min trading days) —")
os.environ["RORT_PARITY_SETTLED_MIN_TDAYS"] = "4"
check("Sat 07-25 N=4 first", candidates("2026-07-25", 1), ["2026-07-20"])
check("Mon 07-27 N=4 first", candidates("2026-07-27", 1), ["2026-07-20"])
os.environ["RORT_PARITY_SETTLED_MIN_TDAYS"] = "3"
check("Sat 07-25 N=3 first", candidates("2026-07-25", 1), ["2026-07-21"])
check("Tue 07-28 N=3 first", candidates("2026-07-28", 1), ["2026-07-22"])
# The recurrence pattern this kills: every observed red day was exactly T+2
# trading days old under the legacy walk (07-20@07-23, 07-21@07-24, 07-22@07-25).
for today, red_day in [("2026-07-23", "2026-07-20"), ("2026-07-24", "2026-07-21"),
                       ("2026-07-25", "2026-07-22")]:
    got = candidates(today, 1)[0]
    check(f"N=3 never picks the {red_day} red day on {today}",
          got != red_day and ss._elapsed_weekdays(d(got), d(today)) >= 3, True)
os.environ.pop("RORT_PARITY_SETTLED_MIN_TDAYS", None)

print("— retrue window list —")
os.environ.pop("RORT_NIGHTLY_SETTLE_RETRUE_WINDOW", None)
got = [x.date().isoformat() for x in ss._retrue_day_list(utc("2026-07-22"), d("2026-07-25"))]
check("default window=1 → pointer only", got, ["2026-07-22"])
os.environ["RORT_NIGHTLY_SETTLE_RETRUE_WINDOW"] = "4"
got = [x.date().isoformat() for x in ss._retrue_day_list(utc("2026-07-20"), d("2026-07-25"))]
check("window=4 keeps pointer + newest 3",
      got, ["2026-07-20", "2026-07-22", "2026-07-23", "2026-07-24"])
got = [x.date().isoformat() for x in ss._retrue_day_list(utc("2026-07-22"), d("2026-07-25"))]
check("window=4, short range → all weekdays",
      got, ["2026-07-22", "2026-07-23", "2026-07-24"])
got = [x.date().isoformat() for x in ss._retrue_day_list(utc("2026-07-24"), d("2026-07-25"))]
check("window=4, pointer=yesterday → pointer only", got, ["2026-07-24"])
os.environ.pop("RORT_NIGHTLY_SETTLE_RETRUE_WINDOW", None)

print()
if FAILURES:
    print(f"❌ {len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("✅ all settled-day walk / retrue-window checks passed")
