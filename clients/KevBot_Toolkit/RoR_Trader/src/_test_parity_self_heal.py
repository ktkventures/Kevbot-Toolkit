"""#125 (V1.17): unit test for the parity suite's self-heal flow — monkeypatched
data sources, no DB and no network. Run:
    python src/_test_parity_self_heal.py
Asserts: (1) default OFF → revision-drift still FAILs (no behavior change);
(2) armed → drift item heals (retrue called once per symbol) and reports PASS
with the self-healed annotation; (3) a diff that SURVIVES the retrue (a real
pipeline bug, not Polygon churn) still FAILs; (4) row-count mismatches are
never healed (signature guard)."""
import os
import sys
import datetime as dt

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

import fidelity_parity_suite as fps
import settle_sweeper
import bar_cache as bc
import data_loader as dl

FAILURES = []


def check(name, cond, note=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {note}")
    if not cond:
        FAILURES.append(name)


def frame(close_vals):
    idx = pd.date_range("2026-07-22 14:00:00+00:00", periods=len(close_vals),
                        freq="10s")
    return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0,
                         "close": close_vals, "volume": 100.0}, index=idx)


class Fake:
    """Cache serves DRIFTED values until retrue_range is called for the symbol."""
    def __init__(self, healable=True, rows_short=0):
        self.healed = set()
        self.retrue_calls = []
        self.healable = healable
        self.rows_short = rows_short

    def native(self, *a, **k):
        return frame([2.0, 2.0, 2.0, 2.0])

    def cached(self, sym, **k):
        good = sym in self.healed and self.healable
        f = frame([2.0, 2.0, 2.0, 2.0] if good else [2.0, 2.01, 2.0, 2.0])
        return f.iloc[:len(f) - self.rows_short] if self.rows_short else f

    def retrue(self, sym, s, e):
        self.retrue_calls.append((sym, s.date().isoformat()))
        self.healed.add(sym)
        return {"symbol": sym, "sec_rows": 4, "min_rows": 1, "days": 1}


def run_matrix(fake, matrix):
    orig = (bc.cached_load_market_data, dl.load_from_polygon,
            settle_sweeper.retrue_range, fps.CACHE_PARITY_MATRIX)
    bc.cached_load_market_data = lambda sym, **k: fake.cached(sym, **k)
    dl.load_from_polygon = fake.native
    settle_sweeper.retrue_range = fake.retrue
    fps.CACHE_PARITY_MATRIX = matrix
    try:
        return fps._cache_parity(dt.date(2026, 7, 22))
    finally:
        (bc.cached_load_market_data, dl.load_from_polygon,
         settle_sweeper.retrue_range, fps.CACHE_PARITY_MATRIX) = orig


print("— default OFF: drift FAILs exactly as before —")
os.environ.pop("RORT_PARITY_SELF_HEAL", None)
fake = Fake()
res = run_matrix(fake, [("SPY", "10Sec")])
check("both sessions FAIL", [s for _, s, _ in res] == ["FAIL", "FAIL"], str(res))
check("no retrue attempted", fake.retrue_calls == [])

print("— armed: drift self-heals to PASS, one retrue per symbol —")
os.environ["RORT_PARITY_SELF_HEAL"] = "1"
fake = Fake()
res = run_matrix(fake, [("SPY", "10Sec"), ("SPY", "1Min")])
check("all four items PASS", [s for _, s, _ in res] == ["PASS"] * 4, str(res))
check("triggering item carries the annotation; healed-symbol siblings pass clean",
      "self-healed revision drift (was OHLCVdiffs=1)" in res[0][2]
      and not any("self-healed" in d_ for _, _, d_ in res[1:]))
check("retrue called ONCE for SPY", fake.retrue_calls == [("SPY", "2026-07-22")],
      str(fake.retrue_calls))

print("— armed: a diff that survives retrue still FAILs (real bug) —")
fake = Fake(healable=False)
res = run_matrix(fake, [("SPY", "10Sec")])
check("still FAIL", [s for _, s, _ in res] == ["FAIL", "FAIL"], str(res))
check("post-heal count reported", all("self-heal attempted, still OHLCVdiffs=1" in d_
                                      for _, _, d_ in res))

print("— armed: rowMM≠0 is NOT the drift signature → no heal —")
fake = Fake(rows_short=1)
res = run_matrix(fake, [("SPY", "10Sec")])
check("FAIL without heal", [s for _, s, _ in res] == ["FAIL", "FAIL"], str(res))
check("no retrue attempted", fake.retrue_calls == [])
os.environ.pop("RORT_PARITY_SELF_HEAL", None)

print()
if FAILURES:
    print(f"❌ {len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("✅ all self-heal flow checks passed")
