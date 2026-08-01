"""Board #73 step 7 — settle (or bound) audit §2.1's pivot tie-break.

WHY THIS EXISTS
---------------
The rebuild shipped behind RORT_SR_CHANNELS_PINE with `pivot_tie="left_strict"`
as an UNEXAMINED DEFAULT. Audit §2.1 framed the open question as a binary —
"which side of ta.pivothigh admits equality, left or right" — and assigned it to
a live TradingView chart. Step 4's gate parametrises over `tie` but passes the
SAME value to both sides, so it is an equivalence gate for a GIVEN tie and
cannot settle the choice. Arming bakes the default in.

THE PREMISE THIS SCRIPT TESTS
-----------------------------
§2.1 asserts, without a citation, that TradingView's rule is ASYMMETRIC ("one
side admits equality"). Every reachable secondary description of ta.pivothigh
instead describes it as SYMMETRIC-STRICT: the source value must be strictly
greater than every neighbour on BOTH sides, so any tie anywhere in the
2*prd+1 window kills the pivot. That is a THIRD candidate the audit never
considered — and it happens to be exactly the LEGACY pack's rule.

If symmetric-strict is right, then:
  * §2.1 is NOT a divergence at all — legacy's pivot detection was faithful,
    and the 18-55% "miss rate" the audit measured is measured against an
    instrument that assumed the asymmetric premise;
  * BOTH shipped tie options are OVER-inclusive w.r.t. the Pine, so arming
    either one bakes in a pivot set larger than TradingView's.

The rebuild's value under §2.2 (newest-first iteration), §2.3 (simultaneous
high+low keeps only the high) and §2.4 (freeze between pivots) is UNAFFECTED
either way — those three are read directly off the Pine source with no
ambiguity. Only pivot DETECTION is in question.

=====================================================================
EXPECTED RESULT — WRITTEN DOWN BEFORE THE RUN
=====================================================================
A. both_strict pivots are a STRICT SUBSET of left_strict's and of
   right_strict's, on every symbol. (Provable: both_strict's reject
   condition is the union of the other two's.) A run showing equality
   everywhere would mean real bars carry no ties in the window and the whole
   question is moot -- which contradicts the 102 field-diffs already measured.
B. [WITHDRAWN — NOT TESTED. The first cut of this script tried to compare the
   LEGACY class's pivot set against both_strict and reported "DIFFERS" on all
   three symbols. That was an INSTRUMENT BUG, not a result:
   `SrChannelsIncremental._pivot_highs` is a bounded LIST OF VALUES (len 312
   on a 2000-bar run, NaN where there is no pivot), not an index-keyed map, so
   the comparison put value-space against index-space and could only ever say
   "DIFFERS". Aligning the legacy class's rolling buffer to absolute bar
   indices is real work and is NOT attempted here. Reading the source,
   legacy's rule (`is_pivot` AND `max_count == 1` over [p-prd, p+prd])
   LOOKS like the same predicate as both_strict — but that is a reading, and
   this task's whole standard is that mappings are MEASURED, not read. Treat
   it as an open question, not as a finding either way.]
C. Choosing both_strict over the shipped left_strict default MOVES the
   output materially -- same order as the 102 left-vs-right field-diffs,
   not zero. If it were zero the default would be harmless and the arm
   could proceed unexamined.

Run from RoR_Trader/:  .venv/bin/python tools/sr_channels_tiebreak_73.py
"""

import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "user_packs", "sr_channels"))
sys.path.insert(0, _HERE)

from dotenv import load_dotenv                                    # noqa: E402
_ENV = os.environ.get("RORT_ENV_FILE") or os.path.join(_ROOT, "src", ".env")
if not os.path.exists(_ENV):
    _ENV = ("/home/kevin/projects/Kevbot-Toolkit/clients/KevBot_Toolkit"
            "/RoR_Trader/src/.env")
load_dotenv(_ENV, override=True)
os.environ.setdefault("USE_DB", "true")

from indicator_incremental import SrChannelsPineIncremental       # noqa: E402,F401
from sr_channels_pine_reference import PineSRChannels              # noqa: E402
from sr_channels_pine_realbars import fetch_real_bars, PINE_FIELDS  # noqa: E402

PRD = 10


class PineSRChannelsBothStrict(PineSRChannels):
    """The transliteration with a SYMMETRIC-STRICT ta.pivothigh.

    Identical to PineSRChannels in every other clause — this overrides one
    method so the only thing that varies is the tie-break.
    """

    def _pivot_at(self, ctr, hi_side):
        n = len(self.h)
        if ctr - self.prd < 0 or ctr + self.prd > n - 1:
            return None
        get = self._src1 if hi_side else self._src2
        val = get(ctr)
        for j in range(ctr - self.prd, ctr + self.prd + 1):
            if j == ctr:
                continue
            v = get(j)
            beats = v > val if hi_side else v < val
            if beats or v == val:      # any tie ANYWHERE kills it
                return None
        return val


def pivot_set(bars, rule):
    """Standalone pivot detection over the whole bar array.

    Deliberately does NOT reuse any class's internals: the point is to
    compare the RULES, so all three are evaluated by one code path.
    Returns {(centre_index, 'H'|'L')}.
    """
    hi = [b["high"] for b in bars]
    lo = [b["low"] for b in bars]
    out = set()
    n = len(bars)
    for ctr in range(PRD, n - PRD):
        for side, arr in (("H", hi), ("L", lo)):
            val = arr[ctr]
            ok = True
            for j in range(ctr - PRD, ctr + PRD + 1):
                if j == ctr:
                    continue
                v = arr[j]
                beats = v > val if side == "H" else v < val
                if beats:
                    ok = False
                    break
                if v == val:
                    left = j < ctr
                    if rule == "both_strict":
                        ok = False
                    elif rule == "left_strict" and left:
                        ok = False
                    elif rule == "right_strict" and not left:
                        ok = False
                    if not ok:
                        break
            if ok:
                out.add((ctr, side))
    return out


def field_diffs(a_out, b_out, warmup=400):
    d = defaultdict(int)
    for i in range(warmup, len(a_out)):
        for f in PINE_FIELDS:
            av = a_out[i]["src_" + f] if "src_" + f in a_out[i] else a_out[i][f]
            bv = b_out[i]["src_" + f] if "src_" + f in b_out[i] else b_out[i][f]
            if av != bv:
                d[f] += 1
    return dict(d)


def main():
    symbols = sys.argv[1].split(",") if len(sys.argv) > 1 else ["SPY", "NVDA", "TSLA"]
    print(__doc__.split("Run from")[0])
    print("=" * 72)
    data = fetch_real_bars(symbols)

    print()
    print("=" * 72)
    print("A · PIVOT SETS UNDER THE THREE CANDIDATE RULES")
    print("     bs=both_strict  ls=left_strict  rs=right_strict")
    all_a_ok = True
    for sym, bars in data.items():
        if len(bars) < 700:
            print(f"  {sym}: only {len(bars)} bars — SKIPPED")
            continue
        bs = pivot_set(bars, "both_strict")
        ls = pivot_set(bars, "left_strict")
        rs = pivot_set(bars, "right_strict")
        subset = bs < ls and bs < rs
        all_a_ok = all_a_ok and subset
        print(f"  {sym:5s} |bs|={len(bs):4d} |ls|={len(ls):4d} |rs|={len(rs):4d}")
        print(f"        bs STRICT subset of both ls and rs : "
              f"{'YES' if subset else 'NO'}   "
              f"(ls\\bs={len(ls - bs)}, rs\\bs={len(rs - bs)})")

    print()
    print("=" * 72)
    print("C · HOW MUCH DOES THE UNEXAMINED DEFAULT MOVE THE OUTPUT?")
    print("    reference transliteration, tie-break varied, everything else equal.")
    tot_ls = tot_rs = 0
    for sym, bars in data.items():
        if len(bars) < 700:
            continue
        ref_bs = PineSRChannelsBothStrict()
        o_bs = [ref_bs.update_bar(b) for b in bars]
        r_ls = PineSRChannels(pivot_tie="left_strict")
        o_ls = [r_ls.update_bar(b) for b in bars]
        r_rs = PineSRChannels(pivot_tie="right_strict")
        o_rs = [r_rs.update_bar(b) for b in bars]
        d_ls = field_diffs(o_ls, o_bs)
        d_rs = field_diffs(o_rs, o_bs)
        tot_ls += sum(d_ls.values())
        tot_rs += sum(d_rs.values())
        print(f"  {sym:5s} SHIPPED-DEFAULT(left_strict) vs both_strict : "
              f"{sum(d_ls.values()):4d} field-diffs {d_ls}")
        print(f"        right_strict                 vs both_strict : "
              f"{sum(d_rs.values()):4d} field-diffs {d_rs}")

    print()
    print("=" * 72)
    print(f"TOTAL left_strict  (the SHIPPED default) vs both_strict : {tot_ls}")
    print(f"TOTAL right_strict                       vs both_strict : {tot_rs}")
    print()
    print(f"  [{'PASS' if all_a_ok else 'FAIL'}] A: both_strict is a strict subset of both asymmetric rules")
    print("  [ -- ] B: WITHDRAWN — see the docstring; not tested by this script")
    print(f"  [{'PASS' if tot_ls > 0 else 'FAIL'}] C: the shipped default is NOT interchangeable with both_strict")
    return 0 if (all_a_ok and tot_ls > 0) else 1


if __name__ == "__main__":
    sys.exit(main())
