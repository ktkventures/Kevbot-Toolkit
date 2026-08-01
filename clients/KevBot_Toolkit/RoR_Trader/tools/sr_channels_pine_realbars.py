"""Board #73 step 7 — verify the sr_channels Pine rebuild on REAL BARS.

Step 4's gate (src/test_sr_channels_pine_rebuild.py) is DETERMINISTIC-SYNTHETIC
by design: a Gaussian random walk quantised to a tick, so anyone can reproduce
it with no DB. This script runs the same differential against real market bars
pulled from `bar_cache`, which is the one thing the synthetic fixture cannot
manufacture:

  * real closes repeat far more often than a Gaussian walk's do, so the
    ASYMMETRIC pivot tie-break (audit §2.1) is exercised orders of magnitude
    harder than on synthetic bars;
  * real sessions contain flat bars (H==L, O==C) and overnight gaps;
  * real prices spend long stretches pinned to round numbers, which is exactly
    where "one value unshifted per confirmation bar" (§2.3) and the
    newest-first array (§2.2) can diverge.

=====================================================================
EXPECTED AGREEMENT — WRITTEN DOWN BEFORE THE RESULT WAS READ
=====================================================================
1. NEW vs PINE: EXACTLY IDENTICAL. 100.000% on every Pine-derived field on
   every scored bar, both tie-breaks, every symbol. Not "close", not ">=99%".
   Both consume the same OHLC, both are deterministic, and the rebuild is a
   line-for-line port of the transliteration. ONE mismatched bar is a bug.
   A near-miss (e.g. 99.9%) is the WORST outcome: it means a real-bar
   condition exists that the synthetic fixture never generates.

2. LEGACY vs PINE: MUST STILL DISAGREE, materially. The audit measured ~1 in 5
   of our break signals as not-a-Pine-break on synthetic bars. If legacy came
   out identical on real bars the differential would be vacuous and the whole
   rebuild unjustified, so this is a real prediction that can fail.

3. Channels must actually FORM on these bars (num_channels >= 2 somewhere,
   breaks fire more than a handful of times). A silent all-zero run would make
   points 1 and 2 pass for the wrong reason.

4. FLAG OFF IS A STRICT NO-OP: with RORT_SR_CHANNELS_PINE unset the pack
   registry resolves sr_channels to the LEGACY class; with it set, to the Pine
   class. Nothing else about the pack changes.

Run from RoR_Trader/:  .venv/bin/python tools/sr_channels_pine_realbars.py
"""

import os
import sys
import json
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "user_packs", "sr_channels"))
sys.path.insert(0, _HERE)

from dotenv import load_dotenv                                   # noqa: E402
# src/.env is gitignored, so a worktree has none — fall back to the main
# checkout's copy (set RORT_ENV_FILE to override).
_ENV = os.environ.get("RORT_ENV_FILE") or os.path.join(_ROOT, "src", ".env")
if not os.path.exists(_ENV):
    _ENV = ("/home/kevin/projects/Kevbot-Toolkit/clients/KevBot_Toolkit"
            "/RoR_Trader/src/.env")
load_dotenv(_ENV, override=True)
os.environ.setdefault("USE_DB", "true")

from indicator_incremental import (                              # noqa: E402
    SrChannelsIncremental,
    SrChannelsPineIncremental,
)
from sr_channels_pine_reference import PineSRChannels             # noqa: E402

PINE_FIELDS = ("in_channel", "res_broken", "sup_broken", "num_channels")


def fetch_real_bars(symbols, timeframe="1Min", limit_per_symbol=2000):
    """LIGHT read of settled bar_cache rows. One query per symbol, capped."""
    from db import get_admin_client
    client = get_admin_client()
    out = {}
    for sym in symbols:
        r = (client.table("bar_cache")
             .select("ts,open,high,low,close,volume")
             .eq("symbol", sym)
             .eq("timeframe", timeframe)
             .eq("settled", True)
             .order("ts", desc=True)
             .limit(limit_per_symbol)
             .execute())
        rows = sorted(r.data or [], key=lambda x: x["ts"])
        bars = [{"open": float(b["open"]), "high": float(b["high"]),
                 "low": float(b["low"]), "close": float(b["close"]),
                 "volume": float(b["volume"] or 0.0)} for b in rows]
        out[sym] = bars
        print(f"  {sym} {timeframe}: {len(bars)} settled bars "
              f"[{rows[0]['ts']} .. {rows[-1]['ts']}]" if bars else f"  {sym}: EMPTY")
    return out


def run_triplet(bars, tie, warmup=400):
    """Drive all three implementations over the same real bars.

    `tie` is passed to BOTH the rebuild and the transliteration — this is
    the equivalence gate for a GIVEN tie-break, not a test of which
    tie-break is right. §2.1's open question is measured separately by
    tie_sensitivity().
    """
    new = SrChannelsPineIncremental(pivot_tie=tie)
    ref = PineSRChannels(pivot_tie=tie)
    old = SrChannelsIncremental()
    n_out = [new.update_bar(b) for b in bars]
    p_out = [ref.update_bar(b) for b in bars]
    o_out = [old.update_bar(b) for b in bars]
    return n_out, p_out, o_out, warmup


def score(n_out, p_out, o_out, warmup):
    n = len(p_out)
    res = {"scored": 0,
           "new_mismatch": defaultdict(int), "old_mismatch": defaultdict(int),
           "first_new_mismatch": None,
           "pine_fires": defaultdict(int), "old_fires": defaultdict(int),
           "max_channels": 0.0}
    for i in range(warmup, n):
        res["scored"] += 1
        a, b, c = n_out[i], p_out[i], o_out[i]
        res["max_channels"] = max(res["max_channels"], float(b["num_channels"]))
        for f in PINE_FIELDS:
            av, bv, cv = a["src_" + f], b[f], c["src_" + f]
            if av != bv:
                res["new_mismatch"][f] += 1
                if res["first_new_mismatch"] is None:
                    res["first_new_mismatch"] = (i, f, av, bv)
            if cv != bv:
                res["old_mismatch"][f] += 1
            if f != "num_channels":
                res["pine_fires"][f] += 1 if bv == 1.0 else 0
                res["old_fires"][f] += 1 if cv == 1.0 else 0
    return res


def tie_sensitivity(bars, warmup=400):
    """§2.1 — the ONE clause the audit could not settle offline.

    Run the REBUILD against itself under both tie-breaks. Step 4's gate
    parametrises over `tie` but passes the SAME value to both sides, so it
    is an equivalence gate for a GIVEN tie — it cannot, by construction,
    settle which tie TradingView actually uses. This measures how much
    that unsettled choice moves the output on real bars. (Measured
    non-zero on the synthetic fixture too: 7-193 field-diffs across the
    gate's own seeds/ticks. The choice is load-bearing either way.)
    """
    ls = SrChannelsPineIncremental(pivot_tie="left_strict")
    rs = SrChannelsPineIncremental(pivot_tie="right_strict")
    a = [ls.update_bar(b) for b in bars]
    c = [rs.update_bar(b) for b in bars]
    diff = defaultdict(int)
    for i in range(warmup, len(bars)):
        for f in PINE_FIELDS:
            if a[i]["src_" + f] != c[i]["src_" + f]:
                diff[f] += 1
    return dict(diff), len(ls._pivotvals), len(rs._pivotvals)


def flag_resolution_check():
    """Rail 4: the manifest flag_variants switch, both directions."""
    import importlib
    import pack_registry
    importlib.reload(pack_registry)
    manifest = json.load(open(os.path.join(
        _ROOT, "user_packs", "sr_channels", "manifest.json")))
    flag = "RORT_SR_CHANNELS_PINE"
    prior = os.environ.pop(flag, None)
    off = pack_registry.select_flag_variant(manifest)
    os.environ[flag] = "1"
    on = pack_registry.select_flag_variant(manifest)
    if prior is None:
        os.environ.pop(flag, None)
    else:
        os.environ[flag] = prior
    return off, on


def main():
    symbols = sys.argv[1].split(",") if len(sys.argv) > 1 else ["SPY", "NVDA", "TSLA"]
    print(__doc__.split("Run from")[0])
    print("=" * 72)
    print("FETCHING REAL BARS (settled bar_cache rows, light read)")
    data = fetch_real_bars(symbols)

    print()
    print("=" * 72)
    print("FLAG RESOLUTION (rail 4)")
    off, on = flag_resolution_check()
    print(f"  flag UNSET -> select_flag_variant = {off!r}")
    print(f"  flag =1     -> select_flag_variant = "
          f"{ (on or {}).get('env') } -> { (on or {}).get('overrides') or on }")

    total_new = 0
    total_old = 0
    total_scored = 0
    print()
    print("=" * 72)
    print("DIFFERENTIAL ON REAL BARS")
    for sym, bars in data.items():
        if len(bars) < 700:
            print(f"  {sym}: only {len(bars)} bars — SKIPPED (needs >700)")
            continue
        for tie in ("left_strict", "right_strict"):
            n_out, p_out, o_out, warmup = run_triplet(bars, tie)
            r = score(n_out, p_out, o_out, warmup)
            nm = sum(r["new_mismatch"].values())
            om = sum(r["old_mismatch"].values())
            total_new += nm
            total_old += om
            total_scored += r["scored"] * len(PINE_FIELDS)
            print(f"  {sym:5s} tie={tie:13s} scored={r['scored']:5d} bars "
                  f"x {len(PINE_FIELDS)} fields")
            print(f"        NEW vs PINE mismatches = {nm}"
                  + (f"   FIRST: {r['first_new_mismatch']}" if nm else "   [EXACT]"))
            print(f"        LEGACY vs PINE mismatches = {om}  "
                  f"({dict(r['old_mismatch'])})")
            print(f"        pine fires: res_broken={r['pine_fires']['res_broken']} "
                  f"sup_broken={r['pine_fires']['sup_broken']} "
                  f"in_channel={r['pine_fires']['in_channel']} | "
                  f"legacy fires: res={r['old_fires']['res_broken']} "
                  f"sup={r['old_fires']['sup_broken']} | "
                  f"max_channels={r['max_channels']}")

    print()
    print("=" * 72)
    print("TIE-BREAK SENSITIVITY (audit §2.1 — the unsettled clause)")
    print("  rebuild(left_strict) vs rebuild(right_strict), same real bars.")
    tie_total = 0
    for sym, bars in data.items():
        if len(bars) < 700:
            continue
        d, npl, npr = tie_sensitivity(bars)
        tie_total += sum(d.values())
        print(f"  {sym:5s} field-diffs={sum(d.values()):4d} {dict(d)} | "
              f"live pivots kept: left_strict={npl} right_strict={npr}")
    print(f"  TOTAL tie-break-induced field differences: {tie_total}")
    print("  -> if >0, the two variants are NOT interchangeable on real bars,")
    print("     and step 4's tie-parametrised gate cannot settle the choice.")

    print()
    print("=" * 72)
    print(f"TOTAL field-comparisons scored : {total_scored}")
    print(f"NEW    vs PINE mismatches      : {total_new}")
    print(f"LEGACY vs PINE mismatches      : {total_old}")
    print()
    ok_new = total_new == 0
    ok_old = total_old > 0
    print(f"  [{'PASS' if ok_new else 'FAIL'}] prediction 1: new == pine exactly")
    print(f"  [{'PASS' if ok_old else 'FAIL'}] prediction 2: legacy still differs")
    return 0 if (ok_new and ok_old) else 1


if __name__ == "__main__":
    sys.exit(main())
