"""Measure the behavioural gap: shipped sr_channels pack vs Pine transliteration.

Board #73 step 2 evidence. No DB, no network — deterministic synthetic bars so
the result is reproducible by anyone reviewing the audit.
"""

import os
import sys
import time
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "user_packs", "sr_channels"))
sys.path.insert(0, _HERE)

from indicator_incremental import SrChannelsIncremental      # noqa: E402
from sr_channels_pine_reference import PineSRChannels        # noqa: E402


def make_bars(n, seed=7, px=100.0, tick=0.01, vol=0.0011):
    """Random walk quantised to a real tick size (ties are the point)."""
    rnd = random.Random(seed)
    bars = []
    for _ in range(n):
        drift = rnd.gauss(0, vol)
        o = px
        c = px * (1 + drift)
        hi = max(o, c) * (1 + abs(rnd.gauss(0, vol / 2)))
        lo = min(o, c) * (1 - abs(rnd.gauss(0, vol / 2)))
        q = lambda v: round(round(v / tick) * tick, 2)
        bars.append({"open": q(o), "high": q(hi), "low": q(lo),
                     "close": q(c), "volume": 1000.0})
        px = c
    return bars


def run(n=1400, seed=7, warmup=400, tie="left_strict", tick=0.01, label=""):
    bars = make_bars(n, seed=seed, tick=tick)

    ours = SrChannelsIncremental()          # defaults == Pine defaults
    pine = PineSRChannels(pivot_tie=tie)

    t0 = time.perf_counter()
    o_out = [ours.update_bar(b) for b in bars]
    t_ours = time.perf_counter() - t0

    t0 = time.perf_counter()
    p_out = [pine.update_bar(b) for b in bars]
    t_pine = time.perf_counter() - t0

    keys = ("in_channel", "res_broken", "sup_broken")
    scored = 0
    dis = {k: 0 for k in keys}
    fires_o = {k: 0 for k in keys}
    fires_p = {k: 0 for k in keys}
    nch_dis = 0
    for i in range(warmup, n):
        scored += 1
        a, b = o_out[i], p_out[i]
        for k in keys:
            av = a["src_" + k] if k != "in_channel" else a["src_in_channel"]
            bv = b[k]
            fires_o[k] += 1 if av == 1.0 else 0
            fires_p[k] += 1 if bv == 1.0 else 0
            if av != bv:
                dis[k] += 1
        if a["src_num_channels"] != b["num_channels"]:
            nch_dis += 1

    print(f"\n=== {label or 'run'}  (n={n}, seed={seed}, tie={tie}, tick={tick}) ===")
    print(f"bars scored (post-warmup): {scored}   pine pivot-recompute events: {pine.pivot_events}")
    print(f"per-bar update_bar cost   ours {t_ours/n*1000:7.3f} ms | pine-faithful {t_pine/n*1000:7.3f} ms"
          f"  ({t_ours/max(t_pine,1e-9):.0f}x)")
    for k in keys:
        shared = (fires_o[k] + fires_p[k] - dis[k]) // 2
        only_o = fires_o[k] - shared
        only_p = fires_p[k] - shared
        print(f"  {k:<14} disagree {dis[k]:5d}/{scored} = {dis[k]/scored*100:6.2f}%"
              f"   | fires: ours {fires_o[k]:5d}  pine {fires_p[k]:5d}  shared {shared:5d}"
              f"   spurious(ours-only) {only_o:4d} = {only_o/max(fires_o[k],1)*100:5.1f}%"
              f"   missed(pine-only) {only_p:4d} = {only_p/max(fires_p[k],1)*100:5.1f}%")
    print(f"  num_channels   disagree {nch_dis:5d}/{scored} = {nch_dis/scored*100:6.2f}%")
    print(f"  Pine L138-143 sort ever fired: {pine.sort_ever_fired}  (expected False = dead code)")
    return dis, scored


def pivot_census(n=1400, seed=7, tick=0.01):
    """How many pivots does each detection rule find? Ours must be a subset."""
    bars = make_bars(n, seed=seed, tick=tick)
    prd = 10
    hs = [b["high"] for b in bars]
    ls = [b["low"] for b in bars]

    def ours_pivot(ctr):
        lo_i, hi_i = max(0, ctr - prd), min(n - 1, ctr + prd)
        v = hs[ctr]
        cnt = 0
        for j in range(lo_i, hi_i + 1):
            if hs[j] > v:
                return False
            if hs[j] == v:
                cnt += 1
        return cnt == 1

    def tv_pivot(ctr, left_strict):
        if ctr - prd < 0 or ctr + prd > n - 1:
            return False
        v = hs[ctr]
        for j in range(ctr - prd, ctr):
            if hs[j] > v or (left_strict and hs[j] == v):
                return False
        for j in range(ctr + 1, ctr + prd + 1):
            if hs[j] > v or ((not left_strict) and hs[j] == v):
                return False
        return True

    a = sum(1 for i in range(prd, n - prd) if ours_pivot(i))
    b = sum(1 for i in range(prd, n - prd) if tv_pivot(i, True))
    c = sum(1 for i in range(prd, n - prd) if tv_pivot(i, False))
    subset_b = all((not ours_pivot(i)) or tv_pivot(i, True) for i in range(prd, n - prd))
    subset_c = all((not ours_pivot(i)) or tv_pivot(i, False) for i in range(prd, n - prd))
    print(f"\n=== pivot-high census (tick={tick}) ===")
    print(f"  ours (strict-unique whole window): {a}")
    print(f"  TV variant A (left-strict):        {b}   ours-subset-of-A: {subset_b}")
    print(f"  TV variant B (right-strict):       {c}   ours-subset-of-B: {subset_c}")
    print(f"  pivots we MISS vs A: {b - a} ({(b-a)/max(b,1)*100:.1f}%)  vs B: {c - a} ({(c-a)/max(c,1)*100:.1f}%)")


if __name__ == "__main__":
    pivot_census(tick=0.01)
    pivot_census(tick=0.05)
    run(seed=7, label="seed 7")
    run(seed=13, label="seed 13")
    run(seed=7, tie="right_strict", label="seed 7 / other tie-break")
