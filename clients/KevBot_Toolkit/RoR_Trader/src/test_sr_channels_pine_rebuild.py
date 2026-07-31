"""Board #73 step 4 — sr_channels rebuilt against the Pine, behind
RORT_SR_CHANNELS_PINE (default OFF).

The gate is DIFFERENTIAL, not assertive: the pack's faithful class is
run bar-for-bar against `tools/sr_channels_pine_reference.PineSRChannels`
— the transliteration written for the step-2 audit, carrying the Pine's
own line numbers — over identical deterministic bars. Equivalence is
measured, not claimed.

Every rail named in docs/_active/Audit_SR_Channels_vs_Pine.md gets a
test that FAILS against the legacy class, so each one is demonstrably
load-bearing rather than decorative.

No DB, no network, no fixtures on disk — bars are generated from a
seeded LCG so any reader can reproduce the numbers.
"""

import json
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent
_PACK = _ROOT / "user_packs" / "sr_channels"

for _p in (str(_SRC), str(_ROOT / "tools"), str(_PACK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pack_registry  # noqa: E402
from sr_channels_pine_reference import PineSRChannels  # noqa: E402

FLAG = "RORT_SR_CHANNELS_PINE"


# ---------------------------------------------------------------------
# Deterministic bars — a seeded LCG, so no numpy/random dependency and
# byte-identical output on every machine.
# ---------------------------------------------------------------------

def make_bars(n=900, seed=7, tick=0.01, start=100.0):
    """Quantised OHLC. `tick` matters: equal highs are what quantised
    prices DO, and they are exactly what the legacy pivot rule threw
    away (audit §2.1)."""
    state = seed
    px = start
    bars = []

    def rnd():
        nonlocal state
        state = (1103515245 * state + 12345) % (2 ** 31)
        return state / (2 ** 31)

    for _ in range(n):
        drift = (rnd() - 0.5) * 0.9
        px = max(5.0, px + drift)
        o = px
        c = px + (rnd() - 0.5) * 0.6
        h = max(o, c) + rnd() * 0.4
        low = min(o, c) - rnd() * 0.4
        q = lambda v: round(round(v / tick) * tick, 10)  # noqa: E731
        bars.append({"open": q(o), "high": q(h), "low": q(low),
                     "close": q(c), "volume": 1000.0})
        px = c
    return bars


def _load_pack_module(name):
    """Import a pack file the way pack_registry does (bare module name,
    so `from indicator_incremental import X` resolves)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, _PACK / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_INC = _load_pack_module("indicator_incremental")
_INTERP = _load_pack_module("interpreter")
PineInc = _INC.SrChannelsPineIncremental
LegacyInc = _INC.SrChannelsIncremental

MANIFEST = json.loads((_PACK / "manifest.json").read_text())

# The four fields the Pine itself computes. Our `src_nearest_*` columns
# have no Pine counterpart (audit §1.9), so they are not part of the
# equivalence contract — these are.
PINE_FIELDS = ("in_channel", "res_broken", "sup_broken", "num_channels")


def run_pair(bars, tie="left_strict"):
    ours = PineInc(pivot_tie=tie)
    ref = PineSRChannels(pivot_tie=tie)
    out_ours, out_ref = [], []
    for b in bars:
        out_ours.append(ours.update_bar(b))
        out_ref.append(ref.update_bar(b))
    return ours, ref, out_ours, out_ref


# =====================================================================
# THE GATE — behavioural equivalence with the Pine
# =====================================================================

@pytest.mark.parametrize("seed", [7, 13, 21])
@pytest.mark.parametrize("tie", ["left_strict", "right_strict"])
def test_pine_variant_is_bar_for_bar_identical_to_the_transliteration(seed, tie):
    """THE deliverable. Every bar, every Pine-derived field, exact."""
    bars = make_bars(seed=seed)
    _, _, ours, ref = run_pair(bars, tie=tie)

    mismatches = []
    for i, (a, b) in enumerate(zip(ours, ref)):
        for f in PINE_FIELDS:
            if a["src_" + f] != b[f]:
                mismatches.append((i, f, a["src_" + f], b[f]))
    assert not mismatches, (
        f"seed={seed} tie={tie}: {len(mismatches)} bar/field mismatches vs "
        f"the Pine transliteration; first 20: {mismatches[:20]}"
    )

    # Non-vacuity: agreeing on "nothing ever happened" is not evidence.
    assert max(r["num_channels"] for r in ref) >= 2, "no channels formed"
    assert sum(r["in_channel"] for r in ref) > 10, "price never in a channel"
    assert sum(r["res_broken"] for r in ref) > 5, "no resistance breaks"
    assert sum(r["sup_broken"] for r in ref) > 5, "no support breaks"


@pytest.mark.parametrize("tick", [0.01, 0.05])
def test_equivalence_holds_on_coarser_ticks(tick):
    """Coarser quantisation = denser equal highs = the exact case the
    legacy pivot rule discarded (audit §2.1: 53-56% missed at $0.05)."""
    bars = make_bars(seed=13, tick=tick)
    _, _, ours, ref = run_pair(bars)
    for i, (a, b) in enumerate(zip(ours, ref)):
        assert a["src_in_channel"] == b["in_channel"], f"bar {i} tick={tick}"
        assert a["src_res_broken"] == b["res_broken"], f"bar {i} tick={tick}"
        assert a["src_sup_broken"] == b["sup_broken"], f"bar {i} tick={tick}"


def test_legacy_class_does_NOT_match_the_pine():
    """Guards the gate itself: if this ever passes, the differential
    test above has stopped measuring anything."""
    bars = make_bars(seed=7)
    legacy = LegacyInc()
    ref = PineSRChannels()
    diffs = 0
    for b in bars:
        a = legacy.update_bar(b)
        r = ref.update_bar(b)
        if (a["src_in_channel"] != r["in_channel"]
                or a["src_res_broken"] != r["res_broken"]
                or a["src_sup_broken"] != r["sup_broken"]):
            diffs += 1
    assert diffs > 0, (
        "the legacy class now matches the Pine — the measured gap that "
        "justified board #73 has vanished; re-run the audit")


# =====================================================================
# RAILS — each fails against the legacy class
# =====================================================================

def test_rail_2_4_channels_are_FROZEN_between_pivot_bars():
    """§2.4, the big one. Pine's `suportresistance` is `var` and is
    rewritten only inside `if bool(ph) or bool(pl)` (L88). The legacy
    class rebuilt every bar, so its levels slid under the price
    mid-cross."""
    bars = make_bars(seed=21)
    eng = PineInc()
    prev_sr = None
    changes = 0
    pivot_bars = 0
    for b in bars:
        before = list(eng._sr)
        before_pv = len(eng._pivotvals)
        eng.update_bar(b)
        after = list(eng._sr)
        if len(eng._pivotvals) != before_pv or eng._pivotlocs[:1] == [
                len(eng._closes) - 1 + eng._trimmed]:
            pivot_bars += 1
        elif after != before:
            changes += 1
        prev_sr = after
    assert prev_sr is not None
    assert changes == 0, (
        f"{changes} bars changed the channel set without confirming a "
        f"pivot — the set is not frozen")
    assert pivot_bars > 0, "no pivot bars in the fixture — test is vacuous"


def test_rail_2_4_legacy_drifts_between_pivot_bars():
    """The exact §2.4 claim, measured against the legacy class: its
    levels move on bars that confirmed NO new pivot. That is the drift
    the freeze removes, and it must reproduce or the rail above is
    guarding nothing."""
    bars = make_bars(seed=21)
    eng = LegacyInc()

    def pivot_count():
        return (sum(1 for v in eng._pivot_highs if v == v)
                + sum(1 for v in eng._pivot_lows if v == v))

    prev_top = float("nan")
    prev_pivots = 0
    drifted = 0
    non_pivot_bars = 0
    for b in bars:
        out = eng.update_bar(b)
        now_pivots = pivot_count()
        new_pivot = now_pivots != prev_pivots
        top = out["src_nearest_top"]
        if not new_pivot and prev_top == prev_top and top == top:
            non_pivot_bars += 1
            if top != prev_top:
                drifted += 1
        prev_pivots = now_pivots
        prev_top = top

    assert non_pivot_bars > 100, "fixture has too few non-pivot bars"
    assert drifted > 0, (
        "legacy levels never moved on a non-pivot bar — the drift §2.4 "
        "measured is not reproducing")


def test_rail_2_2_pivot_array_is_newest_first():
    """§2.2 — PINE L49 `array.unshift`. Order is load-bearing: the
    channel-growing loop in get_sr_vals walks the array in order."""
    bars = make_bars(seed=7)
    eng = PineInc()
    for b in bars:
        eng.update_bar(b)
    assert len(eng._pivotlocs) >= 2, "fixture produced too few pivots"
    assert eng._pivotlocs == sorted(eng._pivotlocs, reverse=True), (
        "pivotlocs is not strictly newest-first")


def test_rail_2_3_simultaneous_high_and_low_keeps_only_the_high():
    """§2.3 — PINE L49 `ph ? ph : pl` unshifts ONE value per bar. On
    Close/Open source a flat run makes both confirm on the same bar;
    the Pine drops the low."""
    prd = 4
    n = 2 * prd + 1
    # A flat plateau on close/open: every bar identical, so the centre
    # bar is both the (tie-tolerant) max and the min.
    bars = [{"open": 50.0, "high": 50.5, "low": 49.5, "close": 50.0,
             "volume": 1.0} for _ in range(n + 5)]
    eng = PineInc(pivot_period=prd, pivot_source="Close/Open",
                  pivot_tie="right_strict")
    ref = PineSRChannels(prd=prd, ppsrc="Close/Open",
                         pivot_tie="right_strict")
    for b in bars:
        eng.update_bar(b)
        ref.update_bar(b)
    assert eng._pivotvals == ref.pivotvals, (
        f"ours={eng._pivotvals} ref={ref.pivotvals}")
    # One entry per confirmation bar — never two.
    assert len(eng._pivotvals) == len(set(eng._pivotlocs)), (
        "more pivot values than confirmation bars — a high and a low "
        "were both kept for one bar")


def test_rail_2_1_pivot_set_is_a_superset_of_the_legacy_rule():
    """§2.1 — the legacy rule demanded a UNIQUE extremum over the whole
    2*prd+1 window, which rejects a superset of what TradingView
    rejects. The faithful rule must find at least as many pivots."""
    bars = make_bars(seed=13, tick=0.05)
    prd = 10

    # Count every pivot each RULE admits over the same bars and the same
    # windows — comparing the rules, not the two classes' bookkeeping.
    highs = [b["high"] for b in bars]
    lows = [b["low"] for b in bars]

    def legacy_admits(p, series, want_max):
        lo, hi = max(0, p - prd), min(len(series) - 1, p + prd)
        v = series[p]
        eq = 0
        for j in range(lo, hi + 1):
            if (series[j] > v) if want_max else (series[j] < v):
                return False
            if series[j] == v:
                eq += 1
        return eq == 1                      # UNIQUE extremum — the old rule

    def pine_admits(p, series, want_max, left_strict=True):
        if p - prd < 0 or p + prd > len(series) - 1:
            return False
        v = series[p]
        for j in range(p - prd, p):
            beats = series[j] > v if want_max else series[j] < v
            if beats or (left_strict and series[j] == v):
                return False
        for j in range(p + 1, p + prd + 1):
            beats = series[j] > v if want_max else series[j] < v
            if beats or ((not left_strict) and series[j] == v):
                return False
        return True

    legacy_set, pine_set = set(), set()
    for p in range(prd, len(bars) - prd):
        for series, want_max, tag in ((highs, True, "H"), (lows, False, "L")):
            if legacy_admits(p, series, want_max):
                legacy_set.add((p, tag))
            if pine_admits(p, series, want_max):
                pine_set.add((p, tag))

    assert pine_set, "no pivots at all — fixture is degenerate"
    assert legacy_set <= pine_set, (
        f"legacy admitted {len(legacy_set - pine_set)} pivots the faithful "
        f"rule rejects — the subset relation the audit proved has inverted")
    assert len(pine_set) > len(legacy_set), (
        f"faithful rule found no MORE pivots than legacy "
        f"({len(pine_set)} vs {len(legacy_set)}) — §2.1 is not reproducing")


def test_rail_2_5_no_channels_before_the_300_bar_warmup():
    """§2.5 — `ta.highest(300)` is `na` until 300 bars exist, so the
    Pine emits nothing. The legacy class used a partial window from bar
    0 and emitted channels the Pine never would."""
    bars = make_bars(n=299, seed=7)
    eng = PineInc()
    for b in bars:
        out = eng.update_bar(b)
        assert out["src_num_channels"] == 0.0
        assert out["src_in_channel"] == 0.0
        assert out["src_res_broken"] == 0.0
        assert out["src_sup_broken"] == 0.0
    assert PineInc.WARMUP_BARS == 300


def test_rail_2_5_legacy_emits_channels_during_warmup():
    """The divergence §2.5 named must actually reproduce."""
    bars = make_bars(n=299, seed=7)
    eng = LegacyInc()
    early = sum(1 for b in bars if eng.update_bar(b)["src_num_channels"] > 0)
    assert early > 0, "legacy no longer emits partial-window channels"


def test_rail_2_7_prev_level_equals_live_level_between_pivot_bars():
    """§2.7 — the C-type break and the L-type intra-bar cross used to
    fire off different levels. Freezing collapses that: on any
    non-pivot bar `*_prev` IS the level the break was evaluated
    against."""
    bars = make_bars(seed=7)
    eng = PineInc()
    outs, sizes = [], []
    for b in bars:
        outs.append(eng.update_bar(b))
        sizes.append(len(eng._pivotvals))
    checked = 0
    for i in range(1, len(outs)):
        if sizes[i] != sizes[i - 1]:
            continue  # pivot bar — the set legitimately moved
        a, b_ = outs[i]["src_nearest_top_prev"], outs[i - 1]["src_nearest_top"]
        if a != a and b_ != b_:
            continue  # both NaN
        assert a == b_, f"bar {i}: prev={a} live-1={b_}"
        checked += 1
    assert checked > 100, f"only {checked} bars checked — test is weak"


def test_greedy_pick_is_non_increasing_so_pine_sort_is_dead():
    """PINE L138-143 is a bubble sort that never fires (and is itself
    buggy). We omit it — this asserts the invariant that makes the
    omission safe, rather than assuming it."""
    for seed in (7, 13, 21):
        bars = make_bars(seed=seed)
        eng = PineInc()
        for b in bars:
            eng.update_bar(b)
            active = [s for s in eng._stren if s != 0.0]
            assert active == sorted(active, reverse=True), (
                f"seed={seed}: strengths not non-increasing: {eng._stren}")


def test_maxnumsr_matches_the_pine_off_by_one():
    """PINE L10 `input.int(defval=6) - 1`, then used as an INCLUSIVE
    loop bound. Net channel cap is min(10, max_num_sr)."""
    eng = PineInc(max_num_sr=6)
    assert eng.maxnumsr == 5
    assert min(9, eng.maxnumsr) + 1 == 6
    assert min(9, PineInc(max_num_sr=10).maxnumsr) + 1 == 10
    assert min(9, PineInc(max_num_sr=1).maxnumsr) + 1 == 1


# =====================================================================
# INTERPRETER — audit §2.6
# =====================================================================

def _interp_frame(bars, engine):
    import pandas as pd
    rows = []
    for b in bars:
        out = engine.update_bar(b)
        row = dict(out)
        row["close"] = b["close"]
        rows.append(row)
    return pd.DataFrame(rows)


def test_rail_2_6_pine_interpreter_never_asserts_direction_inside_a_channel():
    """§2.6 — the Pine's own colour rule (L158) says a channel is
    NEITHER support nor resistance while price sits inside it (grey
    inch_col). The legacy interpreter split that case on the midpoint."""
    bars = make_bars(seed=21)
    df = _interp_frame(bars, PineInc())
    states = _INTERP.interpret_sr_channels_pine(df)
    inside = df["src_in_channel"] == 1.0
    assert inside.sum() > 0, "fixture never puts price inside a channel"
    assert set(states[inside].dropna()) == {"IN_CHANNEL"}
    assert "IN_RESISTANCE" not in set(states.dropna())
    assert "IN_SUPPORT" not in set(states.dropna())


def test_rail_2_6_legacy_interpreter_does_assert_direction():
    """The contradiction §2.6 named must reproduce."""
    bars = make_bars(seed=21)
    df = _interp_frame(bars, LegacyInc())
    states = _INTERP.interpret_sr_channels(df)
    assert {"IN_RESISTANCE", "IN_SUPPORT"} & set(states.dropna())


def test_rail_2_6_above_resistance_means_above_EVERY_channel():
    """§2.6 — legacy computed it against the nearest channel only, so
    price could be 'ABOVE_RESISTANCE' beneath five stronger channels."""
    bars = make_bars(seed=7)
    df = _interp_frame(bars, PineInc())
    states = _INTERP.interpret_sr_channels_pine(df)
    above = states == "ABOVE_RESISTANCE"
    assert above.sum() > 0, "fixture never puts price above every channel"
    assert (df.loc[above, "src_ch_below"]
            == df.loc[above, "src_num_channels"]).all()
    below = states == "BELOW_SUPPORT"
    if below.sum():
        assert (df.loc[below, "src_ch_above"]
                == df.loc[below, "src_num_channels"]).all()


# =====================================================================
# THE FLAG — default OFF, and OFF is a strict no-op
# =====================================================================

def test_flag_helper_is_runtime_read(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert pack_registry.env_flag_on(FLAG) is False
    monkeypatch.setenv(FLAG, "1")
    assert pack_registry.env_flag_on(FLAG) is True
    monkeypatch.setenv(FLAG, "0")
    assert pack_registry.env_flag_on(FLAG) is False
    monkeypatch.setenv(FLAG, "")
    assert pack_registry.env_flag_on(FLAG) is False


def test_flag_off_selects_nothing(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert pack_registry.select_flag_variant(MANIFEST) is None


def test_flag_on_selects_the_pine_variant(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    v = pack_registry.select_flag_variant(MANIFEST)
    assert v is not None
    assert v["incremental_class"] == "SrChannelsPineIncremental"
    assert v["indicator_function"] == "calculate_sr_channels_pine"
    assert v["interpreter_function"] == "interpret_sr_channels_pine"


def test_flag_off_load_is_a_strict_no_op(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    pack = pack_registry.load_single_pack(_PACK)
    assert pack.is_valid, pack.validation_errors
    assert pack.incremental_class.__name__ == "SrChannelsIncremental"
    assert pack.indicator_func.__name__ == "calculate_sr_channels"
    assert pack.interpreter_func.__name__ == "interpret_sr_channels"
    assert "IN_RESISTANCE" in pack.manifest["outputs"]
    assert "src_ch_above" not in pack.manifest["indicator_columns"]


def test_flag_on_load_swaps_callables_and_manifest(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    pack = pack_registry.load_single_pack(_PACK)
    assert pack.is_valid, pack.validation_errors
    assert pack.incremental_class.__name__ == "SrChannelsPineIncremental"
    assert pack.indicator_func.__name__ == "calculate_sr_channels_pine"
    assert pack.interpreter_func.__name__ == "interpret_sr_channels_pine"
    # trigger_function is not overridden — it is shared, verbatim.
    assert pack.trigger_func.__name__ == "detect_sr_channels_triggers"
    assert pack.manifest["outputs"] == [
        "ABOVE_RESISTANCE", "IN_CHANNEL", "BETWEEN_LEVELS", "BELOW_SUPPORT"]
    assert "src_ch_above" in pack.manifest["indicator_columns"]


def test_flag_on_does_not_mutate_the_on_disk_manifest(monkeypatch):
    """A flag flip must not leak into an unarmed process."""
    monkeypatch.setenv(FLAG, "1")
    pack_registry.load_single_pack(_PACK)
    monkeypatch.delenv(FLAG, raising=False)
    pack = pack_registry.load_single_pack(_PACK)
    assert "IN_RESISTANCE" in pack.manifest["outputs"]


def test_armed_variant_naming_a_missing_symbol_is_a_hard_error(monkeypatch):
    """feedback_no_silent_defaults — an armed flag must never fall back
    to the legacy implementation in silence."""
    monkeypatch.setenv("RORT_SR_CHANNELS_BOGUS", "1")
    errors = []
    fake = {"flag_variants": [{"env": "RORT_SR_CHANNELS_BOGUS",
                              "incremental_class": "NoSuchClass"}]}
    out = pack_registry._apply_flag_variant(
        fake, errors, "sr_channels",
        ind_module=None, interp_module=None, inc_module=_INC,
        indicator_func=None, interpreter_func=None,
        trigger_func=None, incremental_class=LegacyInc)
    assert errors, "missing symbol resolved silently"
    assert out[4] is LegacyInc


# =====================================================================
# BATCH == INCREMENTAL (live/backtest parity for the new variant)
# =====================================================================

def test_pine_batch_matches_pine_incremental():
    import pandas as pd
    ind = _load_pack_module("indicator")
    bars = make_bars(seed=13)
    df = pd.DataFrame(bars)
    out = ind.calculate_sr_channels_pine(df)
    eng = PineInc()
    for i, b in enumerate(bars):
        step = eng.update_bar(b)
        for k in ("src_num_channels", "src_in_channel", "src_res_broken",
                  "src_sup_broken", "src_ch_above", "src_ch_below"):
            assert out[k].iloc[i] == step[k], f"bar {i} col {k}"
