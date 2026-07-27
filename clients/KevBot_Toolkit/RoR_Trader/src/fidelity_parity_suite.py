"""Fidelity parity suite — the regression gate for fidelity-impacting changes.

WHY THIS EXISTS
---------------
Backtest≈live (same trades within ~5s) is the project's north star. Speed/cache
work keeps almost touching the data-load + engine paths that decide trades, and a
silent shift there moves us away from that north star — hard to find our way back.
On 2026-06-26 the REST Bars cache upsampled 1Min→sub-minute for uncached TFs and
wrecked 10Sec/15Sec backtests (193→29 trades) — and slipped through because the
validation only checked a *cached* TF (30Sec) and a cache-*off* run, never the
uncached-sub-minute path that actually broke.

This suite closes that gap with two checks. Run it BEFORE trusting any change that
touches data loading, resampling, warmup, the cache, or the engine — and any time
you flip a fidelity kill-switch (BAR_CACHE_ENABLED, RORT_RIGHTSIZE_WARMUP, …).

  1. CACHE PARITY MATRIX — `cached_load_market_data(sym, tf)` must be byte-identical
     to native `load_from_polygon(sym, tf)` for EVERY (symbol, TF), INCLUDING the
     uncached sub-minute combos (the exact 2026-06-26 gap). OHLCV, both sessions.

  2. NORTH-STAR CANARY KILL-SWITCH PARITY — a designated strong-divergence strategy
     (sid 267: confluence + entry/exit trigger, ~90% combined, 10Sec) must produce
     BYTE-IDENTICAL trades with each fidelity flag ON vs OFF. If the canary's
     backtest trades don't change, its backtest↔alert divergence can't change — so
     this protects the north star by construction. Pick a canary that has a
     confluence AND a trigger AND strong performance, so a regression has somewhere
     to show up. Re-pick (or add) canaries as the fleet evolves; keep at least one.

Exit code is non-zero on ANY mismatch, so it can gate CI / a deploy.

USAGE
-----
    python src/fidelity_parity_suite.py                 # full suite
    python src/fidelity_parity_suite.py --quick         # cache matrix only (no engine)
    python src/fidelity_parity_suite.py --day 2026-06-16 # pin the settled comparison day

NOTE: not yet hermetic — it compares the cache vs a live Polygon pull on a settled
day (deterministic for settled data). A golden-fixture version (recorded bars +
expected trades) would let this run with no Polygon/DB; tracked as a follow-up.
"""
import os
import sys
import argparse
import warnings
import datetime as dt

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, "src")

# ── Configuration ──────────────────────────────────────────────────────────────
NORTH_STAR_CANARY = 267  # TSLA-CANARY-10s-LooseConf: confluence + trigger, ~90%, 10Sec, fast
ADMIN_USER = "19d47e46-f718-49a6-af32-5f5407f5b170"

# Coarse-secondary canary: a 1Day-gated strategy (price-only SWING_123) used to
# assert RORT_COARSE_SECONDARY_FROM_1MIN (build the coarse gate from 1Min + short
# primary warmup) is fidelity-safe vs the legacy resample-from-primary path.
COARSE_SECONDARY_CANARY = 338  # TEST-COARSEGATE-10s-1d-TSLA
COARSE_SECONDARY_TOL = 3       # shares M-RS1's warmup-edge class (fix shrinks primary warmup)

# (symbol, timeframe) combos to assert cache==native. MUST include uncached
# sub-minute (the 2026-06-26 bug class) + cached-native + coarse + 1Sec.
CACHE_PARITY_MATRIX = [
    ("TSLA", "1Sec"),    # native fine layer
    ("TSLA", "10Sec"),   # UNCACHED sub-minute → must derive from 1Sec (the bug)
    ("TSLA", "15Sec"),   # UNCACHED sub-minute → must derive from 1Sec (the bug)
    ("TSLA", "30Sec"),   # cached-native sub-minute
    ("TSLA", "1Min"),    # native coarse layer
    ("TSLA", "5Min"),    # coarse → downsample from 1Min
    ("SPY",  "10Sec"),   # UNCACHED sub-minute, different symbol
    ("SPY",  "1Min"),    # native coarse
]

# Fidelity kill-switches to assert ~ON==OFF on the canary. Each:
# (env_var, on_value, max_trade_count_diff).
#   tol=0  → MUST be byte-identical (pure optimizations: the cache shifts WHERE
#            bars come from, never WHICH trades fire). Any diff = a regression.
#   tol>0  → a documented small tradeoff is acceptable (RORT_RIGHTSIZE_WARMUP
#            trades a hair of warmup-edge accuracy for speed — observed ≤1
#            boundary trade on some strategies; accepted when M-RS1 shipped).
KILL_SWITCHES = [
    ("BAR_CACHE_ENABLED", "1", 0),
    ("RORT_RIGHTSIZE_WARMUP", "1", 3),
]

PRICE_TOL = 1e-4  # a hundredth of a cent — below this is float-storage noise, not a real diff

# Symbols the data-worker streams write-through into bar_cache (the 2026-06-29
# burn-in set). Keep in sync with the Bar Cache Supply registry as coverage grows
# (Design_BarCache_WriteThrough_Unification §3.5). The write-through gate validates
# each of these against a fresh Polygon pull on settled bars.
WRITETHROUGH_SYMBOLS = ["TSLA", "SPY", "DIA", "KO", "TSLL"]
WRITETHROUGH_TFS = ["1Sec", "1Min"]  # the two NATIVE source layers the stream persists


def _settled_test_day(override: str | None) -> dt.date:
    """A settled TRADING day that actually has data — not a weekend OR a holiday.
    Candidate walk lives in settle_sweeper.iter_settled_candidates (shared with
    the nightly retrue's pointer so the two can never target different days,
    #125): legacy = first weekday >= 3 calendar days back; with
    RORT_PARITY_SETTLED_MIN_TDAYS=N, first weekday with N full trading days
    elapsed (3 calendar days is only T+2 TRADING days across a weekend — inside
    Polygon's revision-churn window, the #103/#116/#125 false-red class). Picks
    the first candidate with substantial TSLA 1Sec cache coverage (proves it's
    a real trading day with data). Avoids the silent holiday skip that made an
    earlier version 'pass' while testing nothing."""
    if override:
        return dt.date.fromisoformat(override)
    import bar_cache as bc
    from settle_sweeper import iter_settled_candidates
    for d in iter_settled_candidates(dt.date.today()):
        s = dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc)
        df = bc.read_bars("TSLA", "1Sec", s, s + dt.timedelta(days=1))
        if df is not None and len(df) > 10000:  # a real trading day's worth of 1s bars
            return d
    raise RuntimeError("no settled trading day with cache data found in the candidate window")


def _self_heal_enabled() -> bool:
    """#125 (V1.17): RORT_PARITY_SELF_HEAL=1 lets a cache-parity item that fails
    with the revision-drift signature (pure OHLCV value diffs, rowMM=0, on the
    settled day) re-true that symbol/day from Polygon REST and re-compare ONCE.
    Rationale: the cache is a snapshot of Polygon-at-last-true-up; Polygon keeps
    revising a day into its 3rd trading day after (measured #125), so no
    scheduled heal can guarantee equality at an arbitrary later read — the only
    deterministic point to true-up is comparison time. A REAL pipeline bug still
    fails: diffs that survive a fresh REST true-up are ours, not Polygon's.
    Heals via settle_sweeper.retrue_range (same op as the manual SOP); WRITES to
    bar_cache, hence opt-in and default OFF."""
    return os.environ.get("RORT_PARITY_SELF_HEAL", "").strip().lower() in (
        "1", "true", "yes", "on")


def _cache_parity(day: dt.date) -> list[tuple[str, bool, str]]:
    """cached_load_market_data == native load_from_polygon, byte-identical, every (sym,TF)."""
    import bar_cache as bc
    import data_loader as dl
    os.environ["BAR_CACHE_ENABLED"] = "1"
    s = dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc)
    e = s + dt.timedelta(days=1) - dt.timedelta(seconds=1)
    results = []
    healed_syms: set[str] = set()  # one retrue heals ALL of a symbol's TF items

    def _compare(sym, tf, sess, nat):
        cached = bc.cached_load_market_data(sym, timeframe=tf, start_date=s, end_date=e, session=sess)
        if cached is None:
            return None, None, cached
        cols = ["open", "high", "low", "close", "volume"]
        j = nat[cols].join(cached[cols], lsuffix="_n", rsuffix="_c", how="inner")
        diffs = sum(int(((j[c + "_n"] - j[c + "_c"]).abs() > PRICE_TOL).sum()) for c in cols)
        rowmm = abs(len(nat) - len(cached))
        return diffs, rowmm, cached

    for sym, tf in CACHE_PARITY_MATRIX:
        for sess in ("24/7", "RTH"):
            label = f"cache-parity {sym}/{tf}/{sess}"
            try:
                nat = dl.load_from_polygon(sym, timeframe=tf, start_date=s.replace(hour=0), end_date=s.replace(hour=0), session=sess)
                nat = nat[(nat.index >= s) & (nat.index <= e)] if nat is not None else None
                if nat is None or len(nat) == 0:
                    results.append((label, "SKIP", "no native data"))
                    continue
                diffs, rowmm, cached = _compare(sym, tf, sess, nat)
                if cached is None:
                    results.append((label, "FAIL", f"cache returned None, native has {len(nat)}"))
                    continue
                detail = f"native={len(nat)} cached={len(cached)} OHLCVdiffs={diffs} rowMM={rowmm}"
                # #125: revision-drift signature (value diffs only, no row gap)
                # → true-up this symbol's day and re-compare once. The re-compare
                # is against the SAME native pull, so a heal can't paper over a
                # row-count hole or a real value bug (those survive the retrue).
                if diffs and rowmm == 0 and _self_heal_enabled():
                    if sym not in healed_syms:
                        import settle_sweeper
                        os.environ.setdefault("RORT_BARCACHE_WRITETHROUGH", "1")
                        settle_sweeper.retrue_range(sym, s, s)
                        healed_syms.add(sym)
                    diffs2, rowmm2, cached2 = _compare(sym, tf, sess, nat)
                    if cached2 is not None and diffs2 == 0 and rowmm2 == 0:
                        results.append((label, "PASS",
                                        f"native={len(nat)} cached={len(cached2)} OHLCVdiffs=0 rowMM=0 "
                                        f"| self-healed revision drift (was OHLCVdiffs={diffs})"))
                        continue
                    detail += f" | self-heal attempted, still OHLCVdiffs={diffs2} rowMM={rowmm2}"
                status = "PASS" if (diffs == 0 and rowmm == 0) else "FAIL"
                results.append((label, status, detail))
            except Exception as ex:  # noqa: BLE001
                results.append((label, "FAIL", f"EXC {str(ex)[:120]}"))
    return results


def _canary_killswitch_parity() -> list[tuple[str, bool, str]]:
    """Canary trades must be byte-identical with each fidelity flag ON vs OFF."""
    import importlib
    from db import get_strategy_by_id_admin
    _on_vals = {f: v for f, v, _ in KILL_SWITCHES}

    def _trades(test_flag, on):
        # Clean prod-like baseline (all switches ON), then override ONLY the test
        # flag — so a flag's effect is isolated, not contaminated by a prior loop.
        for f, v, _ in KILL_SWITCHES:
            os.environ[f] = v
        if on:
            os.environ[test_flag] = _on_vals[test_flag]
        else:
            os.environ.pop(test_flag, None)
        import data_loader, services, bar_cache
        importlib.reload(bar_cache); importlib.reload(data_loader); importlib.reload(services)
        if hasattr(data_loader, "clear_1s_cache"):
            data_loader.clear_1s_cache()
        res = services.get_strategy_trades(get_strategy_by_id_admin(NORTH_STAR_CANARY, ADMIN_USER))
        df = res[0] if isinstance(res, tuple) else res
        if df is None or len(df) == 0:
            return 0, frozenset()
        col = "entry_fill_ts" if "entry_fill_ts" in df.columns else "entry_time"
        return len(df), frozenset(df[col].astype(str))

    results = []
    for flag, _on, tol in KILL_SWITCHES:
        label = f"canary {NORTH_STAR_CANARY} parity: {flag} ON==OFF (tol={tol})"
        try:
            n_off, s_off = _trades(flag, False)
            n_on, s_on = _trades(flag, True)
            symdiff = len(s_off ^ s_on)
            countdiff = abs(n_off - n_on)
            ok = (symdiff == 0) if tol == 0 else (countdiff <= tol)
            detail = f"OFF={n_off} ON={n_on} symdiff={symdiff} countdiff={countdiff}"
            if n_off == 0:
                results.append((label, "FAIL", f"{detail} (canary produced 0 trades — registry/window broken)"))
            else:
                results.append((label, "PASS" if ok else "FAIL", detail))
        except Exception as ex:  # noqa: BLE001
            results.append((label, "FAIL", f"EXC {str(ex)[:120]}"))
    return results


def _coarse_secondary_parity() -> list[tuple[str, bool, str]]:
    """A coarse (1Day) gated strategy's trades must match with the coarse-
    secondary source flipped: RORT_COARSE_SECONDARY_FROM_1MIN OFF (resample the
    coarse gate from the sub-minute PRIMARY — the ~360x-blowup legacy path that
    hung sid 338's UAD) vs ON (build it from 1Min + short primary warmup). For a
    price-only daily gate the secondary OHLC is byte-identical (volume aside, the
    documented 1Min!=Σsub-minute divergence — irrelevant to price gates); the
    only trade-shift vector is the primary-warmup shrink, same class as
    RORT_RIGHTSIZE_WARMUP → tol. Runs under the prod baseline (rightsize+cache
    ON). NOTE: the OFF leg loads ~363d of the 10Sec primary → slow (minutes)."""
    import importlib
    from db import get_strategy_by_id_admin

    def _trades(on):
        os.environ["RORT_RIGHTSIZE_WARMUP"] = "1"
        os.environ["BAR_CACHE_ENABLED"] = "1"
        if on:
            os.environ["RORT_COARSE_SECONDARY_FROM_1MIN"] = "1"
        else:
            os.environ.pop("RORT_COARSE_SECONDARY_FROM_1MIN", None)
        import data_loader, services, bar_cache, strategy_data
        importlib.reload(bar_cache); importlib.reload(data_loader)
        importlib.reload(strategy_data); importlib.reload(services)
        if hasattr(data_loader, "clear_1s_cache"):
            data_loader.clear_1s_cache()
        res = services.get_strategy_trades(
            get_strategy_by_id_admin(COARSE_SECONDARY_CANARY, ADMIN_USER))
        df = res[0] if isinstance(res, tuple) else res
        if df is None or len(df) == 0:
            return 0, frozenset()
        col = "entry_fill_ts" if "entry_fill_ts" in df.columns else "entry_time"
        return len(df), frozenset(df[col].astype(str))

    label = (f"coarse-secondary canary {COARSE_SECONDARY_CANARY} parity: "
             f"RORT_COARSE_SECONDARY_FROM_1MIN ON==OFF (tol={COARSE_SECONDARY_TOL})")
    try:
        n_off, s_off = _trades(False)
        n_on, s_on = _trades(True)
        symdiff = len(s_off ^ s_on)
        countdiff = abs(n_off - n_on)
        ok = countdiff <= COARSE_SECONDARY_TOL
        detail = f"OFF={n_off} ON={n_on} symdiff={symdiff} countdiff={countdiff}"
        if n_off == 0:
            return [(label, "FAIL", f"{detail} (0 trades — registry/window broken)")]
        return [(label, "PASS" if ok else "FAIL", detail)]
    except Exception as ex:  # noqa: BLE001
        return [(label, "FAIL", f"EXC {str(ex)[:120]}")]


def _writethrough_parity(day: dt.date) -> list[tuple[str, str, str]]:
    """The bars the data-worker WROTE THROUGH into bar_cache must be byte-identical
    to a fresh Polygon REST pull, on SETTLED bars (§6 #1 — write-through must not
    become a divergence source).

    Reads the bar_cache table DIRECTLY (`bc.read_bars`, no delta-fetch) so it
    validates what the stream actually persisted — NOT a read-path backfill that
    would mask a missing-bar gap. Only settled bars are checked: bars older than
    the settle window (RORT_SHADOW_SETTLE_MIN, default 16) are permanent truth; the
    unsettled tail is provisional by design and excluded. For TODAY the window is
    [open, now-settle-pad]; for a past day the whole day is settled.

    Volume IS compared here (unlike the resample caveat): the stream persists NATIVE
    1Sec + 1Min, and `load_from_polygon` returns the same native bars, so volume must
    match exactly — that is precisely the property Step 1 said to protect."""
    import bar_cache as bc
    import data_loader as dl
    os.environ["BAR_CACHE_ENABLED"] = "1"
    s = dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc)
    pad = dt.timedelta(minutes=bc._settle_min() + 5)
    now = dt.datetime.now(dt.timezone.utc)
    e_settled = min(s + dt.timedelta(days=1) - dt.timedelta(seconds=1), now - pad)
    if e_settled <= s:
        return [(f"writethrough {day} (no settled window yet)", "SKIP", "too early in the day")]
    results = []
    for sym in WRITETHROUGH_SYMBOLS:
        for tf in WRITETHROUGH_TFS:
            label = f"writethrough {sym}/{tf}/RTH"
            try:
                wrote = bc.read_bars(sym, tf, s, e_settled)  # raw table read — what the stream wrote
                nat = dl.load_from_polygon(sym, timeframe=tf, start_date=s.replace(hour=0),
                                           end_date=s.replace(hour=0), session="RTH")
                if nat is not None:
                    nat = nat[(nat.index >= s) & (nat.index <= e_settled)]
                if nat is None or len(nat) == 0:
                    results.append((label, "SKIP", "no native data"))
                    continue
                if wrote is None or len(wrote) == 0:
                    results.append((label, "FAIL",
                                    f"bar_cache empty, native={len(nat)} — stream not writing {sym}/{tf}?"))
                    continue
                wrote = dl._filter_session(wrote, "RTH")
                wrote = wrote[(wrote.index >= s) & (wrote.index <= e_settled)]
                cols = ["open", "high", "low", "close", "volume"]
                # Two distinct signals, not one. (a) WITHIN the window the stream
                # actually covered ([first-wrote-bar, e_settled]), native must equal
                # wrote byte-identical — holes or value diffs HERE mean the stream is
                # a divergence source = real FAIL. (b) A LEADING gap (native bars
                # before the stream's first written bar) is missing *history*, not a
                # streaming bug — happens when write-through is enabled mid-session or
                # a symbol is newly streamed. That's a backfill task (blocks
                # BAR_CACHE_ENABLED for the symbol per §5), surfaced as WARN not FAIL.
                cover_start = wrote.index.min()
                lead_gap = int((nat.index < cover_start).sum())
                nat_cov = nat[nat.index >= cover_start]
                j = nat_cov[cols].join(wrote[cols], lsuffix="_n", rsuffix="_c", how="inner")
                diffs = sum(int(((j[c + "_n"] - j[c + "_c"]).abs() > PRICE_TOL).sum()) for c in cols)
                rowmm = abs(len(nat_cov) - len(wrote))
                within_ok = (diffs == 0 and rowmm == 0)
                detail = (f"native={len(nat)} wrote={len(wrote)} | within-coverage: "
                          f"OHLCVdiffs={diffs} rowMM={rowmm} | lead_gap={lead_gap}")
                if not within_ok:
                    results.append((label, "FAIL", detail + "  ← holes/diffs WITHIN coverage (stream bug)"))
                elif lead_gap > 0:
                    results.append((label, "WARN", detail +
                                    f"  ← faithful, but {lead_gap} bars of pre-coverage history missing "
                                    f"(backfill before BAR_CACHE_ENABLED)"))
                else:
                    results.append((label, "PASS", detail))
            except Exception as ex:  # noqa: BLE001
                results.append((label, "FAIL", f"EXC {str(ex)[:120]}"))
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="cache matrix only (skip the engine canary)")
    ap.add_argument("--coarse", action="store_true",
                    help="also run the coarse-secondary canary parity (slow: ~363d primary load)")
    ap.add_argument("--writethrough", action="store_true",
                    help="also run the write-through parity gate (bar_cache stream writes == Polygon, "
                         "settled bars). Defaults to TODAY (where the stream's data lives); --day overrides.")
    ap.add_argument("--day", default=None, help="settled comparison day YYYY-MM-DD")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv("src/.env", override=True)
    os.environ["USE_DB"] = "true"
    os.environ.setdefault("BAR_CACHE_PAGE_SIZE", "50000")
    from db import set_admin_user_context
    set_admin_user_context(ADMIN_USER)
    import pack_registry
    pack_registry.scan_and_load_all()

    day = _settled_test_day(args.day)
    print(f"=== FIDELITY PARITY SUITE (settled day {day}, north-star canary {NORTH_STAR_CANARY}) ===")

    results = []
    results += _cache_parity(day)
    if not args.quick:
        results += _canary_killswitch_parity()
    if args.coarse:
        results += _coarse_secondary_parity()
    if args.writethrough:
        wt_day = dt.date.fromisoformat(args.day) if args.day else dt.date.today()
        print(f"--- write-through gate: settled bars on {wt_day} ---")
        results += _writethrough_parity(wt_day)

    print()
    for label, status, detail in results:
        print(f"  [{status}] {label:42} {detail}")
    print()
    failures = sum(1 for _, s, _ in results if s == "FAIL")
    passes = sum(1 for _, s, _ in results if s == "PASS")
    skips = sum(1 for _, s, _ in results if s == "SKIP")
    warns = sum(1 for _, s, _ in results if s == "WARN")
    if failures:
        print(f"❌ {failures} FAILED ({passes} pass, {warns} warn, {skips} skip) — fidelity regression. Do NOT ship.")
        return 1
    if passes == 0:
        print(f"❌ 0 real checks ran ({skips} skipped — no data). The suite tested NOTHING; "
              f"fix the comparison day so the bug class is actually exercised.")
        return 1
    msg = f"✅ all {passes} checks passed — cache byte-identical across TFs; canary trades stable under flags."
    if warns:
        msg += (f"  ⚠️ {warns} WARN (faithful but incomplete coverage — backfill needed; not a regression).")
    if skips:
        msg += f"  ({skips} skipped for no data)"
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
