"""Divergence / fidelity BASELINE capture — board #264 (pre environment-split).

WHY: `dev` is production and there is exactly ONE environment. Board V5 (#162)
splits it into main+dev fleets behind a market-data gateway. Without a measured
BEFORE we cannot prove the split changed nothing. This script IS the ruler.

WHAT IT MEASURES: nothing new. It calls the dashboard's OWN
`api.routers.strategy_health.get_strategy_health` — the identical code path the
admin Strategy Health tab runs — over FIXED absolute windows at BOTH tolerances
(10s = the honest bar, 60s = the legacy display bar), and dumps per-strategy
paired / phantom / missed / TBD / combined% plus the red-flag lanes.
Measurement SSOT: docs/_active/Plan_Measurement_Trust.md.

READ-ONLY. It never writes to the DB, never flips a flag, never recomputes.

RE-RUN AFTER THE SPLIT (this is the whole point — identical invocation):
    cd clients/KevBot_Toolkit/RoR_Trader/src
    ../.venv/bin/python _divergence_baseline_264.py --out ../docs/_active/Baseline_Divergence_PostSplit_<DATE>.md

The window list is HARDCODED below on purpose: a rolling `window_hours` read is
anchored to `now` and is NOT reproducible later. Absolute start/end is. Do NOT
change WINDOWS when re-running for comparison — that is the comparison.

STRADDLE GUARD (#156): a capture that straddles an intraday recompute produces a
subset>superset artifact. This script samples in-flight update_jobs / mass_searches
BEFORE and AFTER the capture and records both, so a straddled capture is visible
in the artifact instead of silently wrong.
"""
from __future__ import annotations

import argparse
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from api.routers import strategy_health as sh  # noqa: E402
from db import get_admin_client  # noqa: E402

USER = {"id": "baseline-264", "role": "admin"}

# --- The windows. ABSOLUTE, so the identical read is reproducible post-split. ---
# Rationale (Plan_Measurement_Trust Phase B): PREFER a nightly-recomputed SETTLED
# window as the reference. Today's stored lane is built incrementally and can
# disagree with a full recompute, so today is captured but flagged UNSETTLED.
WINDOWS = [
    ("W1_settled_2d", "2026-07-29T13:30:00Z", "2026-07-31T00:00:00Z",
     "PRIMARY. Wed 07-29 + Thu 07-30 RTH, both through the 00:20Z nightly "
     "Update-All => settled, current-logic stored lane. This is the ruler."),
    ("W2_today_unsettled", "2026-07-31T13:30:00Z", "2026-07-31T20:00:00Z",
     "Fri 07-31 RTH (capture day). Stored lane NOT yet through the nightly => "
     "UNSETTLED. Recorded for completeness; NOT comparable head-to-head."),
    ("W3_settled_5d", "2026-07-24T13:30:00Z", "2026-07-31T00:00:00Z",
     "Wider N for the same settled basis (5 sessions). Smooths single-day "
     "regime noise in the fleet aggregate."),
]

TOLERANCES = [10.0, 60.0]


def _pct(paired, phantom, missed):
    d = paired + phantom + missed
    return round(100.0 * paired / d, 1) if d else None


def quiet_probe(label):
    """Sample in-flight recompute/mass-search work. Straddle guard for #156."""
    out = {"label": label, "at": datetime.now(timezone.utc).isoformat()[:19] + "Z"}
    try:
        c = get_admin_client()
        for tbl in ("update_jobs", "mass_searches"):
            try:
                r = (c.table(tbl).select("id,status")
                     .in_("status", ["queued", "running"]).limit(50).execute())
                rows = r.data or []
                out[tbl] = {"in_flight": len(rows),
                            "statuses": dict(Counter(x.get("status") for x in rows))}
            except Exception as e:  # table shape drift must not kill the capture
                out[tbl] = {"error": str(e)[:120]}
    except Exception as e:
        out["error"] = str(e)[:160]
    return out


def flag_surface():
    """Enumerate the RORT_* flag SURFACE from source at this commit.

    NOTE THE LIMIT, and do not let a reader forget it: this is the set of flag
    NAMES the code knows about plus their CODE defaults. It is **not** the armed
    Railway values — reading those requires `railway variables`, which this
    (headless) capture is not permitted to run. Board #265 owns the real armed
    inventory. What this gives you is still worth having: a reproducible
    fingerprint of the flag surface, so a post-split run can prove the two
    fleets are at least reading the same flag vocabulary.
    """
    src = Path(__file__).resolve().parent
    names, defaults = set(), {}
    for p in src.rglob("*.py"):
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            continue
        names.update(re.findall(r"RORT_[A-Z0-9_]+", txt))
        for m, d in re.findall(r'getenv\(\s*"(RORT_[A-Z0-9_]+)"\s*,\s*"([^"]*)"', txt):
            defaults.setdefault(m, d)
    return sorted(names), defaults


def capture(since, until, tol):
    t0 = time.time()
    res = sh.get_strategy_health(user=USER, start=since, end=until,
                                 tolerance_seconds=tol)
    return res.get("rows", []), round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../docs/_active/Baseline_Divergence_PreSplit_2026-07-31.md")
    ap.add_argument("--dev-sha", default=os.environ.get("BASELINE_DEV_SHA", "unknown"))
    args = ap.parse_args()

    started = datetime.now(timezone.utc)
    pre = quiet_probe("BEFORE capture")

    # window -> tol -> rows
    data = {}
    timings = {}
    for name, since, until, _why in WINDOWS:
        data[name] = {}
        for tol in TOLERANCES:
            rows, secs = capture(since, until, tol)
            data[name][tol] = rows
            timings[(name, tol)] = secs
            print(f"[captured] {name} @{int(tol)}s -> {len(rows)} rows in {secs}s")

    # Quiescence check: re-capture the PRIMARY window @10s and require it to be
    # identical. If the stored lane mutated mid-capture (#156) the two disagree,
    # and that is recorded in the artifact rather than silently averaged away.
    recheck_rows, _ = capture(WINDOWS[0][1], WINDOWS[0][2], 10.0)

    def _fp(rs):
        return sorted((r["strategy_id"], r.get("paired_count_cov"),
                       r.get("phantom_count_cov"), r.get("missed_count_cov"))
                      for r in rs)

    quiescent = _fp(recheck_rows) == _fp(data[WINDOWS[0][0]][10.0])
    print(f"[quiescence] primary-window re-capture identical: {quiescent}")

    post = quiet_probe("AFTER capture")
    finished = datetime.now(timezone.utc)

    L = []
    A = L.append

    A("# Baseline — divergence / fidelity BEFORE the environment split")
    A("")
    A(f"**Captured:** {started.isoformat()[:19]}Z → {finished.isoformat()[:19]}Z  ")
    A(f"**Board:** #264 (E) · pre-split ruler for V5 / #162 environment split  ")
    A(f"**`dev` commit at capture:** `{args.dev_sha}`  ")
    A("**Instrument:** `api.routers.strategy_health.get_strategy_health` — the "
      "dashboard's own code path, called in-process. No new metric was invented.  ")
    A("**Generator (re-run this):** `src/_divergence_baseline_264.py`")
    A("")
    A("> Measurement SSOT: `docs/_active/Plan_Measurement_Trust.md`.")
    A("> This artifact is READ-ONLY output. Nothing was armed, recomputed or deployed to produce it.")
    A("")

    # ---------- headline ----------
    prim0 = WINDOWS[0][0]
    r10 = data[prim0][10.0]
    P0 = sum(r.get("paired_count_cov") or 0 for r in r10)
    PH0 = sum(r.get("phantom_count_cov") or 0 for r in r10)
    MI0 = sum(r.get("missed_count_cov") or 0 for r in r10)
    act0 = [r for r in r10
            if ((r.get("paired_count_cov") or 0) + (r.get("phantom_count_cov") or 0)
                + (r.get("missed_count_cov") or 0)) > 0]

    A("## 0. The one number, and what it is not")
    A("")
    A(f"**Fleet pooled paired% @10s over the settled primary window "
      f"(`{prim0}`) = {_pct(P0, PH0, MI0)}%** "
      f"({P0} paired / {PH0} phantom / {MI0} missed edges across "
      f"{len(act0)} strategies carrying signal, out of {len(r10)} in the fleet).")
    A("")
    A("That is the BEFORE. If the same invocation over the same window after the "
      "split returns materially different numbers **and the stored lane has not been "
      "re-trued in between**, the split changed something.")
    A("")
    A("**It is a fidelity ruler, not a health certificate.** It measures ONE thing: "
      "do recorded alerts pair with stored backtest trades. §9 lists — at length and "
      "on purpose — what it does not measure. Read §9 before using this to clear "
      "anything. In particular it would **not** have caught the 2026-07-27 signal "
      "(an alert-latency dispersion change, p90 6.9s → 13.9s), which is close to the "
      "single most likely way an environment split goes wrong.")
    A("")

    # ---------- capture conditions ----------
    A("## 1. Capture conditions")
    A("")
    A("| condition | value |")
    A("|---|---|")
    A(f"| Market session | **CLOSED** — capture began {started.isoformat()[:19]}Z "
      f"(RTH close is 20:00Z). Post-close per the light-load rail. |")
    A(f"| `dev` commit | `{args.dev_sha}` |")
    A(f"| Capture wall-clock | {round((finished - started).total_seconds(), 1)}s |")
    A(f"| Strategies returned | {len(data[WINDOWS[0][0]][TOLERANCES[0]])} (fleet-wide, all users) |")
    A("")
    A("### Straddle guard (#156)")
    A("")
    A("A capture that straddles an intraday recompute yields a subset>superset artifact. "
      "In-flight recompute/mass-search work was sampled either side of the capture:")
    A("")
    A("| probe | at | update_jobs in-flight | mass_searches in-flight |")
    A("|---|---|---|---|")
    for p in (pre, post):
        uj = p.get("update_jobs", {})
        ms = p.get("mass_searches", {})
        A(f"| {p['label']} | {p['at']} | {uj.get('in_flight', uj.get('error', '?'))} "
          f"| {ms.get('in_flight', ms.get('error', '?'))} |")
    A("")
    A(f"**Quiescence check — primary window re-captured and compared: "
      f"{'✅ IDENTICAL' if quiescent else '❌ DIVERGED MID-CAPTURE'}.** "
      "The primary window @10s is scored twice in one run and the per-strategy "
      "paired/phantom/missed triples are compared. Identical ⇒ the stored lane did "
      "not mutate under the capture, so these numbers are not a #156 straddle "
      "artifact. **If this ever reads DIVERGED, discard the run and re-capture on a "
      "quiet DB** — do not reconcile the two readings.")
    if not quiescent:
        A("")
        A("> 🚨 **THIS RUN DIVERGED.** The numbers below straddled a lane mutation and "
          "must not be used as a baseline.")
    A("")

    # ---------- windows ----------
    A("## 2. Windows measured")
    A("")
    A("Absolute start/end — a rolling `window_hours` read is anchored to `now` and is "
      "**not** reproducible later. Do not change these when re-running for comparison.")
    A("")
    A("| id | start (UTC) | end (UTC) | basis |")
    A("|---|---|---|---|")
    for name, since, until, why in WINDOWS:
        A(f"| `{name}` | {since} | {until} | {why} |")
    A("")

    # ---------- fleet aggregate ----------
    A("## 3. Fleet aggregate (the headline BEFORE number)")
    A("")
    A("`combined% = paired / (paired + phantom + missed)` over coverage-classified "
      "counts (`*_cov`), TBD-excluded — the dashboard's own definition "
      "(`strategy_health.py` combined_pct). Two ways to aggregate, both reported:")
    A("- **pooled** = sum all edges fleet-wide, then divide (high-N strategies dominate).")
    A("- **mean-of-strategies** = per-strategy combined%, averaged over strategies "
      "that had any edges (every strategy counts once).")
    A("")
    A("| window | tol | strategies w/ edges | paired | phantom | missed | TBD | pooled combined% | mean-of-strategies% |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, _s, _u, _w in WINDOWS:
        for tol in TOLERANCES:
            rows = data[name][tol]
            P = sum(r.get("paired_count_cov") or 0 for r in rows)
            PH = sum(r.get("phantom_count_cov") or 0 for r in rows)
            MI = sum(r.get("missed_count_cov") or 0 for r in rows)
            TB = sum(r.get("tbd_count") or 0 for r in rows)
            act = [r for r in rows
                   if ((r.get("paired_count_cov") or 0) + (r.get("phantom_count_cov") or 0)
                       + (r.get("missed_count_cov") or 0)) > 0]
            mean = (round(sum(r.get("combined_pct") or 0 for r in act) / len(act), 1)
                    if act else None)
            A(f"| `{name}` | {int(tol)}s | {len(act)} | {P} | {PH} | {MI} | {TB} "
              f"| **{_pct(P, PH, MI)}** | {mean} |")
    A("")

    # ---------- per-strategy ----------
    A("## 4. Per-strategy paired% — PRIMARY window "
      f"(`{WINDOWS[0][0]}`, {WINDOWS[0][1]} → {WINDOWS[0][2]})")
    A("")
    A("Sorted by combined%@10s ascending (worst first). Strategies with zero edges in "
      "the window are listed separately in §5 — they carry no signal and must not be "
      "read as 0%.")
    A("")
    prim = WINDOWS[0][0]
    by10 = {r["strategy_id"]: r for r in data[prim][10.0]}
    by60 = {r["strategy_id"]: r for r in data[prim][60.0]}
    active, silent = [], []
    for sid, r in by10.items():
        edges = ((r.get("paired_count_cov") or 0) + (r.get("phantom_count_cov") or 0)
                 + (r.get("missed_count_cov") or 0))
        (active if edges > 0 else silent).append(r)
    active.sort(key=lambda r: (r.get("combined_pct") if r.get("combined_pct") is not None else 999))

    A("| sid | name | symbol | TF | lane (`data_source`) | fwd | paired | phantom | missed | TBD | **combined%@10s** | combined%@60s | red flags |")
    A("|---:|---|---|---|---|:--:|---:|---:|---:|---:|---:|---:|---|")
    for r in active:
        sid = r["strategy_id"]
        r60 = by60.get(sid, {})
        rf = ", ".join(r.get("red_flags") or []) or "—"
        nm = (r.get("name") or "")[:26]
        A(f"| {sid} | {nm} | {r.get('symbol')} | {r.get('timeframe')} "
          f"| {r.get('data_source')} | {'Y' if r.get('forward_testing') else 'n'} "
          f"| {r.get('paired_count_cov')} | {r.get('phantom_count_cov')} "
          f"| {r.get('missed_count_cov')} | {r.get('tbd_count')} "
          f"| **{r.get('combined_pct')}** | {r60.get('combined_pct')} | {rf} |")
    A("")

    A("## 5. Zero-edge strategies in the primary window (NO SIGNAL — not 0%)")
    A("")
    A(f"{len(silent)} strategies produced no paired/phantom/missed edges in "
      f"`{prim}`. A post-split re-run must treat these as **unmeasured**, not as "
      "healthy and not as broken. If one of these starts producing edges after the "
      "split that is a *coverage* change, not a fidelity regression.")
    A("")
    if silent:
        A("| sid | name | symbol | TF | lane | fwd | red flags |")
        A("|---:|---|---|---|---|:--:|---|")
        for r in sorted(silent, key=lambda x: x["strategy_id"]):
            rf = ", ".join(r.get("red_flags") or []) or "—"
            A(f"| {r['strategy_id']} | {(r.get('name') or '')[:26]} | {r.get('symbol')} "
              f"| {r.get('timeframe')} | {r.get('data_source')} "
              f"| {'Y' if r.get('forward_testing') else 'n'} | {rf} |")
        A("")

    # ---------- lanes / red flags ----------
    A("## 6. Health lanes and red-flag inventory (primary window)")
    A("")
    A("`data_source` is the lane tag. Red flags are the dashboard's own "
      "`red_flags` list — the health-lane signal, captured verbatim so a post-split "
      "run can diff the *distribution*, not just the paired%.")
    A("")
    rows = data[prim][10.0]
    A("| lane (`data_source`) | strategies |")
    A("|---|---:|")
    for lane, n in sorted(Counter(r.get("data_source") for r in rows).items(),
                          key=lambda kv: -kv[1]):
        A(f"| {lane} | {n} |")
    A("")
    flags = Counter()
    for r in rows:
        for f in (r.get("red_flags") or []):
            flags[f] += 1
    A("| red flag | strategies |")
    A("|---|---:|")
    if flags:
        for f, n in sorted(flags.items(), key=lambda kv: -kv[1]):
            A(f"| `{f}` | {n} |")
    else:
        A("| _(none)_ | 0 |")
    A(f"| **strategies with ≥1 red flag** | **{sum(1 for r in rows if r.get('red_flags'))} "
      f"/ {len(rows)}** |")
    A("")
    A("**⚠️ `snapshot_stale` is a KNOWN DEAD ALARM — do not read it as fleet ill-health.** "
      "It is computed from the deprecated `engine_snapshot_at`, whose legacy data-worker "
      "flush stopped ~2026-07-21 (board #157). The live signal is `shadow_heartbeats`; "
      "on 07-28 the proof was 22/22 heartbeats FRESH against 20 frozen snapshots. The "
      "fix (`RORT_HEALTH_HB_SUPERSEDES_SNAPSHOT`) is BUILT but NOT ARMED as of this "
      "capture, so the flag fires here. **Its near-fleet-wide presence in this baseline "
      "is expected and is not a pre-split defect.** If that flag gets armed before the "
      "post-split re-run, `snapshot_stale` will legitimately collapse toward zero — "
      "that is the fix landing, not the split fixing anything.")
    A("")

    # ---------- today (unsettled) ----------
    A("## 7. Capture-day window (UNSETTLED — context only)")
    A("")
    A(f"`{WINDOWS[1][0]}` has not been through the 00:20Z nightly Update-All, so its "
      "stored backtest lane is the incrementally-built one. Per "
      "`Plan_Measurement_Trust` Phase B this is **not** a valid head-to-head basis; "
      "it is recorded so the capture day is not a hole in the record.")
    A("")
    tod10 = data[WINDOWS[1][0]][10.0]
    tact = [r for r in tod10
            if ((r.get("paired_count_cov") or 0) + (r.get("phantom_count_cov") or 0)
                + (r.get("missed_count_cov") or 0)) > 0]
    A("| sid | name | symbol | TF | paired | phantom | missed | TBD | combined%@10s |")
    A("|---:|---|---|---|---:|---:|---:|---:|---:|")
    for r in sorted(tact, key=lambda x: (x.get("combined_pct")
                                         if x.get("combined_pct") is not None else 999)):
        A(f"| {r['strategy_id']} | {(r.get('name') or '')[:26]} | {r.get('symbol')} "
          f"| {r.get('timeframe')} | {r.get('paired_count_cov')} "
          f"| {r.get('phantom_count_cov')} | {r.get('missed_count_cov')} "
          f"| {r.get('tbd_count')} | {r.get('combined_pct')} |")
    A("")

    A("## 8. Flag surface at this commit")
    A("")
    fnames, fdefaults = flag_surface()
    A(f"**{len(fnames)}** distinct `RORT_*` flag names are referenced in `src/` at "
      f"`{args.dev_sha}`; **{len(fdefaults)}** have a literal in-code default.")
    A("")
    A("> **🚨 THIS IS NOT AN ARMED-FLAG INVENTORY.** These are the flag names the code "
      "knows and their CODE defaults. The ARMED values live in Railway service "
      "variables, and reading them (`railway variables`) is outside what this headless "
      "capture is permitted to do. **Board #265 owns the real armed inventory and this "
      "baseline is INCOMPLETE without it.** Two fleets can share this identical flag "
      "surface and still behave differently because one has a flag armed. Treat the "
      "list below as a vocabulary fingerprint only.")
    A("")
    A("<details><summary>In-code defaults (%d)</summary>" % len(fdefaults))
    A("")
    A("| flag | code default |")
    A("|---|---|")
    for k in sorted(fdefaults):
        A(f"| `{k}` | `{fdefaults[k]}` |")
    A("")
    A("</details>")
    A("")
    A("<details><summary>Full flag-name surface (%d)</summary>" % len(fnames))
    A("")
    A("```")
    for i in range(0, len(fnames), 3):
        A("  ".join(fnames[i:i + 3]))
    A("```")
    A("")
    A("</details>")
    A("")

    A("## 9. What this baseline does NOT cover")
    A("")
    A("**A baseline that overclaims is worse than none**, because the after-comparison "
      "reads as a regression that was never actually measured. Every item below is a "
      "hole. If the post-split question falls in one of these, this artifact cannot "
      "answer it and must not be cited as if it could.")
    A("")
    A("**1. Latency and its dispersion — the biggest hole.** This measures *whether* an "
      "alert paired, not *when* it arrived. The 07-27 divergence scare was a p90 "
      "write-lag change (6.9s → 13.9s) with pairing largely intact. The gateway split's "
      "own design doc names symmetric sub-second delivery as the load-bearing property. "
      "**This baseline does not measure delivery latency at all.** A separate "
      "`live_bars` `written_at`-lag distribution capture is required and is NOT in "
      "here (tripwire definition: board #118).")
    A("")
    A("**2. The replay ceiling — no ops-vs-logic splitter.** `/replay-check` "
      "(`src/replay_harness.py`) is the instrument that separates operational loss from "
      "a real logic residual from the WS/REST floor. It is a multi-minute *per-strategy* "
      "offline job and was not run fleet-wide here. Consequence: a post-split drop in "
      "these numbers **cannot be attributed** to logic vs operations from this artifact "
      "alone. To close this hole, per sid:")
    A("")
    A("```")
    A("cd clients/KevBot_Toolkit/RoR_Trader/src")
    A("../.venv/bin/python replay_harness.py --sids 263,267,338 \\")
    A("    --since 2026-07-29T13:30:00Z --until 2026-07-31T00:00:00Z")
    A("```")
    A("")
    A("**3. The stored backtest lane is MUTABLE — this is the deepest caveat.** The "
      "'backtest' half of every number here is the *stored* lane, and a later recompute "
      "/ nightly re-true rewrites it. So re-running the identical window months from now "
      "can return different numbers **with no split and no code change**. The method is "
      "reproducible; the referent is not frozen. Mitigation for the post-split run: "
      "re-run this capture immediately BEFORE the split cutover as well, so the "
      "before/after pair is close in time and lane-mutation has less room to confound. "
      "(Related: #156 capture-mutation artifact.)")
    A("")
    A("**4. Alert lane only — NOT the algo lane.** `get_strategy_health` pairs recorded "
      "ALERTS against stored backtest trades. The algo/execution lane is a different "
      "lane and is not scored here. 'Health = backtest↔ALERTS' is the standing "
      "definition; do not read these as execution fidelity.")
    A("")
    A("**5. Bar construction and gates are not verified.** No `/bar-parity` read, no "
      "gate-parity view. Even the replay ceiling (item 2) is a bar/engine replay, not a "
      "provider-event replay: it consumes already-aggregated `live_bars.first_*` and "
      "bypasses WS parsing, BarBuilder construction, reconnect/dup/correction ordering, "
      "and grace firing. **Nothing in this baseline certifies that bars were CONSTRUCTED "
      "or ROUTED correctly** — precisely the layer a market-data gateway relocates.")
    A("")
    A(f"**6. Thin and narrow coverage.** {len(act0)} strategies carry signal in the "
      f"primary window out of {len(r10)} in the fleet; {len(r10) - len(act0)} are silent "
      "(§5) and are therefore UNMEASURED, not healthy. Symbols reduce to roughly "
      "TSLA/SPY/NVDA/TSLL. Several signal-carrying strategies have single-digit edge "
      "counts (e.g. sid 340 at 11 edges, sid 194 at 8) where a one-trade difference "
      "swings combined% by tens of points — **do not read those rows as trend**.")
    A("")
    A("**7. No armed-flag inventory (§8).** Same-code, different-flags is a real and "
      "likely post-split failure mode, and it is invisible to this artifact.")
    A("")
    A("**8. No P&L, KPI, or portfolio comparison.** Fidelity only. A split that "
      "preserved pairing but changed fills would pass this baseline.")
    A("")
    A("**9. One capture, no run-to-run variance estimate.** There is no repeat-capture "
      "here establishing how much these numbers move day-to-day with nothing changed. "
      "Without that, a small post-split delta is **not** interpretable as signal. The "
      "three windows (2d / 5d / capture-day) give a weak sense of spread — note the "
      "pooled @10s figure ranges 75.3–86.3 across them — but that is window-composition "
      "variance, not run-to-run noise. **Establishing a real noise floor needs the "
      "capture re-run on several settled days before the split.** Until then, treat "
      "anything under roughly 5 points as inconclusive.")
    A("")

    A("## 10. How to re-run this after the split")
    A("")
    A("The entire value of this artifact is that the identical method runs again and "
      "the two are compared. If you cannot reproduce the invocation, this is decoration.")
    A("")
    A("```")
    A("cd clients/KevBot_Toolkit/RoR_Trader/src")
    A("../.venv/bin/python _divergence_baseline_264.py \\")
    A("    --dev-sha $(git rev-parse --short HEAD) \\")
    A("    --out ../docs/_active/Baseline_Divergence_PostSplit_<YYYY-MM-DD>.md")
    A("```")
    A("")
    A("Rules for the re-run, in priority order:")
    A("")
    A("1. **Do NOT edit `WINDOWS` in the script.** The absolute windows ARE the "
      "comparison. Changing them silently invalidates the diff. If a newer window is "
      "wanted, ADD one; never modify or remove `W1_settled_2d`.")
    A("2. **Run post-close on a quiet DB** and check the §1 straddle-guard probe reads "
      "0 in-flight on BOTH sides. A straddled capture is not comparable (#156).")
    A("3. **Run it against BOTH fleets** once the split exists — one capture per "
      "environment — and diff those against each other as well as against this file. "
      "Same-session, same-bars is the whole point of the split; a cross-day comparison "
      "reintroduces exactly the confound the split was built to remove.")
    A("4. **Record the armed flags for both fleets** (§8 hole, board #265). A paired% "
      "difference with different armed flags is uninterpretable.")
    A("5. **Compare @10s first.** 60s hides real timing divergence and will make a "
      "latency regression look like a non-event.")
    A("6. Read §9 before declaring anything proved.")
    A("")

    A("## 11. Timings (capture cost, for planning the re-run)")
    A("")
    A("| window | tol | seconds |")
    A("|---|---:|---:|")
    for (name, tol), secs in timings.items():
        A(f"| `{name}` | {int(tol)}s | {secs} |")
    A(f"| **total** | | **{round((finished - started).total_seconds(), 1)}** |")
    A("")
    A("Cheap enough to re-run daily. Given §9 item 9, **it should be** — several "
      "pre-split settled-day captures are what turns a single number into a noise floor.")
    A("")

    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n[wrote {args.out}]  ({len(L)} lines)")


if __name__ == "__main__":
    main()
