"""LATENCY-DISPERSION baseline capture — board #264 §12 (pre environment-split).

WHY THIS EXISTS SEPARATELY FROM `_divergence_baseline_264.py`:
that script measures *whether* an alert paired. It does NOT measure *when* the
data arrived, and §9 item 1 of its own artifact names that as the biggest hole.
The 2026-07-27 scare was a p90 write-lag move 6.9s -> 13.9s with pairing largely
intact — the pairing ruler would have read that day as healthy. The V5/#162
gateway design rests on symmetric sub-second delivery between fleets, and the
#163 pilot runs a fleet ~3.5s behind BY CONSTRUCTION; without a lag baseline
"the pilot is slow by design" and "something got slower" are indistinguishable.

WHAT IT MEASURES: the distribution of `live_bars.written_at` lag, defined as

    lag_s = written_at - (bar_start + timeframe_seconds)
          = how long after the bar CLOSED the row first landed in the DB.

That is the #118 tripwire's own definition ("watch live_bars ws_agg 1Min
written_at - (bar_start+60s) per symbol; med >6s RTH = saturation returning"),
reused verbatim rather than invented here.

WHY `written_at` AND NOT `last_updated_at`: `written_at` is a NOT NULL DEFAULT
now() column that the writer never puts in its upsert payload, and no trigger
touches it on UPDATE — so it holds the FIRST-arrival time and is preserved
across Polygon rebroadcasts. `last_updated_at` is bumped by
`live_bars_preserve_first_trg` on every rebroadcast; the gap between the two is
settle duration, reported separately in the artifact.

**The referent here is IMMUTABLE**, which makes this a stronger baseline than
the pairing one: a recompute rewrites the stored backtest lane, but nothing
rewrites `written_at`. The one way this population can move is late INSERTs of
previously-absent bars (rest_insert / rest_correction / rest_backfill), which is
why every table below is scoped BY SOURCE.

READ-ONLY. One aggregate SELECT per section, computed server-side (percentiles
in SQL, no bulk row transfer) so the DB cost stays trivial. It never writes,
never flips a flag, never recomputes.

RE-RUN AFTER THE SPLIT (identical invocation — this is the whole point):

    cd clients/KevBot_Toolkit/RoR_Trader/src
    ../.venv/bin/python _lag_baseline_264.py \
        --out ../docs/_active/_lag_section_<YYYY-MM-DD>.md

Windows are HARDCODED and ABSOLUTE and are the SAME W1/W2/W3 as
`_divergence_baseline_264.py`, so the two halves of the baseline describe the
same bars. Do NOT edit them when re-running for comparison — that IS the
comparison.
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv(override=True)

import psycopg  # noqa: E402

# --- The windows. IDENTICAL to _divergence_baseline_264.py WINDOWS. ---
WINDOWS = [
    ("W1_settled_2d", "2026-07-29T13:30:00Z", "2026-07-31T00:00:00Z",
     "PRIMARY. Same window as the pairing baseline's ruler."),
    ("W2_today_unsettled", "2026-07-31T13:30:00Z", "2026-07-31T20:00:00Z",
     "Fri 07-31 RTH (capture day)."),
    ("W3_settled_5d", "2026-07-24T13:30:00Z", "2026-07-31T00:00:00Z",
     "Wider N, 5 sessions."),
]

# RTH in UTC for July (ET = UTC-4): 13:30:00Z -> 20:00:00Z.
# Stated explicitly because a DST-naive re-run in winter would be wrong by 1h.
RTH_START_UTC = "13:30:00"
RTH_END_UTC = "20:00:00"

# The lag expression, once, so every query in this file measures the same thing.
LAG = ("extract(epoch from (written_at - (bar_start + "
       "timeframe_seconds * interval '1 second')))")

PCTLS = ("count(*) n, "
         "round(min({L})::numeric, 2) lo, "
         "round(percentile_cont(0.50) within group (order by {L})::numeric, 2) p50, "
         "round(percentile_cont(0.90) within group (order by {L})::numeric, 2) p90, "
         "round(percentile_cont(0.99) within group (order by {L})::numeric, 2) p99, "
         "round(max({L})::numeric, 2) hi, "
         "round(avg({L})::numeric, 2) mean").format(L=LAG)

RTH_CLAUSE = (f"and (bar_start at time zone 'UTC')::time >= '{RTH_START_UTC}' "
              f"and (bar_start at time zone 'UTC')::time < '{RTH_END_UTC}' "
              "and extract(isodow from bar_start at time zone 'UTC') <= 5")

TF_LABEL = {10: "10Sec", 15: "15Sec", 30: "30Sec", 60: "1Min", 120: "2Min",
            180: "3Min", 300: "5Min", 600: "10Min", 900: "15Min",
            1800: "30Min", 3600: "1Hour", 7200: "2Hour", 14400: "4Hour",
            86400: "1Day"}


def tf(seconds: int) -> str:
    return TF_LABEL.get(seconds, f"{seconds}s")


def q(cur, sql: str, params: tuple = ()) -> list[tuple]:
    cur.execute(sql, params)
    return cur.fetchall()


def md_table(header: list[str], rows: list[list], align: list[str] | None = None) -> str:
    align = align or (["---"] * len(header))
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)


def by_source_tf(cur, since, until, rth: bool) -> list[tuple]:
    return q(cur, f"""
        select source, timeframe_seconds, {PCTLS}
        from live_bars
        where bar_start >= %s and bar_start < %s
        {RTH_CLAUSE if rth else ''}
        group by 1, 2
        having count(*) > 0
        order by 1, 2
    """, (since, until))


def tripwire_118(cur, since, until) -> list[tuple]:
    """ws_agg 1Min, RTH, PER SYMBOL — the #118 tripwire's exact shape."""
    return q(cur, f"""
        select symbol, {PCTLS}
        from live_bars
        where bar_start >= %s and bar_start < %s
          and source = 'ws_agg' and timeframe_seconds = 60
        {RTH_CLAUSE}
        group by 1 order by 1
    """, (since, until))


def per_symbol(cur, since, until, source, tf_seconds) -> list[tuple]:
    return q(cur, f"""
        select symbol, {PCTLS}
        from live_bars
        where bar_start >= %s and bar_start < %s
          and source = %s and timeframe_seconds = %s
        {RTH_CLAUSE}
        group by 1 order by 1
    """, (since, until, source, tf_seconds))


def per_session(cur, since, until, source, tf_seconds) -> list[tuple]:
    """One row per RTH session — the day-to-day spread of the SAME metric.

    This is the closest thing to a noise floor that a single capture can give:
    if the split moves p50 by less than the day-to-day spread already visible
    here, the move is not interpretable as signal.
    """
    return q(cur, f"""
        select (bar_start at time zone 'UTC')::date d, {PCTLS}
        from live_bars
        where bar_start >= %s and bar_start < %s
          and source = %s and timeframe_seconds = %s
        {RTH_CLAUSE}
        group by 1 order by 1
    """, (since, until, source, tf_seconds))


def pooled_daily(cur, since, until) -> list[tuple]:
    """All engine-consumed WS sources, ALL TFs, FULL 24h, one row per UTC date.

    Not a lane anyone trades — it exists purely to reconcile with the 2026-07-27
    reference figure ("p90 write-lag 6.9s -> 13.9s"), whose exact recipe is not
    recorded anywhere this capture can read. See the artifact section for what
    this does and does not establish.
    """
    return q(cur, f"""
        select (bar_start at time zone 'UTC')::date d, count(*) n,
               round(percentile_cont(0.50) within group (order by {LAG})::numeric, 2) p50,
               round(percentile_cont(0.90) within group (order by {LAG})::numeric, 2) p90,
               round(percentile_cont(0.99) within group (order by {LAG})::numeric, 2) p99
        from live_bars
        where bar_start >= %s and bar_start < %s
          and source in ('ws', 'ws_agg')
        group by 1 order by 1
    """, (since, until))


def settle_duration(cur, since, until) -> list[tuple]:
    """last_updated_at - written_at = how long the bar kept being rebroadcast."""
    d = "extract(epoch from (last_updated_at - written_at))"
    return q(cur, f"""
        select source, timeframe_seconds, count(*) n,
               sum(case when last_updated_at > written_at then 1 else 0 end) rebroadcast,
               round(percentile_cont(0.50) within group (order by {d})::numeric, 2) p50,
               round(percentile_cont(0.90) within group (order by {d})::numeric, 2) p90,
               round(max({d})::numeric, 2) hi
        from live_bars
        where bar_start >= %s and bar_start < %s
          and last_updated_at is not null
          and source in ('ws', 'ws_agg')
        {RTH_CLAUSE}
        group by 1, 2 having count(*) >= 50 order by 1, 2
    """, (since, until))


def render(cur, dev_sha: str) -> str:
    t0 = time.time()
    now = datetime.now(timezone.utc).isoformat()[:19] + "Z"
    w1, w1s, w1e, _ = WINDOWS[0][0], WINDOWS[0][1], WINDOWS[0][2], WINDOWS[0][3]

    L = []
    A = L.append

    A("## 12. Latency-dispersion baseline — `live_bars` `written_at` lag")
    A("")
    A(f"**Captured:** {now} · **`dev` commit:** `{dev_sha}` · "
      "**Generator (re-run this):** `src/_lag_baseline_264.py`")
    A("")
    A("> **Why this section exists.** §9 item 1 named latency dispersion as this "
      "baseline's biggest hole, and M inserted a step to close it. §0–§11 measure "
      "*whether* an alert paired; this section measures *when the data arrived*. "
      "They are deliberately in ONE artifact — a split baseline invites re-running "
      "half of it.")
    A("")

    # ---- 12.0 the numbers, up front -------------------------------------
    tw = tripwire_118(cur, w1s, w1e)
    hd = {(s, t): r for s, t, *r in by_source_tf(cur, w1s, w1e, rth=True)}
    agg1m = hd.get(("ws_agg", 60))
    ws10 = hd.get(("ws", 10))
    worst = max((float(r[3] or 0) for r in tw), default=0.0)

    A("### 12.0 The numbers")
    A("")
    if agg1m and ws10:
        A(md_table(
            ["lane", "n", "**p50**", "**p90**", "p99", "max", "mean"],
            [["`ws_agg` 1Min (the #118 lane)", agg1m[0], f"**{agg1m[2]}s**",
              f"**{agg1m[3]}s**", f"{agg1m[4]}s", f"{agg1m[5]}s", f"{agg1m[6]}s"],
             ["`ws` 10Sec (the direct-WS control)", ws10[0], f"**{ws10[2]}s**",
              f"**{ws10[3]}s**", f"{ws10[4]}s", f"{ws10[5]}s", f"{ws10[6]}s"]],
            ["---", "---:", "---:", "---:", "---:", "---:", "---:"]))
        A("")
    A(f"**#118 tripwire (ws_agg 1Min p50 > 6s RTH): worst symbol = {worst:.2f}s — "
      f"{'🔴 TRIPPED' if worst > 6.0 else '🟢 CLEAR'}.** "
      "That is the BEFORE. If the same invocation after the split returns "
      "materially different numbers, the split changed delivery.")
    A("")
    A("**Read p90, not p50.** The centre of this distribution barely moves — five "
      "settled sessions (§12.5) put the `ws_agg` 1Min p50 inside a ~0.2s band. "
      "The 07-27 scare was a **p90** move (6.9s → 13.9s) that left p50 and pairing "
      "largely intact, which is precisely why §0–§11 would have read that day as "
      "healthy. p90 is where this section earns its keep.")
    A("")

    # ---- 12.1 definition -------------------------------------------------
    A("### 12.1 What is measured (exact definition)")
    A("")
    A("```")
    A("lag_s = written_at - (bar_start + timeframe_seconds)")
    A("      = seconds after the bar CLOSED that the row first landed in the DB")
    A("```")
    A("")
    A("- **`written_at`, not `last_updated_at`.** `written_at` is `NOT NULL DEFAULT "
      "now()`; `live_bars_writer._write_bar_sync` never includes it in the upsert "
      "payload and no trigger touches it on UPDATE, so it holds **first arrival** "
      "and survives Polygon rebroadcasts. `last_updated_at` is bumped by "
      "`live_bars_preserve_first_trg` on every rebroadcast — that pair is reported "
      "separately in §12.6 as *settle duration*.")
    A("- **This is the #118 tripwire's own definition**, reused verbatim: *watch "
      "`live_bars` ws_agg 1Min `written_at - (bar_start+60s)` per symbol; "
      "**med >6s RTH = saturation returning***.")
    A("- **`ws` and `ws_agg` are different producers, not two views of one thing.** "
      "`ws` = bars Polygon delivers directly (the 10/15/30Sec fleet TFs, plus a "
      "sparse scatter of coarse rows); `ws_agg` = bars the worker aggregates "
      "client-side and closes on the wall-clock boundary (1Min and coarser). "
      "#118 was a `ws_agg`-only pathology: the boundary burst "
      "is what saturates, and the direct `ws` path stayed clean throughout. "
      "**Any post-split comparison must keep them separate** or a `ws_agg` "
      "regression will be diluted by clean `ws` rows.")
    A("- **RTH = `bar_start` time-of-day in `[13:30Z, 20:00Z)` on Mon–Fri.** Correct "
      "for July (ET = UTC-4). ⚠️ A re-run between November and March must widen to "
      "`14:30Z–21:00Z` or it will silently measure the wrong hours.")
    A("")
    A("**The referent is IMMUTABLE — this is a stronger baseline than §0–§11.** "
      "§9 item 3 warns that the stored backtest lane is mutable, so re-reading the "
      "same window later can return different pairing numbers with no split and no "
      "code change. Nothing rewrites `written_at`. The only way this population "
      "moves is a *late INSERT* of a bar that was previously absent "
      "(`rest_insert` / `rest_correction` / `rest_backfill`), which is exactly why "
      "every table below is scoped **by `source`** and never pooled across sources.")
    A("")

    # ---- 12.2 conditions -------------------------------------------------
    A("### 12.2 Capture conditions")
    A("")
    A(md_table(
        ["condition", "value"],
        [["Market session", "**CLOSED** — captured after 20:00Z on 2026-07-31, "
          "same post-close / quiet-DB rail as the §1 pairing capture"],
         ["Instrument", "one aggregate `SELECT` per section over `live_bars`, "
          "percentiles computed **server-side** (`percentile_cont`) — no bulk row "
          "transfer, negligible DB load"],
         ["Mutations", "**none.** Read-only; nothing armed, recomputed or deployed"],
         ["`dev` commit", f"`{dev_sha}`"],
         ["Window", f"`{w1}` = {w1s} → {w1e} — **the same absolute window as §3–§6**"]]))
    A("")
    A("No straddle guard is needed here (contrast §1): a recompute cannot mutate "
      "`written_at`, so there is no subset>superset failure mode for this metric.")
    A("")

    # ---- 12.3 headline ---------------------------------------------------
    A(f"### 12.3 PRIMARY — `{w1}` RTH, by source × timeframe")
    A("")
    A("All figures in **seconds after bar close**. Sorted by source then TF.")
    A("")
    rows = by_source_tf(cur, w1s, w1e, rth=True)
    body, neg = [], []
    for src, tfs, n, lo, p50, p90, p99, hi, mean in rows:
        flag = ""
        if p50 is not None and p50 < 0:
            flag = "⚠️ forming-bar writes"
            neg.append((src, tfs, p50))
        elif n < 100:
            flag = "low-n"
        body.append([f"`{src}`", tf(tfs), n, lo, f"**{p50}**", f"**{p90}**",
                     p99, hi, mean, flag])
    A(md_table(["source", "TF", "n", "min", "p50", "p90", "p99", "max", "mean", "note"],
               body,
               ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---"]))
    A("")
    if neg:
        A("⚠️ **Negative-median groups are NOT delivery latency and must not be "
          "compared as such.** A row whose median lag is negative was written while "
          "the bar was still FORMING (the worker emits a partial for coarse TFs), so "
          "`written_at` precedes bar close by construction. Groups affected: "
          + ", ".join(f"`{s}`/{tf(t)}" for s, t, _ in neg) + ". They are shown for "
          "completeness and because a *change* in them post-split is still "
          "informative, but the tripwire read in §12.4 excludes them.")
        A("")
    A("**Reading that table:**")
    A("")
    A("- **Only `ws` and `ws_agg` are delivery latency.** `warmup_seed`, "
      "`rest_insert`, `rest_correction` and `rest_backfill` are gap-fill/seed "
      "writes whose lag records *when a job ran*, not how fast data arrived. Their "
      "large numbers here are correct and expected. Do not put them in a verdict "
      "(§12.9 hole 5).")
    A("- **The coarse `ws_agg` TFs (2Min+) read LOWER than `ws_agg` 1Min, and that "
      "is not a paradox.** §12.6 shows those rows are ~100% rebroadcast: the "
      "coarse bar is written promptly when its boundary passes and then updated as "
      "it settles, so `written_at` is early by construction. **`ws_agg` 1Min is "
      "the lane the fleet actually trades on and the lane #118 named — it is the "
      "ruler; the coarse rows are context.**")
    A("- **`ws` sub-minute has a hard floor around 3.1s.** min, p50 and p90 all sit "
      "just above 3s across every sub-minute row. That floor is a property of the "
      "current pipeline, not of the market, and it is the single most useful thing "
      "to re-measure after the split: **a gateway that preserves pairing but moves "
      "this floor has changed the system in a way §0–§11 cannot see.**")
    A("")

    # ---- 12.4 tripwire ---------------------------------------------------
    A(f"### 12.4 #118 tripwire read — `ws_agg` 1Min per symbol, RTH, `{w1}`")
    A("")
    A("**Threshold: p50 > 6.0s RTH = minute-boundary saturation returning** "
      "(board #118). June-2026 healthy baseline was 4.0–4.8s; the 07-08→07-21 "
      "degraded era ran 12.6–12.7s; post-cull sessions have run 3.2–3.5s.")
    A("")
    body = []
    for sym, n, lo, p50, p90, p99, hi, mean in tw:
        v = float(p50 or 0)
        verdict = "🔴 OVER" if v > 6.0 else ("🟡 close" if v > 4.8 else "🟢 under")
        body.append([sym, n, lo, f"**{p50}**", f"**{p90}**", p99, hi, mean, verdict])
    A(md_table(["symbol", "n", "min", "p50", "p90", "p99", "max", "mean", "p50 vs 6s"],
               body, ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---"]))
    A("")
    A(f"**Where today sits: worst per-symbol p50 = {worst:.2f}s against the 6.0s "
      f"tripwire — {'🔴 TRIPPED' if worst > 6.0 else '🟢 CLEAR'}.** "
      "This is the pre-split reading. A post-split run that moves a symbol from "
      "under to over has reproduced #118 on the new topology.")
    A("")
    # Two things in this table are worth naming rather than leaving for a reader
    # to spot, because both are pre-split facts a post-split run will trip over.
    spread = [(r[0], float(r[4] or 0)) for r in tw]
    hi_sym, hi_p90 = max(spread, key=lambda x: x[1])
    lo_sym, lo_p90 = min(spread, key=lambda x: x[1])
    means = [(r[0], float(r[7] or 0), float(r[3] or 0)) for r in tw]
    A("**Two pre-split facts in that table, stated so a post-split reader does not "
      "rediscover them as regressions:**")
    A("")
    A(f"1. **`mean` sits far above `p50` for every symbol** "
      + ", ".join(f"{s} {m:.2f} vs {p:.2f}" for s, m, p in means)
      + ". The distribution is **already heavy-tailed pre-split** — a small "
        "minority of minutes land tens of seconds late (see the p99/max columns). "
        "That is the BEFORE state, not a defect introduced later.")
    A(f"2. **Symbol dispersion is not uniform: {hi_sym} p90 = {hi_p90:.2f}s against "
      f"{lo_sym} at {lo_p90:.2f}s**, a {hi_p90 - lo_p90:.2f}s spread across symbols "
      "on identical infrastructure. A per-fleet comparison after the split must "
      "compare **like symbol to like symbol**; pooling symbols would let a "
      "single-symbol regression hide inside this existing spread.")
    A("")

    # ---- 12.5 per-session ------------------------------------------------
    A("### 12.5 Day-to-day spread of the same metric (the poor man's noise floor)")
    A("")
    A("§9 item 9 says a single capture gives no run-to-run variance estimate, so a "
      "small post-split delta is uninterpretable. For THIS metric the sessions "
      "inside `W3_settled_5d` give a usable spread for free. **A post-split move "
      "smaller than the spread below is not signal.**")
    A("")
    for src, tfs in (("ws_agg", 60), ("ws", 10)):
        sess = per_session(cur, WINDOWS[2][1], WINDOWS[2][2], src, tfs)
        if not sess:
            continue
        A(f"**`{src}` / {tf(tfs)} — one row per RTH session**")
        A("")
        A(md_table(["session (UTC date)", "n", "min", "p50", "p90", "p99", "max", "mean"],
                   [[str(d), n, lo, f"**{p50}**", p90, p99, hi, mean]
                    for d, n, lo, p50, p90, p99, hi, mean in sess],
                   ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
        A("")
        # tuple order: (date, n, lo, p50, p90, p99, hi, mean)
        p50s = [float(r[3]) for r in sess if r[3] is not None]
        p90s = [float(r[4]) for r in sess if r[4] is not None]
        p99s = [float(r[5]) for r in sess if r[5] is not None]
        if p50s:
            A(f"→ **p50 spans {min(p50s):.2f}–{max(p50s):.2f}s** "
              f"(range {max(p50s) - min(p50s):.2f}s) · "
              f"**p90 spans {min(p90s):.2f}–{max(p90s):.2f}s** "
              f"(range {max(p90s) - min(p90s):.2f}s) · "
              f"p99 spans {min(p99s):.2f}–{max(p99s):.2f}s "
              f"(range {max(p99s) - min(p99s):.2f}s).")
            A("")
            A(f"**Read the ranges, not just the medians: the centre of this "
              f"distribution is stable to well under a second day-to-day, while "
              f"the p99 tail moves by {max(p99s) - min(p99s):.0f}s across the same "
              f"five sessions with nothing changed.** For `{src}`/{tf(tfs)} that "
              f"makes ~{max(p50s) - min(p50s):.2f}s a **lower bound** on p50 noise "
              f"and ~{max(p90s) - min(p90s):.2f}s a lower bound on p90 noise — "
              "five sessions of one regime cannot establish an upper bound, so "
              "treat these as 'at least this much movement means nothing', not as "
              "significance thresholds. A p99 move is not interpretable from five "
              "sessions at all. **The p90 is the usable middle and the one to "
              "compare first.**")
            A("")

    # ---- 12.5b per-symbol sub-minute ------------------------------------
    A(f"### 12.5b `ws` sub-minute per symbol, RTH, `{w1}`")
    A("")
    A("The direct-WS path, which #118 found IMMUNE to the boundary burst. It is "
      "the control: if a post-split run moves `ws_agg` and NOT this, the cause is "
      "boundary-chain saturation; if it moves BOTH, the cause is upstream of the "
      "worker (feed, gateway, network).")
    A("")
    body = []
    for tfs in (10, 15, 30):
        for sym, n, lo, p50, p90, p99, hi, mean in per_symbol(cur, w1s, w1e, "ws", tfs):
            body.append([tf(tfs), sym, n, lo, f"**{p50}**", f"**{p90}**", p99, hi, mean])
    A(md_table(["TF", "symbol", "n", "min", "p50", "p90", "p99", "max", "mean"], body,
               ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
    A("")

    # ---- 12.5c reconciliation with the 07-27 figure ----------------------
    A("### 12.5c Reconciliation with the 2026-07-27 reference figure — **it does "
      "not fully reconcile, and that matters**")
    A("")
    A("The step that commissioned this section cites the 07-27 scare as **p90 "
      "write-lag 6.9s → 13.9s**. Anyone comparing a post-split run against §12 "
      "will reach for those numbers, so the relationship has to be stated rather "
      "than assumed.")
    A("")
    A("**Under this section's primary scoping (`ws_agg` 1Min, RTH, per session), "
      "07-27 reads p90 = 5.23s — it is NOT elevated** (§12.5). So the 07-27 "
      "figures were not produced by the recipe §12.3/§12.4 use.")
    A("")
    A("Widening the scope to **all engine-consumed WS sources, all TFs, full 24h** "
      "does land on the reference number:")
    A("")
    pd_rows = pooled_daily(cur, "2026-07-22", "2026-08-01")
    A(md_table(["UTC date", "n", "p50", "p90", "p99"],
               [[str(d), n, p50, f"**{p90}**", p99] for d, n, p50, p90, p99 in pd_rows],
               ["---", "---:", "---:", "---:", "---:"]))
    A("")
    A("⚠️ The final row is the capture day and is **still accumulating** "
      "(post-close and overnight bars land after this capture), so its `n` is not "
      "comparable to the complete sessions above it. Every other table in §12 ends "
      "at or before 2026-07-31T00:00:00Z and is frozen.")
    A("")
    p90pool = [float(r[3]) for r in pd_rows if r[3] is not None]
    A(f"**That pooled p90 sits at {min(p90pool):.2f}–{max(p90pool):.2f}s on every "
      f"one of these {len(pd_rows)} sessions — the 6.9s half of the reference "
      f"figure, essentially exactly.** That is strong circumstantial evidence the "
      "07-27 recipe was this pooled/unscoped one.")
    A("")
    A("**But it is circumstantial, and this artifact does not claim more.** The "
      "07-27 recipe is not recorded anywhere this capture can read: the elevated "
      "13.9s half is not reproducible here (07-27 pooled p90 reads "
      f"{[float(r[3]) for r in pd_rows if str(r[0]) == '2026-07-27'][0]:.2f}s), "
      "which is consistent with the incident having been measured on an intraday "
      "slice rather than a whole session, but is equally consistent with a "
      "different metric altogether. **Two consequences, both binding:**")
    A("")
    A("1. **Compare §12 against §12's own re-run, never against the 07-27 "
      "numbers.** Cross-recipe comparison is exactly the overclaiming §9 forbids.")
    A("2. The pooled table above is included **only** as a legacy bridge. It mixes "
      "producers with different latency profiles and TFs whose bar-close semantics "
      "differ, so a move in it is not attributable. **`ws_agg` 1Min RTH (§12.4) "
      "remains the ruler.**")
    A("")

    # ---- 12.6 settle duration -------------------------------------------
    A(f"### 12.6 Settle duration — `last_updated_at − written_at`, RTH, `{w1}`")
    A("")
    A("How long a bar kept being rebroadcast/corrected after first arrival. "
      "Distinct from delivery lag and reported so a post-split change in "
      "*correction behaviour* is not mistaken for a change in *delivery*. "
      "Groups with n < 50 omitted.")
    A("")
    sd = settle_duration(cur, w1s, w1e)
    A(md_table(["source", "TF", "n", "rows rebroadcast", "p50", "p90", "max"],
               [[f"`{s}`", tf(t), n, f"{rb} ({100.0 * rb / n:.1f}%)", p50, p90, hi]
                for s, t, n, rb, p50, p90, hi in sd],
               ["---", "---", "---:", "---:", "---:", "---:", "---:"]))
    A("")

    # ---- 12.7 other windows ---------------------------------------------
    A("### 12.7 The other two pairing windows (same metric, for one-to-one comparability)")
    A("")
    A("§3 reports pairing over `W2_today_unsettled` and `W3_settled_5d` as well. "
      "The same lag metric over those exact windows, so every pairing row in this "
      "artifact has a latency row beside it. **`W2` is the capture day and is "
      "UNSETTLED for pairing — but lag is unaffected by settlement**, so W2's lag "
      "figures ARE directly comparable (a rare case where W2 is usable "
      "head-to-head).")
    A("")
    for wid, ws, we, _note in WINDOWS[1:]:
        A(f"**`{wid}`** ({ws} → {we}) — RTH, `ws_agg`/1Min and `ws` sub-minute only")
        A("")
        rows = [r for r in by_source_tf(cur, ws, we, rth=True)
                if (r[0] == "ws_agg" and r[1] == 60) or (r[0] == "ws" and r[1] in (10, 15, 30))]
        A(md_table(["source", "TF", "n", "min", "p50", "p90", "p99", "max", "mean"],
                   [[f"`{s}`", tf(t), n, lo, f"**{p50}**", f"**{p90}**", p99, hi, mean]
                    for s, t, n, lo, p50, p90, p99, hi, mean in rows],
                   ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
        A("")

    # ---- 12.8 full-window (no RTH scoping) ------------------------------
    A(f"### 12.8 `{w1}` WITHOUT RTH scoping (the literal pairing window)")
    A("")
    A("§3–§6 score the whole `W1` span, extended hours included. This is the same "
      "literal span for lag, so nobody has to wonder whether the RTH scoping above "
      "changed the answer. Extended-hours tape is thin and `ws_agg` closes on the "
      "boundary regardless of prints, so these run higher — **use §12.3, not this, "
      "as the ruler**; this is here to keep the two halves honestly aligned.")
    A("")
    rows = [r for r in by_source_tf(cur, w1s, w1e, rth=False)
            if (r[0] == "ws_agg" and r[1] == 60) or (r[0] == "ws" and r[1] in (10, 15, 30))]
    A(md_table(["source", "TF", "n", "min", "p50", "p90", "p99", "max", "mean"],
               [[f"`{s}`", tf(t), n, lo, f"**{p50}**", f"**{p90}**", p99, hi, mean]
                for s, t, n, lo, p50, p90, p99, hi, mean in rows],
               ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
    A("")

    # ---- 12.9 holes ------------------------------------------------------
    A("### 12.9 What the LATENCY baseline does NOT cover")
    A("")
    A("Same rule as §9: a baseline that overclaims is worse than none. These are "
      "holes in §12 specifically, on top of every hole in §9.")
    A("")
    A("1. **This is DB-write lag, not end-to-end alert latency.** The chain is "
      "`Polygon emits → worker receives → bar closes → live_bars write → shadow/"
      "monitor pipeline → alert insert`. `written_at` instruments ONE hop. #118 "
      "happened to be visible there because the write sat inside the saturated "
      "synchronous chain — but a regression that lands entirely AFTER the write "
      "(monitor pipeline, alert insert) is invisible to this section. "
      "`alerts.fired_at` vs bar close would cover that hop and is NOT captured here.")
    A("2. **No provider-side latency.** `bar_start` is Polygon's timestamp; the lag "
      "from real-world trade to Polygon emitting is upstream of everything we can "
      "see and is folded into these numbers as an unknown constant. If the gateway "
      "changes the *provider connection* (not just the routing), part of any move "
      "will be provider-side and this metric cannot separate it.")
    A("3. **Distribution shape only at 4 points.** p50/p90/p99/max. A bimodal "
      "distribution — say, most bars fast and a periodic slow cohort — can move "
      "materially with all four points roughly intact. If a post-split delta is "
      "ambiguous, pull the raw lag column for the window and compare full "
      "distributions rather than adding percentiles to this table after the fact.")
    A("4. **Absent bars are invisible.** A bar the worker never wrote contributes "
      "no row and therefore no lag. **Infinite latency reads as no data, not as a "
      "big number** — so §12 must always be read next to a coverage/row-count "
      "check (the `n` columns are there for exactly this; a post-split `n` that "
      "drops materially is a coverage regression masquerading as a clean latency "
      "table).")
    A("5. **`rest_insert` / `rest_correction` / `rest_backfill` rows are gap-fill, "
      "not delivery.** Their lag measures when a backfill ran, which is a "
      "scheduling fact, not a latency fact. They are reported in §12.3 for "
      "completeness and excluded from every verdict.")
    A("6. **Two RTH sessions in the primary window.** §12.5 widens to five for the "
      "spread, but that is still five days of one market regime. A quiet-tape week "
      "and a volatile week are not interchangeable here.")
    A("7. **No per-strategy attribution.** This is per (source, TF, symbol). It "
      "cannot say which strategy ate the lag, and it does not connect a lag figure "
      "to a specific missed/phantom edge in §4.")
    A("8. **No continuity with the 07-27 reference figure.** §12.5c shows the "
      "07-27 numbers are not reproducible under this section's scoping and only "
      "half-reproducible under a pooled one. **§12 therefore establishes a NEW "
      "series starting today, not a continuation of an existing one** — it cannot "
      "be used to re-litigate 07-27, and pre-07-31 latency history remains "
      "un-baselined.")
    A("9. **The day-to-day spread in §12.5 is a LOWER bound on noise, not a "
      "significance threshold.** Five sessions of one market regime, all on the "
      "post-cull fleet. It can prove a small move is meaningless; it cannot prove "
      "a moderate move is meaningful. Closing this properly needs the capture run "
      "on more pre-split sessions — the same ask §9 item 9 makes of the pairing "
      "half, and the generator is cheap enough to run daily.")
    A("")

    A("### 12.10 How to re-run §12 after the split")
    A("")
    A("```")
    A("cd clients/KevBot_Toolkit/RoR_Trader/src")
    A("../.venv/bin/python _lag_baseline_264.py \\")
    A("    --dev-sha $(git rev-parse --short HEAD) \\")
    A("    --out ../docs/_active/_lag_section_<YYYY-MM-DD>.md")
    A("```")
    A("")
    A("1. **Do NOT edit `WINDOWS`.** They are the same absolute windows as "
      "`_divergence_baseline_264.py`. Changing either copy desynchronises the two "
      "halves of this artifact.")
    A("2. **Run it against BOTH fleets** and diff them against each other as well "
      "as against this file. Symmetric delivery between fleets is the #164 "
      "gateway's load-bearing claim; a cross-fleet diff is the only thing that "
      "tests it.")
    A("3. **For the #163 pilot, expect a constant offset by construction** "
      "(~3.5s). Subtract the *designed* offset before reading a regression — and "
      "if the offset is not constant across symbols and TFs, that itself is the "
      "finding.")
    A("4. **Compare p90 before p50.** The 07-27 scare moved p90 6.9→13.9s. A p50 "
      "that holds while p90 doubles is the exact shape of the failure this "
      "section exists to catch.")
    A("5. **Check `n` first** (hole 4). A clean latency table over half the rows "
      "is worse than a dirty one over all of them.")
    A("6. Read §12.9 and §9 before declaring anything proved.")
    A("")
    A(f"*§12 capture cost: {time.time() - t0:.1f}s of DB time.*")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-sha", default=os.environ.get("DEV_SHA", "unknown"))
    ap.add_argument("--out", default=None, help="write markdown here (default: stdout)")
    ap.add_argument("--append", default=None,
                    help="append the section to an existing artifact")
    args = ap.parse_args()

    dsn = os.environ.get("SUPABASE_CONNECTION_STRING")
    if not dsn:
        raise SystemExit("SUPABASE_CONNECTION_STRING missing — refusing to guess "
                         "(no silent defaults).")

    with psycopg.connect(dsn, connect_timeout=20, prepare_threshold=None,
                         autocommit=True) as conn, conn.cursor() as cur:
        md = render(cur, args.dev_sha)

    if args.append:
        with open(args.append, "a", encoding="utf-8") as fh:
            fh.write("\n" + md + "\n")
        print(f"appended {len(md)} chars to {args.append}")
    elif args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(md + "\n")
        print(f"wrote {len(md)} chars to {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
