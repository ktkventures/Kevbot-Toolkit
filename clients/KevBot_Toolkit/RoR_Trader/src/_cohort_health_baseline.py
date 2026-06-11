"""Build the cohort alert<->backtest health baseline doc (2026-06-10), using the
same /by-hour and /by-deploy computations the admin Strategy Health tab uses
(±5s pairing). Captures the BEFORE state prior to the cache flat-bar cleanup.

NOTE: these pair recorded ALERTS vs REST BACKTEST trades — neither is cache-
derived, so the flat-candle cleanup won't move them. They DO reflect the engine
gap-fill deploy (0d3affb, ~2026-06-10T19:05Z). by-deploy history is stale
(last entry 06-06) so today's deploy isn't its own bucket yet — flagged below.

Run from src/:  ../.venv/bin/python _cohort_health_baseline.py
"""
from __future__ import annotations
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv(override=True)

from api.routers import strategy_health as sh

COHORT = {303: 'UT_BOT_V4', 277: 'BOLLINGER', 293: 'SR_CHANNELS',
          299: 'SUPERTREND', 301: 'SWING_123', 267: 'TSLA SWING_123'}
USER = {'id': 'baseline', 'role': 'admin'}
OUT = '../docs/Cohort_Health_Baseline_2026-06-10.md'


def combined(p, ph, m):
    d = p + ph + m
    return round(100 * p / d, 1) if d else None


def main():
    h = sh.get_strategy_health_by_hour(since=None, until=None, pair_window_s=5.0,
                                       stability_tail_minutes=15.0, user=USER)
    sbh = h.get('strategies_by_hour', {})

    # per-strategy aggregate over the whole 7d window
    agg = {sid: {'alerts': 0, 'bt': 0, 'paired': 0, 'phantom': 0, 'missed': 0}
           for sid in COHORT}
    # per-hour cohort totals
    per_hour = {}
    for hour, rows in sbh.items():
        ph_tot = {'paired': 0, 'phantom': 0, 'missed': 0, 'alerts': 0}
        for r in rows:
            sid = r.get('strategy_id')
            if sid not in COHORT:
                continue
            a = agg[sid]
            a['alerts'] += r.get('alerts', 0); a['bt'] += r.get('bt_events', 0)
            a['paired'] += r.get('paired', 0); a['phantom'] += r.get('phantom', 0)
            a['missed'] += r.get('missed', 0)
            ph_tot['paired'] += r.get('paired', 0); ph_tot['phantom'] += r.get('phantom', 0)
            ph_tot['missed'] += r.get('missed', 0); ph_tot['alerts'] += r.get('alerts', 0)
        if ph_tot['paired'] + ph_tot['phantom'] + ph_tot['missed'] > 0:
            per_hour[hour] = ph_tot

    # by-deploy (for the bucket note)
    try:
        d = sh.get_strategy_health_by_deploy(since=None, until=None, pair_window_s=5.0,
                                             stability_tail_minutes=15.0, warmup_minutes=10.0, user=USER)
        deploys = d.get('deploys', [])
        sbd = d.get('strategies_by_deploy', {})
    except Exception as e:
        deploys, sbd = [], {}

    lines = []
    lines.append('# Cohort Health Baseline — 2026-06-10 (pre cache-cleanup)\n')
    lines.append(f"_Generated {datetime.now(timezone.utc).isoformat()[:19]}Z · pairing ±5s · "
                 f"alerts↔REST-backtest · last ~15min clipped (BT-lane lag)._\n")
    lines.append("**Context:** baseline BEFORE deleting the SPY 10s flat gap-fill cache rows. "
                 "These metrics pair recorded alerts vs REST backtest trades — the cache cleanup "
                 "will NOT move them (display-only). The engine gap-fill fix `0d3affb` deployed "
                 "~2026-06-10T19:05Z; this window is almost entirely PRE-fix, so treat it as the "
                 "buggy-period baseline to compare post-fix days against.\n")

    lines.append('## Per-strategy (7-day window, ±5s)\n')
    lines.append('| sid | strategy | alerts | bt | paired | phantom | missed | combined% | alert-pair% |')
    lines.append('|----:|----------|------:|---:|------:|------:|-----:|--------:|-----------:|')
    cohort_tot = {'alerts': 0, 'bt': 0, 'paired': 0, 'phantom': 0, 'missed': 0}
    for sid, label in COHORT.items():
        a = agg[sid]
        for k in cohort_tot:
            cohort_tot[k] += a[k]
        cp = combined(a['paired'], a['phantom'], a['missed'])
        ap = round(100 * a['paired'] / a['alerts'], 1) if a['alerts'] else None
        lines.append(f"| {sid} | {label} | {a['alerts']} | {a['bt']} | {a['paired']} | "
                     f"{a['phantom']} | {a['missed']} | {cp} | {ap} |")
    ct = cohort_tot
    lines.append(f"| — | **COHORT** | {ct['alerts']} | {ct['bt']} | {ct['paired']} | "
                 f"{ct['phantom']} | {ct['missed']} | **{combined(ct['paired'],ct['phantom'],ct['missed'])}** | "
                 f"{round(100*ct['paired']/ct['alerts'],1) if ct['alerts'] else None} |\n")

    lines.append('## Cohort combined% by hour (most recent 24)\n')
    lines.append('| hour (UTC) | paired | phantom | missed | combined% |')
    lines.append('|-----------|------:|------:|-----:|--------:|')
    for hour in sorted(per_hour)[-24:]:
        t = per_hour[hour]
        lines.append(f"| {hour} | {t['paired']} | {t['phantom']} | {t['missed']} | "
                     f"{combined(t['paired'],t['phantom'],t['missed'])} |")
    lines.append('')

    lines.append('## By-deploy note\n')
    if deploys:
        last = deploys[0]
        lines.append(f"deploy history covers {len(deploys)} commits; most-recent tracked = "
                     f"`{last.get('sha','?')}` ({last.get('subject','')[:50]}) — window to now. "
                     f"**Today's deploys (c0116ea / a9c59fe / 0d3affb) are NOT in deploy_history.json yet**, "
                     f"so the gap-fill fix isn't its own bucket. Update deploy_history to split pre/post-fix.\n")
    else:
        lines.append("by-deploy returned no deploys (stale/missing deploy_history.json).\n")

    lines.append('## Caveats\n')
    lines.append('- ±5s pairing; last ~15min clipped (BT-lane lag tail).\n'
                 '- Cache flat-cleanup will NOT change these (alerts/BT not cache-derived); the '
                 'engine deploy `0d3affb` is what should improve post-fix days.\n'
                 '- Re-run this same script after a full post-fix day to compare.\n')

    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\n[wrote {OUT}]')


if __name__ == '__main__':
    main()
