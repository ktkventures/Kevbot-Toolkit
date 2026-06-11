'use client';

/**
 * ParityDriftRibbonV2 — same per-bar parity microscope as ParityDriftRibbon,
 * plus a Live-values toggle: compare the backtest (REST) lens against the live
 * cache using either:
 *   - first  = decision-time WS values (`first_close` — what the engine SAW the
 *              moment each bar formed and acted on), or
 *   - latest = REST-corrected values (what the cache HEALED to afterward; this
 *              is what the v1 ribbon shows).
 *
 * Not a splice: each bar's `first` is that bar's OWN decision-time value, so a
 * per-bar comparison is faithful without a tip/body blend (the splice only
 * matters for a continuous as-of chart, which this table is not).
 *
 * The original ParityDriftRibbon (latest-only) is intentionally left untouched;
 * this is the v2 sibling that adds the toggle.
 */

import { useMemo, useState } from 'react';

// ---- tolerances (also surfaced on-screen) ----
const PRICE_ABS_MATCH = 0.01;
const PRICE_REL_MATCH = 0.0002;
const PRICE_REL_MINOR = 0.001;
const NUM_ABS_MATCH = 1e-6;
const NUM_REL_MATCH = 0.001;
const NUM_REL_MINOR = 0.01;
const MAX_BARS = 300;

type Status = 'match' | 'minor' | 'major' | 'na';

const STATUS_COLOR: Record<Status, string> = {
  match: 'rgba(76,175,80,0.85)',
  minor: 'rgba(234,179,8,0.9)',
  major: 'rgba(239,68,68,0.88)',
  na: 'rgba(120,120,120,0.22)',
};

function toUnixSec(t: any): number {
  if (t == null) return NaN;
  if (typeof t === 'number') return t > 1e12 ? Math.floor(t / 1000) : t;
  const ms = new Date(t).getTime();
  return isNaN(ms) ? NaN : Math.floor(ms / 1000);
}
function pretty(s: string): string { return (s || '').replace(/_/g, ' '); }
function num(v: any): number | null {
  if (v == null || v === '') return null;
  const n = Number(v);
  return isFinite(n) ? n : null;
}
function comparePrice(a: number, b: number): Status {
  const d = Math.abs(a - b);
  if (d <= PRICE_ABS_MATCH) return 'match';
  const rel = d / Math.max(Math.abs(b), 1e-9);
  if (rel <= PRICE_REL_MATCH) return 'match';
  if (rel <= PRICE_REL_MINOR) return 'minor';
  return 'major';
}
function compareNum(a: number, b: number): Status {
  const d = Math.abs(a - b);
  if (d <= NUM_ABS_MATCH) return 'match';
  const rel = d / Math.max(Math.abs(b), 1e-9);
  if (rel <= NUM_REL_MATCH) return 'match';
  if (rel <= NUM_REL_MINOR) return 'minor';
  return 'major';
}

interface MetricDef { key: string; label: string; kind: 'exists' | 'price' | 'num' | 'state'; col?: string; fidelity?: 'PB' | 'CB'; }
interface Cell { status: Status; tip: string; }
interface RowResult { def: MetricDef; cells: Cell[]; match: number; minor: number; major: number; comparable: number; }

interface Props {
  backtestBars: any[];
  alertFirstBars: any[];
  alertLatestBars: any[];
  overlayNames?: string[];
  oscNames?: string[];
  heatmapConds?: any[];
  timezone?: string | null;
  /** Initial live-values lens (default 'first' — decision-time). */
  defaultSource?: 'first' | 'latest';
  /** Optional custom window (Unix sec) — same semantics as v1. */
  startUtc?: number | null;
  endUtc?: number | null;
}

const CUSTOM_MAX_BARS = 2600;

export default function ParityDriftRibbonV2({
  backtestBars,
  alertFirstBars,
  alertLatestBars,
  overlayNames = [],
  oscNames = [],
  heatmapConds = [],
  timezone = null,
  defaultSource = 'first',
  startUtc = null,
  endUtc = null,
}: Props) {
  const [liveSource, setLiveSource] = useState<'first' | 'latest'>(defaultSource);
  const alertBars = liveSource === 'first' ? alertFirstBars : alertLatestBars;

  const result = useMemo(() => {
    const bt = (backtestBars || []).filter((b) => b && b.timestamp != null);
    const al = (alertBars || []).filter((b) => b && b.timestamp != null);
    if (bt.length === 0 || al.length === 0) return null;

    const sortByTime = (arr: any[]) => [...arr].sort((x, y) => toUnixSec(x.timestamp) - toUnixSec(y.timestamp));
    const btS = sortByTime(bt);
    const alS = sortByTime(al);
    const btIdx = new Map<number, number>(); btS.forEach((b, i) => btIdx.set(toUnixSec(b.timestamp), i));
    const alIdx = new Map<number, number>(); alS.forEach((b, i) => alIdx.set(toUnixSec(b.timestamp), i));

    const minMax = (m: Map<number, number>) => { let lo = Infinity, hi = -Infinity; m.forEach((_, t) => { if (t < lo) lo = t; if (t > hi) hi = t; }); return [lo, hi] as const; };
    const [btLo, btHi] = minMax(btIdx);
    const [alLo, alHi] = minMax(alIdx);
    const lo = Math.max(btLo, alLo);
    const hi = Math.min(btHi, alHi);

    const timeSet = new Set<number>();
    btIdx.forEach((_, t) => timeSet.add(t));
    alIdx.forEach((_, t) => timeSet.add(t));
    let allTimes = Array.from(timeSet).filter((t) => isFinite(t)).sort((a, b) => a - b);
    const customWindow = startUtc != null && endUtc != null && startUtc < endUtc;
    if (customWindow) {
      allTimes = allTimes.filter((t) => t >= (startUtc as number) && t <= (endUtc as number));
    } else if (lo <= hi) {
      allTimes = allTimes.filter((t) => t >= lo && t <= hi);
    }
    const cap = customWindow ? CUSTOM_MAX_BARS : MAX_BARS;
    const truncated = allTimes.length > cap;
    const times = truncated ? allTimes.slice(-cap) : allTimes;

    let common = 0, btOnly = 0, alOnly = 0;
    for (const t of times) {
      const inBt = btIdx.has(t), inAl = alIdx.has(t);
      if (inBt && inAl) common++;
      else if (inBt) btOnly++;
      else alOnly++;
    }

    const defs: MetricDef[] = [
      { key: 'exists', label: 'Bar exists', kind: 'exists' },
      { key: 'open', label: 'Open', kind: 'price', col: 'open' },
      { key: 'high', label: 'High', kind: 'price', col: 'high' },
      { key: 'low', label: 'Low', kind: 'price', col: 'low' },
      { key: 'close', label: 'Close', kind: 'price', col: 'close' },
    ];
    const seen = new Set<string>();
    for (const c of [...overlayNames, ...oscNames]) {
      if (!c || seen.has(c)) continue;
      seen.add(c);
      defs.push({ key: `ind:${c}`, label: pretty(c), kind: 'num', col: c });
    }
    for (const cond of heatmapConds) {
      if (!cond?.column) continue;
      defs.push({ key: `state:${cond.column}`, label: `${pretty(cond.label || cond.column)} ⟨state⟩`, kind: 'state', col: cond.column, fidelity: cond.fidelity === 'PB' ? 'PB' : 'CB' });
    }

    const fmtTime = (t: number) => {
      try {
        return new Date(t * 1000).toLocaleString('en-US', { timeZone: timezone || undefined, month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
      } catch { return new Date(t * 1000).toISOString().slice(0, 19); }
    };
    const stateAt = (side: any[], idxMap: Map<number, number>, t: number, col: string, pb: boolean): any => {
      const i = idxMap.get(t);
      if (i == null) return undefined;
      const bar = pb && i > 0 ? side[i - 1] : side[i];
      return bar ? bar[`_state_${col}`] : undefined;
    };

    const rows: RowResult[] = defs.map((def) => {
      const cells: Cell[] = [];
      let match = 0, minor = 0, major = 0, comparable = 0;
      for (const t of times) {
        const bBar = btIdx.has(t) ? btS[btIdx.get(t)!] : null;
        const aBar = alIdx.has(t) ? alS[alIdx.get(t)!] : null;
        const when = fmtTime(t);
        let status: Status = 'na';
        let tip = when;

        if (def.kind === 'exists') {
          const inBt = !!bBar, inAl = !!aBar;
          status = inBt && inAl ? 'match' : 'major';
          tip = `${when}\nbacktest: ${inBt ? 'bar' : '—'} · live(${liveSource}): ${inAl ? 'bar' : '—'}`;
        } else if (!bBar || !aBar) {
          status = 'na';
          tip = `${when}\n${!bBar ? 'no backtest bar' : 'no live bar'}`;
        } else if (def.kind === 'price' || def.kind === 'num') {
          const a = num(bBar[def.col!]);
          const b = num(aBar[def.col!]);
          if (a == null || b == null) {
            status = 'na';
            tip = `${when}\n${def.label}: ${a == null ? 'backtest n/a' : 'live n/a'}`;
          } else {
            status = def.kind === 'price' ? comparePrice(a, b) : compareNum(a, b);
            const d = a - b;
            tip = `${when}\n${def.label}\nbacktest: ${a}\nlive(${liveSource}): ${b}\nΔ ${d >= 0 ? '+' : ''}${d.toPrecision(4)}`;
          }
        } else {
          const pb = def.fidelity === 'PB';
          const sa = stateAt(btS, btIdx, t, def.col!, pb);
          const sb = stateAt(alS, alIdx, t, def.col!, pb);
          if (sa == null || sb == null) {
            status = 'na';
            tip = `${when}\n${def.label}: ${sa == null ? 'backtest n/a' : 'live n/a'}`;
          } else {
            status = String(sa) === String(sb) ? 'match' : 'major';
            tip = `${when}\n${def.label}${pb ? ' [PB]' : ''}\nbacktest: ${sa}\nlive(${liveSource}): ${sb}`;
          }
        }

        if (status !== 'na') { comparable++; if (status === 'match') match++; else if (status === 'minor') minor++; else major++; }
        cells.push({ status, tip });
      }
      return { def, cells, match, minor, major, comparable };
    });

    return { rows, times, truncated, common, btOnly, alOnly, total: allTimes.length };
  }, [backtestBars, alertBars, liveSource, overlayNames, oscNames, heatmapConds, timezone, startUtc, endUtc]);

  const [hover, setHover] = useState<string | null>(null);

  const sourceToggle = (
    <div className="flex items-center gap-1 text-xs">
      <span style={{ color: 'var(--text-muted)' }}>Live values:</span>
      {(['first', 'latest'] as const).map((s) => (
        <button
          key={s}
          onClick={() => setLiveSource(s)}
          title={s === 'first'
            ? 'Decision-time WS values (first_close) — what the live engine SAW when each bar formed'
            : 'REST-corrected values — what the cache HEALED to (matches the v1 ribbon)'}
          className="px-2 py-0.5 rounded transition-colors"
          style={{
            background: liveSource === s ? 'var(--accent)' : 'var(--bg-input)',
            color: liveSource === s ? 'white' : 'var(--text-muted)',
            border: liveSource === s ? 'none' : '1px solid var(--border)',
            cursor: 'pointer',
          }}
        >
          {s === 'first' ? 'first (decision-time)' : 'latest (corrected)'}
        </button>
      ))}
    </div>
  );

  if (!result) {
    return (
      <div className="space-y-2">
        <div className="flex items-center justify-between flex-wrap gap-2">{sourceToggle}</div>
        <p className="text-xs py-2" style={{ color: 'var(--text-muted)' }}>
          Need both lenses loaded — backtest bars: <strong>{backtestBars?.length || 0}</strong>,
          {' '}live <strong>{liveSource}</strong> bars: <strong>{alertBars?.length || 0}</strong>.
          {(alertBars?.length || 0) === 0 && ` (No live cache ${liveSource} rows for this strategy/window yet.)`}
        </p>
      </div>
    );
  }

  const { rows, common, btOnly, alOnly, truncated, total } = result;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between flex-wrap gap-2 text-xs">
        <div className="flex items-center gap-4" style={{ color: 'var(--text-muted)' }}>
          <span>Bars: <strong style={{ color: 'var(--text-primary)' }}>{common}</strong> common</span>
          {btOnly > 0 && <span style={{ color: 'var(--red)' }}>{btOnly} backtest-only</span>}
          {alOnly > 0 && <span style={{ color: 'var(--red)' }}>{alOnly} live-only</span>}
          {truncated && <span>(showing last {rows[0]?.cells.length} of {total})</span>}
        </div>
        {sourceToggle}
      </div>
      <div className="flex items-center gap-3 text-xs justify-end">
        <Legend color={STATUS_COLOR.match} label="match" />
        <Legend color={STATUS_COLOR.minor} label="minor drift" />
        <Legend color={STATUS_COLOR.major} label="major drift" />
        <Legend color={STATUS_COLOR.na} label="missing / n-a" />
      </div>

      <div className="rounded" style={{ border: '1px solid var(--border)', overflow: 'hidden' }}>
        {rows.map((row, ri) => {
          const pct = row.comparable > 0 ? (row.match / row.comparable) * 100 : null;
          const pctColor = pct == null ? 'var(--text-muted)' : pct >= 99 ? 'var(--green)' : pct >= 90 ? 'rgba(234,179,8,0.95)' : 'var(--red)';
          return (
            <div key={row.def.key} className="flex items-stretch" style={{ borderTop: ri === 0 ? 'none' : '1px solid var(--border)' }}>
              <div className="flex items-center justify-between gap-2 px-2 shrink-0" style={{ width: 210, background: 'var(--bg-input)', fontSize: 11 }} title={row.def.kind === 'state' ? `gate/confluence state (${row.def.fidelity})` : row.def.kind}>
                <span className="truncate" style={{ color: 'var(--text-secondary)' }}>{row.def.label}</span>
                <span style={{ color: pctColor, fontVariantNumeric: 'tabular-nums' }}>{pct == null ? '—' : `${pct.toFixed(pct >= 99.95 ? 0 : 1)}%`}</span>
              </div>
              <div className="flex flex-1" style={{ height: 16 }}>
                {row.cells.map((c, ci) => (
                  <div key={ci} title={c.tip} onMouseEnter={() => setHover(c.tip)} style={{ flex: '1 1 0', minWidth: 1, background: STATUS_COLOR[c.status], borderRight: row.cells.length <= 120 ? '1px solid rgba(0,0,0,0.15)' : 'none', cursor: 'crosshair' }} />
                ))}
              </div>
            </div>
          );
        })}
      </div>

      <div className="flex items-start justify-between gap-3 text-[10px]" style={{ color: 'var(--text-muted)' }}>
        <pre className="whitespace-pre-wrap m-0" style={{ fontFamily: 'inherit', minHeight: 14 }}>
          {hover || `Live = ${liveSource === 'first' ? 'first-write (what the engine saw at each bar’s decision)' : 'REST-corrected (post-heal)'}. Hover a cell for time + both values + Δ.`}
        </pre>
        <span className="text-right shrink-0" style={{ maxWidth: 320 }}>
          match = price within {PRICE_ABS_MATCH}¢/{(PRICE_REL_MATCH * 100).toFixed(2)}% (indicators {(NUM_REL_MATCH * 100).toFixed(1)}%);
          {' '}minor &lt; {(PRICE_REL_MINOR * 100).toFixed(1)}% / {(NUM_REL_MINOR * 100).toFixed(0)}%; states exact.
        </span>
      </div>
    </div>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1" style={{ color: 'var(--text-muted)' }}>
      <span style={{ width: 10, height: 10, background: color, borderRadius: 2, display: 'inline-block' }} />
      {label}
    </span>
  );
}
