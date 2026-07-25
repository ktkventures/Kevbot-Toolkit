'use client';

/**
 * Strategy Health V1 — per-strategy freshness + red-flag table.
 *
 * Surfaces the signals the data-worker + recompute paths write back to
 * the DB so we can answer "is the fleet actually getting kept current"
 * without tailing Railway logs.
 *
 * Columns:
 *   strategy | symbol | timeframe | bar parity (live vs REST, W2-6a) |
 *   BT current (shadow heartbeat) | KPI age |
 *   last trade age | trades | paired | phantom | missed | flags
 *
 * Sort: default by red-flag count desc (most-broken first), click any
 * header to re-sort. Filter chips at the top scope to a single flag.
 */

import { useEffect, useMemo, useState } from 'react';
import Card from '@/components/Card';
import { apiFetch } from '@/lib/api/client';
import Last10PairingModal from '@/views/Last10PairingModal';
import StrategyNotesModal from '@/views/StrategyNotesModal';
import type { Task } from '@/views/taskBoardShared';
import {
  useStrategyHealth,
  type StrategyHealthFlag,
  type StrategyHealthRow,
} from '@/hooks/queries/useStrategyHealth';
import {
  useBarParity,
  type BarParityPair,
} from '@/hooks/queries/useBarParity';
import { useSetSnapshotSubscription } from '@/hooks/mutations/useStrategyMutations';

// ── Formatting helpers ────────────────────────────────────────────────

function fmtAge(seconds: number | null | undefined): string {
  if (seconds == null) return '—';
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  if (seconds < 86400) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return m === 0 ? `${h}h` : `${h}h ${m}m`;
  }
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  return h === 0 ? `${d}d` : `${d}d ${h}h`;
}

/** Age in seconds of an ISO timestamp relative to the server's `now`
 *  (falls back to client clock). Used for W2-0 bt_current_through, which
 *  the API ships as a raw timestamp rather than a precomputed age. */
function ageSecFrom(iso: string | null | undefined,
                    nowIso: string | undefined): number | null {
  if (!iso) return null;
  const t = Date.parse(iso);
  if (isNaN(t)) return null;
  const n = nowIso ? Date.parse(nowIso) : Date.now();
  if (isNaN(n)) return null;
  return Math.max(0, Math.round((n - t) / 1000));
}

/** Color-coded age cell. Thresholds tuned for the data-worker cadence:
 *  green ≤ short, yellow ≤ medium, red beyond. */
function ageStyle(seconds: number | null | undefined,
                  thresholds: { green: number; yellow: number }): React.CSSProperties {
  if (seconds == null) return { color: 'var(--text-muted)' };
  let color = 'var(--text-primary)';
  let bg = 'transparent';
  if (seconds <= thresholds.green) {
    bg = 'rgba(76,175,80,0.15)'; color = '#7fd081';
  } else if (seconds <= thresholds.yellow) {
    bg = 'rgba(255,193,7,0.15)'; color = '#ffc107';
  } else {
    bg = 'rgba(244,67,54,0.18)'; color = '#ef5350';
  }
  return {
    background: bg, color,
    padding: '2px 6px', borderRadius: 3,
    fontVariantNumeric: 'tabular-nums',
    whiteSpace: 'nowrap',
  };
}

const FLAG_LABEL: Record<StrategyHealthFlag, string> = {
  legacy_no_confluence_id: 'legacy',
  no_baseline: 'no baseline',
  snapshot_missing: 'no snapshot',
  snapshot_stale: 'snapshot stale',
  kpis_stale: 'KPIs stale',
  kpis_marked_stale: 'KPIs marked stale',
  data_refresh_stale: 'data stale',
  no_recent_trades: 'no recent trades',
  parity_fail: 'parity fail',
  has_discrepancies: 'discrepancies',
  phantom_alerts: 'phantom',
  missed_alerts: 'missed',
};

/** Most flags are amber; only a few are hard-red. */
const FLAG_TONE: Record<StrategyHealthFlag, 'red' | 'amber' | 'gray'> = {
  legacy_no_confluence_id: 'gray',
  no_baseline: 'gray',
  snapshot_missing: 'red',
  snapshot_stale: 'red',
  kpis_stale: 'amber',
  kpis_marked_stale: 'amber',
  data_refresh_stale: 'amber',
  no_recent_trades: 'amber',
  parity_fail: 'red',
  has_discrepancies: 'amber',
  phantom_alerts: 'amber',   // alert without backtest — investigate
  missed_alerts: 'red',      // backtest fired, algo didn't — bigger deal
};

function flagChipStyle(tone: 'red' | 'amber' | 'gray'): React.CSSProperties {
  const palette = {
    red:   { bg: 'rgba(244,67,54,0.18)', fg: '#ef5350' },
    amber: { bg: 'rgba(255,193,7,0.18)', fg: '#ffc107' },
    gray:  { bg: 'rgba(120,120,120,0.18)', fg: 'var(--text-muted)' },
  }[tone];
  return {
    background: palette.bg, color: palette.fg,
    padding: '1px 6px', borderRadius: 3, fontSize: 11,
    whiteSpace: 'nowrap', marginRight: 4, marginBottom: 2,
    display: 'inline-block',
  };
}

// ── Sortable table ────────────────────────────────────────────────────

type SortKey =
  | 'flags' | 'name' | 'symbol' | 'timeframe'
  | 'btCurrent' | 'kpis' | 'lastTrade' | 'trades'
  | 'lastBt' | 'lastAlert' | 'barParity'
  | 'combined' | 'paired' | 'phantom' | 'missed' | 'tbd'
  | 'sid' | 'last10';

/** Last-10 score shape from /api/strategy-health-last10 (board #70). */
type Last10Score = { points: number; denom: number };

/** Join key between StrategyHealthRow and BarParityPair — parity is
 *  computed per unique (symbol, timeframe), shared across strategies. */
function parityKey(symbol: string | null, timeframe: string | null): string {
  return `${symbol ?? ''}|${timeframe ?? ''}`;
}

function rowSortValue(
  r: StrategyHealthRow,
  key: SortKey,
  parityByPair?: Map<string, BarParityPair>,
  last10?: Map<number, Last10Score>,
): number | string {
  switch (key) {
    case 'flags':     return -r.red_flags.length; // most flags first by default asc
    case 'sid':       return r.strategy_id;
    case 'last10': {
      const s10 = last10?.get(r.strategy_id);
      // best ratio first on asc; no-data rows sort last
      return s10 && s10.denom > 0 ? -(s10.points / s10.denom) : 1;
    }
    case 'name':      return r.name ?? '';
    case 'symbol':    return r.symbol ?? '';
    case 'timeframe': return r.timeframe ?? '';
    // W2-0: BT Current sorts by heartbeat coverage recency — newest boundary
    // first on asc (equivalent to smallest age first). Nulls sort last.
    case 'btCurrent': {
      const t = r.bt_current_through ? Date.parse(r.bt_current_through) : NaN;
      return isNaN(t) ? Number.POSITIVE_INFINITY : -t;
    }
    case 'kpis':      return r.kpis_age_sec ?? Number.POSITIVE_INFINITY;
    case 'lastTrade': return r.last_entry_age_sec ?? Number.POSITIVE_INFINITY;
    case 'lastBt':    return r.last_backtest_created_age_sec ?? Number.POSITIVE_INFINITY;
    case 'lastAlert': return r.last_alert_age_sec ?? Number.POSITIVE_INFINITY;
    case 'trades':    return -r.trade_count_backtest;
    // 2026-06-16: per-strategy coverage-classified counts (match By-Hour).
    // TBD makes the comparison apples-to-apples naturally — the old
    // fleet-wide global-fair cutoff is retired (it over-filtered).
    case 'combined':  return -(r.combined_pct ?? -1); // nulls (no data) sort last
    // W2-6a: Bar Parity sorts worst-last on asc, nulls (not computable)
    // last — mirrors the 'combined' convention.
    case 'barParity':
      return -(parityByPair?.get(parityKey(r.symbol, r.timeframe))
        ?.parity_pct ?? -1);
    case 'paired':    return -r.paired_count_cov;
    case 'phantom':   return -r.phantom_count_cov;
    case 'missed':    return -r.missed_count_cov;
    case 'tbd':       return -r.tbd_count;
  }
}

// Combined% cell color — matches the By-Hour tab convention.
function combinedPctColor(v: number | null): string {
  if (v === null || v === undefined) return 'var(--text-muted)';
  if (v >= 90) return '#7fd081';   // green
  if (v >= 70) return 'var(--text)';
  if (v >= 40) return '#ffc107';   // amber
  return '#ef5350';                // red
}

// W2-6a — Bar Parity % chip (age-chip styling, like ageStyle). Parity is
// the worst-of-OHLC field match rate vs REST truth, so the thresholds sit
// high: even a clean WS feed drops a few bars' worth of late prints.
function barParityStyle(v: number | null | undefined): React.CSSProperties {
  let color = 'var(--text-muted)';
  let bg = 'transparent';
  if (v != null) {
    if (v >= 98) {
      bg = 'rgba(76,175,80,0.15)'; color = '#7fd081';
    } else if (v >= 90) {
      bg = 'rgba(255,193,7,0.15)'; color = '#ffc107';
    } else {
      bg = 'rgba(244,67,54,0.18)'; color = '#ef5350';
    }
  }
  return {
    background: bg, color,
    padding: '2px 6px', borderRadius: 3,
    fontVariantNumeric: 'tabular-nums',
    whiteSpace: 'nowrap',
    fontWeight: 600,
  };
}

// Tooltip for the Bar Parity cell: definition + per-field breakdown +
// coverage stats, or the not-computable reason.
function barParityTitle(p: BarParityPair | undefined): string {
  const def = 'Bar Parity (data) = DATA-LAYER check, OHLC-only, recent window: '
    + 'worst-of-OHLC field match rate, live bars (WS) '
    + 'vs REST truth (bar_cache resampled to TF), tolerance max($0.01, 0.02%), '
    + 'recent window anchored at the newest settled live bar. '
    + 'No indicators or engine lens — for full per-indicator parity per '
    + 'strategy see the Model Parity tab.';
  if (!p) return `${def} No computation for this (symbol, TF) pair yet.`;
  if (p.parity_pct == null) {
    return `${def} Not computable: ${p.reason ?? 'unknown'}.`;
  }
  const f = p.field_pcts;
  const fieldsLine = f
    ? ` O ${f.open}% · H ${f.high}% · L ${f.low}% · C ${f.close}% ·`
    : '';
  return `${def}${fieldsLine} n=${p.bars_compared} bars (src ${p.source_timeframe ?? '—'})`
    + ` · live-only ${p.bars_live_only} · rest-only ${p.bars_rest_only}`
    + ` · backfilled ${p.bars_backfilled}`
    + ` · window ${p.window_start ?? '—'} → ${p.window_end ?? '—'}`;
}

// ── Component ─────────────────────────────────────────────────────────

const WINDOW_OPTIONS: { hours: number; label: string }[] = [
  { hours: 1,   label: '1h' },
  { hours: 3,   label: '3h' },
  { hours: 6,   label: '6h' },
  { hours: 12,  label: '12h' },
  { hours: 24,  label: '24h' },
  { hours: 48,  label: '48h' },
  { hours: 72,  label: '72h' },
  { hours: 168, label: '7d' },
];

function windowLabel(hours: number): string {
  const m = WINDOW_OPTIONS.find(o => o.hours === hours);
  if (m) return m.label;
  if (hours >= 24) return `${Math.round(hours / 24)}d`;
  return `${hours}h`;
}

// Format an ISO timestamp into a datetime-local input value
// (YYYY-MM-DDTHH:MM in the browser's local TZ).
function isoToLocalInput(iso: string | null): string {
  if (!iso) return '';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return '';
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

// Parse a datetime-local input value (browser local TZ) into an ISO
// 8601 string with UTC offset.
function localInputToIso(s: string): string {
  if (!s) return '';
  const d = new Date(s);  // Date constructor interprets as local TZ
  if (isNaN(d.getTime())) return '';
  return d.toISOString();
}

export default function StrategyHealthV1() {
  // Mode = 'rolling' uses windowHours chips. 'custom' uses fixed
  // start/end timestamps (frozen — no auto-refetch).
  const [mode, setMode] = useState<'rolling' | 'custom'>('rolling');
  const [windowHours, setWindowHours] = useState<number>(24);
  // Custom-mode inputs (as datetime-local strings, browser TZ).
  // Default start = 24h ago; default end = now. Backend converts both
  // to UTC.
  const [customStartInput, setCustomStartInput] = useState<string>(
    () => isoToLocalInput(new Date(Date.now() - 24 * 3600 * 1000).toISOString()));
  const [customEndInput, setCustomEndInput] = useState<string>(
    () => isoToLocalInput(new Date().toISOString()));

  const customStartIso = mode === 'custom' ? localInputToIso(customStartInput) : null;
  const customEndIso = mode === 'custom' ? localInputToIso(customEndInput) : null;

  // ── APPLIED filter state (Kevin, 07-20) ──────────────────────────────
  // The health query key depends ONLY on these applied values, NOT on the
  // live filter controls above — so editing a window chip or a custom date
  // no longer re-fetches (which used to blank the whole page with "Loading
  // strategy health…" and hit the DB on every keystroke). The draft/control
  // state is committed into applied state ONLY when the user clicks Refresh.
  const [appliedMode, setAppliedMode] = useState<'rolling' | 'custom'>('rolling');
  const [appliedWindowHours, setAppliedWindowHours] = useState<number>(24);
  const [appliedStartIso, setAppliedStartIso] = useState<string | null>(null);
  const [appliedEndIso, setAppliedEndIso] = useState<string | null>(null);

  // Commit the draft controls → applied. Changing an applied value changes
  // the React Query key, which fetches the newly-selected window.
  const applyFilters = () => {
    setAppliedMode(mode);
    setAppliedWindowHours(windowHours);
    setAppliedStartIso(mode === 'custom' ? localInputToIso(customStartInput) : null);
    setAppliedEndIso(mode === 'custom' ? localInputToIso(customEndInput) : null);
  };

  // Draft controls differ from what's currently applied/shown → prompt Apply.
  const filtersDirty =
    mode !== appliedMode ||
    (mode === 'rolling' && windowHours !== appliedWindowHours) ||
    (mode === 'custom' &&
      (customStartIso !== appliedStartIso || customEndIso !== appliedEndIso));

  const { data, isLoading, error, dataUpdatedAt, refetch } =
    useStrategyHealth({
      windowHours: appliedWindowHours,
      start: appliedMode === 'custom' ? appliedStartIso : null,
      end: appliedMode === 'custom' ? appliedEndIso : null,
    });

  // W2-6a — Bar Parity % (live_bars vs REST truth), computed server-side
  // per unique (symbol, timeframe) pair and cached ~5 min. Independent of
  // the main health query so a slow/failed parity fetch never blocks the
  // table — cells just show '—' until it lands.
  const barParity = useBarParity();
  const parityByPair = useMemo(() => {
    const m = new Map<string, BarParityPair>();
    for (const p of barParity.data?.pairs ?? []) {
      m.set(parityKey(p.symbol, p.timeframe), p);
    }
    return m;
  }, [barParity.data]);

  // Last-10 health scores (board #70) — independent fetch so a slow score
  // computation never blocks the main table; cells show '—' until it lands.
  const [last10, setLast10] = useState<Map<number, Last10Score>>(new Map());
  const [last10Sid, setLast10Sid] = useState<{ sid: number; name: string } | null>(null);
  useEffect(() => {
    apiFetch<{ scores: Record<string, Last10Score> }>('/api/strategy-health-last10')
      .then((d) => setLast10(new Map(
        Object.entries(d.scores || {}).map(([k, v]) => [Number(k), v]))))
      .catch(() => { /* keep '—' cells */ });
  }, []);

  // Phase B (board #70): bug chips from open board tasks whose
  // affected_sids contains the row's sid, + per-sid notes counts.
  const [bugTasks, setBugTasks] = useState<Task[]>([]);
  const [noteCounts, setNoteCounts] = useState<Map<number, number>>(new Map());
  const [notesSid, setNotesSid] = useState<{ sid: number; name: string } | null>(null);
  const loadNotes = () => {
    apiFetch<{ sid: number }[]>('/api/strategy-notes')
      .then((rows) => {
        const m = new Map<number, number>();
        (rows || []).forEach((n) => m.set(n.sid, (m.get(n.sid) || 0) + 1));
        setNoteCounts(m);
      })
      .catch(() => { /* counts stay empty */ });
  };
  useEffect(() => {
    apiFetch<Task[]>('/api/dev-tasks?include_done=false')
      .then((ts) => setBugTasks((ts || []).filter(
        (t) => (t.affected_sids || []).length > 0)))
      .catch(() => { /* chips stay empty */ });
    loadNotes();
  }, []);

  const [sortKey, setSortKey] = useState<SortKey>('flags');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');
  const [activeFlag, setActiveFlag] = useState<StrategyHealthFlag | null>(null);
  const [includeLegacy, setIncludeLegacy] = useState(false);
  // 2026-05-27 — "Apples-to-apples" mode: when ON (default), the
  // phantom/missed/paired columns show the lag-tail-excluded counts.
  // The raw counts (full window) inflate when one source hasn't caught
  // up to the other; the fair counts only pair events that BOTH sides
  // had a chance to capture. Default ON so the page shows honest numbers.
  const filteredRows = useMemo(() => {
    if (!data?.rows) return [];
    let rows = data.rows;
    if (!includeLegacy) {
      rows = rows.filter(r => !r.red_flags.includes('legacy_no_confluence_id'));
    }
    if (activeFlag) {
      rows = rows.filter(r => r.red_flags.includes(activeFlag));
    }
    const sorted = [...rows].sort((a, b) => {
      const av = rowSortValue(a, sortKey, parityByPair, last10);
      const bv = rowSortValue(b, sortKey, parityByPair, last10);
      const cmp = typeof av === 'number' && typeof bv === 'number'
        ? av - bv
        : String(av).localeCompare(String(bv));
      return sortDir === 'asc' ? cmp : -cmp;
    });
    return sorted;
  }, [data, sortKey, sortDir, activeFlag, includeLegacy, parityByPair, last10]);

  // Summary: counts by flag, scoped to non-legacy strategies.
  const summary = useMemo(() => {
    const visible = (data?.rows ?? []).filter(
      r => includeLegacy || !r.red_flags.includes('legacy_no_confluence_id'));
    const flagCounts = new Map<StrategyHealthFlag, number>();
    let healthy = 0;
    for (const r of visible) {
      if (r.red_flags.length === 0) {
        healthy++;
      } else {
        for (const f of r.red_flags) {
          flagCounts.set(f, (flagCounts.get(f) ?? 0) + 1);
        }
      }
    }
    return { total: visible.length, healthy, flagCounts };
  }, [data, includeLegacy]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('asc'); }
  };

  const setSnapshotSub = useSetSnapshotSubscription();

  if (isLoading && !data) {
    return (
      <div className="p-6">
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Loading strategy health…</p>
      </div>
    );
  }
  if (error) {
    return (
      <div className="p-6">
        <p className="text-sm" style={{ color: 'var(--red)' }}>
          Failed to load: {String((error as any).message || error)}
        </p>
      </div>
    );
  }

  const lastUpdated = dataUpdatedAt
    ? new Date(dataUpdatedAt).toISOString().slice(11, 19)
    : '—';

  // Sorted list of flag chips for the filter row.
  const flagChips = Array.from(summary.flagCounts.entries())
    .sort((a, b) => b[1] - a[1]);

  const arrow = (k: SortKey) =>
    sortKey === k ? (sortDir === 'asc' ? ' ▲' : ' ▼') : '';

  return (
    <div className="p-4 space-y-4">
      <div>
        <h1 className="text-2xl font-semibold mb-1">Strategy Health</h1>
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          Per-strategy freshness + red flags. Change filters, then click Apply to load · last updated {lastUpdated} UTC.
        </p>
      </div>

      {/* Summary scoreboard */}
      <Card>
        <div className="flex flex-wrap items-center gap-3 text-xs"
             style={{ color: 'var(--text-muted)' }}>
          <div>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
              {summary.total}
            </span>
            {' strategies'}
          </div>
          <div>
            <span style={{ color: '#7fd081', fontWeight: 600 }}>
              {summary.healthy} healthy
            </span>
          </div>
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <span>Window:</span>
            {WINDOW_OPTIONS.map(o => (
              <button
                key={o.hours}
                onClick={() => { setMode('rolling'); setWindowHours(o.hours); }}
                className="px-2 py-0.5 rounded transition-colors"
                style={{
                  background: (mode === 'rolling' && windowHours === o.hours)
                    ? 'var(--accent)' : 'var(--bg-input)',
                  color: (mode === 'rolling' && windowHours === o.hours)
                    ? 'white' : 'var(--text-muted)',
                  border: (mode === 'rolling' && windowHours === o.hours)
                    ? 'none' : '1px solid var(--border)',
                  cursor: 'pointer',
                  fontSize: 11,
                }}
              >
                {o.label}
              </button>
            ))}
            <button
              key="custom"
              onClick={() => setMode('custom')}
              className="px-2 py-0.5 rounded transition-colors"
              style={{
                background: mode === 'custom' ? 'var(--accent)' : 'var(--bg-input)',
                color: mode === 'custom' ? 'white' : 'var(--text-muted)',
                border: mode === 'custom' ? 'none' : '1px solid var(--border)',
                cursor: 'pointer',
                fontSize: 11,
              }}
              title="Pin a fixed [start, end] window. Useful for overnight-into-morning analysis: set start tonight, then update end tomorrow."
            >
              Custom
            </button>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: 4, cursor: 'pointer', marginLeft: 8 }}>
              <input
                type="checkbox"
                checked={includeLegacy}
                onChange={e => setIncludeLegacy(e.target.checked)}
              />
              <span>include legacy</span>
            </label>
            <button
              onClick={() => { if (filtersDirty) applyFilters(); else refetch(); }}
              className="px-3 py-0.5 rounded"
              style={{
                background: filtersDirty ? 'var(--accent)' : 'var(--bg-input)',
                border: filtersDirty ? 'none' : '1px solid var(--border)',
                color: filtersDirty ? 'white' : 'var(--text-muted)',
                cursor: 'pointer',
                fontWeight: filtersDirty ? 600 : 400,
              }}
              title={filtersDirty
                ? 'Apply the selected window/dates and load them'
                : 'Reload the current window'}
            >
              {filtersDirty ? 'Apply' : 'Refresh'}
            </button>
          </div>
        </div>

        {/* Custom-mode datetime inputs — only visible when Custom is selected */}
        {mode === 'custom' && (
          <div className="mt-2 flex flex-wrap items-center gap-2 text-xs"
               style={{ color: 'var(--text-muted)' }}>
            <span>Range:</span>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
              <span>Start</span>
              <input
                type="datetime-local"
                value={customStartInput}
                onChange={e => setCustomStartInput(e.target.value)}
                style={{
                  padding: '2px 6px',
                  background: 'var(--bg-input)',
                  border: '1px solid var(--border)',
                  borderRadius: 3,
                  color: 'var(--text-primary)',
                  fontSize: 11,
                }}
              />
            </label>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
              <span>End</span>
              <input
                type="datetime-local"
                value={customEndInput}
                onChange={e => setCustomEndInput(e.target.value)}
                style={{
                  padding: '2px 6px',
                  background: 'var(--bg-input)',
                  border: '1px solid var(--border)',
                  borderRadius: 3,
                  color: 'var(--text-primary)',
                  fontSize: 11,
                }}
              />
            </label>
            <button
              onClick={() => setCustomEndInput(isoToLocalInput(new Date().toISOString()))}
              className="px-2 py-0.5 rounded"
              style={{
                background: 'var(--bg-input)',
                border: '1px solid var(--border)',
                color: 'var(--text-muted)',
                cursor: 'pointer',
                fontSize: 11,
              }}
              title="Set End to current time"
            >
              End → Now
            </button>
            <span style={{ opacity: 0.7 }}>
              (frozen — no auto-refresh in custom mode)
            </span>
          </div>
        )}

        {/* Filter chips */}
        {flagChips.length > 0 && (
          <div className="mt-2 flex flex-wrap items-center gap-1 text-xs">
            <span style={{ color: 'var(--text-muted)', marginRight: 6 }}>
              Filter:
            </span>
            <button
              onClick={() => setActiveFlag(null)}
              style={{
                ...flagChipStyle('gray'),
                outline: activeFlag === null ? '1px solid var(--accent)' : 'none',
                cursor: 'pointer',
              }}
            >
              all
            </button>
            {flagChips.map(([flag, count]) => (
              <button
                key={flag}
                onClick={() => setActiveFlag(activeFlag === flag ? null : flag)}
                style={{
                  ...flagChipStyle(FLAG_TONE[flag]),
                  outline: activeFlag === flag ? '1px solid var(--accent)' : 'none',
                  cursor: 'pointer',
                }}
              >
                {FLAG_LABEL[flag]} · {count}
              </button>
            ))}
          </div>
        )}
      </Card>

      {/* Main table */}
      <Card>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ color: 'var(--text-muted)', fontSize: 11, textAlign: 'left' }}>
                <Th onClick={() => toggleSort('sid')}       label={`SID${arrow('sid')}`} align="right" />
                <Th onClick={() => toggleSort('name')}      label={`Strategy${arrow('name')}`} />
                <Th onClick={() => toggleSort('symbol')}    label={`Symbol${arrow('symbol')}`} />
                <Th onClick={() => toggleSort('timeframe')} label={`TF${arrow('timeframe')}`} />
                <Th onClick={() => toggleSort('barParity')} label={`Bar Parity (data)${arrow('barParity')}`} align="right" />
                <Th onClick={() => toggleSort('btCurrent')} label={`BT Current${arrow('btCurrent')}`} />
                <Th onClick={() => toggleSort('kpis')}      label={`KPIs${arrow('kpis')}`} />
                <Th onClick={() => toggleSort('lastTrade')} label={`Last trade${arrow('lastTrade')}`} />
                <Th onClick={() => toggleSort('lastBt')}    label={`Last BT${arrow('lastBt')}`} />
                <Th onClick={() => toggleSort('lastAlert')} label={`Last alert${arrow('lastAlert')}`} />
                <Th onClick={() => toggleSort('trades')}    label={`Trades${arrow('trades')}`} align="right" />
                <Th onClick={() => toggleSort('combined')}  label={`Combined %${arrow('combined')}`} align="right" />
                <Th onClick={() => toggleSort('last10')}    label={`Last 10${arrow('last10')}`} align="right" />
                <Th onClick={() => toggleSort('paired')}    label={`Paired ${mode === 'custom' ? 'range' : windowLabel(windowHours)}${arrow('paired')}`} align="right" />
                <Th onClick={() => toggleSort('phantom')}   label={`Phantom ${mode === 'custom' ? 'range' : windowLabel(windowHours)}${arrow('phantom')}`} align="right" />
                <Th onClick={() => toggleSort('missed')}    label={`Missed ${mode === 'custom' ? 'range' : windowLabel(windowHours)}${arrow('missed')}`} align="right" />
                <Th onClick={() => toggleSort('tbd')}       label={`TBD${arrow('tbd')}`} align="right" />
                <th style={{ padding: '6px 8px', fontWeight: 500 }}
                    title="Snapshot subscription. ON = data-worker maintains a backtest snapshot. OFF = strategy is parked in the snapshot lane (alerts unaffected).">
                  Sub
                </th>
                <th style={{ padding: '6px 8px', fontWeight: 500 }}
                    title="Row context: notes (humans + nightly writers) and open board tasks affecting this sid.">
                  Context
                </th>
              </tr>
            </thead>
            <tbody>
              {filteredRows.map(r => {
                // W2-6a: one parity computation per (symbol, TF) pair —
                // strategies sharing a pair show the same number.
                const parity = parityByPair.get(
                  parityKey(r.symbol, r.timeframe));
                const l10 = last10.get(r.strategy_id);
                const l10Ratio = l10 && l10.denom > 0 ? l10.points / l10.denom : null;
                const rowBugs = bugTasks.filter(
                  (t) => (t.affected_sids || []).includes(r.strategy_id));
                const noteCount = noteCounts.get(r.strategy_id) || 0;
                return (
                <tr key={`${r.user_id}:${r.strategy_id}`}
                    style={{ borderTop: '1px solid var(--border)' }}>
                  <td style={{ padding: '6px 8px', textAlign: 'right',
                               fontVariantNumeric: 'tabular-nums' }}>
                    <a href={`/strategies/${r.strategy_id}`} target="_blank"
                       rel="noopener noreferrer"
                       title="open strategy detail in a new tab"
                       style={{ color: 'var(--blue)', textDecoration: 'none' }}>
                      {r.strategy_id}
                    </a>
                  </td>
                  <td style={{ padding: '6px 8px' }}>
                    <div style={{ color: 'var(--text-primary)' }}>
                      {r.name || `sid ${r.strategy_id}`}
                    </div>
                    <div style={{ color: 'var(--text-muted)', fontSize: 11 }}>
                      sid {r.strategy_id} · {r.direction} · {r.backtest_model || '—'}
                      {r.forward_testing ? ' · fwd' : ''}
                    </div>
                  </td>
                  <td style={{ padding: '6px 8px', fontVariantNumeric: 'tabular-nums' }}>
                    {r.symbol || '—'}
                  </td>
                  <td style={{ padding: '6px 8px' }}>{r.timeframe || '—'}</td>
                  {/* W2-6a (2026-07-07): Bar Parity % — messy-vs-clean live
                       data at a glance. Worst-of-OHLC field match rate,
                       live_bars (WS) vs bar_cache REST truth resampled to
                       the strategy TF, over the last ~2 trading hours
                       (≤240 bars, anchored at the newest settled live bar). */}
                  <td style={{ padding: '6px 8px', textAlign: 'right' }}
                      title={barParityTitle(parity)}>
                    <span style={barParityStyle(parity?.parity_pct)}>
                      {parity?.parity_pct == null
                        ? '—'
                        : `${parity.parity_pct.toFixed(1)}%`}
                    </span>
                  </td>
                  {/* W2-0 (2026-07-06): replaced the dead Snapshot column.
                       BT Current = age of the shadow engine's settled coverage
                       boundary (shadow_heartbeats.current_through) — honest
                       "how current is the backtest lane" even when the model
                       is quiet and Last BT ages. green ≤20m, amber ≤2h. */}
                  <td style={{ padding: '6px 8px' }}
                      title={r.bt_current_through
                        ? `current through ${r.bt_current_through}`
                          + ` · heartbeat ${r.bt_heartbeat_at ?? '—'}`
                        : 'no shadow heartbeat on file'}>
                    <span style={ageStyle(
                        ageSecFrom(r.bt_current_through, data?.now),
                        { green: 1200, yellow: 7200 })}>
                      {fmtAge(ageSecFrom(r.bt_current_through, data?.now))}
                    </span>
                  </td>
                  <td style={{ padding: '6px 8px' }}>
                    <span style={ageStyle(r.kpis_age_sec, { green: 3600, yellow: 86400 })}>
                      {fmtAge(r.kpis_age_sec)}
                    </span>
                  </td>
                  <td style={{ padding: '6px 8px' }}>
                    <span style={ageStyle(r.last_entry_age_sec, { green: 86400, yellow: 7 * 86400 })}>
                      {fmtAge(r.last_entry_age_sec)}
                    </span>
                  </td>
                  <td style={{ padding: '6px 8px' }}
                      title={r.last_backtest_created_at ?? 'no backtest trades on file'}>
                    <span style={ageStyle(r.last_backtest_created_age_sec, { green: 3600, yellow: 86400 })}>
                      {fmtAge(r.last_backtest_created_age_sec)}
                    </span>
                  </td>
                  <td style={{ padding: '6px 8px' }}
                      title={r.last_alert_at
                        ? r.last_alert_at
                        + (r.timestamps_upside_down && r.upside_down_delta_sec
                            ? ` (Δ ${Math.round((r.upside_down_delta_sec ?? 0) / 60)}m vs last BT — upside-down)`
                            : '')
                        : 'no alerts in last 30d'}>
                    <span style={ageStyle(r.last_alert_age_sec, { green: 3600, yellow: 86400 })}>
                      {fmtAge(r.last_alert_age_sec)}
                    </span>
                    {r.timestamps_upside_down && (
                      <span
                        style={{ marginLeft: 4, color: '#ffc107', fontSize: 10 }}
                        title="upside-down: alert and backtest timestamps >1h apart"
                      >
                        ⇅
                      </span>
                    )}
                  </td>
                  <td style={{ padding: '6px 8px', textAlign: 'right',
                               fontVariantNumeric: 'tabular-nums',
                               color: r.trade_count_backtest === 0
                                 ? 'var(--text-muted)' : 'var(--text-primary)' }}>
                    {r.trade_count_backtest}
                  </td>
                  {/* 2026-05-29 — apples-to-apples ON uses GLOBAL fair counts
                       (one fleet-wide cutoff). Per-strategy _fair shown in
                       tooltip for reference along with raw. */}
                  <td style={{ padding: '6px 8px', textAlign: 'right',
                               fontVariantNumeric: 'tabular-nums', fontWeight: 600,
                               color: combinedPctColor(r.combined_pct) }}
                      title={`Combined % = paired / (paired + phantom + missed) on `
                        + `apples-to-apples counts: `
                        + `${r.paired_count_global_fair}/(${r.paired_count_global_fair}`
                        + `+${r.phantom_count_global_fair}+${r.missed_count_global_fair}). `
                        + `Excludes ${r.tbd_count} TBD (lane not yet caught up).`}>
                    {r.combined_pct === null ? '—' : `${r.combined_pct.toFixed(1)}%`}
                  </td>
                  {/* Last-10 score (board #70): thresholds are the RATIO
                       equivalents of Kevin's n/20 bands (18/20=90%, 14/20=70%)
                       so shrunk denominators color consistently. */}
                  <td style={{ padding: '6px 8px', textAlign: 'right' }}
                      title="Last-10 completed backtest trades: 1 pt per entry/exit pairing to a live alert within ±10s. Click for per-trade detail.">
                    {l10Ratio == null
                      ? <span style={{ color: 'var(--text-muted)' }}>—</span>
                      : <button
                          onClick={() => setLast10Sid({
                            sid: r.strategy_id,
                            name: r.name || `sid ${r.strategy_id}` })}
                          style={{ background: 'transparent', border: 'none',
                                   cursor: 'pointer', padding: 0, fontSize: 13,
                                   fontWeight: 600,
                                   fontVariantNumeric: 'tabular-nums',
                                   color: l10Ratio >= 0.9 ? '#66bb6a'
                                     : l10Ratio >= 0.7 ? '#ffc107' : '#ef5350' }}>
                          {l10!.points}/{l10!.denom}
                        </button>}
                  </td>
                  <td style={{ padding: '6px 8px', textAlign: 'right',
                               fontVariantNumeric: 'tabular-nums',
                               color: r.paired_count_cov > 0
                                 ? '#7fd081' : 'var(--text-muted)' }}
                      title={`Paired (matched within tolerance). Per-strategy coverage-classified. raw: ${r.paired_count}`}>
                    {r.paired_count_cov || '—'}
                  </td>
                  <td style={{ padding: '6px 8px', textAlign: 'right',
                               fontVariantNumeric: 'tabular-nums',
                               color: r.phantom_count_cov > 0
                                 ? '#ffc107' : 'var(--text-muted)' }}
                      title={`Phantom (alert with no backtest trade, AND older than the backtest lane's coverage — a real divergence, not lag). raw: ${r.phantom_count}`}>
                    {r.phantom_count_cov || '—'}
                  </td>
                  <td style={{ padding: '6px 8px', textAlign: 'right',
                               fontVariantNumeric: 'tabular-nums',
                               color: r.missed_count_cov > 0
                                 ? '#ef5350' : 'var(--text-muted)' }}
                      title={`Missed (backtest trade with no alert). raw: ${r.missed_count}`}>
                    {r.missed_count_cov || '—'}
                  </td>
                  <td style={{ padding: '6px 8px', textAlign: 'right',
                               fontVariantNumeric: 'tabular-nums',
                               color: r.tbd_count > 0 ? 'var(--text)' : 'var(--text-muted)' }}
                      title="TBD = alerts newer than the backtest lane's coverage. Not phantoms — the reference hasn't caught up; they pair on the next append. Excluded from Combined %.">
                    {r.tbd_count || '—'}
                  </td>
                  <td style={{ padding: '6px 8px' }}>
                    <button
                      onClick={() => setSnapshotSub.mutate({
                        id: r.strategy_id,
                        enabled: !r.snapshot_subscribe_enabled,
                      })}
                      disabled={setSnapshotSub.isPending}
                      title={r.snapshot_subscribe_enabled
                        ? 'Subscribed — click to park (data-worker will skip this strategy)'
                        : 'Parked — click to resubscribe'}
                      style={{
                        background: 'transparent',
                        border: '1px solid var(--border)',
                        borderRadius: 3,
                        padding: '1px 6px',
                        fontSize: 11,
                        cursor: setSnapshotSub.isPending ? 'wait' : 'pointer',
                        color: r.snapshot_subscribe_enabled ? '#7fd081' : 'var(--text-muted)',
                      }}
                    >
                      {r.snapshot_subscribe_enabled ? 'ON' : 'OFF'}
                    </button>
                  </td>
                  {/* Context (Kevin 07-25): replaces the Flags chip column —
                       flag data still drives the summary/filter chips up top.
                       📝 notes on top, open affecting-task chips beneath. */}
                  <td style={{ padding: '6px 8px', verticalAlign: 'top' }}>
                    <button
                      onClick={() => setNotesSid({
                        sid: r.strategy_id,
                        name: r.name || `sid ${r.strategy_id}` })}
                      title="strategy notes — click to read/add"
                      style={{ background: 'transparent', border: 'none',
                               cursor: 'pointer', padding: 0, fontSize: 12.5,
                               display: 'block',
                               color: noteCount > 0 ? 'var(--text)' : 'var(--text-muted)' }}>
                      📝{noteCount > 0 ? ` ${noteCount}` : ''}
                    </button>
                    {rowBugs.map((t) => (
                      <a key={t.id} href={`/admin/tasks?task=${t.id}`}
                         target="_blank" rel="noopener noreferrer"
                         title={t.title}
                         style={{
                           display: 'block', width: 'fit-content',
                           padding: '0 6px', borderRadius: 8, fontSize: 10.5,
                           marginTop: 3, border: '1px solid #ef5350',
                           color: '#ef5350', textDecoration: 'none',
                           whiteSpace: 'nowrap',
                         }}>
                        🐛 #{t.id}
                      </a>
                    ))}
                  </td>
                </tr>
                );
              })}
              {filteredRows.length === 0 && (
                <tr>
                  <td colSpan={17} style={{ padding: 16, textAlign: 'center',
                                            color: 'var(--text-muted)' }}>
                    No strategies match the current filter.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {last10Sid && (
        <Last10PairingModal sid={last10Sid.sid} name={last10Sid.name}
          onClose={() => setLast10Sid(null)} />
      )}
      {notesSid && (
        <StrategyNotesModal sid={notesSid.sid} name={notesSid.name}
          onClose={() => setNotesSid(null)} onChanged={loadNotes} />
      )}
    </div>
  );
}

function Th({ label, onClick, align = 'left' }: {
  label: string;
  onClick: () => void;
  align?: 'left' | 'right';
}) {
  return (
    <th
      onClick={onClick}
      style={{
        padding: '6px 8px',
        cursor: 'pointer',
        userSelect: 'none',
        textAlign: align,
        fontWeight: 500,
      }}
    >
      {label}
    </th>
  );
}
