'use client';

/**
 * Nested per-result drill-down for a single mass search (1.17).
 *
 * Renders below an expanded Mass-Results card. Results are grouped by "dough"
 * (the base entry trigger — the expensive precompute the confluences ride on),
 * each dough a collapsible row; expanding a result shows an inline equity
 * preview + KPIs. Reads the live `results` array (1.16 incremental persistence),
 * so rows stream in as the search runs.
 */

import { useState, useMemo } from 'react';
import { useMassResult } from '@/hooks/queries/useMassBuilder';

/* ---- helpers ---- */

function num(v: any): number | null {
  const n = typeof v === 'number' ? v : Number(v);
  return Number.isFinite(n) ? n : null;
}

/** Coerce a result's equity_curve into a flat number[] of cumulative R. */
function toCurve(ec: any): number[] {
  if (!Array.isArray(ec)) return [];
  return ec
    .map((p) => (typeof p === 'number' ? p : num(p?.r ?? p?.value ?? p?.cumR ?? p?.equity)))
    .filter((n): n is number => n != null);
}

function fmtR(v: any): string {
  const n = num(v);
  if (n == null) return '--';
  return (n >= 0 ? '+' : '') + n.toFixed(2);
}
function fmtPct(v: any): string {
  const n = num(v);
  return n == null ? '--' : `${n.toFixed(1)}%`;
}
function fmtNum(v: any, dp = 2): string {
  const n = num(v);
  return n == null ? '--' : n.toFixed(dp);
}

const muted: React.CSSProperties = { color: 'var(--text-muted)' };

/* ---- trigger-set ("dough") identity + readable labels ---- */

function stopLabel(sc: any): string {
  if (!sc) return 'none';
  const m = sc.method || '?';
  if (m === 'atr') return `ATR${sc.atr_mult != null ? ` ${sc.atr_mult}×` : ''}`;
  if (m === 'swing') return `swing${sc.lookback != null ? ` lb${sc.lookback}` : ''}${sc.padding != null ? ` p${sc.padding}` : ''}`;
  if (m === 'fixed_dollar' || m === 'fixed') return `fixed${sc.value != null ? ` $${sc.value}` : ''}`;
  return String(m);
}

/** A trigger-set = the (entry × exit × stop × target) "dough" the confluences
 *  ride on — the expensive part. Group results by it; rows differ by confluence. */
function triggerSet(r: any): { key: string; entry: string; exit: string; stop: string; target: string } {
  const cfg = r?.config || {};
  const entry = cfg.entry_trigger_name || cfg.entry_trigger || '(entry?)';
  const exitsArr = cfg.exit_trigger_names || cfg.exit_triggers || (cfg.exit_trigger ? [cfg.exit_trigger] : []);
  const exit = Array.isArray(exitsArr) ? (exitsArr.join(' + ') || 'signal/none') : String(exitsArr || 'signal/none');
  const stop = stopLabel(cfg.stop_config);
  const target = cfg.target_config?.method ? stopLabel(cfg.target_config) : 'signal';
  const rawExits = [...(cfg.exit_triggers || (cfg.exit_trigger ? [cfg.exit_trigger] : []))].sort().join(',');
  const key = [cfg.entry_trigger || entry, rawExits, JSON.stringify(cfg.stop_config || {}), JSON.stringify(cfg.target_config || null)].join('||');
  return { key, entry, exit, stop, target };
}

/* ---- inline equity sparkline (real result data, lightweight SVG) ---- */

function EquitySparkline({ curve, w = 220, h = 48 }: { curve: number[]; w?: number; h?: number }) {
  if (curve.length < 2) {
    return <span className="text-[10px]" style={muted}>no equity points</span>;
  }
  const min = Math.min(...curve, 0);
  const max = Math.max(...curve, 0);
  const range = max - min || 1;
  const dx = w / (curve.length - 1);
  const y = (v: number) => h - ((v - min) / range) * h;
  const pts = curve.map((v, i) => `${(i * dx).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const last = curve[curve.length - 1];
  const color = last >= 0 ? 'var(--green)' : 'var(--red)';
  const zeroY = y(0);
  return (
    <svg width={w} height={h} style={{ display: 'block' }}>
      <line x1={0} y1={zeroY} x2={w} y2={zeroY} stroke="var(--border)" strokeWidth={1} strokeDasharray="3 3" />
      <polyline points={pts} fill="none" stroke={color} strokeWidth={1.5} />
    </svg>
  );
}

/* ---- one result row (a single combo) ---- */

interface MassResult {
  config?: any;
  kpis?: any;
  equity_curve?: any;
  confluence_str?: string;
  status?: string;
  oos_kpis?: any;
  hifi?: any;
}

function confLabel(r: MassResult): string {
  const c = (r.confluence_str || '').trim();
  return !c || c === 'None' ? '— base (no confluence) —' : c;
}

function KpiGrid({ k, cols = 2 }: { k: any; cols?: number }) {
  const rows: [string, string][] = [
    ['Daily R', fmtR(k.daily_r)],
    ['Win rate', fmtPct(k.win_rate)],
    ['Profit factor', fmtNum(k.profit_factor)],
    ['Trades', String(num(k.total_trades) ?? '--')],
    ['Total R', fmtR(k.total_r)],
    ['Max DD', fmtNum(k.max_drawdown ?? k.max_r_drawdown)],
    ['R²', fmtNum(k.r_squared)],
  ];
  return (
    <div className="grid gap-x-5 gap-y-1 text-[11px]" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
      {rows.map(([label, val]) => (
        <div key={label} className="flex justify-between gap-3">
          <span style={muted}>{label}</span>
          <span className="font-medium">{val}</span>
        </div>
      ))}
    </div>
  );
}

function ResultRow({ r }: { r: MassResult }) {
  const [open, setOpen] = useState(false);
  const cfg = r.config || {};
  const k = r.kpis || {};
  const isBase = confLabel(r).startsWith('—');
  const curve = useMemo(() => toCurve(r.equity_curve), [r.equity_curve]);
  const dailyR = num(k.daily_r);

  return (
    <div style={{ borderTop: '1px solid var(--border)' }}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full grid items-center gap-2 px-2 py-1.5 text-left"
        style={{ gridTemplateColumns: '16px 1fr 70px 64px 56px 64px', fontSize: '0.72rem' }}
      >
        <span style={{ ...muted, fontSize: '0.6rem' }}>{open ? '▼' : '▶'}</span>
        <span className="truncate" style={{ color: isBase ? 'var(--text-muted)' : 'var(--text-secondary)', fontFamily: 'var(--font-mono, monospace)' }}>
          {confLabel(r)}
        </span>
        <span className="font-bold" style={{ color: (dailyR ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>{fmtR(k.daily_r)}</span>
        <span style={muted}>{fmtPct(k.win_rate)}</span>
        <span style={muted}>{fmtNum(k.profit_factor)}</span>
        <span style={muted}>{num(k.total_trades) ?? '--'} tr</span>
      </button>
      {open && (
        <div className="px-6 pb-3 pt-2 space-y-3" style={{ background: 'var(--bg-input)' }}>
          {/* Preview (post-filter KPIs — instant) */}
          <div className="flex flex-wrap gap-6 items-start">
            <div>
              <p className="text-[10px] mb-1" style={muted}>Equity · preview KPIs</p>
              <EquitySparkline curve={curve} />
            </div>
            <KpiGrid k={k} />
            <div className="text-[11px] space-y-0.5" style={muted}>
              <div>exit: <span style={{ color: 'var(--text-secondary)' }}>{(cfg.exit_triggers || []).join(', ') || cfg.exit_trigger || '—'}</span></div>
              <div>stop: <span style={{ color: 'var(--text-secondary)' }}>{cfg.stop_config?.method || '—'}</span></div>
              <div>target: <span style={{ color: 'var(--text-secondary)' }}>{cfg.target_config?.method || 'signal'}</span></div>
              {r.oos_kpis && <div style={{ color: 'var(--accent)' }}>OOS daily R: {fmtR(r.oos_kpis?.daily_r)}</div>}
            </div>
          </div>
          {/* Engine-accurate inline re-run is deferred: the generic
              /api/backtest/run endpoint can't reproduce a mass result (it loads
              the visible window with NO indicator warmup → cold indicators →
              0 trades). Needs a warmup-aware rerun endpoint (reuse
              load_strategy_data's warmup + visible-trim). Preview KPIs stand. */}
        </div>
      )}
    </div>
  );
}

/* ---- a trigger-set ("dough") group: one entry × exit × stop ---- */

interface TriggerSetGroup { key: string; entry: string; exit: string; stop: string; target: string; results: MassResult[]; best: number; }

function DoughGroup({ g, defaultOpen }: { g: TriggerSetGroup; defaultOpen: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  const titleAttr = `entry: ${g.entry}\nexit: ${g.exit}\nstop: ${g.stop} · target: ${g.target}`;
  return (
    <div className="rounded-lg overflow-hidden" style={{ border: '1px solid var(--border)' }}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-3 py-2 gap-3"
        style={{ background: 'var(--bg-card)' }}
        title={titleAttr}
      >
        <div className="flex items-center gap-2 min-w-0">
          <span style={{ ...muted, fontSize: '0.65rem' }}>{open ? '▼' : '▶'}</span>
          <div className="min-w-0 text-left">
            <div className="text-xs font-semibold truncate" style={{ color: 'var(--text-secondary)' }}>
              <span style={{ color: 'var(--green)' }}>{g.entry}</span>
              <span style={muted}> → </span>
              <span style={{ color: 'var(--red)' }}>{g.exit}</span>
            </div>
            <div className="text-[10px] truncate" style={muted}>
              stop: <span style={{ color: 'var(--text-secondary)' }}>{g.stop}</span> · target: <span style={{ color: 'var(--text-secondary)' }}>{g.target}</span>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className="text-[10px] px-1.5 py-0.5 rounded-full" style={{ background: 'var(--bg-input)', ...muted }}>
            {g.results.length} {g.results.length === 1 ? 'confluence' : 'confluences'}
          </span>
          <span className="text-xs" style={muted}>
            best <span className="font-bold" style={{ color: (g.best ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>{fmtR(g.best)}</span>
          </span>
        </div>
      </button>
      {open && (
        <div>
          <div className="grid gap-2 px-2 py-1" style={{ gridTemplateColumns: '16px 1fr 70px 64px 56px 64px', fontSize: '0.6rem', ...muted, background: 'var(--bg-input)' }}>
            <span></span><span>confluence</span><span>daily R</span><span>WR</span><span>PF</span><span>trades</span>
          </div>
          {g.results.map((r, i) => <ResultRow key={i} r={r} />)}
        </div>
      )}
    </div>
  );
}

/* ---- the panel ---- */

export default function MassSearchResultsPanel({ searchId, isRunning }: { searchId: string; isRunning: boolean }) {
  const { data, isLoading } = useMassResult(searchId);
  const results: MassResult[] = Array.isArray(data?.results) ? data!.results : [];

  const groups = useMemo<TriggerSetGroup[]>(() => {
    const map = new Map<string, TriggerSetGroup>();
    for (const r of results) {
      const ts = triggerSet(r);
      let g = map.get(ts.key);
      if (!g) { g = { ...ts, results: [], best: -1e9 }; map.set(ts.key, g); }
      g.results.push(r);
    }
    const arr = Array.from(map.values());
    for (const g of arr) {
      g.results.sort((a, b) => (num(b.kpis?.daily_r) ?? -1e9) - (num(a.kpis?.daily_r) ?? -1e9));
      g.best = num(g.results[0]?.kpis?.daily_r) ?? -1e9;
    }
    arr.sort((a, b) => b.best - a.best);
    return arr;
  }, [results]);

  // Fail-loud: surface combos that errored out (cross-pack drops, data gaps).
  const diag = (data?.summary?.diagnostics) || {};
  const failed = num(diag.backtests_failed) ?? 0;

  return (
    <div className="mt-3 pt-3 space-y-2" style={{ borderTop: '1px solid var(--border)' }}>
      <div className="flex items-center gap-2 flex-wrap">
        <p className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>
          {results.length} result{results.length === 1 ? '' : 's'} across {groups.length} trigger-set{groups.length === 1 ? '' : 's'}
        </p>
        {isRunning && (
          <span className="text-[10px] px-1.5 py-0.5 rounded-full flex items-center gap-1" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
            <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: 'var(--accent)' }} />
            streaming live
          </span>
        )}
        {failed > 0 && (
          <span className="text-[10px] px-1.5 py-0.5 rounded-full" title="Some trigger backtests errored and were skipped — results may be incomplete."
                style={{ background: 'rgba(255,80,80,0.15)', color: 'var(--red)' }}>
            ⚠ {failed} backtest{failed === 1 ? '' : 's'} failed
          </span>
        )}
      </div>

      {isLoading && results.length === 0 && (
        <p className="text-xs" style={muted}>Loading results…</p>
      )}
      {!isLoading && results.length === 0 && (
        <p className="text-xs" style={muted}>
          {isRunning ? 'No results yet — the dough is baking. Rows appear as each confluence lands.' : 'No results recorded for this search.'}
        </p>
      )}

      {groups.map((g, i) => (
        <DoughGroup key={g.key} g={g} defaultOpen={i === 0 && groups.length <= 3} />
      ))}
    </div>
  );
}
