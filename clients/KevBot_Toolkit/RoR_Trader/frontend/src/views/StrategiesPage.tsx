'use client';
/**
 * My Strategies — Clean API-first page, V5 design spec.
 * Data layer built around actual Supabase API response shapes.
 */
import { useState, useMemo, useEffect } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import SparkLine from '@/charts/SparkLine';
import { useStrategies } from '@/hooks/queries/useStrategies';
import { useDeleteStrategy, useDuplicateStrategy } from '@/hooks/mutations/useStrategyMutations';

// --- Types (match API response) ---
interface KpiSet { win_rate?: number; profit_factor?: number; daily_r?: number; daily_roi?: number; trades_per_day?: number; max_r_drawdown?: number; }
interface Strategy {
  id: number; name: string; symbol: string; direction: 'LONG' | 'SHORT'; timeframe: string;
  trading_session?: string; forward_testing?: boolean; forward_test_start?: string;
  alert_tracking_enabled?: boolean; tags?: string[]; data_days?: number;
  kpis?: KpiSet & { total_r?: number; avg_r?: number; r_squared?: number; total_trades?: number; total_pnl?: number; final_balance?: number; };
  fwd_kpis?: KpiSet; alert_kpis?: KpiSet;
  equity_curve_data?: { exit_times?: string[]; cumulative_r?: number[]; boundary_index?: number | null; };
  entry_trigger_confluence_id?: string; exit_trigger_confluence_ids?: string[];
  confluence?: string[]; stop_config?: { method?: string; [k: string]: any };
  target_config?: { method?: string; [k: string]: any } | null; created_at?: string;
}

// --- Constants ---
const BT = '#2196F3', FWD = '#FF9800', LIVE = '#4CAF50', EXEC = '#2196F3', FID = '#26C6DA';
const selCSS: React.CSSProperties = { padding: '6px 10px', borderRadius: '6px', border: '1px solid var(--border)', background: 'var(--bg-input)', color: 'var(--text-primary)', fontSize: '12px' };
const PULSE = `@keyframes pulse{0%{transform:scale(1);opacity:.5}100%{transform:scale(2.5);opacity:0}}`;
const STATUS_CLR: Record<string, string> = { 'On Track': 'var(--green)', Outperforming: '#2196F3', Underperforming: '#FF9800', 'Insufficient Data': 'var(--text-muted)', 'Backtest Only': 'var(--text-muted)' };

// --- Helpers ---
const fmtK = (v?: number | null, d = 2, s = '') => v == null ? '--' : `${v >= 0 ? '+' : ''}${v.toFixed(d)}${s}`;
const daysSince = (s?: string) => !s ? 0 : Math.max(0, Math.round((Date.now() - new Date(s).getTime()) / 86400000));
const tog = (on: boolean): React.CSSProperties => ({ background: on ? 'var(--accent-muted)' : 'var(--bg-input)', color: on ? 'var(--accent)' : 'var(--text-muted)', border: on ? '1px solid var(--accent)' : '1px solid var(--border)' });
function fmtStop(c?: { method?: string; [k: string]: any } | null) { if (!c?.method) return 'None'; const m = c.method; if (m === 'atr') return `ATR ${c.atr_mult||1.5}x`; if (m === 'fixed_dollar') return `$${c.amount||'?'} fixed`; if (m === 'percentage') return `${c.percentage||'?'}%`; if (m === 'swing') return `Swing ${c.lookback||5}-bar`; return m; }
function fmtTarget(c?: { method?: string; [k: string]: any } | null) { if (!c?.method) return 'Signal exit only'; const m = c.method; if (m === 'risk_reward') return `${c.rr_ratio||2}:1 R:R`; if (m === 'atr') return `ATR ${c.atr_mult||2}x`; if (m === 'fixed_dollar') return `$${c.amount||'?'}`; return m; }
function parseExec(s: string) { const m = s.match(/^(\[[A-Z0-9]+\])\s*(.+)$/); return m ? { tag: m[1], rest: m[2] } : { tag: '', rest: s }; }
function deriveStatus(s: Strategy) { return s.forward_testing ? 'On Track' : 'Backtest Only'; }

// --- Inline sub-components ---
function ExecBadge({ tag }: { tag: string }) {
  if (!tag) return null;
  return <span className="font-medium px-1 py-0.5 rounded-full mr-1" style={{ color: EXEC, background: EXEC + '20', fontSize: '10px' }}>{tag}</span>;
}

function KpiComparisonTable({ kpiMode, k, fk, ak, btTPD, fwdDays }: { kpiMode: string; k: Strategy['kpis']; fk?: KpiSet; ak?: KpiSet; btTPD: number; fwdDays: number }) {
  const btRow = { label: 'Backtest', color: BT, wr: k?.win_rate, pf: k?.profit_factor, dr: k?.daily_r, roi: k?.daily_roi, tpd: btTPD, dd: k?.max_r_drawdown };
  const fwdRow = { label: 'Forward', color: FWD, wr: fk?.win_rate, pf: fk?.profit_factor, dr: fk?.daily_r, roi: fk?.daily_roi, tpd: fk?.trades_per_day ?? null, dd: fk?.max_r_drawdown };
  const alertRow = { label: 'Alerts', color: LIVE, wr: ak?.win_rate, pf: ak?.profit_factor, dr: ak?.daily_r, roi: ak?.daily_roi, tpd: ak?.trades_per_day ?? null, dd: ak?.max_r_drawdown };
  const rowA = kpiMode === 'FWD vs Alerts' ? fwdRow : btRow;
  const rowB = kpiMode === 'BT vs FWD' ? fwdRow : alertRow;
  const cell = (v?: number | null, s = '') => v != null ? `${v >= 0 && s !== '%' && s !== 'R' ? '+' : ''}${v.toFixed(s === '%' ? 1 : 2)}${s}` : '--';
  const delta = (a?: number | null, b?: number | null) => {
    if (a == null || b == null) return <span>--</span>;
    const d = a - b; const c = d > 0 ? 'var(--green)' : d < 0 ? 'var(--red)' : 'var(--text-muted)';
    return <span style={{ color: c }}>{d >= 0 ? '+' : ''}{d.toFixed(1)}</span>;
  };
  const Row = ({ r, bg }: { r: typeof rowA; bg?: string }) => (
    <div className="grid grid-cols-7 text-xs px-2 py-1.5 border-t" style={{ borderColor: 'var(--border)', background: bg }}>
      <span className="text-[10px] font-medium" style={{ color: r.color }}>{r.label}</span>
      <span>{r.wr != null ? `${r.wr.toFixed(1)}%` : '--'}</span><span>{r.pf != null ? r.pf.toFixed(2) : '--'}</span>
      <span>{cell(r.dr)}</span><span>{r.roi != null ? `${r.roi.toFixed(2)}%` : '--'}</span>
      <span>{r.tpd != null ? r.tpd.toFixed(1) : '--'}</span><span>{r.dd != null ? `${r.dd.toFixed(1)}R` : '--'}</span>
    </div>
  );
  return (
    <div className="mb-3 rounded-lg overflow-hidden border" style={{ borderColor: 'var(--border)' }}>
      <div className="grid grid-cols-7 text-[10px] font-medium px-2 py-1.5" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
        <span /><span>WR</span><span>PF</span><span>Daily R</span><span>Daily ROI</span><span>TPD</span><span>Max DD</span>
      </div>
      <Row r={rowA} /><Row r={rowB} />
      <div className="grid grid-cols-7 text-xs px-2 py-1.5 border-t" style={{ borderColor: 'var(--border)', background: 'var(--bg-input)' }}>
        <span className="text-[10px] font-medium" style={{ color: 'var(--text-muted)' }}>{'\u0394'}</span>
        <span>{delta(rowB.wr, rowA.wr)}</span><span>{delta(rowB.pf, rowA.pf)}</span>
        <span>{delta(rowB.dr, rowA.dr)}</span><span>{delta(rowB.roi, rowA.roi)}</span>
        <span>{delta(rowB.tpd, rowA.tpd)}</span><span>{delta(rowB.dd, rowA.dd)}</span>
      </div>
    </div>
  );
}

// --- Main Component ---
export default function StrategiesPage() {
  const { data: strategies, isLoading, error } = useStrategies();
  const deleteMut = useDeleteStrategy();
  const dupMut = useDuplicateStrategy();

  useEffect(() => { if (typeof document !== 'undefined' && !document.getElementById('strat-pulse')) { const s = document.createElement('style'); s.id = 'strat-pulse'; s.textContent = PULSE; document.head.appendChild(s); } }, []);

  const [tickerF, setTickerF] = useState('All');
  const [dirF, setDirF] = useState('All');
  const [tagF, setTagF] = useState('All');
  const [sortBy, setSortBy] = useState('Newest First');
  const [kpiMode, setKpiMode] = useState('Overall');
  const [chartH, setChartH] = useState(64);
  const [xAxis, setXAxis] = useState<'time'|'trade'>('time');
  const [hwm, setHwm] = useState(true);
  const [edge, setEdge] = useState(false);
  const [conf, setConf] = useState(false);
  const [selIds, setSelIds] = useState<Set<number>>(new Set());

  const toggleSel = (id: number) => setSelIds(p => { const n = new Set(p); n.has(id) ? n.delete(id) : n.add(id); return n; });

  const tickers = useMemo(() => !strategies ? [] : Array.from(new Set(strategies.map((s: Strategy) => s.symbol))).sort(), [strategies]);
  const tags = useMemo(() => !strategies ? [] : Array.from(new Set(strategies.flatMap((s: Strategy) => s.tags || []))).sort(), [strategies]);

  const filtered = useMemo(() => {
    if (!strategies) return [];
    let r = [...strategies] as Strategy[];
    if (tickerF !== 'All') r = r.filter(s => s.symbol === tickerF);
    if (dirF !== 'All') r = r.filter(s => s.direction === dirF);
    if (tagF !== 'All') r = r.filter(s => (s.tags || []).includes(tagF));
    if (sortBy === 'Newest First') r.sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''));
    else if (sortBy === 'Name A-Z') r.sort((a, b) => a.name.localeCompare(b.name));
    else if (sortBy === 'Win Rate (High)') r.sort((a, b) => (b.kpis?.win_rate || 0) - (a.kpis?.win_rate || 0));
    else if (sortBy === 'Profit Factor (High)') r.sort((a, b) => (b.kpis?.profit_factor || 0) - (a.kpis?.profit_factor || 0));
    else if (sortBy === 'Daily R (High)') r.sort((a, b) => (b.kpis?.daily_r || 0) - (a.kpis?.daily_r || 0));
    return r;
  }, [strategies, tickerF, dirF, tagF, sortBy]);

  // Loading
  if (isLoading) return (
    <div>
      <PageHeader title="My Strategies" subtitle="Loading..." />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
        {[1,2,3,4].map(i => (
          <Card key={i}><div className="animate-pulse space-y-3">
            <div className="h-4 rounded w-1/3" style={{ background: 'var(--border)' }} />
            <div className="h-3 rounded w-2/3" style={{ background: 'var(--border)' }} />
            <div className="h-16 rounded" style={{ background: 'var(--bg-input)' }} />
            <div className="grid grid-cols-6 gap-2">{[1,2,3,4,5,6].map(j => <div key={j} className="h-8 rounded" style={{ background: 'var(--border)' }} />)}</div>
          </div></Card>
        ))}
      </div>
    </div>
  );

  // Error
  if (error) return (
    <div>
      <PageHeader title="My Strategies" subtitle="Error loading strategies" />
      <Card><div className="text-center py-8" style={{ color: 'var(--red)' }}>Failed to load strategies. Check your connection and try again.</div></Card>
    </div>
  );

  return (
    <div>
      <PageHeader
        title="My Strategies"
        subtitle={`${filtered.length} strateg${filtered.length === 1 ? 'y' : 'ies'}`}
        actions={
          <div className="flex items-center gap-2">
            <span className="text-xs px-2 py-1 rounded-full flex items-center gap-1.5" style={{ background: 'var(--green)15', color: 'var(--green)' }}>
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--green)' }} />Live
            </span>
            <Link href="/strategy-builder"><button className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: '#fff' }}>+ New Strategy</button></Link>
          </div>
        }
      />

      {/* Bulk action bar */}
      {selIds.size > 0 && (
        <div className="flex items-center gap-4 mt-4 mb-3 px-4 py-3 rounded-lg" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
          <span className="text-sm" style={{ color: 'var(--text-muted)' }}>{selIds.size} selected</span>
          <button className="text-sm px-3 py-1 rounded" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)', cursor: 'pointer' }} onClick={() => setSelIds(new Set(filtered.map(s => s.id)))}>Select All</button>
          <button className="text-sm px-3 py-1 rounded" style={{ background: 'var(--red)' + '18', color: 'var(--red)', border: 'none', cursor: 'pointer' }} onClick={() => { if (confirm(`Delete ${selIds.size} strategies?`)) { selIds.forEach(id => deleteMut.mutate(id)); setSelIds(new Set()); } }}>Delete Selected</button>
          <button className="text-sm px-3 py-1 rounded font-medium" style={{ background: 'var(--accent)', color: '#fff', border: 'none', cursor: 'pointer' }}>Create Portfolio</button>
          <span className="flex-1" />
          <button className="text-sm px-3 py-1 rounded" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-muted)', cursor: 'pointer' }} onClick={() => setSelIds(new Set())}>Clear</button>
        </div>
      )}

      {/* Filter row */}
      <div className="flex flex-wrap gap-2 mb-3 mt-4">
        <select style={selCSS} value={tickerF} onChange={e => setTickerF(e.target.value)}><option value="All">All Tickers</option>{tickers.map(t => <option key={t} value={t}>{t}</option>)}</select>
        <select style={selCSS} value={dirF} onChange={e => setDirF(e.target.value)}><option value="All">All Directions</option><option value="LONG">LONG</option><option value="SHORT">SHORT</option></select>
        <select style={selCSS} value={tagF} onChange={e => setTagF(e.target.value)}><option value="All">All Tags</option>{tags.map(t => <option key={t} value={t}>{t}</option>)}</select>
        <select style={selCSS} value={sortBy} onChange={e => setSortBy(e.target.value)}>{['Newest First','Name A-Z','Win Rate (High)','Profit Factor (High)','Daily R (High)'].map(s => <option key={s} value={s}>{s}</option>)}</select>
      </div>

      {/* Viewing preferences */}
      <div className="flex flex-wrap items-center gap-4 mb-4 text-[10px]" style={{ color: 'var(--text-muted)' }}>
        <div className="flex items-center gap-1.5">
          <span>KPIs:</span>
          <select className="px-2 py-1 rounded text-[10px] font-medium" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} value={kpiMode} onChange={e => setKpiMode(e.target.value)}>
            <option value="Overall">Overall</option><option value="BT vs FWD">Backtest vs Forward</option><option value="FWD vs Alerts">Forward vs Alerts</option><option value="BT vs Alerts">Backtest vs Alerts</option>
          </select>
        </div>
        <div className="w-px h-4" style={{ background: 'var(--border)' }} />
        <div className="flex items-center gap-1.5">
          <span>Chart Height:</span>
          {[{v:48,l:'S'},{v:64,l:'M'},{v:96,l:'L'},{v:140,l:'XL'}].map(o => <button key={o.v} onClick={() => setChartH(o.v)} className="px-2 py-1 rounded font-medium" style={tog(chartH===o.v)}>{o.l}</button>)}
        </div>
        <div className="w-px h-4" style={{ background: 'var(--border)' }} />
        <div className="flex items-center gap-1.5">
          <span>X-axis:</span>
          {(['time','trade'] as const).map(m => <button key={m} onClick={() => setXAxis(m)} className="px-2 py-1 rounded font-medium" style={tog(xAxis===m)}>{m==='time'?'Time':'Trade #'}</button>)}
        </div>
        <div className="w-px h-4" style={{ background: 'var(--border)' }} />
        <div className="flex items-center gap-1.5"><span>HWM:</span>{(['On','Off'] as const).map(v => <button key={v} onClick={() => setHwm(v==='On')} className="px-2 py-1 rounded font-medium" style={tog((hwm?'On':'Off')===v)}>{v}</button>)}</div>
        <div className="w-px h-4" style={{ background: 'var(--border)' }} />
        <div className="flex items-center gap-1.5"><span>Edge Check:</span>{(['On','Off'] as const).map(v => <button key={v} onClick={() => setEdge(v==='On')} className="px-2 py-1 rounded font-medium" style={tog((edge?'On':'Off')===v)}>{v}</button>)}</div>
        <div className="w-px h-4" style={{ background: 'var(--border)' }} />
        <div className="flex items-center gap-1.5"><span>Confidence Bands:</span>{(['On','Off'] as const).map(v => <button key={v} onClick={() => setConf(v==='On')} className="px-2 py-1 rounded font-medium" style={tog((conf?'On':'Off')===v)}>{v}</button>)}</div>
      </div>

      {/* Empty state */}
      {filtered.length === 0 && (
        <Card><div className="text-center py-12">
          <p className="text-lg mb-2" style={{ color: 'var(--text-secondary)' }}>{(strategies?.length||0)===0?'No strategies yet. Create your first strategy!':'No strategies match the current filters.'}</p>
          {(strategies?.length||0)===0 && <Link href="/strategy-builder"><button className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: '#fff' }}>Go to Strategy Builder</button></Link>}
        </div></Card>
      )}

      {/* Strategy cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {filtered.map(strat => {
          const k = strat.kpis || {};
          const eq = strat.equity_curve_data;
          const fwdDays = daysSince(strat.forward_test_start);
          const status = deriveStatus(strat);
          const sClr = STATUS_CLR[status] || 'var(--text-muted)';
          const btDays = strat.data_days || 30;
          const btTPD = k.trades_per_day ?? (btDays > 0 ? (k.total_trades || 0) / btDays : 0);

          return (
            <Card key={strat.id}>
              {/* Header */}
              <div className="flex items-center gap-2 mb-1">
                {strat.alert_tracking_enabled && (
                  <div className="relative flex-shrink-0 w-2.5 h-2.5">
                    <div className="w-2.5 h-2.5 rounded-full" style={{ background: 'var(--green)' }} />
                    <div className="w-2.5 h-2.5 rounded-full absolute top-0 left-0" style={{ background: 'var(--green)', animation: 'pulse 2s ease-out infinite', opacity: 0.5 }} />
                  </div>
                )}
                <Link href={`/strategies/${strat.id}`} className="font-semibold text-base hover:underline" style={{ color: 'var(--text-primary)' }}>{strat.name}</Link>
                <span className="text-xs px-2 py-0.5 rounded-full font-medium" style={{ color: sClr, background: sClr + '20' }}>{status}</span>
                <span className="flex-1" />
                {/* Sigma badges */}
                {strat.forward_testing && (
                  <div className="flex items-center gap-1 flex-shrink-0">
                    <span className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded" style={{ color: FWD, background: FWD + '18' }}>{'\u2014\u03c3'}</span>
                    {strat.alert_tracking_enabled && (<><span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>|</span><span className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded" style={{ color: LIVE, background: LIVE + '18' }}>{'\u2014\u03c3'}</span></>)}
                  </div>
                )}
              </div>

              {/* Tags */}
              {(strat.tags||[]).length > 0 && (
                <div className="flex flex-wrap gap-1 mb-1">{strat.tags!.map(tag => <span key={tag} className="text-xs px-2 py-0.5 rounded-full" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)', fontSize: '0.72rem' }}>{tag}</span>)}</div>
              )}

              {/* Meta line */}
              <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
                {strat.symbol} {strat.direction} | {strat.timeframe}
                {strat.trading_session && strat.trading_session !== 'RTH' && <span style={{ color: '#9C27B0' }}> | {strat.trading_session}</span>}
                <span style={{ color: BT }}> | BT {btDays}d</span>
                {strat.forward_testing && <span style={{ color: FWD }}> | Fwd {fwdDays}d</span>}
              </p>

              {/* Equity sparkline */}
              {eq?.cumulative_r && eq.cumulative_r.length > 1 ? (
                <div className="rounded-lg mb-1 overflow-hidden" style={{ background: 'var(--bg-input)' }}>
                  <SparkLine data={eq.cumulative_r} height={chartH} boundaryIndex={eq.boundary_index} />
                </div>
              ) : (
                <div className="rounded-lg mb-1 flex items-center justify-center" style={{ background: 'var(--bg-input)', height: chartH }}>
                  <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>No equity data</span>
                </div>
              )}

              {/* Legend */}
              <div className="flex gap-3 mb-3 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                <span><span style={{ color: BT }}>{'\u2014'}</span> Backtest</span>
                <span><span style={{ color: FWD }}>{'\u2014'}</span> Forward</span>
                {strat.alert_tracking_enabled && <span><span style={{ color: LIVE }}>{'\u2014'}</span> Alerts</span>}
              </div>

              {/* KPIs */}
              {kpiMode === 'Overall' ? (
                <div className="grid grid-cols-6 gap-2 mb-3">
                  {[
                    { label: 'WR', value: k.win_rate != null ? `${k.win_rate.toFixed(1)}%` : '--' },
                    { label: 'PF', value: k.profit_factor != null ? k.profit_factor.toFixed(2) : '--' },
                    { label: 'Daily R', value: fmtK(k.daily_r) },
                    { label: 'Daily ROI', value: k.daily_roi != null ? `${fmtK(k.daily_roi)}%` : '--' },
                    { label: 'TPD', value: btTPD != null ? btTPD.toFixed(1) : '--' },
                    { label: 'Max DD', value: k.max_r_drawdown != null ? `${k.max_r_drawdown.toFixed(1)}R` : '--' },
                  ].map(m => (
                    <div key={m.label} className="text-center">
                      <div className="text-[10px] mb-0.5" style={{ color: 'var(--text-muted)' }}>{m.label}</div>
                      <div className="text-sm font-mono font-medium" style={{ color: 'var(--text-primary)' }}>{m.value}</div>
                    </div>
                  ))}
                </div>
              ) : (
                <KpiComparisonTable kpiMode={kpiMode} k={strat.kpis} fk={strat.fwd_kpis} ak={strat.alert_kpis} btTPD={btTPD} fwdDays={fwdDays} />
              )}

              {/* Strategy variables */}
              <div className="space-y-1.5 mb-3">
                {/* Entry */}
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>entry:</span>
                  {strat.entry_trigger_confluence_id ? (() => { const { tag, rest } = parseExec(strat.entry_trigger_confluence_id); return <span className="text-[10px] font-mono" style={{ color: 'var(--text-secondary)' }}><ExecBadge tag={tag} />{rest}</span>; })()
                    : <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>none</span>}
                </div>
                {/* Exit(s) */}
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>exit:</span>
                  {(strat.exit_trigger_confluence_ids||[]).length > 0 ? strat.exit_trigger_confluence_ids!.map((ex, i) => { const { tag, rest } = parseExec(ex); return (
                    <span key={i} className="text-[10px] font-mono" style={{ color: 'var(--text-secondary)' }}><ExecBadge tag={tag} />{rest}{i < strat.exit_trigger_confluence_ids!.length - 1 && <span style={{ color: 'var(--text-muted)' }}>{' , '}</span>}</span>
                  ); }) : <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>none</span>}
                </div>
                {/* Stop + Target */}
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>stop:</span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded" style={{ color: 'var(--red)', background: 'var(--red)' + '18' }}>{fmtStop(strat.stop_config)}</span>
                  <span className="text-[10px] ml-1" style={{ color: 'var(--text-muted)' }}>target:</span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded" style={{ color: 'var(--green)', background: 'var(--green)' + '18' }}>{fmtTarget(strat.target_config)}</span>
                </div>
                {/* Confluence */}
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>confluence:</span>
                  {(strat.confluence||[]).length > 0 ? strat.confluence!.map(c => (
                    <span key={c} className="text-[10px] font-mono flex items-center gap-1">
                      <span className="px-1 py-0.5 rounded-full font-medium" style={{ color: FID, background: FID + '20', fontSize: '9px' }}>[PB]</span>
                      <span className="px-1.5 py-0.5 rounded" style={{ color: 'var(--accent)', background: 'var(--accent)' + '15' }}>{c}</span>
                    </span>
                  )) : <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>none</span>}
                </div>
              </div>

              {/* Action row */}
              <div className="flex items-center gap-2 pt-2" style={{ borderTop: '1px solid var(--border)' }}>
                <Link href={`/strategies/${strat.id}`}><button className="text-xs px-3 py-1.5 rounded" style={{ background: 'var(--bg-input)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}>View</button></Link>
                <button className="text-xs px-3 py-1.5 rounded" style={{ background: 'var(--bg-input)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }} onClick={() => dupMut.mutate(strat.id)}>Clone</button>
                <button className="text-xs px-3 py-1.5 rounded" style={{ background: 'var(--red)' + '15', color: 'var(--red)', border: '1px solid var(--red)' + '30' }} onClick={() => { if (confirm(`Delete "${strat.name}"?`)) deleteMut.mutate(strat.id); }}>Delete</button>
                <span className="flex-1" />
                {/* Alert toggle */}
                <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Alerts</span>
                <div className="relative w-7 h-4 rounded-full cursor-pointer flex-shrink-0" style={{ background: strat.alert_tracking_enabled ? 'var(--accent)' : 'var(--bg-input)', border: strat.alert_tracking_enabled ? 'none' : '1px solid var(--border)' }} title={strat.alert_tracking_enabled ? 'Alerts on' : 'Alerts off'}>
                  <div className="absolute top-0.5 w-3 h-3 rounded-full transition-all" style={{ background: 'white', left: strat.alert_tracking_enabled ? '12px' : '2px' }} />
                </div>
                {/* Bulk checkbox */}
                <input type="checkbox" checked={selIds.has(strat.id)} onChange={() => toggleSel(strat.id)} className="w-4 h-4 rounded cursor-pointer flex-shrink-0" style={{ accentColor: 'var(--accent)' }} title="Select for bulk actions" />
              </div>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
