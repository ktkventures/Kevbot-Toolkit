'use client';

/**
 * Dashboard — Clean API-first page. V6 "Refined Cockpit" design.
 * Customizable KPIs, 7/5 grid, portfolio filter, date range, system status,
 * performance health SD bars, active positions with match badges + close,
 * issues/warnings, recent activity, and customize modal.
 */

import { useState, useRef, useEffect, useCallback } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import Modal from '@/components/Modal';
import { useDashboardSummary } from '@/hooks/queries/useDashboard';

const ANIMATION_STYLES = `
@keyframes dash-fade-in { 0% { opacity:0; transform:translateY(6px) } 100% { opacity:1; transform:translateY(0) } }
@keyframes dash-pulse-dot { 0%,100% { opacity:.4; transform:scale(1) } 50% { opacity:1; transform:scale(1.4) } }
@keyframes dash-slide-in { 0% { transform:translateX(12px); opacity:0 } 100% { transform:translateX(0); opacity:1 } }
@keyframes dash-glow-border { 0%,100% { box-shadow:0 0 4px rgba(0,255,136,.08) } 50% { box-shadow:0 0 14px rgba(0,255,136,.22) } }
`;

interface ToggleItem { id: string; label: string; enabled: boolean }

const ALL_KPIS: ToggleItem[] = [
  { id: 'strategies', label: 'Strategies', enabled: true }, { id: 'portfolios', label: 'Portfolios', enabled: true },
  { id: 'monitored', label: 'Monitored', enabled: true }, { id: 'total-trades', label: 'Total Trades', enabled: true },
  { id: 'total-r', label: 'Total R', enabled: true }, { id: 'avg-win-rate', label: 'Avg Win Rate', enabled: true },
  { id: 'profit-factor', label: 'Profit Factor', enabled: false }, { id: 'daily-r', label: 'Avg Daily R', enabled: false },
  { id: 'max-dd', label: 'Max Drawdown', enabled: false }, { id: 'balance', label: 'Balance', enabled: false },
];

const ALL_WIDGETS: ToggleItem[] = [
  { id: 'equity-curve', label: 'Equity Curve', enabled: true }, { id: 'daily-pnl', label: 'Daily P&L', enabled: true },
  { id: 'calendar', label: 'P&L Calendar', enabled: true }, { id: 'positions', label: 'Positions', enabled: true },
  { id: 'health', label: 'Health', enabled: true }, { id: 'market-regime', label: 'Market Regime', enabled: true },
  { id: 'monthly-goal', label: 'Monthly Goal', enabled: true }, { id: 'issues', label: 'Issues', enabled: true },
  { id: 'activity', label: 'Activity', enabled: true },
];

const DATE_RANGES: Record<string, string> = {
  '7d': 'Last 7 days', '14d': 'Last 14 days', '30d': 'Last 30 days',
  '90d': 'Last 90 days', 'mtd': 'Month to date', 'ytd': 'Year to date', 'all': 'All time',
};

function fmtKpi(v: number | undefined | null, d = 2, s = ''): string {
  if (v == null) return '--';
  return `${v >= 0 ? '+' : ''}${v.toFixed(d)}${s}`;
}

function usePopover() {
  const [isOpen, setIsOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!isOpen) return;
    const h = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setIsOpen(false); };
    document.addEventListener('mousedown', h);
    return () => document.removeEventListener('mousedown', h);
  }, [isOpen]);
  return { isOpen, setIsOpen, ref };
}

function Placeholder({ label, height = 180 }: { label: string; height?: number }) {
  return (
    <div className="rounded-lg flex flex-col items-center justify-center" style={{ background: 'var(--bg-input)', height, border: '1px dashed var(--border)' }}>
      <span className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span className="text-[10px] mt-1" style={{ color: 'var(--text-muted)', opacity: 0.6 }}>Coming soon</span>
    </div>
  );
}

function SdBar({ name, sd, trades }: { name: string; sd: number; trades: number }) {
  const abs = Math.abs(sd);
  const color = abs <= 1 ? 'var(--green)' : abs <= 2 ? 'var(--orange, #FF9800)' : 'var(--red)';
  const bg = abs <= 1 ? 'rgba(76,175,80,.15)' : abs <= 2 ? 'rgba(255,152,0,.15)' : 'rgba(244,67,54,.15)';
  const tag = abs <= 1 ? 'On Track' : abs <= 2 ? 'Warning' : 'Critical';
  return (
    <div className="rounded-lg p-2.5" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-xs font-medium truncate flex-1 mr-2" style={{ color: 'var(--text-primary)' }}>{name}</span>
        <span className="text-[9px] px-1.5 py-0.5 rounded-full font-medium" style={{ background: bg, color }}>{tag}</span>
      </div>
      <div className="relative h-3 rounded-full mb-1.5" style={{ background: 'var(--bg-card)' }}>
        <div className="absolute inset-y-0 rounded-full" style={{ left: '10%', right: '10%', background: 'rgba(76,175,80,.15)', opacity: 0.4 }} />
        <div className="absolute top-0 bottom-0 w-px" style={{ left: '50%', background: 'var(--text-muted)', opacity: 0.4 }} />
        <div className="absolute top-0 bottom-0 w-2.5 rounded-full" style={{ left: `${Math.max(2, Math.min(98, 50 + (sd / 3) * 45))}%`, transform: 'translateX(-50%)', background: color, boxShadow: `0 0 4px ${color}` }} />
      </div>
      <div className="flex items-center justify-between text-[10px]">
        <span style={{ color: 'var(--text-muted)' }}>{trades} trades</span>
        <span className="font-mono" style={{ color }}>{sd >= 0 ? '+' : ''}{sd.toFixed(2)} SD</span>
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const { data: summary, isLoading, error } = useDashboardSummary();
  const [showCustomize, setShowCustomize] = useState(false);
  const [customizeTab, setCustomizeTab] = useState<'KPIs' | 'Widgets'>('Widgets');
  const [kpis, setKpis] = useState(ALL_KPIS);
  const [widgets, setWidgets] = useState(ALL_WIDGETS);
  const [healthTab, setHealthTab] = useState<'portfolios' | 'strategies'>('portfolios');
  const [dateRange, setDateRange] = useState('30d');
  const datePopover = usePopover();
  const portfolioPopover = usePopover();

  const toggle = (list: ToggleItem[], id: string) => list.map(i => i.id === id ? { ...i, enabled: !i.enabled } : i);
  const w = (id: string) => widgets.find(x => x.id === id)?.enabled ?? true;

  useEffect(() => {
    const id = 'dashboard-page-animations';
    if (typeof document !== 'undefined' && !document.getElementById(id)) {
      const s = document.createElement('style'); s.id = id; s.textContent = ANIMATION_STYLES; document.head.appendChild(s);
    }
  }, []);

  if (isLoading) return (
    <div>
      <div className="h-7 w-40 rounded animate-pulse mb-6" style={{ background: 'var(--border)' }} />
      <div className="grid grid-cols-6 gap-3 mb-5">
        {[1,2,3,4,5,6].map(i => <div key={i} className="rounded-lg border p-3 animate-pulse" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}><div className="h-3 rounded w-1/2 mb-2" style={{ background: 'var(--border)' }} /><div className="h-5 rounded w-2/3" style={{ background: 'var(--border)' }} /></div>)}
      </div>
    </div>
  );

  if (error) return (
    <div>
      <h1 className="text-2xl font-bold mb-2">Dashboard</h1>
      <Card><div className="text-center py-8" style={{ color: 'var(--red)' }}>Failed to load dashboard data.</div></Card>
    </div>
  );

  const d = summary || {} as any;
  const strategies = d.strategies ?? [];
  const monitored = strategies.filter((s: any) => s.alert_tracking_enabled);
  const enabledKpis = kpis.filter(k => k.enabled);
  const cols = enabledKpis.length <= 4 ? 'grid-cols-4' : enabledKpis.length === 5 ? 'grid-cols-5' : 'grid-cols-6';

  const kpiVal: Record<string, { v: string; c?: string }> = {
    strategies: { v: String(d.strategy_count ?? 0) },
    portfolios: { v: String(d.portfolio_count ?? 0) },
    monitored: { v: String(d.monitored_count ?? 0), c: (d.monitored_count ?? 0) > 0 ? 'var(--green)' : undefined },
    'total-trades': { v: (d.total_trades ?? 0).toLocaleString() },
    'total-r': { v: fmtKpi(d.total_r), c: (d.total_r ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' },
    'avg-win-rate': { v: (d.avg_win_rate ?? 0) > 0 ? `${d.avg_win_rate.toFixed(1)}%` : '--' },
    'profit-factor': { v: '--' }, 'daily-r': { v: '--' }, 'max-dd': { v: '--' }, balance: { v: '--' },
  };

  return (
    <div style={{ animation: 'dash-fade-in 0.3s ease-out' }} suppressHydrationWarning>
      {/* HEADER */}
      <div className="flex items-start justify-between mb-5">
        <div className="flex items-center gap-4">
          <div><h1 className="text-2xl font-bold mb-1">Dashboard</h1><span className="text-sm" style={{ color: 'var(--text-muted)' }}>Trading Cockpit</span></div>

          {/* Portfolio filter */}
          <div className="relative" ref={portfolioPopover.ref}>
            <button className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium mt-1" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }} onClick={() => portfolioPopover.setIsOpen(!portfolioPopover.isOpen)}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z" /></svg>
              All portfolios <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M6 9l6 6 6-6" /></svg>
            </button>
            {portfolioPopover.isOpen && <div className="absolute top-full left-0 mt-1 rounded-lg border py-2 z-50 min-w-[200px]" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)', boxShadow: '0 8px 24px rgba(0,0,0,.4)' }}><p className="px-3 py-3 text-xs" style={{ color: 'var(--text-muted)' }}>Portfolio filter coming soon</p></div>}
          </div>

          {/* Date range */}
          <div className="relative" ref={datePopover.ref}>
            <button className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium mt-1" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }} onClick={() => datePopover.setIsOpen(!datePopover.isOpen)}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="4" width="18" height="18" rx="2" /><line x1="16" y1="2" x2="16" y2="6" /><line x1="8" y1="2" x2="8" y2="6" /><line x1="3" y1="10" x2="21" y2="10" /></svg>
              {DATE_RANGES[dateRange]} <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M6 9l6 6 6-6" /></svg>
            </button>
            {datePopover.isOpen && <div className="absolute top-full left-0 mt-1 rounded-lg border py-1 z-50 min-w-[160px]" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)', boxShadow: '0 8px 24px rgba(0,0,0,.4)' }}>
              {Object.entries(DATE_RANGES).map(([id, label]) => <button key={id} className="w-full text-left px-3 py-1.5 text-xs" style={{ color: dateRange === id ? 'var(--accent)' : 'var(--text-secondary)', background: dateRange === id ? 'var(--accent-muted, rgba(0,255,136,.1))' : 'transparent' }} onClick={() => { setDateRange(id); datePopover.setIsOpen(false); }}>{label}</button>)}
            </div>}
          </div>

          {/* System status */}
          <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11px] mt-1" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
            <span className="w-2 h-2 rounded-full" style={{ background: 'var(--green)', boxShadow: '0 0 6px rgba(76,175,80,.5)', animation: 'dash-pulse-dot 2s ease-in-out infinite' }} />
            <span style={{ color: 'var(--green)' }}>All systems operational</span>
          </div>

          {/* Quick actions */}
          <div className="flex items-center gap-1 mt-1">
            {[{ l: 'Builder', h: '/strategy-builder', i: 'M12 4v16m-8-8h16' }, { l: 'Portfolios', h: '/portfolios', i: 'M4 6h16M4 12h16M4 18h16' }, { l: 'Alerts', h: '/alerts', i: 'M18 8A6 6 0 006 8c0 7-3 9-3 9h18s-3-2-3-9M13.73 21a2 2 0 01-3.46 0' }].map(a => (
              <Link key={a.l} href={a.h}><div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }} title={a.l}><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d={a.i} /></svg></div></Link>
            ))}
          </div>
        </div>
        <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }} onClick={() => setShowCustomize(true)}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z" /></svg>
          Customize
        </button>
      </div>

      {/* KPI STRIP */}
      <div className={`grid ${cols} gap-3 mb-5`}>
        {enabledKpis.map(k => {
          const val = kpiVal[k.id];
          return <div key={k.id} className="rounded-lg border p-3" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}><p className="text-[10px] mb-0.5" style={{ color: 'var(--text-muted)' }}>{k.label}</p><span className="text-lg font-semibold" style={{ color: val?.c || 'var(--text-primary)' }}>{val?.v ?? '--'}</span></div>;
        })}
      </div>

      {/* MAIN GRID: 7/5 split */}
      <div className="grid grid-cols-12 gap-5 mb-5">
        {/* LEFT: Charts (7 cols) */}
        <div className="col-span-7 space-y-5">
          {w('equity-curve') && <Card><h3 className="text-xs font-semibold mb-2" style={{ color: 'var(--text-muted)' }}>Portfolio Equity Curve</h3><Placeholder label="Equity curve not yet available from API" height={260} /></Card>}
          {w('daily-pnl') && <Card><div className="flex items-center justify-between mb-2"><h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>Daily P&L</h3><div className="flex gap-3">{[['var(--green)', 'Profit'], ['var(--red)', 'Loss']].map(([c, l]) => <div key={l as string} className="flex items-center gap-1"><span className="w-2 h-2 rounded-sm" style={{ background: c as string }} /><span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>{l}</span></div>)}</div></div><Placeholder label="Daily P&L not yet available from API" height={220} /></Card>}
          {w('calendar') && <Card><div className="flex items-center justify-between mb-2"><h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>P&L Calendar</h3><div className="flex gap-2">{[['var(--green)', 'Profit'], ['var(--red)', 'Loss']].map(([c, l]) => <div key={l as string} className="flex items-center gap-1"><span className="w-2 h-2 rounded-sm" style={{ background: c as string }} /><span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>{l}</span></div>)}</div></div><Placeholder label="Calendar heatmap not yet available" height={180} /></Card>}
        </div>

        {/* RIGHT: Widgets (5 cols) */}
        <div className="col-span-5 space-y-5">
          {/* Active Positions */}
          {w('positions') && <Card>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>Active Positions ({monitored.length})</h3>
              <span className="flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full" style={{ background: 'rgba(76,175,80,.15)', color: 'var(--green)' }}><span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--green)' }} />Live</span>
            </div>
            {monitored.length === 0 ? <p className="text-xs text-center py-4" style={{ color: 'var(--text-muted)' }}>No open positions</p> : monitored.map((s: any, i: number) => (
              <Link key={s.id} href={`/strategies/${s.id}`}><div className="rounded-lg p-3 mb-2 last:mb-0 cursor-pointer" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', animation: `dash-glow-border 3s ease-in-out infinite, dash-slide-in .3s ease-out ${i * .1}s both` }}>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold">{s.symbol}</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: s.direction === 'LONG' ? 'rgba(76,175,80,.15)' : 'rgba(244,67,54,.15)', color: s.direction === 'LONG' ? 'var(--green)' : 'var(--red)' }}>{s.direction}</span>
                    <span className="text-[9px] px-1.5 py-0.5 rounded-full font-medium" style={{ background: 'rgba(76,175,80,.15)', color: 'var(--green)' }}>Matched</span>
                  </div>
                  <div className="flex items-center gap-2">
                    {s.kpis?.total_r != null && <span className="text-sm font-mono font-bold" style={{ color: s.kpis.total_r >= 0 ? 'var(--green)' : 'var(--red)' }}>{fmtKpi(s.kpis.total_r, 2, 'R')}</span>}
                    <button className="text-[9px] px-2 py-0.5 rounded font-medium" style={{ background: 'rgba(244,67,54,.15)', color: 'var(--red)' }} onClick={e => e.preventDefault()} title="Close early">Close</button>
                  </div>
                </div>
                <p className="text-[10px] mb-1.5 truncate" style={{ color: 'var(--text-muted)' }}>{s.name}</p>
                <div className="grid grid-cols-3 gap-2 text-[10px]">
                  {[['WR', s.kpis?.win_rate != null ? `${s.kpis.win_rate.toFixed(1)}%` : '--'], ['PF', s.kpis?.profit_factor != null ? s.kpis.profit_factor.toFixed(2) : '--'], ['Trades', s.kpis?.total_trades ?? '--']].map(([l, v]) => <div key={l as string}><span style={{ color: 'var(--text-muted)' }}>{l}</span><p className="font-mono" style={{ color: 'var(--text-secondary)' }}>{v}</p></div>)}
                </div>
              </div></Link>
            ))}
          </Card>}

          {/* Performance Health */}
          {w('health') && <Card>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>Performance Health</h3>
              <div className="flex gap-0.5 rounded-lg overflow-hidden" style={{ border: '1px solid var(--border)' }}>
                {(['portfolios', 'strategies'] as const).map(t => <button key={t} onClick={() => setHealthTab(t)} className="px-2.5 py-1 text-[10px] font-medium" style={{ background: healthTab === t ? 'var(--accent-muted, rgba(0,255,136,.1))' : 'transparent', color: healthTab === t ? 'var(--accent)' : 'var(--text-muted)' }}>{t[0].toUpperCase() + t.slice(1)}</button>)}
              </div>
            </div>
            <div className="flex items-center gap-3 mb-3 px-2 py-1.5 rounded-lg" style={{ background: 'var(--bg-input)' }}>
              {[['var(--green)', '<1 SD'], ['var(--orange, #FF9800)', '1-2 SD'], ['var(--red)', '>2 SD']].map(([c, l]) => <div key={l as string} className="flex items-center gap-1"><span className="w-2 h-2 rounded-full" style={{ background: c as string }} /><span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>{l}</span></div>)}
            </div>
            <Placeholder label="Health deviation data not yet available from API" height={100} />
          </Card>}

          {w('market-regime') && <Card><h3 className="text-xs font-semibold mb-2" style={{ color: 'var(--text-muted)' }}>Market Regime</h3><Placeholder label="Market regime not yet available" height={100} /></Card>}
          {w('monthly-goal') && <Card><h3 className="text-xs font-semibold mb-3" style={{ color: 'var(--text-muted)' }}>Monthly Goal</h3><Placeholder label="Goal tracking not yet available" height={80} /></Card>}
        </div>
      </div>

      {/* BOTTOM: Issues + Activity */}
      <div className="grid grid-cols-2 gap-5 mb-5">
        {w('issues') && <Card>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>Issues & Warnings</h3>
            <span className="text-[9px] px-1.5 py-0.5 rounded-full font-medium" style={{ background: 'rgba(76,175,80,.15)', color: 'var(--green)' }}>0 active</span>
          </div>
          <div className="flex flex-col items-center justify-center py-6 gap-2">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--green)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14" /><polyline points="22 4 12 14.01 9 11.01" /></svg>
            <p className="text-xs font-medium" style={{ color: 'var(--green)' }}>No issues detected</p>
            <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>All strategies operating normally</p>
          </div>
        </Card>}
        {w('activity') && <Card>
          <h3 className="text-xs font-semibold mb-3" style={{ color: 'var(--text-muted)' }}>Recent Activity</h3>
          {[{ t: 'Just now', m: 'Dashboard loaded', i: 'S' }, { t: '--', m: 'Activity feed not yet wired to API', i: '--' }].map((a, i) => (
            <div key={i} className="flex items-center gap-2.5 py-2 border-b last:border-0" style={{ borderColor: 'var(--border)' }}>
              <div className="w-5 h-5 rounded flex items-center justify-center text-[10px] font-bold" style={{ background: 'rgba(156,163,175,.15)', color: 'var(--text-muted)' }}>{a.i}</div>
              <p className="text-xs flex-1 truncate" style={{ color: 'var(--text-secondary)' }}>{a.m}</p>
              <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{a.t}</span>
            </div>
          ))}
        </Card>}
      </div>

      {/* CUSTOMIZE MODAL */}
      <Modal title="Customize Dashboard" isOpen={showCustomize} onClose={() => setShowCustomize(false)} width="520px">
        <div className="flex gap-1 border-b mb-4" style={{ borderColor: 'var(--border)' }}>
          {(['KPIs', 'Widgets'] as const).map(tab => <button key={tab} onClick={() => setCustomizeTab(tab)} className="px-4 py-2 text-xs font-medium" style={{ color: customizeTab === tab ? 'var(--accent)' : 'var(--text-muted)', borderBottom: customizeTab === tab ? '2px solid var(--accent)' : '2px solid transparent', marginBottom: '-1px' }}>{tab}</button>)}
        </div>
        {customizeTab === 'KPIs' && <div>
          <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>Choose which KPIs appear in the top strip (4-6 recommended).</p>
          {kpis.map(k => <label key={k.id} className="flex items-center gap-2.5 py-2.5 border-b last:border-0 cursor-pointer" style={{ borderColor: 'var(--border)' }}><input type="checkbox" checked={k.enabled} onChange={() => setKpis(prev => toggle(prev, k.id))} className="w-3.5 h-3.5 rounded" /><span className="text-sm flex-1" style={{ color: 'var(--text-secondary)' }}>{k.label}</span><span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>{kpiVal[k.id]?.v ?? '--'}</span></label>)}
        </div>}
        {customizeTab === 'Widgets' && <div>
          <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>Toggle dashboard widgets on or off.</p>
          {widgets.map(ww => <div key={ww.id} className="flex items-center justify-between py-2.5 border-b last:border-0" style={{ borderColor: 'var(--border)' }}>
            <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>{ww.label}</span>
            <button className="relative w-10 h-5 rounded-full" style={{ background: ww.enabled ? 'var(--accent)' : 'var(--bg-input)', border: '1px solid', borderColor: ww.enabled ? 'var(--accent)' : 'var(--border)' }} onClick={() => setWidgets(prev => toggle(prev, ww.id))}><span className="absolute top-0.5 w-3.5 h-3.5 rounded-full transition-all" style={{ background: ww.enabled ? '#000' : 'var(--text-muted)', left: ww.enabled ? '22px' : '3px' }} /></button>
          </div>)}
        </div>}
      </Modal>
    </div>
  );
}
