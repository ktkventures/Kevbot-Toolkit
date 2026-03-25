'use client';

/**
 * My Strategies — Clean API-first page.
 *
 * Visual design copied from V5 (versions/V5.tsx), data layer built
 * around actual Supabase API response shapes. No mock data, no fallbacks.
 */

import { useState, useMemo, useEffect } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import SparkLine from '@/charts/SparkLine';
import { useStrategies } from '@/hooks/queries/useStrategies';
import { useDeleteStrategy, useDuplicateStrategy } from '@/hooks/mutations/useStrategyMutations';

// ---------------------------------------------------------------------------
// Types — match the actual API response from /api/strategies
// ---------------------------------------------------------------------------

interface Strategy {
  id: number;
  name: string;
  symbol: string;
  direction: 'LONG' | 'SHORT';
  timeframe: string;
  trading_session?: string;
  forward_testing?: boolean;
  forward_test_start?: string;
  alert_tracking_enabled?: boolean;
  tags?: string[];
  data_days?: number;
  kpis?: {
    win_rate?: number;
    profit_factor?: number;
    total_r?: number;
    daily_r?: number;
    avg_r?: number;
    r_squared?: number;
    max_r_drawdown?: number;
    total_trades?: number;
    total_pnl?: number;
    final_balance?: number;
  };
  equity_curve_data?: {
    exit_times?: string[];
    cumulative_r?: number[];
    boundary_index?: number | null;
  };
  entry_trigger_confluence_id?: string;
  exit_trigger_confluence_ids?: string[];
  confluence?: string[];
  stop_config?: { method?: string; [key: string]: any };
  target_config?: { method?: string; [key: string]: any } | null;
  created_at?: string;
}

// ---------------------------------------------------------------------------
// Style constants (from V5)
// ---------------------------------------------------------------------------

const EQ_BT_COLOR = '#2196F3';
const EQ_FWD_COLOR = '#FF9800';
const EQ_LIVE_COLOR = '#4CAF50';

const selectStyle: React.CSSProperties = {
  padding: '6px 10px',
  borderRadius: '6px',
  border: '1px solid var(--border)',
  background: 'var(--bg-input)',
  color: 'var(--text-primary)',
  fontSize: '12px',
};

const PULSE_CSS = `@keyframes pulse{0%{transform:scale(1);opacity:.5}100%{transform:scale(2.5);opacity:0}}`;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function kpi(v: number | undefined, decimals = 2, suffix = ''): string {
  if (v === undefined || v === null) return '--';
  return `${v >= 0 ? '+' : ''}${v.toFixed(decimals)}${suffix}`;
}

function daysSince(dateStr?: string): number {
  if (!dateStr) return 0;
  const d = new Date(dateStr);
  return Math.max(0, Math.round((Date.now() - d.getTime()) / 86400000));
}

function formatStopMethod(config?: { method?: string } | null): string {
  if (!config?.method) return 'None';
  const m = config.method;
  if (m === 'atr') return `ATR ${config.atr_mult || 1.5}x`;
  if (m === 'fixed_dollar') return `$${config.amount || '?'} fixed`;
  if (m === 'percentage') return `${config.percentage || '?'}%`;
  if (m === 'swing') return `Swing ${config.lookback || 5}-bar`;
  return m;
}

function formatTargetMethod(config?: { method?: string } | null): string {
  if (!config?.method) return 'Signal exit only';
  const m = config.method;
  if (m === 'risk_reward') return `${config.rr_ratio || 2}:1 R:R`;
  if (m === 'atr') return `ATR ${config.atr_mult || 2}x`;
  if (m === 'fixed_dollar') return `$${config.amount || '?'}`;
  return m;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function StrategiesPage() {
  const { data: strategies, isLoading, error } = useStrategies();
  const deleteMutation = useDeleteStrategy();
  const duplicateMutation = useDuplicateStrategy();

  // Inject pulse CSS
  useEffect(() => {
    if (typeof document !== 'undefined' && !document.getElementById('strat-pulse')) {
      const s = document.createElement('style');
      s.id = 'strat-pulse';
      s.textContent = PULSE_CSS;
      document.head.appendChild(s);
    }
  }, []);

  // Filter state
  const [tickerFilter, setTickerFilter] = useState('All');
  const [directionFilter, setDirectionFilter] = useState('All');
  const [tagFilter, setTagFilter] = useState('All');
  const [sortBy, setSortBy] = useState('Newest First');
  const [kpiMode, setKpiMode] = useState('Overall');
  const [chartHeight, setChartHeight] = useState(64);

  const allTickers = useMemo(() => {
    if (!strategies) return [];
    return Array.from(new Set(strategies.map((s: Strategy) => s.symbol))).sort();
  }, [strategies]);

  const allTags = useMemo(() => {
    if (!strategies) return [];
    return Array.from(new Set(strategies.flatMap((s: Strategy) => s.tags || []))).sort();
  }, [strategies]);

  const filtered = useMemo(() => {
    if (!strategies) return [];
    let result = [...strategies] as Strategy[];
    if (tickerFilter !== 'All') result = result.filter((s) => s.symbol === tickerFilter);
    if (directionFilter !== 'All') result = result.filter((s) => s.direction === directionFilter);
    if (tagFilter !== 'All') result = result.filter((s) => (s.tags || []).includes(tagFilter));

    // Sort
    if (sortBy === 'Newest First') result.sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''));
    else if (sortBy === 'Name A-Z') result.sort((a, b) => a.name.localeCompare(b.name));
    else if (sortBy === 'Win Rate (High)') result.sort((a, b) => (b.kpis?.win_rate || 0) - (a.kpis?.win_rate || 0));
    else if (sortBy === 'Profit Factor (High)') result.sort((a, b) => (b.kpis?.profit_factor || 0) - (a.kpis?.profit_factor || 0));
    else if (sortBy === 'Daily R (High)') result.sort((a, b) => (b.kpis?.daily_r || 0) - (a.kpis?.daily_r || 0));

    return result;
  }, [strategies, tickerFilter, directionFilter, tagFilter, sortBy]);

  // ---------------------------------------------------------------------------
  // Loading / Error / Empty states
  // ---------------------------------------------------------------------------

  if (isLoading) {
    return (
      <div>
        <PageHeader title="My Strategies" subtitle="Loading..." />
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i}>
              <div className="animate-pulse space-y-3">
                <div className="h-4 rounded w-1/3" style={{ background: 'var(--border)' }} />
                <div className="h-3 rounded w-2/3" style={{ background: 'var(--border)' }} />
                <div className="h-16 rounded" style={{ background: 'var(--bg-input)' }} />
                <div className="grid grid-cols-6 gap-2">
                  {[1, 2, 3, 4, 5, 6].map((j) => (
                    <div key={j} className="h-8 rounded" style={{ background: 'var(--border)' }} />
                  ))}
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div>
        <PageHeader title="My Strategies" subtitle="Error loading strategies" />
        <Card>
          <div className="text-center py-8" style={{ color: 'var(--red)' }}>
            Failed to load strategies. Check your connection and try again.
          </div>
        </Card>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div>
      <PageHeader
        title="My Strategies"
        subtitle={`${filtered.length} strateg${filtered.length === 1 ? 'y' : 'ies'}`}
        actions={
          <div className="flex items-center gap-2">
            <span
              className="text-xs px-2 py-1 rounded-full flex items-center gap-1.5"
              style={{ background: 'var(--green)' + '15', color: 'var(--green)' }}
            >
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--green)' }} />
              Live
            </span>
            <Link href="/strategy-builder">
              <button
                className="px-4 py-2 rounded-lg text-sm font-medium"
                style={{ background: 'var(--accent)', color: '#fff' }}
              >
                + New Strategy
              </button>
            </Link>
          </div>
        }
      />

      {/* Filter row */}
      <div className="flex flex-wrap gap-2 mb-3 mt-4">
        <select style={selectStyle} value={tickerFilter} onChange={(e) => setTickerFilter(e.target.value)}>
          <option value="All">All Tickers</option>
          {allTickers.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>
        <select style={selectStyle} value={directionFilter} onChange={(e) => setDirectionFilter(e.target.value)}>
          <option value="All">All Directions</option>
          <option value="LONG">LONG</option>
          <option value="SHORT">SHORT</option>
        </select>
        <select style={selectStyle} value={tagFilter} onChange={(e) => setTagFilter(e.target.value)}>
          <option value="All">All Tags</option>
          {allTags.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>
        <select style={selectStyle} value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
          {['Newest First', 'Name A-Z', 'Win Rate (High)', 'Profit Factor (High)', 'Daily R (High)'].map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </div>

      {/* KPI mode + chart preferences */}
      <div className="flex items-center gap-4 mb-4 text-[10px]" style={{ color: 'var(--text-muted)' }}>
        <div className="flex items-center gap-1.5">
          <span>KPIs:</span>
          <select
            className="px-2 py-1 rounded text-[10px] font-medium"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
            value={kpiMode}
            onChange={(e) => setKpiMode(e.target.value)}
          >
            <option value="Overall">Overall</option>
          </select>
        </div>
        <div className="w-px h-4" style={{ background: 'var(--border)' }} />
        <div className="flex items-center gap-1.5">
          <span>Chart:</span>
          {[{ v: 48, l: 'S' }, { v: 64, l: 'M' }, { v: 96, l: 'L' }, { v: 140, l: 'XL' }].map((o) => (
            <button
              key={o.v}
              onClick={() => setChartHeight(o.v)}
              className="px-2 py-1 rounded font-medium"
              style={{
                background: chartHeight === o.v ? 'var(--accent-muted)' : 'var(--bg-input)',
                color: chartHeight === o.v ? 'var(--accent)' : 'var(--text-muted)',
                border: chartHeight === o.v ? '1px solid var(--accent)' : '1px solid var(--border)',
              }}
            >
              {o.l}
            </button>
          ))}
        </div>
      </div>

      {/* Empty state */}
      {filtered.length === 0 && (
        <Card>
          <div className="text-center py-12">
            <p className="text-lg mb-2" style={{ color: 'var(--text-secondary)' }}>
              {(strategies?.length || 0) === 0
                ? 'No strategies yet. Create your first strategy!'
                : 'No strategies match the current filters.'}
            </p>
            {(strategies?.length || 0) === 0 && (
              <Link href="/strategy-builder">
                <button className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: '#fff' }}>
                  Go to Strategy Builder
                </button>
              </Link>
            )}
          </div>
        </Card>
      )}

      {/* Strategy cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {filtered.map((strat) => {
          const k = strat.kpis || {};
          const eq = strat.equity_curve_data;
          const fwdDays = daysSince(strat.forward_test_start);
          const totalTrades = k.total_trades || 0;

          return (
            <Card key={strat.id}>
              {/* Header: monitored dot + name + status */}
              <div className="flex items-center gap-2 mb-1">
                {strat.alert_tracking_enabled && (
                  <div className="relative flex-shrink-0 w-2.5 h-2.5">
                    <div className="w-2.5 h-2.5 rounded-full" style={{ background: 'var(--green)' }} />
                    <div className="w-2.5 h-2.5 rounded-full absolute top-0 left-0" style={{ background: 'var(--green)', animation: 'pulse 2s ease-out infinite', opacity: 0.5 }} />
                  </div>
                )}
                <Link href={`/strategies/${strat.id}`} className="font-semibold text-base hover:underline" style={{ color: 'var(--text-primary)' }}>
                  {strat.name}
                </Link>
                {strat.forward_testing && (
                  <span className="text-xs px-2 py-0.5 rounded-full font-medium" style={{ color: EQ_FWD_COLOR, background: EQ_FWD_COLOR + '20' }}>
                    Forward Testing
                  </span>
                )}
              </div>

              {/* Tags */}
              {(strat.tags || []).length > 0 && (
                <div className="flex flex-wrap gap-1 mb-1">
                  {strat.tags!.map((tag) => (
                    <span key={tag} className="text-xs px-2 py-0.5 rounded-full" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)', fontSize: '0.72rem' }}>
                      {tag}
                    </span>
                  ))}
                </div>
              )}

              {/* Meta line */}
              <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
                {strat.symbol} {strat.direction} | {strat.timeframe}
                {strat.trading_session && strat.trading_session !== 'RTH' && (
                  <span style={{ color: '#9C27B0' }}> | {strat.trading_session}</span>
                )}
                <span style={{ color: EQ_BT_COLOR }}> | BT {strat.data_days || 30}d</span>
                {strat.forward_testing && <span style={{ color: EQ_FWD_COLOR }}> | Fwd {fwdDays}d</span>}
              </p>

              {/* Equity sparkline */}
              {eq?.cumulative_r && eq.cumulative_r.length > 1 ? (
                <div className="rounded-lg mb-2 overflow-hidden" style={{ background: 'var(--bg-input)' }}>
                  <SparkLine
                    data={eq.cumulative_r}
                    height={chartHeight}
                    boundaryIndex={eq.boundary_index}
                  />
                </div>
              ) : (
                <div className="rounded-lg mb-2 flex items-center justify-center" style={{ background: 'var(--bg-input)', height: chartHeight }}>
                  <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>No equity data</span>
                </div>
              )}

              {/* KPIs */}
              <div className="grid grid-cols-6 gap-2 mb-3">
                {[
                  { label: 'WR', value: k.win_rate != null ? `${k.win_rate.toFixed(1)}%` : '--' },
                  { label: 'PF', value: k.profit_factor != null ? k.profit_factor.toFixed(2) : '--' },
                  { label: 'Daily R', value: kpi(k.daily_r) },
                  { label: 'Total R', value: kpi(k.total_r) },
                  { label: 'Trades', value: totalTrades.toString() },
                  { label: 'Max DD', value: k.max_r_drawdown != null ? `${k.max_r_drawdown.toFixed(1)}R` : '--' },
                ].map((m) => (
                  <div key={m.label} className="text-center">
                    <div className="text-[10px] mb-0.5" style={{ color: 'var(--text-muted)' }}>{m.label}</div>
                    <div className="text-sm font-mono font-medium" style={{ color: 'var(--text-primary)' }}>{m.value}</div>
                  </div>
                ))}
              </div>

              {/* Strategy variables */}
              <div className="text-[10px] font-mono space-y-0.5 mb-3" style={{ color: 'var(--text-muted)' }}>
                <div>
                  <span>entry: </span>
                  <span style={{ color: 'var(--accent)' }}>{strat.entry_trigger_confluence_id || 'none'}</span>
                  <span> | exit: </span>
                  <span style={{ color: 'var(--accent)' }}>{(strat.exit_trigger_confluence_ids || []).join(', ') || 'none'}</span>
                </div>
                <div>
                  <span style={{ color: 'var(--red)' }}>stop: {formatStopMethod(strat.stop_config)}</span>
                  <span> | </span>
                  <span style={{ color: 'var(--green)' }}>target: {formatTargetMethod(strat.target_config)}</span>
                </div>
                {(strat.confluence || []).length > 0 && (
                  <div>
                    <span>confluence: </span>
                    {strat.confluence!.map((c) => (
                      <span key={c} className="px-1 py-0.5 rounded mr-1" style={{ background: 'var(--accent)15', color: 'var(--accent)' }}>{c}</span>
                    ))}
                  </div>
                )}
              </div>

              {/* Action row */}
              <div className="flex items-center gap-2 pt-2" style={{ borderTop: '1px solid var(--border)' }}>
                <Link href={`/strategies/${strat.id}`}>
                  <button className="text-xs px-3 py-1.5 rounded" style={{ background: 'var(--bg-input)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}>
                    View
                  </button>
                </Link>
                <button
                  className="text-xs px-3 py-1.5 rounded"
                  style={{ background: 'var(--bg-input)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}
                  onClick={() => duplicateMutation.mutate(strat.id)}
                >
                  Clone
                </button>
                <button
                  className="text-xs px-3 py-1.5 rounded"
                  style={{ background: 'var(--red)15', color: 'var(--red)', border: '1px solid var(--red)30' }}
                  onClick={() => {
                    if (confirm(`Delete "${strat.name}"?`)) {
                      deleteMutation.mutate(strat.id);
                    }
                  }}
                >
                  Delete
                </button>
              </div>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
