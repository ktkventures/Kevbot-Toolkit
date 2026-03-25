'use client';

/**
 * My Portfolios — Clean API-first page.
 *
 * Visual design derived from V5 (versions/V5.tsx), data layer built
 * around actual Supabase API response shapes. No mock data.
 */

import { useState, useMemo } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import { usePortfolios, PortfolioDTO } from '@/hooks/queries/usePortfolios';
import { useDeletePortfolio, useDuplicatePortfolio } from '@/hooks/mutations/usePortfolioMutations';

// ---------------------------------------------------------------------------
// Style constants (from V5)
// ---------------------------------------------------------------------------

const selectStyle: React.CSSProperties = {
  padding: '6px 10px',
  borderRadius: '6px',
  border: '1px solid var(--border)',
  background: 'var(--bg-input)',
  color: 'var(--text-primary)',
  fontSize: '12px',
};

const statusColors: Record<string, string> = {
  active: 'var(--green)',
  paused: 'var(--text-muted)',
  error: 'var(--red)',
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmtDollar(v: number | undefined): string {
  if (v === undefined || v === null) return '--';
  return `${v >= 0 ? '+' : ''}$${Math.abs(v).toLocaleString()}`;
}

function fmtPct(v: number | undefined): string {
  if (v === undefined || v === null) return '--';
  return `${v.toFixed(1)}%`;
}

function getPortfolioStatus(p: PortfolioDTO): { label: string; color: string } {
  if (!p.enabled) return { label: 'Disabled', color: 'var(--text-muted)' };
  return { label: 'Active', color: 'var(--green)' };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function PortfoliosPage() {
  const { data: portfolios, isLoading, error } = usePortfolios();
  const deleteMutation = useDeletePortfolio();
  const duplicateMutation = useDuplicatePortfolio();

  // Filters
  const [sortBy, setSortBy] = useState('Newest First');
  const [tagFilter, setTagFilter] = useState('All');
  const [kpiMode, setKpiMode] = useState('Overall');

  const allTags = useMemo(() => {
    if (!portfolios) return [];
    return Array.from(new Set(portfolios.flatMap((p: PortfolioDTO) => p.tags || []))).sort();
  }, [portfolios]);

  const filtered = useMemo(() => {
    if (!portfolios) return [];
    let result = [...portfolios] as PortfolioDTO[];

    if (tagFilter !== 'All') {
      result = result.filter((p) => (p.tags || []).includes(tagFilter));
    }

    if (sortBy === 'Newest First') result.sort((a, b) => b.id - a.id);
    else if (sortBy === 'Oldest First') result.sort((a, b) => a.id - b.id);
    else if (sortBy === 'Name A-Z') result.sort((a, b) => (a.name || '').localeCompare(b.name || ''));
    else if (sortBy === 'P&L (High)') result.sort((a, b) => ((b.kpis?.total_pnl || 0) - (a.kpis?.total_pnl || 0)));
    else if (sortBy === 'Win Rate (High)') result.sort((a, b) => ((b.kpis?.win_rate || 0) - (a.kpis?.win_rate || 0)));

    return result;
  }, [portfolios, sortBy, tagFilter]);

  // Combined metrics
  const combined = useMemo(() => {
    if (!filtered.length) return null;
    const totalPnl = filtered.reduce((s, p) => s + (p.kpis?.total_pnl || 0), 0);
    const avgWR = filtered.reduce((s, p) => s + (p.kpis?.win_rate || 0), 0) / filtered.length;
    const avgPF = filtered.reduce((s, p) => s + (p.kpis?.profit_factor || 0), 0) / filtered.length;
    const totalTrades = filtered.reduce((s, p) => s + (p.kpis?.total_trades || 0), 0);
    const totalStrategies = filtered.reduce((s, p) => s + (p.strategies || []).length, 0);
    return { totalPnl, avgWR, avgPF, totalTrades, totalStrategies };
  }, [filtered]);

  // ---------------------------------------------------------------------------
  // Loading / Error / Empty states
  // ---------------------------------------------------------------------------

  if (isLoading) {
    return (
      <div>
        <PageHeader title="My Portfolios" subtitle="Loading..." />
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i}>
              <div className="animate-pulse space-y-3">
                <div className="h-4 rounded w-1/3" style={{ background: 'var(--border)' }} />
                <div className="h-3 rounded w-2/3" style={{ background: 'var(--border)' }} />
                <div className="h-16 rounded" style={{ background: 'var(--bg-input)' }} />
                <div className="grid grid-cols-4 gap-2">
                  {[1, 2, 3, 4].map((j) => (
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
        <PageHeader title="My Portfolios" subtitle="Error loading portfolios" />
        <Card>
          <div className="text-center py-8" style={{ color: 'var(--red)' }}>
            Failed to load portfolios. Check your connection and try again.
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
        title="My Portfolios"
        subtitle={`${filtered.length} portfolio${filtered.length === 1 ? '' : 's'}`}
        actions={
          <div className="flex items-center gap-2">
            <span
              className="text-xs px-2 py-1 rounded-full flex items-center gap-1.5"
              style={{ background: 'var(--green)' + '15', color: 'var(--green)' }}
            >
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--green)' }} />
              Live
            </span>
            <Link href="/portfolios/new">
              <button
                className="px-4 py-2 rounded-lg text-sm font-medium"
                style={{ background: 'var(--accent)', color: '#fff' }}
              >
                + New Portfolio
              </button>
            </Link>
          </div>
        }
      />

      {/* Filter row */}
      <div className="flex flex-wrap gap-2 mb-2 mt-4">
        <select style={selectStyle} value={tagFilter} onChange={(e) => setTagFilter(e.target.value)}>
          <option value="All">All Tags</option>
          {allTags.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>
        <select style={selectStyle} value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
          {['Newest First', 'Oldest First', 'Name A-Z', 'P&L (High)', 'Win Rate (High)'].map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </div>

      {/* KPI comparison mode selector (from V5) */}
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
            <option value="BT vs FWD">Backtest vs Forward</option>
            <option value="FWD vs Alerts">Forward vs Alerts</option>
            <option value="BT vs Alerts">Backtest vs Alerts</option>
          </select>
        </div>
      </div>

      {/* Combined Metrics card (from V5) */}
      {combined && filtered.length > 0 && (
        <Card className="mb-4">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-medium">
              Combined Metrics
              <span className="text-xs font-normal ml-2" style={{ color: 'var(--text-muted)' }}>
                {filtered.length} portfolio{filtered.length !== 1 ? 's' : ''} &middot; {combined.totalStrategies} strategies
              </span>
            </h4>
          </div>
          <div className="grid grid-cols-3 sm:grid-cols-5 gap-3">
            {[
              { label: 'Total P&L', value: fmtDollar(combined.totalPnl), color: combined.totalPnl >= 0 ? 'var(--green)' : 'var(--red)' },
              { label: 'Avg Win Rate', value: fmtPct(combined.avgWR) },
              { label: 'Avg PF', value: combined.avgPF.toFixed(2) },
              { label: 'Total Trades', value: String(combined.totalTrades) },
              { label: 'Strategies', value: String(combined.totalStrategies) },
            ].map((kpi) => (
              <div key={kpi.label}>
                <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
                <p className="text-sm font-bold" style={(kpi as { color?: string }).color ? { color: (kpi as { color: string }).color } : undefined}>{kpi.value}</p>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Empty state */}
      {filtered.length === 0 && (
        <Card>
          <div className="text-center py-12">
            <p className="text-lg mb-2" style={{ color: 'var(--text-secondary)' }}>
              {(portfolios?.length || 0) === 0
                ? 'No portfolios yet. Create your first portfolio!'
                : 'No portfolios match the current filters.'}
            </p>
            {(portfolios?.length || 0) === 0 && (
              <Link href="/portfolios/new">
                <button className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: '#fff' }}>
                  Create Portfolio
                </button>
              </Link>
            )}
          </div>
        </Card>
      )}

      {/* Portfolio cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {filtered.map((port) => {
          const kpis = port.kpis || {};
          const stratCount = (port.strategies || []).length;
          const status = getPortfolioStatus(port);
          const reqSet = port.requirement_set_id;
          const reqPassing = port.req_passing;
          const reqTotal = port.req_total;

          return (
            <Card key={port.id}>
              {/* Header: name + status */}
              <div className="flex items-center gap-2 mb-1">
                {port.enabled && (
                  <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: 'var(--green)' }} />
                )}
                <Link
                  href={`/portfolios/${port.id}`}
                  className="font-semibold text-base hover:underline"
                  style={{ color: 'var(--text-primary)' }}
                >
                  {port.name || 'Untitled Portfolio'}
                </Link>
                <span
                  className="text-xs px-2 py-0.5 rounded-full font-medium"
                  style={{ color: status.color, background: status.color + '20' }}
                >
                  {status.label}
                </span>
              </div>

              {/* Tags */}
              {(port.tags || []).length > 0 && (
                <div className="flex flex-wrap gap-1 mb-1">
                  {port.tags!.map((tag) => (
                    <span key={tag} className="text-xs px-2 py-0.5 rounded-full" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)', fontSize: '0.72rem' }}>
                      {tag}
                    </span>
                  ))}
                </div>
              )}

              {/* Meta line with requirement set badge */}
              <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
                {stratCount} strateg{stratCount === 1 ? 'y' : 'ies'}
                {port.account?.starting_balance != null && (
                  <span> | ${port.account.starting_balance.toLocaleString()} balance</span>
                )}
                {reqSet && (
                  <span style={{ color: 'var(--accent)' }}>
                    {' '}| Req Set #{reqSet}
                    {reqPassing != null && reqTotal != null && (
                      <span
                        className="ml-1 text-[10px] px-1.5 py-0.5 rounded-full font-medium"
                        style={{
                          color: reqPassing === reqTotal ? 'var(--green)' : 'var(--orange)',
                          background: reqPassing === reqTotal ? 'var(--green)' + '20' : 'var(--orange)' + '20',
                        }}
                      >
                        {reqPassing}/{reqTotal} pass
                      </span>
                    )}
                  </span>
                )}
              </p>

              {/* Strategy pills — symbol + direction */}
              {stratCount > 0 && (
                <div className="flex flex-wrap gap-1.5 mb-3">
                  {(port.strategies || []).slice(0, 5).map((s: any, i: number) => (
                    <span key={i} className="text-[10px] px-2 py-0.5 rounded-full inline-flex items-center gap-1" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
                      <span className="font-mono font-medium" style={{ color: 'var(--text-secondary)' }}>{s.symbol || '???'}</span>
                      {s.direction && (
                        <span style={{ color: s.direction === 'LONG' ? 'var(--green)' : 'var(--red)', fontSize: '0.6rem' }}>
                          {s.direction === 'LONG' ? '\u2191' : '\u2193'}
                        </span>
                      )}
                    </span>
                  ))}
                  {stratCount > 5 && (
                    <span className="text-[10px] px-2 py-0.5 rounded-full" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
                      +{stratCount - 5} more
                    </span>
                  )}
                </div>
              )}

              {/* Equity curve placeholder */}
              <div
                className="rounded-lg mb-3 flex items-center justify-center"
                style={{ background: 'var(--bg-input)', height: 64 }}
              >
                <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>Equity curve</span>
              </div>

              {/* KPIs — V5-style with comparison mode support */}
              <div className="grid grid-cols-4 gap-2 mb-3">
                {kpiMode === 'Overall' ? (
                  <>
                    {[
                      { label: 'WR', value: kpis.win_rate != null ? fmtPct(kpis.win_rate) : '--' },
                      { label: 'PF', value: kpis.profit_factor != null ? kpis.profit_factor.toFixed(2) : '--' },
                      { label: 'Total P&L', value: kpis.total_pnl != null ? fmtDollar(kpis.total_pnl) : '--' },
                      { label: 'Trades', value: kpis.total_trades != null ? String(kpis.total_trades) : '--' },
                    ].map((m) => (
                      <div key={m.label} className="text-center">
                        <div className="text-[10px] mb-0.5" style={{ color: 'var(--text-muted)' }}>{m.label}</div>
                        <div className="text-sm font-mono font-medium" style={{ color: 'var(--text-primary)' }}>{m.value}</div>
                      </div>
                    ))}
                  </>
                ) : (
                  <>
                    {[
                      { label: 'WR', value: kpis.win_rate != null ? fmtPct(kpis.win_rate) : '--' },
                      { label: 'PF', value: kpis.profit_factor != null ? kpis.profit_factor.toFixed(2) : '--' },
                      { label: 'Max DD', value: kpis.max_dd_pct != null ? `${kpis.max_dd_pct.toFixed(1)}%` : '--', color: 'var(--red)' },
                      { label: 'Trades', value: kpis.total_trades != null ? String(kpis.total_trades) : '--' },
                    ].map((m) => (
                      <div key={m.label} className="text-center">
                        <div className="text-[10px] mb-0.5" style={{ color: 'var(--text-muted)' }}>{m.label}</div>
                        <div className="text-sm font-mono font-medium" style={(m as { color?: string }).color ? { color: (m as { color: string }).color } : { color: 'var(--text-primary)' }}>{m.value}</div>
                      </div>
                    ))}
                  </>
                )}
              </div>

              {/* Action row */}
              <div className="flex items-center gap-2 pt-2" style={{ borderTop: '1px solid var(--border)' }}>
                <Link href={`/portfolios/${port.id}`}>
                  <button className="text-xs px-3 py-1.5 rounded" style={{ background: 'var(--bg-input)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}>
                    View
                  </button>
                </Link>
                <button
                  className="text-xs px-3 py-1.5 rounded"
                  style={{ background: 'var(--bg-input)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}
                  onClick={() => duplicateMutation.mutate(port.id)}
                >
                  Clone
                </button>
                <button
                  className="text-xs px-3 py-1.5 rounded"
                  style={{ background: 'var(--red)15', color: 'var(--red)', border: '1px solid var(--red)30' }}
                  onClick={() => {
                    if (confirm(`Delete "${port.name}"?`)) {
                      deleteMutation.mutate(port.id);
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
