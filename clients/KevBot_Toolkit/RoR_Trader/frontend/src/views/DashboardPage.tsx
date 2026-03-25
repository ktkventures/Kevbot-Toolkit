'use client';

/**
 * Dashboard — Clean API-first page.
 *
 * Visual design based on V6 "Refined Cockpit" (versions/V6.tsx).
 * Data layer built around the /api/dashboard/summary endpoint.
 * No mock data — real API data or clean placeholder states.
 */

import { useEffect } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import EquityCurve from '@/charts/EquityCurve';
import { useDashboardSummary } from '@/hooks/queries/useDashboard';

// ---------------------------------------------------------------------------
// CSS Animations (injected once on mount)
// ---------------------------------------------------------------------------

const ANIMATION_STYLES = `
@keyframes dash-fade-in {
  0% { opacity: 0; transform: translateY(6px); }
  100% { opacity: 1; transform: translateY(0); }
}
@keyframes dash-pulse-dot {
  0%, 100% { opacity: 0.4; transform: scale(1); }
  50% { opacity: 1; transform: scale(1.4); }
}
@keyframes dash-slide-in {
  0% { transform: translateX(12px); opacity: 0; }
  100% { transform: translateX(0); opacity: 1; }
}
`;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmtKpi(v: number | undefined | null, decimals = 2, suffix = ''): string {
  if (v === undefined || v === null) return '--';
  return `${v >= 0 ? '+' : ''}${v.toFixed(decimals)}${suffix}`;
}

// ---------------------------------------------------------------------------
// Skeleton Loading State
// ---------------------------------------------------------------------------

function DashboardSkeleton() {
  return (
    <div>
      <PageHeader title="Dashboard" subtitle="Loading..." />

      {/* KPI strip skeleton */}
      <div className="grid grid-cols-4 gap-3 mb-5">
        {[1, 2, 3, 4].map((i) => (
          <div
            key={i}
            className="rounded-lg border p-3 animate-pulse"
            style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}
          >
            <div className="h-3 rounded w-1/2 mb-2" style={{ background: 'var(--border)' }} />
            <div className="h-5 rounded w-2/3" style={{ background: 'var(--border)' }} />
          </div>
        ))}
      </div>

      {/* Main content skeleton */}
      <div className="grid grid-cols-12 gap-5 mb-5">
        {/* Left column */}
        <div className="col-span-7 space-y-5">
          {[260, 220, 180].map((h, i) => (
            <Card key={i}>
              <div className="animate-pulse">
                <div className="h-3 rounded w-1/3 mb-3" style={{ background: 'var(--border)' }} />
                <div className="rounded" style={{ background: 'var(--bg-input)', height: h }} />
              </div>
            </Card>
          ))}
        </div>
        {/* Right column */}
        <div className="col-span-5 space-y-5">
          {[200, 240, 140, 100].map((h, i) => (
            <Card key={i}>
              <div className="animate-pulse">
                <div className="h-3 rounded w-1/2 mb-3" style={{ background: 'var(--border)' }} />
                <div className="rounded" style={{ background: 'var(--bg-input)', height: h }} />
              </div>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Placeholder Widget — for sections without API data yet
// ---------------------------------------------------------------------------

function PlaceholderWidget({
  label,
  height = 180,
  icon,
}: {
  label: string;
  height?: number;
  icon?: string;
}) {
  return (
    <div
      className="rounded-lg flex flex-col items-center justify-center"
      style={{ background: 'var(--bg-input)', height, border: '1px dashed var(--border)' }}
    >
      {icon && (
        <span className="text-2xl mb-2" style={{ color: 'var(--text-muted)', opacity: 0.5 }}>
          {icon}
        </span>
      )}
      <span className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>
        {label}
      </span>
      <span className="text-[10px] mt-1" style={{ color: 'var(--text-muted)', opacity: 0.6 }}>
        Coming soon
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Monitored Strategies Widget (right column)
// ---------------------------------------------------------------------------

function MonitoredStrategies({
  strategies,
}: {
  strategies: Array<{
    id: number;
    name: string;
    symbol: string;
    direction: string;
    kpis: Record<string, number>;
    alert_tracking_enabled: boolean;
  }>;
}) {
  const monitored = strategies.filter((s) => s.alert_tracking_enabled);

  return (
    <Card>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>
          Active Positions ({monitored.length})
        </h3>
        <span
          className="flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full"
          style={{ background: 'rgba(76,175,80,0.15)', color: 'var(--green)' }}
        >
          <span
            className="w-1.5 h-1.5 rounded-full"
            style={{ background: 'var(--green)' }}
          />
          Live
        </span>
      </div>

      {monitored.length === 0 ? (
        <p className="text-xs text-center py-4" style={{ color: 'var(--text-muted)' }}>
          No monitored strategies
        </p>
      ) : (
        monitored.map((strat, idx) => (
          <Link key={strat.id} href={`/strategies/${strat.id}`}>
            <div
              className="rounded-lg p-3 mb-2 last:mb-0 cursor-pointer transition-colors"
              style={{
                background: 'var(--bg-input)',
                border: '1px solid var(--border)',
                animation: `dash-slide-in 0.3s ease-out ${idx * 0.1}s both`,
              }}
            >
              {/* Top row: symbol + direction */}
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                    {strat.symbol}
                  </span>
                  <span
                    className="text-[10px] px-1.5 py-0.5 rounded"
                    style={{
                      background: strat.direction === 'LONG' ? 'rgba(76,175,80,0.15)' : 'rgba(244,67,54,0.15)',
                      color: strat.direction === 'LONG' ? 'var(--green)' : 'var(--red)',
                    }}
                  >
                    {strat.direction}
                  </span>
                  <span
                    className="w-2 h-2 rounded-full flex-shrink-0"
                    style={{ background: 'var(--green)', animation: 'dash-pulse-dot 2s ease-in-out infinite' }}
                  />
                </div>
                {strat.kpis?.total_r != null && (
                  <span
                    className="text-sm font-mono font-bold"
                    style={{ color: strat.kpis.total_r >= 0 ? 'var(--green)' : 'var(--red)' }}
                  >
                    {fmtKpi(strat.kpis.total_r, 2, 'R')}
                  </span>
                )}
              </div>

              {/* Strategy name */}
              <p className="text-[10px] mb-1.5 truncate" style={{ color: 'var(--text-muted)' }}>
                {strat.name}
              </p>

              {/* KPI row */}
              <div className="grid grid-cols-3 gap-2 text-[10px]">
                <div>
                  <span style={{ color: 'var(--text-muted)' }}>WR</span>
                  <p className="font-mono" style={{ color: 'var(--text-secondary)' }}>
                    {strat.kpis?.win_rate != null ? `${strat.kpis.win_rate.toFixed(1)}%` : '--'}
                  </p>
                </div>
                <div>
                  <span style={{ color: 'var(--text-muted)' }}>PF</span>
                  <p className="font-mono" style={{ color: 'var(--text-secondary)' }}>
                    {strat.kpis?.profit_factor != null ? strat.kpis.profit_factor.toFixed(2) : '--'}
                  </p>
                </div>
                <div>
                  <span style={{ color: 'var(--text-muted)' }}>Trades</span>
                  <p className="font-mono" style={{ color: 'var(--text-secondary)' }}>
                    {strat.kpis?.total_trades ?? '--'}
                  </p>
                </div>
              </div>
            </div>
          </Link>
        ))
      )}
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function DashboardPage() {
  const { data: summary, isLoading, error } = useDashboardSummary();

  // Inject animation styles on mount
  useEffect(() => {
    const id = 'dashboard-page-animations';
    if (typeof document !== 'undefined' && !document.getElementById(id)) {
      const style = document.createElement('style');
      style.id = id;
      style.textContent = ANIMATION_STYLES;
      document.head.appendChild(style);
    }
  }, []);

  // ---------------------------------------------------------------------------
  // Loading state
  // ---------------------------------------------------------------------------

  if (isLoading) {
    return <DashboardSkeleton />;
  }

  // ---------------------------------------------------------------------------
  // Error state
  // ---------------------------------------------------------------------------

  if (error) {
    return (
      <div>
        <PageHeader title="Dashboard" subtitle="Error loading dashboard" />
        <Card>
          <div className="text-center py-8" style={{ color: 'var(--red)' }}>
            Failed to load dashboard data. Check your connection and try again.
          </div>
        </Card>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // Derived data
  // ---------------------------------------------------------------------------

  const strategyCount = summary?.strategy_count ?? 0;
  const portfolioCount = summary?.portfolio_count ?? 0;
  const monitoredCount = summary?.monitored_count ?? 0;
  const totalTrades = summary?.total_trades ?? 0;
  const totalR = summary?.total_r ?? 0;
  const avgWinRate = summary?.avg_win_rate ?? 0;
  const strategies = summary?.strategies ?? [];

  // Header KPI metrics — derived from real API data
  const headerKpis = [
    { label: 'Strategies', value: strategyCount.toString() },
    { label: 'Portfolios', value: portfolioCount.toString() },
    { label: 'Monitored', value: monitoredCount.toString(), highlight: monitoredCount > 0 },
    { label: 'Total Trades', value: totalTrades.toLocaleString() },
    { label: 'Total R', value: fmtKpi(totalR), color: totalR >= 0 ? 'var(--green)' : 'var(--red)' },
    { label: 'Avg Win Rate', value: avgWinRate > 0 ? `${avgWinRate.toFixed(1)}%` : '--' },
  ];

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div style={{ animation: 'dash-fade-in 0.3s ease-out' }} suppressHydrationWarning>
      <PageHeader
        title="Dashboard"
        subtitle="Trading Cockpit"
        actions={
          <div className="flex items-center gap-2">
            <span
              className="text-xs px-2 py-1 rounded-full flex items-center gap-1.5"
              style={{ background: 'rgba(76,175,80,0.15)', color: 'var(--green)' }}
            >
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--green)' }} />
              Live
            </span>

            {/* Quick action icons */}
            <div className="flex items-center gap-1">
              {[
                { label: 'Strategy Builder', href: '/strategy-builder', icon: 'M12 4v16m-8-8h16' },
                { label: 'Portfolios', href: '/portfolios', icon: 'M4 6h16M4 12h16M4 18h16' },
                { label: 'Alerts', href: '/alerts', icon: 'M18 8A6 6 0 006 8c0 7-3 9-3 9h18s-3-2-3-9M13.73 21a2 2 0 01-3.46 0' },
              ].map((action) => (
                <Link key={action.label} href={action.href}>
                  <div
                    className="w-8 h-8 rounded-lg flex items-center justify-center transition-colors"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
                    title={action.label}
                  >
                    <svg
                      width="14" height="14" viewBox="0 0 24 24" fill="none"
                      stroke="var(--text-muted)" strokeWidth="2"
                      strokeLinecap="round" strokeLinejoin="round"
                    >
                      <path d={action.icon} />
                    </svg>
                  </div>
                </Link>
              ))}
            </div>
          </div>
        }
      />

      {/* ============ KPI STRIP ============ */}
      <div className="grid grid-cols-6 gap-3 mb-5">
        {headerKpis.map((kpi) => (
          <div
            key={kpi.label}
            className="rounded-lg border p-3"
            style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}
          >
            <p className="text-[10px] mb-0.5" style={{ color: 'var(--text-muted)' }}>
              {kpi.label}
            </p>
            <div className="flex items-baseline gap-1.5">
              <span
                className="text-lg font-semibold"
                style={{ color: kpi.color || ('highlight' in kpi && kpi.highlight ? 'var(--green)' : 'var(--text-primary)') }}
              >
                {kpi.value}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* ============ MAIN CONTENT: 7/5 grid split ============ */}
      <div className="grid grid-cols-12 gap-5 mb-5">

        {/* -- LEFT COLUMN: Charts (7 cols) -- */}
        <div className="col-span-7 space-y-5">

          {/* Portfolio Equity Curve — placeholder (no equity curve data from this endpoint) */}
          <Card>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>
                Portfolio Equity Curve
              </h3>
            </div>
            <PlaceholderWidget
              label="Equity curve data not yet available from API"
              height={260}
              icon="~"
            />
          </Card>

          {/* Daily P&L Bar Chart — placeholder */}
          <Card>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>
                Daily P&L
              </h3>
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-sm" style={{ background: 'var(--green)' }} />
                  <span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Profit</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-sm" style={{ background: 'var(--red)' }} />
                  <span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Loss</span>
                </div>
              </div>
            </div>
            <PlaceholderWidget
              label="Daily P&L data not yet available from API"
              height={220}
              icon="|"
            />
          </Card>

          {/* P&L Calendar Heatmap — placeholder */}
          <Card>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>
                P&L Calendar
              </h3>
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-sm" style={{ background: 'var(--green)' }} />
                  <span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Profit</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-sm" style={{ background: 'var(--red)' }} />
                  <span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Loss</span>
                </div>
              </div>
            </div>
            <PlaceholderWidget
              label="Calendar heatmap not yet available from API"
              height={180}
              icon="#"
            />
          </Card>
        </div>

        {/* -- RIGHT COLUMN: Widgets (5 cols) -- */}
        <div className="col-span-5 space-y-5">

          {/* Active Positions / Monitored Strategies — real data */}
          <MonitoredStrategies strategies={strategies} />

          {/* Performance Health — placeholder */}
          <Card>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>
                Performance Health
              </h3>
              <div className="flex gap-0.5 rounded-lg overflow-hidden" style={{ border: '1px solid var(--border)' }}>
                {['Portfolios', 'Strategies'].map((label) => (
                  <span
                    key={label}
                    className="px-2.5 py-1 text-[10px] font-medium"
                    style={{ background: 'transparent', color: 'var(--text-muted)' }}
                  >
                    {label}
                  </span>
                ))}
              </div>
            </div>

            {/* SD legend */}
            <div className="flex items-center gap-3 mb-3 px-2 py-1.5 rounded-lg" style={{ background: 'var(--bg-input)' }}>
              <div className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full" style={{ background: 'var(--green)' }} />
                <span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>&lt;1 SD</span>
              </div>
              <div className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full" style={{ background: 'var(--orange, #FF9800)' }} />
                <span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>1-2 SD</span>
              </div>
              <div className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full" style={{ background: 'var(--red)' }} />
                <span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>&gt;2 SD</span>
              </div>
            </div>

            <PlaceholderWidget
              label="Health deviation data not yet available from API"
              height={120}
            />
          </Card>

          {/* Market Regime — placeholder */}
          <Card>
            <h3 className="text-xs font-semibold mb-2" style={{ color: 'var(--text-muted)' }}>
              Market Regime
            </h3>
            <PlaceholderWidget
              label="Market regime data not yet available from API"
              height={100}
              icon="O"
            />
          </Card>

          {/* Monthly Goal Tracker — placeholder */}
          <Card>
            <h3 className="text-xs font-semibold mb-3" style={{ color: 'var(--text-muted)' }}>
              Monthly Goal
            </h3>
            <PlaceholderWidget
              label="Goal tracking not yet available from API"
              height={80}
            />
          </Card>
        </div>
      </div>

      {/* ============ BOTTOM ROW: Issues + Activity ============ */}
      <div className="grid grid-cols-2 gap-5 mb-5">
        {/* Issues & Warnings — placeholder */}
        <Card>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>
              Issues & Warnings
            </h3>
            <span
              className="text-[9px] px-1.5 py-0.5 rounded-full font-medium"
              style={{ background: 'rgba(76,175,80,0.15)', color: 'var(--green)' }}
            >
              0 active
            </span>
          </div>
          <p className="text-xs text-center py-4" style={{ color: 'var(--text-muted)' }}>
            No active issues
          </p>
        </Card>

        {/* Recent Activity Feed — placeholder */}
        <Card>
          <h3 className="text-xs font-semibold mb-3" style={{ color: 'var(--text-muted)' }}>
            Recent Activity
          </h3>
          <p className="text-xs text-center py-4" style={{ color: 'var(--text-muted)' }}>
            Activity feed not yet available from API
          </p>
        </Card>
      </div>
    </div>
  );
}
