'use client';

/**
 * Portfolio Detail — Faithful copy of V5 design with mock data replaced by API hooks.
 * Source: src/app/portfolios/[id]/versions/V5.tsx
 *
 * Data convention:
 *   Real value = live from API
 *   -- = wired but no data yet
 *   {{field}} = not wired, needs backend work
 */

import { useState, useEffect, useMemo } from 'react';
import Link from 'next/link';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import Card from '@/components/Card';
import Modal from '@/components/Modal';
import MetricCard from '@/components/MetricCard';
import ChartPlaceholder from '@/components/ChartPlaceholder';
import EquityCurve from '@/charts/EquityCurve';
import DistributionChart from '@/charts/DistributionChart';
import CombinedEquityCurve from '@/charts/CombinedEquityCurve';
import DrawdownChart from '@/charts/DrawdownChart';
import DailyBarChart from '@/charts/DailyBarChart';
import SparkLine from '@/charts/SparkLine';
import BalanceHistoryChart from '@/charts/BalanceHistoryChart';
import {
  usePortfolios,
  usePortfolio,
  usePortfolioCompute,
  usePortfolioTrades,
  usePortfolioAnomalies,
  usePortfolioAccount,
  usePortfolioWorstCase,
  usePortfolioCapitalUtilization,
  usePortfolioOpenPositions,
  usePortfolioRequirementsCheck,
} from '@/hooks/queries/usePortfolios';
import { useStrategies } from '@/hooks/queries/useStrategies';
import {
  useDeletePortfolio, useDuplicatePortfolio, useReanalyzePortfolio,
  useAddLedgerEntry, useRemoveLedgerEntry, useUpdatePortfolio,
} from '@/hooks/mutations/usePortfolioMutations';
import { useWebhookGroups } from '@/hooks/queries/useWebhookGroups';
import { useWebhookDeliveryLog } from '@/hooks/queries/useWebhooks';
import PortfolioDetailV5 from '@/app/portfolios/[id]/versions/V5';

const statusColors: Record<string, string> = {
  'On Track': 'var(--green)',
  'Outperforming': 'var(--blue)',
  'Underperforming': 'var(--red)',
  'Insufficient Data': 'var(--text-muted)',
};

/* =========================================================================
   EMPTY-STATE DEFAULTS (replace mock constants — tabs use these until worker provides data)
   ========================================================================= */

// ---- Live Dashboard ----
const EMPTY_LIVE_KPIS = { alertTrades: 0, winRate: 0, totalPnL: 0, expectedPnL: 0, vsPlan: 0 };
const EMPTY_BUYING_POWER = { startingBalance: 0, currentBalance: 0, allocated: 0, available: 0, utilization: 0 };

// ---- Performance ----
const EMPTY_PERF_KPIS = { trades: 0, winRate: 0, pf: 0, totalPnL: 0, balance: 0, maxDD: 0 };

// ---- Prop Firm ----
const EMPTY_REQ_SET = { name: '--', status: 'Insufficient Data' as string };

// ---- Account ----
const EMPTY_ACCOUNT = { currentBalance: 0, startingBalance: 0, netDeposits: 0, tradingPnL: 0 };

// Mock data has been removed. All tab components now show empty states / -- for data
// that needs worker compute (Live Dashboard, Performance, Prop Firm Check, Webhooks).
// Account tab wires to usePortfolioAccount.

/* =========================================================================
   HELPER COMPONENTS
   ========================================================================= */

// ---- SVG Mini Equity Curve ----
function MiniEquityCurve({ data, width = 320, height = 80, gradientId = 'eqGrad', color }: {
  data: number[];
  width?: number;
  height?: number;
  gradientId?: string;
  color?: string;
}) {
  if (data.length < 2) return null;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const padding = 4;
  const chartH = height - padding * 2;
  const chartW = width - padding * 2;

  const points = data.map((v, i) => {
    const x = padding + (i / (data.length - 1)) * chartW;
    const y = padding + chartH - ((v - min) / range) * chartH;
    return `${x},${y}`;
  });

  const pathD = `M${points.join(' L')}`;
  const fillD = `${pathD} L${padding + chartW},${height} L${padding},${height} Z`;
  const finalVal = data[data.length - 1];
  const strokeColor = color || (finalVal >= 0 ? 'var(--green)' : 'var(--red)');

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={strokeColor} stopOpacity={0.25} />
          <stop offset="100%" stopColor={strokeColor} stopOpacity={0} />
        </linearGradient>
      </defs>
      <path d={fillD} fill={`url(#${gradientId})`} />
      <path d={pathD} fill="none" stroke={strokeColor} strokeWidth="2" strokeLinejoin="round" />
    </svg>
  );
}

// ---- R-Distribution Sparkline ----
function RDistributionSparkline({ data, width = 200, height = 40 }: {
  data: number[];
  width?: number;
  height?: number;
}) {
  const max = Math.max(...data.map(Math.abs)) || 1;
  const barW = (width - data.length) / data.length;
  const mid = height / 2;

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      {data.map((v, i) => {
        const barH = (Math.abs(v) / max) * (height / 2 - 2);
        const y = v >= 0 ? mid - barH : mid;
        return (
          <rect
            key={i}
            x={i * (barW + 1)}
            y={y}
            width={barW}
            height={barH}
            fill={v >= 0 ? 'var(--green)' : 'var(--red)'}
            opacity={0.8}
            rx={1}
          />
        );
      })}
      <line x1={0} y1={mid} x2={width} y2={mid} stroke="var(--border)" strokeWidth={0.5} />
    </svg>
  );
}

// ---- Progress Bar ----
function ProgressBar({ pct, color = 'var(--accent)', height = 8 }: {
  pct: number;
  color?: string;
  height?: number;
}) {
  const clampedPct = Math.min(100, Math.max(0, pct));
  return (
    <div className="w-full rounded-full overflow-hidden" style={{ background: 'var(--bg-input)', height }}>
      <div
        className="rounded-full transition-all"
        style={{ width: `${clampedPct}%`, height: '100%', background: color }}
      />
    </div>
  );
}

// ---- Status Badge ----
function StatusBadge({ status }: { status: 'Healthy' | 'Warning' | 'Critical' | string }) {
  const colors: Record<string, { bg: string; text: string }> = {
    Healthy: { bg: 'var(--green-muted)', text: 'var(--green)' },
    Warning: { bg: 'var(--orange-muted)', text: 'var(--orange)' },
    Critical: { bg: 'var(--red-muted)', text: 'var(--red)' },
    Streaming: { bg: 'var(--green-muted)', text: 'var(--green)' },
    Disabled: { bg: 'rgba(156,163,175,0.15)', text: 'var(--text-muted)' },
    Running: { bg: 'var(--green-muted)', text: 'var(--green)' },
  };
  const c = colors[status] || { bg: 'var(--accent-muted)', text: 'var(--accent)' };

  return (
    <span
      className="text-xs px-2 py-0.5 rounded-full font-medium"
      style={{ background: c.bg, color: c.text }}
    >
      {status}
    </span>
  );
}

// ---- Confidence Stars ----
function ConfidenceStars({ rating }: { rating: number }) {
  return (
    <span className="inline-flex gap-0.5">
      {[1, 2, 3, 4, 5].map((star) => (
        <span
          key={star}
          className="text-sm"
          style={{ color: star <= rating ? 'var(--orange)' : 'var(--border)' }}
        >
          *
        </span>
      ))}
    </span>
  );
}

/* =========================================================================
   TAB COMPONENTS
   ========================================================================= */

const TABS = [
  'Live Dashboard', 'Performance', 'Strategies', 'Prop Firm Check',
  'Account', 'Webhooks', 'Design Ref',
];

// ---- 1. Live Dashboard ----
function LiveDashboardTab({ portfolioId }: { portfolioId?: number }) {
  const [bpDate, setBpDate] = useState(new Date().toISOString().split('T')[0]);
  const [showTradeChart, setShowTradeChart] = useState<number | null>(null);
  const [anomalyTab, setAnomalyTab] = useState('All');
  const [dataMode, setDataMode] = useState<'Planned' | 'Executed'>('Planned');

  // Live data hooks
  const { data: portfolio } = usePortfolio(portfolioId ?? null);
  const { data: computeData } = usePortfolioCompute(portfolioId ?? null, ['kpis']);
  const { data: openPositionsData } = usePortfolioOpenPositions(portfolioId ?? null);
  const { data: capUtil } = usePortfolioCapitalUtilization(portfolioId ?? null, 'alerts');
  // alert_kpis is only surfaced on the ?preview=true list endpoint (not on
  // the single-portfolio GET or compute). Reuse the list query so it stays
  // cached across page navigation from the Portfolios list.
  const { data: portfolioList } = usePortfolios({ preview: true });

  const preview = useMemo(
    () => (portfolioList || []).find((p: any) => p?.id === portfolioId),
    [portfolioList, portfolioId],
  );

  const alertKpis: any =
    (preview as any)?.alert_kpis ||
    (portfolio as any)?.alert_kpis ||
    (computeData as any)?.alert_kpis ||
    {};
  const plannedKpis: any =
    (preview as any)?.kpis ||
    (portfolio as any)?.kpis ||
    (computeData as any)?.kpis ||
    {};
  const openPositions: any[] = (openPositionsData as any)?.open_positions || [];
  const startingBalance = (portfolio as any)?.starting_balance ?? 0;
  const currentNotional = openPositions.reduce(
    (sum: number, p: any) => sum + (Number(p.buying_power_used) || 0), 0,
  );
  const currentRisk = openPositions.reduce(
    (sum: number, p: any) => sum + (Number(p.risk_per_trade) || 0), 0,
  );
  // For cash-account semantics, "available" clamps at zero when notional
  // exceeds balance. Users can see both the raw notional and the risk.
  const currentAvailable = Math.max(0, startingBalance - currentRisk);
  const peakAllocated = (capUtil as any)?.peak_allocated ?? (capUtil as any)?.peak ?? null;

  const fmtR = (v: any) => v == null ? '--' : `${v >= 0 ? '+' : ''}${Number(v).toFixed(2)}R`;
  const fmtPct = (v: any) => v == null ? '--' : `${Number(v).toFixed(1)}%`;
  const fmtUsd = (v: any) => v == null ? '--' : `$${Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
  const fmtTime = (iso?: string) => {
    if (!iso || iso === '--') return '--';
    try {
      const d = new Date(iso);
      if (isNaN(d.getTime())) return '--';
      return d.toLocaleString('en-US', {
        month: 'short', day: 'numeric',
        hour: '2-digit', minute: '2-digit', hour12: false,
      });
    } catch { return '--'; }
  };
  const durationFrom = (iso?: string) => {
    if (!iso) return '--';
    const d = new Date(iso); if (isNaN(d.getTime())) return '--';
    const mins = Math.floor((Date.now() - d.getTime()) / 60000);
    if (mins < 60) return `${mins}m`;
    const h = Math.floor(mins / 60); const m = mins % 60;
    if (h < 24) return `${h}h${m > 0 ? ` ${m}m` : ''}`;
    const days = Math.floor(h / 24);
    return `${days}d ${h % 24}h`;
  };

  const shiftDate = (days: number) => {
    const d = new Date(bpDate);
    d.setDate(d.getDate() + days);
    setBpDate(d.toISOString().split('T')[0]);
  };

  return (
    <div>
      {/* Visibility disclaimer + data mode toggle */}
      <div className="flex items-center justify-between mb-4 px-3 py-2 rounded-lg" style={{ background: 'var(--accent-muted)', border: '1px solid var(--accent)30' }}>
        <span className="text-xs" style={{ color: 'var(--accent)' }}>
          Dashboard metrics reflect <strong>visible strategies</strong> from the Strategies tab.
        </span>
        <div className="flex items-center gap-4 flex-shrink-0">
          <div className="flex items-center gap-2">
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Data:</span>
            {(['Planned', 'Executed'] as const).map((mode) => (
              <button
                key={mode}
                className="text-xs px-2.5 py-1 rounded-full"
                style={{
                  background: dataMode === mode ? 'var(--bg-card)' : 'transparent',
                  color: dataMode === mode ? 'var(--text-primary)' : 'var(--text-muted)',
                  border: dataMode === mode ? '1px solid var(--border)' : '1px solid transparent',
                  cursor: 'pointer', fontWeight: dataMode === mode ? 600 : 400,
                }}
                onClick={() => setDataMode(mode)}
                title={mode === 'Planned' ? 'Planned quantity — assumes unlimited buying power, shows strategy performance' : 'Executed quantity — actual transactions based on available buying power'}
              >
                {mode}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>TQ:</span>
            <select className="text-xs px-2 py-1 rounded-full" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-primary)', cursor: 'pointer' }} defaultValue="None">
              <option value="None">None</option>
              <option value="ttp">Trade The Pool</option>
              <option value="ftmo">FTMO</option>
              <option value="topstep">Topstep</option>
              <option value="custom">My Custom Rules</option>
            </select>
          </div>
        </div>
      </div>

      {/* KPI Row — live (alert-sourced) vs planned (backtest) */}
      <div className="grid grid-cols-5 gap-3 mb-6">
        <MetricCard
          label="Alert Trades"
          value={alertKpis.total_trades != null ? String(alertKpis.total_trades) : '--'}
        />
        <MetricCard
          label="Win Rate"
          value={fmtPct(alertKpis.win_rate)}
        />
        <MetricCard
          label="Total P&L"
          value={fmtR(alertKpis.total_r)}
        />
        <MetricCard
          label="Expected P&L"
          value={fmtR(plannedKpis.total_r)}
        />
        <MetricCard
          label="vs Plan"
          value={
            alertKpis.total_r != null && plannedKpis.total_r != null && plannedKpis.total_r !== 0
              ? `${((alertKpis.total_r / plannedKpis.total_r - 1) * 100).toFixed(1)}%`
              : '--'
          }
        />
      </div>

      {/* Performance vs Plan Chart */}
      <Card className="mb-6">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Performance vs Plan</h3>
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--text-muted)' }}>
              <span className="w-3 h-0.5 inline-block" style={{ background: 'var(--accent)', opacity: 0.5 }} /> Benchmark
            </span>
            <span className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--text-muted)' }}>
              <span className="w-3 h-0.5 inline-block rounded" style={{ background: 'rgba(99,102,241,0.2)' }} /> Confidence Band
            </span>
            <span className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--text-muted)' }}>
              <span className="w-3 h-0.5 inline-block" style={{ background: 'var(--green)' }} /> Actual
            </span>
          </div>
        </div>
        <div className="flex items-center justify-center py-12" style={{ height: 400, color: 'var(--text-muted)' }}>
          <div className="text-center">
            <p className="text-sm mb-1">Performance vs Plan chart requires portfolio compute data.</p>
            <p className="text-xs">Click <strong>Re-Analyze</strong> to generate benchmark + actual equity curves.</p>
          </div>
        </div>
      </Card>

      {/* Open Positions — live from /api/portfolios/{id}/open-positions.
          Notional = qty × entry_price (total position value, not margin).
          Risk = risk_per_trade (actual $ at risk until stop hits). */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
          Open Positions
          <span className="ml-2 text-xs px-1.5 py-0.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
            {openPositions.length}
          </span>
        </h3>

        <div style={{ overflowX: 'auto' }}>
          <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                {['Strategy', 'Symbol', 'Dir', 'Entry', 'Stop', 'Qty', 'Notional', 'Risk', 'Duration'].map((h) => (
                  <th key={h} className="text-left py-2 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)', background: 'var(--bg-secondary)' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {openPositions.length === 0 ? (
                <tr><td colSpan={9} className="py-6 text-center text-sm" style={{ color: 'var(--text-muted)' }}>
                  No open positions.
                </td></tr>
              ) : openPositions.map((p: any, i: number) => (
                <tr key={p.strategy_id ?? i} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td className="py-2 px-3 text-sm">{p.strategy_name || '--'}</td>
                  <td className="py-2 px-3 text-sm" style={{ fontFamily: 'monospace' }}>{p.symbol || '--'}</td>
                  <td className="py-2 px-3 text-sm" style={{ color: p.direction === 'LONG' ? 'var(--green)' : 'var(--red)', fontWeight: 600 }}>
                    {p.direction || '--'}
                  </td>
                  <td className="py-2 px-3 text-sm" style={{ fontFamily: 'monospace' }}>
                    {p.entry_price != null ? `$${Number(p.entry_price).toFixed(2)}` : '--'}
                    {p.entry_time && (
                      <div className="text-xs" style={{ color: 'var(--text-muted)' }}>{fmtTime(p.entry_time)}</div>
                    )}
                  </td>
                  <td className="py-2 px-3 text-sm" style={{ fontFamily: 'monospace', color: 'var(--red)' }}>
                    {p.stop_price != null ? `$${Number(p.stop_price).toFixed(2)}` : '--'}
                  </td>
                  <td className="py-2 px-3 text-sm" style={{ fontFamily: 'monospace' }}>{p.quantity ?? '--'}</td>
                  <td
                    className="py-2 px-3 text-sm"
                    style={{ fontFamily: 'monospace' }}
                    title="Notional value: quantity × entry price. Total position size."
                  >
                    {p.buying_power_used != null ? fmtUsd(p.buying_power_used) : '--'}
                  </td>
                  <td
                    className="py-2 px-3 text-sm"
                    style={{ fontFamily: 'monospace', color: 'var(--orange)' }}
                    title="Capital at risk: risk_per_trade (loss if stop hits)."
                  >
                    {p.risk_per_trade != null ? fmtUsd(p.risk_per_trade) : '--'}
                  </td>
                  <td className="py-2 px-3 text-sm" style={{ color: 'var(--text-muted)' }}>{durationFrom(p.entry_time)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Buying Power Tracker */}
      <Card className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Buying Power Tracker</h3>
          <div className="flex items-center gap-1">
            <button
              className="text-xs px-2 py-1 rounded"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-muted)', cursor: 'pointer' }}
              onClick={() => shiftDate(-1)}
            >
              &larr;
            </button>
            <input
              type="date"
              value={bpDate}
              onChange={(e) => setBpDate(e.target.value)}
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '4px 8px', borderRadius: '6px', fontSize: '0.75rem' }}
            />
            <button
              className="text-xs px-2 py-1 rounded"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-muted)', cursor: 'pointer' }}
              onClick={() => shiftDate(1)}
            >
              &rarr;
            </button>
          </div>
        </div>
        <div className="grid grid-cols-4 gap-4 mb-4">
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Starting Balance</p>
            <p className="text-lg font-semibold">{fmtUsd(startingBalance)}</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Available</p>
            <p className="text-lg font-semibold" style={{ color: 'var(--green)' }}>{fmtUsd(currentAvailable)}</p>
            <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>balance − risk</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Committed</p>
            <p className="text-sm font-semibold" style={{ color: 'var(--orange)' }}>
              {fmtUsd(currentNotional)} <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>notional</span>
            </p>
            <p className="text-sm font-semibold" style={{ color: 'var(--orange)' }}>
              {fmtUsd(currentRisk)} <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>risk</span>
            </p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Peak Allocated Today</p>
            <p className="text-lg font-semibold">{peakAllocated != null ? fmtUsd(peakAllocated) : '--'}</p>
          </div>
        </div>
        <ChartPlaceholder label={`24-hour buying power chart (${bpDate}): Available BP over time — needs worker data`} height={180} />
        <div className="flex items-center justify-between mt-2">
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Risk utilization: {startingBalance > 0 ? `${((currentRisk / startingBalance) * 100).toFixed(1)}%` : '--'}
            {' '}&middot; Notional utilization: {startingBalance > 0 ? `${((currentNotional / startingBalance) * 100).toFixed(0)}%` : '--'}
            {' '}&middot; Concurrent positions: {openPositions.length}
          </p>
        </div>
      </Card>

      {/* Anomaly Detection */}
      <Card className="mb-6">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            Anomaly Detection
          </h3>
          <div className="flex gap-1">
            {['All', 'Alert Issues', 'Performance'].map((t) => (
              <button
                key={t}
                className="text-xs px-2.5 py-1 rounded-full"
                style={{
                  background: anomalyTab === t ? 'var(--accent-muted)' : 'var(--bg-input)',
                  color: anomalyTab === t ? 'var(--accent)' : 'var(--text-muted)',
                  border: anomalyTab === t ? '1px solid var(--accent)' : '1px solid var(--border)',
                  cursor: 'pointer',
                }}
                onClick={() => setAnomalyTab(t)}
              >
                {t}
              </button>
            ))}
          </div>
        </div>
        <p className="text-xs py-6 text-center" style={{ color: 'var(--text-muted)' }}>
          No anomalies detected. Anomalies will appear here once the portfolio monitor is active.
        </p>
      </Card>

      {/* Trade History — matching Streamlit 13 columns */}
      <Card>
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Trade History</h3>

        <div style={{ overflowX: 'auto' }}>
          <table className="w-full text-sm" style={{ borderCollapse: 'collapse', minWidth: 900 }}>
            <thead>
              <tr>
                {['#', 'Strategy', 'Symbol', 'Dir', 'Entry $', 'Exit $', 'Reason', 'P Qty', 'E Qty', 'R', 'P&L', 'Status', ''].map((h) => (
                  <th key={h} className="text-left py-2 px-2 text-xs font-medium" style={{ color: 'var(--text-muted)', background: 'var(--bg-secondary)', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr><td colSpan={13} className="py-6 text-center text-sm" style={{ color: 'var(--text-muted)' }}>No trade history — worker data not available yet.</td></tr>
            </tbody>
          </table>
        </div>

        {/* Trade chart modal */}
        {/* Trade chart modal — will be wired when worker provides live trade data */}
      </Card>
    </div>
  );
}

// ---- 2. Performance ----
function PerformanceTab({ portfolioId }: { portfolioId?: number }) {
  const [mcShuffle, setMcShuffle] = useState<'daily' | 'weekly' | 'individual'>('daily');
  const [mcSims, setMcSims] = useState(1000);
  const [mcTriggered, setMcTriggered] = useState(false);

  const { data: perfData } = usePortfolioCompute(
    portfolioId ?? null,
    ['kpis', 'equity_curve_with_strategies', 'daily_pnl', 'correlation', 'drawdown'],
  );
  const { data: mcData, refetch: refetchMc, isFetching: mcFetching } = usePortfolioCompute(
    mcTriggered ? (portfolioId ?? null) : null,
    ['monte_carlo'],
  );
  const { data: worstCase } = usePortfolioWorstCase(portfolioId ?? null);
  const { data: capUtil } = usePortfolioCapitalUtilization(portfolioId ?? null, 'backtest');
  const { data: reqCheck } = usePortfolioRequirementsCheck(portfolioId ?? null);

  const k = perfData?.kpis || {};
  const ecData = perfData?.equity_curve_with_strategies;
  const dpnlRaw: any[] = Array.isArray(perfData?.daily_pnl) ? perfData!.daily_pnl : [];
  const drawdownSeries: any[] = Array.isArray(perfData?.drawdown) ? perfData!.drawdown : [];
  const corr = perfData?.correlation;
  const mc = mcData?.monte_carlo;

  // Extract threshold rules for reference lines
  const rules = reqCheck?.rules || [];
  const maxDDRule = rules.find((r: any) => r.type === 'max_total_drawdown_pct');
  const maxDailyLossRule = rules.find((r: any) => r.type === 'max_daily_loss_pct');
  const dailyPauseRule = rules.find((r: any) => r.type === 'daily_pause_pct');
  const startingBalance = (k.final_balance != null && k.total_pnl != null)
    ? k.final_balance - k.total_pnl
    : 10000;

  // Formatters
  const fmtDollar = (v: number | null | undefined, dp = 0) =>
    v == null ? '--' : `${v < 0 ? '-' : ''}$${Math.abs(v).toLocaleString(undefined, { maximumFractionDigits: dp })}`;
  const fmtPct = (v: number | null | undefined, dp = 2) =>
    v == null ? '--' : `${v.toFixed(dp)}%`;
  const fmtPf = (v: number | null | undefined) => {
    if (v == null) return '--';
    if (!isFinite(v)) return '∞';
    return v.toFixed(2);
  };

  // Combined equity curve data
  const combinedSeries: number[] = ecData?.combined?.cumulative_pnl || [];
  const combinedTimestamps: (string | null)[] = ecData?.combined?.exit_time || [];
  const strategiesMap = useStrategies().data;

  const strategyOverlays = useMemo(() => {
    const overlays = ecData?.strategies || {};
    const strategyEntries = Object.entries(overlays).slice(0, 8); // cap to 8 overlays
    const palette = ['var(--accent)', 'var(--green)', 'var(--orange)', '#8B5CF6', '#06B6D4', '#EC4899', '#14B8A6', '#F59E0B'];
    return strategyEntries.map(([sid, series]: [string, any], i) => {
      const strat = strategiesMap?.find((s: any) => String(s.id) === sid);
      return {
        name: strat?.name || `Strategy ${sid}`,
        color: palette[i % palette.length],
        points: (series?.cumulative_pnl || []) as number[],
      };
    });
  }, [ecData, strategiesMap]);

  // Daily P&L distribution values
  const dpnlValues = dpnlRaw.map((d) => d.daily_pnl ?? d.pnl ?? d.value ?? 0);
  const bestDay = dpnlValues.length ? Math.max(...dpnlValues) : null;
  const worstDay = dpnlValues.length ? Math.min(...dpnlValues) : null;

  // Daily Peak Capital bars from capital-utilization timeline
  const peakCapitalByDay = useMemo(() => {
    const timeline: any[] = capUtil?.timeline || [];
    if (!Array.isArray(timeline) || timeline.length === 0) return [];
    const byDate: Record<string, number> = {};
    for (const e of timeline) {
      const ts = e.timestamp || e.date || e.time;
      if (!ts) continue;
      const d = typeof ts === 'string' ? ts.split('T')[0] : String(ts);
      const cap = e.capital_deployed ?? 0;
      if (!(d in byDate) || cap > byDate[d]) byDate[d] = cap;
    }
    return Object.entries(byDate)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([date, value]) => ({ date, value }));
  }, [capUtil]);

  // Daily P&L vs Limits data — color-coded by breach
  const dailyPnlBars = useMemo(() => {
    const maxLossPL = maxDailyLossRule?.threshold != null ? -(startingBalance * maxDailyLossRule.threshold / 100) : null;
    const pausePL = dailyPauseRule?.threshold != null ? -(startingBalance * dailyPauseRule.threshold / 100) : null;
    return dpnlRaw.map((d: any) => {
      const dateRaw = d.date || d.day || '';
      const date = typeof dateRaw === 'string' && dateRaw.length > 10 ? dateRaw.slice(0, 10) : String(dateRaw);
      const value = d.daily_pnl ?? d.pnl ?? d.value ?? 0;
      let fill: string | undefined;
      if (maxLossPL != null && value <= maxLossPL) fill = 'var(--red)';
      else if (pausePL != null && value <= pausePL) fill = 'var(--orange)';
      else fill = value >= 0 ? 'var(--green)' : 'var(--red)';
      return { date, value, fill };
    });
  }, [dpnlRaw, maxDailyLossRule, dailyPauseRule, startingBalance]);

  const dailyPnlRefLines = useMemo(() => {
    const lines: { value: number; color: string; label: string; dashed?: boolean }[] = [];
    if (maxDailyLossRule?.threshold != null) {
      lines.push({
        value: -(startingBalance * maxDailyLossRule.threshold / 100),
        color: 'var(--red)',
        label: `Max Loss -${maxDailyLossRule.threshold}%`,
      });
    }
    if (dailyPauseRule?.threshold != null) {
      lines.push({
        value: -(startingBalance * dailyPauseRule.threshold / 100),
        color: 'var(--orange)',
        label: `Pause -${dailyPauseRule.threshold}%`,
      });
    }
    return lines;
  }, [maxDailyLossRule, dailyPauseRule, startingBalance]);

  const handleRunMC = () => {
    setMcTriggered(true);
    refetchMc();
  };

  return (
    <div>
      {/* KPI Row */}
      <div className="grid grid-cols-6 gap-3 mb-6">
        <MetricCard label="Trades" value={(k.total_trades ?? 0).toString()} />
        <MetricCard label="Win Rate" value={fmtPct(k.win_rate, 1)} />
        <MetricCard label="PF" value={fmtPf(k.profit_factor)} />
        <MetricCard label="Total P&L" value={fmtDollar(k.total_pnl)} />
        <MetricCard label="Balance" value={fmtDollar(k.final_balance)} />
        <MetricCard label="Max DD" value={fmtPct(k.max_drawdown_pct, 2)} />
      </div>

      {/* Combined Equity Curve */}
      <Card className="mb-6">
        <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Combined Equity Curve</h3>
          <div className="flex items-center gap-3 flex-wrap">
            {strategyOverlays.map((o) => (
              <span key={o.name} className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--text-muted)' }}>
                <span
                  className="w-3 h-0.5 inline-block"
                  style={{
                    background: `repeating-linear-gradient(to right, ${o.color} 0 4px, transparent 4px 7px)`,
                    opacity: 0.7,
                  }}
                />
                {o.name}
              </span>
            ))}
            <span className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--text-muted)' }}>
              <span className="w-3 h-0.5 inline-block" style={{ background: 'var(--text-primary)' }} />
              <strong>Combined</strong>
            </span>
          </div>
        </div>
        {combinedSeries.length === 0 ? (
          <div className="flex items-center justify-center py-8" style={{ color: 'var(--text-muted)', height: 320 }}>
            <span className="text-xs">No equity data — click Re-Analyze to compute.</span>
          </div>
        ) : (
          <CombinedEquityCurve
            combined={combinedSeries}
            strategyOverlays={strategyOverlays}
            timestamps={combinedTimestamps}
            xAxis="time"
            height={320}
          />
        )}
      </Card>

      {/* Drawdown Analysis */}
      <Card className="mb-6">
        <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Drawdown Analysis</h3>
          <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-muted)' }}>
            <span className="flex items-center gap-1">
              <span className="w-3 h-0.5 inline-block" style={{ background: 'var(--red)' }} /> Drawdown
            </span>
            {maxDDRule && (
              <span className="flex items-center gap-1">
                <span className="w-3 h-0.5 inline-block" style={{ background: 'var(--orange)' }} /> Max DD Limit (-{maxDDRule.threshold}%)
              </span>
            )}
          </div>
        </div>
        {drawdownSeries.length === 0 ? (
          <div className="flex items-center justify-center" style={{ height: 220, color: 'var(--text-muted)' }}>
            <span className="text-xs">No drawdown data — click Re-Analyze.</span>
          </div>
        ) : (
          <DrawdownChart
            data={drawdownSeries}
            maxDDLimit={maxDDRule?.threshold ?? null}
            height={220}
          />
        )}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-3">
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Max Drawdown</p>
            <p className="text-sm font-semibold" style={{ color: 'var(--red)' }}>
              {fmtPct(k.max_drawdown_pct, 2)} ({fmtDollar(k.max_drawdown_dollars)})
            </p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Profitable Days</p>
            <p className="text-sm font-semibold">
              {k.profitable_days_count != null && k.total_trading_days != null
                ? `${fmtPct(k.profitable_days_pct, 0)} (${k.profitable_days_count}/${k.total_trading_days})`
                : '--'}
            </p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Avg Daily P&L</p>
            <p className="text-sm font-semibold" style={{ color: (k.avg_daily_pnl ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
              {fmtDollar(k.avg_daily_pnl)} (std: {fmtDollar(k.daily_pnl_std)})
            </p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Trades / Day</p>
            <p className="text-sm font-semibold">
              {k.trades_per_day != null ? k.trades_per_day.toFixed(2) : '--'}
            </p>
          </div>
        </div>
      </Card>

      {/* Daily P&L Distribution + Strategy Correlation Heatmap */}
      <div className="grid grid-cols-2 gap-6">
        <Card>
          <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Daily P&L Distribution</h3>
          {(() => {
            const dpnl = perfData?.daily_pnl;
            const vals = Array.isArray(dpnl) ? dpnl.map((d: any) => d.daily_pnl ?? d.pnl ?? d.value ?? 0) : [];
            if (vals.length === 0) return <div className="flex items-center justify-center py-8" style={{ height: 300, color: 'var(--text-muted)' }}><span className="text-xs">No daily P&L data — click Re-Analyze</span></div>;
            return <DistributionChart values={vals} bins={15} height={300} />;
          })()}
          <div className="flex items-center gap-6 mt-3">
            <div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Avg Daily</p>
              <p className="text-sm font-semibold" style={{ color: (k.avg_daily_pnl ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
                {fmtDollar(k.avg_daily_pnl)}
              </p>
            </div>
            <div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Std Dev</p>
              <p className="text-sm font-semibold">{fmtDollar(k.daily_pnl_std)}</p>
            </div>
            <div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Best Day</p>
              <p className="text-sm font-semibold" style={{ color: 'var(--green)' }}>{fmtDollar(bestDay)}</p>
            </div>
            <div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Worst Day</p>
              <p className="text-sm font-semibold" style={{ color: 'var(--red)' }}>{fmtDollar(worstDay)}</p>
            </div>
          </div>
        </Card>

        <Card>
          <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Strategy Correlation Heatmap</h3>
          {(() => {
            const corr = perfData?.correlation;
            if (!corr || Object.keys(corr).length === 0) {
              return <p className="text-xs py-8 text-center" style={{ color: 'var(--text-muted)' }}>Correlation data will appear once enough trade history is available — click Re-Analyze.</p>;
            }
            const keys = Object.keys(corr);
            const getColor = (v: number | null) => {
              if (v == null) return 'var(--bg-input)';
              const abs = Math.abs(v);
              if (v >= 0) return `rgba(76,175,80,${abs * 0.7})`;
              return `rgba(244,67,54,${abs * 0.7})`;
            };
            return (
              <div style={{ overflowX: 'auto' }}>
                <table className="text-xs" style={{ borderCollapse: 'collapse' }}>
                  <thead>
                    <tr><th style={{ padding: '4px 8px' }} />{keys.map(k => <th key={k} style={{ padding: '4px 8px', color: 'var(--text-muted)', fontWeight: 500 }}>{k}</th>)}</tr>
                  </thead>
                  <tbody>
                    {keys.map(row => (
                      <tr key={row}>
                        <td style={{ padding: '4px 8px', fontWeight: 600, color: 'var(--text-secondary)' }}>{row}</td>
                        {keys.map(col => {
                          const val = corr[row]?.[col] ?? null;
                          return <td key={col} style={{ padding: '4px 8px', textAlign: 'center', background: getColor(val), color: 'white', fontFamily: 'monospace', borderRadius: 2 }}>{val != null ? val.toFixed(2) : '--'}</td>;
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
          })()}
        </Card>
      </div>

      {/* ---- Risk Analytics ---- */}
      <h3 className="text-sm font-semibold mt-6 mb-4" style={{ color: 'var(--text-primary)' }}>Risk Analytics</h3>

      {/* Daily Peak Capital Deployed */}
      <Card className="mb-4">
        <h4 className="text-sm font-medium mb-3">Daily Peak Capital Deployed</h4>
        <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
          Shows the maximum buying power used on each day across the date range. The red dashed line marks your account balance — days that approach or exceed it indicate buying power constraints that may have affected executed quantities.
        </p>
        {peakCapitalByDay.length === 0 ? (
          <div className="flex items-center justify-center" style={{ height: 220, color: 'var(--text-muted)' }}>
            <span className="text-xs">
              {capUtil?.unavailable_reason
                ? capUtil.unavailable_reason
                : 'Capital utilization data not available for this portfolio.'}
            </span>
          </div>
        ) : (() => {
          const accountBalance = startingBalance;
          const coloredBars = peakCapitalByDay.map((d) => ({
            ...d,
            fill: d.value > accountBalance ? 'var(--red)' : 'var(--accent)',
          }));
          return (
            <DailyBarChart
              data={coloredBars}
              referenceLines={[{ value: accountBalance, color: 'var(--red)', label: `Balance $${accountBalance.toFixed(0)}` }]}
              height={220}
            />
          );
        })()}
        <div className="grid grid-cols-4 gap-4 mt-3">
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Peak Capital Deployed</p>
            <p className="text-sm font-bold">{fmtDollar(capUtil?.peak_capital_deployed)}</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Peak as % of Final</p>
            <p className="text-sm font-bold">{fmtPct(capUtil?.peak_capital_pct, 1)}</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Insufficient-Capital Events</p>
            <p className="text-sm font-bold" style={{
              color: (capUtil?.insufficient_capital_events?.length ?? 0) > 0 ? 'var(--red)' : 'var(--green)',
            }}>
              {capUtil?.insufficient_capital_events?.length ?? 0}
            </p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Max Concurrent Positions</p>
            <p className="text-sm font-bold">{capUtil?.max_concurrent_positions ?? '--'}</p>
          </div>
        </div>
      </Card>

      {/* Daily P&L vs Limits */}
      <Card className="mb-4">
        <h4 className="text-sm font-medium mb-3">Daily P&L vs Limits</h4>
        {dailyPnlBars.length === 0 ? (
          <div className="flex items-center justify-center" style={{ height: 220, color: 'var(--text-muted)' }}>
            <span className="text-xs">No daily P&L data — click Re-Analyze.</span>
          </div>
        ) : (
          <DailyBarChart
            data={dailyPnlBars}
            referenceLines={dailyPnlRefLines}
            height={220}
          />
        )}
        <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
          {worstCase?.days_breach_daily_pause ?? 0} days breaching daily pause limit
          &middot; {worstCase?.days_breach_max_daily_loss ?? 0} days breaching max daily loss limit
        </p>
      </Card>

      {/* Worst-Case Analysis */}
      <Card className="mb-4">
        <h4 className="text-sm font-medium mb-3">Worst-Case Analysis</h4>
        {!worstCase ? (
          <p className="text-xs py-4 text-center" style={{ color: 'var(--text-muted)' }}>
            Loading worst-case analysis…
          </p>
        ) : (
          <>
            <div className="grid grid-cols-3 sm:grid-cols-5 gap-3 mb-4">
              <div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Worst Single Day</p>
                <p className="text-sm font-bold" style={{ color: 'var(--red)' }}>
                  {fmtDollar(worstCase.worst_day_dollars)}
                </p>
                <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                  {worstCase.worst_day_date || '--'}
                </p>
              </div>
              <div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Worst Losing Streak</p>
                <p className="text-sm font-bold" style={{ color: 'var(--red)' }}>
                  {worstCase.worst_streak_days ?? 0} days
                </p>
                <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                  {fmtDollar(worstCase.worst_streak_dollars)}
                </p>
              </div>
              <div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Worst 5-Day Rolling</p>
                <p className="text-sm font-bold" style={{ color: 'var(--red)' }}>
                  {fmtDollar(worstCase.worst_5day_rolling_dd)}
                </p>
              </div>
              <div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Days Breaching Pause</p>
                <p className="text-sm font-bold" style={{
                  color: (worstCase.days_breach_daily_pause ?? 0) > 0 ? 'var(--orange)' : 'var(--green)',
                }}>
                  {worstCase.days_breach_daily_pause ?? 0}
                </p>
              </div>
              <div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Days Breaching Max Loss</p>
                <p className="text-sm font-bold" style={{
                  color: (worstCase.days_breach_max_daily_loss ?? 0) > 0 ? 'var(--red)' : 'var(--green)',
                }}>
                  {worstCase.days_breach_max_daily_loss ?? 0}
                </p>
              </div>
            </div>
            <h5 className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>Top 5 Worst Days</h5>
            <div style={{ overflowX: 'auto' }}>
              <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    {['Date', 'P&L ($)', 'P&L (%)', 'Cumulative DD %', 'Status'].map((h) => (
                      <th key={h} className="text-left py-2 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)', background: 'var(--bg-secondary)' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(worstCase.top_5_worst_days || []).length === 0 ? (
                    <tr><td colSpan={5} className="py-4 text-center text-sm" style={{ color: 'var(--text-muted)' }}>No daily P&L data.</td></tr>
                  ) : (worstCase.top_5_worst_days || []).map((d: any) => (
                    <tr key={d.date} style={{ borderBottom: '1px solid var(--border)' }}>
                      <td className="py-2 px-3 text-xs font-mono">{d.date}</td>
                      <td className="py-2 px-3 text-xs font-mono" style={{ color: 'var(--red)' }}>
                        {fmtDollar(d.pnl_dollars)}
                      </td>
                      <td className="py-2 px-3 text-xs font-mono" style={{ color: 'var(--red)' }}>
                        {fmtPct(d.pnl_pct, 2)}
                      </td>
                      <td className="py-2 px-3 text-xs">{fmtPct(d.cumulative_dd_pct, 2)}</td>
                      <td className="py-2 px-3">
                        {d.breach_status && d.breach_status !== 'None' ? (
                          <span className="text-xs px-1.5 py-0.5 rounded" style={{
                            background: d.breach_status === 'MAX_LOSS' ? 'rgba(239,68,68,0.2)' : 'rgba(251,146,60,0.2)',
                            color: d.breach_status === 'MAX_LOSS' ? 'var(--red)' : 'var(--orange)',
                          }}>
                            {d.breach_status}
                          </span>
                        ) : (
                          <span className="text-xs px-1.5 py-0.5 rounded" style={{ background: 'rgba(34,197,94,0.15)', color: 'var(--green)' }}>
                            Normal
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </Card>

      {/* Monte Carlo Simulation */}
      <Card>
        <h4 className="text-sm font-medium mb-3">Monte Carlo Simulation</h4>
        <div className="flex items-center gap-4 mb-4 flex-wrap">
          <label className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Shuffle Mode:
            <select
              className="ml-2"
              value={mcShuffle}
              onChange={(e) => setMcShuffle(e.target.value as 'daily' | 'weekly' | 'individual')}
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '4px 8px', borderRadius: '6px', fontSize: '0.75rem' }}
            >
              <option value="daily">Daily</option>
              <option value="weekly">Weekly</option>
              <option value="individual">Individual</option>
            </select>
          </label>
          <label className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Simulations:
            <input
              type="range" min={500} max={5000} step={500}
              value={mcSims}
              onChange={(e) => setMcSims(Number(e.target.value))}
              style={{ marginLeft: 8, verticalAlign: 'middle' }}
            />
            <span className="font-mono ml-1">{mcSims}</span>
          </label>
          <button
            onClick={handleRunMC}
            disabled={!portfolioId || mcFetching}
            className="px-3 py-1.5 rounded text-xs font-medium"
            style={{
              background: 'var(--accent)', color: 'white', border: 'none',
              cursor: mcFetching ? 'not-allowed' : 'pointer',
              opacity: mcFetching ? 0.6 : 1,
            }}
          >
            {mcFetching ? 'Running…' : 'Run Simulation'}
          </button>
        </div>
        {!mc ? (
          <p className="text-xs py-4 text-center" style={{ color: 'var(--text-muted)' }}>
            {mcTriggered ? 'Running simulation…' : 'Click Run Simulation to generate Monte Carlo metrics.'}
          </p>
        ) : (
          <>
            <div className="grid grid-cols-3 sm:grid-cols-6 gap-3 mb-4">
              <div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Bust Probability</p>
                <p className="text-sm font-bold" style={{
                  color: (mc.bust_probability ?? 0) > 5 ? 'var(--red)' : 'var(--green)',
                }}>
                  {fmtPct(mc.bust_probability, 1)}
                </p>
              </div>
              <div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Daily Pause Prob</p>
                <p className="text-sm font-bold">{fmtPct(mc.daily_pause_probability, 1)}</p>
              </div>
              <div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Max Loss Prob</p>
                <p className="text-sm font-bold">{fmtPct(mc.max_daily_loss_probability, 1)}</p>
              </div>
              <div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Median Max DD</p>
                <p className="text-sm font-bold" style={{ color: 'var(--red)' }}>{fmtPct(mc.median_max_dd, 1)}</p>
              </div>
              <div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>95th Pctl DD</p>
                <p className="text-sm font-bold" style={{ color: 'var(--red)' }}>{fmtPct(mc.p95_max_dd, 1)}</p>
              </div>
              <div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Expected Worst Day</p>
                <p className="text-sm font-bold" style={{ color: 'var(--red)' }}>{fmtPct(mc.expected_worst_day, 1)}</p>
              </div>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {Array.isArray(mc.max_dd_values) && mc.max_dd_values.length > 0 ? (
                <div>
                  <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>Max Drawdown Distribution</p>
                  <DistributionChart values={mc.max_dd_values} bins={40} height={200} />
                </div>
              ) : null}
              {Array.isArray(mc.worst_day_values) && mc.worst_day_values.length > 0 ? (
                <div>
                  <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>Worst Day Distribution</p>
                  <DistributionChart values={mc.worst_day_values} bins={40} height={200} />
                </div>
              ) : null}
            </div>
            <p className="text-xs mt-3" style={{ color: 'var(--text-muted)' }}>
              {mc.n_simulations?.toLocaleString?.() ?? mc.n_simulations} simulations · {mc.shuffle_mode ?? mcShuffle} shuffle
            </p>
          </>
        )}
      </Card>
    </div>
  );
}

// ---- 3. Strategies ----
function StrategiesTab({ portfolioId }: { portfolioId?: number }) {
  const [visibility, setVisibility] = useState<Record<string, boolean>>({});
  const [activeState, setActiveState] = useState<Record<string, boolean>>({});

  const { data: portfolio } = usePortfolio(portfolioId ?? null);
  const { data: allStrategies } = useStrategies();
  const { data: perfData } = usePortfolioCompute(
    portfolioId ?? null,
    ['kpis', 'equity_curve_with_strategies'],
  );

  const toggleVisibility = (id: string) => setVisibility((prev) => ({ ...prev, [id]: !prev[id] }));
  const toggleActive = (id: string) => setActiveState((prev) => ({ ...prev, [id]: !prev[id] }));

  // Build per-strategy row data from the portfolio's strategies allocation + strategy details
  const portStrategies: any[] = portfolio?.strategies || [];
  const strategiesMap = useMemo(() => {
    const m = new Map<number, any>();
    for (const s of allStrategies || []) m.set(s.id, s);
    return m;
  }, [allStrategies]);

  const overlays = perfData?.equity_curve_with_strategies?.strategies || {};
  const portfolioTotalPnl = perfData?.kpis?.total_pnl ?? 0;

  const rows = useMemo(() => {
    return portStrategies.map((alloc) => {
      const sid = alloc.strategy_id;
      const strat = strategiesMap.get(sid) || {};
      const kpis = strat.kpis || {};
      const eq = overlays[String(sid)];
      const equityCurve: number[] = eq?.cumulative_pnl || [];
      // P&L contribution: the strategy's portion of total portfolio P&L (last cumulative value)
      const contribution = equityCurve.length > 0
        ? equityCurve[equityCurve.length - 1]
        : null;
      const contributionPct = portfolioTotalPnl !== 0 && contribution != null
        ? (contribution / portfolioTotalPnl) * 100
        : null;
      return {
        id: String(sid),
        name: strat.name || `Strategy ${sid}`,
        symbol: strat.symbol || '--',
        direction: strat.direction || 'LONG',
        timeframe: strat.timeframe,
        riskPerTrade: alloc.risk_per_trade ?? 0,
        winRate: kpis.win_rate,
        pf: kpis.profit_factor,
        trades: kpis.total_trades,
        avgR: kpis.avg_r ?? kpis.average_r,
        // Strategy-level KPIs use R-based drawdown (max_r_drawdown), not %
        maxDrawdownR: kpis.max_r_drawdown,
        status: strat.status || 'Healthy',
        pnlContribution: contribution,
        pnlContributionPct: contributionPct,
        equityCurve,
      };
    });
  }, [portStrategies, strategiesMap, overlays, portfolioTotalPnl]);

  const visibleCount = rows.filter((r) => visibility[r.id] !== false).length;
  const activeCount = rows.filter((r) => activeState[r.id] !== false).length;

  const fmtPct = (v: number | null | undefined, dp = 1) =>
    v == null ? '--' : `${v.toFixed(dp)}%`;
  const fmtPf = (v: number | null | undefined) => {
    if (v == null) return '--';
    if (!isFinite(v)) return '∞';
    return v.toFixed(2);
  };
  const fmtDollar = (v: number | null | undefined) =>
    v == null ? '--' : `${v < 0 ? '-' : ''}$${Math.abs(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;

  return (
    <div>
      {/* Controls row */}
      <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          {rows.length} strategies &middot; {visibleCount} visible on dashboard &middot; {activeCount} active
        </p>
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          Live/forward-test sigma badges arrive with live alert data (M8.5)
        </p>
      </div>

      <div className="space-y-4">
        {rows.length === 0 ? (
          <Card>
            <p className="text-sm text-center py-6" style={{ color: 'var(--text-muted)' }}>
              No strategies assigned to this portfolio yet. Edit the portfolio to add strategies.
            </p>
          </Card>
        ) : rows.map((strat) => {
          const isVisible = visibility[strat.id] !== false;
          const isActive = activeState[strat.id] !== false;
          return (
            <div key={strat.id} style={!isVisible ? { opacity: 0.6 } : undefined}><Card>
              {/* Header row */}
              <div className="flex items-start justify-between mb-3">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <p className="font-medium">{strat.name}</p>
                    <StatusBadge status={strat.status} />
                    {!isActive && (
                      <span className="text-xs px-1.5 py-0.5 rounded" style={{ background: 'var(--orange)' + '20', color: 'var(--orange)' }}>Paused</span>
                    )}
                    {!isVisible && (
                      <span className="text-xs px-1.5 py-0.5 rounded" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>Hidden</span>
                    )}
                  </div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    {strat.symbol} &middot; {strat.direction}
                    {strat.timeframe ? ` · ${strat.timeframe}` : ''} &middot; ${strat.riskPerTrade}/trade
                  </p>
                </div>
              </div>

              {/* KPIs */}
              <div className="grid grid-cols-5 gap-4 mb-3">
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Win Rate</p>
                  <p className="text-sm font-semibold">{fmtPct(strat.winRate)}</p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Profit Factor</p>
                  <p className="text-sm font-semibold">{fmtPf(strat.pf)}</p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Trades</p>
                  <p className="text-sm font-semibold">{strat.trades ?? '--'}</p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Max DD (R)</p>
                  <p className="text-sm font-semibold" style={{ color: 'var(--red)' }}>
                    {strat.maxDrawdownR != null ? `${strat.maxDrawdownR.toFixed(2)}R` : '--'}
                  </p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>P&L Contribution</p>
                  <p className="text-sm font-semibold" style={{ color: (strat.pnlContribution ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
                    {strat.pnlContribution == null
                      ? '--'
                      : <>
                          {strat.pnlContribution >= 0 ? '+' : ''}{fmtDollar(strat.pnlContribution)}
                          {strat.pnlContributionPct != null && (
                            <span className="text-[10px] ml-1" style={{ color: 'var(--text-muted)' }}>
                              ({strat.pnlContributionPct.toFixed(0)}%)
                            </span>
                          )}
                        </>}
                  </p>
                </div>
              </div>

              {/* Equity curve (strategy's contribution to portfolio) */}
              <div className="rounded-lg overflow-hidden mb-2" style={{ background: 'var(--bg-input)' }}>
                {strat.equityCurve.length > 1 ? (
                  <SparkLine data={strat.equityCurve} height={64} />
                ) : (
                  <div className="flex items-center justify-center" style={{ height: 64, color: 'var(--text-muted)' }}>
                    <span className="text-xs">No equity data for this strategy</span>
                  </div>
                )}
              </div>

              {/* Action row */}
              <div className="flex items-center gap-2 pt-2" style={{ borderTop: '1px solid var(--border)' }}>
                <Link
                  href={`/strategies/${strat.id}`}
                  className="px-3 py-1 rounded text-xs"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)', textDecoration: 'none' }}
                >
                  View Strategy
                </Link>
                <button
                  className="px-3 py-1 rounded text-xs"
                  style={{
                    background: isVisible ? 'var(--accent-muted)' : 'var(--bg-input)',
                    border: isVisible ? '1px solid var(--accent)' : '1px solid var(--border)',
                    color: isVisible ? 'var(--accent)' : 'var(--text-muted)',
                    cursor: 'pointer',
                  }}
                  onClick={() => toggleVisibility(strat.id)}
                >
                  {isVisible ? 'Visible' : 'Hidden'}
                </button>
                <span className="flex-1" />
                {/* Active/Paused toggle */}
                <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{isActive ? 'Active' : 'Paused'}</span>
                <div
                  className="relative w-7 h-4 rounded-full cursor-pointer flex-shrink-0"
                  style={{ background: isActive ? 'var(--green)' : 'var(--bg-input)', border: isActive ? 'none' : '1px solid var(--border)' }}
                  onClick={() => toggleActive(strat.id)}
                  title={isActive ? 'Active — webhooks will fire for this strategy' : 'Paused — alerts still track but webhooks are suppressed'}
                >
                  <div className="absolute top-0.5 w-3 h-3 rounded-full transition-all" style={{ background: 'white', left: isActive ? '12px' : '2px' }} />
                </div>
              </div>
            </Card></div>
          );
        })}
      </div>
    </div>
  );
}

// ---- 4. Prop Firm Check ----
function PropFirmCheckTab({ portfolioId }: { portfolioId?: number }) {
  const { data: mcData } = usePortfolioCompute(portfolioId ?? null, ['monte_carlo']);
  const { data: reqCheck } = usePortfolioRequirementsCheck(portfolioId ?? null);
  const mc = mcData?.monte_carlo || null;

  // Map backend rules into UI shape (V5 structure)
  const complianceRules = (reqCheck?.rules || []).map((r: any) => {
    const pct = r.threshold && r.actual != null
      ? Math.min(100, Math.abs((r.actual / r.threshold) * 100))
      : 0;
    return {
      name: r.name || r.type,
      type: r.type,
      threshold: r.limit_display ?? String(r.threshold ?? '--'),
      currentValue: r.value_display ?? String(r.actual ?? '--'),
      currentPct: pct,
      passing: !!r.passed,
      violations: [] as string[],
      margin: r.margin,
    };
  });

  const requirementSetData = {
    name: reqCheck?.firm_name || '--',
    status: reqCheck?.overall_pass === false
      ? 'Violations'
      : reqCheck?.overall_pass === true
        ? 'All Passing'
        : EMPTY_REQ_SET.status,
  };
  const passingCount = complianceRules.filter((r: any) => r.passing).length;
  const totalRules = complianceRules.length;

  return (
    <div>
      {/* Header */}
      <Card className="mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold">{requirementSetData.name}</h3>
            <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>
              {passingCount}/{totalRules} rules passing
            </p>
          </div>
          <span
            className="text-sm px-3 py-1 rounded-full font-medium"
            style={{
              background: requirementSetData.status === 'Violations' ? 'var(--red-muted)' : 'var(--green-muted)',
              color: requirementSetData.status === 'Violations' ? 'var(--red)' : 'var(--green)',
            }}
          >
            {requirementSetData.status === 'Violations' ? 'Has Violations' : 'All Passing'}
          </span>
        </div>
      </Card>

      {/* Compliance Rules */}
      <div className="space-y-4 mb-6">
        {complianceRules.map((rule, i) => (
          <Card key={i}>
            <div className="flex items-start justify-between mb-3">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <p className="font-medium">{rule.name}</p>
                  <span
                    className="text-xs px-2 py-0.5 rounded-full font-medium"
                    style={{
                      background: rule.passing ? 'var(--green-muted)' : 'var(--red-muted)',
                      color: rule.passing ? 'var(--green)' : 'var(--red)',
                    }}
                  >
                    {rule.passing ? 'PASS' : 'FAIL'}
                  </span>
                </div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  Threshold: {rule.threshold} &middot; margin {rule.margin != null ? (rule.margin >= 0 ? '+' : '') + rule.margin.toFixed(2) : '--'}
                </p>
              </div>
              <div className="text-right">
                <p className="text-sm font-semibold">{rule.currentValue}</p>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>current</p>
              </div>
            </div>

            {/* Progress Bar */}
            <div className="mb-2">
              <div className="flex items-center justify-between mb-1">
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  {rule.currentPct.toFixed(1)}% of limit
                </p>
              </div>
              <ProgressBar
                pct={rule.currentPct}
                color={rule.passing ? (rule.currentPct > 75 ? 'var(--orange)' : 'var(--green)') : 'var(--red)'}
                height={8}
              />
            </div>

            {/* Violation History */}
            {rule.violations.length > 0 && (
              <div className="mt-3 pt-3 border-t" style={{ borderColor: 'var(--border)' }}>
                {rule.violations.map((violation: string, vi: number) => (
                  <div key={vi} className="flex items-center gap-2 py-1">
                    <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: 'var(--red)' }} />
                    <p className="text-xs" style={{ color: 'var(--red)' }}>{violation}</p>
                  </div>
                ))}
              </div>
            )}
          </Card>
        ))}
      </div>

      {/* Daily Limit Tracker */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Daily Loss Limit Tracker</h3>
        {(() => {
          const maxLoss = complianceRules.find((r: any) => r.type === 'max_daily_loss_pct');
          const pause = complianceRules.find((r: any) => r.type === 'daily_pause_pct');
          const rule = maxLoss || pause;
          if (!rule) {
            return (
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                No daily loss rule in the active requirement set.
              </p>
            );
          }
          return (
            <>
              <div className="grid grid-cols-4 gap-4 mb-4">
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Daily Limit</p>
                  <p className="text-lg font-semibold">{rule.threshold}</p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Worst Observed Day</p>
                  <p className="text-lg font-semibold" style={{ color: 'var(--orange)' }}>{rule.currentValue}</p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Margin to Limit</p>
                  <p className="text-lg font-semibold" style={{ color: (rule.margin ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
                    {rule.margin != null ? `${rule.margin >= 0 ? '+' : ''}${rule.margin.toFixed(2)}%` : '--'}
                  </p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Status</p>
                  <p className="text-lg font-semibold" style={{ color: rule.passing ? 'var(--green)' : 'var(--red)' }}>
                    {rule.passing ? 'PASS' : 'FAIL'}
                  </p>
                </div>
              </div>
              <ProgressBar
                pct={rule.currentPct}
                color={rule.passing ? (rule.currentPct > 75 ? 'var(--orange)' : 'var(--green)') : 'var(--red)'}
                height={12}
              />
              <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
                {rule.currentPct.toFixed(1)}% of daily limit used at worst observed day.
              </p>
            </>
          );
        })()}
      </Card>

      {/* Worst Case Analysis */}
      <Card>
        <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Worst Case Analysis (Monte Carlo)</h3>
        {!mc ? (
          <div className="flex items-center justify-center py-8" style={{ color: 'var(--text-muted)' }}>
            <span className="text-xs">Monte Carlo data loading or not available — click Re-Analyze to compute</span>
          </div>
        ) : (
          <>
            <div className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
              {mc.n_simulations?.toLocaleString() ?? '--'} simulations ({mc.shuffle_mode ?? '--'} shuffle)
            </div>
            <div className="flex items-center gap-6 mt-3">
              <div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>95th Percentile DD</p>
                <p className="text-sm font-semibold" style={{ color: 'var(--orange)' }}>
                  {mc.p95_max_dd != null ? `${mc.p95_max_dd.toFixed(1)}%` : '--'}
                </p>
              </div>
              <div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Median Max DD</p>
                <p className="text-sm font-semibold" style={{ color: 'var(--red)' }}>
                  {mc.median_max_dd != null ? `${mc.median_max_dd.toFixed(1)}%` : '--'}
                </p>
              </div>
              <div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Probability of Ruin</p>
                <p className="text-sm font-semibold" style={{ color: mc.bust_probability > 5 ? 'var(--red)' : 'var(--green)' }}>
                  {mc.bust_probability != null ? `${mc.bust_probability.toFixed(1)}%` : '--'}
                </p>
              </div>
              <div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Expected Worst Day</p>
                <p className="text-sm font-semibold">
                  {mc.expected_worst_day != null ? `${mc.expected_worst_day.toFixed(1)}%` : '--'}
                </p>
              </div>
            </div>
          </>
        )}
      </Card>
    </div>
  );
}

// ---- 5. Account ----
function AccountTab({ portfolioId }: { portfolioId?: number }) {
  const [modalDate, setModalDate] = useState<string | null>(null);
  const [activeModalTab, setActiveModalTab] = useState('Trades');
  const [depositAmount, setDepositAmount] = useState('');
  const [depositDate, setDepositDate] = useState(new Date().toISOString().split('T')[0]);
  const [depositNote, setDepositNote] = useState('');
  const [withdrawAmount, setWithdrawAmount] = useState('');
  const [withdrawDate, setWithdrawDate] = useState(new Date().toISOString().split('T')[0]);
  const [withdrawNote, setWithdrawNote] = useState('');

  const { data: account } = usePortfolioAccount(portfolioId ?? null);
  const addLedger = useAddLedgerEntry();
  const removeLedger = useRemoveLedgerEntry();

  const ledger: any[] = Array.isArray(account?.ledger) ? account!.ledger : [];

  // Compute metrics
  const deposits = ledger.filter((e) => e.type === 'deposit').reduce((s, e) => s + Math.abs(e.amount ?? 0), 0);
  const withdrawals = ledger.filter((e) => e.type === 'withdrawal').reduce((s, e) => s + Math.abs(e.amount ?? 0), 0);
  const tradingPnl = ledger.filter((e) => e.type === 'trading_pnl').reduce((s, e) => s + (e.amount ?? 0), 0);
  const netDeposits = deposits - withdrawals;
  const currentBalance = account?.balance ?? 0;
  // Starting balance = first deposit entry amount, or 0
  const startingBalance = (() => {
    const firstDeposit = ledger.find((e) => e.type === 'deposit');
    return firstDeposit ? Math.abs(firstDeposit.amount ?? 0) : 0;
  })();

  // Running balance history for chart
  const sortedLedger = [...ledger].sort((a, b) => String(a.date || '').localeCompare(String(b.date || '')));
  let running = 0;
  const balanceHistory = sortedLedger.map((e) => {
    running += e.amount ?? 0;
    return { date: e.date, balance: running };
  });
  // Transform for DailyBarChart (reusable chart component — it's actually a line series, not a bar, so use EquityCurve-style)
  const balanceHistoryBars = balanceHistory.map((pt) => ({ date: pt.date, value: pt.balance }));

  const modalEntry = ledger.find((e: any) => e.date === modalDate);

  const fmtDollar = (v: number | null | undefined) =>
    v == null ? '--' : `${v < 0 ? '-' : ''}$${Math.abs(v).toLocaleString(undefined, { maximumFractionDigits: 2 })}`;

  const handleDeposit = () => {
    if (!portfolioId || !depositAmount || Number(depositAmount) <= 0) return;
    addLedger.mutate({
      id: portfolioId,
      entry: { entry_type: 'deposit', amount: Number(depositAmount), date: depositDate, note: depositNote },
    }, {
      onSuccess: () => { setDepositAmount(''); setDepositNote(''); },
    });
  };
  const handleWithdraw = () => {
    if (!portfolioId || !withdrawAmount || Number(withdrawAmount) <= 0) return;
    addLedger.mutate({
      id: portfolioId,
      entry: { entry_type: 'withdrawal', amount: Number(withdrawAmount), date: withdrawDate, note: withdrawNote },
    }, {
      onSuccess: () => { setWithdrawAmount(''); setWithdrawNote(''); },
    });
  };
  const handleDelete = (entryId: string) => {
    if (!portfolioId) return;
    removeLedger.mutate({ portfolioId, entryId });
  };

  return (
    <div>
      {/* Balance Metrics */}
      <div className="grid grid-cols-4 gap-3 mb-6">
        <MetricCard label="Current Balance" value={fmtDollar(currentBalance)} />
        <MetricCard label="Starting Balance" value={fmtDollar(startingBalance)} />
        <MetricCard label="Net Deposits" value={fmtDollar(netDeposits)} />
        <MetricCard label="Trading P&L" value={fmtDollar(tradingPnl)} />
      </div>

      {/* Balance History Chart */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Balance History</h3>
        <BalanceHistoryChart data={balanceHistory} height={200} />
      </Card>

      {/* Deposit / Withdrawal */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <Card>
          <h4 className="text-sm font-medium mb-3">Add Deposit</h4>
          <div className="flex flex-col gap-3">
            <div>
              <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>Amount ($)</label>
              <input type="number" placeholder="0.00" min="0.01" step="0.01"
                value={depositAmount}
                onChange={(e) => setDepositAmount(e.target.value)}
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '8px 12px', borderRadius: '8px', fontSize: '0.875rem', width: '100%' }} />
            </div>
            <div>
              <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>Date</label>
              <input type="date"
                value={depositDate}
                onChange={(e) => setDepositDate(e.target.value)}
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '8px 12px', borderRadius: '8px', fontSize: '0.875rem', width: '100%' }} />
            </div>
            <div>
              <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>Note (optional)</label>
              <input type="text" placeholder="e.g. Initial funding"
                value={depositNote}
                onChange={(e) => setDepositNote(e.target.value)}
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '8px 12px', borderRadius: '8px', fontSize: '0.875rem', width: '100%' }} />
            </div>
            <button
              onClick={handleDeposit}
              disabled={!portfolioId || !depositAmount || Number(depositAmount) <= 0 || addLedger.isPending}
              className="px-4 py-2 rounded-lg text-sm font-medium"
              style={{
                background: 'var(--green)', color: 'white', border: 'none',
                cursor: (!depositAmount || addLedger.isPending) ? 'not-allowed' : 'pointer',
                opacity: (!depositAmount || addLedger.isPending) ? 0.5 : 1,
              }}
            >
              {addLedger.isPending ? 'Saving…' : 'Add Deposit'}
            </button>
          </div>
        </Card>
        <Card>
          <h4 className="text-sm font-medium mb-3">Add Withdrawal</h4>
          <div className="flex flex-col gap-3">
            <div>
              <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>Amount ($)</label>
              <input type="number" placeholder="0.00" min="0.01" step="0.01"
                value={withdrawAmount}
                onChange={(e) => setWithdrawAmount(e.target.value)}
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '8px 12px', borderRadius: '8px', fontSize: '0.875rem', width: '100%' }} />
            </div>
            <div>
              <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>Date</label>
              <input type="date"
                value={withdrawDate}
                onChange={(e) => setWithdrawDate(e.target.value)}
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '8px 12px', borderRadius: '8px', fontSize: '0.875rem', width: '100%' }} />
            </div>
            <div>
              <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>Note (optional)</label>
              <input type="text" placeholder="e.g. Profit withdrawal"
                value={withdrawNote}
                onChange={(e) => setWithdrawNote(e.target.value)}
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '8px 12px', borderRadius: '8px', fontSize: '0.875rem', width: '100%' }} />
            </div>
            <button
              onClick={handleWithdraw}
              disabled={!portfolioId || !withdrawAmount || Number(withdrawAmount) <= 0 || addLedger.isPending}
              className="px-4 py-2 rounded-lg text-sm font-medium"
              style={{
                background: 'var(--red)', color: 'white', border: 'none',
                cursor: (!withdrawAmount || addLedger.isPending) ? 'not-allowed' : 'pointer',
                opacity: (!withdrawAmount || addLedger.isPending) ? 0.5 : 1,
              }}
            >
              {addLedger.isPending ? 'Saving…' : 'Add Withdrawal'}
            </button>
          </div>
        </Card>
      </div>

      {/* Ledger */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Ledger</h3>

        {/* Header */}
        <div className="grid grid-cols-12 gap-2 pb-2 border-b" style={{ borderColor: 'var(--border)' }}>
          <p className="col-span-2 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Date</p>
          <p className="col-span-3 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Type</p>
          <p className="col-span-2 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Amount</p>
          <p className="col-span-3 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Summary</p>
          <p className="col-span-2 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Action</p>
        </div>

        {/* Rows */}
        {ledger.length === 0 ? (
          <p className="text-sm text-center py-4" style={{ color: 'var(--text-muted)' }}>
            No ledger entries yet. Add a deposit to get started.
          </p>
        ) : (
          ledger.map((entry, i) => (
            <div
              key={entry.id || entry.date || i}
              className="grid grid-cols-12 gap-2 py-3 border-b items-center"
              style={{ borderColor: 'var(--border)' }}
            >
              <p className="col-span-2 text-sm">{entry.date || '--'}</p>
              <p className="col-span-3 text-sm" style={{ color: 'var(--text-secondary)', textTransform: 'capitalize' }}>
                {String(entry.type || '').replace(/_/g, ' ')}
              </p>
              <p className="col-span-2 text-sm font-medium" style={{ color: (entry.amount ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
                {(entry.amount ?? 0) >= 0 ? '+' : ''}${Math.abs(entry.amount ?? 0).toFixed(2)}
              </p>
              <p className="col-span-3 text-xs" style={{ color: 'var(--text-muted)' }}>
                {entry.note || entry.summary || '--'}
              </p>
              <div className="col-span-2 flex gap-2">
                {entry.type === 'trading_pnl' && (
                  <button
                    onClick={() => { setModalDate(entry.date); setActiveModalTab('Trades'); }}
                    className="px-3 py-1 rounded text-xs"
                    style={{ background: 'var(--accent-muted)', color: 'var(--accent)', border: 'none', cursor: 'pointer' }}
                  >
                    Details
                  </button>
                )}
                {entry.type !== 'trading_pnl' && entry.id != null && (
                  <button
                    onClick={() => handleDelete(String(entry.id))}
                    disabled={removeLedger.isPending}
                    className="px-3 py-1 rounded text-xs"
                    style={{
                      background: 'var(--red-muted, rgba(239,68,68,0.15))',
                      color: 'var(--red)',
                      border: 'none',
                      cursor: removeLedger.isPending ? 'not-allowed' : 'pointer',
                      opacity: removeLedger.isPending ? 0.5 : 1,
                    }}
                  >
                    Delete
                  </button>
                )}
              </div>
            </div>
          ))
        )}
        {!ledger.some((e) => e.type === 'trading_pnl') && (
          <p className="text-[11px] mt-3 pt-3 border-t" style={{ color: 'var(--text-muted)', borderColor: 'var(--border)' }}>
            Automatic daily Trading P&L entries arrive when live alerts are active (M8.5).
          </p>
        )}
      </Card>

      {/* Journal Section */}
      <Card>
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Trading Journal</h3>
        <div className="space-y-3">
          {ledger.filter((e) => e.journal).map((entry) => (
            <div key={entry.date} className="rounded-lg p-4 border" style={{ borderColor: 'var(--border)', background: 'var(--bg-input)' }}>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-3">
                  <p className="text-sm font-medium">{entry.date}</p>
                  <span
                    className="text-xs px-2 py-0.5 rounded"
                    style={{
                      background: entry.journal!.mood === 'Frustrated' ? 'var(--red-muted)' : 'var(--green-muted)',
                      color: entry.journal!.mood === 'Frustrated' ? 'var(--red)' : 'var(--green)',
                    }}
                  >
                    {entry.journal!.mood}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Confidence:</span>
                  <ConfidenceStars rating={entry.journal!.confidence} />
                </div>
              </div>
              <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>{entry.journal!.notes}</p>
              <div className="mt-2 flex items-center gap-2">
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Day P&L:</span>
                <span className="text-xs font-medium" style={{ color: entry.amount >= 0 ? 'var(--green)' : 'var(--red)' }}>
                  {entry.amount >= 0 ? '+' : ''}${entry.amount.toFixed(2)}
                </span>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Change History */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Change History</h3>
        <p className="text-sm text-center py-4" style={{ color: 'var(--text-muted)' }}>No change history yet.</p>
      </Card>

      {/* Daily Detail Modal */}
      <Modal
        title={`Daily Detail \u2014 ${modalDate || ''}`}
        isOpen={!!modalDate}
        onClose={() => setModalDate(null)}
        width="700px"
      >
        {/* Modal tabs */}
        <div className="flex gap-1 border-b mb-5" style={{ borderColor: 'var(--border)' }}>
          {['Trades', 'Portfolio Changes', 'Notes'].map((t) => (
            <button
              key={t}
              onClick={() => setActiveModalTab(t)}
              className="px-4 py-2 text-sm"
              style={{
                color: activeModalTab === t ? 'var(--accent)' : 'var(--text-muted)',
                borderBottom: activeModalTab === t ? '2px solid var(--accent)' : '2px solid transparent',
                marginBottom: '-1px',
              }}
            >
              {t}
            </button>
          ))}
        </div>

        {activeModalTab === 'Trades' && modalEntry && (
          <div>
            <div className="rounded-lg p-4 mb-4" style={{ background: 'var(--accent-muted)' }}>
              <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>Entry Amount</p>
              <p className="text-2xl font-bold" style={{ color: (modalEntry.amount ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
                {(modalEntry.amount ?? 0) >= 0 ? '+' : ''}${Math.abs(modalEntry.amount ?? 0).toFixed(2)}
              </p>
            </div>

            {/* Trade rows */}
            <div className="grid grid-cols-12 gap-2 pb-2 border-b" style={{ borderColor: 'var(--border)' }}>
              <p className="col-span-1 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>#</p>
              <p className="col-span-8 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Trade</p>
              <p className="col-span-3 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>P&L</p>
            </div>
            {(modalEntry.trades || []).length === 0 ? (
              <p className="text-xs py-3 text-center" style={{ color: 'var(--text-muted)' }}>
                No trade detail available for this entry.
              </p>
            ) : (modalEntry.trades || []).map((trade: any, i: number) => (
              <div key={i} className="grid grid-cols-12 gap-2 py-2.5 border-b" style={{ borderColor: 'var(--border)' }}>
                <p className="col-span-1 text-sm" style={{ color: 'var(--text-muted)' }}>{i + 1}</p>
                <p className="col-span-8 text-sm" style={{ color: 'var(--text-secondary)' }}>{trade.note}</p>
                <p className="col-span-3 text-sm font-medium" style={{ color: (trade.amount ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
                  {(trade.amount ?? 0) >= 0 ? '+' : ''}${Math.abs(trade.amount ?? 0).toFixed(2)}
                </p>
              </div>
            ))}
          </div>
        )}

        {activeModalTab === 'Portfolio Changes' && modalEntry && (
          <div>
            {(modalEntry.changes || []).length > 0 ? (
              (modalEntry.changes || []).map((change: string, i: number) => (
                <div key={i} className="flex items-center gap-3 py-3 border-b" style={{ borderColor: 'var(--border)' }}>
                  <span className="text-xs px-2 py-0.5 rounded" style={{ background: 'var(--orange-muted)', color: 'var(--orange)' }}>
                    Change
                  </span>
                  <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>{change}</span>
                </div>
              ))
            ) : (
              <p style={{ color: 'var(--text-muted)' }}>No portfolio changes on this date.</p>
            )}
          </div>
        )}

        {activeModalTab === 'Notes' && (
          <div>
            {modalEntry?.journal ? (
              <div className="mb-4 rounded-lg p-4" style={{ background: 'var(--bg-input)' }}>
                <div className="flex items-center gap-3 mb-2">
                  <span className="text-xs px-2 py-0.5 rounded" style={{
                    background: modalEntry.journal.mood === 'Frustrated' ? 'var(--red-muted)' : 'var(--green-muted)',
                    color: modalEntry.journal.mood === 'Frustrated' ? 'var(--red)' : 'var(--green)',
                  }}>
                    {modalEntry.journal.mood}
                  </span>
                  <ConfidenceStars rating={modalEntry.journal.confidence} />
                </div>
                <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>{modalEntry.journal.notes}</p>
              </div>
            ) : null}
            <textarea
              className="w-full rounded-lg p-3 text-sm"
              style={{
                background: 'var(--bg-input)',
                border: '1px solid var(--border)',
                color: 'var(--text-primary)',
                resize: 'vertical',
              }}
              rows={5}
              placeholder="Add your thoughts, observations, or context for this trading day..."
              defaultValue={modalEntry?.journal?.notes || ''}
            />
            <div className="flex items-center gap-3 mt-3">
              <select
                className="rounded-lg px-3 py-2 text-sm"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                defaultValue={modalEntry?.journal?.mood || ''}
              >
                <option value="">Mood...</option>
                {['Focused', 'Confident', 'Calm', 'Anxious', 'Frustrated', 'Excited'].map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
              <select
                className="rounded-lg px-3 py-2 text-sm"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                defaultValue={modalEntry?.journal?.confidence || 3}
              >
                {[1, 2, 3, 4, 5].map((c) => (
                  <option key={c} value={c}>Confidence: {c}/5</option>
                ))}
              </select>
              <button
                className="px-4 py-2 rounded-lg text-sm font-medium ml-auto"
                style={{ background: 'var(--accent)', color: 'white' }}
              >
                Save Note
              </button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}

// ---- 6. Webhooks ----
function WebhooksTab({ portfolioId }: { portfolioId?: number }) {
  const { data: portfolio } = usePortfolio(portfolioId ?? null);
  const { data: groups } = useWebhookGroups();
  const updatePortfolio = useUpdatePortfolio();
  const [saving, setSaving] = useState(false);

  const currentGroupId: string = portfolio?.webhook_group_id || '';
  const currentGroup = (groups || []).find((g) => g.id === currentGroupId);
  const configuredCount = currentGroup
    ? (() => {
        const isShared = (currentGroup as any).url_mode === 'shared';
        const sharedUrlSet = ((currentGroup as any).shared_url || '').trim().length > 0;
        return Object.values(currentGroup.events || {}).filter((c: any) => {
          if (!c) return false;
          const hasUrl = isShared ? sharedUrlSet : ((c.url || '').trim().length > 0);
          const hasPayload = (c.payload || '').trim().length > 0;
          return hasUrl && hasPayload;
        }).length;
      })()
    : 0;

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    if (!portfolioId || !portfolio) return;
    const next = e.target.value;
    setSaving(true);
    updatePortfolio.mutate(
      { id: portfolioId, portfolio: { ...portfolio, webhook_group_id: next || null } },
      { onSettled: () => setSaving(false) },
    );
  };

  return (
    <div>
      {/* Webhook Group Selector */}
      <Card className="mb-6">
        <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
          <h3 className="text-sm font-medium">Webhook Group</h3>
          <Link
            href="/alerts/webhook-groups"
            className="text-xs px-2 py-1 rounded"
            style={{ color: 'var(--accent)', textDecoration: 'none' }}
          >
            Manage groups →
          </Link>
        </div>
        <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
          Select your webhook group. Each group contains URL + payload for all 11 execution-driven events — set up once per account, then assigned here.
        </p>
        <div className="flex items-center gap-3">
          <select
            className="flex-1"
            value={currentGroupId}
            onChange={handleChange}
            disabled={saving}
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '8px 14px', borderRadius: '8px', fontSize: '0.875rem' }}
          >
            <option value="">No group (webhooks disabled)</option>
            {(groups || []).map((g) => (
              <option key={g.id} value={g.id}>
                {g.name}{g.service ? ` — ${g.service}` : ''}
              </option>
            ))}
          </select>
          {currentGroupId && (
            <Link
              href={`/alerts/webhook-groups/${currentGroupId}`}
              className="px-3 py-2 rounded-lg text-xs font-medium"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--accent)', textDecoration: 'none', whiteSpace: 'nowrap' }}
            >
              View Group
            </Link>
          )}
        </div>
        {currentGroup && (
          <p className="text-xs mt-3" style={{ color: 'var(--text-muted)' }}>
            {configuredCount} / 11 events configured in this group.
            {configuredCount === 0 && (
              <span style={{ color: 'var(--orange)' }}> Group is empty — no webhooks will fire until events are configured.</span>
            )}
          </p>
        )}
        {(groups || []).length === 0 && (
          <p className="text-xs mt-3" style={{ color: 'var(--text-muted)' }}>
            No webhook groups yet. <Link href="/alerts/webhook-groups" style={{ color: 'var(--accent)' }}>Create one →</Link>
          </p>
        )}
      </Card>

      {/* Delivery History — wired to /api/webhooks/delivery-log */}
      <WebhookDeliveryHistory portfolioId={portfolioId} />
    </div>
  );
}

// ---- Webhook Delivery History ----
// Pulls user-scoped delivery log and filters to this portfolio's strategies.
// Empty state still shown when nothing has fired yet.
function WebhookDeliveryHistory({ portfolioId }: { portfolioId?: number }) {
  const { data: portfolio } = usePortfolio(portfolioId ?? null);
  const { data: deliveries, isLoading } = useWebhookDeliveryLog({ limit: 100 });

  const portfolioStrategyNames = useMemo(() => {
    const names = new Set<string>();
    for (const s of (portfolio?.strategies as any[]) || []) {
      const name = s?.name ?? s?.strategy_name ?? s?.strategy?.name;
      if (name) names.add(String(name));
    }
    return names;
  }, [portfolio]);

  const filtered = useMemo(() => {
    if (!deliveries) return [];
    if (portfolioStrategyNames.size === 0) return deliveries;
    return deliveries.filter((d: any) =>
      portfolioStrategyNames.has(d.strategy_name || d.strategy || '')
    );
  }, [deliveries, portfolioStrategyNames]);

  const fmtTime = (iso: string) => {
    if (!iso || iso === '--') return '--';
    try {
      const d = new Date(iso);
      if (isNaN(d.getTime())) return '--';
      return d.toLocaleString('en-US', {
        month: 'short', day: 'numeric',
        hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
      });
    } catch { return '--'; }
  };

  const statusColor = (code: number, success?: boolean) => {
    if (success === true || (code >= 200 && code < 300)) return 'var(--green)';
    if (code === 0) return 'var(--text-muted)';
    return 'var(--red)';
  };

  return (
    <Card className="mb-6">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium">Webhook Delivery History</h3>
        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
          {filtered.length} recent
        </span>
      </div>

      {isLoading ? (
        <p className="text-sm text-center py-6" style={{ color: 'var(--text-muted)' }}>
          Loading...
        </p>
      ) : filtered.length === 0 ? (
        <p className="text-sm text-center py-6" style={{ color: 'var(--text-muted)' }}>
          No webhook deliveries yet for this portfolio&apos;s strategies.
          Deliveries will appear here once alerts fire and webhooks dispatch.
        </p>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                {['Time', 'Event', 'Strategy', 'Symbol', 'Status', 'Latency'].map((h) => (
                  <th
                    key={h}
                    className="text-left py-2 px-3 text-xs font-medium"
                    style={{
                      color: 'var(--text-muted)', background: 'var(--bg-secondary)',
                      textTransform: 'uppercase', letterSpacing: '0.05em',
                    }}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.slice(0, 50).map((d: any, i: number) => {
                const statusCode = d.status_code ?? d.status ?? 0;
                const latency = d.latency_ms ?? d.latency ?? null;
                return (
                  <tr key={d.id ?? `${d.alert_id}-${i}`}>
                    <td style={{
                      padding: '8px 12px', fontSize: '0.75rem', fontFamily: 'monospace',
                      borderBottom: '1px solid var(--border)', color: 'var(--text-secondary)',
                    }}>
                      {fmtTime(d.sent_at || d.timestamp || d.alert_timestamp)}
                    </td>
                    <td style={{
                      padding: '8px 12px', fontSize: '0.8125rem',
                      borderBottom: '1px solid var(--border)', color: 'var(--text-secondary)',
                    }}>
                      <span className="text-xs px-2 py-0.5 rounded-full" style={{
                        background: 'var(--accent-muted)', color: 'var(--accent)',
                      }}>
                        {d.event_type || d.event || d.alert_type || '--'}
                      </span>
                    </td>
                    <td style={{
                      padding: '8px 12px', fontSize: '0.8125rem',
                      borderBottom: '1px solid var(--border)', color: 'var(--text-secondary)',
                    }}>
                      {d.strategy_name || d.strategy || '--'}
                    </td>
                    <td style={{
                      padding: '8px 12px', fontSize: '0.8125rem', fontFamily: 'monospace',
                      borderBottom: '1px solid var(--border)', color: 'var(--text-secondary)',
                    }}>
                      {d.symbol || '--'}
                      {d.direction && <span style={{ color: 'var(--text-muted)', marginLeft: 4 }}>{d.direction}</span>}
                    </td>
                    <td style={{
                      padding: '8px 12px', fontSize: '0.8125rem', fontFamily: 'monospace',
                      borderBottom: '1px solid var(--border)',
                      color: statusColor(statusCode, d.success),
                      fontWeight: 600,
                    }}>
                      {statusCode || (d.success ? 'OK' : 'FAIL')}
                    </td>
                    <td style={{
                      padding: '8px 12px', fontSize: '0.75rem', fontFamily: 'monospace',
                      borderBottom: '1px solid var(--border)', color: 'var(--text-muted)',
                    }}>
                      {latency != null ? `${Number(latency).toFixed(0)}ms` : '--'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

/* =========================================================================
   MAIN V2 COMPONENT
   ========================================================================= */

const PULSE_CSS = `@keyframes pulse { 0%, 100% { transform: scale(1); opacity: 0.5; } 50% { transform: scale(2.2); opacity: 0; } }`;
const EQ_FWD_COLOR = '#FF9800';
const EQ_LIVE_COLOR = '#4CAF50';

const btnSecondary: React.CSSProperties = {
  background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)',
  padding: '6px 14px', borderRadius: '8px', fontSize: '0.875rem', cursor: 'pointer',
};

interface PortfolioDetailPageProps {
  portfolioId: number;
}

export default function PortfolioDetailPage({ portfolioId }: PortfolioDetailPageProps) {
  // ---- API Hooks (ALL before early returns) ----
  const { data: portfolio, isLoading: portfolioLoading, error: portfolioError } = usePortfolio(portfolioId);
  const { data: computeData } = usePortfolioCompute(portfolioId, ['kpis']);
  const { data: trades } = usePortfolioTrades(portfolioId);
  const { data: anomalyData } = usePortfolioAnomalies(portfolioId);
  const { data: account } = usePortfolioAccount(portfolioId);
  const { data: apiStrategiesRaw } = useStrategies();
  const deleteMut = useDeletePortfolio();
  const dupMut = useDuplicatePortfolio();
  const reanalyzeMut = useReanalyzePortfolio();

  useEffect(() => {
    const id = 'portfolio-detail-pulse-css';
    if (!document.getElementById(id)) {
      const s = document.createElement('style'); s.id = id; s.textContent = PULSE_CSS; document.head.appendChild(s);
    }
  }, []);

  // Map API data
  const p = portfolio || {} as any;
  const k = p.kpis || computeData?.kpis || {};
  const portName = p.name || '--';
  const portEnabled = p.enabled ?? false;
  const portTags = p.tags || [];
  const portStrategies = p.strategies || [];
  const portStartingBalance = p.starting_balance ?? account?.starting_balance ?? 0;
  const portCompoundRate = p.compound_rate ?? 0;
  const portWebhookTemplate = p.webhook_template_name || null;
  const portRequirementSet = p.requirement_set_name || null;

  // ---- Loading / Error states (after all hooks) ----
  if (portfolioLoading) {
    return (
      <div style={{ padding: '32px' }}>
        <h1 className="text-xl font-bold mb-4" style={{ color: 'var(--text-primary)' }}>Portfolio</h1>
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <Card key={i}><div className="animate-pulse space-y-3"><div className="h-4 rounded w-1/3" style={{ background: 'var(--border)' }} /><div className="h-3 rounded w-2/3" style={{ background: 'var(--border)' }} /><div className="h-16 rounded" style={{ background: 'var(--bg-input)' }} /></div></Card>
          ))}
        </div>
      </div>
    );
  }
  if (portfolioError) {
    return (
      <div style={{ padding: '32px' }}>
        <h1 className="text-xl font-bold mb-4" style={{ color: 'var(--text-primary)' }}>Portfolio</h1>
        <Card><div className="text-center py-8" style={{ color: 'var(--red)' }}>Failed to load portfolio.</div></Card>
      </div>
    );
  }

  return (
    <div>
      <PageHeader
        title={portName}
        backHref="/portfolios"
        actions={
          <>
            <button style={btnSecondary} onClick={() => reanalyzeMut.mutate(portfolioId)} disabled={reanalyzeMut.isPending}>
              {reanalyzeMut.isPending ? 'Analyzing...' : 'Re-Analyze'}
            </button>
            <button style={btnSecondary}>Edit</button>
            <button style={btnSecondary} onClick={() => dupMut.mutate(portfolioId)}>Clone</button>
            <button style={{ ...btnSecondary, background: 'var(--red-muted)', color: 'var(--red)', border: 'none' }} onClick={() => deleteMut.mutate(portfolioId)}>Delete</button>
          </>
        }
      />

      {/* Status badges + sigma + pulse dot */}
      <div className="flex items-center gap-3 mb-2 flex-wrap">
        <span className="text-xs font-semibold px-2.5 py-1 rounded-full" style={{ color: statusColors[k.status as string] || 'var(--text-muted)', background: (statusColors[k.status as string] || 'var(--text-muted)') + '20' }}>
          {k.status || 'Insufficient Data'}
        </span>
        {k.total_trades > 0 && (
          <span className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded" style={{ color: EQ_FWD_COLOR, background: EQ_FWD_COLOR + '18' }}>
            {k.total_trades} trades
          </span>
        )}
        {k.total_r != null && (
          <span className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded" style={{ color: k.total_r >= 0 ? EQ_LIVE_COLOR : 'var(--red)', background: (k.total_r >= 0 ? EQ_LIVE_COLOR : 'var(--red)') + '18' }}>
            {k.total_r >= 0 ? '+' : ''}{Number(k.total_r).toFixed(1)}R
          </span>
        )}
        {portEnabled && (
          <span className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--green)' }}>
            <span style={{ position: 'relative', display: 'inline-block', width: 8, height: 8 }}>
              <span style={{ position: 'absolute', inset: 0, borderRadius: '50%', background: 'var(--green)', opacity: 0.5, animation: 'pulse 2s ease-in-out infinite' }} />
              <span style={{ position: 'absolute', inset: '25%', borderRadius: '50%', background: 'var(--green)' }} />
            </span>
            Enabled
          </span>
        )}
        {portTags.map((tag: string) => (
          <span key={tag} className="text-xs px-2 py-0.5 rounded-full" style={{ color: 'var(--text-muted)', background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
            {tag}
          </span>
        ))}
      </div>

      {/* Meta line */}
      <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
        {portStrategies.length} strategies &middot; ${portStartingBalance.toLocaleString()} balance
        {portCompoundRate > 0 && <> &middot; {(portCompoundRate * 100).toFixed(0)}% scaling</>}
        {portWebhookTemplate && <span style={{ color: 'var(--accent)' }}> &middot; {portWebhookTemplate}</span>}
        {portRequirementSet && <span style={{ color: 'var(--green)' }}> &middot; {portRequirementSet}</span>}
      </p>

      <TabBar tabs={TABS}>
        {(tab) => (
          <div>
            {tab === 'Live Dashboard' && <LiveDashboardTab portfolioId={portfolioId} />}
            {tab === 'Performance' && <PerformanceTab portfolioId={portfolioId} />}
            {tab === 'Strategies' && <StrategiesTab portfolioId={portfolioId} />}
            {tab === 'Prop Firm Check' && <PropFirmCheckTab portfolioId={portfolioId} />}
            {tab === 'Account' && <AccountTab portfolioId={portfolioId} />}
            {tab === 'Webhooks' && <WebhooksTab portfolioId={portfolioId} />}
            {tab === 'Design Ref' && (
              <div>
                <div className="mb-4 p-3 rounded-lg" style={{ background: 'var(--accent-muted)', border: '1px solid var(--accent)' }}>
                  <p className="text-xs" style={{ color: 'var(--accent)' }}>
                    <strong>V5 Design Reference</strong> — original wireframe with mock data. Use this as the source-of-truth design while the other tabs are being wired with real data.
                  </p>
                </div>
                <PortfolioDetailV5 />
              </div>
            )}
          </div>
        )}
      </TabBar>
    </div>
  );
}
