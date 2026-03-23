'use client';

import { useState } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

/* ========================================================================
   Types
   ======================================================================== */

interface Portfolio {
  id: string;
  name: string;
  strategies: number;
  startingBalance: number;
  currentBalance: number;
  status: 'Live' | 'Paper' | 'Paused';
  deployStatus: 'Railway' | 'Local' | null;
  trades: number;
  winRate: number;
  pf: number;
  totalPnL: number;
  maxDD: number;
  requirementSet: string | null;
  compliant: boolean;
  violations: number;
  strategyNames: string[];
  equityCurve: number[];
  avgRiskPerTrade: number;
  compoundRate: number;
  avgTradesPerDay: number;
  webhookCount: number;
}

/* ========================================================================
   Mock Data
   ======================================================================== */

const mockPortfolios: Portfolio[] = [
  {
    id: '1',
    name: 'My Portfolio',
    strategies: 8,
    startingBalance: 80000,
    currentBalance: 84230,
    status: 'Live',
    deployStatus: 'Railway',
    trades: 847,
    winRate: 55.2,
    pf: 1.89,
    totalPnL: 4230,
    maxDD: -2.1,
    requirementSet: 'TTP 50k',
    compliant: true,
    violations: 0,
    strategyNames: ['SPY LONG', 'NVDA LONG', 'AAPL LONG', 'META LONG'],
    equityCurve: [0, 200, 150, 400, 600, 500, 800, 1200, 1100, 1500, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4230],
    avgRiskPerTrade: 250,
    compoundRate: 5,
    avgTradesPerDay: 3.2,
    webhookCount: 2,
  },
  {
    id: '2',
    name: 'Scalping Portfolio',
    strategies: 3,
    startingBalance: 25000,
    currentBalance: 26180,
    status: 'Paper',
    deployStatus: 'Local',
    trades: 312,
    winRate: 51.8,
    pf: 1.45,
    totalPnL: 1180,
    maxDD: -3.5,
    requirementSet: 'FTMO 100k',
    compliant: false,
    violations: 2,
    strategyNames: ['SPY LONG', 'QQQ SHORT', 'TSLA LONG'],
    equityCurve: [0, 100, -50, 200, 350, 300, 500, 450, 600, 700, 800, 750, 900, 1000, 1100, 1180],
    avgRiskPerTrade: 100,
    compoundRate: 0,
    avgTradesPerDay: 5.1,
    webhookCount: 0,
  },
  {
    id: '3',
    name: 'Swing Trading',
    strategies: 5,
    startingBalance: 50000,
    currentBalance: 50000,
    status: 'Paused',
    deployStatus: null,
    trades: 0,
    winRate: 0,
    pf: 0,
    totalPnL: 0,
    maxDD: 0,
    requirementSet: null,
    compliant: true,
    violations: 0,
    strategyNames: ['MSFT LONG', 'GOOG LONG', 'AMZN LONG', 'NFLX LONG', 'AMD LONG'],
    equityCurve: [],
    avgRiskPerTrade: 500,
    compoundRate: 0,
    avgTradesPerDay: 0,
    webhookCount: 0,
  },
];

type StatusFilter = 'All' | 'Live' | 'Paper' | 'Paused';

/* ========================================================================
   Helper: SVG Mini Equity Curve
   ======================================================================== */

function MiniEquityCurve({ data, gradientId, height = 80 }: {
  data: number[];
  gradientId: string;
  height?: number;
}) {
  if (data.length < 2) return null;

  const width = 400;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const pad = 4;
  const chartH = height - pad * 2;
  const chartW = width - pad * 2;

  const points = data.map((v, i) => {
    const x = pad + (i / (data.length - 1)) * chartW;
    const y = pad + chartH - ((v - min) / range) * chartH;
    return `${x},${y}`;
  });

  const pathD = `M${points.join(' L')}`;
  const fillD = `${pathD} L${pad + chartW},${height} L${pad},${height} Z`;
  const finalVal = data[data.length - 1];
  const color = finalVal >= 0 ? 'var(--green)' : 'var(--red)';
  const colorRgb = finalVal >= 0 ? '76,175,80' : '244,67,54';

  // Zero line
  const zeroY = pad + chartH - ((0 - min) / range) * chartH;

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={`rgba(${colorRgb}, 0.2)`} />
          <stop offset="100%" stopColor={`rgba(${colorRgb}, 0)`} />
        </linearGradient>
      </defs>
      <path d={fillD} fill={`url(#${gradientId})`} />
      <path d={pathD} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" />
      {min < 0 && (
        <line x1={pad} y1={zeroY} x2={pad + chartW} y2={zeroY} stroke="gray" strokeWidth="0.5" strokeDasharray="4,4" opacity="0.4" />
      )}
    </svg>
  );
}

/* ========================================================================
   Helper: Status Badge
   ======================================================================== */

function StatusBadge({ status }: { status: Portfolio['status'] }) {
  const styles: Record<Portfolio['status'], { bg: string; color: string }> = {
    Live: { bg: 'var(--green-muted)', color: 'var(--green)' },
    Paper: { bg: 'var(--accent-muted)', color: 'var(--accent)' },
    Paused: { bg: 'rgba(156,163,175,0.15)', color: 'var(--text-muted)' },
  };
  const s = styles[status];

  return (
    <span
      className="text-xs px-2 py-0.5 rounded-full font-medium"
      style={{ background: s.bg, color: s.color }}
    >
      {status}
    </span>
  );
}

/* ========================================================================
   Helper: Deploy Badge
   ======================================================================== */

function DeployBadge({ deployStatus }: { deployStatus: Portfolio['deployStatus'] }) {
  if (!deployStatus) return null;

  const isRailway = deployStatus === 'Railway';
  return (
    <span
      className="text-xs px-2 py-0.5 rounded-full"
      style={{
        background: isRailway ? 'rgba(168,85,247,0.15)' : 'rgba(156,163,175,0.1)',
        color: isRailway ? '#a855f7' : 'var(--text-muted)',
      }}
    >
      {isRailway ? 'Railway' : 'Local'}
    </span>
  );
}

/* ========================================================================
   Helper: Compliance Badge
   ======================================================================== */

function ComplianceBadge({ compliant, violations, requirementSet }: {
  compliant: boolean;
  violations: number;
  requirementSet: string | null;
}) {
  if (!requirementSet) return null;

  return (
    <div className="flex items-center gap-2 mt-2">
      <span
        className="text-xs px-2 py-0.5 rounded"
        style={{ background: 'var(--bg-input)', color: 'var(--text-secondary)' }}
      >
        {requirementSet}
      </span>
      {compliant ? (
        <span className="text-xs flex items-center gap-1" style={{ color: 'var(--green)' }}>
          <span>&#10003;</span> All rules passing
        </span>
      ) : (
        <span className="text-xs flex items-center gap-1" style={{ color: 'var(--red)' }}>
          <span>&#10007;</span> {violations} violation{violations !== 1 ? 's' : ''}
        </span>
      )}
    </div>
  );
}

/* ========================================================================
   Portfolio Card
   ======================================================================== */

function PortfolioCard({ portfolio, onClone, onDelete }: {
  portfolio: Portfolio;
  onClone: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  const [confirmDelete, setConfirmDelete] = useState(false);
  const p = portfolio;

  return (
    <Card>
      {/* Header: Name + Badges */}
      <div className="flex items-start justify-between mb-2">
        <div className="flex-1 min-w-0">
          <h3 className="text-lg font-bold truncate">{p.name}</h3>
          <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
            {p.strategies} strategies | ${p.startingBalance.toLocaleString()} balance
            {p.compoundRate > 0 && ` | ${p.compoundRate}% scaling`}
          </p>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0 ml-2">
          <StatusBadge status={p.status} />
          <DeployBadge deployStatus={p.deployStatus} />
        </div>
      </div>

      {/* Strategy names + metadata caption */}
      <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
        {p.strategyNames.slice(0, 4).join(', ')}
        {p.strategies > 4 ? '...' : ''}
        {p.avgTradesPerDay > 0 && ` | ${p.avgTradesPerDay.toFixed(1)} trades/day`}
        {p.webhookCount > 0 && ` | ${p.webhookCount} webhook(s)`}
      </p>

      {/* Action Buttons (early, always visible per Streamlit pattern) */}
      <div className="flex gap-2 mb-3">
        <Link href={`/portfolios/${p.id}`} className="flex-1">
          <button
            className="w-full py-1.5 rounded-lg text-xs font-medium border"
            style={{ borderColor: 'var(--border)', color: 'var(--text-secondary)' }}
          >
            View
          </button>
        </Link>
        <Link href={`/portfolios/${p.id}/edit`} className="flex-1">
          <button
            className="w-full py-1.5 rounded-lg text-xs font-medium border"
            style={{ borderColor: 'var(--border)', color: 'var(--text-secondary)' }}
          >
            Edit
          </button>
        </Link>
        <button
          className="flex-1 py-1.5 rounded-lg text-xs font-medium border"
          style={{ borderColor: 'var(--border)', color: 'var(--text-secondary)' }}
          onClick={() => onClone(p.id)}
        >
          Clone
        </button>
        <button
          className="flex-1 py-1.5 rounded-lg text-xs font-medium border"
          style={{ borderColor: 'var(--border)', color: 'var(--red)' }}
          onClick={() => setConfirmDelete(true)}
        >
          Delete
        </button>
      </div>

      {/* Inline delete confirmation */}
      {confirmDelete && (
        <div
          className="rounded-lg p-3 mb-3 border"
          style={{ borderColor: 'var(--red)', background: 'var(--red-muted)' }}
        >
          <p className="text-sm mb-2" style={{ color: 'var(--text-secondary)' }}>
            Delete &ldquo;{p.name}&rdquo;? This cannot be undone.
          </p>
          <div className="flex gap-2">
            <button
              className="px-3 py-1 rounded text-xs font-medium"
              style={{ background: 'var(--red)', color: 'white' }}
              onClick={() => { onDelete(p.id); setConfirmDelete(false); }}
            >
              Yes, Delete
            </button>
            <button
              className="px-3 py-1 rounded text-xs font-medium border"
              style={{ borderColor: 'var(--border)', color: 'var(--text-secondary)' }}
              onClick={() => setConfirmDelete(false)}
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Mini Equity Curve */}
      {p.equityCurve.length >= 2 && (
        <div className="rounded-lg mb-3" style={{ background: 'var(--bg-input)' }}>
          <MiniEquityCurve data={p.equityCurve} gradientId={`portEq_${p.id}`} height={80} />
        </div>
      )}

      {/* KPI Row (6 metrics) */}
      {p.trades > 0 ? (
        <div className="grid grid-cols-6 gap-2">
          {[
            { label: 'Trades', value: String(p.trades) },
            { label: 'Win Rate', value: `${p.winRate.toFixed(1)}%` },
            { label: 'PF', value: p.pf.toFixed(2) },
            {
              label: 'Total P&L',
              value: `$${p.totalPnL >= 0 ? '+' : ''}${p.totalPnL.toLocaleString()}`,
              color: p.totalPnL >= 0 ? 'var(--green)' : 'var(--red)',
            },
            {
              label: 'Balance',
              value: `$${p.currentBalance.toLocaleString()}`,
            },
            { label: 'Max DD', value: `${p.maxDD.toFixed(1)}%`, color: 'var(--red)' },
          ].map((kpi) => (
            <div key={kpi.label}>
              <p className="text-xs mb-0.5" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
              <p
                className="text-sm font-semibold"
                style={kpi.color ? { color: kpi.color } : undefined}
              >
                {kpi.value}
              </p>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>No trades yet</p>
      )}

      {/* Compliance Badge */}
      <ComplianceBadge
        compliant={p.compliant}
        violations={p.violations}
        requirementSet={p.requirementSet}
      />
    </Card>
  );
}

/* ========================================================================
   Portfolios V2 Component
   ======================================================================== */

export default function PortfoliosV2() {
  const [filter, setFilter] = useState<StatusFilter>('All');

  const filtered = filter === 'All'
    ? mockPortfolios
    : mockPortfolios.filter((p) => p.status === filter);

  const handleClone = (id: string) => {
    console.log('Clone portfolio', id);
  };

  const handleDelete = (id: string) => {
    console.log('Delete portfolio', id);
  };

  // ---------- EMPTY STATE ----------
  if (mockPortfolios.length === 0) {
    return (
      <div>
        <PageHeader title="My Portfolios" />
        <Card>
          <div className="text-center py-12">
            <p className="text-lg font-medium mb-2">No portfolios yet</p>
            <p className="text-sm mb-5" style={{ color: 'var(--text-muted)' }}>
              Create your first portfolio to combine strategies and track combined performance!
            </p>
            <Link href="/portfolios/new">
              <button
                className="px-5 py-2.5 rounded-lg text-sm font-medium"
                style={{ background: 'var(--accent)', color: 'white' }}
              >
                + New Portfolio
              </button>
            </Link>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div>
      <PageHeader
        title="My Portfolios"
        actions={
          <Link href="/portfolios/new">
            <button
              className="px-4 py-2 rounded-lg text-sm font-medium"
              style={{ background: 'var(--accent)', color: 'white' }}
            >
              + New Portfolio
            </button>
          </Link>
        }
      />

      {/* Status Filters */}
      <div className="flex gap-2 mb-5">
        {(['All', 'Live', 'Paper', 'Paused'] as StatusFilter[]).map((f) => {
          const isActive = filter === f;
          const count = f === 'All'
            ? mockPortfolios.length
            : mockPortfolios.filter((p) => p.status === f).length;

          return (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className="px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors"
              style={{
                borderColor: isActive ? 'var(--accent)' : 'var(--border)',
                background: isActive ? 'var(--accent-muted)' : 'transparent',
                color: isActive ? 'var(--accent)' : 'var(--text-muted)',
              }}
            >
              {f} ({count})
            </button>
          );
        })}
      </div>

      {/* Portfolio Grid */}
      {filtered.length === 0 ? (
        <Card>
          <p className="text-center py-8 text-sm" style={{ color: 'var(--text-muted)' }}>
            No portfolios match the selected filter.
          </p>
        </Card>
      ) : (
        <div className="grid grid-cols-2 gap-6">
          {filtered.map((portfolio) => (
            <PortfolioCard
              key={portfolio.id}
              portfolio={portfolio}
              onClone={handleClone}
              onDelete={handleDelete}
            />
          ))}
        </div>
      )}
    </div>
  );
}
