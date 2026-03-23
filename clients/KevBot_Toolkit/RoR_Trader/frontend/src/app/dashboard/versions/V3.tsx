'use client';

import Link from 'next/link';
import Card from '@/components/Card';
import MetricCard from '@/components/MetricCard';

/* ========================================================================= */
/* MOCK DATA                                                                  */
/* ========================================================================= */

const mockKPIs = {
  totalStrategies: 12,
  alertsToday: 7,
  portfolioPnL: '+$247',
  portfolioPnLDelta: '+1.2%',
  activePositions: 2,
};

const mockBestStrategy = {
  id: '2',
  name: 'SPY LONG - Mass #1',
  winRate: 62.5,
  pf: 3.12,
  dailyR: 2.41,
};

const mockPortfolioHealth = {
  id: '1',
  name: 'My Portfolio',
  balance: '$84,230',
  pnl: '+$4,230',
  drawdown: '-2.1%',
};

const mockActivity = [
  { time: '10:32', text: 'NVDA LONG entry signal fired', color: 'var(--green)' },
  { time: '10:15', text: 'NVDA LONG exit -- +1.2R', color: 'var(--red)' },
  { time: '09:45', text: 'SPY LONG entry via L0', color: 'var(--green)' },
  { time: '09:30', text: 'Market open -- RTH session', color: 'var(--text-muted)' },
  { time: 'Yest.', text: 'Saved TSLA LONG - Mass #5', color: 'var(--text-muted)' },
];

/* ========================================================================= */
/* COMPONENT                                                                  */
/* ========================================================================= */

export default function DashboardV3() {
  return (
    <div>
      {/* Header */}
      <h1 className="text-2xl font-bold mb-5">Dashboard</h1>

      {/* Top row: 4 KPIs */}
      <div className="grid grid-cols-4 gap-4 mb-5">
        <MetricCard label="Strategies" value={String(mockKPIs.totalStrategies)} />
        <MetricCard label="Alerts Today" value={String(mockKPIs.alertsToday)} />
        <MetricCard
          label="Portfolio P&L"
          value={mockKPIs.portfolioPnL}
          delta={mockKPIs.portfolioPnLDelta}
          positive
        />
        <MetricCard label="Active Positions" value={String(mockKPIs.activePositions)} />
      </div>

      {/* Center: Two side-by-side cards */}
      <div className="grid grid-cols-2 gap-4 mb-5">
        {/* Best Strategy */}
        <Card>
          <div className="flex items-start justify-between mb-3">
            <h3 className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Best Strategy</h3>
            <Link href={`/strategies/${mockBestStrategy.id}`}>
              <span className="text-xs" style={{ color: 'var(--accent)', cursor: 'pointer' }}>View</span>
            </Link>
          </div>
          <h2 className="text-base font-bold mb-3">{mockBestStrategy.name}</h2>
          <div className="grid grid-cols-3 gap-3">
            <div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>WR</p>
              <p className="text-sm font-semibold">{mockBestStrategy.winRate}%</p>
            </div>
            <div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>PF</p>
              <p className="text-sm font-semibold">{mockBestStrategy.pf.toFixed(2)}</p>
            </div>
            <div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Daily R</p>
              <p className="text-sm font-semibold" style={{ color: 'var(--green)' }}>
                +{mockBestStrategy.dailyR.toFixed(2)}
              </p>
            </div>
          </div>
        </Card>

        {/* Portfolio Health */}
        <Card>
          <div className="flex items-start justify-between mb-3">
            <h3 className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Portfolio Health</h3>
            <Link href={`/portfolios/${mockPortfolioHealth.id}`}>
              <span className="text-xs" style={{ color: 'var(--accent)', cursor: 'pointer' }}>View</span>
            </Link>
          </div>
          <h2 className="text-base font-bold mb-3">{mockPortfolioHealth.name}</h2>
          <div className="grid grid-cols-3 gap-3">
            <div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Balance</p>
              <p className="text-sm font-semibold">{mockPortfolioHealth.balance}</p>
            </div>
            <div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>P&L</p>
              <p className="text-sm font-semibold" style={{ color: 'var(--green)' }}>
                {mockPortfolioHealth.pnl}
              </p>
            </div>
            <div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Drawdown</p>
              <p className="text-sm font-semibold" style={{ color: 'var(--red)' }}>
                {mockPortfolioHealth.drawdown}
              </p>
            </div>
          </div>
        </Card>
      </div>

      {/* Bottom: Compact activity feed */}
      <Card>
        <h3 className="text-xs font-medium mb-3" style={{ color: 'var(--text-muted)' }}>Recent Activity</h3>
        <div className="space-y-0">
          {mockActivity.map((item, idx) => (
            <div
              key={idx}
              className="flex items-center gap-3 py-2 border-b last:border-0"
              style={{ borderColor: 'var(--border)' }}
            >
              <span
                className="text-xs w-12 shrink-0 font-medium"
                style={{ color: 'var(--text-muted)' }}
              >
                {item.time}
              </span>
              <span
                className="w-1.5 h-1.5 rounded-full shrink-0"
                style={{ background: item.color }}
              />
              <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                {item.text}
              </span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
