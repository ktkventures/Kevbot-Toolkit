'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import MetricCard from '@/components/MetricCard';
import ChartPlaceholder from '@/components/ChartPlaceholder';
import TabBar from '@/components/TabBar';

const mockPlatformStats = {
  totalUsers: 2847, newThisMonth: 342, dau: 456,
  mrr: 18420, arr: 221040, mrrGrowth: 12.3,
  totalPortfolios: 67, totalStrategies: 234, totalPacks: 45,
  dailyAlerts: 12847, alertSuccessRate: 99.2, uptime: 99.97,
  webhookDeliveryRate: 99.8, errorRate: 0.03,
};

const topCreators = [
  { name: 'Alex K.', role: 'Portfolio Builder', subscribers: 128, revenue: 3840, growth: 15.2 },
  { name: 'Sarah M.', role: 'Pack Creator', subscribers: 0, revenue: 1800, growth: 8.4 },
  { name: 'James T.', role: 'Portfolio Builder', subscribers: 95, revenue: 2850, growth: 22.1 },
  { name: 'Maria L.', role: 'Strategy Builder', subscribers: 0, revenue: 960, growth: 5.7 },
  { name: 'David R.', role: 'Portfolio Builder', subscribers: 67, revenue: 2010, growth: -3.2 },
];

const recentActivity = [
  { time: '2m ago', action: 'New user registered', detail: 'user@example.com' },
  { time: '5m ago', action: 'Portfolio published', detail: 'Momentum Alpha v2 by Alex K.' },
  { time: '12m ago', action: 'Subscription started', detail: 'user123 → Swing Master portfolio' },
  { time: '18m ago', action: 'Pack updated', detail: 'VWAP Extended by Sarah M.' },
  { time: '25m ago', action: 'Alert triggered', detail: '847 alerts dispatched (batch)' },
  { time: '31m ago', action: 'User upgraded', detail: 'john@email.com → Portfolio Builder' },
  { time: '45m ago', action: 'Strategy backtested', detail: 'EMA Trend Follow — 2.4 PF' },
  { time: '1h ago', action: 'Payout processed', detail: '$1,240 to 3 creators' },
];

export default function AdminV2() {
  const [moderationCount] = useState(4);

  return (
    <div>
      <PageHeader
        title="Admin Dashboard"
        subtitle="Platform metrics, revenue, and management"
        actions={
          <span
            className="px-3 py-1.5 rounded-lg text-xs font-medium"
            style={{ background: 'var(--orange-muted)', color: 'var(--orange)' }}
          >
            {moderationCount} items pending review
          </span>
        }
      />

      <TabBar tabs={['Overview', 'Revenue', 'Content', 'System']}>
        {(tab) => {
          if (tab === 'Overview') return (
            <div className="space-y-6">
              {/* KPIs */}
              <div>
                <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-muted)' }}>USERS</h3>
                <div className="grid grid-cols-3 gap-4 mb-6">
                  <MetricCard label="Total Users" value={mockPlatformStats.totalUsers.toLocaleString()} delta={`+${mockPlatformStats.newThisMonth} this month`} positive />
                  <MetricCard label="Daily Active" value={mockPlatformStats.dau.toString()} delta="16% of total" />
                  <MetricCard label="New This Month" value={mockPlatformStats.newThisMonth.toString()} delta="+18% vs last month" positive />
                </div>

                <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-muted)' }}>REVENUE</h3>
                <div className="grid grid-cols-3 gap-4 mb-6">
                  <MetricCard label="MRR" value={`$${mockPlatformStats.mrr.toLocaleString()}`} delta={`+${mockPlatformStats.mrrGrowth}%`} positive />
                  <MetricCard label="ARR" value={`$${mockPlatformStats.arr.toLocaleString()}`} />
                  <MetricCard label="MRR Growth" value={`${mockPlatformStats.mrrGrowth}%`} delta="vs last month" positive />
                </div>

                <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-muted)' }}>CONTENT</h3>
                <div className="grid grid-cols-3 gap-4">
                  <MetricCard label="Portfolios" value={mockPlatformStats.totalPortfolios.toString()} />
                  <MetricCard label="Strategies" value={mockPlatformStats.totalStrategies.toString()} />
                  <MetricCard label="Packs" value={mockPlatformStats.totalPacks.toString()} />
                </div>
              </div>

              {/* Charts */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                <Card>
                  <h3 className="text-sm font-semibold mb-3">MRR Over Time</h3>
                  <ChartPlaceholder label="MRR trend chart" height={220} />
                </Card>
                <Card>
                  <h3 className="text-sm font-semibold mb-3">User Growth</h3>
                  <ChartPlaceholder label="User growth chart" height={220} />
                </Card>
              </div>

              {/* Top Creators + Activity */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                <Card>
                  <h3 className="text-sm font-semibold mb-3">Top Creators</h3>
                  <div className="space-y-3">
                    {topCreators.map((c, i) => (
                      <div key={c.name} className="flex items-center gap-3">
                        <span className="text-xs font-mono w-5" style={{ color: 'var(--text-muted)' }}>#{i + 1}</span>
                        <div
                          className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-semibold"
                          style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                        >
                          {c.name.charAt(0)}
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium truncate">{c.name}</p>
                          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{c.role}</p>
                        </div>
                        <div className="text-right">
                          <p className="text-sm font-semibold">${c.revenue.toLocaleString()}</p>
                          <p className="text-xs" style={{ color: c.growth >= 0 ? 'var(--green)' : 'var(--red)' }}>
                            {c.growth >= 0 ? '+' : ''}{c.growth}%
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>

                <Card>
                  <h3 className="text-sm font-semibold mb-3">Recent Activity</h3>
                  <div className="space-y-3">
                    {recentActivity.map((a, i) => (
                      <div key={i} className="flex items-start gap-3">
                        <span className="text-xs w-12 shrink-0 pt-0.5" style={{ color: 'var(--text-muted)' }}>{a.time}</span>
                        <div className="min-w-0">
                          <p className="text-sm">{a.action}</p>
                          <p className="text-xs truncate" style={{ color: 'var(--text-muted)' }}>{a.detail}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>
              </div>
            </div>
          );

          if (tab === 'Revenue') return (
            <div className="space-y-5">
              <div className="grid grid-cols-4 gap-4">
                <MetricCard label="MRR" value={`$${mockPlatformStats.mrr.toLocaleString()}`} delta={`+${mockPlatformStats.mrrGrowth}%`} positive />
                <MetricCard label="ARR" value={`$${mockPlatformStats.arr.toLocaleString()}`} />
                <MetricCard label="Platform Fee Revenue" value="$2,763" delta="+9.8%" positive />
                <MetricCard label="Creator Payouts" value="$15,657" delta="85% of gross" />
              </div>
              <Card>
                <h3 className="text-sm font-semibold mb-3">Revenue Breakdown</h3>
                <ChartPlaceholder label="Revenue waterfall / breakdown chart" height={300} />
              </Card>
            </div>
          );

          if (tab === 'Content') return (
            <div className="space-y-5">
              <div className="grid grid-cols-3 gap-4">
                <MetricCard label="Portfolios" value={mockPlatformStats.totalPortfolios.toString()} delta="+5 this month" positive />
                <MetricCard label="Strategies" value={mockPlatformStats.totalStrategies.toString()} delta="+22 this month" positive />
                <MetricCard label="Packs" value={mockPlatformStats.totalPacks.toString()} delta="+3 this month" positive />
              </div>
              <Card>
                <h3 className="text-sm font-semibold mb-3">Content Moderation Queue</h3>
                <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                  {moderationCount} items pending review. Visit the Curation page for details.
                </p>
              </Card>
            </div>
          );

          if (tab === 'System') return (
            <div className="space-y-5">
              <div className="grid grid-cols-4 gap-4">
                <MetricCard label="Uptime" value={`${mockPlatformStats.uptime}%`} positive />
                <MetricCard label="Webhook Delivery" value={`${mockPlatformStats.webhookDeliveryRate}%`} positive />
                <MetricCard label="Error Rate" value={`${mockPlatformStats.errorRate}%`} />
                <MetricCard label="Daily Alerts" value={mockPlatformStats.dailyAlerts.toLocaleString()} delta={`${mockPlatformStats.alertSuccessRate}% success`} positive />
              </div>
              <Card>
                <h3 className="text-sm font-semibold mb-3">System Health</h3>
                <ChartPlaceholder label="Uptime / error rate chart" height={250} />
              </Card>
            </div>
          );

          return null;
        }}
      </TabBar>
    </div>
  );
}
