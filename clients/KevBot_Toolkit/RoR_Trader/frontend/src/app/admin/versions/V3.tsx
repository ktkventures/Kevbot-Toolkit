'use client';

import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import MetricCard from '@/components/MetricCard';

const mockPlatformStats = {
  totalUsers: 2847, newThisMonth: 342, dau: 456,
  mrr: 18420, arr: 221040, mrrGrowth: 12.3,
  totalPortfolios: 67, totalStrategies: 234, totalPacks: 45,
  dailyAlerts: 12847, alertSuccessRate: 99.2, uptime: 99.97,
};

const recentActivity = [
  { time: '2m ago', action: 'New user registered', detail: 'user@example.com' },
  { time: '5m ago', action: 'Portfolio published', detail: 'Momentum Alpha v2 by Alex K.' },
  { time: '12m ago', action: 'Subscription started', detail: 'user123 subscribed to Swing Master' },
  { time: '18m ago', action: 'Pack updated', detail: 'VWAP Extended by Sarah M.' },
  { time: '25m ago', action: 'Alert batch dispatched', detail: '847 alerts sent successfully' },
];

export default function AdminV3() {
  return (
    <div>
      <PageHeader
        title="Admin Dashboard"
        subtitle="Platform metrics at a glance"
      />

      {/* Compact KPI Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        <MetricCard label="Users" value={mockPlatformStats.totalUsers.toLocaleString()} delta={`+${mockPlatformStats.newThisMonth}`} positive />
        <MetricCard label="MRR" value={`$${mockPlatformStats.mrr.toLocaleString()}`} delta={`+${mockPlatformStats.mrrGrowth}%`} positive />
        <MetricCard label="DAU" value={mockPlatformStats.dau.toString()} />
        <MetricCard label="ARR" value={`$${mockPlatformStats.arr.toLocaleString()}`} />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        <MetricCard label="Portfolios" value={mockPlatformStats.totalPortfolios.toString()} />
        <MetricCard label="Strategies" value={mockPlatformStats.totalStrategies.toString()} />
        <MetricCard label="Packs" value={mockPlatformStats.totalPacks.toString()} />
        <MetricCard label="Daily Alerts" value={mockPlatformStats.dailyAlerts.toLocaleString()} delta={`${mockPlatformStats.alertSuccessRate}% ok`} positive />
      </div>

      {/* Activity Feed */}
      <Card>
        <h3 className="text-sm font-semibold mb-3">Recent Activity</h3>
        <div className="space-y-2">
          {recentActivity.map((a, i) => (
            <div key={i} className="flex items-center gap-3 py-1.5" style={{ borderBottom: i < recentActivity.length - 1 ? '1px solid var(--border)' : 'none' }}>
              <span className="text-xs w-12 shrink-0" style={{ color: 'var(--text-muted)' }}>{a.time}</span>
              <span className="text-sm flex-1">{a.action}</span>
              <span className="text-xs truncate max-w-[200px]" style={{ color: 'var(--text-muted)' }}>{a.detail}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
