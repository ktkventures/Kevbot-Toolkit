'use client';

import PageHeader from '@/components/PageHeader';
import MetricCard from '@/components/MetricCard';

const mockPlatformStats = {
  totalUsers: 2847, newThisMonth: 342, dau: 456,
  mrr: 18420, arr: 221040, mrrGrowth: 12.3,
  totalPortfolios: 67, totalStrategies: 234, totalPacks: 45,
  dailyAlerts: 12847, alertSuccessRate: 99.2, uptime: 99.97,
};

export default function AdminV1() {
  return (
    <div>
      <PageHeader
        title="Admin Dashboard"
        subtitle="Platform overview and management"
      />

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard label="Total Users" value={mockPlatformStats.totalUsers.toLocaleString()} delta={`+${mockPlatformStats.newThisMonth} this month`} positive />
        <MetricCard label="Monthly Revenue" value={`$${mockPlatformStats.mrr.toLocaleString()}`} delta={`+${mockPlatformStats.mrrGrowth}% growth`} positive />
        <MetricCard label="Active Portfolios" value={mockPlatformStats.totalPortfolios.toString()} />
        <MetricCard label="Daily Alert Volume" value={mockPlatformStats.dailyAlerts.toLocaleString()} delta={`${mockPlatformStats.alertSuccessRate}% success`} positive />
      </div>
    </div>
  );
}
