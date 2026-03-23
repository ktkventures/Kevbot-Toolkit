'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import MetricCard from '@/components/MetricCard';
import TabBar from '@/components/TabBar';

const mockCreatorStats = {
  totalEarnings: 4280,
  thisMonth: 890,
  lastMonth: 720,
  subscribers: 342,
  newThisMonth: 47,
  published: { portfolios: 2, strategies: 5, packs: 1 },
  avgRating: 4.6,
  totalReviews: 201,
};

const mockItems = [
  { id: '1', name: 'Momentum Alpha', type: 'Portfolio' as const, subscribers: 156, monthlyRevenue: 490, rating: 4.8, trend: 'up' as const },
  { id: '2', name: 'VWAP Scalper Pro', type: 'Portfolio' as const, subscribers: 89, monthlyRevenue: 230, rating: 4.5, trend: 'up' as const },
  { id: '3', name: 'EMA Momentum', type: 'Strategy' as const, subscribers: 45, monthlyRevenue: 90, rating: 4.3, trend: 'down' as const },
  { id: '4', name: 'RSI Zones Pack', type: 'Pack' as const, subscribers: 52, monthlyRevenue: 80, rating: 4.7, trend: 'up' as const },
];

const mockMonthlyRevenue = [
  { month: 'Oct', value: 380 },
  { month: 'Nov', value: 520 },
  { month: 'Dec', value: 610 },
  { month: 'Jan', value: 780 },
  { month: 'Feb', value: 720 },
  { month: 'Mar', value: 890 },
];

const mockSubscriberGrowth = [
  { month: 'Oct', value: 180 },
  { month: 'Nov', value: 215 },
  { month: 'Dec', value: 248 },
  { month: 'Jan', value: 280 },
  { month: 'Feb', value: 310 },
  { month: 'Mar', value: 342 },
];

const mockRecentActivity = [
  { time: '2 hours ago', text: 'New subscriber to Momentum Alpha', type: 'sub' },
  { time: '5 hours ago', text: 'Review (4.9) on VWAP Scalper Pro', type: 'review' },
  { time: '1 day ago', text: '3 new subscribers to RSI Zones Pack', type: 'sub' },
  { time: '2 days ago', text: 'Payout of $722.00 processed', type: 'payout' },
  { time: '3 days ago', text: 'New subscriber to EMA Momentum', type: 'sub' },
];

const mockPayouts = [
  { date: '2026-03-01', amount: 722, method: 'PayPal', status: 'Paid' },
  { date: '2026-02-01', amount: 663, method: 'PayPal', status: 'Paid' },
  { date: '2026-01-01', amount: 581, method: 'PayPal', status: 'Paid' },
];

const typeColors: Record<string, { color: string; bg: string }> = {
  Portfolio: { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  Strategy: { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  Pack: { color: 'var(--purple)', bg: 'var(--purple-muted)' },
};

function SVGLineChart({ data, color, height = 120 }: { data: { month: string; value: number }[]; color: string; height?: number }) {
  const maxVal = Math.max(...data.map((d) => d.value));
  const minVal = Math.min(...data.map((d) => d.value));
  const range = maxVal - minVal || 1;
  const padding = 40;
  const chartWidth = 500;
  const chartHeight = height;

  const points = data.map((d, i) => ({
    x: padding + (i / (data.length - 1)) * (chartWidth - padding * 2),
    y: padding / 2 + (1 - (d.value - minVal) / range) * (chartHeight - padding),
  }));

  const linePath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');
  const areaPath = `${linePath} L ${points[points.length - 1].x} ${chartHeight} L ${points[0].x} ${chartHeight} Z`;

  return (
    <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} className="w-full" style={{ height }}>
      <defs>
        <linearGradient id={`grad-${color.replace(/[^a-z]/g, '')}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0.02" />
        </linearGradient>
      </defs>
      <path d={areaPath} fill={`url(#grad-${color.replace(/[^a-z]/g, '')})`} />
      <path d={linePath} fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
      {points.map((p, i) => (
        <g key={i}>
          <circle cx={p.x} cy={p.y} r="4" fill={color} />
          <text x={p.x} y={chartHeight - 4} textAnchor="middle" fontSize="10" fill="var(--text-muted)">
            {data[i].month}
          </text>
          <text x={p.x} y={p.y - 10} textAnchor="middle" fontSize="9" fill="var(--text-secondary)">
            {data[i].value}
          </text>
        </g>
      ))}
    </svg>
  );
}

export default function CreatorDashboardV2() {
  const [selectedItem, setSelectedItem] = useState<string | null>(null);
  const stats = mockCreatorStats;
  const totalPublished = stats.published.portfolios + stats.published.strategies + stats.published.packs;

  const topItem = [...mockItems].sort((a, b) => b.monthlyRevenue - a.monthlyRevenue)[0];

  const revenueByType = {
    Portfolio: mockItems.filter((i) => i.type === 'Portfolio').reduce((s, i) => s + i.monthlyRevenue, 0),
    Strategy: mockItems.filter((i) => i.type === 'Strategy').reduce((s, i) => s + i.monthlyRevenue, 0),
    Pack: mockItems.filter((i) => i.type === 'Pack').reduce((s, i) => s + i.monthlyRevenue, 0),
  };
  const totalMonthlyRevenue = Object.values(revenueByType).reduce((s, v) => s + v, 0);

  return (
    <div>
      <PageHeader
        title="Creator Dashboard"
        subtitle="Earnings, subscribers, and content performance at a glance"
        actions={
          <button
            className="px-4 py-2 rounded-lg text-sm font-medium"
            style={{ background: 'var(--accent)', color: 'white' }}
          >
            + Publish New
          </button>
        }
      />

      {/* KPI Cards */}
      <div className="grid grid-cols-4 gap-4 mb-8">
        <MetricCard
          label="Total Earnings"
          value={`$${stats.totalEarnings.toLocaleString()}`}
          delta={`+$${stats.thisMonth - stats.lastMonth} vs last month`}
          positive
        />
        <MetricCard
          label="Active Subscribers"
          value={stats.subscribers.toString()}
          delta={`+${stats.newThisMonth} this month`}
          positive
        />
        <MetricCard
          label="Published Items"
          value={totalPublished.toString()}
          delta={`${stats.published.portfolios}P / ${stats.published.strategies}S / ${stats.published.packs}Pk`}
        />
        <MetricCard
          label="Avg Rating"
          value={`${stats.avgRating} / 5.0`}
          delta={`${stats.totalReviews} reviews`}
        />
      </div>

      <TabBar tabs={['Overview', 'Performance', 'Payouts']}>
        {(tab) => (
          <div>
            {tab === 'Overview' && (
              <div className="grid grid-cols-3 gap-6">
                {/* Left column: Charts */}
                <div className="col-span-2 space-y-6">
                  <Card>
                    <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
                      Monthly Revenue
                    </h3>
                    <SVGLineChart data={mockMonthlyRevenue} color="var(--accent)" height={160} />
                  </Card>
                  <Card>
                    <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
                      Subscriber Growth
                    </h3>
                    <SVGLineChart data={mockSubscriberGrowth} color="var(--green)" height={140} />
                  </Card>
                </div>

                {/* Right column: Top Item + Activity */}
                <div className="space-y-6">
                  {/* Top Performing */}
                  <Card>
                    <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>
                      Top Performer
                    </h3>
                    <div className="text-center py-3">
                      <div
                        className="w-14 h-14 rounded-xl flex items-center justify-center text-2xl font-bold mx-auto mb-3"
                        style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                      >
                        #1
                      </div>
                      <p className="font-semibold text-sm">{topItem.name}</p>
                      <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                        {topItem.subscribers} subs &middot; ${topItem.monthlyRevenue}/mo
                      </p>
                      <p className="text-xs mt-1" style={{ color: 'var(--green)' }}>
                        Rating: {topItem.rating}/5.0
                      </p>
                    </div>
                  </Card>

                  {/* Revenue Breakdown */}
                  <Card>
                    <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>
                      Revenue by Type
                    </h3>
                    <div className="space-y-3">
                      {(Object.entries(revenueByType) as [string, number][]).map(([type, amount]) => {
                        const pct = totalMonthlyRevenue > 0 ? (amount / totalMonthlyRevenue) * 100 : 0;
                        return (
                          <div key={type}>
                            <div className="flex justify-between text-xs mb-1">
                              <span style={{ color: typeColors[type]?.color }}>{type}</span>
                              <span style={{ color: 'var(--text-muted)' }}>${amount}/mo ({pct.toFixed(0)}%)</span>
                            </div>
                            <div className="h-2 rounded-full" style={{ background: 'var(--bg-input)' }}>
                              <div
                                className="h-2 rounded-full transition-all"
                                style={{ width: `${pct}%`, background: typeColors[type]?.color }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </Card>

                  {/* Recent Activity */}
                  <Card>
                    <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>
                      Recent Activity
                    </h3>
                    <div className="space-y-3">
                      {mockRecentActivity.map((activity, i) => (
                        <div key={i} className="flex gap-3">
                          <div
                            className="w-2 h-2 rounded-full mt-1.5 flex-shrink-0"
                            style={{
                              background:
                                activity.type === 'sub'
                                  ? 'var(--green)'
                                  : activity.type === 'review'
                                    ? 'var(--blue)'
                                    : 'var(--accent)',
                            }}
                          />
                          <div>
                            <p className="text-xs" style={{ color: 'var(--text-primary)' }}>{activity.text}</p>
                            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{activity.time}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </Card>
                </div>
              </div>
            )}

            {tab === 'Performance' && (
              <div>
                {/* Item performance table */}
                <Card>
                  <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
                    Per-Item Performance
                  </h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr style={{ borderBottom: '1px solid var(--border)' }}>
                          {['Name', 'Type', 'Subscribers', 'Monthly Revenue', 'Rating', 'Trend'].map((h) => (
                            <th
                              key={h}
                              className="text-left py-3 px-3 text-xs font-medium"
                              style={{ color: 'var(--text-muted)' }}
                            >
                              {h}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {mockItems.map((item) => (
                          <tr
                            key={item.id}
                            className="cursor-pointer transition-colors"
                            style={{ borderBottom: '1px solid var(--border)' }}
                            onClick={() => setSelectedItem(selectedItem === item.id ? null : item.id)}
                            onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-input)')}
                            onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                          >
                            <td className="py-3 px-3 font-medium">{item.name}</td>
                            <td className="py-3 px-3">
                              <span
                                className="text-xs px-2 py-0.5 rounded"
                                style={{ color: typeColors[item.type]?.color, background: typeColors[item.type]?.bg }}
                              >
                                {item.type}
                              </span>
                            </td>
                            <td className="py-3 px-3">{item.subscribers}</td>
                            <td className="py-3 px-3">${item.monthlyRevenue}</td>
                            <td className="py-3 px-3">{item.rating}/5.0</td>
                            <td className="py-3 px-3">
                              <span
                                className="text-xs px-2 py-0.5 rounded font-medium"
                                style={{
                                  color: item.trend === 'up' ? 'var(--green)' : item.trend === 'down' ? 'var(--red)' : 'var(--text-muted)',
                                  background: item.trend === 'up' ? 'var(--green-muted)' : item.trend === 'down' ? 'var(--red-muted)' : 'var(--bg-input)',
                                }}
                              >
                                {item.trend === 'up' ? 'Trending Up' : item.trend === 'down' ? 'Declining' : 'Stable'}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>
              </div>
            )}

            {tab === 'Payouts' && (
              <div className="space-y-6">
                <div className="grid grid-cols-3 gap-4">
                  <MetricCard label="Next Payout" value="$890.00" delta="April 1, 2026" />
                  <MetricCard label="Lifetime Earned" value={`$${stats.totalEarnings.toLocaleString()}`} delta="Since Oct 2025" />
                  <MetricCard label="Platform Fees Paid" value="$756" delta="15% rate" />
                </div>
                <Card>
                  <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
                    Payout History
                  </h3>
                  <table className="w-full text-sm">
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--border)' }}>
                        {['Date', 'Amount', 'Method', 'Status'].map((h) => (
                          <th key={h} className="text-left py-3 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {mockPayouts.map((p, i) => (
                        <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                          <td className="py-3 px-3">{p.date}</td>
                          <td className="py-3 px-3 font-medium">${p.amount.toFixed(2)}</td>
                          <td className="py-3 px-3">{p.method}</td>
                          <td className="py-3 px-3">
                            <span
                              className="text-xs px-2 py-0.5 rounded"
                              style={{ color: 'var(--green)', background: 'var(--green-muted)' }}
                            >
                              {p.status}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Card>
              </div>
            )}
          </div>
        )}
      </TabBar>
    </div>
  );
}
