'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import MetricCard from '@/components/MetricCard';

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

const typeColors: Record<string, { color: string; bg: string }> = {
  Portfolio: { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  Strategy: { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  Pack: { color: 'var(--purple)', bg: 'var(--purple-muted)' },
};

const trendIcons: Record<string, string> = {
  up: '+',
  down: '-',
  stable: '=',
};

export default function CreatorDashboardV1() {
  const [sortBy, setSortBy] = useState<'subscribers' | 'revenue' | 'rating'>('subscribers');

  const stats = mockCreatorStats;
  const totalPublished = stats.published.portfolios + stats.published.strategies + stats.published.packs;

  const sorted = [...mockItems].sort((a, b) => {
    if (sortBy === 'subscribers') return b.subscribers - a.subscribers;
    if (sortBy === 'revenue') return b.monthlyRevenue - a.monthlyRevenue;
    return b.rating - a.rating;
  });

  return (
    <div>
      <PageHeader
        title="Creator Dashboard"
        subtitle="Your command center for earnings, subscribers, and content performance"
      />

      {/* KPI Cards */}
      <div className="grid grid-cols-4 gap-4 mb-8">
        <MetricCard
          label="Total Earnings"
          value={`$${stats.totalEarnings.toLocaleString()}`}
          delta={`+$${stats.thisMonth - stats.lastMonth} vs last month`}
          positive={stats.thisMonth > stats.lastMonth}
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
          value={stats.avgRating.toFixed(1)}
          delta={`${stats.totalReviews} reviews`}
        />
      </div>

      {/* Sort controls */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Published Items</h2>
        <div className="flex gap-2">
          {(['subscribers', 'revenue', 'rating'] as const).map((key) => (
            <button
              key={key}
              onClick={() => setSortBy(key)}
              className="px-3 py-1.5 rounded-lg text-xs capitalize transition-colors"
              style={{
                background: sortBy === key ? 'var(--accent-muted)' : 'var(--bg-input)',
                color: sortBy === key ? 'var(--accent)' : 'var(--text-muted)',
                border: `1px solid ${sortBy === key ? 'var(--accent)' : 'var(--border)'}`,
              }}
            >
              {key}
            </button>
          ))}
        </div>
      </div>

      {/* Published items list */}
      <div className="space-y-3">
        {sorted.map((item) => (
          <Card key={item.id}>
            <div className="flex items-center gap-4">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-semibold text-sm">{item.name}</span>
                  <span
                    className="text-xs px-2 py-0.5 rounded"
                    style={{
                      color: typeColors[item.type]?.color,
                      background: typeColors[item.type]?.bg,
                    }}
                  >
                    {item.type}
                  </span>
                </div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  {item.subscribers} subscribers &middot; ${item.monthlyRevenue}/mo
                </p>
              </div>

              <div className="flex items-center gap-6 flex-shrink-0">
                <div className="text-center">
                  <p className="text-sm font-semibold">{item.subscribers}</p>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>subs</p>
                </div>
                <div className="text-center">
                  <p className="text-sm font-semibold">${item.monthlyRevenue}</p>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>monthly</p>
                </div>
                <div className="text-center">
                  <p className="text-sm font-semibold">{item.rating}</p>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>rating</p>
                </div>
                <div
                  className="w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold"
                  style={{
                    color: item.trend === 'up' ? 'var(--green)' : item.trend === 'down' ? 'var(--red)' : 'var(--text-muted)',
                    background: item.trend === 'up' ? 'var(--green-muted)' : item.trend === 'down' ? 'var(--red-muted)' : 'var(--bg-input)',
                  }}
                >
                  {trendIcons[item.trend]}
                </div>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
