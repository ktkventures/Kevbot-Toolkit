'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

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
  { id: '3', name: 'EMA Momentum', type: 'Strategy' as const, subscribers: 45, monthlyRevenue: 90, rating: 4.3, trend: 'stable' as const },
  { id: '4', name: 'RSI Zones Pack', type: 'Pack' as const, subscribers: 52, monthlyRevenue: 80, rating: 4.7, trend: 'up' as const },
];

const typeColors: Record<string, { color: string; bg: string }> = {
  Portfolio: { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  Strategy: { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  Pack: { color: 'var(--purple)', bg: 'var(--purple-muted)' },
};

function MiniSparkline({ positive }: { positive: boolean }) {
  const color = positive ? 'var(--green)' : 'var(--red)';
  const points = positive
    ? 'M0,20 L8,18 L16,15 L24,16 L32,10 L40,8 L48,4'
    : 'M0,4 L8,8 L16,10 L24,14 L32,16 L40,18 L48,20';

  return (
    <svg width="48" height="24" viewBox="0 0 48 24">
      <path d={points} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export default function CreatorDashboardV3() {
  const [sortCol, setSortCol] = useState<'name' | 'subs' | 'rev' | 'rating'>('rev');
  const [sortAsc, setSortAsc] = useState(false);

  const stats = mockCreatorStats;
  const totalPublished = stats.published.portfolios + stats.published.strategies + stats.published.packs;

  const kpis = [
    { label: 'Earnings', value: `$${stats.totalEarnings.toLocaleString()}`, accent: 'var(--accent)' },
    { label: 'This Month', value: `$${stats.thisMonth}`, accent: 'var(--green)' },
    { label: 'Subscribers', value: stats.subscribers.toString(), accent: 'var(--blue)' },
    { label: 'New Subs', value: `+${stats.newThisMonth}`, accent: 'var(--green)' },
    { label: 'Published', value: totalPublished.toString(), accent: 'var(--purple)' },
    { label: 'Rating', value: stats.avgRating.toFixed(1), accent: 'var(--orange)' },
  ];

  function handleSort(col: typeof sortCol) {
    if (sortCol === col) {
      setSortAsc(!sortAsc);
    } else {
      setSortCol(col);
      setSortAsc(false);
    }
  }

  const sorted = [...mockItems].sort((a, b) => {
    let diff = 0;
    if (sortCol === 'name') diff = a.name.localeCompare(b.name);
    else if (sortCol === 'subs') diff = b.subscribers - a.subscribers;
    else if (sortCol === 'rev') diff = b.monthlyRevenue - a.monthlyRevenue;
    else diff = b.rating - a.rating;
    return sortAsc ? -diff : diff;
  });

  const sortIndicator = (col: typeof sortCol) =>
    sortCol === col ? (sortAsc ? ' ^' : ' v') : '';

  return (
    <div>
      <PageHeader title="Creator Dashboard" subtitle="Compact overview" />

      {/* KPI Strip */}
      <div
        className="flex gap-0 rounded-xl overflow-hidden border mb-6"
        style={{ borderColor: 'var(--border)' }}
      >
        {kpis.map((kpi, i) => (
          <div
            key={kpi.label}
            className="flex-1 text-center py-3 px-2"
            style={{
              background: 'var(--bg-card)',
              borderRight: i < kpis.length - 1 ? '1px solid var(--border)' : 'none',
            }}
          >
            <p className="text-xs mb-0.5" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
            <p className="text-lg font-bold" style={{ color: kpi.accent }}>{kpi.value}</p>
          </div>
        ))}
      </div>

      {/* Revenue sparkline */}
      <Card className="mb-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Revenue Trend (6mo)</p>
            <p className="text-sm font-semibold mt-1" style={{ color: 'var(--green)' }}>+23.6% growth</p>
          </div>
          <MiniSparkline positive />
        </div>
      </Card>

      {/* Compact table */}
      <Card>
        <table className="w-full text-sm">
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              <th
                className="text-left py-2 px-2 text-xs font-medium cursor-pointer select-none"
                style={{ color: 'var(--text-muted)' }}
                onClick={() => handleSort('name')}
              >
                Item{sortIndicator('name')}
              </th>
              <th
                className="text-center py-2 px-2 text-xs font-medium cursor-pointer select-none"
                style={{ color: 'var(--text-muted)' }}
                onClick={() => handleSort('subs')}
              >
                Subs{sortIndicator('subs')}
              </th>
              <th
                className="text-center py-2 px-2 text-xs font-medium cursor-pointer select-none"
                style={{ color: 'var(--text-muted)' }}
                onClick={() => handleSort('rev')}
              >
                Rev/mo{sortIndicator('rev')}
              </th>
              <th
                className="text-center py-2 px-2 text-xs font-medium cursor-pointer select-none"
                style={{ color: 'var(--text-muted)' }}
                onClick={() => handleSort('rating')}
              >
                Rating{sortIndicator('rating')}
              </th>
              <th className="text-center py-2 px-2 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>
                Trend
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((item) => (
              <tr key={item.id} style={{ borderBottom: '1px solid var(--border)' }}>
                <td className="py-2.5 px-2">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-sm">{item.name}</span>
                    <span
                      className="text-xs px-1.5 py-0.5 rounded"
                      style={{ color: typeColors[item.type]?.color, background: typeColors[item.type]?.bg }}
                    >
                      {item.type[0]}
                    </span>
                  </div>
                </td>
                <td className="py-2.5 px-2 text-center text-sm">{item.subscribers}</td>
                <td className="py-2.5 px-2 text-center text-sm font-medium">${item.monthlyRevenue}</td>
                <td className="py-2.5 px-2 text-center text-sm">{item.rating}</td>
                <td className="py-2.5 px-2 text-center">
                  <MiniSparkline positive={item.trend === 'up'} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  );
}
