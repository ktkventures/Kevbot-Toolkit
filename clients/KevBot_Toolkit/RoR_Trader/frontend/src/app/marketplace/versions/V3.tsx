'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

interface ListItem {
  id: string;
  name: string;
  creator: string;
  price: number;
  rating: number;
  subscribers: number;
  verification: 'backtest' | 'forward' | 'live';
  winRate: number;
  pf: number;
  dailyR: number;
  type: 'portfolio' | 'strategy' | 'pack';
}

const allItems: ListItem[] = [
  { id: 'p1', name: 'Momentum Alpha', creator: 'TradeMaster', price: 49, rating: 4.8, subscribers: 342, verification: 'live', winRate: 58.2, pf: 2.31, dailyR: 1.85, type: 'portfolio' },
  { id: 'p2', name: 'VWAP Scalper Pro', creator: 'QuantEdge', price: 29, rating: 4.5, subscribers: 521, verification: 'forward', winRate: 54.0, pf: 1.92, dailyR: 1.45, type: 'portfolio' },
  { id: 'p3', name: 'Steady Eddie', creator: 'RoR Team', price: 0, rating: 4.2, subscribers: 1203, verification: 'live', winRate: 52.1, pf: 1.65, dailyR: 0.95, type: 'portfolio' },
  { id: 'p4', name: 'Tech Momentum', creator: 'AlgoTrader99', price: 39, rating: 4.6, subscribers: 189, verification: 'forward', winRate: 56.3, pf: 2.08, dailyR: 1.72, type: 'portfolio' },
  { id: 'p5', name: 'Conservative Growth', creator: 'RoR Team', price: 0, rating: 4.0, subscribers: 876, verification: 'live', winRate: 50.5, pf: 1.42, dailyR: 0.68, type: 'portfolio' },
  { id: 'p6', name: 'Swing Master', creator: 'SwingKing', price: 79, rating: 4.9, subscribers: 156, verification: 'live', winRate: 61.5, pf: 3.12, dailyR: 2.41, type: 'portfolio' },
  { id: 's1', name: 'EMA Cross Scalper', creator: 'TradeMaster', price: 15, rating: 4.6, subscribers: 189, verification: 'live', winRate: 55.8, pf: 2.1, dailyR: 1.6, type: 'strategy' },
  { id: 's2', name: 'VWAP Bounce', creator: 'QuantEdge', price: 10, rating: 4.3, subscribers: 312, verification: 'forward', winRate: 52.4, pf: 1.78, dailyR: 1.2, type: 'strategy' },
  { id: 's3', name: 'MACD Trend Follow', creator: 'RoR Team', price: 0, rating: 4.1, subscribers: 854, verification: 'live', winRate: 49.2, pf: 1.55, dailyR: 0.85, type: 'strategy' },
  { id: 'pk1', name: 'Advanced EMA Pack', creator: 'TradeMaster', price: 8, rating: 4.5, subscribers: 402, verification: 'backtest', winRate: 0, pf: 0, dailyR: 0, type: 'pack' },
  { id: 'pk2', name: 'Volume Profile Suite', creator: 'QuantEdge', price: 12, rating: 4.4, subscribers: 178, verification: 'backtest', winRate: 0, pf: 0, dailyR: 0, type: 'pack' },
];

type SortKey = 'name' | 'rating' | 'subscribers' | 'price' | 'winRate' | 'pf';

const verificationDot: Record<string, string> = {
  backtest: 'var(--text-muted)',
  forward: 'var(--blue)',
  live: 'var(--orange)',
};

const verificationLabel: Record<string, string> = {
  backtest: 'BT',
  forward: 'FT',
  live: 'LV',
};

export default function MarketplaceV3() {
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState<SortKey>('rating');
  const [typeFilter, setTypeFilter] = useState<'all' | 'portfolio' | 'strategy' | 'pack'>('all');

  const filtered = allItems
    .filter((item) => {
      if (typeFilter !== 'all' && item.type !== typeFilter) return false;
      if (search && !item.name.toLowerCase().includes(search.toLowerCase()) && !item.creator.toLowerCase().includes(search.toLowerCase())) return false;
      return true;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'name': return a.name.localeCompare(b.name);
        case 'rating': return b.rating - a.rating;
        case 'subscribers': return b.subscribers - a.subscribers;
        case 'price': return a.price - b.price;
        case 'winRate': return b.winRate - a.winRate;
        case 'pf': return b.pf - a.pf;
        default: return 0;
      }
    });

  return (
    <div>
      <PageHeader title="Marketplace" subtitle="Quick scan view" />

      {/* Controls */}
      <div className="flex gap-3 mb-4">
        <input
          className="flex-1 px-4 py-2 rounded-lg text-sm"
          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
          placeholder="Quick search..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <select
          className="px-3 py-2 rounded-lg text-sm"
          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value as typeof typeFilter)}
        >
          <option value="all">All Types</option>
          <option value="portfolio">Portfolios</option>
          <option value="strategy">Strategies</option>
          <option value="pack">Packs</option>
        </select>
        <select
          className="px-3 py-2 rounded-lg text-sm"
          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as SortKey)}
        >
          <option value="rating">Sort: Rating</option>
          <option value="subscribers">Sort: Subscribers</option>
          <option value="price">Sort: Price</option>
          <option value="winRate">Sort: Win Rate</option>
          <option value="pf">Sort: Profit Factor</option>
          <option value="name">Sort: Name</option>
        </select>
      </div>

      <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>{filtered.length} results</p>

      {/* List view */}
      <Card>
        {/* Header row */}
        <div className="flex items-center gap-3 pb-3 border-b text-xs font-medium" style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}>
          <span className="w-6">V</span>
          <span className="flex-1">Name</span>
          <span className="w-16 text-center">Type</span>
          <span className="w-16 text-right">WR</span>
          <span className="w-16 text-right">PF</span>
          <span className="w-16 text-right">Daily R</span>
          <span className="w-16 text-right">Rating</span>
          <span className="w-20 text-right">Subs</span>
          <span className="w-20 text-right">Price</span>
          <span className="w-24"></span>
        </div>

        {/* Rows */}
        <div className="divide-y" style={{ borderColor: 'var(--border)' }}>
          {filtered.map((item) => (
            <div
              key={item.id}
              className="flex items-center gap-3 py-3 hover:bg-[var(--bg-input)] transition-colors cursor-pointer rounded"
            >
              {/* Verification dot */}
              <span className="w-6 flex justify-center">
                <span
                  className="w-2.5 h-2.5 rounded-full inline-block"
                  style={{ background: verificationDot[item.verification] }}
                  title={item.verification}
                />
              </span>

              {/* Name + creator */}
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate">{item.name}</p>
                <p className="text-xs truncate" style={{ color: 'var(--text-muted)' }}>{item.creator}</p>
              </div>

              {/* Type */}
              <span className="w-16 text-center text-xs px-1.5 py-0.5 rounded" style={{ background: 'var(--bg-input)', color: 'var(--text-secondary)' }}>{item.type}</span>

              {/* Metrics */}
              <span className="w-16 text-right text-sm" style={{ color: item.winRate > 0 ? (item.winRate >= 55 ? 'var(--green)' : 'var(--text-primary)') : 'var(--text-muted)' }}>
                {item.winRate > 0 ? `${item.winRate}%` : '-'}
              </span>
              <span className="w-16 text-right text-sm" style={{ color: item.pf > 0 ? (item.pf >= 2 ? 'var(--green)' : 'var(--text-primary)') : 'var(--text-muted)' }}>
                {item.pf > 0 ? item.pf.toFixed(2) : '-'}
              </span>
              <span className="w-16 text-right text-sm" style={{ color: item.dailyR > 0 ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                {item.dailyR > 0 ? item.dailyR.toFixed(2) : '-'}
              </span>

              {/* Rating */}
              <span className="w-16 text-right text-sm">
                <span style={{ color: 'var(--orange)' }}>&#9733;</span> {item.rating}
              </span>

              {/* Subscribers */}
              <span className="w-20 text-right text-sm" style={{ color: 'var(--text-secondary)' }}>{item.subscribers.toLocaleString()}</span>

              {/* Price */}
              <span className="w-20 text-right text-sm font-medium" style={{ color: item.price === 0 ? 'var(--green)' : 'var(--text-primary)' }}>
                {item.price === 0 ? 'FREE' : `$${item.price}/mo`}
              </span>

              {/* Action */}
              <span className="w-24">
                <button className="w-full px-3 py-1.5 rounded-lg text-xs font-medium" style={{ background: 'var(--accent)', color: 'white' }}>Subscribe</button>
              </span>
            </div>
          ))}
        </div>

        {filtered.length === 0 && (
          <div className="text-center py-8">
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No results found.</p>
          </div>
        )}
      </Card>

      {/* Legend */}
      <div className="flex items-center gap-4 mt-4">
        {Object.entries(verificationDot).map(([key, color]) => (
          <div key={key} className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: color }} />
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{verificationLabel[key]} = {key}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
