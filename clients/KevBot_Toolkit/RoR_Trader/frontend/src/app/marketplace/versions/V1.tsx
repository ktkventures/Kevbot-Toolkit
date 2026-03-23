'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';

interface MarketplaceItem {
  id: string;
  name: string;
  creator: string;
  price: number;
  rating: number;
  reviews: number;
  subscribers: number;
  type: 'portfolio' | 'strategy' | 'pack';
  category: string;
  free: boolean;
}

const mockPortfolios: MarketplaceItem[] = [
  { id: 'p1', name: 'Momentum Alpha', creator: 'TradeMaster', price: 49, rating: 4.8, reviews: 127, subscribers: 342, type: 'portfolio', category: 'Momentum', free: false },
  { id: 'p2', name: 'VWAP Scalper Pro', creator: 'QuantEdge', price: 29, rating: 4.5, reviews: 83, subscribers: 521, type: 'portfolio', category: 'Scalping', free: false },
  { id: 'p3', name: 'Steady Eddie', creator: 'RoR Team', price: 0, rating: 4.2, reviews: 45, subscribers: 1203, type: 'portfolio', category: 'Conservative', free: true },
  { id: 'p4', name: 'Tech Momentum', creator: 'AlgoTrader99', price: 39, rating: 4.6, reviews: 62, subscribers: 189, type: 'portfolio', category: 'Momentum', free: false },
  { id: 'p5', name: 'Conservative Growth', creator: 'RoR Team', price: 0, rating: 4.0, reviews: 31, subscribers: 876, type: 'portfolio', category: 'Conservative', free: true },
  { id: 'p6', name: 'Swing Master', creator: 'SwingKing', price: 79, rating: 4.9, reviews: 201, subscribers: 156, type: 'portfolio', category: 'Swing', free: false },
  { id: 'p7', name: 'Mean Reversion Suite', creator: 'StatArb', price: 59, rating: 4.4, reviews: 74, subscribers: 98, type: 'portfolio', category: 'Mean Reversion', free: false },
  { id: 'p8', name: 'Breakout Hunter', creator: 'BreakoutPro', price: 35, rating: 4.3, reviews: 56, subscribers: 267, type: 'portfolio', category: 'Breakout', free: false },
];

const mockStrategies: MarketplaceItem[] = [
  { id: 's1', name: 'EMA Cross Scalper', creator: 'TradeMaster', price: 15, rating: 4.6, reviews: 42, subscribers: 189, type: 'strategy', category: 'Scalping', free: false },
  { id: 's2', name: 'VWAP Bounce', creator: 'QuantEdge', price: 10, rating: 4.3, reviews: 28, subscribers: 312, type: 'strategy', category: 'Intraday', free: false },
  { id: 's3', name: 'MACD Trend Follow', creator: 'RoR Team', price: 0, rating: 4.1, reviews: 67, subscribers: 854, type: 'strategy', category: 'Trend', free: true },
  { id: 's4', name: 'ATR Breakout', creator: 'BreakoutPro', price: 20, rating: 4.7, reviews: 35, subscribers: 145, type: 'strategy', category: 'Breakout', free: false },
];

const mockPacks: MarketplaceItem[] = [
  { id: 'pk1', name: 'Advanced EMA Pack', creator: 'TradeMaster', price: 8, rating: 4.5, reviews: 31, subscribers: 402, type: 'pack', category: 'Moving Averages', free: false },
  { id: 'pk2', name: 'Volume Profile Suite', creator: 'QuantEdge', price: 12, rating: 4.4, reviews: 19, subscribers: 178, type: 'pack', category: 'Volume', free: false },
  { id: 'pk3', name: 'Momentum Essentials', creator: 'RoR Team', price: 0, rating: 4.2, reviews: 56, subscribers: 1024, type: 'pack', category: 'Momentum', free: true },
];

const categories = ['All', 'Momentum', 'Scalping', 'Swing', 'Conservative', 'Breakout', 'Mean Reversion', 'Trend', 'Intraday', 'Moving Averages', 'Volume'];

function StarRating({ rating }: { rating: number }) {
  return (
    <div className="flex items-center gap-1">
      {[1, 2, 3, 4, 5].map((star) => (
        <span
          key={star}
          className="text-xs"
          style={{ color: star <= Math.round(rating) ? 'var(--orange)' : 'var(--text-muted)' }}
        >
          &#9733;
        </span>
      ))}
      <span className="text-xs ml-1" style={{ color: 'var(--text-muted)' }}>{rating.toFixed(1)}</span>
    </div>
  );
}

export default function MarketplaceV1() {
  const [search, setSearch] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('All');

  function getItems(tab: string): MarketplaceItem[] {
    let items: MarketplaceItem[] = [];
    if (tab === 'Portfolios') items = mockPortfolios;
    else if (tab === 'Strategies') items = mockStrategies;
    else items = mockPacks;

    if (search) {
      items = items.filter((item) =>
        item.name.toLowerCase().includes(search.toLowerCase()) ||
        item.creator.toLowerCase().includes(search.toLowerCase())
      );
    }
    if (selectedCategory !== 'All') {
      items = items.filter((item) => item.category === selectedCategory);
    }
    return items;
  }

  function renderCard(item: MarketplaceItem) {
    return (
      <Card key={item.id} className="hover:border-[var(--accent)] transition-colors cursor-pointer">
        <div className="flex flex-col gap-3">
          <div className="flex items-start justify-between">
            <div>
              <h3 className="font-semibold text-sm">{item.name}</h3>
              <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>by {item.creator}</p>
            </div>
            {item.free ? (
              <span className="text-xs font-medium px-2 py-0.5 rounded" style={{ background: 'var(--green-muted)', color: 'var(--green)' }}>FREE</span>
            ) : (
              <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>${item.price}/mo</span>
            )}
          </div>

          <div className="flex items-center justify-between">
            <StarRating rating={item.rating} />
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{item.reviews} reviews</span>
          </div>

          <div className="flex items-center justify-between pt-2 border-t" style={{ borderColor: 'var(--border)' }}>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{item.subscribers.toLocaleString()} subscribers</span>
            <span className="text-xs px-2 py-0.5 rounded" style={{ background: 'var(--bg-input)', color: 'var(--text-secondary)' }}>{item.category}</span>
          </div>
        </div>
      </Card>
    );
  }

  return (
    <div>
      <PageHeader
        title="Marketplace"
        subtitle="Browse portfolios, strategies, and confluence packs"
      />

      {/* Search bar */}
      <div className="mb-6">
        <input
          className="w-full px-4 py-3 rounded-lg text-sm"
          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
          placeholder="Search portfolios, strategies, packs..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      {/* Category pills */}
      <div className="flex flex-wrap gap-2 mb-6">
        {categories.map((cat) => (
          <button
            key={cat}
            className="px-3 py-1.5 rounded-full text-xs transition-colors"
            style={{
              background: selectedCategory === cat ? 'var(--accent)' : 'var(--bg-input)',
              color: selectedCategory === cat ? 'white' : 'var(--text-secondary)',
              border: selectedCategory === cat ? 'none' : '1px solid var(--border)',
            }}
            onClick={() => setSelectedCategory(cat)}
          >
            {cat}
          </button>
        ))}
      </div>

      {/* Tabs */}
      <TabBar tabs={['Portfolios', 'Strategies', 'Packs']}>
        {(tab) => {
          const items = getItems(tab);
          return (
            <div>
              <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
                {items.length} {tab.toLowerCase()} found
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {items.map(renderCard)}
              </div>
              {items.length === 0 && (
                <div className="text-center py-12">
                  <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No {tab.toLowerCase()} match your filters.</p>
                </div>
              )}
            </div>
          );
        }}
      </TabBar>
    </div>
  );
}
