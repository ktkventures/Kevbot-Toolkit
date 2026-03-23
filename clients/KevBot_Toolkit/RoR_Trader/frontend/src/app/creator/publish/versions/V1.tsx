'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

type PublishType = 'Portfolio' | 'Strategy' | 'Pack';

const mockUnpublished = {
  Portfolio: [
    { id: 'p1', name: 'Scalp Machine', strategies: 4, winRate: 62.3 },
    { id: 'p2', name: 'Swing Daily', strategies: 3, winRate: 58.1 },
  ],
  Strategy: [
    { id: 's1', name: 'RSI Reversal', winRate: 55.7, profitFactor: 1.8 },
    { id: 's2', name: 'Breakout Rider', winRate: 51.2, profitFactor: 2.1 },
    { id: 's3', name: 'MACD Cross Daily', winRate: 60.4, profitFactor: 1.6 },
  ],
  Pack: [
    { id: 'pk1', name: 'Volume Profile Pack', indicators: 3 },
    { id: 'pk2', name: 'Trend Strength Pack', indicators: 4 },
  ],
};

const alreadyPublished = [
  { id: '1', name: 'Momentum Alpha', type: 'Portfolio' as const, price: 30, subscribers: 156 },
  { id: '2', name: 'VWAP Scalper Pro', type: 'Portfolio' as const, price: 25, subscribers: 89 },
  { id: '3', name: 'EMA Momentum', type: 'Strategy' as const, price: 5, subscribers: 45 },
  { id: '4', name: 'RSI Zones Pack', type: 'Pack' as const, price: 8, subscribers: 52 },
];

const typeColors: Record<string, { color: string; bg: string }> = {
  Portfolio: { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  Strategy: { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  Pack: { color: 'var(--purple)', bg: 'var(--purple-muted)' },
};

export default function CreatorPublishV1() {
  const [selectedType, setSelectedType] = useState<PublishType>('Portfolio');
  const [selectedItem, setSelectedItem] = useState<string | null>(null);
  const [price, setPrice] = useState('30');
  const [published, setPublished] = useState(false);

  const items = mockUnpublished[selectedType] || [];

  function handlePublish() {
    if (selectedItem && price) {
      setPublished(true);
      setTimeout(() => setPublished(false), 2000);
    }
  }

  return (
    <div>
      <PageHeader
        title="Publish to Marketplace"
        subtitle="Share your portfolios, strategies, or packs with the community"
        backHref="/creator/dashboard"
      />

      <div className="grid grid-cols-2 gap-6">
        {/* Publish form */}
        <Card>
          <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
            New Listing
          </h3>

          {/* Type selector */}
          <div className="mb-4">
            <label className="text-sm mb-2 block" style={{ color: 'var(--text-secondary)' }}>
              What would you like to publish?
            </label>
            <div className="flex gap-2">
              {(['Portfolio', 'Strategy', 'Pack'] as PublishType[]).map((type) => (
                <button
                  key={type}
                  onClick={() => { setSelectedType(type); setSelectedItem(null); }}
                  className="flex-1 py-2.5 rounded-lg text-sm font-medium transition-colors"
                  style={{
                    background: selectedType === type ? typeColors[type]?.bg : 'var(--bg-input)',
                    color: selectedType === type ? typeColors[type]?.color : 'var(--text-muted)',
                    border: `1px solid ${selectedType === type ? typeColors[type]?.color : 'var(--border)'}`,
                  }}
                >
                  {type}
                </button>
              ))}
            </div>
          </div>

          {/* Item selector */}
          <div className="mb-4">
            <label className="text-sm mb-2 block" style={{ color: 'var(--text-secondary)' }}>
              Select Item
            </label>
            <div className="space-y-2">
              {items.map((item) => (
                <div
                  key={item.id}
                  className="flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-colors"
                  style={{
                    background: selectedItem === item.id ? typeColors[selectedType]?.bg : 'var(--bg-input)',
                    border: `1px solid ${selectedItem === item.id ? typeColors[selectedType]?.color : 'var(--border)'}`,
                  }}
                  onClick={() => setSelectedItem(item.id)}
                >
                  <div
                    className="w-4 h-4 rounded-full border-2 flex-shrink-0"
                    style={{
                      borderColor: selectedItem === item.id ? typeColors[selectedType]?.color : 'var(--text-muted)',
                      background: selectedItem === item.id ? typeColors[selectedType]?.color : 'transparent',
                    }}
                  />
                  <div className="flex-1">
                    <p className="text-sm font-medium">{item.name}</p>
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                      {'strategies' in item && `${item.strategies} strategies`}
                      {'profitFactor' in item && `PF: ${item.profitFactor}`}
                      {'indicators' in item && `${item.indicators} indicators`}
                      {'winRate' in item && ` | WR: ${item.winRate}%`}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Price */}
          <div className="mb-6">
            <label className="text-sm mb-2 block" style={{ color: 'var(--text-secondary)' }}>
              Monthly Subscription Price
            </label>
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium" style={{ color: 'var(--text-muted)' }}>$</span>
              <input
                type="number"
                className="flex-1 px-3 py-2 rounded-lg text-sm"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                value={price}
                onChange={(e) => setPrice(e.target.value)}
                min="1"
              />
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>/month</span>
            </div>
            {price && (
              <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                Platform fee: ${(parseFloat(price) * 0.15).toFixed(2)} (15%) | You receive: ${(parseFloat(price) * 0.85).toFixed(2)}
              </p>
            )}
          </div>

          {/* Publish button */}
          <button
            className="w-full py-2.5 rounded-lg text-sm font-medium transition-colors"
            style={{
              background: selectedItem ? 'var(--accent)' : 'var(--bg-input)',
              color: selectedItem ? 'white' : 'var(--text-muted)',
              cursor: selectedItem ? 'pointer' : 'not-allowed',
            }}
            disabled={!selectedItem}
            onClick={handlePublish}
          >
            {published ? 'Published!' : 'Publish to Marketplace'}
          </button>
        </Card>

        {/* Already published */}
        <Card>
          <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
            Your Published Items
          </h3>
          <div className="space-y-3">
            {alreadyPublished.map((item) => (
              <div
                key={item.id}
                className="flex items-center justify-between p-3 rounded-lg"
                style={{ background: 'var(--bg-input)' }}
              >
                <div className="flex items-center gap-3">
                  <span
                    className="text-xs px-2 py-0.5 rounded"
                    style={{ color: typeColors[item.type]?.color, background: typeColors[item.type]?.bg }}
                  >
                    {item.type}
                  </span>
                  <div>
                    <p className="text-sm font-medium">{item.name}</p>
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                      ${item.price}/mo &middot; {item.subscribers} subscribers
                    </p>
                  </div>
                </div>
                <div className="flex gap-2">
                  <button
                    className="px-3 py-1.5 rounded text-xs"
                    style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                  >
                    Edit
                  </button>
                  <button
                    className="px-3 py-1.5 rounded text-xs"
                    style={{ background: 'var(--red-muted)', color: 'var(--red)' }}
                  >
                    Unpublish
                  </button>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}
