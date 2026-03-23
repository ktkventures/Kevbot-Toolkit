'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

type PublishType = 'Portfolio' | 'Strategy' | 'Pack';

const mockUnpublished: Record<PublishType, { id: string; name: string }[]> = {
  Portfolio: [
    { id: 'p1', name: 'Scalp Machine' },
    { id: 'p2', name: 'Swing Daily' },
  ],
  Strategy: [
    { id: 's1', name: 'RSI Reversal' },
    { id: 's2', name: 'Breakout Rider' },
    { id: 's3', name: 'MACD Cross Daily' },
  ],
  Pack: [
    { id: 'pk1', name: 'Volume Profile Pack' },
    { id: 'pk2', name: 'Trend Strength Pack' },
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

export default function CreatorPublishV3() {
  const [type, setType] = useState<PublishType>('Portfolio');
  const [item, setItem] = useState('');
  const [price, setPrice] = useState('30');
  const [published, setPublished] = useState(false);

  const items = mockUnpublished[type] || [];

  return (
    <div>
      <PageHeader title="Publish" subtitle="Quick publish" backHref="/creator/dashboard" />

      <div className="grid grid-cols-2 gap-6">
        {/* Streamlined publish card */}
        <Card>
          <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
            Publish to Marketplace
          </h3>

          <div className="space-y-4">
            {/* Type */}
            <div>
              <label className="text-xs mb-1.5 block" style={{ color: 'var(--text-muted)' }}>Type</label>
              <div className="flex gap-2">
                {(['Portfolio', 'Strategy', 'Pack'] as PublishType[]).map((t) => (
                  <button
                    key={t}
                    className="flex-1 py-2 rounded-lg text-xs font-medium"
                    style={{
                      background: type === t ? typeColors[t]?.bg : 'var(--bg-input)',
                      color: type === t ? typeColors[t]?.color : 'var(--text-muted)',
                      border: `1px solid ${type === t ? typeColors[t]?.color : 'var(--border)'}`,
                    }}
                    onClick={() => { setType(t); setItem(''); }}
                  >
                    {t}
                  </button>
                ))}
              </div>
            </div>

            {/* Item */}
            <div>
              <label className="text-xs mb-1.5 block" style={{ color: 'var(--text-muted)' }}>Item</label>
              <select
                className="w-full px-3 py-2 rounded-lg text-sm"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                value={item}
                onChange={(e) => setItem(e.target.value)}
              >
                <option value="">Select {type.toLowerCase()}...</option>
                {items.map((i) => (
                  <option key={i.id} value={i.id}>{i.name}</option>
                ))}
              </select>
            </div>

            {/* Price */}
            <div>
              <label className="text-xs mb-1.5 block" style={{ color: 'var(--text-muted)' }}>Price ($/month)</label>
              <input
                type="number"
                className="w-full px-3 py-2 rounded-lg text-sm"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                value={price}
                onChange={(e) => setPrice(e.target.value)}
                min="1"
              />
              {price && (
                <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                  You receive ${(parseFloat(price) * 0.85).toFixed(2)}/sub after 15% fee
                </p>
              )}
            </div>

            {/* Publish */}
            <button
              className="w-full py-2.5 rounded-lg text-sm font-medium"
              style={{
                background: item ? 'var(--accent)' : 'var(--bg-input)',
                color: item ? 'white' : 'var(--text-muted)',
                cursor: item ? 'pointer' : 'not-allowed',
              }}
              disabled={!item}
              onClick={() => { setPublished(true); setTimeout(() => setPublished(false), 2000); }}
            >
              {published ? 'Published!' : 'Publish'}
            </button>
          </div>
        </Card>

        {/* Published list */}
        <Card>
          <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
            Published ({alreadyPublished.length})
          </h3>
          <div className="space-y-2">
            {alreadyPublished.map((p) => (
              <div
                key={p.id}
                className="flex items-center justify-between px-3 py-2.5 rounded-lg"
                style={{ background: 'var(--bg-input)' }}
              >
                <div className="flex items-center gap-2">
                  <span
                    className="text-xs px-1.5 py-0.5 rounded"
                    style={{ color: typeColors[p.type]?.color, background: typeColors[p.type]?.bg }}
                  >
                    {p.type[0]}
                  </span>
                  <span className="text-sm font-medium">{p.name}</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    ${p.price}/mo &middot; {p.subscribers}
                  </span>
                  <button
                    className="text-xs px-2 py-1 rounded"
                    style={{ color: 'var(--red)', background: 'var(--red-muted)' }}
                  >
                    x
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
