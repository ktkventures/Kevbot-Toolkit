'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

interface CurationItem {
  id: string;
  name: string;
  creator: string;
  type: 'portfolio' | 'strategy' | 'pack';
  inFreeRotation: boolean;
  subscribers: number;
  winRate: number;
}

const mockItems: CurationItem[] = [
  { id: 'p1', name: 'Momentum Alpha', creator: 'Alex K.', type: 'portfolio', inFreeRotation: true, subscribers: 128, winRate: 62.4 },
  { id: 'p2', name: 'Swing Master Pro', creator: 'David R.', type: 'portfolio', inFreeRotation: true, subscribers: 67, winRate: 58.1 },
  { id: 'p3', name: 'Scalper\'s Edge', creator: 'Lisa P.', type: 'portfolio', inFreeRotation: false, subscribers: 45, winRate: 71.2 },
  { id: 'p4', name: 'Trend Rider', creator: 'Alex K.', type: 'portfolio', inFreeRotation: false, subscribers: 34, winRate: 55.8 },
  { id: 'p5', name: 'Conservative Growth', creator: 'Maria L.', type: 'portfolio', inFreeRotation: true, subscribers: 89, winRate: 64.7 },
  { id: 'p6', name: 'Crypto Momentum', creator: 'James T.', type: 'portfolio', inFreeRotation: false, subscribers: 23, winRate: 49.3 },
];

export default function AdminCurationV1() {
  const [items, setItems] = useState(mockItems);

  function toggleFreeRotation(id: string) {
    setItems((prev) => prev.map((item) => item.id === id ? { ...item, inFreeRotation: !item.inFreeRotation } : item));
  }

  const freeCount = items.filter((i) => i.inFreeRotation).length;

  return (
    <div>
      <PageHeader
        title="Content Curation"
        subtitle={`${freeCount} items in free rotation`}
        backHref="/admin"
      />

      <div className="space-y-3">
        {items.map((item) => (
          <Card key={item.id} className="flex items-center gap-4">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <p className="font-medium">{item.name}</p>
                {item.inFreeRotation && (
                  <span className="px-2 py-0.5 rounded-full text-xs" style={{ background: 'var(--green-muted)', color: 'var(--green)' }}>
                    Free Rotation
                  </span>
                )}
              </div>
              <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
                by {item.creator} &middot; {item.subscribers} subscribers &middot; {item.winRate}% win rate
              </p>
            </div>
            <button
              onClick={() => toggleFreeRotation(item.id)}
              className="px-3 py-1.5 rounded-lg text-sm font-medium shrink-0"
              style={{
                background: item.inFreeRotation ? 'var(--red-muted)' : 'var(--green-muted)',
                color: item.inFreeRotation ? 'var(--red)' : 'var(--green)',
                border: `1px solid ${item.inFreeRotation ? 'var(--red)' : 'var(--green)'}`,
              }}
            >
              {item.inFreeRotation ? 'Remove' : 'Add to Free Rotation'}
            </button>
          </Card>
        ))}
      </div>
    </div>
  );
}
