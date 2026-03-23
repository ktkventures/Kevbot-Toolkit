'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

interface CurationItem {
  id: string;
  name: string;
  creator: string;
  inFreeRotation: boolean;
  subscribers: number;
  winRate: number;
}

const mockItems: CurationItem[] = [
  { id: 'p1', name: 'Momentum Alpha', creator: 'Alex K.', inFreeRotation: true, subscribers: 128, winRate: 62.4 },
  { id: 'p2', name: 'Swing Master Pro', creator: 'David R.', inFreeRotation: true, subscribers: 67, winRate: 58.1 },
  { id: 'p3', name: 'Conservative Growth', creator: 'Maria L.', inFreeRotation: true, subscribers: 89, winRate: 64.7 },
  { id: 'p4', name: 'Scalper\'s Edge', creator: 'Lisa P.', inFreeRotation: false, subscribers: 45, winRate: 71.2 },
  { id: 'p5', name: 'Trend Rider', creator: 'Alex K.', inFreeRotation: false, subscribers: 34, winRate: 55.8 },
  { id: 'p6', name: 'Crypto Momentum', creator: 'James T.', inFreeRotation: false, subscribers: 23, winRate: 49.3 },
];

export default function AdminCurationV3() {
  const [items, setItems] = useState(mockItems);
  const [draggedId, setDraggedId] = useState<string | null>(null);

  const freeRotation = items.filter((i) => i.inFreeRotation);
  const available = items.filter((i) => !i.inFreeRotation);

  function moveToRotation(id: string) {
    setItems((prev) => prev.map((item) => item.id === id ? { ...item, inFreeRotation: true } : item));
  }

  function removeFromRotation(id: string) {
    setItems((prev) => prev.map((item) => item.id === id ? { ...item, inFreeRotation: false } : item));
  }

  function handleDragStart(id: string) {
    setDraggedId(id);
  }

  function handleDragEnd() {
    setDraggedId(null);
  }

  function handleDrop(targetId: string) {
    if (!draggedId || draggedId === targetId) return;

    setItems((prev) => {
      const rotationItems = prev.filter((i) => i.inFreeRotation);
      const dragIdx = rotationItems.findIndex((i) => i.id === draggedId);
      const targetIdx = rotationItems.findIndex((i) => i.id === targetId);

      if (dragIdx === -1 || targetIdx === -1) return prev;

      const newRotation = [...rotationItems];
      const [moved] = newRotation.splice(dragIdx, 1);
      newRotation.splice(targetIdx, 0, moved);

      const nonRotation = prev.filter((i) => !i.inFreeRotation);
      return [...newRotation, ...nonRotation];
    });
    setDraggedId(null);
  }

  return (
    <div>
      <PageHeader
        title="Content Curation"
        subtitle="Drag to reorder free rotation"
        backHref="/admin"
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Free Rotation List */}
        <div>
          <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-muted)' }}>
            FREE ROTATION ({freeRotation.length})
          </h3>
          <div className="space-y-2">
            {freeRotation.map((item, i) => (
              <div
                key={item.id}
                draggable
                onDragStart={() => handleDragStart(item.id)}
                onDragEnd={handleDragEnd}
                onDragOver={(e) => e.preventDefault()}
                onDrop={() => handleDrop(item.id)}
                className="flex items-center gap-3 p-3 rounded-lg cursor-grab active:cursor-grabbing transition-opacity"
                style={{
                  background: 'var(--bg-card)',
                  border: `1px solid ${draggedId === item.id ? 'var(--accent)' : 'var(--border)'}`,
                  opacity: draggedId === item.id ? 0.5 : 1,
                }}
              >
                <span className="text-xs font-mono w-5" style={{ color: 'var(--text-muted)' }}>#{i + 1}</span>
                <span className="cursor-grab" style={{ color: 'var(--text-muted)' }}>&#9776;</span>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium">{item.name}</p>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>by {item.creator}</p>
                </div>
                <button
                  onClick={() => removeFromRotation(item.id)}
                  className="text-xs px-2 py-1 rounded shrink-0"
                  style={{ background: 'var(--red-muted)', color: 'var(--red)' }}
                >
                  Remove
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Available Items */}
        <div>
          <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-muted)' }}>
            AVAILABLE ({available.length})
          </h3>
          <div className="space-y-2">
            {available.map((item) => (
              <Card key={item.id} className="flex items-center gap-3">
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium">{item.name}</p>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    by {item.creator} &middot; {item.subscribers} subs &middot; {item.winRate}% WR
                  </p>
                </div>
                <button
                  onClick={() => moveToRotation(item.id)}
                  className="text-xs px-2 py-1 rounded shrink-0"
                  style={{ background: 'var(--green-muted)', color: 'var(--green)' }}
                >
                  Add
                </button>
              </Card>
            ))}
            {available.length === 0 && (
              <p className="text-sm text-center py-4" style={{ color: 'var(--text-muted)' }}>All items are in rotation</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
