'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

interface CurationItem {
  id: string;
  name: string;
  creator: string;
  inFreeRotation: boolean;
  featured: boolean;
  subscribers: number;
  winRate: number;
  pf: number;
  dailyR: number;
  score: number;
  abVariant?: 'A' | 'B';
}

const mockItems: CurationItem[] = [
  { id: 'p1', name: 'Momentum Alpha', creator: 'Alex K.', inFreeRotation: true, featured: true, subscribers: 128, winRate: 62.4, pf: 2.1, dailyR: 0.34, score: 92 },
  { id: 'p2', name: 'Swing Master Pro', creator: 'David R.', inFreeRotation: true, featured: false, subscribers: 67, winRate: 58.1, pf: 1.8, dailyR: 0.28, score: 78 },
  { id: 'p3', name: 'Conservative Growth', creator: 'Maria L.', inFreeRotation: true, featured: false, subscribers: 89, winRate: 64.7, pf: 2.3, dailyR: 0.22, score: 85 },
  { id: 'p4', name: 'Scalper\'s Edge', creator: 'Lisa P.', inFreeRotation: false, featured: false, subscribers: 45, winRate: 71.2, pf: 2.8, dailyR: 0.41, score: 88, abVariant: 'A' },
  { id: 'p5', name: 'Trend Rider', creator: 'Alex K.', inFreeRotation: false, featured: false, subscribers: 34, winRate: 55.8, pf: 1.5, dailyR: 0.19, score: 62 },
  { id: 'p6', name: 'Crypto Momentum', creator: 'James T.', inFreeRotation: false, featured: false, subscribers: 23, winRate: 49.3, pf: 1.2, dailyR: 0.15, score: 45, abVariant: 'B' },
];

function getScoreColor(score: number): string {
  if (score >= 80) return 'var(--green)';
  if (score >= 60) return 'var(--blue)';
  if (score >= 40) return 'var(--orange)';
  return 'var(--red)';
}

export default function AdminCurationV4() {
  const [items, setItems] = useState(mockItems);
  const [view, setView] = useState<'board' | 'suggestions'>('board');

  const freeRotation = items.filter((i) => i.inFreeRotation);
  const available = items.filter((i) => !i.inFreeRotation);
  const suggestions = [...items].filter((i) => !i.inFreeRotation).sort((a, b) => b.score - a.score);

  function toggleFreeRotation(id: string) {
    setItems((prev) => prev.map((item) => item.id === id ? { ...item, inFreeRotation: !item.inFreeRotation } : item));
  }

  function toggleFeatured(id: string) {
    setItems((prev) => prev.map((item) => item.id === id ? { ...item, featured: !item.featured } : item));
  }

  return (
    <div>
      <PageHeader
        title="Content Curation"
        subtitle="Visual board with performance-based suggestions"
        backHref="/admin"
        actions={
          <div className="flex gap-1 rounded-lg p-0.5" style={{ background: 'var(--bg-input)' }}>
            {(['board', 'suggestions'] as const).map((v) => (
              <button
                key={v}
                onClick={() => setView(v)}
                className="px-3 py-1.5 rounded-md text-xs font-medium transition-colors capitalize"
                style={{
                  background: view === v ? 'var(--bg-card)' : 'transparent',
                  color: view === v ? 'var(--text-primary)' : 'var(--text-muted)',
                }}
              >
                {v}
              </button>
            ))}
          </div>
        }
      />

      {view === 'board' ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Free Rotation Column */}
          <div>
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-2" style={{ color: 'var(--green)' }}>
              <span className="w-2 h-2 rounded-full" style={{ background: 'var(--green)' }} />
              FREE ROTATION ({freeRotation.length})
            </h3>
            <div className="space-y-3">
              {freeRotation.map((item) => (
                <div
                  key={item.id}
                  className="rounded-xl border p-4 transition-all"
                  style={{ background: 'var(--bg-card)', borderColor: item.featured ? 'var(--accent)' : 'var(--border)' }}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <div className="flex items-center gap-2">
                        <p className="font-semibold">{item.name}</p>
                        {item.featured && (
                          <span className="px-1.5 py-0.5 rounded text-xs" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>Featured</span>
                        )}
                      </div>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>by {item.creator}</p>
                    </div>
                    <div
                      className="w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold"
                      style={{ background: getScoreColor(item.score) + '22', color: getScoreColor(item.score), border: `2px solid ${getScoreColor(item.score)}` }}
                    >
                      {item.score}
                    </div>
                  </div>

                  <div className="grid grid-cols-4 gap-2 mb-3">
                    {[
                      { label: 'Subs', value: item.subscribers.toString() },
                      { label: 'WR', value: `${item.winRate}%` },
                      { label: 'PF', value: item.pf.toString() },
                      { label: 'Daily R', value: item.dailyR.toString() },
                    ].map((m) => (
                      <div key={m.label} className="text-center p-1.5 rounded" style={{ background: 'var(--bg-input)' }}>
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{m.label}</p>
                        <p className="text-sm font-semibold">{m.value}</p>
                      </div>
                    ))}
                  </div>

                  <div className="flex gap-2">
                    <button
                      onClick={() => toggleFeatured(item.id)}
                      className="flex-1 py-1.5 rounded-lg text-xs font-medium"
                      style={{
                        background: item.featured ? 'var(--accent)' : 'var(--bg-input)',
                        color: item.featured ? 'var(--bg-primary)' : 'var(--text-secondary)',
                        border: `1px solid ${item.featured ? 'var(--accent)' : 'var(--border)'}`,
                      }}
                    >
                      {item.featured ? 'Unfeature' : 'Feature'}
                    </button>
                    <button
                      onClick={() => toggleFreeRotation(item.id)}
                      className="flex-1 py-1.5 rounded-lg text-xs font-medium"
                      style={{ background: 'var(--red-muted)', color: 'var(--red)', border: '1px solid var(--red)' }}
                    >
                      Remove
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Available Column */}
          <div>
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-2" style={{ color: 'var(--text-muted)' }}>
              <span className="w-2 h-2 rounded-full" style={{ background: 'var(--text-muted)' }} />
              AVAILABLE ({available.length})
            </h3>
            <div className="space-y-3">
              {available.map((item) => (
                <div
                  key={item.id}
                  className="rounded-xl border p-4"
                  style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <div className="flex items-center gap-2">
                        <p className="font-medium">{item.name}</p>
                        {item.abVariant && (
                          <span className="px-1.5 py-0.5 rounded text-xs" style={{ background: 'var(--purple)22', color: 'var(--purple)' }}>
                            A/B: {item.abVariant}
                          </span>
                        )}
                      </div>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>by {item.creator}</p>
                    </div>
                    <div
                      className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold"
                      style={{ color: getScoreColor(item.score) }}
                    >
                      {item.score}
                    </div>
                  </div>
                  <button
                    onClick={() => toggleFreeRotation(item.id)}
                    className="w-full py-1.5 rounded-lg text-xs font-medium"
                    style={{ background: 'var(--green-muted)', color: 'var(--green)', border: '1px solid var(--green)' }}
                  >
                    Add to Rotation
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        /* Suggestions View */
        <Card>
          <h3 className="text-sm font-semibold mb-4">Performance-Based Suggestions</h3>
          <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
            Items ranked by quality score (subscribers, win rate, profit factor, daily R)
          </p>
          <div className="space-y-3">
            {suggestions.map((item, i) => (
              <div
                key={item.id}
                className="flex items-center gap-4 p-3 rounded-lg"
                style={{ background: 'var(--bg-input)' }}
              >
                <span className="text-lg font-bold w-8 text-center" style={{ color: getScoreColor(item.score) }}>
                  #{i + 1}
                </span>
                <div className="flex-1 min-w-0">
                  <p className="font-medium">{item.name}</p>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    {item.winRate}% WR &middot; {item.pf} PF &middot; {item.subscribers} subs
                  </p>
                </div>
                <div
                  className="px-3 py-1 rounded-full text-sm font-bold"
                  style={{ background: getScoreColor(item.score) + '22', color: getScoreColor(item.score) }}
                >
                  {item.score}
                </div>
                <button
                  onClick={() => toggleFreeRotation(item.id)}
                  className="px-3 py-1.5 rounded-lg text-xs font-medium shrink-0"
                  style={{ background: 'var(--green-muted)', color: 'var(--green)' }}
                >
                  Add
                </button>
              </div>
            ))}
            {suggestions.length === 0 && (
              <p className="text-sm text-center py-4" style={{ color: 'var(--text-muted)' }}>All items are already in rotation</p>
            )}
          </div>
        </Card>
      )}
    </div>
  );
}
