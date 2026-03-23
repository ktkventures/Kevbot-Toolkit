'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import MetricCard from '@/components/MetricCard';
import Modal from '@/components/Modal';

interface CurationItem {
  id: string;
  name: string;
  creator: string;
  type: 'portfolio' | 'strategy' | 'pack';
  inFreeRotation: boolean;
  featured: boolean;
  subscribers: number;
  winRate: number;
  pf: number;
  drawdown: number;
  reviewStatus: 'approved' | 'pending' | 'rejected';
  submittedAt: string;
}

const mockItems: CurationItem[] = [
  { id: 'p1', name: 'Momentum Alpha', creator: 'Alex K.', type: 'portfolio', inFreeRotation: true, featured: true, subscribers: 128, winRate: 62.4, pf: 2.1, drawdown: 4.2, reviewStatus: 'approved', submittedAt: '2026-02-10' },
  { id: 'p2', name: 'Swing Master Pro', creator: 'David R.', type: 'portfolio', inFreeRotation: true, featured: false, subscribers: 67, winRate: 58.1, pf: 1.8, drawdown: 6.1, reviewStatus: 'approved', submittedAt: '2026-02-15' },
  { id: 'p3', name: 'Scalper\'s Edge', creator: 'Lisa P.', type: 'portfolio', inFreeRotation: false, featured: false, subscribers: 45, winRate: 71.2, pf: 2.8, drawdown: 2.8, reviewStatus: 'approved', submittedAt: '2026-02-20' },
  { id: 'p4', name: 'Trend Rider', creator: 'Alex K.', type: 'portfolio', inFreeRotation: false, featured: false, subscribers: 34, winRate: 55.8, pf: 1.5, drawdown: 7.3, reviewStatus: 'pending', submittedAt: '2026-03-18' },
  { id: 'p5', name: 'Conservative Growth', creator: 'Maria L.', type: 'portfolio', inFreeRotation: true, featured: false, subscribers: 89, winRate: 64.7, pf: 2.3, drawdown: 3.1, reviewStatus: 'approved', submittedAt: '2026-02-25' },
  { id: 'p6', name: 'Crypto Momentum', creator: 'James T.', type: 'portfolio', inFreeRotation: false, featured: false, subscribers: 23, winRate: 49.3, pf: 1.2, drawdown: 12.5, reviewStatus: 'pending', submittedAt: '2026-03-19' },
  { id: 's1', name: 'EMA Breakout Strategy', creator: 'Sarah M.', type: 'strategy', inFreeRotation: false, featured: false, subscribers: 0, winRate: 68.0, pf: 2.5, drawdown: 3.5, reviewStatus: 'pending', submittedAt: '2026-03-20' },
  { id: 'k1', name: 'VWAP Extended Pack', creator: 'Sarah M.', type: 'pack', inFreeRotation: false, featured: false, subscribers: 0, winRate: 0, pf: 0, drawdown: 0, reviewStatus: 'pending', submittedAt: '2026-03-21' },
];

const statusColors: Record<string, { color: string; bg: string }> = {
  approved: { color: 'var(--green)', bg: 'var(--green-muted)' },
  pending: { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  rejected: { color: 'var(--red)', bg: 'var(--red-muted)' },
};

export default function AdminCurationV2() {
  const [items, setItems] = useState(mockItems);
  const [detailItem, setDetailItem] = useState<CurationItem | null>(null);

  const freeRotation = items.filter((i) => i.inFreeRotation);
  const pendingReview = items.filter((i) => i.reviewStatus === 'pending');

  function toggleFreeRotation(id: string) {
    setItems((prev) => prev.map((item) => item.id === id ? { ...item, inFreeRotation: !item.inFreeRotation } : item));
  }

  function toggleFeatured(id: string) {
    setItems((prev) => prev.map((item) => item.id === id ? { ...item, featured: !item.featured } : item));
  }

  function approveItem(id: string) {
    setItems((prev) => prev.map((item) => item.id === id ? { ...item, reviewStatus: 'approved' as const } : item));
    setDetailItem(null);
  }

  function rejectItem(id: string) {
    setItems((prev) => prev.map((item) => item.id === id ? { ...item, reviewStatus: 'rejected' as const } : item));
    setDetailItem(null);
  }

  return (
    <div>
      <PageHeader
        title="Content Curation"
        subtitle="Free rotation, featured items, and content review"
        backHref="/admin"
      />

      <div className="grid grid-cols-4 gap-4 mb-6">
        <MetricCard label="Free Rotation" value={freeRotation.length.toString()} />
        <MetricCard label="Featured" value={items.filter((i) => i.featured).length.toString()} />
        <MetricCard label="Pending Review" value={pendingReview.length.toString()} delta={pendingReview.length > 0 ? 'Needs attention' : undefined} positive={false} />
        <MetricCard label="Total Published" value={items.filter((i) => i.reviewStatus === 'approved').length.toString()} />
      </div>

      <TabBar tabs={['Free Rotation', 'Featured', 'Review Queue', 'All Content']}>
        {(tab) => {
          let filteredItems = items;
          if (tab === 'Free Rotation') filteredItems = items.filter((i) => i.inFreeRotation);
          if (tab === 'Featured') filteredItems = items.filter((i) => i.featured);
          if (tab === 'Review Queue') filteredItems = items.filter((i) => i.reviewStatus === 'pending');

          return (
            <div className="space-y-3">
              {filteredItems.length === 0 && (
                <p className="text-sm text-center py-8" style={{ color: 'var(--text-muted)' }}>No items in this category</p>
              )}
              {filteredItems.map((item) => (
                <Card key={item.id} className="flex items-center gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <p className="font-medium">{item.name}</p>
                      <span className="px-2 py-0.5 rounded-full text-xs" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
                        {item.type}
                      </span>
                      <span className="px-2 py-0.5 rounded-full text-xs" style={{ background: statusColors[item.reviewStatus].bg, color: statusColors[item.reviewStatus].color }}>
                        {item.reviewStatus}
                      </span>
                      {item.featured && (
                        <span className="px-2 py-0.5 rounded-full text-xs" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>Featured</span>
                      )}
                    </div>
                    <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                      by {item.creator}
                      {item.type === 'portfolio' && <> &middot; {item.subscribers} subs &middot; {item.winRate}% WR &middot; {item.pf} PF &middot; {item.drawdown}% DD</>}
                    </p>
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    {item.reviewStatus === 'pending' && (
                      <button
                        onClick={() => setDetailItem(item)}
                        className="px-3 py-1.5 rounded-lg text-xs font-medium"
                        style={{ background: 'var(--accent-muted)', color: 'var(--accent)', border: '1px solid var(--accent)' }}
                      >
                        Review
                      </button>
                    )}
                    {item.reviewStatus === 'approved' && item.type === 'portfolio' && (
                      <>
                        <button
                          onClick={() => toggleFeatured(item.id)}
                          className="px-3 py-1.5 rounded-lg text-xs font-medium"
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
                          className="px-3 py-1.5 rounded-lg text-xs font-medium"
                          style={{
                            background: item.inFreeRotation ? 'var(--red-muted)' : 'var(--green-muted)',
                            color: item.inFreeRotation ? 'var(--red)' : 'var(--green)',
                            border: `1px solid ${item.inFreeRotation ? 'var(--red)' : 'var(--green)'}`,
                          }}
                        >
                          {item.inFreeRotation ? 'Remove' : 'Add to Rotation'}
                        </button>
                      </>
                    )}
                  </div>
                </Card>
              ))}
            </div>
          );
        }}
      </TabBar>

      {/* Review Modal */}
      <Modal title="Content Review" isOpen={!!detailItem} onClose={() => setDetailItem(null)} width="520px">
        {detailItem && (
          <div className="space-y-4">
            <div>
              <h3 className="font-semibold text-lg">{detailItem.name}</h3>
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                by {detailItem.creator} &middot; Submitted {detailItem.submittedAt}
              </p>
            </div>

            {detailItem.type === 'portfolio' && (
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Win Rate</p>
                  <p className="text-lg font-semibold">{detailItem.winRate}%</p>
                </div>
                <div className="p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Profit Factor</p>
                  <p className="text-lg font-semibold">{detailItem.pf}</p>
                </div>
                <div className="p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Max Drawdown</p>
                  <p className="text-lg font-semibold">{detailItem.drawdown}%</p>
                </div>
                <div className="p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Subscribers</p>
                  <p className="text-lg font-semibold">{detailItem.subscribers}</p>
                </div>
              </div>
            )}

            <div className="flex gap-3">
              <button
                onClick={() => approveItem(detailItem.id)}
                className="flex-1 py-2.5 rounded-lg text-sm font-medium"
                style={{ background: 'var(--green)', color: 'var(--bg-primary)' }}
              >
                Approve
              </button>
              <button
                onClick={() => rejectItem(detailItem.id)}
                className="flex-1 py-2.5 rounded-lg text-sm font-medium"
                style={{ background: 'var(--red)', color: 'var(--bg-primary)' }}
              >
                Reject
              </button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
