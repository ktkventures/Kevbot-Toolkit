'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

interface PropFirm {
  id: string;
  name: string;
  accounts: string[];
  evalPrice: string;
  profitTarget: string;
  maxDD: string;
  dailyLoss: string;
  minDays: number;
  compatible: number;
  rating: number;
}

const mockFirms: PropFirm[] = [
  { id: 'ttp', name: 'Trade The Pool', accounts: ['25K', '50K', '100K', '200K'], evalPrice: '$99-$399', profitTarget: '$3,000-$12,000', maxDD: '6%', dailyLoss: '2%', minDays: 5, compatible: 12, rating: 4.6 },
  { id: 'ftmo', name: 'FTMO', accounts: ['10K', '25K', '50K', '100K', '200K'], evalPrice: '$155-$1,080', profitTarget: '10%', maxDD: '10%', dailyLoss: '5%', minDays: 4, compatible: 8, rating: 4.5 },
  { id: 'topstep', name: 'Topstep', accounts: ['50K', '100K', '150K'], evalPrice: '$49-$149/mo', profitTarget: '$3,000-$9,000', maxDD: '$2,000-$4,500', dailyLoss: '$1,000-$2,000', minDays: 0, compatible: 6, rating: 4.3 },
  { id: 'apex', name: 'Apex Trader', accounts: ['25K', '50K', '100K', '250K'], evalPrice: '$37-$187/mo', profitTarget: '$1,500-$15,000', maxDD: '$1,500-$6,250', dailyLoss: 'N/A', minDays: 7, compatible: 5, rating: 4.1 },
];

function StarRating({ rating }: { rating: number }) {
  return (
    <div className="flex items-center gap-1">
      {[1, 2, 3, 4, 5].map((star) => (
        <span key={star} className="text-xs" style={{ color: star <= Math.round(rating) ? 'var(--orange)' : 'var(--text-muted)' }}>&#9733;</span>
      ))}
      <span className="text-xs ml-1" style={{ color: 'var(--text-muted)' }}>{rating.toFixed(1)}</span>
    </div>
  );
}

export default function PropFirmsV1() {
  const [selectedFirm, setSelectedFirm] = useState<PropFirm | null>(null);

  return (
    <div>
      <PageHeader
        title="Prop Firm Hub"
        subtitle="Browse compatible prop firms and find the right fit"
        backHref="/marketplace"
      />

      {selectedFirm ? (
        <div>
          <button
            onClick={() => setSelectedFirm(null)}
            className="text-sm mb-4 flex items-center gap-1"
            style={{ color: 'var(--text-muted)' }}
          >
            &larr; Back to Firms
          </button>

          <Card className="mb-4">
            <div className="flex items-start justify-between mb-4">
              <div>
                <h2 className="text-xl font-bold">{selectedFirm.name}</h2>
                <StarRating rating={selectedFirm.rating} />
              </div>
              <button className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: 'white' }}>
                Sign Up (Affiliate)
              </button>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
              <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Evaluation Price</p>
                <p className="text-sm font-semibold">{selectedFirm.evalPrice}</p>
              </div>
              <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Profit Target</p>
                <p className="text-sm font-semibold">{selectedFirm.profitTarget}</p>
              </div>
              <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Max Drawdown</p>
                <p className="text-sm font-semibold" style={{ color: 'var(--red)' }}>{selectedFirm.maxDD}</p>
              </div>
              <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Daily Loss Limit</p>
                <p className="text-sm font-semibold" style={{ color: 'var(--red)' }}>{selectedFirm.dailyLoss}</p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Min Trading Days</p>
                <p className="text-sm font-semibold">{selectedFirm.minDays === 0 ? 'None' : selectedFirm.minDays}</p>
              </div>
              <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Account Sizes</p>
                <div className="flex flex-wrap gap-1 mt-1">
                  {selectedFirm.accounts.map((acc) => (
                    <span key={acc} className="text-xs px-1.5 py-0.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>{acc}</span>
                  ))}
                </div>
              </div>
            </div>
          </Card>

          <Card>
            <h3 className="text-sm font-medium mb-3">{selectedFirm.compatible} Compatible Portfolios</h3>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>These marketplace portfolios meet {selectedFirm.name}&apos;s evaluation requirements</p>
            <div className="mt-3 space-y-2">
              {Array.from({ length: Math.min(selectedFirm.compatible, 5) }, (_, i) => (
                <div key={i} className="flex items-center justify-between py-2 px-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                  <span className="text-sm">Portfolio {i + 1}</span>
                  <button className="text-xs" style={{ color: 'var(--accent)' }}>View</button>
                </div>
              ))}
              {selectedFirm.compatible > 5 && (
                <p className="text-xs text-center pt-2" style={{ color: 'var(--text-muted)' }}>+{selectedFirm.compatible - 5} more portfolios</p>
              )}
            </div>
          </Card>
        </div>
      ) : (
        <>
          <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>{mockFirms.length} prop firms available</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {mockFirms.map((firm) => (
              <div key={firm.id} onClick={() => setSelectedFirm(firm)} className="cursor-pointer">
              <Card className="hover:border-[var(--accent)] transition-colors">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <div className="w-12 h-12 rounded-lg flex items-center justify-center text-lg font-bold mb-2" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                      {firm.name.charAt(0)}
                    </div>
                    <h3 className="font-semibold text-sm">{firm.name}</h3>
                    <StarRating rating={firm.rating} />
                  </div>
                  <button
                    className="px-3 py-1.5 rounded-lg text-xs font-medium"
                    style={{ background: 'var(--accent)', color: 'white' }}
                    onClick={(e) => { e.stopPropagation(); }}
                  >
                    Sign Up
                  </button>
                </div>

                <div className="grid grid-cols-2 gap-3 mb-3">
                  <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>Eval Price</p><p className="text-sm font-semibold">{firm.evalPrice}</p></div>
                  <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>Profit Target</p><p className="text-sm font-semibold">{firm.profitTarget}</p></div>
                  <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>Max DD</p><p className="text-sm font-semibold" style={{ color: 'var(--red)' }}>{firm.maxDD}</p></div>
                  <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>Daily Loss</p><p className="text-sm font-semibold" style={{ color: 'var(--red)' }}>{firm.dailyLoss}</p></div>
                </div>

                <div className="flex items-center justify-between pt-3 border-t" style={{ borderColor: 'var(--border)' }}>
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Min {firm.minDays === 0 ? 'No min' : `${firm.minDays} days`}</span>
                  <span className="text-xs font-medium" style={{ color: 'var(--accent)' }}>{firm.compatible} compatible portfolios</span>
                </div>
              </Card>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
