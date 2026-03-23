'use client';

import { useState, useEffect } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import ChartPlaceholder from '@/components/ChartPlaceholder';

type PublishType = 'Portfolio' | 'Strategy' | 'Pack';

const mockUnpublished = {
  Portfolio: [
    { id: 'p1', name: 'Scalp Machine', strategies: 4, winRate: 62.3, subscribers_est: 120 },
    { id: 'p2', name: 'Swing Daily', strategies: 3, winRate: 58.1, subscribers_est: 60 },
  ],
  Strategy: [
    { id: 's1', name: 'RSI Reversal', winRate: 55.7, profitFactor: 1.8, subscribers_est: 40 },
    { id: 's2', name: 'Breakout Rider', winRate: 51.2, profitFactor: 2.1, subscribers_est: 35 },
  ],
  Pack: [
    { id: 'pk1', name: 'Volume Profile Pack', indicators: 3, subscribers_est: 50 },
    { id: 'pk2', name: 'Trend Strength Pack', indicators: 4, subscribers_est: 45 },
  ],
};

const mockCompetitive = [
  { name: 'Similar Portfolio A', price: 35, subscribers: 210, rating: 4.6 },
  { name: 'Similar Portfolio B', price: 25, subscribers: 145, rating: 4.3 },
  { name: 'Similar Portfolio C', price: 40, subscribers: 98, rating: 4.8 },
];

const typeColors: Record<string, { color: string; bg: string }> = {
  Portfolio: { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  Strategy: { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  Pack: { color: 'var(--purple)', bg: 'var(--purple-muted)' },
};

const stepEmojis = ['1', '2', '3', '4'];

function AnimatedNumber({ value, prefix = '' }: { value: number; prefix?: string }) {
  const [current, setCurrent] = useState(0);
  useEffect(() => {
    let v = 0;
    const step = value / 30;
    const timer = setInterval(() => {
      v += step;
      if (v >= value) { setCurrent(value); clearInterval(timer); }
      else setCurrent(Math.floor(v));
    }, 30);
    return () => clearInterval(timer);
  }, [value]);
  return <span>{prefix}{current.toLocaleString()}</span>;
}

export default function CreatorPublishV4() {
  const [step, setStep] = useState(1);
  const [type, setType] = useState<PublishType>('Portfolio');
  const [selectedItem, setSelectedItem] = useState<string | null>(null);
  const [price, setPrice] = useState(30);
  const [showPreview, setShowPreview] = useState(false);

  const items = mockUnpublished[type] || [];
  const selected = items.find((i) => i.id === selectedItem);
  const estimatedSubs = selected ? selected.subscribers_est : 100;
  const projectedMonthly = price * estimatedSubs * 0.85;
  const projectedAnnual = projectedMonthly * 12;
  const avgCompetitorPrice = mockCompetitive.reduce((s, c) => s + c.price, 0) / mockCompetitive.length;

  return (
    <div>
      <style>{`
        @keyframes wizardSlide {
          from { opacity: 0; transform: translateX(20px); }
          to { opacity: 1; transform: translateX(0); }
        }
        .wizard-step {
          animation: wizardSlide 0.3s ease-out;
        }
        @keyframes revenueGlow {
          0%, 100% { text-shadow: 0 0 8px transparent; }
          50% { text-shadow: 0 0 16px var(--green); }
        }
        .revenue-glow {
          animation: revenueGlow 2s ease-in-out infinite;
        }
      `}</style>

      <PageHeader
        title="Publish Wizard"
        subtitle="Visual guided publishing with revenue projections"
        backHref="/creator/dashboard"
      />

      {/* Animated step indicator */}
      <div className="flex items-center justify-center gap-3 mb-8">
        {stepEmojis.map((emoji, i) => {
          const stepNum = i + 1;
          const isActive = step === stepNum;
          const isComplete = step > stepNum;
          return (
            <div key={i} className="flex items-center gap-3">
              <button
                onClick={() => stepNum <= step && setStep(stepNum)}
                className="w-10 h-10 rounded-xl flex items-center justify-center text-sm font-bold transition-all"
                style={{
                  background: isActive ? 'var(--accent)' : isComplete ? 'var(--green)' : 'var(--bg-input)',
                  color: isActive || isComplete ? 'white' : 'var(--text-muted)',
                  transform: isActive ? 'scale(1.15)' : 'scale(1)',
                  boxShadow: isActive ? '0 0 20px var(--accent-muted)' : 'none',
                  cursor: stepNum <= step ? 'pointer' : 'default',
                }}
              >
                {isComplete ? '!' : emoji}
              </button>
              {i < stepEmojis.length - 1 && (
                <div
                  className="w-12 h-1 rounded-full transition-colors"
                  style={{ background: isComplete ? 'var(--green)' : 'var(--border)' }}
                />
              )}
            </div>
          );
        })}
      </div>

      <div className="wizard-step" key={step}>
        {/* Step 1: Choose type */}
        {step === 1 && (
          <div className="grid grid-cols-3 gap-6 max-w-3xl mx-auto">
            {(['Portfolio', 'Strategy', 'Pack'] as PublishType[]).map((t) => (
              <Card key={t}>
                <div
                  className="cursor-pointer text-center py-4 transition-transform hover:scale-[1.02]"
                  onClick={() => { setType(t); setSelectedItem(null); setStep(2); }}
                >
                  <div
                    className="w-16 h-16 rounded-2xl flex items-center justify-center text-2xl font-bold mx-auto mb-4"
                    style={{
                      background: typeColors[t]?.bg,
                      color: typeColors[t]?.color,
                      boxShadow: `0 4px 20px ${typeColors[t]?.bg}`,
                    }}
                  >
                    {t[0]}
                  </div>
                  <h3 className="font-semibold mb-1">{t}</h3>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    {t === 'Portfolio' && 'Curated strategy collections'}
                    {t === 'Strategy' && 'Individual trading strategies'}
                    {t === 'Pack' && 'Indicator & trigger packs'}
                  </p>
                </div>
              </Card>
            ))}
          </div>
        )}

        {/* Step 2: Select item */}
        {step === 2 && (
          <div className="max-w-2xl mx-auto">
            <Card>
              <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
                Choose {type}
              </h3>
              <div className="space-y-3">
                {items.map((item) => (
                  <div
                    key={item.id}
                    className="flex items-center gap-4 p-4 rounded-lg cursor-pointer transition-all hover:scale-[1.01]"
                    style={{
                      background: selectedItem === item.id ? typeColors[type]?.bg : 'var(--bg-input)',
                      border: `2px solid ${selectedItem === item.id ? typeColors[type]?.color : 'transparent'}`,
                    }}
                    onClick={() => setSelectedItem(item.id)}
                  >
                    <div
                      className="w-10 h-10 rounded-lg flex items-center justify-center font-bold"
                      style={{ background: typeColors[type]?.bg, color: typeColors[type]?.color }}
                    >
                      {item.name[0]}
                    </div>
                    <div className="flex-1">
                      <p className="text-sm font-medium">{item.name}</p>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        {'strategies' in item && `${item.strategies} strategies | `}
                        {'winRate' in item && `WR: ${item.winRate}% | `}
                        {'profitFactor' in item && `PF: ${item.profitFactor} | `}
                        {'indicators' in item && `${item.indicators} indicators | `}
                        Est. {item.subscribers_est} subs
                      </p>
                    </div>
                  </div>
                ))}
              </div>
              <div className="flex gap-3 mt-6">
                <button
                  className="px-4 py-2 rounded-lg text-sm"
                  style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                  onClick={() => setStep(1)}
                >
                  Back
                </button>
                <button
                  className="flex-1 py-2 rounded-lg text-sm font-medium"
                  style={{
                    background: selectedItem ? 'var(--accent)' : 'var(--bg-input)',
                    color: selectedItem ? 'white' : 'var(--text-muted)',
                    cursor: selectedItem ? 'pointer' : 'not-allowed',
                  }}
                  disabled={!selectedItem}
                  onClick={() => setStep(3)}
                >
                  Next: Set Price
                </button>
              </div>
            </Card>
          </div>
        )}

        {/* Step 3: Price + Revenue Projection */}
        {step === 3 && (
          <div className="grid grid-cols-2 gap-6 max-w-4xl mx-auto">
            <Card>
              <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
                Set Your Price
              </h3>
              <div className="mb-6">
                <label className="text-xs mb-2 block" style={{ color: 'var(--text-muted)' }}>Monthly Subscription</label>
                <div className="flex items-center gap-3">
                  <span className="text-2xl font-bold" style={{ color: 'var(--text-muted)' }}>$</span>
                  <input
                    type="range"
                    min="5"
                    max="200"
                    value={price}
                    onChange={(e) => setPrice(parseInt(e.target.value))}
                    className="flex-1"
                  />
                  <span className="text-2xl font-bold min-w-[60px] text-right">${price}</span>
                </div>
                <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
                  After 15% platform fee: ${(price * 0.85).toFixed(2)}/subscriber
                </p>
              </div>

              {/* Revenue projection */}
              <div className="p-4 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                <h4 className="text-xs font-medium mb-3" style={{ color: 'var(--text-muted)' }}>
                  Revenue Projection (est. {estimatedSubs} subscribers)
                </h4>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>Monthly</span>
                    <span className="text-lg font-bold revenue-glow" style={{ color: 'var(--green)' }}>
                      <AnimatedNumber value={Math.round(projectedMonthly)} prefix="$" />
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>Annual</span>
                    <span className="text-lg font-bold" style={{ color: 'var(--accent)' }}>
                      <AnimatedNumber value={Math.round(projectedAnnual)} prefix="$" />
                    </span>
                  </div>
                </div>
              </div>

              {/* Preview mockup */}
              <div className="mt-6">
                <button
                  className="text-xs mb-2"
                  style={{ color: 'var(--accent)' }}
                  onClick={() => setShowPreview(!showPreview)}
                >
                  {showPreview ? 'Hide' : 'Show'} Listing Preview
                </button>
                {showPreview && (
                  <div className="p-4 rounded-lg border" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
                    <div className="flex items-center gap-2 mb-2">
                      <span
                        className="text-xs px-2 py-0.5 rounded"
                        style={{ color: typeColors[type]?.color, background: typeColors[type]?.bg }}
                      >
                        {type}
                      </span>
                      <span className="text-sm font-semibold">{selected?.name || 'Your Item'}</span>
                    </div>
                    <ChartPlaceholder label="Performance chart" height={120} />
                    <div className="flex justify-between mt-3">
                      <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        {'winRate' in (selected || {}) ? `WR: ${(selected as { winRate: number }).winRate}%` : ''}
                      </span>
                      <span className="text-sm font-bold" style={{ color: 'var(--accent)' }}>${price}/mo</span>
                    </div>
                  </div>
                )}
              </div>
            </Card>

            {/* Competitive Analysis */}
            <Card>
              <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
                Competitive Analysis
              </h3>
              <div className="mb-4 p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Average competitor price</p>
                <p className="text-xl font-bold" style={{ color: 'var(--accent)' }}>
                  ${avgCompetitorPrice.toFixed(0)}/mo
                </p>
                <p className="text-xs mt-1" style={{ color: price > avgCompetitorPrice ? 'var(--orange)' : 'var(--green)' }}>
                  Your price is {price > avgCompetitorPrice ? 'above' : 'below'} average
                </p>
              </div>

              <div className="space-y-3">
                {mockCompetitive.map((comp, i) => (
                  <div key={i} className="flex items-center justify-between p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                    <div>
                      <p className="text-sm font-medium">{comp.name}</p>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        {comp.subscribers} subs &middot; {comp.rating}/5.0
                      </p>
                    </div>
                    <span className="font-semibold text-sm">${comp.price}/mo</span>
                  </div>
                ))}
              </div>

              <div className="flex gap-3 mt-6">
                <button
                  className="px-4 py-2 rounded-lg text-sm"
                  style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                  onClick={() => setStep(2)}
                >
                  Back
                </button>
                <button
                  className="flex-1 py-2 rounded-lg text-sm font-medium"
                  style={{ background: 'var(--accent)', color: 'white' }}
                  onClick={() => setStep(4)}
                >
                  Next: Review
                </button>
              </div>
            </Card>
          </div>
        )}

        {/* Step 4: Review & Publish */}
        {step === 4 && (
          <div className="max-w-xl mx-auto">
            <Card>
              <div className="text-center mb-6">
                <div
                  className="w-16 h-16 rounded-2xl flex items-center justify-center text-2xl font-bold mx-auto mb-4"
                  style={{ background: typeColors[type]?.bg, color: typeColors[type]?.color }}
                >
                  {type[0]}
                </div>
                <h3 className="text-lg font-semibold">{selected?.name || 'Your Item'}</h3>
                <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>
                  {type} &middot; ${price}/month
                </p>
              </div>

              <div className="space-y-3 mb-6">
                <div className="flex justify-between p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                  <span className="text-sm" style={{ color: 'var(--text-muted)' }}>Your revenue/sub</span>
                  <span className="text-sm font-medium" style={{ color: 'var(--green)' }}>${(price * 0.85).toFixed(2)}/mo</span>
                </div>
                <div className="flex justify-between p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                  <span className="text-sm" style={{ color: 'var(--text-muted)' }}>Projected monthly</span>
                  <span className="text-sm font-medium" style={{ color: 'var(--green)' }}>${Math.round(projectedMonthly).toLocaleString()}</span>
                </div>
                <div className="flex justify-between p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                  <span className="text-sm" style={{ color: 'var(--text-muted)' }}>Projected annual</span>
                  <span className="text-sm font-medium" style={{ color: 'var(--accent)' }}>${Math.round(projectedAnnual).toLocaleString()}</span>
                </div>
              </div>

              <div className="flex gap-3">
                <button
                  className="px-4 py-2 rounded-lg text-sm"
                  style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                  onClick={() => setStep(3)}
                >
                  Back
                </button>
                <button
                  className="flex-1 py-3 rounded-lg text-sm font-bold transition-all"
                  style={{
                    background: 'var(--accent)',
                    color: 'white',
                    boxShadow: '0 4px 20px var(--accent-muted)',
                  }}
                  onClick={() => {
                    // Reset to show success state
                    setStep(1);
                    setSelectedItem(null);
                  }}
                >
                  Publish to Marketplace
                </button>
              </div>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
