'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import ChartPlaceholder from '@/components/ChartPlaceholder';
import TabBar from '@/components/TabBar';

type PublishType = 'Portfolio' | 'Strategy' | 'Pack';

const mockUnpublished = {
  Portfolio: [
    { id: 'p1', name: 'Scalp Machine', strategies: 4, winRate: 62.3, profitFactor: 2.1, maxDD: -8.2, verificationLevel: 'Forward Tested' as const },
    { id: 'p2', name: 'Swing Daily', strategies: 3, winRate: 58.1, profitFactor: 1.7, maxDD: -12.5, verificationLevel: 'Backtest Only' as const },
  ],
  Strategy: [
    { id: 's1', name: 'RSI Reversal', winRate: 55.7, profitFactor: 1.8, maxDD: -6.1, verificationLevel: 'Live Verified' as const },
    { id: 's2', name: 'Breakout Rider', winRate: 51.2, profitFactor: 2.1, maxDD: -9.4, verificationLevel: 'Forward Tested' as const },
    { id: 's3', name: 'MACD Cross Daily', winRate: 60.4, profitFactor: 1.6, maxDD: -7.8, verificationLevel: 'Backtest Only' as const },
  ],
  Pack: [
    { id: 'pk1', name: 'Volume Profile Pack', indicators: 3, outputs: 8, verificationLevel: 'N/A' as const },
    { id: 'pk2', name: 'Trend Strength Pack', indicators: 4, outputs: 12, verificationLevel: 'N/A' as const },
  ],
};

const alreadyPublished = [
  { id: '1', name: 'Momentum Alpha', type: 'Portfolio' as const, price: 30, subscribers: 156, rating: 4.8 },
  { id: '2', name: 'VWAP Scalper Pro', type: 'Portfolio' as const, price: 25, subscribers: 89, rating: 4.5 },
  { id: '3', name: 'EMA Momentum', type: 'Strategy' as const, price: 5, subscribers: 45, rating: 4.3 },
  { id: '4', name: 'RSI Zones Pack', type: 'Pack' as const, price: 8, subscribers: 52, rating: 4.7 },
];

const propFirmTags = ['FTMO', 'TTP', 'Topstep', 'MFF', 'E8'];
const categoryTags = ['Scalping', 'Swing', 'Day Trading', 'Momentum', 'Mean Reversion', 'Trend Following'];

const typeColors: Record<string, { color: string; bg: string }> = {
  Portfolio: { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  Strategy: { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  Pack: { color: 'var(--purple)', bg: 'var(--purple-muted)' },
};

const verificationColors: Record<string, { color: string; bg: string }> = {
  'Backtest Only': { color: 'var(--text-muted)', bg: 'var(--bg-input)' },
  'Forward Tested': { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  'Live Verified': { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  'N/A': { color: 'var(--text-muted)', bg: 'var(--bg-input)' },
};

const typeDescriptions: Record<PublishType, string> = {
  Portfolio: 'Curated collection of strategies with diversification and risk management. Subscribers get all alerts.',
  Strategy: 'Individual trading strategy. Portfolio builders can include it in their portfolios.',
  Pack: 'Technical indicator pack with interpreters and triggers. Strategy builders can use it.',
};

export default function CreatorPublishV2() {
  const [step, setStep] = useState(1);
  const [selectedType, setSelectedType] = useState<PublishType | null>(null);
  const [selectedItem, setSelectedItem] = useState<string | null>(null);
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [price, setPrice] = useState('30');
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  const [selectedPropFirms, setSelectedPropFirms] = useState<string[]>([]);
  const [acceptedTerms, setAcceptedTerms] = useState(false);

  function toggleTag(tag: string, list: string[], setter: (v: string[]) => void) {
    setter(list.includes(tag) ? list.filter((t) => t !== tag) : [...list, tag]);
  }

  const items = selectedType ? (mockUnpublished[selectedType] || []) : [];
  const selected = items.find((i) => i.id === selectedItem);

  const stepLabels = ['Type', 'Select', 'Configure', 'Verification', 'Review'];

  return (
    <div>
      <PageHeader
        title="Publish to Marketplace"
        subtitle="List your content for subscribers to discover"
        backHref="/creator/dashboard"
      />

      <TabBar tabs={['New Listing', 'Published Items']}>
        {(tab) => (
          <div>
            {tab === 'New Listing' && (
              <div>
                {/* Step indicator */}
                <div className="flex items-center gap-2 mb-8">
                  {stepLabels.map((label, i) => {
                    const stepNum = i + 1;
                    const isActive = step === stepNum;
                    const isComplete = step > stepNum;
                    return (
                      <div key={label} className="flex items-center gap-2">
                        <div className="flex items-center gap-2">
                          <div
                            className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold"
                            style={{
                              background: isActive ? 'var(--accent)' : isComplete ? 'var(--green)' : 'var(--bg-input)',
                              color: isActive || isComplete ? 'white' : 'var(--text-muted)',
                            }}
                          >
                            {isComplete ? '!' : stepNum}
                          </div>
                          <span
                            className="text-xs font-medium"
                            style={{ color: isActive ? 'var(--accent)' : isComplete ? 'var(--green)' : 'var(--text-muted)' }}
                          >
                            {label}
                          </span>
                        </div>
                        {i < stepLabels.length - 1 && (
                          <div className="w-8 h-px" style={{ background: isComplete ? 'var(--green)' : 'var(--border)' }} />
                        )}
                      </div>
                    );
                  })}
                </div>

                {/* Step 1: Choose type */}
                {step === 1 && (
                  <div className="grid grid-cols-3 gap-4">
                    {(['Portfolio', 'Strategy', 'Pack'] as PublishType[]).map((type) => (
                      <Card key={type}>
                        <div
                          className="cursor-pointer p-2"
                          onClick={() => { setSelectedType(type); setStep(2); }}
                        >
                          <div
                            className="w-12 h-12 rounded-xl flex items-center justify-center text-xl font-bold mb-3"
                            style={{ background: typeColors[type]?.bg, color: typeColors[type]?.color }}
                          >
                            {type[0]}
                          </div>
                          <h3 className="font-semibold text-sm mb-2">{type}</h3>
                          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                            {typeDescriptions[type]}
                          </p>
                        </div>
                      </Card>
                    ))}
                  </div>
                )}

                {/* Step 2: Select item */}
                {step === 2 && selectedType && (
                  <div>
                    <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
                      Select a {selectedType} to Publish
                    </h3>
                    <div className="space-y-3">
                      {items.map((item) => (
                        <Card key={item.id}>
                          <div
                            className="flex items-center gap-4 cursor-pointer"
                            onClick={() => { setSelectedItem(item.id); setTitle(item.name); setStep(3); }}
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
                                {'winRate' in item && `WR: ${item.winRate}%`}
                                {'profitFactor' in item && ` | PF: ${item.profitFactor}`}
                                {'maxDD' in item && ` | MaxDD: ${item.maxDD}%`}
                                {'strategies' in item && `${item.strategies} strategies`}
                                {'indicators' in item && `${item.indicators} indicators, ${item.outputs} outputs`}
                              </p>
                            </div>
                            {'verificationLevel' in item && (
                              <span
                                className="text-xs px-2 py-0.5 rounded"
                                style={{
                                  color: verificationColors[item.verificationLevel as string]?.color,
                                  background: verificationColors[item.verificationLevel as string]?.bg,
                                }}
                              >
                                {item.verificationLevel}
                              </span>
                            )}
                          </div>
                        </Card>
                      ))}
                    </div>
                    <button
                      className="mt-4 px-4 py-2 rounded-lg text-sm"
                      style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                      onClick={() => setStep(1)}
                    >
                      Back
                    </button>
                  </div>
                )}

                {/* Step 3: Configure listing */}
                {step === 3 && (
                  <div className="max-w-2xl">
                    <Card>
                      <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
                        Configure Listing
                      </h3>
                      <div className="space-y-5">
                        <div>
                          <label className="text-sm mb-1 block" style={{ color: 'var(--text-secondary)' }}>Title</label>
                          <input
                            className="w-full px-3 py-2 rounded-lg text-sm"
                            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                            value={title}
                            onChange={(e) => setTitle(e.target.value)}
                          />
                        </div>
                        <div>
                          <label className="text-sm mb-1 block" style={{ color: 'var(--text-secondary)' }}>Description</label>
                          <textarea
                            className="w-full px-3 py-2 rounded-lg text-sm"
                            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', minHeight: '100px' }}
                            value={description}
                            onChange={(e) => setDescription(e.target.value)}
                            placeholder="Describe your strategy's edge, ideal market conditions, risk management approach..."
                          />
                        </div>
                        <div>
                          <label className="text-sm mb-1 block" style={{ color: 'var(--text-secondary)' }}>Category Tags</label>
                          <div className="flex flex-wrap gap-2">
                            {categoryTags.map((tag) => (
                              <button
                                key={tag}
                                className="px-3 py-1 rounded-full text-xs transition-colors"
                                style={{
                                  background: selectedCategories.includes(tag) ? 'var(--accent-muted)' : 'var(--bg-input)',
                                  color: selectedCategories.includes(tag) ? 'var(--accent)' : 'var(--text-muted)',
                                  border: `1px solid ${selectedCategories.includes(tag) ? 'var(--accent)' : 'var(--border)'}`,
                                }}
                                onClick={() => toggleTag(tag, selectedCategories, setSelectedCategories)}
                              >
                                {tag}
                              </button>
                            ))}
                          </div>
                        </div>
                        <div>
                          <label className="text-sm mb-1 block" style={{ color: 'var(--text-secondary)' }}>Prop Firm Compatibility</label>
                          <div className="flex flex-wrap gap-2">
                            {propFirmTags.map((tag) => (
                              <button
                                key={tag}
                                className="px-3 py-1 rounded-full text-xs transition-colors"
                                style={{
                                  background: selectedPropFirms.includes(tag) ? 'var(--green-muted)' : 'var(--bg-input)',
                                  color: selectedPropFirms.includes(tag) ? 'var(--green)' : 'var(--text-muted)',
                                  border: `1px solid ${selectedPropFirms.includes(tag) ? 'var(--green)' : 'var(--border)'}`,
                                }}
                                onClick={() => toggleTag(tag, selectedPropFirms, setSelectedPropFirms)}
                              >
                                {tag}
                              </button>
                            ))}
                          </div>
                        </div>
                        <div>
                          <label className="text-sm mb-1 block" style={{ color: 'var(--text-secondary)' }}>Monthly Price</label>
                          <div className="flex items-center gap-2">
                            <span className="text-sm" style={{ color: 'var(--text-muted)' }}>$</span>
                            <input
                              type="number"
                              className="w-32 px-3 py-2 rounded-lg text-sm"
                              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                              value={price}
                              onChange={(e) => setPrice(e.target.value)}
                            />
                            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>/month</span>
                          </div>
                          {price && (
                            <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                              Platform fee: ${(parseFloat(price) * 0.15).toFixed(2)} | Net to you: ${(parseFloat(price) * 0.85).toFixed(2)}/subscriber
                            </p>
                          )}
                        </div>
                        <div>
                          <label className="text-sm mb-1 block" style={{ color: 'var(--text-secondary)' }}>Preview Chart</label>
                          <ChartPlaceholder label="Equity curve or performance chart preview" height={200} />
                        </div>
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
                          className="px-4 py-2 rounded-lg text-sm font-medium"
                          style={{ background: 'var(--accent)', color: 'white' }}
                          onClick={() => setStep(4)}
                        >
                          Next: Verification
                        </button>
                      </div>
                    </Card>
                  </div>
                )}

                {/* Step 4: Verification */}
                {step === 4 && (
                  <div className="max-w-2xl">
                    <Card>
                      <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
                        Verification Status
                      </h3>
                      {selected && 'verificationLevel' in selected && (
                        <div
                          className="p-4 rounded-lg mb-4"
                          style={{ background: 'var(--bg-input)' }}
                        >
                          <div className="flex items-center gap-3 mb-2">
                            <span
                              className="text-sm px-3 py-1 rounded font-medium"
                              style={{
                                color: verificationColors[selected.verificationLevel as string]?.color,
                                background: verificationColors[selected.verificationLevel as string]?.bg,
                              }}
                            >
                              {selected.verificationLevel}
                            </span>
                          </div>
                          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                            {selected.verificationLevel === 'Backtest Only' && 'This item has only been backtested. It can be listed at a lower price tier ($5-15/mo). Forward test for 3+ months to unlock higher pricing.'}
                            {selected.verificationLevel === 'Forward Tested' && 'This item has been forward-tested for 3+ months. It qualifies for mid-tier pricing ($15-50/mo). Continue live trading to reach Live Verified status.'}
                            {selected.verificationLevel === 'Live Verified' && 'This item has been live-verified with 6+ months of real alerts. It qualifies for premium pricing ($50-200/mo).'}
                            {selected.verificationLevel === 'N/A' && 'Verification is not applicable for packs.'}
                          </p>
                        </div>
                      )}
                      <div className="space-y-3 mb-6">
                        {[
                          { level: 'Backtest Only', desc: '$5-15/mo tier', icon: 'B' },
                          { level: 'Forward Tested', desc: '$15-50/mo tier (3+ months)', icon: 'F' },
                          { level: 'Live Verified', desc: '$50-200/mo tier (6+ months, 500+ alerts)', icon: 'L' },
                        ].map((v) => (
                          <div key={v.level} className="flex items-center gap-3 p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                            <div
                              className="w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold"
                              style={{
                                color: verificationColors[v.level]?.color,
                                background: verificationColors[v.level]?.bg,
                              }}
                            >
                              {v.icon}
                            </div>
                            <div>
                              <p className="text-sm font-medium">{v.level}</p>
                              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{v.desc}</p>
                            </div>
                          </div>
                        ))}
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
                          className="px-4 py-2 rounded-lg text-sm font-medium"
                          style={{ background: 'var(--accent)', color: 'white' }}
                          onClick={() => setStep(5)}
                        >
                          Next: Review
                        </button>
                      </div>
                    </Card>
                  </div>
                )}

                {/* Step 5: Review & Publish */}
                {step === 5 && (
                  <div className="max-w-2xl">
                    <Card>
                      <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
                        Review & Publish
                      </h3>
                      <div className="space-y-3 mb-6">
                        <div className="flex justify-between p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                          <span className="text-sm" style={{ color: 'var(--text-muted)' }}>Type</span>
                          <span className="text-sm font-medium">{selectedType}</span>
                        </div>
                        <div className="flex justify-between p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                          <span className="text-sm" style={{ color: 'var(--text-muted)' }}>Title</span>
                          <span className="text-sm font-medium">{title || 'Untitled'}</span>
                        </div>
                        <div className="flex justify-between p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                          <span className="text-sm" style={{ color: 'var(--text-muted)' }}>Price</span>
                          <span className="text-sm font-medium">${price}/month</span>
                        </div>
                        <div className="flex justify-between p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                          <span className="text-sm" style={{ color: 'var(--text-muted)' }}>Categories</span>
                          <span className="text-sm">{selectedCategories.join(', ') || 'None'}</span>
                        </div>
                        <div className="flex justify-between p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                          <span className="text-sm" style={{ color: 'var(--text-muted)' }}>Prop Firms</span>
                          <span className="text-sm">{selectedPropFirms.join(', ') || 'None'}</span>
                        </div>
                      </div>

                      {/* Terms */}
                      <div
                        className="flex items-start gap-3 p-4 rounded-lg mb-6"
                        style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
                      >
                        <button
                          className="w-5 h-5 rounded border-2 flex-shrink-0 mt-0.5 flex items-center justify-center text-xs"
                          style={{
                            borderColor: acceptedTerms ? 'var(--accent)' : 'var(--text-muted)',
                            background: acceptedTerms ? 'var(--accent)' : 'transparent',
                            color: acceptedTerms ? 'white' : 'transparent',
                          }}
                          onClick={() => setAcceptedTerms(!acceptedTerms)}
                        >
                          {acceptedTerms ? '!' : ''}
                        </button>
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                          I agree to the Marketplace Terms of Service. I understand that RoR Trader is a signal service platform, not investment management. Past performance does not guarantee future results.
                        </p>
                      </div>

                      <div className="flex gap-3">
                        <button
                          className="px-4 py-2 rounded-lg text-sm"
                          style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                          onClick={() => setStep(4)}
                        >
                          Back
                        </button>
                        <button
                          className="flex-1 py-2.5 rounded-lg text-sm font-medium"
                          style={{
                            background: acceptedTerms ? 'var(--accent)' : 'var(--bg-input)',
                            color: acceptedTerms ? 'white' : 'var(--text-muted)',
                            cursor: acceptedTerms ? 'pointer' : 'not-allowed',
                          }}
                          disabled={!acceptedTerms}
                        >
                          Publish to Marketplace
                        </button>
                      </div>
                    </Card>
                  </div>
                )}
              </div>
            )}

            {tab === 'Published Items' && (
              <div className="space-y-3">
                {alreadyPublished.map((item) => (
                  <Card key={item.id}>
                    <div className="flex items-center justify-between">
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
                            ${item.price}/mo &middot; {item.subscribers} subs &middot; {item.rating}/5.0
                          </p>
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <button
                          className="px-3 py-1.5 rounded text-xs"
                          style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                        >
                          Edit Listing
                        </button>
                        <button
                          className="px-3 py-1.5 rounded text-xs"
                          style={{ background: 'var(--red-muted)', color: 'var(--red)' }}
                        >
                          Unpublish
                        </button>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            )}
          </div>
        )}
      </TabBar>
    </div>
  );
}
