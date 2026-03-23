'use client';

import { useState, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import MetricCard from '@/components/MetricCard';

interface Portfolio {
  id: string;
  name: string;
  creator: string;
  price: number;
  rating: number;
  reviews: number;
  subscribers: number;
  verification: 'backtest' | 'forward' | 'live';
  winRate: number;
  pf: number;
  dailyR: number;
  maxDD: number;
  alerts: number;
  propFirms: string[];
  free: boolean;
}

interface Strategy {
  id: string;
  name: string;
  creator: string;
  price: number;
  rating: number;
  reviews: number;
  verification: 'backtest' | 'forward' | 'live';
  winRate: number;
  pf: number;
  category: string;
}

interface Pack {
  id: string;
  name: string;
  creator: string;
  price: number;
  rating: number;
  reviews: number;
  category: string;
  indicators: number;
}

const mockPortfolios: Portfolio[] = [
  { id: 'p1', name: 'Momentum Alpha', creator: 'TradeMaster', price: 49, rating: 4.8, reviews: 127, subscribers: 342, verification: 'live', winRate: 58.2, pf: 2.31, dailyR: 1.85, maxDD: -2.1, alerts: 1847, propFirms: ['TTP', 'FTMO'], free: false },
  { id: 'p2', name: 'VWAP Scalper Pro', creator: 'QuantEdge', price: 29, rating: 4.5, reviews: 83, subscribers: 521, verification: 'forward', winRate: 54.0, pf: 1.92, dailyR: 1.45, maxDD: -2.8, alerts: 623, propFirms: ['TTP'], free: false },
  { id: 'p3', name: 'Steady Eddie', creator: 'RoR Team', price: 0, rating: 4.2, reviews: 45, subscribers: 1203, verification: 'live', winRate: 52.1, pf: 1.65, dailyR: 0.95, maxDD: -1.5, alerts: 2341, propFirms: ['TTP', 'FTMO', 'Topstep'], free: true },
  { id: 'p4', name: 'Tech Momentum', creator: 'AlgoTrader99', price: 39, rating: 4.6, reviews: 62, subscribers: 189, verification: 'forward', winRate: 56.3, pf: 2.08, dailyR: 1.72, maxDD: -3.1, alerts: 412, propFirms: ['FTMO'], free: false },
  { id: 'p5', name: 'Conservative Growth', creator: 'RoR Team', price: 0, rating: 4.0, reviews: 31, subscribers: 876, verification: 'live', winRate: 50.5, pf: 1.42, dailyR: 0.68, maxDD: -1.2, alerts: 3102, propFirms: ['TTP', 'Topstep'], free: true },
  { id: 'p6', name: 'Swing Master', creator: 'SwingKing', price: 79, rating: 4.9, reviews: 201, subscribers: 156, verification: 'live', winRate: 61.5, pf: 3.12, dailyR: 2.41, maxDD: -1.8, alerts: 987, propFirms: ['TTP', 'FTMO'], free: false },
];

const mockStrategies: Strategy[] = [
  { id: 's1', name: 'EMA Cross Scalper', creator: 'TradeMaster', price: 15, rating: 4.6, reviews: 42, verification: 'live', winRate: 55.8, pf: 2.1, category: 'Scalping' },
  { id: 's2', name: 'VWAP Bounce', creator: 'QuantEdge', price: 10, rating: 4.3, reviews: 28, verification: 'forward', winRate: 52.4, pf: 1.78, category: 'Intraday' },
  { id: 's3', name: 'MACD Trend Follow', creator: 'RoR Team', price: 0, rating: 4.1, reviews: 67, verification: 'live', winRate: 49.2, pf: 1.55, category: 'Trend' },
  { id: 's4', name: 'ATR Breakout', creator: 'BreakoutPro', price: 20, rating: 4.7, reviews: 35, verification: 'forward', winRate: 57.1, pf: 2.45, category: 'Breakout' },
];

const mockPacks: Pack[] = [
  { id: 'pk1', name: 'Advanced EMA Pack', creator: 'TradeMaster', price: 8, rating: 4.5, reviews: 31, category: 'Moving Averages', indicators: 6 },
  { id: 'pk2', name: 'Volume Profile Suite', creator: 'QuantEdge', price: 12, rating: 4.4, reviews: 19, category: 'Volume', indicators: 4 },
  { id: 'pk3', name: 'Momentum Essentials', creator: 'RoR Team', price: 0, rating: 4.2, reviews: 56, category: 'Momentum', indicators: 5 },
];

const verificationBadge: Record<string, { label: string; color: string; bg: string }> = {
  backtest: { label: 'Backtest Only', color: 'var(--text-muted)', bg: 'var(--bg-input)' },
  forward: { label: 'Forward Tested', color: 'var(--blue)', bg: 'var(--blue-muted)' },
  live: { label: 'Live Verified', color: 'var(--orange)', bg: 'var(--orange-muted)' },
};

type SortKey = 'performance' | 'subscribers' | 'newest' | 'price';
type AssetFilter = 'all' | 'stocks' | 'crypto';

function StarRating({ rating, count }: { rating: number; count: number }) {
  return (
    <div className="flex items-center gap-1">
      {[1, 2, 3, 4, 5].map((star) => (
        <span key={star} className="text-xs" style={{ color: star <= Math.round(rating) ? 'var(--orange)' : 'var(--text-muted)' }}>&#9733;</span>
      ))}
      <span className="text-xs ml-1" style={{ color: 'var(--text-muted)' }}>{rating.toFixed(1)} ({count})</span>
    </div>
  );
}

function MiniEquityCurve({ seed }: { seed: number }) {
  const points = useMemo(() => {
    const pts: number[] = [];
    let val = 50;
    for (let i = 0; i < 30; i++) {
      val += (Math.sin(seed * 0.7 + i * 0.3) * 3) + (Math.random() - 0.4) * 4;
      val = Math.max(10, Math.min(90, val));
      pts.push(val);
    }
    return pts;
  }, [seed]);

  const pathData = points.map((p, i) => `${(i / 29) * 100},${100 - p}`).join(' ');
  const isPositive = points[points.length - 1] > points[0];

  return (
    <svg viewBox="0 0 100 100" className="w-full h-12" preserveAspectRatio="none">
      <polyline
        points={pathData}
        fill="none"
        stroke={isPositive ? 'var(--green)' : 'var(--red)'}
        strokeWidth="2"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}

export default function MarketplaceV2() {
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState<SortKey>('performance');
  const [assetFilter, setAssetFilter] = useState<AssetFilter>('all');
  const [minWR, setMinWR] = useState(0);
  const [maxPrice, setMaxPrice] = useState(200);
  const [propFirmFilter, setPropFirmFilter] = useState('all');
  const [showFilters, setShowFilters] = useState(false);

  const freeRotation = mockPortfolios.filter((p) => p.free);

  function sortPortfolios(items: Portfolio[]): Portfolio[] {
    const sorted = [...items];
    switch (sortBy) {
      case 'performance': return sorted.sort((a, b) => b.pf - a.pf);
      case 'subscribers': return sorted.sort((a, b) => b.subscribers - a.subscribers);
      case 'newest': return sorted.reverse();
      case 'price': return sorted.sort((a, b) => a.price - b.price);
      default: return sorted;
    }
  }

  function filterPortfolios(items: Portfolio[]): Portfolio[] {
    let filtered = items;
    if (search) {
      filtered = filtered.filter((p) =>
        p.name.toLowerCase().includes(search.toLowerCase()) ||
        p.creator.toLowerCase().includes(search.toLowerCase())
      );
    }
    if (minWR > 0) filtered = filtered.filter((p) => p.winRate >= minWR);
    if (maxPrice < 200) filtered = filtered.filter((p) => p.price <= maxPrice);
    if (propFirmFilter !== 'all') filtered = filtered.filter((p) => p.propFirms.includes(propFirmFilter));
    return sortPortfolios(filtered);
  }

  const filteredPortfolios = filterPortfolios(mockPortfolios);

  return (
    <div>
      <PageHeader title="Marketplace" subtitle="Discover verified portfolios, strategies, and confluence packs" />

      {/* Summary metrics */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <MetricCard label="Total Portfolios" value={mockPortfolios.length.toString()} />
        <MetricCard label="Live Verified" value={mockPortfolios.filter((p) => p.verification === 'live').length.toString()} />
        <MetricCard label="Total Subscribers" value={mockPortfolios.reduce((s, p) => s + p.subscribers, 0).toLocaleString()} />
        <MetricCard label="Free Portfolios" value={freeRotation.length.toString()} />
      </div>

      {/* Search + Sort + Filters */}
      <div className="flex gap-3 mb-4">
        <input
          className="flex-1 px-4 py-2.5 rounded-lg text-sm"
          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
          placeholder="Search by name, creator..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <select
          className="px-3 py-2 rounded-lg text-sm"
          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as SortKey)}
        >
          <option value="performance">Best Performing</option>
          <option value="subscribers">Most Subscribers</option>
          <option value="newest">Newest</option>
          <option value="price">Price: Low to High</option>
        </select>
        <button
          className="px-4 py-2 rounded-lg text-sm"
          style={{ background: showFilters ? 'var(--accent)' : 'var(--bg-input)', color: showFilters ? 'white' : 'var(--text-secondary)', border: showFilters ? 'none' : '1px solid var(--border)' }}
          onClick={() => setShowFilters(!showFilters)}
        >
          Filters {showFilters ? '▲' : '▼'}
        </button>
      </div>

      {/* Advanced Filters */}
      {showFilters && (
        <Card className="mb-6">
          <div className="grid grid-cols-4 gap-4">
            <div>
              <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Asset Type</label>
              <select className="w-full px-3 py-2 rounded-lg text-sm" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} value={assetFilter} onChange={(e) => setAssetFilter(e.target.value as AssetFilter)}>
                <option value="all">All Assets</option>
                <option value="stocks">Stocks</option>
                <option value="crypto">Crypto</option>
              </select>
            </div>
            <div>
              <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Min Win Rate</label>
              <input type="range" min="0" max="70" value={minWR} onChange={(e) => setMinWR(Number(e.target.value))} className="w-full" />
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{minWR}%</span>
            </div>
            <div>
              <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Max Price</label>
              <input type="range" min="0" max="200" value={maxPrice} onChange={(e) => setMaxPrice(Number(e.target.value))} className="w-full" />
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>${maxPrice}/mo</span>
            </div>
            <div>
              <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Prop Firm Compatible</label>
              <select className="w-full px-3 py-2 rounded-lg text-sm" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} value={propFirmFilter} onChange={(e) => setPropFirmFilter(e.target.value)}>
                <option value="all">Any</option>
                <option value="TTP">Trade The Pool</option>
                <option value="FTMO">FTMO</option>
                <option value="Topstep">Topstep</option>
              </select>
            </div>
          </div>
        </Card>
      )}

      <TabBar tabs={['Portfolios', 'Strategies', 'Packs']}>
        {(tab) => (
          <div>
            {tab === 'Portfolios' && (
              <>
                {/* Free Rotation */}
                {freeRotation.length > 0 && (
                  <div className="mb-8">
                    <h2 className="text-lg font-semibold mb-3">Free Rotation</h2>
                    <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>Curated portfolios available for free — subscribe to lock in access</p>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      {freeRotation.map((p) => (
                        <Card key={p.id}>
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-xs font-medium px-2 py-0.5 rounded" style={{ background: 'var(--green-muted)', color: 'var(--green)' }}>FREE</span>
                            <span className="text-xs px-2 py-0.5 rounded" style={{ background: verificationBadge[p.verification].bg, color: verificationBadge[p.verification].color }}>{verificationBadge[p.verification].label}</span>
                          </div>
                          <h3 className="font-semibold text-sm mb-1">{p.name}</h3>
                          <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>by {p.creator}</p>
                          <MiniEquityCurve seed={parseInt(p.id.replace('p', ''), 10)} />
                          <div className="grid grid-cols-3 gap-2 mt-3 text-center">
                            <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>WR</p><p className="text-sm font-semibold">{p.winRate}%</p></div>
                            <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>PF</p><p className="text-sm font-semibold">{p.pf}</p></div>
                            <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>Daily R</p><p className="text-sm font-semibold">{p.dailyR}</p></div>
                          </div>
                          <button className="w-full mt-3 px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--green)', color: 'white' }}>Subscribe Free</button>
                        </Card>
                      ))}
                    </div>
                  </div>
                )}

                {/* All Portfolios */}
                <h2 className="text-lg font-semibold mb-3">All Portfolios</h2>
                <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>{filteredPortfolios.length} portfolios</p>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {filteredPortfolios.map((p) => (
                    <Card key={p.id} className="hover:border-[var(--accent)] transition-colors cursor-pointer">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs px-2 py-0.5 rounded" style={{ background: verificationBadge[p.verification].bg, color: verificationBadge[p.verification].color }}>{verificationBadge[p.verification].label}</span>
                        <StarRating rating={p.rating} count={p.reviews} />
                      </div>
                      <h3 className="font-semibold text-sm">{p.name}</h3>
                      <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>by <span style={{ color: 'var(--accent)' }}>{p.creator}</span></p>

                      <MiniEquityCurve seed={parseInt(p.id.replace('p', ''), 10)} />

                      <div className="grid grid-cols-4 gap-2 mt-3 text-center">
                        <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>WR</p><p className="text-sm font-semibold" style={{ color: p.winRate >= 55 ? 'var(--green)' : 'var(--text-primary)' }}>{p.winRate}%</p></div>
                        <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>PF</p><p className="text-sm font-semibold" style={{ color: p.pf >= 2 ? 'var(--green)' : 'var(--text-primary)' }}>{p.pf}</p></div>
                        <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>Daily R</p><p className="text-sm font-semibold">{p.dailyR}</p></div>
                        <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>Max DD</p><p className="text-sm font-semibold" style={{ color: 'var(--red)' }}>{p.maxDD}%</p></div>
                      </div>

                      <div className="flex items-center justify-between mt-3 pt-3 border-t" style={{ borderColor: 'var(--border)' }}>
                        <div className="flex items-center gap-2">
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{p.subscribers.toLocaleString()} subs</span>
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{p.alerts.toLocaleString()} alerts</span>
                        </div>
                        {p.free ? (
                          <button className="px-3 py-1.5 rounded-lg text-xs font-medium" style={{ background: 'var(--green)', color: 'white' }}>Subscribe Free</button>
                        ) : (
                          <button className="px-3 py-1.5 rounded-lg text-xs font-medium" style={{ background: 'var(--accent)', color: 'white' }}>Subscribe ${p.price}/mo</button>
                        )}
                      </div>

                      {p.propFirms.length > 0 && (
                        <div className="flex gap-1 mt-2">
                          {p.propFirms.map((firm) => (
                            <span key={firm} className="text-xs px-1.5 py-0.5 rounded" style={{ background: 'var(--purple)', color: 'white', fontSize: '10px' }}>{firm}</span>
                          ))}
                        </div>
                      )}
                    </Card>
                  ))}
                </div>
              </>
            )}

            {tab === 'Strategies' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {mockStrategies.map((s) => (
                  <Card key={s.id} className="hover:border-[var(--accent)] transition-colors cursor-pointer">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs px-2 py-0.5 rounded" style={{ background: verificationBadge[s.verification].bg, color: verificationBadge[s.verification].color }}>{verificationBadge[s.verification].label}</span>
                      <span className="text-xs px-2 py-0.5 rounded" style={{ background: 'var(--bg-input)', color: 'var(--text-secondary)' }}>{s.category}</span>
                    </div>
                    <h3 className="font-semibold text-sm">{s.name}</h3>
                    <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>by <span style={{ color: 'var(--accent)' }}>{s.creator}</span></p>
                    <div className="grid grid-cols-2 gap-3">
                      <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>Win Rate</p><p className="text-sm font-semibold">{s.winRate}%</p></div>
                      <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>Profit Factor</p><p className="text-sm font-semibold">{s.pf}</p></div>
                    </div>
                    <div className="flex items-center justify-between mt-3 pt-3 border-t" style={{ borderColor: 'var(--border)' }}>
                      <StarRating rating={s.rating} count={s.reviews} />
                      {s.price === 0 ? (
                        <span className="text-xs font-medium px-2 py-0.5 rounded" style={{ background: 'var(--green-muted)', color: 'var(--green)' }}>FREE</span>
                      ) : (
                        <span className="text-sm font-semibold">${s.price}/mo</span>
                      )}
                    </div>
                  </Card>
                ))}
              </div>
            )}

            {tab === 'Packs' && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {mockPacks.map((pk) => (
                  <Card key={pk.id} className="hover:border-[var(--accent)] transition-colors cursor-pointer">
                    <span className="text-xs px-2 py-0.5 rounded mb-2 inline-block" style={{ background: 'var(--bg-input)', color: 'var(--text-secondary)' }}>{pk.category}</span>
                    <h3 className="font-semibold text-sm">{pk.name}</h3>
                    <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>by {pk.creator} &middot; {pk.indicators} indicators</p>
                    <div className="flex items-center justify-between">
                      <StarRating rating={pk.rating} count={pk.reviews} />
                      {pk.price === 0 ? (
                        <span className="text-xs font-medium px-2 py-0.5 rounded" style={{ background: 'var(--green-muted)', color: 'var(--green)' }}>FREE</span>
                      ) : (
                        <span className="text-sm font-semibold">${pk.price}/mo</span>
                      )}
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
