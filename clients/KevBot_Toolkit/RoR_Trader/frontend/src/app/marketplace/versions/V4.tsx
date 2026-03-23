'use client';

import { useState, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import ChartPlaceholder from '@/components/ChartPlaceholder';
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
  creatorBio?: string;
  monthsActive?: number;
}

const mockPortfolios: Portfolio[] = [
  { id: 'p1', name: 'Momentum Alpha', creator: 'TradeMaster', price: 49, rating: 4.8, reviews: 127, subscribers: 342, verification: 'live', winRate: 58.2, pf: 2.31, dailyR: 1.85, maxDD: -2.1, alerts: 1847, propFirms: ['TTP', 'FTMO'], free: false, creatorBio: 'Full-time prop trader, 5+ years experience', monthsActive: 14 },
  { id: 'p2', name: 'VWAP Scalper Pro', creator: 'QuantEdge', price: 29, rating: 4.5, reviews: 83, subscribers: 521, verification: 'forward', winRate: 54.0, pf: 1.92, dailyR: 1.45, maxDD: -2.8, alerts: 623, propFirms: ['TTP'], free: false, creatorBio: 'Quantitative analyst turned retail trader', monthsActive: 8 },
  { id: 'p3', name: 'Steady Eddie', creator: 'RoR Team', price: 0, rating: 4.2, reviews: 45, subscribers: 1203, verification: 'live', winRate: 52.1, pf: 1.65, dailyR: 0.95, maxDD: -1.5, alerts: 2341, propFirms: ['TTP', 'FTMO', 'Topstep'], free: true, creatorBio: 'Official RoR team portfolio', monthsActive: 18 },
  { id: 'p4', name: 'Tech Momentum', creator: 'AlgoTrader99', price: 39, rating: 4.6, reviews: 62, subscribers: 189, verification: 'forward', winRate: 56.3, pf: 2.08, dailyR: 1.72, maxDD: -3.1, alerts: 412, propFirms: ['FTMO'], free: false, creatorBio: 'Software engineer building systematic strategies', monthsActive: 6 },
  { id: 'p5', name: 'Conservative Growth', creator: 'RoR Team', price: 0, rating: 4.0, reviews: 31, subscribers: 876, verification: 'live', winRate: 50.5, pf: 1.42, dailyR: 0.68, maxDD: -1.2, alerts: 3102, propFirms: ['TTP', 'Topstep'], free: true, creatorBio: 'Official RoR team portfolio', monthsActive: 18 },
  { id: 'p6', name: 'Swing Master', creator: 'SwingKing', price: 79, rating: 4.9, reviews: 201, subscribers: 156, verification: 'live', winRate: 61.5, pf: 3.12, dailyR: 2.41, maxDD: -1.8, alerts: 987, propFirms: ['TTP', 'FTMO'], free: false, creatorBio: 'Swing trader focused on clean setups', monthsActive: 11 },
];

const risingStars: Portfolio[] = [
  { id: 'r1', name: 'Crypto Momentum', creator: 'CryptoAlpha', price: 25, rating: 4.7, reviews: 12, subscribers: 47, verification: 'forward', winRate: 59.8, pf: 2.65, dailyR: 2.1, maxDD: -3.5, alerts: 89, propFirms: [], free: false, monthsActive: 2 },
  { id: 'r2', name: 'Small Cap Hunter', creator: 'MicroCap', price: 35, rating: 4.4, reviews: 8, subscribers: 23, verification: 'forward', winRate: 57.2, pf: 2.38, dailyR: 1.95, maxDD: -4.2, alerts: 56, propFirms: ['TTP'], free: false, monthsActive: 1 },
];

const verificationBadge: Record<string, { label: string; color: string; bg: string }> = {
  backtest: { label: 'Backtest Only', color: 'var(--text-muted)', bg: 'var(--bg-input)' },
  forward: { label: 'Forward Tested', color: 'var(--blue)', bg: 'var(--blue-muted)' },
  live: { label: 'Live Verified', color: 'var(--orange)', bg: 'var(--orange-muted)' },
};

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

function MiniEquityCurve({ seed, height = 60 }: { seed: number; height?: number }) {
  const points = useMemo(() => {
    const pts: number[] = [];
    let val = 50;
    for (let i = 0; i < 40; i++) {
      val += (Math.sin(seed * 0.7 + i * 0.3) * 3) + (Math.random() - 0.4) * 4;
      val = Math.max(10, Math.min(90, val));
      pts.push(val);
    }
    return pts;
  }, [seed]);

  const pathData = points.map((p, i) => `${(i / 39) * 100},${100 - p}`).join(' ');
  const isPositive = points[points.length - 1] > points[0];

  return (
    <svg viewBox="0 0 100 100" style={{ width: '100%', height }} preserveAspectRatio="none">
      <polyline points={pathData} fill="none" stroke={isPositive ? 'var(--green)' : 'var(--red)'} strokeWidth="2" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function ScatterPoint({ x, y, size, label, color }: { x: number; y: number; size: number; label: string; color: string }) {
  return (
    <g>
      <circle cx={x} cy={y} r={size} fill={color} opacity={0.7} />
      <text x={x} y={y - size - 4} textAnchor="middle" fill="var(--text-secondary)" fontSize="8">{label}</text>
    </g>
  );
}

export default function MarketplaceV4() {
  const [selectedCreator, setSelectedCreator] = useState<string | null>(null);

  const top3 = [...mockPortfolios].sort((a, b) => b.pf - a.pf).slice(0, 3);
  const leaderboard = [...mockPortfolios].sort((a, b) => b.dailyR - a.dailyR);

  const creators = useMemo(() => {
    const map = new Map<string, { name: string; portfolios: number; totalSubs: number; avgRating: number; bio?: string }>();
    mockPortfolios.forEach((p) => {
      const existing = map.get(p.creator);
      if (existing) {
        existing.portfolios++;
        existing.totalSubs += p.subscribers;
        existing.avgRating = (existing.avgRating + p.rating) / 2;
      } else {
        map.set(p.creator, { name: p.creator, portfolios: 1, totalSubs: p.subscribers, avgRating: p.rating, bio: p.creatorBio });
      }
    });
    return Array.from(map.values());
  }, []);

  const recommended = mockPortfolios.filter((p) => p.winRate > 55 && p.pf > 2).slice(0, 3);

  return (
    <div>
      <PageHeader title="Marketplace" subtitle="Discover, compare, and subscribe to the best trading portfolios" />

      {/* Hero — Top 3 Portfolios */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold mb-1">Featured Portfolios</h2>
        <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>Top performing portfolios on the platform</p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {top3.map((p, idx) => (
            <Card key={p.id} className={idx === 0 ? 'ring-2 ring-[var(--orange)]' : ''}>
              <div className="flex items-center justify-between mb-3">
                <span className="text-2xl font-bold" style={{ color: idx === 0 ? 'var(--orange)' : idx === 1 ? 'var(--text-secondary)' : 'var(--orange)' }}>
                  #{idx + 1}
                </span>
                <span className="text-xs px-2 py-0.5 rounded" style={{ background: verificationBadge[p.verification].bg, color: verificationBadge[p.verification].color }}>{verificationBadge[p.verification].label}</span>
              </div>
              <h3 className="text-lg font-bold mb-1">{p.name}</h3>
              <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>by {p.creator}</p>
              <MiniEquityCurve seed={parseInt(p.id.replace('p', ''), 10)} height={80} />
              <div className="grid grid-cols-4 gap-2 mt-4 text-center">
                <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>WR</p><p className="text-lg font-bold" style={{ color: 'var(--green)' }}>{p.winRate}%</p></div>
                <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>PF</p><p className="text-lg font-bold" style={{ color: 'var(--green)' }}>{p.pf}</p></div>
                <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>Daily R</p><p className="text-lg font-bold">{p.dailyR}</p></div>
                <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>Max DD</p><p className="text-lg font-bold" style={{ color: 'var(--red)' }}>{p.maxDD}%</p></div>
              </div>
              <div className="flex items-center justify-between mt-4 pt-3 border-t" style={{ borderColor: 'var(--border)' }}>
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{p.subscribers.toLocaleString()} subscribers</span>
                {p.free ? (
                  <button className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--green)', color: 'white' }}>Subscribe Free</button>
                ) : (
                  <button className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: 'white' }}>Subscribe ${p.price}/mo</button>
                )}
              </div>
            </Card>
          ))}
        </div>
      </div>

      {/* Rising Stars */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold mb-1">Rising Stars</h2>
        <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>New portfolios showing strong early performance</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {risingStars.map((p) => (
            <Card key={p.id}>
              <div className="flex items-start gap-4">
                <div className="w-2 h-full rounded-full flex-shrink-0" style={{ background: 'var(--green)', minHeight: '80px' }} />
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <h3 className="font-semibold text-sm">{p.name}</h3>
                    <span className="text-xs px-2 py-0.5 rounded" style={{ background: 'var(--green-muted)', color: 'var(--green)' }}>NEW &middot; {p.monthsActive}mo</span>
                  </div>
                  <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>by {p.creator}</p>
                  <div className="grid grid-cols-4 gap-3 text-center">
                    <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>WR</p><p className="text-sm font-semibold" style={{ color: 'var(--green)' }}>{p.winRate}%</p></div>
                    <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>PF</p><p className="text-sm font-semibold" style={{ color: 'var(--green)' }}>{p.pf}</p></div>
                    <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>Daily R</p><p className="text-sm font-semibold">{p.dailyR}</p></div>
                    <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>Subs</p><p className="text-sm font-semibold">{p.subscribers}</p></div>
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>

      {/* Performance Leaderboard */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold mb-4">Performance Leaderboard</h2>
        <Card>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b" style={{ borderColor: 'var(--border)' }}>
                  <th className="text-left py-2 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Rank</th>
                  <th className="text-left py-2 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Portfolio</th>
                  <th className="text-left py-2 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Creator</th>
                  <th className="text-center py-2 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Verification</th>
                  <th className="text-right py-2 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>WR</th>
                  <th className="text-right py-2 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>PF</th>
                  <th className="text-right py-2 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Daily R</th>
                  <th className="text-right py-2 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Max DD</th>
                  <th className="text-right py-2 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Alerts</th>
                  <th className="text-right py-2 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Subs</th>
                </tr>
              </thead>
              <tbody>
                {leaderboard.map((p, idx) => (
                  <tr key={p.id} className="border-b hover:bg-[var(--bg-input)] transition-colors cursor-pointer" style={{ borderColor: 'var(--border)' }}>
                    <td className="py-3 px-3 font-bold" style={{ color: idx < 3 ? 'var(--orange)' : 'var(--text-muted)' }}>#{idx + 1}</td>
                    <td className="py-3 px-3 font-medium">{p.name}</td>
                    <td className="py-3 px-3" style={{ color: 'var(--accent)' }}>{p.creator}</td>
                    <td className="py-3 px-3 text-center"><span className="text-xs px-2 py-0.5 rounded" style={{ background: verificationBadge[p.verification].bg, color: verificationBadge[p.verification].color }}>{verificationBadge[p.verification].label}</span></td>
                    <td className="py-3 px-3 text-right" style={{ color: p.winRate >= 55 ? 'var(--green)' : 'var(--text-primary)' }}>{p.winRate}%</td>
                    <td className="py-3 px-3 text-right" style={{ color: p.pf >= 2 ? 'var(--green)' : 'var(--text-primary)' }}>{p.pf}</td>
                    <td className="py-3 px-3 text-right font-semibold">{p.dailyR}</td>
                    <td className="py-3 px-3 text-right" style={{ color: 'var(--red)' }}>{p.maxDD}%</td>
                    <td className="py-3 px-3 text-right" style={{ color: 'var(--text-secondary)' }}>{p.alerts.toLocaleString()}</td>
                    <td className="py-3 px-3 text-right" style={{ color: 'var(--text-secondary)' }}>{p.subscribers.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      {/* Risk vs Return Scatter Plot */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold mb-4">Risk vs Return</h2>
        <Card>
          <div className="relative">
            <svg viewBox="0 0 400 250" className="w-full" style={{ height: '300px' }}>
              {/* Grid lines */}
              {[0, 1, 2, 3, 4].map((i) => (
                <line key={`h${i}`} x1="50" y1={50 + i * 40} x2="380" y2={50 + i * 40} stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4" />
              ))}
              {[0, 1, 2, 3, 4, 5].map((i) => (
                <line key={`v${i}`} x1={50 + i * 66} y1="40" x2={50 + i * 66} y2="220" stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4" />
              ))}
              {/* Axis labels */}
              <text x="215" y="245" textAnchor="middle" fill="var(--text-muted)" fontSize="10">Max Drawdown (Risk) →</text>
              <text x="15" y="130" textAnchor="middle" fill="var(--text-muted)" fontSize="10" transform="rotate(-90, 15, 130)">Daily R (Return) →</text>
              {/* Data points */}
              {mockPortfolios.map((p) => {
                const x = 50 + (Math.abs(p.maxDD) / 5) * 330;
                const y = 210 - (p.dailyR / 3) * 170;
                const size = Math.max(4, Math.min(15, p.subscribers / 100));
                const color = p.verification === 'live' ? 'var(--orange)' : p.verification === 'forward' ? 'var(--blue)' : 'var(--text-muted)';
                return <ScatterPoint key={p.id} x={x} y={y} size={size} label={p.name.split(' ')[0]} color={color} />;
              })}
            </svg>
            <div className="flex items-center gap-4 mt-2 justify-center">
              <div className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full inline-block" style={{ background: 'var(--orange)' }} /><span className="text-xs" style={{ color: 'var(--text-muted)' }}>Live Verified</span></div>
              <div className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full inline-block" style={{ background: 'var(--blue)' }} /><span className="text-xs" style={{ color: 'var(--text-muted)' }}>Forward Tested</span></div>
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Size = subscriber count</span>
            </div>
          </div>
        </Card>
      </div>

      {/* Creator Spotlight */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold mb-4">Creator Spotlight</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {creators.map((c) => (
            <div key={c.name} onClick={() => setSelectedCreator(selectedCreator === c.name ? null : c.name)} className="cursor-pointer">
            <Card className="hover:border-[var(--accent)] transition-colors">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                  {c.name.charAt(0)}
                </div>
                <div>
                  <h3 className="font-semibold text-sm">{c.name}</h3>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{c.portfolios} portfolio{c.portfolios > 1 ? 's' : ''}</p>
                </div>
              </div>
              {c.bio && <p className="text-xs mb-3" style={{ color: 'var(--text-secondary)' }}>{c.bio}</p>}
              <div className="grid grid-cols-2 gap-2 text-center">
                <div className="rounded-lg p-2" style={{ background: 'var(--bg-input)' }}>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Subscribers</p>
                  <p className="text-sm font-semibold">{c.totalSubs.toLocaleString()}</p>
                </div>
                <div className="rounded-lg p-2" style={{ background: 'var(--bg-input)' }}>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Avg Rating</p>
                  <p className="text-sm font-semibold"><span style={{ color: 'var(--orange)' }}>&#9733;</span> {c.avgRating.toFixed(1)}</p>
                </div>
              </div>
              {selectedCreator === c.name && (
                <div className="mt-3 pt-3 border-t" style={{ borderColor: 'var(--border)' }}>
                  <p className="text-xs font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Portfolios by {c.name}:</p>
                  {mockPortfolios.filter((p) => p.creator === c.name).map((p) => (
                    <div key={p.id} className="flex items-center justify-between py-1">
                      <span className="text-xs">{p.name}</span>
                      <span className="text-xs" style={{ color: 'var(--green)' }}>PF {p.pf}</span>
                    </div>
                  ))}
                </div>
              )}
            </Card>
            </div>
          ))}
        </div>
      </div>

      {/* Recommended for You */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold mb-1">Recommended for You</h2>
        <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>Based on portfolios with strong risk-adjusted returns</p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {recommended.map((p) => (
            <Card key={p.id}>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs px-2 py-0.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>Recommended</span>
                <StarRating rating={p.rating} count={p.reviews} />
              </div>
              <h3 className="font-semibold text-sm">{p.name}</h3>
              <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>by {p.creator}</p>
              <MiniEquityCurve seed={parseInt(p.id.replace('p', ''), 10)} />
              <div className="flex items-center justify-between mt-3">
                <div className="flex gap-3 text-xs" style={{ color: 'var(--text-secondary)' }}>
                  <span>WR {p.winRate}%</span>
                  <span>PF {p.pf}</span>
                </div>
                <button className="px-3 py-1.5 rounded-lg text-xs font-medium" style={{ background: 'var(--accent)', color: 'white' }}>
                  {p.free ? 'Subscribe Free' : `$${p.price}/mo`}
                </button>
              </div>
            </Card>
          ))}
        </div>
      </div>

      {/* Category Icons */}
      <div>
        <h2 className="text-lg font-semibold mb-4">Browse by Category</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
          {[
            { name: 'Momentum', icon: '↗', color: 'var(--green)', count: 12 },
            { name: 'Scalping', icon: '⚡', color: 'var(--orange)', count: 8 },
            { name: 'Swing', icon: '↔', color: 'var(--blue)', count: 6 },
            { name: 'Mean Reversion', icon: '↩', color: 'var(--purple)', count: 4 },
            { name: 'Breakout', icon: '↑', color: 'var(--teal)', count: 5 },
            { name: 'Conservative', icon: '■', color: 'var(--accent)', count: 7 },
          ].map((cat) => (
            <Card key={cat.name} className="text-center cursor-pointer hover:border-[var(--accent)] transition-colors">
              <div className="text-2xl mb-2">{cat.icon}</div>
              <p className="text-sm font-medium">{cat.name}</p>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{cat.count} portfolios</p>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
