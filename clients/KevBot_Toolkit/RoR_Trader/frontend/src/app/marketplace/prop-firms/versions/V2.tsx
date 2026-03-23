'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import MetricCard from '@/components/MetricCard';

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
  reviews: number;
  compatiblePortfolios: { name: string; winRate: number; pf: number; dailyR: number }[];
}

const mockFirms: PropFirm[] = [
  {
    id: 'ttp', name: 'Trade The Pool', accounts: ['25K', '50K', '100K', '200K'],
    evalPrice: '$99-$399', profitTarget: '$3,000-$12,000', maxDD: '6%', dailyLoss: '2%',
    minDays: 5, compatible: 12, rating: 4.6, reviews: 234,
    compatiblePortfolios: [
      { name: 'Momentum Alpha', winRate: 58.2, pf: 2.31, dailyR: 1.85 },
      { name: 'Steady Eddie', winRate: 52.1, pf: 1.65, dailyR: 0.95 },
      { name: 'Swing Master', winRate: 61.5, pf: 3.12, dailyR: 2.41 },
      { name: 'Conservative Growth', winRate: 50.5, pf: 1.42, dailyR: 0.68 },
    ],
  },
  {
    id: 'ftmo', name: 'FTMO', accounts: ['10K', '25K', '50K', '100K', '200K'],
    evalPrice: '$155-$1,080', profitTarget: '10%', maxDD: '10%', dailyLoss: '5%',
    minDays: 4, compatible: 8, rating: 4.5, reviews: 189,
    compatiblePortfolios: [
      { name: 'Momentum Alpha', winRate: 58.2, pf: 2.31, dailyR: 1.85 },
      { name: 'Tech Momentum', winRate: 56.3, pf: 2.08, dailyR: 1.72 },
      { name: 'Swing Master', winRate: 61.5, pf: 3.12, dailyR: 2.41 },
    ],
  },
  {
    id: 'topstep', name: 'Topstep', accounts: ['50K', '100K', '150K'],
    evalPrice: '$49-$149/mo', profitTarget: '$3,000-$9,000', maxDD: '$2,000-$4,500', dailyLoss: '$1,000-$2,000',
    minDays: 0, compatible: 6, rating: 4.3, reviews: 156,
    compatiblePortfolios: [
      { name: 'Steady Eddie', winRate: 52.1, pf: 1.65, dailyR: 0.95 },
      { name: 'Conservative Growth', winRate: 50.5, pf: 1.42, dailyR: 0.68 },
    ],
  },
  {
    id: 'apex', name: 'Apex Trader', accounts: ['25K', '50K', '100K', '250K'],
    evalPrice: '$37-$187/mo', profitTarget: '$1,500-$15,000', maxDD: '$1,500-$6,250', dailyLoss: 'N/A',
    minDays: 7, compatible: 5, rating: 4.1, reviews: 98,
    compatiblePortfolios: [
      { name: 'Conservative Growth', winRate: 50.5, pf: 1.42, dailyR: 0.68 },
    ],
  },
];

const comparisonRules = ['Evaluation Price', 'Profit Target', 'Max Drawdown', 'Daily Loss Limit', 'Min Trading Days', 'Account Sizes', 'Compatible Portfolios', 'Rating'];

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

function getFirmValue(firm: PropFirm, rule: string): string {
  switch (rule) {
    case 'Evaluation Price': return firm.evalPrice;
    case 'Profit Target': return firm.profitTarget;
    case 'Max Drawdown': return firm.maxDD;
    case 'Daily Loss Limit': return firm.dailyLoss;
    case 'Min Trading Days': return firm.minDays === 0 ? 'None' : `${firm.minDays} days`;
    case 'Account Sizes': return firm.accounts.join(', ');
    case 'Compatible Portfolios': return `${firm.compatible} portfolios`;
    case 'Rating': return `${firm.rating}/5 (${firm.reviews})`;
    default: return '-';
  }
}

export default function PropFirmsV2() {
  const [accountSize, setAccountSize] = useState(50000);
  const [riskPercent, setRiskPercent] = useState(1.5);
  const [dailyRTarget, setDailyRTarget] = useState(1.0);

  const requiredDailyR = ((accountSize * (riskPercent / 100)) > 0)
    ? dailyRTarget
    : 0;

  return (
    <div>
      <PageHeader title="Prop Firm Hub" subtitle="Compare firms, find compatible portfolios, calculate requirements" backHref="/marketplace" />

      {/* Summary */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <MetricCard label="Firms Listed" value={mockFirms.length.toString()} />
        <MetricCard label="Total Compatible Portfolios" value={mockFirms.reduce((s, f) => s + f.compatible, 0).toString()} />
        <MetricCard label="Lowest Eval Price" value="$37/mo" />
        <MetricCard label="Avg Rating" value={(mockFirms.reduce((s, f) => s + f.rating, 0) / mockFirms.length).toFixed(1)} />
      </div>

      <TabBar tabs={['Firm Cards', 'Comparison Table', 'Account Calculator']}>
        {(tab) => (
          <div>
            {tab === 'Firm Cards' && (
              <div className="space-y-6">
                {mockFirms.map((firm) => (
                  <Card key={firm.id}>
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex items-center gap-4">
                        <div className="w-14 h-14 rounded-lg flex items-center justify-center text-xl font-bold" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                          {firm.name.charAt(0)}
                        </div>
                        <div>
                          <h2 className="text-lg font-bold">{firm.name}</h2>
                          <StarRating rating={firm.rating} count={firm.reviews} />
                        </div>
                      </div>
                      <button className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: 'white' }}>
                        Sign Up (Affiliate)
                      </button>
                    </div>

                    {/* Key rules */}
                    <div className="grid grid-cols-5 gap-4 mb-4">
                      <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Eval Price</p>
                        <p className="text-sm font-semibold">{firm.evalPrice}</p>
                      </div>
                      <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Profit Target</p>
                        <p className="text-sm font-semibold">{firm.profitTarget}</p>
                      </div>
                      <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Max DD</p>
                        <p className="text-sm font-semibold" style={{ color: 'var(--red)' }}>{firm.maxDD}</p>
                      </div>
                      <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Daily Loss</p>
                        <p className="text-sm font-semibold" style={{ color: 'var(--red)' }}>{firm.dailyLoss}</p>
                      </div>
                      <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Min Days</p>
                        <p className="text-sm font-semibold">{firm.minDays === 0 ? 'None' : firm.minDays}</p>
                      </div>
                    </div>

                    {/* Account sizes */}
                    <div className="flex items-center gap-2 mb-4">
                      <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Accounts:</span>
                      {firm.accounts.map((acc) => (
                        <span key={acc} className="text-xs px-2 py-0.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>{acc}</span>
                      ))}
                    </div>

                    {/* Compatible portfolios */}
                    <div className="border-t pt-3" style={{ borderColor: 'var(--border)' }}>
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="text-sm font-medium">{firm.compatible} compatible portfolios pass this evaluation</h3>
                        <button className="text-xs" style={{ color: 'var(--accent)' }}>Find portfolios for {firm.name} &rarr;</button>
                      </div>
                      <div className="space-y-1">
                        {firm.compatiblePortfolios.map((p) => (
                          <div key={p.name} className="flex items-center justify-between py-2 px-3 rounded" style={{ background: 'var(--bg-input)' }}>
                            <span className="text-sm">{p.name}</span>
                            <div className="flex gap-4 text-xs" style={{ color: 'var(--text-secondary)' }}>
                              <span>WR {p.winRate}%</span>
                              <span>PF {p.pf}</span>
                              <span>Daily R {p.dailyR}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            )}

            {tab === 'Comparison Table' && (
              <Card>
                <h3 className="text-sm font-medium mb-4">Side-by-Side Comparison</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b" style={{ borderColor: 'var(--border)' }}>
                        <th className="text-left py-3 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Rule</th>
                        {mockFirms.map((f) => (
                          <th key={f.id} className="text-center py-3 px-3">
                            <span className="text-sm font-semibold">{f.name}</span>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {comparisonRules.map((rule) => (
                        <tr key={rule} className="border-b" style={{ borderColor: 'var(--border)' }}>
                          <td className="py-3 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>{rule}</td>
                          {mockFirms.map((firm) => (
                            <td key={firm.id} className="py-3 px-3 text-center text-sm">
                              {getFirmValue(firm, rule)}
                            </td>
                          ))}
                        </tr>
                      ))}
                      <tr>
                        <td className="py-3 px-3"></td>
                        {mockFirms.map((firm) => (
                          <td key={firm.id} className="py-3 px-3 text-center">
                            <button className="px-3 py-1.5 rounded-lg text-xs font-medium" style={{ background: 'var(--accent)', color: 'white' }}>Sign Up</button>
                          </td>
                        ))}
                      </tr>
                    </tbody>
                  </table>
                </div>
              </Card>
            )}

            {tab === 'Account Calculator' && (
              <Card>
                <h3 className="text-sm font-medium mb-4">Account Size Calculator</h3>
                <p className="text-xs mb-6" style={{ color: 'var(--text-muted)' }}>Calculate required Daily R based on your account size and risk settings</p>

                <div className="grid grid-cols-3 gap-6 mb-8">
                  <div>
                    <label className="text-xs mb-2 block" style={{ color: 'var(--text-muted)' }}>Account Size</label>
                    <select
                      className="w-full px-3 py-2 rounded-lg text-sm"
                      style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                      value={accountSize}
                      onChange={(e) => setAccountSize(Number(e.target.value))}
                    >
                      <option value={25000}>$25,000</option>
                      <option value={50000}>$50,000</option>
                      <option value={100000}>$100,000</option>
                      <option value={200000}>$200,000</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-xs mb-2 block" style={{ color: 'var(--text-muted)' }}>Risk Per Trade (%)</label>
                    <input
                      type="number"
                      className="w-full px-3 py-2 rounded-lg text-sm"
                      style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                      value={riskPercent}
                      onChange={(e) => setRiskPercent(Number(e.target.value))}
                      min={0.1}
                      max={5}
                      step={0.1}
                    />
                  </div>
                  <div>
                    <label className="text-xs mb-2 block" style={{ color: 'var(--text-muted)' }}>Target Daily R</label>
                    <input
                      type="number"
                      className="w-full px-3 py-2 rounded-lg text-sm"
                      style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                      value={dailyRTarget}
                      onChange={(e) => setDailyRTarget(Number(e.target.value))}
                      min={0.1}
                      max={5}
                      step={0.1}
                    />
                  </div>
                </div>

                {/* Results */}
                <div className="rounded-lg p-4 mb-4" style={{ background: 'var(--bg-input)' }}>
                  <h4 className="text-sm font-medium mb-3">Results</h4>
                  <div className="grid grid-cols-4 gap-4 text-center">
                    <div>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Risk Per Trade</p>
                      <p className="text-lg font-bold">${(accountSize * riskPercent / 100).toFixed(0)}</p>
                    </div>
                    <div>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Target Daily P&L</p>
                      <p className="text-lg font-bold" style={{ color: 'var(--green)' }}>${(accountSize * riskPercent / 100 * dailyRTarget).toFixed(0)}</p>
                    </div>
                    <div>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Weekly Target</p>
                      <p className="text-lg font-bold" style={{ color: 'var(--green)' }}>${(accountSize * riskPercent / 100 * dailyRTarget * 5).toFixed(0)}</p>
                    </div>
                    <div>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Monthly Target</p>
                      <p className="text-lg font-bold" style={{ color: 'var(--green)' }}>${(accountSize * riskPercent / 100 * dailyRTarget * 22).toFixed(0)}</p>
                    </div>
                  </div>
                </div>

                {/* Firm evaluation results */}
                <h4 className="text-sm font-medium mb-3">Evaluation Timeline</h4>
                <div className="space-y-2">
                  {mockFirms.map((firm) => {
                    const dailyPnl = accountSize * riskPercent / 100 * dailyRTarget;
                    const targetAmount = firm.profitTarget.includes('%')
                      ? accountSize * (parseFloat(firm.profitTarget) / 100)
                      : parseFloat(firm.profitTarget.replace(/[$,]/g, '').split('-')[0]);
                    const daysNeeded = Math.max(firm.minDays, Math.ceil(targetAmount / dailyPnl));

                    return (
                      <div key={firm.id} className="flex items-center justify-between py-2 px-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                        <span className="text-sm font-medium">{firm.name}</span>
                        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Target: {firm.profitTarget}</span>
                        <span className="text-sm font-semibold" style={{ color: daysNeeded <= 10 ? 'var(--green)' : daysNeeded <= 20 ? 'var(--orange)' : 'var(--red)' }}>
                          ~{daysNeeded} trading days
                        </span>
                      </div>
                    );
                  })}
                </div>
              </Card>
            )}
          </div>
        )}
      </TabBar>
    </div>
  );
}
