'use client';

import { useState, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
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
  successRate: number;
  avgPassTime: number;
}

const mockFirms: PropFirm[] = [
  { id: 'ttp', name: 'Trade The Pool', accounts: ['25K', '50K', '100K', '200K'], evalPrice: '$99-$399', profitTarget: '$3,000-$12,000', maxDD: '6%', dailyLoss: '2%', minDays: 5, compatible: 12, rating: 4.6, reviews: 234, successRate: 38, avgPassTime: 12 },
  { id: 'ftmo', name: 'FTMO', accounts: ['10K', '25K', '50K', '100K', '200K'], evalPrice: '$155-$1,080', profitTarget: '10%', maxDD: '10%', dailyLoss: '5%', minDays: 4, compatible: 8, rating: 4.5, reviews: 189, successRate: 32, avgPassTime: 15 },
  { id: 'topstep', name: 'Topstep', accounts: ['50K', '100K', '150K'], evalPrice: '$49-$149/mo', profitTarget: '$3,000-$9,000', maxDD: '$2,000-$4,500', dailyLoss: '$1,000-$2,000', minDays: 0, compatible: 6, rating: 4.3, reviews: 156, successRate: 28, avgPassTime: 18 },
  { id: 'apex', name: 'Apex Trader', accounts: ['25K', '50K', '100K', '250K'], evalPrice: '$37-$187/mo', profitTarget: '$1,500-$15,000', maxDD: '$1,500-$6,250', dailyLoss: 'N/A', minDays: 7, compatible: 5, rating: 4.1, reviews: 98, successRate: 25, avgPassTime: 20 },
];

const firmColors: Record<string, string> = {
  ttp: 'var(--blue)',
  ftmo: 'var(--orange)',
  topstep: 'var(--green)',
  apex: 'var(--purple)',
};

export default function PropFirmsV4() {
  const [compareFirms, setCompareFirms] = useState<string[]>([]);
  const [simWinRate, setSimWinRate] = useState(55);
  const [simPF, setSimPF] = useState(2.0);
  const [simDailyR, setSimDailyR] = useState(1.5);
  const [simMaxDD, setSimMaxDD] = useState(2.0);
  const [accountSize, setAccountSize] = useState(50000);
  const [riskPercent, setRiskPercent] = useState(1.5);

  function toggleCompare(id: string) {
    setCompareFirms((prev) =>
      prev.includes(id) ? prev.filter((f) => f !== id) : prev.length < 2 ? [...prev, id] : [prev[1], id]
    );
  }

  const compareA = mockFirms.find((f) => f.id === compareFirms[0]);
  const compareB = mockFirms.find((f) => f.id === compareFirms[1]);

  function simulatePass(firm: PropFirm): { passes: boolean; reason: string } {
    const ddNum = parseFloat(firm.maxDD.replace(/[^0-9.]/g, ''));
    const dailyLossNum = firm.dailyLoss === 'N/A' ? 100 : parseFloat(firm.dailyLoss.replace(/[^0-9.]/g, ''));

    if (simMaxDD > ddNum && firm.maxDD.includes('%')) return { passes: false, reason: `Max DD ${simMaxDD}% exceeds limit of ${firm.maxDD}` };
    if (simMaxDD > dailyLossNum && firm.dailyLoss.includes('%')) return { passes: false, reason: `Daily loss risk exceeds ${firm.dailyLoss}` };
    if (simWinRate < 45) return { passes: false, reason: 'Win rate below safe threshold for evaluation' };
    if (simPF < 1.2) return { passes: false, reason: 'Profit factor too low for consistent evaluation passes' };
    return { passes: true, reason: 'Strategy metrics within evaluation parameters' };
  }

  const monthlyProfit = accountSize * (riskPercent / 100) * simDailyR * 22;

  return (
    <div>
      <PageHeader title="Prop Firm Hub" subtitle="Interactive comparison, simulation, and funding calculator" backHref="/marketplace" />

      {/* Summary */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <MetricCard label="Firms Listed" value={mockFirms.length.toString()} />
        <MetricCard label="Highest Success Rate" value={`${Math.max(...mockFirms.map((f) => f.successRate))}%`} />
        <MetricCard label="Most Compatible" value={mockFirms.sort((a, b) => b.compatible - a.compatible)[0].name} />
        <MetricCard label="Lowest Entry" value="$37/mo" />
      </div>

      {/* Interactive Comparison — Drag to compare */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold mb-1">Compare Firms</h2>
        <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>Select two firms to compare side-by-side</p>

        <div className="flex gap-3 mb-4">
          {mockFirms.map((firm) => (
            <button
              key={firm.id}
              onClick={() => toggleCompare(firm.id)}
              className="px-4 py-2.5 rounded-lg text-sm font-medium transition-all"
              style={{
                background: compareFirms.includes(firm.id) ? firmColors[firm.id] : 'var(--bg-input)',
                color: compareFirms.includes(firm.id) ? 'white' : 'var(--text-secondary)',
                border: compareFirms.includes(firm.id) ? 'none' : '1px solid var(--border)',
              }}
            >
              {firm.name}
            </button>
          ))}
        </div>

        {compareA && compareB && (
          <Card>
            <div className="grid grid-cols-3 gap-4">
              {/* Firm A */}
              <div className="text-center">
                <div className="w-16 h-16 rounded-full flex items-center justify-center text-2xl font-bold mx-auto mb-2" style={{ background: firmColors[compareA.id], color: 'white' }}>
                  {compareA.name.charAt(0)}
                </div>
                <h3 className="font-bold">{compareA.name}</h3>
                <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>{compareA.rating}/5 ({compareA.reviews} reviews)</p>
              </div>

              {/* VS */}
              <div className="flex items-center justify-center">
                <span className="text-2xl font-bold" style={{ color: 'var(--text-muted)' }}>VS</span>
              </div>

              {/* Firm B */}
              <div className="text-center">
                <div className="w-16 h-16 rounded-full flex items-center justify-center text-2xl font-bold mx-auto mb-2" style={{ background: firmColors[compareB.id], color: 'white' }}>
                  {compareB.name.charAt(0)}
                </div>
                <h3 className="font-bold">{compareB.name}</h3>
                <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>{compareB.rating}/5 ({compareB.reviews} reviews)</p>
              </div>
            </div>

            <div className="mt-6 space-y-2">
              {[
                { label: 'Eval Price', a: compareA.evalPrice, b: compareB.evalPrice },
                { label: 'Profit Target', a: compareA.profitTarget, b: compareB.profitTarget },
                { label: 'Max DD', a: compareA.maxDD, b: compareB.maxDD },
                { label: 'Daily Loss', a: compareA.dailyLoss, b: compareB.dailyLoss },
                { label: 'Min Days', a: compareA.minDays === 0 ? 'None' : `${compareA.minDays}`, b: compareB.minDays === 0 ? 'None' : `${compareB.minDays}` },
                { label: 'Compatible', a: `${compareA.compatible} portfolios`, b: `${compareB.compatible} portfolios` },
                { label: 'Success Rate', a: `${compareA.successRate}%`, b: `${compareB.successRate}%` },
                { label: 'Avg Pass Time', a: `${compareA.avgPassTime} days`, b: `${compareB.avgPassTime} days` },
              ].map((row) => (
                <div key={row.label} className="grid grid-cols-3 gap-4 py-2 rounded" style={{ background: 'var(--bg-input)' }}>
                  <div className="text-right text-sm font-medium px-3">{row.a}</div>
                  <div className="text-center text-xs" style={{ color: 'var(--text-muted)' }}>{row.label}</div>
                  <div className="text-left text-sm font-medium px-3">{row.b}</div>
                </div>
              ))}
            </div>
          </Card>
        )}

        {compareFirms.length < 2 && (
          <p className="text-xs text-center py-4" style={{ color: 'var(--text-muted)' }}>
            Select {2 - compareFirms.length} more firm{compareFirms.length === 0 ? 's' : ''} to compare
          </p>
        )}
      </div>

      {/* Evaluation Simulator */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold mb-1">Evaluation Simulator</h2>
        <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>Enter your strategy stats to see which firms you would pass</p>

        <Card className="mb-4">
          <div className="grid grid-cols-4 gap-4 mb-6">
            <div>
              <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Win Rate (%)</label>
              <input type="number" className="w-full px-3 py-2 rounded-lg text-sm" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} value={simWinRate} onChange={(e) => setSimWinRate(Number(e.target.value))} min={1} max={100} />
            </div>
            <div>
              <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Profit Factor</label>
              <input type="number" className="w-full px-3 py-2 rounded-lg text-sm" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} value={simPF} onChange={(e) => setSimPF(Number(e.target.value))} min={0.1} max={10} step={0.1} />
            </div>
            <div>
              <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Daily R</label>
              <input type="number" className="w-full px-3 py-2 rounded-lg text-sm" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} value={simDailyR} onChange={(e) => setSimDailyR(Number(e.target.value))} min={0.1} max={5} step={0.1} />
            </div>
            <div>
              <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Max DD (%)</label>
              <input type="number" className="w-full px-3 py-2 rounded-lg text-sm" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} value={simMaxDD} onChange={(e) => setSimMaxDD(Number(e.target.value))} min={0.1} max={20} step={0.1} />
            </div>
          </div>

          <div className="space-y-3">
            {mockFirms.map((firm) => {
              const result = simulatePass(firm);
              return (
                <div key={firm.id} className="flex items-center justify-between p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                  <div className="flex items-center gap-3">
                    <span className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold" style={{ background: result.passes ? 'var(--green-muted)' : 'var(--red-muted)', color: result.passes ? 'var(--green)' : 'var(--red)' }}>
                      {result.passes ? '\u2713' : '\u2717'}
                    </span>
                    <div>
                      <span className="text-sm font-medium">{firm.name}</span>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{result.reason}</p>
                    </div>
                  </div>
                  <span className="text-sm font-semibold" style={{ color: result.passes ? 'var(--green)' : 'var(--red)' }}>
                    {result.passes ? 'PASS' : 'FAIL'}
                  </span>
                </div>
              );
            })}
          </div>
        </Card>
      </div>

      {/* Funding Calculator */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold mb-1">Funding Calculator</h2>
        <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>Project your earnings with a funded account</p>

        <Card>
          <div className="grid grid-cols-2 gap-4 mb-6">
            <div>
              <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Account Size</label>
              <select className="w-full px-3 py-2 rounded-lg text-sm" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} value={accountSize} onChange={(e) => setAccountSize(Number(e.target.value))}>
                <option value={25000}>$25,000</option>
                <option value={50000}>$50,000</option>
                <option value={100000}>$100,000</option>
                <option value={200000}>$200,000</option>
              </select>
            </div>
            <div>
              <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Risk Per Trade (%)</label>
              <input type="number" className="w-full px-3 py-2 rounded-lg text-sm" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} value={riskPercent} onChange={(e) => setRiskPercent(Number(e.target.value))} min={0.1} max={5} step={0.1} />
            </div>
          </div>

          <div className="grid grid-cols-4 gap-4 text-center">
            <div className="rounded-lg p-4" style={{ background: 'var(--bg-input)' }}>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Daily Profit</p>
              <p className="text-xl font-bold" style={{ color: 'var(--green)' }}>${(monthlyProfit / 22).toFixed(0)}</p>
            </div>
            <div className="rounded-lg p-4" style={{ background: 'var(--bg-input)' }}>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Weekly Profit</p>
              <p className="text-xl font-bold" style={{ color: 'var(--green)' }}>${(monthlyProfit / 4.4).toFixed(0)}</p>
            </div>
            <div className="rounded-lg p-4" style={{ background: 'var(--bg-input)' }}>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Monthly Profit</p>
              <p className="text-xl font-bold" style={{ color: 'var(--green)' }}>${monthlyProfit.toFixed(0)}</p>
            </div>
            <div className="rounded-lg p-4" style={{ background: 'var(--bg-input)' }}>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Yearly Profit</p>
              <p className="text-xl font-bold" style={{ color: 'var(--green)' }}>${(monthlyProfit * 12).toFixed(0)}</p>
            </div>
          </div>

          <p className="text-xs text-center mt-3" style={{ color: 'var(--text-muted)' }}>
            Projections assume {simDailyR}R daily average. Actual results will vary. Past performance does not guarantee future results.
          </p>
        </Card>
      </div>

      {/* Success Rates */}
      <div>
        <h2 className="text-lg font-semibold mb-4">Platform Success Rates</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {mockFirms.map((firm) => (
            <Card key={firm.id} className="text-center">
              <div className="relative w-20 h-20 mx-auto mb-3">
                <svg viewBox="0 0 50 50" className="w-full h-full transform -rotate-90">
                  <circle cx="25" cy="25" r="20" fill="none" stroke="var(--bg-input)" strokeWidth="4" />
                  <circle
                    cx="25" cy="25" r="20" fill="none"
                    stroke={firmColors[firm.id]}
                    strokeWidth="4"
                    strokeDasharray={`${(firm.successRate / 100) * 125.6} 125.6`}
                    strokeLinecap="round"
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-lg font-bold">{firm.successRate}%</span>
                </div>
              </div>
              <h3 className="font-semibold text-sm">{firm.name}</h3>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Avg pass: {firm.avgPassTime} days</p>
              <button className="mt-3 w-full px-3 py-1.5 rounded-lg text-xs font-medium" style={{ background: 'var(--accent)', color: 'white' }}>
                Sign Up
              </button>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
