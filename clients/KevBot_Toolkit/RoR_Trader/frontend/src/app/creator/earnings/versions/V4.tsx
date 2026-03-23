'use client';

import { useState, useEffect } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import MetricCard from '@/components/MetricCard';

const mockEarnings = {
  totalEarned: 4280,
  pendingPayout: 890,
  nextPayoutDate: '2026-04-01',
  platformFeePaid: 756,
  monthlyBreakdown: [
    { month: 'Mar 2026', gross: 1050, platformFee: 158, net: 892 },
    { month: 'Feb 2026', gross: 850, platformFee: 128, net: 722 },
    { month: 'Jan 2026', gross: 780, platformFee: 117, net: 663 },
  ],
  payouts: [
    { date: '2026-03-01', amount: 722, method: 'PayPal', status: 'Paid' as const },
    { date: '2026-02-01', amount: 663, method: 'PayPal', status: 'Paid' as const },
    { date: '2026-01-01', amount: 581, method: 'PayPal', status: 'Paid' as const },
  ],
};

const mockRevenueChain = [
  { label: 'Subscribers Pay', amount: 890, color: 'var(--accent)' },
  { label: 'Platform Fee (15%)', amount: 134, color: 'var(--text-muted)' },
  { label: 'You Receive (50%)', amount: 378, color: 'var(--green)' },
  { label: 'Strategy Creators (25%)', amount: 189, color: 'var(--blue)' },
  { label: 'Pack Creators (10%)', amount: 89, color: 'var(--purple)' },
];

const mockMilestones = [
  { label: 'First Payout', target: 100, reached: true },
  { label: '$1K Earned', target: 1000, reached: true },
  { label: '$5K Earned', target: 5000, reached: false, progress: 4280 },
  { label: '$10K Earned', target: 10000, reached: false, progress: 4280 },
  { label: '100 Subscribers', target: 100, reached: true },
  { label: '500 Subscribers', target: 500, reached: false, progress: 342 },
];

const mockForecast = {
  currentMonthly: 890,
  projected3Mo: 1120,
  projected6Mo: 1450,
  projected12Mo: 2100,
};

function AnimatedCounter({ target, prefix = '' }: { target: number; prefix?: string }) {
  const [value, setValue] = useState(0);

  useEffect(() => {
    let current = 0;
    const step = target / 35;
    const timer = setInterval(() => {
      current += step;
      if (current >= target) {
        setValue(target);
        clearInterval(timer);
      } else {
        setValue(Math.floor(current));
      }
    }, 30);
    return () => clearInterval(timer);
  }, [target]);

  return <span>{prefix}{value.toLocaleString()}</span>;
}

function FlowArrow() {
  return (
    <div className="flex justify-center py-1">
      <svg width="20" height="24" viewBox="0 0 20 24">
        <path d="M10 0 L10 18 M4 14 L10 20 L16 14" fill="none" stroke="var(--text-muted)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  );
}

export default function CreatorEarningsV4() {
  const [showForecast, setShowForecast] = useState(false);

  return (
    <div>
      <style>{`
        @keyframes flowPulse {
          0%, 100% { opacity: 0.4; }
          50% { opacity: 1; }
        }
        @keyframes shimmer {
          0% { background-position: -200% 0; }
          100% { background-position: 200% 0; }
        }
        .flow-node {
          animation: flowPulse 3s ease-in-out infinite;
        }
        .milestone-badge {
          background: linear-gradient(90deg, var(--accent-muted), var(--accent), var(--accent-muted));
          background-size: 200% 100%;
          animation: shimmer 3s infinite;
        }
      `}</style>

      <PageHeader
        title="Earnings & Payouts"
        subtitle="Visualize your revenue flow and track milestones"
        backHref="/creator/dashboard"
      />

      {/* Animated KPI */}
      <Card className="mb-6">
        <div className="grid grid-cols-3 gap-8">
          <div className="text-center">
            <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Total Earned</p>
            <p className="text-3xl font-bold" style={{ color: 'var(--accent)' }}>
              <AnimatedCounter target={mockEarnings.totalEarned} prefix="$" />
            </p>
          </div>
          <div className="text-center">
            <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Pending</p>
            <p className="text-3xl font-bold" style={{ color: 'var(--green)' }}>
              <AnimatedCounter target={mockEarnings.pendingPayout} prefix="$" />
            </p>
          </div>
          <div className="text-center">
            <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Platform Fees</p>
            <p className="text-3xl font-bold" style={{ color: 'var(--orange)' }}>
              <AnimatedCounter target={mockEarnings.platformFeePaid} prefix="$" />
            </p>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-3 gap-6 mb-6">
        {/* Money Flow Visualization */}
        <Card>
          <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
            Revenue Flow (Monthly)
          </h3>
          <div className="space-y-0">
            {mockRevenueChain.map((node, i) => (
              <div key={node.label}>
                <div
                  className="flow-node rounded-lg p-3 text-center"
                  style={{
                    background: 'var(--bg-input)',
                    borderLeft: `3px solid ${node.color}`,
                    animationDelay: `${i * 0.5}s`,
                  }}
                >
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{node.label}</p>
                  <p className="text-lg font-bold" style={{ color: node.color }}>${node.amount}</p>
                </div>
                {i < mockRevenueChain.length - 1 && <FlowArrow />}
              </div>
            ))}
          </div>
        </Card>

        {/* Earnings Forecast */}
        <Card>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
              Earnings Forecast
            </h3>
            <button
              className="text-xs px-2 py-1 rounded"
              style={{
                background: showForecast ? 'var(--accent-muted)' : 'var(--bg-input)',
                color: showForecast ? 'var(--accent)' : 'var(--text-muted)',
              }}
              onClick={() => setShowForecast(!showForecast)}
            >
              {showForecast ? 'Hide' : 'Show'} Projection
            </button>
          </div>

          <div className="space-y-4">
            <div className="p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Current Monthly</p>
              <p className="text-xl font-bold" style={{ color: 'var(--accent)' }}>${mockForecast.currentMonthly}/mo</p>
            </div>

            {showForecast && (
              <>
                {[
                  { label: '3-Month Projection', value: mockForecast.projected3Mo, growth: ((mockForecast.projected3Mo / mockForecast.currentMonthly - 1) * 100).toFixed(0) },
                  { label: '6-Month Projection', value: mockForecast.projected6Mo, growth: ((mockForecast.projected6Mo / mockForecast.currentMonthly - 1) * 100).toFixed(0) },
                  { label: '12-Month Projection', value: mockForecast.projected12Mo, growth: ((mockForecast.projected12Mo / mockForecast.currentMonthly - 1) * 100).toFixed(0) },
                ].map((proj) => (
                  <div key={proj.label} className="p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{proj.label}</p>
                    <div className="flex items-end justify-between">
                      <p className="text-lg font-bold">${proj.value}/mo</p>
                      <span className="text-xs" style={{ color: 'var(--green)' }}>+{proj.growth}%</span>
                    </div>
                  </div>
                ))}
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  Based on current growth trends. Actual results may vary.
                </p>
              </>
            )}
          </div>
        </Card>

        {/* Milestone Badges */}
        <Card>
          <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
            Milestones
          </h3>
          <div className="space-y-3">
            {mockMilestones.map((m, i) => (
              <div
                key={i}
                className="flex items-center gap-3 p-3 rounded-lg"
                style={{ background: 'var(--bg-input)' }}
              >
                <div
                  className={`w-8 h-8 rounded-lg flex items-center justify-center text-xs font-bold flex-shrink-0 ${m.reached ? 'milestone-badge' : ''}`}
                  style={{
                    background: m.reached ? undefined : 'var(--bg-card)',
                    color: m.reached ? 'var(--accent)' : 'var(--text-muted)',
                    border: m.reached ? 'none' : '1px solid var(--border)',
                  }}
                >
                  {m.reached ? 'OK' : '...'}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium" style={{ color: m.reached ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                    {m.label}
                  </p>
                  {!m.reached && m.progress !== undefined && (
                    <div className="mt-1">
                      <div className="h-1.5 rounded-full" style={{ background: 'var(--bg-card)' }}>
                        <div
                          className="h-1.5 rounded-full"
                          style={{
                            width: `${Math.min((m.progress / m.target) * 100, 100)}%`,
                            background: 'var(--accent)',
                          }}
                        />
                      </div>
                      <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
                        {((m.progress / m.target) * 100).toFixed(0)}% complete
                      </p>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Payout History */}
      <Card>
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
          Recent Payouts
        </h3>
        <div className="space-y-2">
          {mockEarnings.payouts.map((p, i) => (
            <div
              key={i}
              className="flex items-center justify-between p-3 rounded-lg"
              style={{ background: 'var(--bg-input)' }}
            >
              <div className="flex items-center gap-4">
                <div
                  className="w-10 h-10 rounded-lg flex items-center justify-center text-lg"
                  style={{ background: 'var(--green-muted)', color: 'var(--green)' }}
                >
                  $
                </div>
                <div>
                  <p className="text-sm font-medium">${p.amount.toFixed(2)}</p>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{p.date} via {p.method}</p>
                </div>
              </div>
              <span
                className="text-xs px-2 py-1 rounded"
                style={{ color: 'var(--green)', background: 'var(--green-muted)' }}
              >
                {p.status}
              </span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
