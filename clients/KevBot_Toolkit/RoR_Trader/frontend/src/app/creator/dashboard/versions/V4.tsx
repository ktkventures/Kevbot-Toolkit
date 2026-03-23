'use client';

import { useState, useEffect } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

const mockCreatorStats = {
  totalEarnings: 4280,
  thisMonth: 890,
  lastMonth: 720,
  subscribers: 342,
  newThisMonth: 47,
  published: { portfolios: 2, strategies: 5, packs: 1 },
  avgRating: 4.6,
  totalReviews: 201,
};

const mockItems = [
  { id: '1', name: 'Momentum Alpha', type: 'Portfolio' as const, subscribers: 156, monthlyRevenue: 490, rating: 4.8, trend: 'up' as const, healthScore: 94 },
  { id: '2', name: 'VWAP Scalper Pro', type: 'Portfolio' as const, subscribers: 89, monthlyRevenue: 230, rating: 4.5, trend: 'up' as const, healthScore: 87 },
  { id: '3', name: 'EMA Momentum', type: 'Strategy' as const, subscribers: 45, monthlyRevenue: 90, rating: 4.3, trend: 'stable' as const, healthScore: 72 },
  { id: '4', name: 'RSI Zones Pack', type: 'Pack' as const, subscribers: 52, monthlyRevenue: 80, rating: 4.7, trend: 'up' as const, healthScore: 91 },
];

const mockHeatmap = [
  // 4 weeks x 7 days of daily earnings
  [12, 18, 0, 22, 35, 40, 28],
  [15, 0, 30, 25, 42, 38, 20],
  [0, 28, 32, 45, 50, 35, 22],
  [30, 35, 28, 0, 55, 48, 40],
];

const mockInsights = [
  { text: 'Your VWAP Scalper Pro gained 12 subscribers this week', type: 'positive' as const },
  { text: 'Momentum Alpha revenue up 24% month-over-month', type: 'positive' as const },
  { text: 'EMA Momentum rating dipped — consider updating description', type: 'warning' as const },
  { text: 'RSI Zones Pack is trending in the marketplace top 10', type: 'positive' as const },
];

const typeColors: Record<string, { color: string; bg: string }> = {
  Portfolio: { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  Strategy: { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  Pack: { color: 'var(--purple)', bg: 'var(--purple-muted)' },
};

const dayLabels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

function AnimatedCounter({ target, prefix = '' }: { target: number; prefix?: string }) {
  const [value, setValue] = useState(0);

  useEffect(() => {
    let current = 0;
    const step = target / 40;
    const timer = setInterval(() => {
      current += step;
      if (current >= target) {
        setValue(target);
        clearInterval(timer);
      } else {
        setValue(Math.floor(current));
      }
    }, 25);
    return () => clearInterval(timer);
  }, [target]);

  return (
    <span>
      {prefix}{value.toLocaleString()}
    </span>
  );
}

function HealthGauge({ score }: { score: number }) {
  const color = score >= 90 ? 'var(--green)' : score >= 70 ? 'var(--orange)' : 'var(--red)';
  const circumference = 2 * Math.PI * 18;
  const offset = circumference - (score / 100) * circumference;

  return (
    <div className="relative w-12 h-12">
      <svg viewBox="0 0 44 44" className="w-full h-full" style={{ transform: 'rotate(-90deg)' }}>
        <circle cx="22" cy="22" r="18" fill="none" stroke="var(--bg-input)" strokeWidth="4" />
        <circle
          cx="22"
          cy="22"
          r="18"
          fill="none"
          stroke={color}
          strokeWidth="4"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          style={{ transition: 'stroke-dashoffset 1s ease-out' }}
        />
      </svg>
      <div
        className="absolute inset-0 flex items-center justify-center text-xs font-bold"
        style={{ color }}
      >
        {score}
      </div>
    </div>
  );
}

function PulseIndicator({ active }: { active: boolean }) {
  return (
    <span
      className="inline-block w-2.5 h-2.5 rounded-full"
      style={{
        background: active ? 'var(--green)' : 'var(--red)',
        boxShadow: active ? '0 0 8px var(--green)' : 'none',
        animation: active ? 'pulse 2s infinite' : 'none',
      }}
    />
  );
}

export default function CreatorDashboardV4() {
  const [expandedInsight, setExpandedInsight] = useState<number | null>(null);
  const stats = mockCreatorStats;

  return (
    <div>
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.6; transform: scale(1.3); }
        }
        @keyframes slideIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-slide-in {
          animation: slideIn 0.4s ease-out both;
        }
      `}</style>

      <PageHeader
        title="Creator Dashboard"
        subtitle="Live insights and performance analytics"
        actions={
          <div className="flex items-center gap-2">
            <PulseIndicator active />
            <span className="text-xs" style={{ color: 'var(--green)' }}>Live</span>
          </div>
        }
      />

      {/* Animated Revenue Hero */}
      <Card className="mb-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Total Lifetime Earnings</p>
            <p className="text-4xl font-bold" style={{ color: 'var(--accent)' }}>
              <AnimatedCounter target={stats.totalEarnings} prefix="$" />
            </p>
            <p className="text-sm mt-2" style={{ color: 'var(--green)' }}>
              +$<AnimatedCounter target={stats.thisMonth} /> this month
            </p>
          </div>
          <div className="text-right">
            <div className="flex items-center gap-3 mb-2">
              <PulseIndicator active />
              <span className="text-sm font-medium">
                <AnimatedCounter target={stats.subscribers} /> active subscribers
              </span>
            </div>
            <p className="text-xs" style={{ color: 'var(--green)' }}>
              +{stats.newThisMonth} new this month
            </p>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-3 gap-6 mb-6">
        {/* Earnings Heatmap */}
        <Card className="col-span-2">
          <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
            Earnings Heatmap (Last 4 Weeks)
          </h3>
          <div className="flex gap-1 mb-2">
            {dayLabels.map((d) => (
              <div key={d} className="flex-1 text-center text-xs" style={{ color: 'var(--text-muted)' }}>
                {d}
              </div>
            ))}
          </div>
          {mockHeatmap.map((week, wi) => (
            <div key={wi} className="flex gap-1 mb-1">
              {week.map((val, di) => {
                const maxVal = Math.max(...mockHeatmap.flat());
                const intensity = maxVal > 0 ? val / maxVal : 0;
                return (
                  <div
                    key={di}
                    className="flex-1 rounded aspect-square flex items-center justify-center text-xs font-mono cursor-pointer transition-transform hover:scale-110"
                    style={{
                      background: val === 0 ? 'var(--bg-input)' : `color-mix(in srgb, var(--green) ${Math.max(intensity * 100, 15)}%, var(--bg-input))`,
                      color: intensity > 0.5 ? 'white' : 'var(--text-muted)',
                    }}
                    title={`$${val} earned`}
                  >
                    {val > 0 ? `$${val}` : ''}
                  </div>
                );
              })}
            </div>
          ))}
          <div className="flex items-center justify-end gap-2 mt-3">
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Less</span>
            {[0, 0.25, 0.5, 0.75, 1].map((v, i) => (
              <div
                key={i}
                className="w-3 h-3 rounded"
                style={{
                  background: v === 0 ? 'var(--bg-input)' : `color-mix(in srgb, var(--green) ${v * 100}%, var(--bg-input))`,
                }}
              />
            ))}
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>More</span>
          </div>
        </Card>

        {/* AI Insights */}
        <Card>
          <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
            AI Insights
          </h3>
          <div className="space-y-3">
            {mockInsights.map((insight, i) => (
              <div
                key={i}
                className="rounded-lg p-3 cursor-pointer transition-all animate-slide-in"
                style={{
                  background: 'var(--bg-input)',
                  borderLeft: `3px solid ${insight.type === 'positive' ? 'var(--green)' : 'var(--orange)'}`,
                  animationDelay: `${i * 100}ms`,
                }}
                onClick={() => setExpandedInsight(expandedInsight === i ? null : i)}
              >
                <p className="text-xs" style={{ color: 'var(--text-primary)' }}>{insight.text}</p>
                {expandedInsight === i && (
                  <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
                    {insight.type === 'positive'
                      ? 'Keep up the good work! This trend is likely to continue.'
                      : 'Consider reviewing this item and making improvements.'}
                  </p>
                )}
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Item Health Scores */}
      <Card>
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
          Item Health Scores
        </h3>
        <div className="grid grid-cols-4 gap-4">
          {mockItems.map((item) => (
            <div
              key={item.id}
              className="rounded-lg p-4 text-center transition-transform hover:scale-[1.02]"
              style={{ background: 'var(--bg-input)' }}
            >
              <div className="flex justify-center mb-3">
                <HealthGauge score={item.healthScore} />
              </div>
              <p className="font-medium text-sm mb-1">{item.name}</p>
              <span
                className="text-xs px-2 py-0.5 rounded"
                style={{ color: typeColors[item.type]?.color, background: typeColors[item.type]?.bg }}
              >
                {item.type}
              </span>
              <div className="flex justify-between mt-3 text-xs" style={{ color: 'var(--text-muted)' }}>
                <span>{item.subscribers} subs</span>
                <span>${item.monthlyRevenue}/mo</span>
              </div>
              <div className="mt-2">
                <div className="h-1.5 rounded-full" style={{ background: 'var(--bg-card)' }}>
                  <div
                    className="h-1.5 rounded-full transition-all"
                    style={{
                      width: `${item.healthScore}%`,
                      background: item.healthScore >= 90 ? 'var(--green)' : item.healthScore >= 70 ? 'var(--orange)' : 'var(--red)',
                    }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
