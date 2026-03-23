'use client';

import { useState, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import MetricCard from '@/components/MetricCard';
import ChartPlaceholder from '@/components/ChartPlaceholder';

interface Subscription {
  id: string;
  name: string;
  creator: string;
  status: 'active' | 'paused';
  cost: number;
  subscribedDate: string;
  trades: number;
  winRate: number;
  pnl: number;
  alerts: number;
  webhookStatus: 'connected' | 'not_setup' | 'error';
  riskPerTrade: number;
  healthScore: number;
  dailyPnl: number[];
}

const mockSubs: Subscription[] = [
  { id: 'sub1', name: 'Momentum Alpha', creator: 'TradeMaster', status: 'active', cost: 49, subscribedDate: '2025-11-15', trades: 87, winRate: 57.5, pnl: 1240, alerts: 312, webhookStatus: 'connected', riskPerTrade: 1.5, healthScore: 92, dailyPnl: [45, -20, 65, 30, -15, 80, 25, 55, -10, 70, 40, -25, 60, 35, 90] },
  { id: 'sub2', name: 'Steady Eddie', creator: 'RoR Team', status: 'active', cost: 0, subscribedDate: '2025-10-01', trades: 142, winRate: 52.1, pnl: 580, alerts: 487, webhookStatus: 'connected', riskPerTrade: 1.0, healthScore: 78, dailyPnl: [15, 10, -5, 20, 8, -12, 18, 25, -3, 14, 22, 5, -8, 16, 11] },
  { id: 'sub3', name: 'VWAP Scalper Pro', creator: 'QuantEdge', status: 'paused', cost: 29, subscribedDate: '2026-01-10', trades: 34, winRate: 47.1, pnl: -120, alerts: 89, webhookStatus: 'error', riskPerTrade: 2.0, healthScore: 35, dailyPnl: [-20, 10, -30, 15, -25, -10, 5, -35, 20, -15, -40, 10, -20, -5, 15] },
  { id: 'sub4', name: 'Tech Momentum', creator: 'AlgoTrader99', status: 'active', cost: 39, subscribedDate: '2026-02-20', trades: 0, winRate: 0, pnl: 0, alerts: 0, webhookStatus: 'not_setup', riskPerTrade: 1.0, healthScore: 50, dailyPnl: [] },
];

const recommendations = [
  { name: 'Swing Master', reason: 'High PF, low DD, complements your momentum strategies', price: 79 },
  { name: 'Conservative Growth', reason: 'Free, live-verified, great diversification', price: 0 },
];

function HealthGauge({ score }: { score: number }) {
  const color = score >= 70 ? 'var(--green)' : score >= 40 ? 'var(--orange)' : 'var(--red)';
  const circumference = 2 * Math.PI * 20;
  const filled = (score / 100) * circumference;

  return (
    <div className="relative w-14 h-14">
      <svg viewBox="0 0 50 50" className="w-full h-full transform -rotate-90">
        <circle cx="25" cy="25" r="20" fill="none" stroke="var(--bg-input)" strokeWidth="4" />
        <circle cx="25" cy="25" r="20" fill="none" stroke={color} strokeWidth="4" strokeDasharray={`${filled} ${circumference}`} strokeLinecap="round" />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-xs font-bold" style={{ color }}>{score}</span>
      </div>
    </div>
  );
}

function MiniPnlChart({ data }: { data: number[] }) {
  if (data.length === 0) return <div className="h-10 flex items-center justify-center text-xs" style={{ color: 'var(--text-muted)' }}>No data</div>;

  const cumulative = data.reduce<number[]>((acc, val) => {
    acc.push((acc.length > 0 ? acc[acc.length - 1] : 0) + val);
    return acc;
  }, []);

  const max = Math.max(...cumulative.map(Math.abs), 1);
  const points = cumulative.map((v, i) => `${(i / (cumulative.length - 1)) * 100},${50 - (v / max) * 40}`).join(' ');
  const isPositive = cumulative[cumulative.length - 1] > 0;

  return (
    <svg viewBox="0 0 100 100" className="w-full h-10" preserveAspectRatio="none">
      <line x1="0" y1="50" x2="100" y2="50" stroke="var(--border)" strokeWidth="0.5" strokeDasharray="2" />
      <polyline points={points} fill="none" stroke={isPositive ? 'var(--green)' : 'var(--red)'} strokeWidth="2" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function WebhookPulse({ status }: { status: string }) {
  const color = status === 'connected' ? 'var(--green)' : status === 'error' ? 'var(--red)' : 'var(--orange)';
  return (
    <div className="flex items-center gap-2">
      <div className="relative">
        <span className="w-3 h-3 rounded-full inline-block" style={{ background: color }} />
        {status === 'connected' && (
          <span className="absolute inset-0 w-3 h-3 rounded-full animate-ping" style={{ background: color, opacity: 0.4 }} />
        )}
      </div>
      <span className="text-xs" style={{ color }}>{status === 'connected' ? 'Live' : status === 'error' ? 'Error' : 'Setup Needed'}</span>
    </div>
  );
}

export default function SubscriptionsV4() {
  const [subs] = useState(mockSubs);

  const totalCost = subs.filter((s) => s.status === 'active').reduce((sum, s) => sum + s.cost, 0);
  const totalPnl = subs.reduce((sum, s) => sum + s.pnl, 0);
  const totalTrades = subs.reduce((sum, s) => sum + s.trades, 0);
  const avgHealth = Math.round(subs.reduce((sum, s) => sum + s.healthScore, 0) / subs.length);

  const combinedDailyPnl = useMemo(() => {
    const maxLen = Math.max(...subs.map((s) => s.dailyPnl.length));
    const combined: number[] = [];
    for (let i = 0; i < maxLen; i++) {
      let dayTotal = 0;
      subs.forEach((s) => {
        if (s.dailyPnl[i] !== undefined) dayTotal += s.dailyPnl[i];
      });
      combined.push(dayTotal);
    }
    return combined;
  }, [subs]);

  return (
    <div>
      <PageHeader title="My Subscriptions" subtitle="Dashboard view with portfolio health monitoring" backHref="/marketplace" />

      {/* Summary Metrics */}
      <div className="grid grid-cols-5 gap-4 mb-6">
        <MetricCard label="Active" value={subs.filter((s) => s.status === 'active').length.toString()} />
        <MetricCard label="Monthly Spend" value={totalCost === 0 ? 'Free' : `$${totalCost}/mo`} />
        <MetricCard label="Combined P&L" value={`${totalPnl >= 0 ? '+' : ''}$${totalPnl.toLocaleString()}`} positive={totalPnl >= 0} />
        <MetricCard label="Total Trades" value={totalTrades.toString()} />
        <MetricCard label="Avg Health" value={`${avgHealth}/100`} positive={avgHealth >= 60} />
      </div>

      {/* Combined P&L Chart */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-3">Combined P&L (All Subscriptions)</h3>
        <div className="h-48">
          {combinedDailyPnl.length > 0 ? (
            <svg viewBox="0 0 400 150" className="w-full h-full" preserveAspectRatio="none">
              {/* Grid */}
              {[0, 1, 2, 3, 4].map((i) => (
                <line key={i} x1="0" y1={i * 37.5} x2="400" y2={i * 37.5} stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4" />
              ))}
              {/* Zero line */}
              <line x1="0" y1="75" x2="400" y2="75" stroke="var(--text-muted)" strokeWidth="0.5" />
              {/* Bars */}
              {(() => {
                const cumulative = combinedDailyPnl.reduce<number[]>((acc, val) => {
                  acc.push((acc.length > 0 ? acc[acc.length - 1] : 0) + val);
                  return acc;
                }, []);
                const max = Math.max(...cumulative.map(Math.abs), 1);
                return cumulative.map((val, i) => {
                  const barWidth = 380 / cumulative.length;
                  const x = 10 + i * barWidth;
                  const barHeight = (Math.abs(val) / max) * 65;
                  const y = val >= 0 ? 75 - barHeight : 75;
                  return (
                    <rect key={i} x={x} y={y} width={barWidth * 0.8} height={barHeight} fill={val >= 0 ? 'var(--green)' : 'var(--red)'} opacity={0.7} rx={1} />
                  );
                });
              })()}
            </svg>
          ) : (
            <ChartPlaceholder label="No P&L data available" height={192} />
          )}
        </div>
      </Card>

      {/* Subscription Health Cards */}
      <h3 className="text-lg font-semibold mb-3">Portfolio Health</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        {subs.map((sub) => (
          <Card key={sub.id} className={sub.status === 'paused' ? 'opacity-60' : ''}>
            <div className="flex items-start gap-4">
              <HealthGauge score={sub.healthScore} />
              <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <h3 className="font-semibold text-sm">{sub.name}</h3>
                    {sub.status === 'paused' && <span className="text-xs px-1.5 py-0.5 rounded" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>Paused</span>}
                  </div>
                  <WebhookPulse status={sub.webhookStatus} />
                </div>
                <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>by {sub.creator} &middot; {sub.cost === 0 ? 'Free' : `$${sub.cost}/mo`}</p>

                <div className="grid grid-cols-4 gap-3 mb-2">
                  <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>Trades</p><p className="text-sm font-semibold">{sub.trades}</p></div>
                  <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>WR</p><p className="text-sm font-semibold" style={{ color: sub.winRate >= 50 ? 'var(--green)' : sub.winRate > 0 ? 'var(--red)' : 'var(--text-muted)' }}>{sub.winRate > 0 ? `${sub.winRate}%` : '-'}</p></div>
                  <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>P&L</p><p className="text-sm font-semibold" style={{ color: sub.pnl >= 0 ? 'var(--green)' : 'var(--red)' }}>{sub.pnl !== 0 ? `${sub.pnl >= 0 ? '+' : ''}$${sub.pnl}` : '-'}</p></div>
                  <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>Alerts</p><p className="text-sm font-semibold">{sub.alerts}</p></div>
                </div>

                <MiniPnlChart data={sub.dailyPnl} />
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* Webhook Health Monitor */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-3">Webhook Health Monitor</h3>
        <div className="space-y-2">
          {subs.map((sub) => (
            <div key={sub.id} className="flex items-center gap-4 py-2">
              <WebhookPulse status={sub.webhookStatus} />
              <span className="text-sm flex-1">{sub.name}</span>
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                {sub.webhookStatus === 'connected' ? `Last ping: ${Math.floor(Math.random() * 5) + 1}s ago` : sub.webhookStatus === 'error' ? 'Last error: 2h ago' : 'No webhook configured'}
              </span>
              {sub.webhookStatus !== 'connected' && (
                <button className="px-3 py-1 rounded text-xs" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                  {sub.webhookStatus === 'error' ? 'Fix' : 'Set Up'}
                </button>
              )}
            </div>
          ))}
        </div>
      </Card>

      {/* Cost vs Revenue */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-3">Cost vs Revenue Analysis</h3>
        <div className="grid grid-cols-3 gap-4 text-center">
          <div className="rounded-lg p-4" style={{ background: 'var(--bg-input)' }}>
            <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Monthly Subscription Cost</p>
            <p className="text-xl font-bold" style={{ color: 'var(--red)' }}>${totalCost}/mo</p>
          </div>
          <div className="rounded-lg p-4" style={{ background: 'var(--bg-input)' }}>
            <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Total P&L Generated</p>
            <p className="text-xl font-bold" style={{ color: totalPnl >= 0 ? 'var(--green)' : 'var(--red)' }}>{totalPnl >= 0 ? '+' : ''}${totalPnl.toLocaleString()}</p>
          </div>
          <div className="rounded-lg p-4" style={{ background: 'var(--bg-input)' }}>
            <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>ROI on Subscriptions</p>
            <p className="text-xl font-bold" style={{ color: totalPnl > totalCost ? 'var(--green)' : 'var(--orange)' }}>
              {totalCost > 0 ? `${Math.round((totalPnl / totalCost) * 100)}%` : 'N/A'}
            </p>
          </div>
        </div>
      </Card>

      {/* Smart Recommendations */}
      <Card>
        <h3 className="text-sm font-medium mb-3">Recommended Additions</h3>
        <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>Based on your current subscriptions</p>
        <div className="space-y-3">
          {recommendations.map((rec) => (
            <div key={rec.name} className="flex items-center justify-between p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
              <div>
                <p className="text-sm font-medium">{rec.name}</p>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{rec.reason}</p>
              </div>
              <button className="px-3 py-1.5 rounded-lg text-xs font-medium" style={{ background: 'var(--accent)', color: 'white' }}>
                {rec.price === 0 ? 'Subscribe Free' : `$${rec.price}/mo`}
              </button>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
