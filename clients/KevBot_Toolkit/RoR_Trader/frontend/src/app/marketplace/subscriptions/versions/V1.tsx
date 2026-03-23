'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import MetricCard from '@/components/MetricCard';

interface Subscription {
  id: string;
  portfolioName: string;
  creator: string;
  status: 'active' | 'paused';
  monthlyCost: number;
  subscribedDate: string;
  performanceSinceSubscribing: number;
  webhookStatus: 'connected' | 'not_setup' | 'error';
  tradesReceived: number;
}

const mockSubscriptions: Subscription[] = [
  { id: 'sub1', portfolioName: 'Momentum Alpha', creator: 'TradeMaster', status: 'active', monthlyCost: 49, subscribedDate: '2025-11-15', performanceSinceSubscribing: 12.4, webhookStatus: 'connected', tradesReceived: 87 },
  { id: 'sub2', portfolioName: 'Steady Eddie', creator: 'RoR Team', status: 'active', monthlyCost: 0, subscribedDate: '2025-10-01', performanceSinceSubscribing: 5.8, webhookStatus: 'connected', tradesReceived: 142 },
  { id: 'sub3', portfolioName: 'VWAP Scalper Pro', creator: 'QuantEdge', status: 'paused', monthlyCost: 29, subscribedDate: '2026-01-10', performanceSinceSubscribing: -1.2, webhookStatus: 'error', tradesReceived: 34 },
  { id: 'sub4', portfolioName: 'Tech Momentum', creator: 'AlgoTrader99', status: 'active', monthlyCost: 39, subscribedDate: '2026-02-20', performanceSinceSubscribing: 3.1, webhookStatus: 'not_setup', tradesReceived: 0 },
];

const webhookColors: Record<string, { color: string; bg: string; label: string }> = {
  connected: { color: 'var(--green)', bg: 'var(--green-muted)', label: 'Connected' },
  not_setup: { color: 'var(--orange)', bg: 'var(--orange-muted)', label: 'Not Set Up' },
  error: { color: 'var(--red)', bg: 'var(--red-muted)', label: 'Error' },
};

const statusColors: Record<string, { color: string; bg: string }> = {
  active: { color: 'var(--green)', bg: 'var(--green-muted)' },
  paused: { color: 'var(--text-muted)', bg: 'var(--bg-input)' },
};

export default function SubscriptionsV1() {
  const [subs, setSubs] = useState(mockSubscriptions);

  const totalCost = subs.filter((s) => s.status === 'active').reduce((sum, s) => sum + s.monthlyCost, 0);
  const activeCount = subs.filter((s) => s.status === 'active').length;

  function toggleStatus(id: string) {
    setSubs((prev) =>
      prev.map((s) =>
        s.id === id ? { ...s, status: s.status === 'active' ? 'paused' : 'active' } : s
      )
    );
  }

  return (
    <div>
      <PageHeader
        title="My Subscriptions"
        subtitle="Manage your active portfolio subscriptions"
        backHref="/marketplace"
        actions={
          <button className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: 'white' }}>
            Browse Marketplace
          </button>
        }
      />

      {/* Summary */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <MetricCard label="Active Subscriptions" value={activeCount.toString()} />
        <MetricCard label="Monthly Spend" value={totalCost === 0 ? 'Free' : `$${totalCost}/mo`} />
        <MetricCard label="Total Trades Received" value={subs.reduce((s, sub) => s + sub.tradesReceived, 0).toLocaleString()} />
      </div>

      {/* Subscription List */}
      {subs.length === 0 ? (
        <Card className="text-center py-12">
          <p className="text-lg font-semibold mb-2">No Subscriptions Yet</p>
          <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>Browse the marketplace to find portfolios that match your trading style.</p>
          <button className="px-6 py-2.5 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: 'white' }}>Browse Marketplace</button>
        </Card>
      ) : (
        <div className="space-y-4">
          {subs.map((sub) => (
            <Card key={sub.id}>
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <h3 className="font-semibold text-sm">{sub.portfolioName}</h3>
                    <span className="text-xs px-2 py-0.5 rounded" style={{ background: statusColors[sub.status].bg, color: statusColors[sub.status].color }}>
                      {sub.status}
                    </span>
                    <span className="text-xs px-2 py-0.5 rounded" style={{ background: webhookColors[sub.webhookStatus].bg, color: webhookColors[sub.webhookStatus].color }}>
                      {webhookColors[sub.webhookStatus].label}
                    </span>
                  </div>
                  <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
                    by {sub.creator} &middot; Subscribed {sub.subscribedDate} &middot; {sub.monthlyCost === 0 ? 'Free' : `$${sub.monthlyCost}/mo`}
                  </p>

                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Performance</p>
                      <p className="text-sm font-semibold" style={{ color: sub.performanceSinceSubscribing >= 0 ? 'var(--green)' : 'var(--red)' }}>
                        {sub.performanceSinceSubscribing >= 0 ? '+' : ''}{sub.performanceSinceSubscribing}%
                      </p>
                    </div>
                    <div>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Trades Received</p>
                      <p className="text-sm font-semibold">{sub.tradesReceived}</p>
                    </div>
                    <div>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Webhook</p>
                      <p className="text-sm font-semibold" style={{ color: webhookColors[sub.webhookStatus].color }}>{webhookColors[sub.webhookStatus].label}</p>
                    </div>
                  </div>
                </div>

                <div className="flex flex-col gap-2 ml-4">
                  <button
                    className="px-3 py-1.5 rounded-lg text-xs"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                    onClick={() => toggleStatus(sub.id)}
                  >
                    {sub.status === 'active' ? 'Pause' : 'Resume'}
                  </button>
                  <button
                    className="px-3 py-1.5 rounded-lg text-xs"
                    style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                  >
                    Configure
                  </button>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
