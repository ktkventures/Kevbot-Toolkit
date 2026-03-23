'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

interface Subscription {
  id: string;
  name: string;
  creator: string;
  status: 'active' | 'paused';
  cost: number;
  webhookStatus: 'connected' | 'not_setup' | 'error';
  webhookUrl: string;
  alerts: number;
}

const mockSubs: Subscription[] = [
  { id: 'sub1', name: 'Momentum Alpha', creator: 'TradeMaster', status: 'active', cost: 49, webhookStatus: 'connected', webhookUrl: 'https://webhook.ttp.com/abc123', alerts: 312 },
  { id: 'sub2', name: 'Steady Eddie', creator: 'RoR Team', status: 'active', cost: 0, webhookStatus: 'connected', webhookUrl: 'https://webhook.alpaca.com/xyz789', alerts: 487 },
  { id: 'sub3', name: 'VWAP Scalper Pro', creator: 'QuantEdge', status: 'paused', cost: 29, webhookStatus: 'error', webhookUrl: 'https://webhook.ttp.com/broken', alerts: 89 },
  { id: 'sub4', name: 'Tech Momentum', creator: 'AlgoTrader99', status: 'active', cost: 39, webhookStatus: 'not_setup', webhookUrl: '', alerts: 0 },
];

const statusDot: Record<string, string> = {
  active: 'var(--green)',
  paused: 'var(--text-muted)',
};

const webhookDot: Record<string, string> = {
  connected: 'var(--green)',
  not_setup: 'var(--orange)',
  error: 'var(--red)',
};

export default function SubscriptionsV3() {
  const [subs, setSubs] = useState(mockSubs);
  const [editingWebhook, setEditingWebhook] = useState<string | null>(null);
  const [testingId, setTestingId] = useState<string | null>(null);

  const totalCost = subs.filter((s) => s.status === 'active').reduce((sum, s) => sum + s.cost, 0);

  function toggleStatus(id: string) {
    setSubs((prev) => prev.map((s) => s.id === id ? { ...s, status: s.status === 'active' ? 'paused' : 'active' } : s));
  }

  function testWebhook(id: string) {
    setTestingId(id);
    setTimeout(() => setTestingId(null), 1500);
  }

  return (
    <div>
      <PageHeader title="My Subscriptions" subtitle="Compact overview" backHref="/marketplace" />

      {/* Total cost badge */}
      <div className="flex items-center justify-between mb-4">
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>{subs.length} subscription{subs.length !== 1 ? 's' : ''}</p>
        <span className="text-sm font-medium px-3 py-1 rounded-lg" style={{ background: 'var(--bg-input)', color: 'var(--text-primary)' }}>
          Total: {totalCost === 0 ? 'Free' : `$${totalCost}/mo`}
        </span>
      </div>

      {/* Compact list */}
      <Card>
        {/* Header */}
        <div className="flex items-center gap-3 pb-3 border-b text-xs font-medium" style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}>
          <span className="w-5"></span>
          <span className="flex-1">Portfolio</span>
          <span className="w-5"></span>
          <span className="w-20 text-right">Cost</span>
          <span className="w-16 text-right">Alerts</span>
          <span className="w-64">Webhook URL</span>
          <span className="w-16"></span>
        </div>

        <div className="divide-y" style={{ borderColor: 'var(--border)' }}>
          {subs.map((sub) => (
            <div key={sub.id}>
              <div className="flex items-center gap-3 py-3">
                {/* Status dot */}
                <span className="w-5 flex justify-center">
                  <span className="w-2.5 h-2.5 rounded-full" style={{ background: statusDot[sub.status] }} title={sub.status} />
                </span>

                {/* Name */}
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{sub.name}</p>
                  <p className="text-xs truncate" style={{ color: 'var(--text-muted)' }}>{sub.creator}</p>
                </div>

                {/* Webhook dot */}
                <span className="w-5 flex justify-center">
                  <span className="w-2.5 h-2.5 rounded-full" style={{ background: webhookDot[sub.webhookStatus] }} title={sub.webhookStatus} />
                </span>

                {/* Cost */}
                <span className="w-20 text-right text-sm" style={{ color: sub.cost === 0 ? 'var(--green)' : 'var(--text-primary)' }}>
                  {sub.cost === 0 ? 'Free' : `$${sub.cost}/mo`}
                </span>

                {/* Alerts */}
                <span className="w-16 text-right text-sm" style={{ color: 'var(--text-secondary)' }}>{sub.alerts}</span>

                {/* Inline webhook URL */}
                <div className="w-64">
                  {editingWebhook === sub.id ? (
                    <div className="flex gap-1">
                      <input
                        className="flex-1 px-2 py-1 rounded text-xs font-mono"
                        style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                        defaultValue={sub.webhookUrl}
                        placeholder="https://..."
                        autoFocus
                      />
                      <button
                        onClick={() => testWebhook(sub.id)}
                        className="px-2 py-1 rounded text-xs"
                        style={{ background: 'var(--accent)', color: 'white' }}
                      >
                        {testingId === sub.id ? '...' : 'Test'}
                      </button>
                      <button
                        onClick={() => setEditingWebhook(null)}
                        className="px-2 py-1 rounded text-xs"
                        style={{ background: 'var(--green)', color: 'white' }}
                      >
                        Save
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => setEditingWebhook(sub.id)}
                      className="text-xs font-mono truncate block w-full text-left px-2 py-1 rounded hover:bg-[var(--bg-input)] transition-colors"
                      style={{ color: sub.webhookUrl ? 'var(--text-secondary)' : 'var(--text-muted)' }}
                    >
                      {sub.webhookUrl || 'Click to set webhook URL...'}
                    </button>
                  )}
                </div>

                {/* Toggle */}
                <span className="w-16 flex justify-end">
                  <button
                    onClick={() => toggleStatus(sub.id)}
                    className="w-10 h-5 rounded-full relative flex-shrink-0 transition-colors"
                    style={{ background: sub.status === 'active' ? 'var(--accent)' : 'var(--bg-input)', border: sub.status === 'active' ? 'none' : '1px solid var(--border)' }}
                  >
                    <div className="w-3.5 h-3.5 rounded-full absolute top-0.5 transition-all" style={{ background: sub.status === 'active' ? 'white' : 'var(--text-muted)', left: sub.status === 'active' ? '22px' : '3px' }} />
                  </button>
                </span>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Legend */}
      <div className="flex items-center gap-6 mt-4">
        <div className="flex items-center gap-4">
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Status:</span>
          <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full" style={{ background: 'var(--green)' }} /><span className="text-xs" style={{ color: 'var(--text-muted)' }}>Active</span></div>
          <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full" style={{ background: 'var(--text-muted)' }} /><span className="text-xs" style={{ color: 'var(--text-muted)' }}>Paused</span></div>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Webhook:</span>
          <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full" style={{ background: 'var(--green)' }} /><span className="text-xs" style={{ color: 'var(--text-muted)' }}>Connected</span></div>
          <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full" style={{ background: 'var(--orange)' }} /><span className="text-xs" style={{ color: 'var(--text-muted)' }}>Not Set Up</span></div>
          <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full" style={{ background: 'var(--red)' }} /><span className="text-xs" style={{ color: 'var(--text-muted)' }}>Error</span></div>
        </div>
      </div>
    </div>
  );
}
