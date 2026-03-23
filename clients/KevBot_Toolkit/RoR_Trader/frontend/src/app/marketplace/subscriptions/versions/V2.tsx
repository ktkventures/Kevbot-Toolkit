'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import MetricCard from '@/components/MetricCard';

interface AlertEntry {
  time: string;
  signal: string;
  symbol: string;
  result: 'win' | 'loss' | 'pending';
}

interface Subscription {
  id: string;
  portfolioName: string;
  creator: string;
  status: 'active' | 'paused';
  monthlyCost: number;
  subscribedDate: string;
  trades: number;
  winRate: number;
  pnl: number;
  alertsReceived: number;
  webhookStatus: 'connected' | 'not_setup' | 'error';
  webhookUrl: string;
  broker: string;
  riskPerTrade: number;
  recentAlerts: AlertEntry[];
}

const mockSubscriptions: Subscription[] = [
  {
    id: 'sub1', portfolioName: 'Momentum Alpha', creator: 'TradeMaster', status: 'active', monthlyCost: 49,
    subscribedDate: '2025-11-15', trades: 87, winRate: 57.5, pnl: 1240, alertsReceived: 312,
    webhookStatus: 'connected', webhookUrl: 'https://webhook.ttp.com/abc123', broker: 'TTP', riskPerTrade: 1.5,
    recentAlerts: [
      { time: '2026-03-21 09:45', signal: 'LONG NVDA', symbol: 'NVDA', result: 'win' },
      { time: '2026-03-21 10:12', signal: 'LONG AAPL', symbol: 'AAPL', result: 'loss' },
      { time: '2026-03-20 14:30', signal: 'SHORT TSLA', symbol: 'TSLA', result: 'win' },
      { time: '2026-03-20 11:15', signal: 'LONG AMD', symbol: 'AMD', result: 'win' },
      { time: '2026-03-19 09:32', signal: 'LONG MSFT', symbol: 'MSFT', result: 'pending' },
    ],
  },
  {
    id: 'sub2', portfolioName: 'Steady Eddie', creator: 'RoR Team', status: 'active', monthlyCost: 0,
    subscribedDate: '2025-10-01', trades: 142, winRate: 52.1, pnl: 580, alertsReceived: 487,
    webhookStatus: 'connected', webhookUrl: 'https://webhook.alpaca.com/xyz789', broker: 'Alpaca', riskPerTrade: 1.0,
    recentAlerts: [
      { time: '2026-03-21 09:31', signal: 'LONG SPY', symbol: 'SPY', result: 'pending' },
      { time: '2026-03-20 15:45', signal: 'SHORT QQQ', symbol: 'QQQ', result: 'win' },
      { time: '2026-03-20 10:20', signal: 'LONG AMZN', symbol: 'AMZN', result: 'loss' },
      { time: '2026-03-19 14:05', signal: 'LONG GOOG', symbol: 'GOOG', result: 'win' },
      { time: '2026-03-19 09:48', signal: 'LONG META', symbol: 'META', result: 'win' },
    ],
  },
  {
    id: 'sub3', portfolioName: 'VWAP Scalper Pro', creator: 'QuantEdge', status: 'paused', monthlyCost: 29,
    subscribedDate: '2026-01-10', trades: 34, winRate: 47.1, pnl: -120, alertsReceived: 89,
    webhookStatus: 'error', webhookUrl: 'https://webhook.ttp.com/broken', broker: 'TTP', riskPerTrade: 2.0,
    recentAlerts: [
      { time: '2026-02-28 10:15', signal: 'LONG NVDA', symbol: 'NVDA', result: 'loss' },
      { time: '2026-02-27 09:45', signal: 'SHORT AMD', symbol: 'AMD', result: 'loss' },
      { time: '2026-02-27 11:30', signal: 'LONG TSLA', symbol: 'TSLA', result: 'win' },
    ],
  },
  {
    id: 'sub4', portfolioName: 'Tech Momentum', creator: 'AlgoTrader99', status: 'active', monthlyCost: 39,
    subscribedDate: '2026-02-20', trades: 0, winRate: 0, pnl: 0, alertsReceived: 0,
    webhookStatus: 'not_setup', webhookUrl: '', broker: '', riskPerTrade: 1.0,
    recentAlerts: [],
  },
];

const webhookStyles: Record<string, { color: string; bg: string; label: string }> = {
  connected: { color: 'var(--green)', bg: 'var(--green-muted)', label: 'Connected' },
  not_setup: { color: 'var(--orange)', bg: 'var(--orange-muted)', label: 'Not Set Up' },
  error: { color: 'var(--red)', bg: 'var(--red-muted)', label: 'Error' },
};

const resultColors: Record<string, string> = {
  win: 'var(--green)',
  loss: 'var(--red)',
  pending: 'var(--orange)',
};

const brokers = ['TTP', 'Alpaca', 'Custom'];

export default function SubscriptionsV2() {
  const [subs, setSubs] = useState(mockSubscriptions);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [wizardStep, setWizardStep] = useState<Record<string, number>>({});
  const [testResult, setTestResult] = useState<Record<string, 'success' | 'fail' | null>>({});

  const totalCost = subs.filter((s) => s.status === 'active').reduce((sum, s) => sum + s.monthlyCost, 0);
  const activeCount = subs.filter((s) => s.status === 'active').length;
  const totalPnl = subs.reduce((sum, s) => sum + s.pnl, 0);

  function toggleStatus(id: string) {
    setSubs((prev) => prev.map((s) => s.id === id ? { ...s, status: s.status === 'active' ? 'paused' : 'active' } : s));
  }

  function getWizardStep(id: string) { return wizardStep[id] || 0; }
  function setStep(id: string, step: number) { setWizardStep((prev) => ({ ...prev, [id]: step })); }

  function runTest(id: string) {
    setTestResult((prev) => ({ ...prev, [id]: null }));
    setTimeout(() => {
      setTestResult((prev) => ({ ...prev, [id]: Math.random() > 0.3 ? 'success' : 'fail' }));
    }, 1500);
  }

  return (
    <div>
      <PageHeader
        title="My Subscriptions"
        subtitle="Manage subscriptions, configure webhooks, monitor performance"
        backHref="/marketplace"
        actions={
          <button className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: 'white' }}>Browse Marketplace</button>
        }
      />

      {/* Summary */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <MetricCard label="Active Subscriptions" value={activeCount.toString()} />
        <MetricCard label="Monthly Spend" value={totalCost === 0 ? 'Free' : `$${totalCost}/mo`} />
        <MetricCard label="Total P&L" value={`${totalPnl >= 0 ? '+' : ''}$${totalPnl.toLocaleString()}`} positive={totalPnl >= 0} />
        <MetricCard label="Total Alerts" value={subs.reduce((s, sub) => s + sub.alertsReceived, 0).toLocaleString()} />
      </div>

      {/* Subscription Cards */}
      {subs.length === 0 ? (
        <Card className="text-center py-12">
          <p className="text-lg font-semibold mb-2">No Subscriptions Yet</p>
          <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>Browse the marketplace to find portfolios that match your trading style.</p>
          <button className="px-6 py-2.5 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: 'white' }}>Browse Marketplace</button>
        </Card>
      ) : (
        <div className="space-y-4">
          {subs.map((sub) => {
            const isExpanded = expandedId === sub.id;
            const step = getWizardStep(sub.id);
            const test = testResult[sub.id];

            return (
              <Card key={sub.id}>
                {/* Main row */}
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className="font-semibold text-sm">{sub.portfolioName}</h3>
                      <span className="text-xs px-2 py-0.5 rounded" style={{ background: sub.status === 'active' ? 'var(--green-muted)' : 'var(--bg-input)', color: sub.status === 'active' ? 'var(--green)' : 'var(--text-muted)' }}>
                        {sub.status}
                      </span>
                      <span className="text-xs px-2 py-0.5 rounded" style={{ background: webhookStyles[sub.webhookStatus].bg, color: webhookStyles[sub.webhookStatus].color }}>
                        {webhookStyles[sub.webhookStatus].label}
                      </span>
                    </div>
                    <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
                      by {sub.creator} &middot; Since {sub.subscribedDate} &middot; {sub.monthlyCost === 0 ? 'Free' : `$${sub.monthlyCost}/mo`}
                    </p>

                    <div className="grid grid-cols-5 gap-4 mb-3">
                      <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>Trades</p><p className="text-sm font-semibold">{sub.trades}</p></div>
                      <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>Win Rate</p><p className="text-sm font-semibold" style={{ color: sub.winRate >= 50 ? 'var(--green)' : 'var(--red)' }}>{sub.winRate > 0 ? `${sub.winRate}%` : '-'}</p></div>
                      <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>P&L</p><p className="text-sm font-semibold" style={{ color: sub.pnl >= 0 ? 'var(--green)' : 'var(--red)' }}>{sub.pnl !== 0 ? `${sub.pnl >= 0 ? '+' : ''}$${sub.pnl}` : '-'}</p></div>
                      <div><p className="text-xs" style={{ color: 'var(--text-muted)' }}>Alerts</p><p className="text-sm font-semibold">{sub.alertsReceived}</p></div>
                      <div>
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Risk/Trade</p>
                        <div className="flex items-center gap-1">
                          <input
                            type="number"
                            className="w-14 px-2 py-1 rounded text-xs text-right"
                            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                            defaultValue={sub.riskPerTrade}
                            min={0.1}
                            max={5}
                            step={0.1}
                          />
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>%</span>
                        </div>
                      </div>
                    </div>

                    {/* Recent Alerts */}
                    {sub.recentAlerts.length > 0 && (
                      <div className="mt-2">
                        <p className="text-xs font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Recent Alerts</p>
                        <div className="space-y-1">
                          {sub.recentAlerts.slice(0, 5).map((alert, idx) => (
                            <div key={idx} className="flex items-center gap-3 text-xs py-1">
                              <span style={{ color: 'var(--text-muted)' }}>{alert.time}</span>
                              <span className="font-mono">{alert.signal}</span>
                              <span className="ml-auto px-1.5 py-0.5 rounded" style={{ color: resultColors[alert.result], background: alert.result === 'win' ? 'var(--green-muted)' : alert.result === 'loss' ? 'var(--red-muted)' : 'var(--orange-muted)' }}>
                                {alert.result}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="flex flex-col gap-2 ml-4 flex-shrink-0">
                    <button onClick={() => toggleStatus(sub.id)} className="px-3 py-1.5 rounded-lg text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>
                      {sub.status === 'active' ? 'Pause' : 'Resume'}
                    </button>
                    <button onClick={() => setExpandedId(isExpanded ? null : sub.id)} className="px-3 py-1.5 rounded-lg text-xs" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                      {isExpanded ? 'Hide Webhook' : 'Configure Webhook'}
                    </button>
                  </div>
                </div>

                {/* Webhook Configuration Wizard */}
                {isExpanded && (
                  <div className="mt-4 pt-4 border-t" style={{ borderColor: 'var(--border)' }}>
                    <h4 className="text-sm font-medium mb-4">Webhook Configuration</h4>

                    {/* Step indicators */}
                    <div className="flex items-center gap-2 mb-6">
                      {['Choose Broker', 'Webhook URL', 'Test Connection', 'Set Risk'].map((label, idx) => (
                        <div key={label} className="flex items-center gap-2">
                          <button
                            onClick={() => setStep(sub.id, idx)}
                            className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-medium"
                            style={{
                              background: step === idx ? 'var(--accent)' : step > idx ? 'var(--green)' : 'var(--bg-input)',
                              color: step === idx || step > idx ? 'white' : 'var(--text-muted)',
                              border: step < idx ? '1px solid var(--border)' : 'none',
                            }}
                          >
                            {step > idx ? '✓' : idx + 1}
                          </button>
                          <span className="text-xs" style={{ color: step === idx ? 'var(--text-primary)' : 'var(--text-muted)' }}>{label}</span>
                          {idx < 3 && <div className="w-8 h-px" style={{ background: 'var(--border)' }} />}
                        </div>
                      ))}
                    </div>

                    {/* Step content */}
                    {step === 0 && (
                      <div className="space-y-3">
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Where do your webhooks go?</p>
                        <div className="grid grid-cols-3 gap-3">
                          {brokers.map((broker) => (
                            <button
                              key={broker}
                              className="p-4 rounded-lg text-center text-sm font-medium transition-colors"
                              style={{ background: sub.broker === broker ? 'var(--accent-muted)' : 'var(--bg-input)', color: sub.broker === broker ? 'var(--accent)' : 'var(--text-secondary)', border: sub.broker === broker ? '2px solid var(--accent)' : '1px solid var(--border)' }}
                              onClick={() => setStep(sub.id, 1)}
                            >
                              {broker}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                    {step === 1 && (
                      <div className="space-y-3">
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Paste your webhook URL from your broker platform</p>
                        <input
                          className="w-full px-4 py-2.5 rounded-lg text-sm font-mono"
                          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                          placeholder="https://your-webhook-url.com/..."
                          defaultValue={sub.webhookUrl}
                        />
                        <button onClick={() => setStep(sub.id, 2)} className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: 'white' }}>Next</button>
                      </div>
                    )}
                    {step === 2 && (
                      <div className="space-y-3">
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Send a test payload to verify the connection</p>
                        <button onClick={() => runTest(sub.id)} className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: 'white' }}>
                          {test === null && testResult[sub.id] === undefined ? 'Send Test' : test === null ? 'Testing...' : 'Retry Test'}
                        </button>
                        {test === 'success' && (
                          <div className="flex items-center gap-2 p-3 rounded-lg" style={{ background: 'var(--green-muted)' }}>
                            <span style={{ color: 'var(--green)' }}>&#10003;</span>
                            <span className="text-sm" style={{ color: 'var(--green)' }}>Connection successful! Test payload delivered.</span>
                          </div>
                        )}
                        {test === 'fail' && (
                          <div className="flex items-center gap-2 p-3 rounded-lg" style={{ background: 'var(--red-muted)' }}>
                            <span style={{ color: 'var(--red)' }}>&#10007;</span>
                            <span className="text-sm" style={{ color: 'var(--red)' }}>Connection failed. Check your URL and try again.</span>
                          </div>
                        )}
                        {(test === 'success') && (
                          <button onClick={() => setStep(sub.id, 3)} className="px-4 py-2 rounded-lg text-sm" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>Next</button>
                        )}
                      </div>
                    )}
                    {step === 3 && (
                      <div className="space-y-3">
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Set your risk per trade (% of account)</p>
                        <div className="flex items-center gap-3">
                          <input
                            type="number"
                            className="w-24 px-3 py-2 rounded-lg text-sm"
                            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                            defaultValue={sub.riskPerTrade}
                            min={0.1}
                            max={5}
                            step={0.1}
                          />
                          <span className="text-sm" style={{ color: 'var(--text-muted)' }}>% per trade</span>
                        </div>
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Recommended: 0.5% - 2.0% for prop firm accounts</p>
                        <button className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--green)', color: 'white' }}>Save Configuration</button>
                      </div>
                    )}
                  </div>
                )}
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
