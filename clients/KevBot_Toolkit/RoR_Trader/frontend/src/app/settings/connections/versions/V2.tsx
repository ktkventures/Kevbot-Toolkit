'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import MetricCard from '@/components/MetricCard';

/* ========================================================================= */
/* TYPES                                                                      */
/* ========================================================================= */

interface ConnectionService {
  name: string;
  provider: string;
  status: 'connected' | 'disconnected' | 'error';
  details: string;
  apiKeyMasked?: string;
  planTier?: string;
  rateLimit?: string;
}

/* ========================================================================= */
/* MOCK DATA                                                                  */
/* ========================================================================= */

const mockServices: ConnectionService[] = [
  {
    name: 'Market Data',
    provider: 'Polygon.io',
    status: 'connected',
    details: 'REST + WebSocket active',
    apiKeyMasked: '****...3x7f',
    planTier: 'Stocks Advanced ($199/mo)',
    rateLimit: '10,000 calls/min',
  },
  {
    name: 'Broker',
    provider: 'Alpaca',
    status: 'disconnected',
    details: 'Paper trading account',
    apiKeyMasked: '****...k9m2',
    planTier: 'Free (Paper)',
    rateLimit: '200 calls/min',
  },
  {
    name: 'Database',
    provider: 'Supabase',
    status: 'connected',
    details: 'PostgreSQL + Auth + Realtime',
  },
  {
    name: 'Deployment',
    provider: 'Railway',
    status: 'connected',
    details: 'Web service + Worker service',
  },
];

/* ========================================================================= */
/* HELPERS                                                                    */
/* ========================================================================= */

function statusColor(status: ConnectionService['status']): string {
  if (status === 'connected') return 'var(--green)';
  if (status === 'error') return 'var(--red)';
  return 'var(--text-muted)';
}

function statusLabel(status: ConnectionService['status']): string {
  if (status === 'connected') return 'Connected';
  if (status === 'error') return 'Error';
  return 'Not Connected';
}

/* ========================================================================= */
/* COMPONENT                                                                  */
/* ========================================================================= */

export default function SettingsConnectionsV2() {
  const [services] = useState(mockServices);
  const [showApiKey, setShowApiKey] = useState<Record<string, boolean>>({});
  const [healthChecking, setHealthChecking] = useState(false);

  const connectedCount = services.filter((s) => s.status === 'connected').length;

  function handleHealthCheck() {
    setHealthChecking(true);
    setTimeout(() => setHealthChecking(false), 2000);
  }

  return (
    <div>
      <PageHeader
        title="Connections"
        subtitle="Manage data providers, broker, and infrastructure connections"
        actions={
          <button
            onClick={handleHealthCheck}
            className="px-4 py-2 rounded-lg text-sm font-medium"
            style={{
              background: 'var(--accent-muted)',
              color: 'var(--accent)',
              opacity: healthChecking ? 0.6 : 1,
            }}
            disabled={healthChecking}
          >
            {healthChecking ? 'Checking...' : 'Health Check'}
          </button>
        }
      />

      {/* Summary */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <MetricCard label="Total Services" value={services.length.toString()} />
        <MetricCard label="Connected" value={connectedCount.toString()} positive />
        <MetricCard label="Disconnected" value={(services.length - connectedCount).toString()} />
        <MetricCard label="Last Check" value="Just now" />
      </div>

      {/* Service Cards */}
      <div className="max-w-3xl space-y-4">
        {services.map((svc) => (
          <Card key={svc.name}>
            <div className="flex items-start justify-between mb-4">
              <div>
                <div className="flex items-center gap-3 mb-1">
                  <h3 className="font-semibold">{svc.name}</h3>
                  <span
                    className="text-xs px-2 py-0.5 rounded-full"
                    style={{ background: 'var(--bg-input)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}
                  >
                    {svc.provider}
                  </span>
                </div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{svc.details}</p>
              </div>
              <div className="flex items-center gap-2">
                <span
                  className="w-2.5 h-2.5 rounded-full"
                  style={{
                    background: statusColor(svc.status),
                    boxShadow: svc.status === 'connected' ? `0 0 6px ${statusColor(svc.status)}` : 'none',
                  }}
                />
                <span className="text-xs font-medium" style={{ color: statusColor(svc.status) }}>
                  {statusLabel(svc.status)}
                </span>
              </div>
            </div>

            {/* API Key & Plan details (for services that have them) */}
            {(svc.apiKeyMasked || svc.planTier) && (
              <div
                className="rounded-lg p-3 space-y-2"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
              >
                {svc.apiKeyMasked && (
                  <div className="flex items-center justify-between">
                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>API Key</span>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>
                        {showApiKey[svc.name] ? 'pk_live_abc123def456ghi789jkl' : svc.apiKeyMasked}
                      </span>
                      <button
                        onClick={() => setShowApiKey((prev) => ({ ...prev, [svc.name]: !prev[svc.name] }))}
                        className="text-xs px-2 py-0.5 rounded"
                        style={{ background: 'var(--bg-card)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}
                      >
                        {showApiKey[svc.name] ? 'Hide' : 'Show'}
                      </button>
                    </div>
                  </div>
                )}
                {svc.planTier && (
                  <div className="flex items-center justify-between">
                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Plan</span>
                    <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>{svc.planTier}</span>
                  </div>
                )}
                {svc.rateLimit && (
                  <div className="flex items-center justify-between">
                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Rate Limit</span>
                    <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>{svc.rateLimit}</span>
                  </div>
                )}
              </div>
            )}

            {/* Action buttons */}
            <div className="flex gap-2 mt-3">
              {svc.status === 'disconnected' && (
                <button
                  className="px-3 py-1.5 rounded-lg text-xs font-medium"
                  style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                >
                  Connect
                </button>
              )}
              {svc.apiKeyMasked && (
                <button
                  className="px-3 py-1.5 rounded-lg text-xs"
                  style={{ background: 'var(--bg-input)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}
                >
                  Update Key
                </button>
              )}
              <button
                className="px-3 py-1.5 rounded-lg text-xs"
                style={{ background: 'var(--bg-input)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}
              >
                Test Connection
              </button>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
