'use client';

/**
 * Webhook Templates — Clean API-first page.
 *
 * Visual design derived from V5 (versions/V5.tsx), data layer built
 * around the webhook templates API endpoint. No mock data.
 */

import { useState, useMemo } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import { useWebhookTemplates } from '@/hooks/queries/useWebhooks';
import { useDeleteWebhookTemplate } from '@/hooks/mutations/useWebhookMutations';

// ---------------------------------------------------------------------------
// Constants — exchange presets matching V5
// ---------------------------------------------------------------------------

const EXCHANGE_PRESETS: Record<string, { color: string; bg: string }> = {
  SignalStack: { color: '#2196F3', bg: '#2196F320' },
  TradeThePool: { color: '#FF9800', bg: '#FF980020' },
  Discord: { color: '#5865F2', bg: '#5865F220' },
  Slack: { color: '#4A154B', bg: '#4A154B20' },
  Custom: { color: 'var(--text-muted)', bg: 'var(--bg-input)' },
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function ExchangeBadge({ exchange }: { exchange: string }) {
  const preset = EXCHANGE_PRESETS[exchange] || EXCHANGE_PRESETS.Custom;
  return (
    <span
      className="text-xs font-medium px-2 py-0.5 rounded-full"
      style={{ color: preset.color, background: preset.bg }}
    >
      {exchange}
    </span>
  );
}

function StatusDot({ status }: { status?: number }) {
  if (!status) return <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Never sent</span>;
  const ok = status >= 200 && status < 300;
  return (
    <span className="flex items-center gap-1.5">
      <span style={{ width: 6, height: 6, borderRadius: '50%', background: ok ? 'var(--green)' : 'var(--red)', display: 'inline-block' }} />
      <span className="text-xs" style={{ color: ok ? 'var(--green)' : 'var(--red)' }}>{ok ? 'Healthy' : `Error ${status}`}</span>
    </span>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function WebhookTemplatesPage() {
  const { data: templates, isLoading, error } = useWebhookTemplates();
  const deleteMutation = useDeleteWebhookTemplate();

  const [search, setSearch] = useState('');

  const filtered = useMemo(() => {
    if (!templates) return [];
    if (!search.trim()) return templates;
    const q = search.toLowerCase();
    return templates.filter((t: any) =>
      (t.name || '').toLowerCase().includes(q) ||
      (t.exchange || t.service || '').toLowerCase().includes(q)
    );
  }, [templates, search]);

  // ---------------------------------------------------------------------------
  // Loading / Error
  // ---------------------------------------------------------------------------

  if (isLoading) {
    return (
      <div>
        <PageHeader title="Webhook Templates" subtitle="Loading..." />
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-4">
          {[1, 2, 3].map((i) => (
            <Card key={i}>
              <div className="animate-pulse space-y-3">
                <div className="h-4 rounded w-2/3" style={{ background: 'var(--border)' }} />
                <div className="h-3 rounded w-1/3" style={{ background: 'var(--border)' }} />
                <div className="grid grid-cols-3 gap-2">
                  {[1, 2, 3].map((j) => (
                    <div key={j} className="h-8 rounded" style={{ background: 'var(--border)' }} />
                  ))}
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div>
        <PageHeader title="Webhook Templates" subtitle="Error" />
        <Card>
          <div className="text-center py-8" style={{ color: 'var(--red)' }}>
            Failed to load webhook templates. Check your connection and try again.
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div>
      <PageHeader
        title="Webhook Templates"
        subtitle="Account-based templates define how your system communicates with each exchange. Each template contains payloads for all 11 webhook event types."
        actions={
          <div className="flex items-center gap-2">
            <span
              className="text-xs px-2 py-1 rounded-full flex items-center gap-1.5"
              style={{ background: 'var(--green)' + '15', color: 'var(--green)' }}
            >
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--green)' }} />
              Live
            </span>
            <Link href="/alerts/webhook-templates/new">
              <button
                className="px-4 py-2 rounded-lg text-sm font-medium"
                style={{ background: 'var(--accent)', color: '#fff' }}
              >
                + New Template
              </button>
            </Link>
          </div>
        }
      />

      {/* Exchange preset legend */}
      <div className="flex flex-wrap gap-2 mt-4 mb-3">
        {Object.keys(EXCHANGE_PRESETS).map((name) => (
          <ExchangeBadge key={name} exchange={name} />
        ))}
      </div>

      {/* Search */}
      <div className="mb-5">
        <input
          type="text"
          placeholder="Search templates..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full sm:w-80"
          style={{
            background: 'var(--bg-input)',
            border: '1px solid var(--border)',
            color: 'var(--text-primary)',
            padding: '8px 14px',
            borderRadius: '8px',
            fontSize: '0.875rem',
          }}
        />
      </div>

      {/* Empty state */}
      {filtered.length === 0 && (
        <Card>
          <div className="text-center py-12">
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              {(templates?.length || 0) === 0
                ? 'No webhook templates yet. Create your first template to start sending signals.'
                : 'No templates match your search.'}
            </p>
          </div>
        </Card>
      )}

      {/* Template cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filtered.map((tpl: any) => {
          const tplId = tpl.id || tpl.template_id || '';
          const exchange = tpl.exchange || tpl.service || 'Custom';
          const url = tpl.url || tpl.webhook_url || '';
          const eventCount = tpl.event_count ?? 11;
          const portfolioCount = tpl.used_by_portfolios ?? 0;
          const lastDelivery = tpl.last_delivery;
          const isDefault = tpl.is_default ?? false;

          return (
            <Link
              key={tplId}
              href={`/alerts/webhook-templates/${tplId}`}
              style={{ textDecoration: 'none', color: 'inherit' }}
            >
              <Card className="h-full">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <p className="text-sm font-semibold">{tpl.name || 'Untitled'}</p>
                    <div className="flex items-center gap-2 mt-1">
                      <ExchangeBadge exchange={exchange} />
                      {isDefault && (
                        <span className="text-xs px-2 py-0.5 rounded-full" style={{ color: 'var(--text-muted)', background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                          Default
                        </span>
                      )}
                    </div>
                  </div>
                  <StatusDot status={lastDelivery?.status} />
                </div>

                {/* Metrics row */}
                <div className="grid grid-cols-3 gap-3 mb-3">
                  <div>
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Portfolios</p>
                    <p className="text-sm font-bold">{portfolioCount}</p>
                  </div>
                  <div>
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Events</p>
                    <p className="text-sm font-bold">{eventCount}</p>
                  </div>
                  <div>
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Last Sent</p>
                    <p className="text-xs font-mono mt-0.5">
                      {lastDelivery?.time ? lastDelivery.time.split(' ')[1] || lastDelivery.time : '--'}
                    </p>
                  </div>
                </div>

                {/* URL preview truncated */}
                {url && (
                  <p
                    className="text-xs font-mono mt-3 truncate"
                    style={{ color: 'var(--text-muted)', maxWidth: '100%' }}
                    title={url}
                  >
                    {url.length > 50 ? url.slice(0, 50) + '...' : url}
                  </p>
                )}
              </Card>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
