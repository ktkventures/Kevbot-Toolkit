'use client';

/**
 * Webhook Templates — Faithful copy of V5 design with mock data replaced by API hooks.
 * Source: V5 design file (323 lines)
 *
 * Data convention:
 *   Real value = live from API
 *   -- = wired but no data yet
 *   {{field}} = not wired, needs backend work
 */

import { useState, useMemo } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import Modal from '@/components/Modal';
import { useWebhookTemplates } from '@/hooks/queries/useWebhooks';
import { useDeleteWebhookTemplate } from '@/hooks/mutations/useWebhookMutations';
import { useCreateWebhookTemplate } from '@/hooks/mutations/useWebhookMutations';

/* ========================================================================= */
/* TYPES                                                                      */
/* ========================================================================= */

interface AccountTemplate {
  id: string;
  name: string;
  exchange: string;
  url: string;
  isDefault: boolean;
  usedByPortfolios: number;
  eventCount: number;
  lastDelivery?: { time: string; status: number };
  entryCfg: string;
  exitCfg: string;
}

/* ========================================================================= */
/* API → V5 TEMPLATE MAPPING                                                   */
/* ========================================================================= */

function apiToTemplate(t: any): AccountTemplate {
  return {
    id: String(t.id ?? ''),
    name: t.name ?? '--',
    exchange: t.exchange ?? t.service ?? '--',
    url: t.url ?? t.webhook_url ?? '',
    isDefault: t.is_default ?? false,
    usedByPortfolios: t.used_by_portfolios ?? t.portfolio_count ?? 0,
    eventCount: t.event_count ?? 11,
    lastDelivery: t.last_delivery ? {
      time: t.last_delivery.time ?? t.last_delivery.timestamp ?? '--',
      status: t.last_delivery.status ?? t.last_delivery.status_code ?? 0,
    } : undefined,
    entryCfg: t.entry_cfg ?? t.entry_config ?? '--',
    exitCfg: t.exit_cfg ?? t.exit_config ?? '--',
  };
}

/* ========================================================================= */
/* STYLES                                                                      */
/* ========================================================================= */

const EXCHANGE_PRESETS = [
  { id: 'signalstack', name: 'SignalStack', description: 'Automated order execution via webhook', urlHint: 'https://app.signalstack.com/hook/...' },
  { id: 'tradethepool', name: 'TradeThePool', description: 'Prop firm trading platform', urlHint: 'https://api.tradethepool.com/webhook/...' },
  { id: 'discord', name: 'Discord', description: 'Post alerts to a Discord channel', urlHint: 'https://discord.com/api/webhooks/...' },
  { id: 'slack', name: 'Slack', description: 'Post alerts to a Slack channel', urlHint: 'https://hooks.slack.com/services/...' },
  { id: 'custom', name: 'Custom', description: 'Any webhook endpoint', urlHint: 'https://...' },
];

const inputStyle: React.CSSProperties = {
  background: 'var(--bg-input)',
  border: '1px solid var(--border)',
  color: 'var(--text-primary)',
  padding: '8px 14px',
  borderRadius: '8px',
  fontSize: '0.875rem',
  width: '100%',
};

const btnSecondary: React.CSSProperties = {
  background: 'var(--bg-card)',
  border: '1px solid var(--border)',
  color: 'var(--text-secondary)',
  padding: '6px 14px',
  borderRadius: '8px',
  fontSize: '0.875rem',
  cursor: 'pointer',
};

const btnPrimary: React.CSSProperties = {
  background: 'var(--accent)',
  color: 'white',
  border: 'none',
  padding: '8px 16px',
  borderRadius: '8px',
  fontSize: '0.875rem',
  cursor: 'pointer',
  fontWeight: 600,
};

/* ========================================================================= */
/* COMPONENT                                                                   */
/* ========================================================================= */

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

export default function WebhookTemplatesPage() {
  /* ---- API hooks (all before any early return) ---- */
  const { data: rawTemplates, isLoading, error } = useWebhookTemplates();
  const deleteMutation = useDeleteWebhookTemplate();
  const createMutation = useCreateWebhookTemplate();

  const router = useRouter();
  const [search, setSearch] = useState('');
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState('');
  const [newExchange, setNewExchange] = useState('signalstack');
  const [newUrl, setNewUrl] = useState('');

  /* ---- Derived data from API ---- */
  const templates: AccountTemplate[] = useMemo(() => {
    if (!rawTemplates) return [];
    return (rawTemplates as any[]).map(apiToTemplate);
  }, [rawTemplates]);

  const filtered = useMemo(() => {
    return templates.filter((t) =>
      t.name.toLowerCase().includes(search.toLowerCase()) ||
      t.exchange.toLowerCase().includes(search.toLowerCase())
    );
  }, [templates, search]);

  const selectedPreset = EXCHANGE_PRESETS.find((p) => p.id === newExchange);

  const handleCreate = () => {
    createMutation.mutate(
      { name: newName, exchange: newExchange, url: newUrl },
      {
        onSuccess: (data: any) => {
          setShowCreate(false);
          const newId = data?.id ?? 'new';
          router.push(`/alerts/webhook-templates/${newId}`);
        },
      }
    );
  };

  /* ---- Loading / Error states ---- */
  if (isLoading) {
    return (
      <div>
        <PageHeader title="Webhook Templates" subtitle="Loading..." />
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-4">
          {[1, 2, 3].map((i) => (
            <Card key={i}>
              <div className="animate-pulse space-y-3">
                <div className="h-4 rounded w-1/3" style={{ background: 'var(--border)' }} />
                <div className="h-3 rounded w-2/3" style={{ background: 'var(--border)' }} />
                <div className="h-16 rounded" style={{ background: 'var(--bg-input)' }} />
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
        <PageHeader title="Webhook Templates" subtitle="Error loading templates" />
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
          <button style={btnPrimary} onClick={() => { setShowCreate(true); setNewName(''); setNewUrl(''); setNewExchange('signalstack'); }}>
            + New Template
          </button>
        }
      />

      {/* Create modal */}
      <Modal title="New Webhook Template" isOpen={showCreate} onClose={() => setShowCreate(false)} width="540px">
        {/* Step 1: Pick exchange */}
        <div className="mb-5">
          <label className="text-xs font-medium block mb-2" style={{ color: 'var(--text-muted)' }}>Exchange / Service</label>
          <div className="grid grid-cols-2 gap-2">
            {EXCHANGE_PRESETS.map((preset) => (
              <button
                key={preset.id}
                className="flex flex-col items-start px-3 py-2.5 rounded-lg text-left transition-colors"
                style={{
                  background: newExchange === preset.id ? 'var(--accent-muted)' : 'var(--bg-input)',
                  border: newExchange === preset.id ? '1px solid var(--accent)' : '1px solid var(--border)',
                  cursor: 'pointer',
                }}
                onClick={() => setNewExchange(preset.id)}
              >
                <span className="text-sm font-medium" style={{ color: newExchange === preset.id ? 'var(--accent)' : 'var(--text-primary)' }}>
                  {preset.name}
                </span>
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{preset.description}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Step 2: Name */}
        <div className="mb-4">
          <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-muted)' }}>Template Name</label>
          <input
            type="text"
            placeholder={`${selectedPreset?.name || ''} — My Account`}
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            style={inputStyle}
          />
        </div>

        {/* Step 3: URL */}
        <div className="mb-6">
          <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-muted)' }}>Webhook URL</label>
          <input
            type="text"
            placeholder={selectedPreset?.urlHint || 'https://...'}
            value={newUrl}
            onChange={(e) => setNewUrl(e.target.value)}
            style={inputStyle}
          />
          <p className="text-xs mt-1.5" style={{ color: 'var(--text-muted)' }}>
            All 11 event payloads will be pre-filled with {selectedPreset?.name || 'default'} defaults. You can customize each one on the detail page.
          </p>
        </div>

        {/* Actions */}
        <div className="flex justify-end gap-2">
          <button style={btnSecondary} onClick={() => setShowCreate(false)}>Cancel</button>
          <button style={btnPrimary} onClick={handleCreate}>Create Template</button>
        </div>
      </Modal>

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

      {/* Template cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filtered.map((tpl) => (
          <Link
            key={tpl.id}
            href={`/alerts/webhook-templates/${tpl.id}`}
            style={{ textDecoration: 'none', color: 'inherit' }}
          >
            <Card className="h-full hover:border-accent transition-colors">
              <div className="flex items-start justify-between mb-3">
                <div>
                  <p className="text-sm font-semibold">{tpl.name}</p>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-xs px-2 py-0.5 rounded-full" style={{ color: 'var(--accent)', background: 'var(--accent-muted)' }}>
                      {tpl.exchange}
                    </span>
                    {tpl.isDefault && (
                      <span className="text-xs px-2 py-0.5 rounded-full" style={{ color: 'var(--text-muted)', background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                        Default
                      </span>
                    )}
                  </div>
                </div>
                <StatusDot status={tpl.lastDelivery?.status} />
              </div>

              {/* Metrics row */}
              <div className="grid grid-cols-3 gap-3 mb-3">
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Portfolios</p>
                  <p className="text-sm font-bold">{tpl.usedByPortfolios}</p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Events</p>
                  <p className="text-sm font-bold">{tpl.eventCount}</p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Last Sent</p>
                  <p className="text-xs font-mono mt-0.5">{tpl.lastDelivery ? tpl.lastDelivery.time.split(' ')[1] : '—'}</p>
                </div>
              </div>

              {/* URL */}
              {tpl.url && (
                <p className="text-xs font-mono mt-3 truncate" style={{ color: 'var(--text-muted)' }}>
                  {tpl.url}
                </p>
              )}
            </Card>
          </Link>
        ))}
      </div>

      {filtered.length === 0 && (
        <Card>
          <div className="py-12 text-center">
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              No templates match your search.
            </p>
          </div>
        </Card>
      )}
    </div>
  );
}
