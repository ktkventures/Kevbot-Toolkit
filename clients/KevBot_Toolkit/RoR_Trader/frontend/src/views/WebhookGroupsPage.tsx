'use client';

/**
 * Webhook Groups — account-based webhook groups (11 events per group).
 * Each group represents an account/service with payloads for each exec event.
 */

import { useState, useMemo } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import Modal from '@/components/Modal';
import {
  useWebhookGroups,
  useCreateWebhookGroup,
  useDeleteWebhookGroup,
  useDuplicateWebhookGroup,
  type WebhookGroup,
} from '@/hooks/queries/useWebhookGroups';

/* ========================================================================= */
/* STYLES                                                                      */
/* ========================================================================= */

const SERVICE_PRESETS = [
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

function countConfiguredEvents(group: WebhookGroup): number {
  const events = group.events || {};
  const isShared = (group as any).url_mode === 'shared';
  const sharedUrlSet = ((group as any).shared_url || '').trim().length > 0;
  return Object.values(events).filter((c: any) => {
    if (!c) return false;
    const hasUrl = isShared ? sharedUrlSet : ((c.url || '').trim().length > 0);
    const hasPayload = (c.payload || '').trim().length > 0;
    return hasUrl && hasPayload;
  }).length;
}

export default function WebhookGroupsPage() {
  const { data: groups, isLoading, error } = useWebhookGroups();
  const createMutation = useCreateWebhookGroup();
  const deleteMutation = useDeleteWebhookGroup();
  const duplicateMutation = useDuplicateWebhookGroup();

  const router = useRouter();
  const [search, setSearch] = useState('');
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState('');
  const [newService, setNewService] = useState('signalstack');

  const filtered = useMemo(() => {
    if (!groups) return [];
    const q = search.toLowerCase();
    return groups.filter((g) =>
      (g.name || '').toLowerCase().includes(q) ||
      (g.service || '').toLowerCase().includes(q)
    );
  }, [groups, search]);

  const selectedPreset = SERVICE_PRESETS.find((p) => p.id === newService);

  const handleCreate = () => {
    createMutation.mutate(
      { name: newName || `${selectedPreset?.name || 'New'} Group`, service: newService },
      {
        onSuccess: (data) => {
          setShowCreate(false);
          router.push(`/alerts/webhook-groups/${data.id}`);
        },
      }
    );
  };

  if (isLoading) {
    return (
      <div>
        <PageHeader title="Webhook Groups" subtitle="Loading…" />
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
        <PageHeader title="Webhook Groups" subtitle="Error loading groups" />
        <Card>
          <div className="text-center py-8" style={{ color: 'var(--red)' }}>
            Failed to load webhook groups. Check your connection and try again.
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div>
      <PageHeader
        title="Webhook Groups"
        subtitle="Account-based webhook groups. Each group contains URL + payload for all 11 execution-driven events — set up once per account, then assign to any portfolio."
        actions={
          <button
            style={btnPrimary}
            onClick={() => { setShowCreate(true); setNewName(''); setNewService('signalstack'); }}
          >
            + New Group
          </button>
        }
      />

      {/* Create modal */}
      <Modal title="New Webhook Group" isOpen={showCreate} onClose={() => setShowCreate(false)} width="540px">
        <div className="mb-5">
          <label className="text-xs font-medium block mb-2" style={{ color: 'var(--text-muted)' }}>Service / Account Type</label>
          <div className="grid grid-cols-2 gap-2">
            {SERVICE_PRESETS.map((preset) => (
              <button
                key={preset.id}
                className="flex flex-col items-start px-3 py-2.5 rounded-lg text-left transition-colors"
                style={{
                  background: newService === preset.id ? 'var(--accent-muted)' : 'var(--bg-input)',
                  border: newService === preset.id ? '1px solid var(--accent)' : '1px solid var(--border)',
                  cursor: 'pointer',
                }}
                onClick={() => setNewService(preset.id)}
              >
                <span className="text-sm font-medium" style={{ color: newService === preset.id ? 'var(--accent)' : 'var(--text-primary)' }}>
                  {preset.name}
                </span>
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{preset.description}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="mb-6">
          <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-muted)' }}>Group Name</label>
          <input
            type="text"
            placeholder={`${selectedPreset?.name || ''} — My Account`}
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            style={inputStyle}
          />
          <p className="text-xs mt-1.5" style={{ color: 'var(--text-muted)' }}>
            All 11 event slots will start empty. Fill in the ones you need on the detail page — blanks are skipped at fire time.
          </p>
        </div>

        <div className="flex justify-end gap-2">
          <button style={btnSecondary} onClick={() => setShowCreate(false)}>Cancel</button>
          <button
            style={{ ...btnPrimary, opacity: createMutation.isPending ? 0.5 : 1, cursor: createMutation.isPending ? 'not-allowed' : 'pointer' }}
            onClick={handleCreate}
            disabled={createMutation.isPending}
          >
            {createMutation.isPending ? 'Creating…' : 'Create Group'}
          </button>
        </div>
      </Modal>

      {/* Search */}
      <div className="mb-5">
        <input
          type="text"
          placeholder="Search groups…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full sm:w-80"
          style={inputStyle}
        />
      </div>

      {/* Group cards */}
      {filtered.length === 0 ? (
        <Card>
          <div className="py-12 text-center">
            <p className="text-sm mb-2" style={{ color: 'var(--text-muted)' }}>
              {search ? 'No groups match your search.' : 'No webhook groups yet.'}
            </p>
            {!search && (
              <button
                style={btnPrimary}
                onClick={() => setShowCreate(true)}
              >
                + Create your first group
              </button>
            )}
          </div>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered.map((g) => {
            const configured = countConfiguredEvents(g);
            return (
              <Link
                key={g.id}
                href={`/alerts/webhook-groups/${g.id}`}
                style={{ textDecoration: 'none', color: 'inherit' }}
              >
                <Card className="h-full hover:border-accent transition-colors">
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <p className="text-sm font-semibold">{g.name || 'Untitled Group'}</p>
                      {g.service && (
                        <span className="text-xs px-2 py-0.5 rounded-full mt-1 inline-block" style={{ color: 'var(--accent)', background: 'var(--accent-muted)' }}>
                          {g.service}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Metrics row */}
                  <div className="grid grid-cols-2 gap-3 mb-3">
                    <div>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Events Configured</p>
                      <p className="text-sm font-bold">
                        {configured} / 11
                      </p>
                    </div>
                    <div>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Status</p>
                      <p className="text-sm font-bold" style={{ color: configured > 0 ? 'var(--green)' : 'var(--text-muted)' }}>
                        {configured === 0 ? 'Empty' : configured === 11 ? 'Fully configured' : 'Partial'}
                      </p>
                    </div>
                  </div>

                  {/* Created */}
                  {g.created_at && (
                    <p className="text-xs font-mono mt-3" style={{ color: 'var(--text-muted)' }}>
                      Created {new Date(g.created_at).toLocaleDateString()}
                    </p>
                  )}
                </Card>
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
}
