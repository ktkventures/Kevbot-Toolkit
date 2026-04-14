'use client';

/**
 * Webhook Group Detail — edit URL + payload for each of 11 events in a group.
 * Each event has its own URL (SignalStack-style unique endpoint per action) and
 * its own JSON payload template with {{placeholders}}. Blank URL = event is skipped.
 */

import { useState, useMemo, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import {
  useWebhookGroup,
  useUpdateWebhookGroup,
  useDeleteWebhookGroup,
  useDuplicateWebhookGroup,
  useTestWebhookGroupEvent,
  type WebhookGroup,
} from '@/hooks/queries/useWebhookGroups';

/* ========================================================================= */
/* CONSTANTS (canonical events + categories, mirror the backend)              */
/* ========================================================================= */

const EVENT_TYPES = [
  { event: 'entry_long_market', label: 'Entry Long (Market)', category: 'entry' },
  { event: 'entry_long_limit', label: 'Entry Long (Limit)', category: 'entry' },
  { event: 'entry_short_market', label: 'Entry Short (Market)', category: 'entry' },
  { event: 'entry_short_limit', label: 'Entry Short (Limit)', category: 'entry' },
  { event: 'exit_long_market', label: 'Exit Long (Market)', category: 'exit' },
  { event: 'exit_long_limit', label: 'Exit Long (Limit)', category: 'exit' },
  { event: 'exit_short_market', label: 'Exit Short (Market)', category: 'exit' },
  { event: 'exit_short_limit', label: 'Exit Short (Limit)', category: 'exit' },
  { event: 'cancel_long', label: 'Cancel Long', category: 'cancel' },
  { event: 'cancel_short', label: 'Cancel Short', category: 'cancel' },
  { event: 'compliance_breach', label: 'Compliance Breach', category: 'compliance' },
] as const;

// Which exec types trigger each event (per docs/Webhook_Event_System.md)
const EVENT_TRIGGERED_BY: Record<string, string[]> = {
  entry_long_market: ['C', 'L', 'LC', 'CC'],
  entry_long_limit: ['L', 'LC'],
  entry_short_market: ['C', 'L', 'LC', 'CC'],
  entry_short_limit: ['L', 'LC'],
  exit_long_market: ['C', 'L', 'LC', 'CC'],
  exit_long_limit: ['L', 'LC'],
  exit_short_market: ['C', 'L', 'LC', 'CC'],
  exit_short_limit: ['L', 'LC'],
  cancel_long: ['L', 'LC'],
  cancel_short: ['L', 'LC'],
  compliance_breach: ['*'],
};

const CATEGORY_COLORS: Record<string, { color: string; bg: string }> = {
  entry: { color: 'var(--green)', bg: 'var(--green-muted)' },
  exit: { color: 'var(--red)', bg: 'var(--red-muted)' },
  cancel: { color: 'var(--orange)', bg: 'rgba(255,152,0,0.12)' },
  compliance: { color: '#AB47BC', bg: 'rgba(171,71,188,0.12)' },
};

const PLACEHOLDERS = [
  { key: '{{symbol}}', example: 'NVDA', category: 'Signal' },
  { key: '{{direction}}', example: 'LONG', category: 'Signal' },
  { key: '{{event_type}}', example: 'entry_long_market', category: 'Signal' },
  { key: '{{order_action}}', example: 'buy', category: 'Order' },
  { key: '{{order_type}}', example: 'market', category: 'Order' },
  { key: '{{order_price}}', example: '142.35', category: 'Order' },
  { key: '{{stop_price}}', example: '141.50', category: 'Order' },
  { key: '{{quantity}}', example: '10', category: 'Order' },
  { key: '{{trigger_name}}', example: 'EMA_CROSS', category: 'Strategy' },
  { key: '{{strategy_name}}', example: 'NVDA LONG - Mass #2', category: 'Strategy' },
  { key: '{{timeframe}}', example: '1Min', category: 'Strategy' },
  { key: '{{confluence_met}}', example: '5M-RVOL-HIGH, 1D-MACD-BULL', category: 'Strategy' },
  { key: '{{portfolio_name}}', example: 'Main Portfolio', category: 'Portfolio' },
  { key: '{{risk_per_trade}}', example: '500.00', category: 'Portfolio' },
  { key: '{{atr}}', example: '1.42', category: 'Indicator' },
  { key: '{{timestamp}}', example: '2026-03-24T14:30:00Z', category: 'Meta' },
];

const TABS = ['Event Payloads', 'Placeholders', 'Settings'];

/* ========================================================================= */
/* PAYLOAD GENERATOR — sensible default per event                             */
/* ========================================================================= */

function makeDefaultPayload(e: typeof EVENT_TYPES[number]): string {
  if (e.category === 'entry') {
    const action = e.event.includes('short') ? 'sell_short' : 'buy';
    const type = e.event.includes('limit') ? 'limit' : 'market';
    const priceField = e.event.includes('limit') ? ',\n  "price": {{order_price}}' : '';
    return `{\n  "action": "${action}",\n  "type": "${type}",\n  "symbol": "{{symbol}}",\n  "qty": {{quantity}}${priceField}\n}`;
  }
  if (e.category === 'exit') {
    const action = e.event.includes('short') ? 'buy_to_cover' : 'sell';
    const type = e.event.includes('limit') ? 'limit' : 'market';
    const priceField = e.event.includes('limit') ? ',\n  "price": {{order_price}}' : '';
    return `{\n  "action": "${action}",\n  "type": "${type}",\n  "symbol": "{{symbol}}",\n  "qty": {{quantity}}${priceField}\n}`;
  }
  if (e.category === 'cancel') {
    const side = e.event.includes('long') ? 'buy' : 'sell';
    return `{\n  "action": "cancel_all",\n  "symbol": "{{symbol}}",\n  "side": "${side}"\n}`;
  }
  return `{\n  "alert": "compliance_breach",\n  "portfolio": "{{portfolio_name}}",\n  "strategy": "{{strategy_name}}"\n}`;
}

/* ========================================================================= */
/* STYLES                                                                      */
/* ========================================================================= */

const inputStyle: React.CSSProperties = {
  background: 'var(--bg-input)',
  border: '1px solid var(--border)',
  color: 'var(--text-primary)',
  padding: '8px 12px',
  borderRadius: '8px',
  fontSize: '0.875rem',
  width: '100%',
};

const codeInputStyle: React.CSSProperties = {
  ...inputStyle,
  fontFamily: 'ui-monospace, "SF Mono", Menlo, Consolas, monospace',
  fontSize: '0.8125rem',
  lineHeight: '1.55',
  minHeight: 140,
  resize: 'vertical',
};

const btnSecondary: React.CSSProperties = {
  background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)',
  padding: '6px 14px', borderRadius: '8px', fontSize: '0.875rem', cursor: 'pointer',
};

const btnPrimary: React.CSSProperties = {
  background: 'var(--accent)', color: 'white', border: 'none',
  padding: '8px 16px', borderRadius: '8px', fontSize: '0.875rem', cursor: 'pointer', fontWeight: 600,
};

const btnDanger: React.CSSProperties = {
  background: 'var(--red-muted, rgba(239,68,68,0.15))', color: 'var(--red)', border: 'none',
  padding: '6px 14px', borderRadius: '8px', fontSize: '0.875rem', cursor: 'pointer',
};

const thStyle: React.CSSProperties = {
  color: 'var(--text-muted)', background: 'var(--bg-secondary)', textAlign: 'left',
  padding: '8px 12px', fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em',
};

const tdStyle: React.CSSProperties = {
  padding: '8px 12px', fontSize: '0.8125rem', borderBottom: '1px solid var(--border)', color: 'var(--text-secondary)',
};

/* ========================================================================= */
/* SUB COMPONENTS                                                              */
/* ========================================================================= */

function EventBadge({ event, category }: { event: string; category: string }) {
  const c = CATEGORY_COLORS[category] || { color: 'var(--text-muted)', bg: 'var(--bg-input)' };
  return (
    <span className="text-xs font-mono font-semibold px-2 py-0.5 rounded-full" style={{ color: c.color, background: c.bg }}>
      {event}
    </span>
  );
}

function resolvePayload(payload: string): string {
  return payload.replace(/\{\{(\w+)\}\}/g, (_, key) => {
    const ph = PLACEHOLDERS.find((p) => p.key === `{{${key}}}`);
    return ph ? ph.example : key;
  });
}

function ExecTypeBadges({ eventKey }: { eventKey: string }) {
  const types = EVENT_TRIGGERED_BY[eventKey] || [];
  if (types.length === 0) return null;
  return (
    <div className="flex items-center gap-1 flex-wrap">
      <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Triggered by:</span>
      {types.map((t) => (
        <span
          key={t}
          className="text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded"
          style={{ color: 'var(--accent)', background: 'var(--accent-muted)' }}
        >
          {t === '*' ? 'Any' : t}
        </span>
      ))}
    </div>
  );
}

/* ========================================================================= */
/* COMPONENT                                                                   */
/* ========================================================================= */

interface WebhookGroupDetailPageProps {
  groupId: string;
}

export default function WebhookGroupDetailPage({ groupId }: WebhookGroupDetailPageProps) {
  const router = useRouter();
  const { data: group, isLoading, error } = useWebhookGroup(groupId);
  const updateMutation = useUpdateWebhookGroup();
  const deleteMutation = useDeleteWebhookGroup();
  const duplicateMutation = useDuplicateWebhookGroup();
  const testMutation = useTestWebhookGroupEvent();

  // Local edit state — mirrors the group but only submits on Save
  const [name, setName] = useState('');
  const [service, setService] = useState('');
  const [urlMode, setUrlMode] = useState<'per_event' | 'shared'>('per_event');
  const [sharedUrl, setSharedUrl] = useState('');
  const [events, setEvents] = useState<Record<string, { url: string; payload: string }>>({});
  const [activeCategory, setActiveCategory] = useState<string>('entry');
  const [selectedEventIdx, setSelectedEventIdx] = useState(0);
  const [testResult, setTestResult] = useState<{ eventKey: string; ok: boolean; message: string } | null>(null);

  // Hydrate local state from API
  useEffect(() => {
    if (!group) return;
    setName(group.name || '');
    setService(group.service || '');
    setUrlMode((group.url_mode as 'per_event' | 'shared') || 'per_event');
    setSharedUrl(group.shared_url || '');
    const hydrated: Record<string, { url: string; payload: string }> = {};
    for (const e of EVENT_TYPES) {
      const cfg = group.events?.[e.event] || {};
      hydrated[e.event] = {
        url: cfg.url || '',
        payload: cfg.payload || '',
      };
    }
    setEvents(hydrated);
  }, [group?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  // Dirty tracking
  const isDirty = useMemo(() => {
    if (!group) return false;
    if ((group.name || '') !== name) return true;
    if ((group.service || '') !== service) return true;
    if (((group.url_mode as string) || 'per_event') !== urlMode) return true;
    if ((group.shared_url || '') !== sharedUrl) return true;
    for (const e of EVENT_TYPES) {
      const current = events[e.event] || { url: '', payload: '' };
      const saved = group.events?.[e.event] || { url: '', payload: '' };
      if (current.url !== (saved.url || '')) return true;
      if (current.payload !== (saved.payload || '')) return true;
    }
    return false;
  }, [group, name, service, urlMode, sharedUrl, events]);

  // "Configured" means: payload set AND (shared URL set OR per-event URL set)
  const configuredCount = Object.entries(events).filter(([key, c]) => {
    const hasUrl = urlMode === 'shared' ? sharedUrl.trim().length > 0 : (c.url || '').trim().length > 0;
    const hasPayload = (c.payload || '').trim().length > 0;
    return hasUrl && hasPayload;
  }).length;

  const filteredEvents = useMemo(
    () => EVENT_TYPES.filter((e) => e.category === activeCategory),
    [activeCategory]
  );
  const selectedEvent = filteredEvents[selectedEventIdx] || filteredEvents[0];
  const selectedCfg = selectedEvent ? (events[selectedEvent.event] || { url: '', payload: '' }) : null;

  const handleUpdateEvent = (eventKey: string, patch: Partial<{ url: string; payload: string }>) => {
    setEvents((prev) => ({
      ...prev,
      [eventKey]: { ...(prev[eventKey] || { url: '', payload: '' }), ...patch },
    }));
  };

  const handleSave = () => {
    updateMutation.mutate({
      id: groupId,
      updates: { name, service, url_mode: urlMode, shared_url: sharedUrl, events },
    });
  };

  const handleDelete = () => {
    if (!confirm(`Delete webhook group "${name}"? This cannot be undone.`)) return;
    deleteMutation.mutate(groupId, {
      onSuccess: () => router.push('/alerts/webhook-groups'),
    });
  };

  const handleDuplicate = () => {
    duplicateMutation.mutate(groupId, {
      onSuccess: (data) => router.push(`/alerts/webhook-groups/${data.id}`),
    });
  };

  const handleTest = (eventKey: string) => {
    setTestResult(null);
    testMutation.mutate(
      { id: groupId, eventType: eventKey },
      {
        onSuccess: (result: any) => {
          setTestResult({
            eventKey,
            ok: result?.success === true || (result?.status_code >= 200 && result?.status_code < 300),
            message: `Status ${result?.status_code ?? '?'} · ${result?.latency_ms ?? '?'}ms`,
          });
        },
        onError: (err: any) => {
          setTestResult({
            eventKey,
            ok: false,
            message: err?.message || 'Test failed',
          });
        },
      }
    );
  };

  // Use a default-generated payload when user hasn't written one yet
  const applyDefaultPayload = (eventKey: string) => {
    const e = EVENT_TYPES.find((evt) => evt.event === eventKey);
    if (!e) return;
    handleUpdateEvent(eventKey, { payload: makeDefaultPayload(e) });
  };

  /* ---- Loading / Error states ---- */
  if (isLoading) {
    return (
      <div>
        <PageHeader title="Webhook Group" subtitle="Loading…" backHref="/alerts/webhook-groups" />
        <div className="space-y-4 mt-4">
          {[1, 2].map((i) => (
            <Card key={i}>
              <div className="animate-pulse space-y-3">
                <div className="h-4 rounded w-1/3" style={{ background: 'var(--border)' }} />
                <div className="h-3 rounded w-2/3" style={{ background: 'var(--border)' }} />
              </div>
            </Card>
          ))}
        </div>
      </div>
    );
  }
  if (error || !group) {
    return (
      <div>
        <PageHeader title="Webhook Group" subtitle="Not found" backHref="/alerts/webhook-groups" />
        <Card>
          <div className="text-center py-8" style={{ color: 'var(--red)' }}>
            Failed to load webhook group.
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div>
      <PageHeader
        title={name || 'Untitled Group'}
        subtitle={
          <div className="flex items-center gap-3">
            {service && (
              <span className="text-xs px-2 py-0.5 rounded-full" style={{ color: 'var(--accent)', background: 'var(--accent-muted)' }}>
                {service}
              </span>
            )}
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
              {configuredCount} / 11 events configured
            </span>
          </div>
        }
        backHref="/alerts/webhook-groups"
        actions={
          <div className="flex items-center gap-2">
            <button style={btnSecondary} onClick={handleDuplicate} disabled={duplicateMutation.isPending}>
              {duplicateMutation.isPending ? 'Copying…' : 'Duplicate'}
            </button>
            <button style={btnDanger} onClick={handleDelete} disabled={deleteMutation.isPending}>
              {deleteMutation.isPending ? 'Deleting…' : 'Delete'}
            </button>
            <button
              style={{ ...btnPrimary, opacity: !isDirty || updateMutation.isPending ? 0.5 : 1, cursor: (!isDirty || updateMutation.isPending) ? 'not-allowed' : 'pointer' }}
              onClick={handleSave}
              disabled={!isDirty || updateMutation.isPending}
            >
              {updateMutation.isPending ? 'Saving…' : isDirty ? 'Save Changes' : 'Saved'}
            </button>
          </div>
        }
      />

      <TabBar tabs={TABS}>
        {(tab) => (
          <div>
            {/* ========== TAB 1: EVENT PAYLOADS ========== */}
            {tab === 'Event Payloads' && (
              <div>
                {/* Shared URL banner (only in shared mode) */}
                {urlMode === 'shared' && (
                  <Card className="mb-4">
                    <div className="flex items-center justify-between flex-wrap gap-2">
                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-medium mb-1" style={{ color: 'var(--accent)' }}>Shared URL mode</p>
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                          All 11 events fire to this single URL. Payload is the only thing that changes per event.
                        </p>
                      </div>
                      <div className="flex items-center gap-2" style={{ minWidth: 0, flex: '1 1 400px' }}>
                        <input
                          type="text"
                          placeholder="https://app.signalstack.com/hook/..."
                          value={sharedUrl}
                          onChange={(e) => setSharedUrl(e.target.value)}
                          style={{ ...inputStyle, fontFamily: 'monospace', fontSize: '0.8125rem' }}
                        />
                      </div>
                    </div>
                  </Card>
                )}

                <div className="grid grid-cols-12 gap-4">
                {/* Category + event selector */}
                <div className="col-span-12 md:col-span-3">
                  <Card>
                    {/* Category pills */}
                    <div className="flex gap-1 flex-wrap mb-3">
                      {(['entry', 'exit', 'cancel', 'compliance'] as const).map((cat) => {
                        const count = EVENT_TYPES.filter((e) => e.category === cat).length;
                        const c = CATEGORY_COLORS[cat];
                        return (
                          <button
                            key={cat}
                            onClick={() => { setActiveCategory(cat); setSelectedEventIdx(0); }}
                            className="text-xs px-2.5 py-1 rounded-full capitalize"
                            style={{
                              background: activeCategory === cat ? c.bg : 'var(--bg-input)',
                              color: activeCategory === cat ? c.color : 'var(--text-muted)',
                              border: activeCategory === cat ? `1px solid ${c.color}` : '1px solid var(--border)',
                              cursor: 'pointer',
                              fontWeight: activeCategory === cat ? 600 : 400,
                            }}
                          >
                            {cat} ({count})
                          </button>
                        );
                      })}
                    </div>

                    {/* Event list within active category */}
                    <div className="flex flex-col gap-1">
                      {filteredEvents.map((e, i) => {
                        const cfg = events[e.event] || { url: '', payload: '' };
                        const hasUrl = urlMode === 'shared'
                          ? sharedUrl.trim().length > 0
                          : (cfg.url || '').trim().length > 0;
                        const configured = hasUrl && (cfg.payload || '').trim().length > 0;
                        const active = i === selectedEventIdx;
                        return (
                          <button
                            key={e.event}
                            onClick={() => setSelectedEventIdx(i)}
                            className="flex items-center justify-between gap-2 px-3 py-2 rounded-lg text-left"
                            style={{
                              background: active ? 'var(--accent-muted)' : 'var(--bg-input)',
                              border: active ? '1px solid var(--accent)' : '1px solid transparent',
                              cursor: 'pointer',
                            }}
                          >
                            <div className="flex-1 min-w-0">
                              <p className="text-sm truncate" style={{ color: active ? 'var(--accent)' : 'var(--text-primary)', fontWeight: active ? 600 : 500 }}>
                                {e.label}
                              </p>
                              <p className="text-[10px] font-mono truncate" style={{ color: 'var(--text-muted)' }}>
                                {e.event}
                              </p>
                            </div>
                            <span
                              style={{
                                width: 8, height: 8, borderRadius: '50%',
                                background: configured ? 'var(--green)' : 'var(--text-muted)',
                                opacity: configured ? 1 : 0.4,
                                flexShrink: 0,
                              }}
                            />
                          </button>
                        );
                      })}
                    </div>
                  </Card>
                </div>

                {/* Event editor */}
                <div className="col-span-12 md:col-span-9">
                  {selectedEvent && selectedCfg && (
                    <Card>
                      <div className="flex items-start justify-between mb-4 flex-wrap gap-2">
                        <div>
                          <p className="text-base font-medium mb-1">{selectedEvent.label}</p>
                          <div className="flex items-center gap-3 flex-wrap">
                            <EventBadge event={selectedEvent.event} category={selectedEvent.category} />
                            <ExecTypeBadges eventKey={selectedEvent.event} />
                          </div>
                        </div>
                      </div>

                      {/* URL — only shown in per_event mode */}
                      {urlMode === 'per_event' ? (
                        <div className="mb-4">
                          <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-muted)' }}>
                            Webhook URL <span style={{ color: 'var(--text-muted)' }}>(blank = skip this event)</span>
                          </label>
                          <input
                            type="text"
                            placeholder="https://app.signalstack.com/hook/..."
                            value={selectedCfg.url}
                            onChange={(e) => handleUpdateEvent(selectedEvent.event, { url: e.target.value })}
                            style={inputStyle}
                          />
                        </div>
                      ) : (
                        <div className="mb-4 px-3 py-2 rounded-lg" style={{ background: 'var(--bg-input)', border: '1px dashed var(--border)' }}>
                          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                            Uses the <strong style={{ color: 'var(--accent)' }}>shared URL</strong> above. Switch to per-event mode in Settings to override.
                          </p>
                        </div>
                      )}

                      {/* Payload */}
                      <div className="mb-4">
                        <div className="flex items-center justify-between mb-1.5">
                          <label className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>
                            Payload Template (JSON with {'{{placeholders}}'})
                          </label>
                          {!selectedCfg.payload && (
                            <button
                              style={{ ...btnSecondary, fontSize: '0.75rem', padding: '4px 10px' }}
                              onClick={() => applyDefaultPayload(selectedEvent.event)}
                            >
                              Insert default
                            </button>
                          )}
                        </div>
                        <textarea
                          value={selectedCfg.payload}
                          onChange={(e) => handleUpdateEvent(selectedEvent.event, { payload: e.target.value })}
                          placeholder="{ }"
                          style={codeInputStyle}
                        />
                      </div>

                      {/* Resolved preview */}
                      {selectedCfg.payload && (
                        <div className="mb-4">
                          <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-muted)' }}>
                            Resolved preview (placeholders filled with examples)
                          </label>
                          <pre
                            className="rounded-lg p-3 text-xs font-mono overflow-auto"
                            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)', maxHeight: 200 }}
                          >
                            {resolvePayload(selectedCfg.payload)}
                          </pre>
                        </div>
                      )}

                      {/* Test + result */}
                      <div className="flex items-center gap-3 flex-wrap">
                        {(() => {
                          const effectiveUrl = urlMode === 'shared' ? sharedUrl : selectedCfg.url;
                          const canTest = Boolean(effectiveUrl) && !testMutation.isPending;
                          return (
                            <button
                              style={{ ...btnSecondary, opacity: canTest ? 1 : 0.5, cursor: canTest ? 'pointer' : 'not-allowed' }}
                              onClick={() => handleTest(selectedEvent.event)}
                              disabled={!canTest}
                            >
                              {testMutation.isPending && testResult?.eventKey !== selectedEvent.event ? 'Sending…' : 'Send test'}
                            </button>
                          );
                        })()}
                        {testResult && testResult.eventKey === selectedEvent.event && (
                          <span className="flex items-center gap-2 text-xs">
                            <span style={{
                              width: 8, height: 8, borderRadius: '50%',
                              background: testResult.ok ? 'var(--green)' : 'var(--red)',
                            }} />
                            <span style={{ color: testResult.ok ? 'var(--green)' : 'var(--red)' }}>
                              {testResult.message}
                            </span>
                          </span>
                        )}
                      </div>

                      {isDirty && (
                        <p className="text-xs mt-4" style={{ color: 'var(--orange)' }}>
                          Unsaved changes — click Save Changes in the header to persist.
                        </p>
                      )}
                    </Card>
                  )}
                </div>
                </div>
              </div>
            )}

            {/* ========== TAB 2: PLACEHOLDERS ========== */}
            {tab === 'Placeholders' && (
              <Card>
                <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
                  These placeholders get substituted with real values at webhook fire time. Use them in any event payload.
                </p>
                <div style={{ overflowX: 'auto' }}>
                  <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                    <thead>
                      <tr>
                        {['Placeholder', 'Category', 'Example'].map((h) => (
                          <th key={h} style={thStyle}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {PLACEHOLDERS.map((p) => (
                        <tr key={p.key}>
                          <td style={{ ...tdStyle, fontFamily: 'monospace', color: 'var(--accent)' }}>{p.key}</td>
                          <td style={tdStyle}>{p.category}</td>
                          <td style={{ ...tdStyle, fontFamily: 'monospace' }}>{p.example}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            )}

            {/* ========== TAB 3: SETTINGS ========== */}
            {tab === 'Settings' && (
              <Card>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                  <div>
                    <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-muted)' }}>Group Name</label>
                    <input
                      type="text"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      style={inputStyle}
                    />
                  </div>
                  <div>
                    <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-muted)' }}>Service / Account Type</label>
                    <input
                      type="text"
                      value={service}
                      onChange={(e) => setService(e.target.value)}
                      placeholder="signalstack, discord, custom…"
                      style={inputStyle}
                    />
                  </div>
                </div>

                {/* URL mode toggle */}
                <div className="mb-4 pt-4 border-t" style={{ borderColor: 'var(--border)' }}>
                  <label className="text-xs font-medium block mb-2" style={{ color: 'var(--text-muted)' }}>URL Mode</label>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mb-3">
                    <button
                      onClick={() => setUrlMode('per_event')}
                      className="flex flex-col items-start px-3 py-2.5 rounded-lg text-left transition-colors"
                      style={{
                        background: urlMode === 'per_event' ? 'var(--accent-muted)' : 'var(--bg-input)',
                        border: urlMode === 'per_event' ? '1px solid var(--accent)' : '1px solid var(--border)',
                        cursor: 'pointer',
                      }}
                    >
                      <span className="text-sm font-medium" style={{ color: urlMode === 'per_event' ? 'var(--accent)' : 'var(--text-primary)' }}>
                        URL per event
                      </span>
                      <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        Each of the 11 events has its own URL. Common for SignalStack-style services where each action has a unique endpoint.
                      </span>
                    </button>
                    <button
                      onClick={() => setUrlMode('shared')}
                      className="flex flex-col items-start px-3 py-2.5 rounded-lg text-left transition-colors"
                      style={{
                        background: urlMode === 'shared' ? 'var(--accent-muted)' : 'var(--bg-input)',
                        border: urlMode === 'shared' ? '1px solid var(--accent)' : '1px solid var(--border)',
                        cursor: 'pointer',
                      }}
                    >
                      <span className="text-sm font-medium" style={{ color: urlMode === 'shared' ? 'var(--accent)' : 'var(--text-primary)' }}>
                        One shared URL
                      </span>
                      <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        All events fire at the same URL (the payload differs per event). Common for generic webhook endpoints that route by payload content.
                      </span>
                    </button>
                  </div>
                  {urlMode === 'shared' && (
                    <div>
                      <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-muted)' }}>Shared URL</label>
                      <input
                        type="text"
                        placeholder="https://..."
                        value={sharedUrl}
                        onChange={(e) => setSharedUrl(e.target.value)}
                        style={{ ...inputStyle, fontFamily: 'monospace', fontSize: '0.8125rem' }}
                      />
                    </div>
                  )}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs" style={{ color: 'var(--text-muted)' }}>
                  <div>
                    <span className="font-medium" style={{ color: 'var(--text-secondary)' }}>Created</span>
                    <p className="mt-1 font-mono">{group.created_at ? new Date(group.created_at).toLocaleString() : '--'}</p>
                  </div>
                  <div>
                    <span className="font-medium" style={{ color: 'var(--text-secondary)' }}>Last Updated</span>
                    <p className="mt-1 font-mono">{group.updated_at ? new Date(group.updated_at).toLocaleString() : '--'}</p>
                  </div>
                  <div>
                    <span className="font-medium" style={{ color: 'var(--text-secondary)' }}>Group ID</span>
                    <p className="mt-1 font-mono">{group.id}</p>
                  </div>
                  <div>
                    <span className="font-medium" style={{ color: 'var(--text-secondary)' }}>Events Configured</span>
                    <p className="mt-1">{configuredCount} / 11</p>
                  </div>
                </div>
              </Card>
            )}
          </div>
        )}
      </TabBar>
    </div>
  );
}
