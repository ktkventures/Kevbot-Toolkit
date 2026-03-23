'use client';

import { useState, useMemo, useCallback } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import MetricCard from '@/components/MetricCard';
import Modal from '@/components/Modal';

// =============================================================================
// V4 Meta: Creative — Webhook Studio
// =============================================================================
// Visual webhook development environment with:
//
// 1. VISUAL PAYLOAD BUILDER — drag placeholders into payload structure
// 2. LIVE WEBHOOK TESTER — send test with response preview panel
// 3. DELIVERY TIMELINE — visual history of webhook deliveries
// 4. PAYLOAD DIFF VIEW — compare template vs rendered output
// 5. TEMPLATE CARDS — rich cards with payload preview, usage, health
// 6. PLACEHOLDER PALETTE — clickable palette to insert variables
// 7. RESPONSE INSPECTOR — headers, status, body from test delivery
// 8. USAGE GRAPH — which strategies use which templates
// =============================================================================

// ---------------------------------------------------------------------------
// CSS Animations
// ---------------------------------------------------------------------------

const ANIMATION_STYLES = `
@keyframes fade-in-up {
  0% { opacity: 0; transform: translateY(12px); }
  100% { opacity: 1; transform: translateY(0); }
}
@keyframes slide-in-right {
  0% { transform: translateX(16px); opacity: 0; }
  100% { transform: translateX(0); opacity: 1; }
}
@keyframes pulse-success {
  0%, 100% { box-shadow: 0 0 6px rgba(0,255,136,0.1); }
  50% { box-shadow: 0 0 20px rgba(0,255,136,0.35); }
}
@keyframes pulse-error {
  0%, 100% { box-shadow: 0 0 6px rgba(255,80,80,0.1); }
  50% { box-shadow: 0 0 20px rgba(255,80,80,0.35); }
}
@keyframes typing-cursor {
  0%, 50% { border-right-color: var(--accent); }
  51%, 100% { border-right-color: transparent; }
}
@keyframes timeline-dot-pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.4); }
}
@keyframes pop-in {
  0% { transform: scale(0.8); opacity: 0; }
  70% { transform: scale(1.03); }
  100% { transform: scale(1); opacity: 1; }
}
@keyframes send-wave {
  0% { transform: translateX(0); opacity: 1; }
  100% { transform: translateX(20px); opacity: 0; }
}
@keyframes receive-wave {
  0% { transform: translateX(-20px); opacity: 0; }
  100% { transform: translateX(0); opacity: 1; }
}
@keyframes code-highlight {
  0% { background: rgba(0,255,136,0.2); }
  100% { background: transparent; }
}
`;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface WebhookTemplate {
  id: string;
  name: string;
  url: string;
  usedBy: number;
  payload: string;
  lastDelivery?: {
    status: number;
    time: string;
    latencyMs: number;
  };
  deliveryHistory: DeliveryEvent[];
}

interface DeliveryEvent {
  id: string;
  time: string;
  status: number;
  latencyMs: number;
  strategy: string;
  signal: string;
}

interface Placeholder {
  key: string;
  label: string;
  example: string;
  category: 'signal' | 'strategy' | 'position' | 'meta';
}

interface TestResponse {
  status: number;
  statusText: string;
  latencyMs: number;
  headers: Record<string, string>;
  body: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PLACEHOLDERS: Placeholder[] = [
  { key: '{{SYMBOL}}', label: 'Symbol', example: 'NVDA', category: 'signal' },
  { key: '{{SIGNAL_TYPE}}', label: 'Signal Type', example: 'ENTRY', category: 'signal' },
  { key: '{{DIRECTION}}', label: 'Direction', example: 'LONG', category: 'signal' },
  { key: '{{PRICE}}', label: 'Price', example: '142.35', category: 'signal' },
  { key: '{{STRATEGY}}', label: 'Strategy Name', example: 'NVDA EMA Scalper', category: 'strategy' },
  { key: '{{TIMEFRAME}}', label: 'Timeframe', example: '1Min', category: 'strategy' },
  { key: '{{EXEC_TYPE}}', label: 'Exec Type', example: 'C', category: 'strategy' },
  { key: '{{PNL}}', label: 'P&L (R)', example: '+1.5R', category: 'position' },
  { key: '{{TIMESTAMP}}', label: 'Timestamp', example: '2026-03-21T10:30:00Z', category: 'meta' },
];

const CATEGORY_COLORS: Record<string, { bg: string; color: string }> = {
  signal: { bg: 'var(--green-muted)', color: 'var(--green)' },
  strategy: { bg: 'var(--accent-muted)', color: 'var(--accent)' },
  position: { bg: 'rgba(100,149,237,0.12)', color: 'rgb(100,149,237)' },
  meta: { bg: 'rgba(255,200,0,0.1)', color: 'rgb(200,170,50)' },
};

// ---------------------------------------------------------------------------
// Mock Data
// ---------------------------------------------------------------------------

const mockDeliveryHistory: DeliveryEvent[] = [
  { id: 'd1', time: '10:32:15', status: 200, latencyMs: 142, strategy: 'NVDA Scalper', signal: 'ENTRY LONG' },
  { id: 'd2', time: '10:28:03', status: 200, latencyMs: 89, strategy: 'SPY Swing', signal: 'EXIT' },
  { id: 'd3', time: '10:15:47', status: 200, latencyMs: 201, strategy: 'NVDA Scalper', signal: 'EXIT +1.2R' },
  { id: 'd4', time: '09:45:12', status: 429, latencyMs: 55, strategy: 'TSLA Momentum', signal: 'ENTRY LONG' },
  { id: 'd5', time: '09:31:08', status: 200, latencyMs: 112, strategy: 'NVDA Scalper', signal: 'ENTRY LONG' },
];

const initialTemplates: WebhookTemplate[] = [
  {
    id: 't1', name: 'TradeThePool Market Buy', usedBy: 3,
    url: 'https://api.tradethepool.com/webhook/abc123',
    payload: '{\n  "action": "buy",\n  "symbol": "{{SYMBOL}}",\n  "qty": 1,\n  "type": "market"\n}',
    lastDelivery: { status: 200, time: '10:32:15', latencyMs: 142 },
    deliveryHistory: mockDeliveryHistory.slice(0, 3),
  },
  {
    id: 't2', name: 'TradeThePool Market Sell', usedBy: 3,
    url: 'https://api.tradethepool.com/webhook/abc123',
    payload: '{\n  "action": "sell",\n  "symbol": "{{SYMBOL}}",\n  "qty": 1,\n  "type": "market"\n}',
    lastDelivery: { status: 200, time: '10:28:03', latencyMs: 89 },
    deliveryHistory: mockDeliveryHistory.slice(1, 4),
  },
  {
    id: 't3', name: 'Discord Alert', usedBy: 2,
    url: 'https://discord.com/api/webhooks/123456/abcdef',
    payload: '{\n  "embeds": [{\n    "title": "{{SIGNAL_TYPE}} Signal",\n    "description": "{{STRATEGY}} @ {{PRICE}}",\n    "color": 65280\n  }]\n}',
    lastDelivery: { status: 200, time: '10:15:47', latencyMs: 201 },
    deliveryHistory: mockDeliveryHistory.slice(2, 5),
  },
  {
    id: 't4', name: 'Slack Notification', usedBy: 1,
    url: 'https://hooks.slack.com/services/T01/B02/xyz789',
    payload: '{\n  "text": "{{SIGNAL_TYPE}}: {{STRATEGY}} at {{PRICE}}"\n}',
    lastDelivery: undefined,
    deliveryHistory: [],
  },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function maskUrl(url: string): string {
  try { const p = new URL(url); return `${p.hostname}`; } catch { return url.substring(0, 30); }
}

function generateId(): string {
  return 'id-' + Math.random().toString(36).substring(2, 9);
}

/** Render payload with placeholder values highlighted */
function renderPayloadPreview(payload: string): string {
  return payload.replace(/\{\{(\w+)\}\}/g, (_, key) => {
    const ph = PLACEHOLDERS.find((p) => p.key === `{{${key}}}`);
    return ph ? ph.example : key;
  });
}

/** Count placeholders in payload */
function countPlaceholders(payload: string): number {
  const matches = payload.match(/\{\{(\w+)\}\}/g);
  return matches ? new Set(matches).size : 0;
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

/** Delivery timeline visualization */
function DeliveryTimeline({ events }: { events: DeliveryEvent[] }) {
  if (events.length === 0) {
    return (
      <div className="py-4 text-center">
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>No deliveries yet</p>
      </div>
    );
  }

  return (
    <div className="space-y-0">
      {events.map((evt, i) => {
        const isSuccess = evt.status >= 200 && evt.status < 300;
        return (
          <div
            key={evt.id}
            className="flex items-start gap-3 pb-3"
            style={{ animation: `slide-in-right 0.3s ease ${i * 0.06}s both` }}
          >
            {/* Timeline dot + line */}
            <div className="flex flex-col items-center flex-shrink-0">
              <div
                className="w-2.5 h-2.5 rounded-full"
                style={{
                  background: isSuccess ? 'var(--green)' : 'var(--red)',
                  animation: i === 0 ? 'timeline-dot-pulse 2s ease infinite' : undefined,
                }}
              />
              {i < events.length - 1 && (
                <div className="w-px flex-1 min-h-[20px]" style={{ background: 'var(--border)' }} />
              )}
            </div>
            {/* Content */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>{evt.time}</span>
                <span
                  className="text-[10px] px-1.5 py-0.5 rounded font-mono font-bold"
                  style={{
                    background: isSuccess ? 'var(--green-muted)' : 'var(--red-muted)',
                    color: isSuccess ? 'var(--green)' : 'var(--red)',
                  }}
                >
                  {evt.status}
                </span>
                <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{evt.latencyMs}ms</span>
              </div>
              <div className="text-xs mt-0.5" style={{ color: 'var(--text-secondary)' }}>
                {evt.strategy} &middot; {evt.signal}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

/** Placeholder palette for payload builder */
function PlaceholderPalette({ onInsert }: { onInsert: (key: string) => void }) {
  const categories = Array.from(new Set(PLACEHOLDERS.map((p) => p.category)));

  return (
    <div className="space-y-2">
      {categories.map((cat) => (
        <div key={cat}>
          <div className="text-[9px] font-medium uppercase mb-1" style={{ color: CATEGORY_COLORS[cat].color }}>
            {cat}
          </div>
          <div className="flex gap-1 flex-wrap">
            {PLACEHOLDERS.filter((p) => p.category === cat).map((ph) => (
              <button
                key={ph.key}
                onClick={() => onInsert(ph.key)}
                className="px-2 py-1 rounded text-[10px] font-mono transition-all"
                style={{
                  background: CATEGORY_COLORS[cat].bg,
                  color: CATEGORY_COLORS[cat].color,
                  border: `1px solid ${CATEGORY_COLORS[cat].color}`,
                  borderColor: 'transparent',
                }}
                onMouseEnter={(e) => { e.currentTarget.style.borderColor = CATEGORY_COLORS[cat].color; e.currentTarget.style.transform = 'scale(1.05)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'transparent'; e.currentTarget.style.transform = 'scale(1)'; }}
                title={`${ph.label}: ${ph.example}`}
              >
                {ph.label}
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

/** Payload diff view: template vs rendered */
function PayloadDiff({ template, rendered }: { template: string; rendered: string }) {
  return (
    <div className="grid grid-cols-2 gap-2">
      <div>
        <div className="text-[9px] font-medium mb-1 uppercase" style={{ color: 'var(--text-muted)' }}>Template</div>
        <pre
          className="text-[11px] p-3 rounded-lg overflow-x-auto leading-relaxed"
          style={{
            background: 'rgba(0,0,0,0.3)',
            color: 'var(--text-secondary)',
            fontFamily: 'monospace',
            border: '1px solid var(--border)',
          }}
        >
          {template}
        </pre>
      </div>
      <div>
        <div className="text-[9px] font-medium mb-1 uppercase" style={{ color: 'var(--accent)' }}>Rendered</div>
        <pre
          className="text-[11px] p-3 rounded-lg overflow-x-auto leading-relaxed"
          style={{
            background: 'rgba(0,255,136,0.03)',
            color: 'var(--text-primary)',
            fontFamily: 'monospace',
            border: '1px solid var(--accent)',
          }}
        >
          {rendered}
        </pre>
      </div>
    </div>
  );
}

/** Test response inspector */
function ResponseInspector({ response }: { response: TestResponse | null }) {
  if (!response) return null;

  const isSuccess = response.status >= 200 && response.status < 300;

  return (
    <div
      className="rounded-lg border p-3 mt-3"
      style={{
        borderColor: isSuccess ? 'var(--green)' : 'var(--red)',
        background: isSuccess ? 'rgba(0,255,136,0.03)' : 'rgba(255,80,80,0.03)',
        animation: `${isSuccess ? 'pulse-success' : 'pulse-error'} 2s ease-in-out 1`,
      }}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span
            className="text-xs font-mono font-bold px-2 py-0.5 rounded"
            style={{
              background: isSuccess ? 'var(--green-muted)' : 'var(--red-muted)',
              color: isSuccess ? 'var(--green)' : 'var(--red)',
            }}
          >
            {response.status} {response.statusText}
          </span>
          <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{response.latencyMs}ms</span>
        </div>
      </div>
      {/* Headers */}
      <div className="mb-2">
        <div className="text-[9px] font-medium mb-1 uppercase" style={{ color: 'var(--text-muted)' }}>Headers</div>
        <div className="space-y-0.5">
          {Object.entries(response.headers).map(([k, v]) => (
            <div key={k} className="text-[10px] font-mono" style={{ color: 'var(--text-secondary)' }}>
              <span style={{ color: 'var(--text-muted)' }}>{k}:</span> {v}
            </div>
          ))}
        </div>
      </div>
      {/* Body */}
      <div>
        <div className="text-[9px] font-medium mb-1 uppercase" style={{ color: 'var(--text-muted)' }}>Body</div>
        <pre className="text-[10px] font-mono p-2 rounded" style={{ background: 'rgba(0,0,0,0.2)', color: 'var(--text-secondary)' }}>
          {response.body}
        </pre>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function WebhookTemplatesV4() {
  const [templates, setTemplates] = useState<WebhookTemplate[]>(initialTemplates);
  const [modalOpen, setModalOpen] = useState(false);
  const [editingTemplate, setEditingTemplate] = useState<WebhookTemplate | null>(null);
  const [isNew, setIsNew] = useState(false);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  // Test state
  const [testingId, setTestingId] = useState<string | null>(null);
  const [testStatus, setTestStatus] = useState<'idle' | 'sending' | 'success' | 'error'>('idle');
  const [testResponse, setTestResponse] = useState<TestResponse | null>(null);

  // Stats
  const totalDeliveries = templates.reduce((s, t) => s + t.deliveryHistory.length, 0);
  const successRate = totalDeliveries > 0
    ? (templates.reduce((s, t) => s + t.deliveryHistory.filter((d) => d.status >= 200 && d.status < 300).length, 0) / totalDeliveries * 100)
    : 0;
  const avgLatency = totalDeliveries > 0
    ? templates.reduce((s, t) => s + t.deliveryHistory.reduce((ss, d) => ss + d.latencyMs, 0), 0) / totalDeliveries
    : 0;

  function handleNew() {
    setEditingTemplate({
      id: generateId(), name: '', url: '', usedBy: 0,
      payload: '{\n  \n}',
      deliveryHistory: [],
    });
    setIsNew(true);
    setModalOpen(true);
  }

  function handleEdit(tmpl: WebhookTemplate) {
    setEditingTemplate({ ...tmpl });
    setIsNew(false);
    setModalOpen(true);
  }

  function handleSave() {
    if (!editingTemplate || !editingTemplate.name.trim()) return;
    setTemplates((prev) => {
      const exists = prev.find((t) => t.id === editingTemplate.id);
      if (exists) return prev.map((t) => t.id === editingTemplate.id ? editingTemplate : t);
      return [...prev, editingTemplate];
    });
    setModalOpen(false);
    setEditingTemplate(null);
  }

  function handleDelete(id: string) {
    setTemplates((prev) => prev.filter((t) => t.id !== id));
    setDeleteConfirmId(null);
    if (expandedId === id) setExpandedId(null);
  }

  function handleTest(id: string) {
    setTestingId(id);
    setTestStatus('sending');
    setTestResponse(null);
    setTimeout(() => {
      setTestStatus('success');
      setTestResponse({
        status: 200,
        statusText: 'OK',
        latencyMs: 127,
        headers: {
          'content-type': 'application/json',
          'x-ratelimit-remaining': '29',
          'x-request-id': 'req_8f3a2b1c',
        },
        body: '{"success": true, "id": "msg_abc123"}',
      });
      setTimeout(() => { setTestingId(null); setTestStatus('idle'); }, 5000);
    }, 1200);
  }

  function insertPlaceholder(key: string) {
    if (!editingTemplate) return;
    // Insert at end of payload (before last })
    const payload = editingTemplate.payload;
    const lastBrace = payload.lastIndexOf('}');
    if (lastBrace > 0) {
      const before = payload.substring(0, lastBrace);
      const after = payload.substring(lastBrace);
      const needsComma = before.trim().endsWith('"') || before.trim().endsWith('}') || before.trim().endsWith(']');
      const newPayload = before + (needsComma ? ',\n  ' : '') + `"field": "${key}"` + '\n' + after;
      setEditingTemplate({ ...editingTemplate, payload: newPayload });
    }
  }

  return (
    <div>
      <style dangerouslySetInnerHTML={{ __html: ANIMATION_STYLES }} />

      <PageHeader
        title="Webhook Studio"
        subtitle="Build, test, and monitor webhook integrations"
        backHref="/alerts"
        actions={
          <button
            onClick={handleNew}
            className="px-4 py-2 rounded-lg text-sm font-medium"
            style={{ background: 'var(--accent)', color: 'white' }}
          >
            + New Template
          </button>
        }
      />

      {/* Stats strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-5">
        <MetricCard label="Templates" value={String(templates.length)} />
        <MetricCard label="Total Deliveries" value={String(totalDeliveries)} delta="all time" positive />
        <MetricCard label="Success Rate" value={`${successRate.toFixed(1)}%`} positive={successRate >= 95} />
        <MetricCard label="Avg Latency" value={`${Math.round(avgLatency)}ms`} delta="response time" positive={avgLatency < 200} />
      </div>

      {/* Template cards */}
      <div className="space-y-3">
        {templates.map((tmpl, i) => {
          const isExpanded = expandedId === tmpl.id;
          const phCount = countPlaceholders(tmpl.payload);

          return (
            <Card key={tmpl.id}>
              <div style={{ animation: `fade-in-up 0.3s ease ${i * 0.06}s both` }}>
                {/* Main row */}
                <div className="flex items-center gap-4">
                  {/* Status indicator */}
                  <div
                    className="w-10 h-10 rounded-xl flex items-center justify-center text-sm font-bold flex-shrink-0"
                    style={{
                      background: tmpl.lastDelivery
                        ? (tmpl.lastDelivery.status < 300 ? 'var(--green-muted)' : 'var(--red-muted)')
                        : 'var(--bg-input)',
                      color: tmpl.lastDelivery
                        ? (tmpl.lastDelivery.status < 300 ? 'var(--green)' : 'var(--red)')
                        : 'var(--text-muted)',
                    }}
                  >
                    {tmpl.lastDelivery ? tmpl.lastDelivery.status : '--'}
                  </div>

                  {/* Info */}
                  <div className="flex-1 min-w-0">
                    <h3 className="text-sm font-bold truncate" style={{ color: 'var(--text-primary)' }}>{tmpl.name}</h3>
                    <div className="flex items-center gap-3 mt-0.5">
                      <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>{maskUrl(tmpl.url)}</span>
                      <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                        {phCount} placeholder{phCount !== 1 ? 's' : ''}
                      </span>
                      <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                        Used by {tmpl.usedBy} strateg{tmpl.usedBy !== 1 ? 'ies' : 'y'}
                      </span>
                    </div>
                  </div>

                  {/* Last delivery info */}
                  {tmpl.lastDelivery && (
                    <div className="text-right flex-shrink-0 hidden md:block">
                      <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Last delivery</div>
                      <div className="text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>
                        {tmpl.lastDelivery.time} ({tmpl.lastDelivery.latencyMs}ms)
                      </div>
                    </div>
                  )}

                  {/* Actions */}
                  <div className="flex items-center gap-1 flex-shrink-0">
                    <button
                      onClick={() => setExpandedId(isExpanded ? null : tmpl.id)}
                      className="px-2 py-1.5 rounded text-[10px] font-medium transition-colors"
                      style={{
                        background: isExpanded ? 'var(--accent-muted)' : 'transparent',
                        color: isExpanded ? 'var(--accent)' : 'var(--text-muted)',
                        border: isExpanded ? '1px solid var(--accent)' : '1px solid transparent',
                      }}
                    >
                      {isExpanded ? 'Collapse' : 'Details'}
                    </button>
                    <button
                      onClick={() => handleEdit(tmpl)}
                      className="w-8 h-8 rounded-lg flex items-center justify-center text-xs transition-colors"
                      style={{ color: 'var(--text-muted)' }}
                      onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--accent-muted)'; e.currentTarget.style.color = 'var(--accent)'; }}
                      onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--text-muted)'; }}
                      title="Edit"
                    >
                      E
                    </button>
                    <button
                      onClick={() => handleTest(tmpl.id)}
                      disabled={testingId === tmpl.id && testStatus === 'sending'}
                      className="w-8 h-8 rounded-lg flex items-center justify-center text-xs transition-colors"
                      style={{
                        color: testingId === tmpl.id && testStatus === 'success' ? 'var(--green)' : 'var(--text-muted)',
                        background: testingId === tmpl.id && testStatus === 'success' ? 'var(--green-muted)' : 'transparent',
                        animation: testingId === tmpl.id && testStatus === 'sending' ? 'send-wave 0.6s ease infinite' : undefined,
                      }}
                      onMouseEnter={(e) => { if (testingId !== tmpl.id) { e.currentTarget.style.background = 'var(--accent-muted)'; e.currentTarget.style.color = 'var(--accent)'; } }}
                      onMouseLeave={(e) => { if (testingId !== tmpl.id) { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--text-muted)'; } }}
                      title="Test"
                    >
                      {testingId === tmpl.id && testStatus === 'sending' ? '...' : testingId === tmpl.id && testStatus === 'success' ? 'OK' : 'T'}
                    </button>
                    {deleteConfirmId === tmpl.id ? (
                      <div className="flex gap-1">
                        <button onClick={() => handleDelete(tmpl.id)} className="w-7 h-7 rounded flex items-center justify-center text-[10px] font-bold" style={{ background: 'var(--red)', color: 'white' }}>Y</button>
                        <button onClick={() => setDeleteConfirmId(null)} className="w-7 h-7 rounded flex items-center justify-center text-[10px]" style={{ border: '1px solid var(--border)', color: 'var(--text-muted)' }}>N</button>
                      </div>
                    ) : (
                      <button
                        onClick={() => setDeleteConfirmId(tmpl.id)}
                        className="w-8 h-8 rounded-lg flex items-center justify-center text-xs transition-colors"
                        style={{ color: 'var(--text-muted)' }}
                        onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--red-muted)'; e.currentTarget.style.color = 'var(--red)'; }}
                        onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--text-muted)'; }}
                        title="Delete"
                      >
                        D
                      </button>
                    )}
                  </div>
                </div>

                {/* Expanded section */}
                {isExpanded && (
                  <div className="mt-4 pt-4 border-t space-y-4" style={{ borderColor: 'var(--border)', animation: 'fade-in-up 0.3s ease both' }}>
                    {/* Payload diff */}
                    <PayloadDiff template={tmpl.payload} rendered={renderPayloadPreview(tmpl.payload)} />

                    {/* Test response (if this template was just tested) */}
                    {testingId === tmpl.id && testResponse && (
                      <ResponseInspector response={testResponse} />
                    )}

                    {/* Delivery timeline */}
                    <div>
                      <h4 className="text-xs font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
                        Recent Deliveries
                      </h4>
                      <DeliveryTimeline events={tmpl.deliveryHistory} />
                    </div>
                  </div>
                )}
              </div>
            </Card>
          );
        })}

        {templates.length === 0 && (
          <Card>
            <div className="flex flex-col items-center justify-center py-16">
              <div className="w-14 h-14 rounded-2xl flex items-center justify-center text-xl mb-4" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>W</div>
              <h3 className="text-sm font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>No Webhook Templates</h3>
              <p className="text-xs text-center max-w-xs" style={{ color: 'var(--text-muted)' }}>Create your first template to start sending alerts to external services.</p>
            </div>
          </Card>
        )}
      </div>

      {/* Edit/Create Modal — Enhanced with placeholder palette + diff */}
      <Modal
        title={isNew ? 'New Template' : `Edit: ${editingTemplate?.name ?? ''}`}
        isOpen={modalOpen}
        onClose={() => { setModalOpen(false); setEditingTemplate(null); }}
        width="720px"
      >
        {editingTemplate && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Name</label>
                <input
                  type="text"
                  value={editingTemplate.name}
                  onChange={(e) => setEditingTemplate({ ...editingTemplate, name: e.target.value })}
                  placeholder="e.g. Discord Alert"
                  className="w-full px-3 py-2 rounded-lg text-sm"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                />
              </div>
              <div>
                <label className="block text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Webhook URL</label>
                <input
                  type="url"
                  value={editingTemplate.url}
                  onChange={(e) => setEditingTemplate({ ...editingTemplate, url: e.target.value })}
                  placeholder="https://hooks.example.com/..."
                  className="w-full px-3 py-2 rounded-lg text-sm"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                />
              </div>
            </div>

            {/* Placeholder palette */}
            <div>
              <label className="block text-xs mb-2" style={{ color: 'var(--text-muted)' }}>
                Placeholders <span className="text-[10px] opacity-60">(click to insert)</span>
              </label>
              <PlaceholderPalette onInsert={insertPlaceholder} />
            </div>

            {/* Payload editor */}
            <div>
              <label className="block text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Payload</label>
              <textarea
                value={editingTemplate.payload}
                onChange={(e) => setEditingTemplate({ ...editingTemplate, payload: e.target.value })}
                rows={8}
                className="w-full px-3 py-2 rounded-lg text-sm"
                style={{
                  background: 'var(--bg-input)', border: '1px solid var(--border)',
                  color: 'var(--text-primary)', fontFamily: 'monospace', resize: 'vertical',
                }}
              />
            </div>

            {/* Live preview */}
            <div>
              <label className="block text-[10px] font-medium mb-1 uppercase" style={{ color: 'var(--accent)' }}>
                Live Preview
              </label>
              <pre
                className="text-[11px] p-3 rounded-lg overflow-x-auto leading-relaxed"
                style={{
                  background: 'rgba(0,255,136,0.03)',
                  color: 'var(--text-primary)',
                  fontFamily: 'monospace',
                  border: '1px solid var(--accent)',
                }}
              >
                {renderPayloadPreview(editingTemplate.payload)}
              </pre>
            </div>

            <div className="flex justify-end gap-3 pt-3 border-t" style={{ borderColor: 'var(--border)' }}>
              <button
                onClick={() => { setModalOpen(false); setEditingTemplate(null); }}
                className="px-4 py-2 rounded-lg text-sm"
                style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
              >
                Cancel
              </button>
              <button
                onClick={handleSave}
                className="px-4 py-2 rounded-lg text-sm font-medium"
                style={{
                  background: 'var(--accent)', color: 'white',
                  opacity: editingTemplate.name.trim() ? 1 : 0.5,
                }}
                disabled={!editingTemplate.name.trim()}
              >
                {isNew ? 'Create' : 'Save'}
              </button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
