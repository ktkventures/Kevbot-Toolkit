'use client';

import { useState, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import Modal from '@/components/Modal';

/* ========================================================================= */
/* TYPES                                                                      */
/* ========================================================================= */

interface WebhookHeader {
  key: string;
  value: string;
}

interface WebhookTemplate {
  id: string;
  name: string;
  description: string;
  url: string;
  contentType: 'JSON' | 'Form';
  usedBy: number;
  payload: string;
  headers: WebhookHeader[];
}

interface TestResult {
  status: number;
  body: string;
}

/* ========================================================================= */
/* MOCK DATA                                                                  */
/* ========================================================================= */

const initialTemplates: WebhookTemplate[] = [
  {
    id: 't1',
    name: 'TradeThePool Market Buy',
    description: 'Places market buy order via TTP webhook',
    url: 'https://api.tradethepool.com/webhook/abc123',
    contentType: 'JSON',
    usedBy: 3,
    payload: '{\n  "action": "buy",\n  "symbol": "{{SYMBOL}}",\n  "qty": 1,\n  "type": "market"\n}',
    headers: [{ key: 'Authorization', value: 'Bearer tok_xxx' }],
  },
  {
    id: 't2',
    name: 'TradeThePool Market Sell',
    description: 'Places market sell order via TTP webhook',
    url: 'https://api.tradethepool.com/webhook/abc123',
    contentType: 'JSON',
    usedBy: 3,
    payload: '{\n  "action": "sell",\n  "symbol": "{{SYMBOL}}",\n  "qty": 1,\n  "type": "market"\n}',
    headers: [{ key: 'Authorization', value: 'Bearer tok_xxx' }],
  },
  {
    id: 't3',
    name: 'Discord Alert',
    description: 'Posts alert to Discord channel',
    url: 'https://discord.com/api/webhooks/123456/abcdef',
    contentType: 'JSON',
    usedBy: 2,
    payload: '{\n  "embeds": [{\n    "title": "{{SIGNAL_TYPE}} Signal",\n    "description": "{{STRATEGY}} @ {{PRICE}}",\n    "color": 65280,\n    "fields": [\n      {"name": "Symbol", "value": "{{SYMBOL}}", "inline": true},\n      {"name": "Direction", "value": "{{DIRECTION}}", "inline": true},\n      {"name": "Exec Type", "value": "{{EXEC_TYPE}}", "inline": true}\n    ]\n  }]\n}',
    headers: [],
  },
  {
    id: 't4',
    name: 'Slack Notification',
    description: 'Posts to Slack #trading channel',
    url: 'https://hooks.slack.com/services/T01/B02/xyz789',
    contentType: 'JSON',
    usedBy: 1,
    payload: '{\n  "text": "{{SIGNAL_TYPE}}: {{STRATEGY}} at {{PRICE}}",\n  "blocks": [\n    {\n      "type": "section",\n      "text": {"type": "mrkdwn", "text": "*{{SIGNAL_TYPE}}* — {{STRATEGY}}\\nPrice: {{PRICE}} | {{TIMEFRAME}} | {{DIRECTION}}"}\n    }\n  ]\n}',
    headers: [],
  },
];

const PLACEHOLDERS = [
  '{{STRATEGY}}',
  '{{SIGNAL_TYPE}}',
  '{{PRICE}}',
  '{{SYMBOL}}',
  '{{DIRECTION}}',
  '{{TIMEFRAME}}',
  '{{TIMESTAMP}}',
  '{{PNL}}',
  '{{EXEC_TYPE}}',
];

const SAMPLE_DATA: Record<string, string> = {
  '{{STRATEGY}}': 'NVDA LONG - Mass #2',
  '{{SIGNAL_TYPE}}': 'ENTRY',
  '{{PRICE}}': '487.50',
  '{{SYMBOL}}': 'NVDA',
  '{{DIRECTION}}': 'LONG',
  '{{TIMEFRAME}}': '1Min',
  '{{TIMESTAMP}}': '2026-03-21T10:32:00Z',
  '{{PNL}}': '+1.2R',
  '{{EXEC_TYPE}}': '[C]',
};

/* ========================================================================= */
/* HELPERS                                                                    */
/* ========================================================================= */

function maskUrl(url: string): string {
  try {
    const parsed = new URL(url);
    const host = parsed.hostname;
    return `https://${host.substring(0, Math.min(12, host.length))}...`;
  } catch {
    return url.substring(0, 24) + '...';
  }
}

function renderPayload(payload: string): string {
  let result = payload;
  for (const [key, val] of Object.entries(SAMPLE_DATA)) {
    result = result.replaceAll(key, val);
  }
  return result;
}

function generateId(): string {
  return 'id-' + Math.random().toString(36).substring(2, 9);
}

/* ========================================================================= */
/* COMPONENT                                                                  */
/* ========================================================================= */

export default function WebhookTemplatesV2() {
  const [templates, setTemplates] = useState<WebhookTemplate[]>(initialTemplates);
  const [editorOpen, setEditorOpen] = useState(false);
  const [editingTemplate, setEditingTemplate] = useState<WebhookTemplate | null>(null);
  const [isNew, setIsNew] = useState(false);
  const [testModalId, setTestModalId] = useState<string | null>(null);
  const [testSignalType, setTestSignalType] = useState<'ENTRY' | 'EXIT'>('ENTRY');
  const [testResult, setTestResult] = useState<TestResult | null>(null);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  const testTemplate = useMemo(
    () => templates.find((t) => t.id === testModalId) ?? null,
    [templates, testModalId]
  );

  // ── Handlers ──

  function handleNew() {
    setEditingTemplate({
      id: generateId(),
      name: '',
      description: '',
      url: '',
      contentType: 'JSON',
      usedBy: 0,
      payload: '{\n  \n}',
      headers: [],
    });
    setIsNew(true);
    setEditorOpen(true);
  }

  function handleEdit(template: WebhookTemplate) {
    setEditingTemplate({ ...template, headers: template.headers.map((h) => ({ ...h })) });
    setIsNew(false);
    setEditorOpen(true);
  }

  function handleClone(template: WebhookTemplate) {
    setEditingTemplate({
      ...template,
      id: generateId(),
      name: `${template.name} (Copy)`,
      usedBy: 0,
      headers: template.headers.map((h) => ({ ...h })),
    });
    setIsNew(true);
    setEditorOpen(true);
  }

  function handleSave() {
    if (!editingTemplate || !editingTemplate.name.trim()) return;
    setTemplates((prev) => {
      const exists = prev.find((t) => t.id === editingTemplate.id);
      if (exists) return prev.map((t) => (t.id === editingTemplate.id ? editingTemplate : t));
      return [...prev, editingTemplate];
    });
    setEditorOpen(false);
    setEditingTemplate(null);
    setIsNew(false);
  }

  function handleDelete(id: string) {
    setTemplates((prev) => prev.filter((t) => t.id !== id));
    setDeleteConfirmId(null);
  }

  function handleAddHeader() {
    if (!editingTemplate) return;
    setEditingTemplate({
      ...editingTemplate,
      headers: [...editingTemplate.headers, { key: '', value: '' }],
    });
  }

  function handleRemoveHeader(index: number) {
    if (!editingTemplate) return;
    setEditingTemplate({
      ...editingTemplate,
      headers: editingTemplate.headers.filter((_, i) => i !== index),
    });
  }

  function handleUpdateHeader(index: number, field: 'key' | 'value', val: string) {
    if (!editingTemplate) return;
    const updated = editingTemplate.headers.map((h, i) =>
      i === index ? { ...h, [field]: val } : h
    );
    setEditingTemplate({ ...editingTemplate, headers: updated });
  }

  function insertPlaceholder(placeholder: string) {
    if (!editingTemplate) return;
    setEditingTemplate({
      ...editingTemplate,
      payload: editingTemplate.payload + placeholder,
    });
  }

  function handleTestSend() {
    // Mock test result
    setTestResult({ status: 200, body: '{"ok": true, "message": "Webhook received"}' });
  }

  // ── Rendered preview ──
  const renderedPreview = useMemo(() => {
    if (!editingTemplate) return '';
    return renderPayload(editingTemplate.payload);
  }, [editingTemplate]);

  const testRenderedPayload = useMemo(() => {
    if (!testTemplate) return '';
    let payload = testTemplate.payload;
    const data = { ...SAMPLE_DATA, '{{SIGNAL_TYPE}}': testSignalType };
    for (const [key, val] of Object.entries(data)) {
      payload = payload.replaceAll(key, val);
    }
    return payload;
  }, [testTemplate, testSignalType]);

  return (
    <div>
      <PageHeader
        title="Webhook Templates"
        subtitle="Reusable payload templates for alert delivery"
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

      <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
        {templates.length} template{templates.length !== 1 ? 's' : ''}
      </p>

      {/* ── Template List ── */}
      <div className="space-y-4">
        {templates.map((tmpl) => (
          <Card key={tmpl.id}>
            <div className="flex items-start justify-between gap-4">
              {/* Left: info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1 flex-wrap">
                  <h3 className="font-semibold text-sm truncate">{tmpl.name}</h3>
                  <span
                    className="text-xs px-2 py-0.5 rounded"
                    style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                  >
                    {tmpl.contentType}
                  </span>
                  {tmpl.usedBy > 0 && (
                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                      Used by {tmpl.usedBy} strateg{tmpl.usedBy === 1 ? 'y' : 'ies'}
                    </span>
                  )}
                </div>
                <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>
                  {tmpl.description}
                </p>
                <div className="flex items-center gap-2">
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>URL:</span>
                  <code
                    className="text-xs px-2 py-0.5 rounded"
                    style={{ background: 'var(--bg-input)', color: 'var(--text-secondary)', fontFamily: 'monospace' }}
                  >
                    {maskUrl(tmpl.url)}
                  </code>
                </div>
              </div>

              {/* Right: actions */}
              <div className="flex flex-col gap-2 shrink-0">
                <button
                  onClick={() => handleEdit(tmpl)}
                  className="px-3 py-1.5 rounded text-xs"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                >
                  Edit
                </button>
                <button
                  onClick={() => { setTestModalId(tmpl.id); setTestResult(null); }}
                  className="px-3 py-1.5 rounded text-xs"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                >
                  Test
                </button>
                <button
                  onClick={() => handleClone(tmpl)}
                  className="px-3 py-1.5 rounded text-xs"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                >
                  Clone
                </button>
                <button
                  onClick={() => setDeleteConfirmId(tmpl.id)}
                  className="px-3 py-1.5 rounded text-xs"
                  style={{ background: 'var(--red-muted)', border: '1px solid transparent', color: 'var(--red)' }}
                >
                  Delete
                </button>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* ── Template Editor Modal ── */}
      <Modal
        title={isNew ? 'New Webhook Template' : `Edit: ${editingTemplate?.name ?? ''}`}
        isOpen={editorOpen}
        onClose={() => { setEditorOpen(false); setEditingTemplate(null); }}
        width="780px"
      >
        {editingTemplate && (
          <div className="space-y-5">
            {/* Name & description */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs mb-1.5" style={{ color: 'var(--text-muted)' }}>Template Name</label>
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
                <label className="block text-xs mb-1.5" style={{ color: 'var(--text-muted)' }}>Description</label>
                <input
                  type="text"
                  value={editingTemplate.description}
                  onChange={(e) => setEditingTemplate({ ...editingTemplate, description: e.target.value })}
                  placeholder="Short description..."
                  className="w-full px-3 py-2 rounded-lg text-sm"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                />
              </div>
            </div>

            {/* URL & content type */}
            <div className="grid grid-cols-3 gap-4">
              <div className="col-span-2">
                <label className="block text-xs mb-1.5" style={{ color: 'var(--text-muted)' }}>Webhook URL</label>
                <input
                  type="url"
                  value={editingTemplate.url}
                  onChange={(e) => setEditingTemplate({ ...editingTemplate, url: e.target.value })}
                  placeholder="https://hooks.example.com/..."
                  className="w-full px-3 py-2 rounded-lg text-sm"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                />
              </div>
              <div>
                <label className="block text-xs mb-1.5" style={{ color: 'var(--text-muted)' }}>Content Type</label>
                <select
                  value={editingTemplate.contentType}
                  onChange={(e) => setEditingTemplate({ ...editingTemplate, contentType: e.target.value as 'JSON' | 'Form' })}
                  className="w-full px-3 py-2 rounded-lg text-sm"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                >
                  <option value="JSON">JSON</option>
                  <option value="Form">Form</option>
                </select>
              </div>
            </div>

            {/* Payload editor */}
            <div>
              <label className="block text-xs mb-1.5" style={{ color: 'var(--text-muted)' }}>Payload Template</label>
              <textarea
                value={editingTemplate.payload}
                onChange={(e) => setEditingTemplate({ ...editingTemplate, payload: e.target.value })}
                rows={8}
                className="w-full px-3 py-2 rounded-lg text-sm"
                style={{
                  background: 'var(--bg-input)',
                  border: '1px solid var(--border)',
                  color: 'var(--text-primary)',
                  fontFamily: 'monospace',
                  resize: 'vertical',
                }}
              />
              {/* Placeholder picker */}
              <div className="flex flex-wrap gap-1.5 mt-2">
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Insert:</span>
                {PLACEHOLDERS.map((ph) => (
                  <button
                    key={ph}
                    onClick={() => insertPlaceholder(ph)}
                    className="text-xs px-2 py-0.5 rounded cursor-pointer transition-colors"
                    style={{
                      background: 'var(--bg-card)',
                      border: '1px solid var(--border)',
                      color: 'var(--accent)',
                      fontFamily: 'monospace',
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--accent-muted)')}
                    onMouseLeave={(e) => (e.currentTarget.style.background = 'var(--bg-card)')}
                  >
                    {ph}
                  </button>
                ))}
              </div>
            </div>

            {/* Headers */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs" style={{ color: 'var(--text-muted)' }}>Custom Headers</label>
                <button
                  onClick={handleAddHeader}
                  className="text-xs px-2 py-1 rounded"
                  style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                >
                  + Add Header
                </button>
              </div>
              {editingTemplate.headers.length === 0 ? (
                <p className="text-xs py-2" style={{ color: 'var(--text-muted)' }}>No custom headers.</p>
              ) : (
                <div className="space-y-2">
                  {editingTemplate.headers.map((header, i) => (
                    <div key={i} className="flex items-center gap-2">
                      <input
                        type="text"
                        value={header.key}
                        onChange={(e) => handleUpdateHeader(i, 'key', e.target.value)}
                        placeholder="Header name"
                        className="flex-1 px-3 py-1.5 rounded-lg text-sm"
                        style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                      />
                      <input
                        type="text"
                        value={header.value}
                        onChange={(e) => handleUpdateHeader(i, 'value', e.target.value)}
                        placeholder="Header value"
                        className="flex-1 px-3 py-1.5 rounded-lg text-sm"
                        style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                      />
                      <button
                        onClick={() => handleRemoveHeader(i)}
                        className="w-6 h-6 rounded flex items-center justify-center text-xs shrink-0"
                        style={{ color: 'var(--red)', background: 'transparent' }}
                        onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--red-muted)')}
                        onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                      >
                        x
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Preview */}
            <div>
              <label className="block text-xs mb-1.5" style={{ color: 'var(--text-muted)' }}>Rendered Preview (sample data)</label>
              <pre
                className="px-3 py-2 rounded-lg text-xs overflow-x-auto"
                style={{
                  background: 'var(--bg-input)',
                  border: '1px solid var(--border)',
                  color: 'var(--text-secondary)',
                  fontFamily: 'monospace',
                  whiteSpace: 'pre-wrap',
                  maxHeight: '160px',
                }}
              >
                {renderedPreview}
              </pre>
            </div>

            {/* Actions */}
            <div className="flex justify-end gap-3 pt-3 border-t" style={{ borderColor: 'var(--border)' }}>
              <button
                onClick={() => { setEditorOpen(false); setEditingTemplate(null); }}
                className="px-4 py-2 rounded-lg text-sm"
                style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
              >
                Cancel
              </button>
              <button
                onClick={handleSave}
                className="px-4 py-2 rounded-lg text-sm font-medium"
                style={{
                  background: 'var(--accent)',
                  color: 'white',
                  opacity: editingTemplate.name.trim() ? 1 : 0.5,
                  cursor: editingTemplate.name.trim() ? 'pointer' : 'not-allowed',
                }}
                disabled={!editingTemplate.name.trim()}
              >
                {isNew ? 'Create Template' : 'Save Changes'}
              </button>
            </div>
          </div>
        )}
      </Modal>

      {/* ── Test Delivery Modal ── */}
      <Modal
        title={testTemplate ? `Test: ${testTemplate.name}` : 'Test Delivery'}
        isOpen={testModalId !== null}
        onClose={() => { setTestModalId(null); setTestResult(null); }}
        width="600px"
      >
        {testTemplate && (
          <div className="space-y-5">
            {/* Signal type selector */}
            <div>
              <label className="block text-xs mb-1.5" style={{ color: 'var(--text-muted)' }}>Signal Type</label>
              <select
                value={testSignalType}
                onChange={(e) => { setTestSignalType(e.target.value as 'ENTRY' | 'EXIT'); setTestResult(null); }}
                className="w-full px-3 py-2 rounded-lg text-sm"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
              >
                <option value="ENTRY">Entry</option>
                <option value="EXIT">Exit</option>
              </select>
            </div>

            {/* Rendered payload preview */}
            <div>
              <label className="block text-xs mb-1.5" style={{ color: 'var(--text-muted)' }}>Rendered Payload</label>
              <pre
                className="px-3 py-2 rounded-lg text-xs overflow-x-auto"
                style={{
                  background: 'var(--bg-input)',
                  border: '1px solid var(--border)',
                  color: 'var(--text-secondary)',
                  fontFamily: 'monospace',
                  whiteSpace: 'pre-wrap',
                  maxHeight: '200px',
                }}
              >
                {testRenderedPayload}
              </pre>
            </div>

            {/* Destination */}
            <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-muted)' }}>
              <span>Sending to:</span>
              <code className="px-2 py-0.5 rounded" style={{ background: 'var(--bg-input)', fontFamily: 'monospace' }}>
                {maskUrl(testTemplate.url)}
              </code>
            </div>

            {/* Test result */}
            {testResult && (
              <div
                className="px-4 py-3 rounded-lg"
                style={{
                  background: testResult.status === 200 ? 'var(--green-muted)' : 'var(--red-muted)',
                  border: `1px solid ${testResult.status === 200 ? 'var(--green)' : 'var(--red)'}`,
                }}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-sm font-medium" style={{ color: testResult.status === 200 ? 'var(--green)' : 'var(--red)' }}>
                    Status: {testResult.status}
                  </span>
                </div>
                <pre className="text-xs" style={{ fontFamily: 'monospace', color: 'var(--text-secondary)', whiteSpace: 'pre-wrap' }}>
                  {testResult.body}
                </pre>
              </div>
            )}

            {/* Actions */}
            <div className="flex justify-end gap-3 pt-3 border-t" style={{ borderColor: 'var(--border)' }}>
              <button
                onClick={() => { setTestModalId(null); setTestResult(null); }}
                className="px-4 py-2 rounded-lg text-sm"
                style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
              >
                Close
              </button>
              <button
                onClick={handleTestSend}
                className="px-4 py-2 rounded-lg text-sm font-medium"
                style={{ background: 'var(--accent)', color: 'white' }}
              >
                Send Test
              </button>
            </div>
          </div>
        )}
      </Modal>

      {/* ── Delete Confirmation Modal ── */}
      <Modal
        title="Delete Template"
        isOpen={deleteConfirmId !== null}
        onClose={() => setDeleteConfirmId(null)}
        width="420px"
      >
        {deleteConfirmId && (() => {
          const target = templates.find((t) => t.id === deleteConfirmId);
          if (!target) return null;
          return (
            <div>
              <p className="text-sm mb-2" style={{ color: 'var(--text-secondary)' }}>
                Delete &ldquo;{target.name}&rdquo;?
              </p>
              {target.usedBy > 0 && (
                <p className="text-xs mb-4 px-3 py-2 rounded-lg" style={{ background: 'var(--red-muted)', color: 'var(--red)' }}>
                  Warning: This template is used by {target.usedBy} strateg{target.usedBy === 1 ? 'y' : 'ies'}. Deleting it will remove the webhook configuration from those strategies.
                </p>
              )}
              <div className="flex justify-end gap-3 mt-4">
                <button
                  onClick={() => setDeleteConfirmId(null)}
                  className="px-4 py-2 rounded-lg text-sm"
                  style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                >
                  Cancel
                </button>
                <button
                  onClick={() => handleDelete(deleteConfirmId)}
                  className="px-4 py-2 rounded-lg text-sm font-medium"
                  style={{ background: 'var(--red)', color: 'white' }}
                >
                  Yes, Delete
                </button>
              </div>
            </div>
          );
        })()}
      </Modal>
    </div>
  );
}
