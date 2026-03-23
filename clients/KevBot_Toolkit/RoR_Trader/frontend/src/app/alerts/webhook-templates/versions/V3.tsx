'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import Modal from '@/components/Modal';

/* ========================================================================= */
/* TYPES                                                                      */
/* ========================================================================= */

interface WebhookTemplate {
  id: string;
  name: string;
  url: string;
  usedBy: number;
  payload: string;
}

/* ========================================================================= */
/* MOCK DATA                                                                  */
/* ========================================================================= */

const initialTemplates: WebhookTemplate[] = [
  {
    id: 't1', name: 'TradeThePool Market Buy',
    url: 'https://api.tradethepool.com/webhook/abc123', usedBy: 3,
    payload: '{\n  "action": "buy",\n  "symbol": "{{SYMBOL}}",\n  "qty": 1,\n  "type": "market"\n}',
  },
  {
    id: 't2', name: 'TradeThePool Market Sell',
    url: 'https://api.tradethepool.com/webhook/abc123', usedBy: 3,
    payload: '{\n  "action": "sell",\n  "symbol": "{{SYMBOL}}",\n  "qty": 1,\n  "type": "market"\n}',
  },
  {
    id: 't3', name: 'Discord Alert',
    url: 'https://discord.com/api/webhooks/123456/abcdef', usedBy: 2,
    payload: '{\n  "embeds": [{\n    "title": "{{SIGNAL_TYPE}} Signal",\n    "description": "{{STRATEGY}} @ {{PRICE}}",\n    "color": 65280\n  }]\n}',
  },
  {
    id: 't4', name: 'Slack Notification',
    url: 'https://hooks.slack.com/services/T01/B02/xyz789', usedBy: 1,
    payload: '{\n  "text": "{{SIGNAL_TYPE}}: {{STRATEGY}} at {{PRICE}}"\n}',
  },
];

/* ========================================================================= */
/* HELPERS                                                                    */
/* ========================================================================= */

function maskUrl(url: string): string {
  try {
    const parsed = new URL(url);
    return `${parsed.hostname.substring(0, 16)}...`;
  } catch {
    return url.substring(0, 20) + '...';
  }
}

function generateId(): string {
  return 'id-' + Math.random().toString(36).substring(2, 9);
}

/* ========================================================================= */
/* COMPONENT                                                                  */
/* ========================================================================= */

export default function WebhookTemplatesV3() {
  const [templates, setTemplates] = useState<WebhookTemplate[]>(initialTemplates);
  const [modalOpen, setModalOpen] = useState(false);
  const [editingTemplate, setEditingTemplate] = useState<WebhookTemplate | null>(null);
  const [isNew, setIsNew] = useState(false);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [testingId, setTestingId] = useState<string | null>(null);
  const [testStatus, setTestStatus] = useState<'idle' | 'sending' | 'success' | 'error'>('idle');

  function handleNew() {
    setEditingTemplate({
      id: generateId(), name: '', url: '', usedBy: 0, payload: '{\n  \n}',
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
  }

  function handleTest(id: string) {
    setTestingId(id);
    setTestStatus('sending');
    // Simulate test
    setTimeout(() => {
      setTestStatus('success');
      setTimeout(() => { setTestingId(null); setTestStatus('idle'); }, 2000);
    }, 800);
  }

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

      {/* Table-like rows */}
      <Card>
        {/* Header row */}
        <div
          className="grid items-center gap-4 pb-2 mb-2 border-b text-xs font-medium"
          style={{ gridTemplateColumns: '1fr 160px 60px 120px', borderColor: 'var(--border)', color: 'var(--text-muted)' }}
        >
          <span>Name</span>
          <span>URL</span>
          <span className="text-center">Used</span>
          <span className="text-right">Actions</span>
        </div>

        {/* Template rows */}
        <div className="space-y-1">
          {templates.map((tmpl) => (
            <div key={tmpl.id}>
              <div
                className="grid items-center gap-4 py-2 rounded-lg px-1 transition-colors"
                style={{ gridTemplateColumns: '1fr 160px 60px 120px' }}
                onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-input)')}
                onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
              >
                <span className="text-sm font-medium truncate" style={{ color: 'var(--text-primary)' }}>
                  {tmpl.name}
                </span>
                <code className="text-xs truncate" style={{ color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                  {maskUrl(tmpl.url)}
                </code>
                <span className="text-xs text-center" style={{ color: 'var(--text-muted)' }}>
                  {tmpl.usedBy}
                </span>
                <div className="flex items-center justify-end gap-1">
                  <button
                    onClick={() => handleEdit(tmpl)}
                    className="w-7 h-7 rounded flex items-center justify-center text-xs transition-colors"
                    style={{ color: 'var(--text-muted)' }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--accent-muted)'; e.currentTarget.style.color = 'var(--accent)'; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--text-muted)'; }}
                    title="Edit"
                  >
                    E
                  </button>
                  <button
                    onClick={() => handleTest(tmpl.id)}
                    className="w-7 h-7 rounded flex items-center justify-center text-xs transition-colors"
                    style={{
                      color: testingId === tmpl.id && testStatus === 'success' ? 'var(--green)' : 'var(--text-muted)',
                      background: testingId === tmpl.id && testStatus === 'success' ? 'var(--green-muted)' : 'transparent',
                    }}
                    onMouseEnter={(e) => { if (testingId !== tmpl.id) { e.currentTarget.style.background = 'var(--accent-muted)'; e.currentTarget.style.color = 'var(--accent)'; } }}
                    onMouseLeave={(e) => { if (testingId !== tmpl.id) { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--text-muted)'; } }}
                    title="Test"
                    disabled={testingId === tmpl.id && testStatus === 'sending'}
                  >
                    {testingId === tmpl.id && testStatus === 'sending' ? '...' : testingId === tmpl.id && testStatus === 'success' ? 'OK' : 'T'}
                  </button>
                  {deleteConfirmId === tmpl.id ? (
                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => handleDelete(tmpl.id)}
                        className="w-7 h-7 rounded flex items-center justify-center text-[10px] font-bold"
                        style={{ background: 'var(--red)', color: 'white' }}
                      >
                        Y
                      </button>
                      <button
                        onClick={() => setDeleteConfirmId(null)}
                        className="w-7 h-7 rounded flex items-center justify-center text-[10px]"
                        style={{ border: '1px solid var(--border)', color: 'var(--text-muted)' }}
                      >
                        N
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => setDeleteConfirmId(tmpl.id)}
                      className="w-7 h-7 rounded flex items-center justify-center text-xs transition-colors"
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
            </div>
          ))}

          {templates.length === 0 && (
            <p className="text-sm py-6 text-center" style={{ color: 'var(--text-muted)' }}>
              No templates yet. Create one to get started.
            </p>
          )}
        </div>
      </Card>

      {/* Edit/Create Modal */}
      <Modal
        title={isNew ? 'New Template' : `Edit: ${editingTemplate?.name ?? ''}`}
        isOpen={modalOpen}
        onClose={() => { setModalOpen(false); setEditingTemplate(null); }}
        width="560px"
      >
        {editingTemplate && (
          <div className="space-y-4">
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
              <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>
                Available placeholders: {'{{STRATEGY}}'}, {'{{SIGNAL_TYPE}}'}, {'{{PRICE}}'}, {'{{SYMBOL}}'}, {'{{DIRECTION}}'}, {'{{TIMEFRAME}}'}, {'{{TIMESTAMP}}'}, {'{{PNL}}'}, {'{{EXEC_TYPE}}'}
              </p>
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
