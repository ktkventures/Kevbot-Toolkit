'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import Modal from '@/components/Modal';
import ChartPlaceholder from '@/components/ChartPlaceholder';

interface UserPack {
  id: string;
  name: string;
  category: string;
  enabled: boolean;
  isDefault: false;
  params: string;
  outputs: string[];
  triggers: number;
  createdAt: string;
}

const mockPacks: UserPack[] = [
  { id: 'my-rsi', name: 'My RSI Pack', category: 'Momentum', enabled: true, isDefault: false, params: 'Period: 14, OB: 70, OS: 30', outputs: ['OVERBOUGHT','NEUTRAL','OVERSOLD'], triggers: 2, createdAt: '2026-03-18' },
];

const categoryColors: Record<string, { color: string; bg: string }> = {
  'Momentum': { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  'Moving Averages': { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  'Volume': { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  'Volatility': { color: 'var(--red)', bg: 'var(--red-muted)' },
  'Trend': { color: 'var(--green)', bg: 'var(--green-muted)' },
  'Custom': { color: 'var(--accent)', bg: 'var(--accent-muted)' },
};

export default function UserPacksV1() {
  const [packs, setPacks] = useState<UserPack[]>(mockPacks);
  const [detailPack, setDetailPack] = useState<UserPack | null>(null);
  const [showDeleteModal, setShowDeleteModal] = useState<string | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newName, setNewName] = useState('');
  const [newCategory, setNewCategory] = useState('Momentum');

  const enabledCount = packs.filter((p) => p.enabled).length;

  function togglePack(id: string) {
    setPacks((prev) => prev.map((p) => p.id === id ? { ...p, enabled: !p.enabled } : p));
  }

  function deletePack(id: string) {
    setPacks((prev) => prev.filter((p) => p.id !== id));
    setShowDeleteModal(null);
    setDetailPack(null);
  }

  // Detail view
  if (detailPack) {
    return (
      <div>
        <PageHeader
          title={detailPack.name}
          subtitle={`${detailPack.category} | Created ${detailPack.createdAt}`}
          backHref="#"
          actions={
            <button
              onClick={() => setDetailPack(null)}
              className="px-4 py-2 rounded-lg text-sm"
              style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
            >
              Back to Packs
            </button>
          }
        />

        <TabBar tabs={['Parameters', 'Plot Settings', 'Outputs & Triggers', 'Preview', 'Danger Zone']}>
          {(tab) => (
            <div>
              {tab === 'Parameters' && (
                <Card>
                  <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Pack Parameters</h3>
                  <div className="space-y-4">
                    {detailPack.params.split(', ').map((param) => {
                      const [label, value] = param.split(': ');
                      return (
                        <div key={label} className="flex items-center gap-4">
                          <label className="text-sm w-40" style={{ color: 'var(--text-secondary)' }}>{label.trim()}</label>
                          <input
                            className="px-3 py-2 rounded-lg text-sm flex-1"
                            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                            defaultValue={value?.trim() || ''}
                          />
                        </div>
                      );
                    })}
                  </div>
                  <button className="mt-6 px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: 'white' }}>
                    Save Changes
                  </button>
                </Card>
              )}

              {tab === 'Plot Settings' && (
                <Card>
                  <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Visual Settings</h3>
                  <div className="space-y-4">
                    {[
                      { label: 'Line Color', value: '#f59e0b' },
                      { label: 'Overbought Zone', value: '#ef444440' },
                      { label: 'Oversold Zone', value: '#22c55e40' },
                    ].map((setting) => (
                      <div key={setting.label} className="flex items-center gap-4">
                        <label className="text-sm w-48" style={{ color: 'var(--text-secondary)' }}>{setting.label}</label>
                        <div className="flex items-center gap-2">
                          <div
                            className="w-8 h-8 rounded border"
                            style={{ background: setting.value, borderColor: 'var(--border)' }}
                          />
                          <input
                            className="px-3 py-2 rounded-lg text-sm font-mono w-28"
                            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                            defaultValue={setting.value}
                          />
                        </div>
                      </div>
                    ))}
                    <div className="flex items-center gap-4">
                      <label className="text-sm w-48" style={{ color: 'var(--text-secondary)' }}>Line Width</label>
                      <input type="range" min="1" max="5" defaultValue="2" className="flex-1" />
                      <span className="text-sm w-8 text-center" style={{ color: 'var(--text-muted)' }}>2px</span>
                    </div>
                  </div>
                  <button className="mt-6 px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: 'white' }}>
                    Save Plot Settings
                  </button>
                </Card>
              )}

              {tab === 'Outputs & Triggers' && (
                <div className="grid grid-cols-2 gap-6">
                  <Card>
                    <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Interpreter Outputs</h3>
                    <div className="space-y-2">
                      {detailPack.outputs.map((output) => (
                        <div
                          key={output}
                          className="flex items-center justify-between px-3 py-2 rounded-lg"
                          style={{ background: 'var(--bg-input)' }}
                        >
                          <span className="text-sm font-mono" style={{ color: 'var(--text-primary)' }}>{output}</span>
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                            {output === 'OVERBOUGHT' ? 'RSI > 70' :
                             output === 'OVERSOLD' ? 'RSI < 30' : 'RSI 30-70'}
                          </span>
                        </div>
                      ))}
                    </div>
                  </Card>

                  <Card>
                    <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Available Triggers</h3>
                    <div className="space-y-2">
                      {[
                        { name: 'RSI Cross Above OB', execType: '[C]' },
                        { name: 'RSI Cross Below OS', execType: '[C]' },
                      ].map((trigger) => (
                        <div
                          key={trigger.name}
                          className="flex items-center justify-between px-3 py-2 rounded-lg"
                          style={{ background: 'var(--bg-input)' }}
                        >
                          <span className="text-sm" style={{ color: 'var(--text-primary)' }}>{trigger.name}</span>
                          <span
                            className="text-xs font-mono px-2 py-0.5 rounded"
                            style={{ color: 'var(--green)', background: 'var(--green-muted)' }}
                          >
                            {trigger.execType}
                          </span>
                        </div>
                      ))}
                    </div>
                  </Card>
                </div>
              )}

              {tab === 'Preview' && (
                <div>
                  <div className="flex gap-3 mb-4">
                    <select
                      className="px-3 py-2 rounded-lg text-sm"
                      style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                    >
                      <option>NVDA</option>
                      <option>SPY</option>
                    </select>
                    <select
                      className="px-3 py-2 rounded-lg text-sm"
                      style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                    >
                      <option>1Min</option>
                      <option>5Min</option>
                    </select>
                  </div>
                  <Card>
                    <ChartPlaceholder label={`Price chart with ${detailPack.name} overlay`} height={400} />
                  </Card>
                </div>
              )}

              {tab === 'Danger Zone' && (
                <Card>
                  <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--red)' }}>Danger Zone</h3>
                  <div className="space-y-6">
                    <div>
                      <label className="text-sm mb-2 block" style={{ color: 'var(--text-secondary)' }}>Rename Pack</label>
                      <div className="flex gap-3">
                        <input
                          className="px-3 py-2 rounded-lg text-sm flex-1"
                          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                          defaultValue={detailPack.name}
                        />
                        <button className="px-4 py-2 rounded-lg text-sm" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>
                          Rename
                        </button>
                      </div>
                    </div>

                    <div className="pt-4 border-t" style={{ borderColor: 'var(--border)' }}>
                      <p className="text-sm mb-2" style={{ color: 'var(--text-secondary)' }}>
                        Delete this pack permanently. This action cannot be undone.
                      </p>
                      <button
                        onClick={() => setShowDeleteModal(detailPack.id)}
                        className="px-4 py-2 rounded-lg text-sm font-medium"
                        style={{ background: 'var(--red)', color: 'white' }}
                      >
                        Delete Pack
                      </button>
                    </div>
                  </div>
                </Card>
              )}
            </div>
          )}
        </TabBar>

        {/* Delete confirmation modal */}
        <Modal title="Delete Pack" isOpen={!!showDeleteModal} onClose={() => setShowDeleteModal(null)} width="420px">
          <p className="text-sm mb-4" style={{ color: 'var(--text-secondary)' }}>
            Are you sure you want to delete <strong>{detailPack.name}</strong>? This action cannot be undone.
          </p>
          <div className="flex gap-3">
            <button
              onClick={() => showDeleteModal && deletePack(showDeleteModal)}
              className="px-4 py-2 rounded-lg text-sm font-medium flex-1"
              style={{ background: 'var(--red)', color: 'white' }}
            >
              Delete
            </button>
            <button
              onClick={() => setShowDeleteModal(null)}
              className="px-4 py-2 rounded-lg text-sm flex-1"
              style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
            >
              Cancel
            </button>
          </div>
        </Modal>
      </div>
    );
  }

  // Main list view
  return (
    <div>
      <PageHeader
        title="User Packs"
        subtitle="Custom indicator packs you've created"
        actions={
          <button
            onClick={() => setShowCreateModal(true)}
            className="px-4 py-2 rounded-lg text-sm font-medium"
            style={{ background: 'var(--accent)', color: 'white' }}
          >
            + Create Pack
          </button>
        }
      />

      <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>
        {packs.length} {packs.length === 1 ? 'pack' : 'packs'}{packs.length > 0 ? `, ${enabledCount} enabled` : ''}
      </p>

      {packs.length === 0 ? (
        <Card>
          <div className="text-center py-12">
            <p className="text-lg font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>No custom packs yet</p>
            <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>
              Create your first pack or start from a template.
            </p>
            <div className="flex gap-3 justify-center">
              <button
                onClick={() => setShowCreateModal(true)}
                className="px-4 py-2 rounded-lg text-sm font-medium"
                style={{ background: 'var(--accent)', color: 'white' }}
              >
                + Create Pack
              </button>
              <button
                className="px-4 py-2 rounded-lg text-sm"
                style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
              >
                Browse Templates
              </button>
            </div>
          </div>
        </Card>
      ) : (
        <div className="space-y-3">
          {packs.map((pack) => (
            <Card key={pack.id}>
              <div className="flex items-center gap-4">
                {/* Toggle */}
                <button
                  onClick={() => togglePack(pack.id)}
                  className="w-10 h-6 rounded-full relative flex-shrink-0 transition-colors"
                  style={{
                    background: pack.enabled ? 'var(--accent)' : 'var(--bg-input)',
                    border: pack.enabled ? 'none' : '1px solid var(--border)',
                  }}
                >
                  <div
                    className="w-4 h-4 rounded-full absolute top-1 transition-all"
                    style={{
                      background: pack.enabled ? 'white' : 'var(--text-muted)',
                      left: pack.enabled ? '22px' : '4px',
                    }}
                  />
                </button>

                {/* Info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-semibold text-sm">{pack.name}</span>
                    <span
                      className="text-xs px-2 py-0.5 rounded"
                      style={{
                        color: categoryColors[pack.category]?.color || 'var(--text-muted)',
                        background: categoryColors[pack.category]?.bg || 'var(--bg-input)',
                      }}
                    >
                      {pack.category}
                    </span>
                  </div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    {pack.params} &middot; {pack.triggers} triggers &middot; Created {pack.createdAt}
                  </p>
                </div>

                {/* Outputs preview */}
                <div className="flex items-center gap-1 flex-shrink-0">
                  {pack.outputs.map((output) => (
                    <span
                      key={output}
                      className="text-xs font-mono px-1.5 py-0.5 rounded"
                      style={{ background: 'var(--bg-input)', color: 'var(--text-secondary)' }}
                    >
                      {output}
                    </span>
                  ))}
                </div>

                {/* Actions */}
                <div className="flex gap-2 flex-shrink-0">
                  <button
                    onClick={() => setDetailPack(pack)}
                    className="px-3 py-1.5 rounded text-xs"
                    style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                  >
                    Edit
                  </button>
                  <button
                    onClick={() => setShowDeleteModal(pack.id)}
                    className="px-3 py-1.5 rounded text-xs"
                    style={{ background: 'var(--red-muted)', color: 'var(--red)' }}
                  >
                    Delete
                  </button>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Create Pack Modal */}
      <Modal title="Create Custom Pack" isOpen={showCreateModal} onClose={() => setShowCreateModal(false)} width="500px">
        <div className="space-y-4">
          <div>
            <label className="text-sm mb-1 block" style={{ color: 'var(--text-secondary)' }}>Pack Name</label>
            <input
              className="w-full px-3 py-2 rounded-lg text-sm"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
              placeholder="e.g. My RSI Pack"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
            />
          </div>
          <div>
            <label className="text-sm mb-1 block" style={{ color: 'var(--text-secondary)' }}>Category</label>
            <select
              className="w-full px-3 py-2 rounded-lg text-sm"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
              value={newCategory}
              onChange={(e) => setNewCategory(e.target.value)}
            >
              {Object.keys(categoryColors).map((cat) => (
                <option key={cat} value={cat}>{cat}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-sm mb-1 block" style={{ color: 'var(--text-secondary)' }}>Starting Template</label>
            <select
              className="w-full px-3 py-2 rounded-lg text-sm"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
            >
              <option value="">Blank (no template)</option>
              <option value="rsi">RSI</option>
              <option value="bollinger">Bollinger Bands</option>
              <option value="stochastic">Stochastic</option>
            </select>
          </div>
          <div className="flex gap-3 pt-2">
            <button
              className="px-4 py-2 rounded-lg text-sm font-medium flex-1"
              style={{ background: 'var(--accent)', color: 'white' }}
              onClick={() => setShowCreateModal(false)}
            >
              Create Pack
            </button>
            <button
              className="px-4 py-2 rounded-lg text-sm flex-1"
              style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
              onClick={() => setShowCreateModal(false)}
            >
              Cancel
            </button>
          </div>
        </div>
      </Modal>

      {/* Delete confirmation modal */}
      <Modal title="Delete Pack" isOpen={!!showDeleteModal} onClose={() => setShowDeleteModal(null)} width="420px">
        <p className="text-sm mb-4" style={{ color: 'var(--text-secondary)' }}>
          Are you sure you want to delete this pack? This action cannot be undone.
        </p>
        <div className="flex gap-3">
          <button
            onClick={() => showDeleteModal && deletePack(showDeleteModal)}
            className="px-4 py-2 rounded-lg text-sm font-medium flex-1"
            style={{ background: 'var(--red)', color: 'white' }}
          >
            Delete
          </button>
          <button
            onClick={() => setShowDeleteModal(null)}
            className="px-4 py-2 rounded-lg text-sm flex-1"
            style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
          >
            Cancel
          </button>
        </div>
      </Modal>
    </div>
  );
}
