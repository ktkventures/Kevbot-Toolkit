'use client';

import { useState, useRef, useEffect } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import Modal from '@/components/Modal';

/* ---------- Types ---------- */
interface PackParam {
  key: string;
  label: string;
  value: number;
  min: number;
  max: number;
}

interface PackTrigger {
  name: string;
  direction: 'LONG' | 'SHORT';
  exec: string;
}

interface UserPack {
  id: string;
  name: string;
  category: string;
  enabled: boolean;
  params: PackParam[];
  outputs: string[];
  triggers: PackTrigger[];
  createdAt: string;
}

/* ---------- Mock Data ---------- */
const mockPacks: UserPack[] = [
  {
    id: 'my-rsi',
    name: 'My RSI Pack',
    category: 'Momentum',
    enabled: true,
    params: [
      { key: 'period', label: 'Period', value: 14, min: 2, max: 100 },
      { key: 'overbought', label: 'Overbought', value: 70, min: 50, max: 95 },
      { key: 'oversold', label: 'Oversold', value: 30, min: 5, max: 50 },
    ],
    outputs: ['OVERBOUGHT', 'NEUTRAL', 'OVERSOLD'],
    triggers: [
      { name: 'RSI Cross Above OB', direction: 'SHORT', exec: '[C]' },
      { name: 'RSI Cross Below OS', direction: 'LONG', exec: '[C]' },
    ],
    createdAt: '2026-03-18',
  },
];

const categoryColors: Record<string, { color: string; bg: string }> = {
  'Momentum': { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  'Moving Averages': { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  'Volume': { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  'Volatility': { color: 'var(--red)', bg: 'var(--red-muted)' },
  'Trend': { color: 'var(--green)', bg: 'var(--green-muted)' },
  'Custom': { color: 'var(--accent)', bg: 'var(--accent-muted)' },
};

const directionColors: Record<string, { color: string; bg: string }> = {
  'LONG': { color: 'var(--green)', bg: 'var(--green-muted)' },
  'SHORT': { color: 'var(--red)', bg: 'var(--red-muted)' },
};

const templateOptions = [
  { value: '', label: 'Blank (no template)' },
  { value: 'rsi', label: 'RSI' },
  { value: 'bollinger', label: 'Bollinger Bands' },
  { value: 'stochastic', label: 'Stochastic' },
];

/* ---------- Helpers ---------- */
function paramSummary(params: PackParam[]): string {
  return params.map((p) => `${p.label}: ${p.value}`).join(', ');
}

/* ---------- Overflow Menu ---------- */
function OverflowMenu({
  onEdit,
  onDelete,
}: {
  onEdit: () => void;
  onDelete: () => void;
}) {
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    if (open) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [open]);

  return (
    <div ref={menuRef} className="relative">
      <button
        onClick={(e) => { e.stopPropagation(); setOpen(!open); }}
        className="w-7 h-7 rounded flex items-center justify-center text-sm transition-colors"
        style={{ color: 'var(--text-muted)', background: 'transparent' }}
        onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-input)')}
        onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
        title="More actions"
      >
        ...
      </button>
      {open && (
        <div
          className="absolute right-0 top-8 z-20 rounded-lg border py-1 min-w-[140px]"
          style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border)', boxShadow: '0 8px 24px rgba(0,0,0,0.3)' }}
        >
          <button
            onClick={(e) => { e.stopPropagation(); setOpen(false); onEdit(); }}
            className="w-full text-left px-3 py-2 text-sm transition-colors"
            style={{ color: 'var(--text-primary)' }}
            onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-card)')}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
          >
            Rename
          </button>
          <div className="my-1 border-t" style={{ borderColor: 'var(--border)' }} />
          <button
            onClick={(e) => { e.stopPropagation(); setOpen(false); onDelete(); }}
            className="w-full text-left px-3 py-2 text-sm transition-colors"
            style={{ color: 'var(--red)' }}
            onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-card)')}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
          >
            Delete
          </button>
        </div>
      )}
    </div>
  );
}

/* ---------- Pack Row (Accordion) ---------- */
function PackRow({
  pack,
  expanded,
  onToggleExpand,
  onToggleEnabled,
  onParamChange,
  onSaveParams,
  onEdit,
  onDelete,
}: {
  pack: UserPack;
  expanded: boolean;
  onToggleExpand: () => void;
  onToggleEnabled: () => void;
  onParamChange: (key: string, value: number) => void;
  onSaveParams: () => void;
  onEdit: () => void;
  onDelete: () => void;
}) {
  const catStyle = categoryColors[pack.category] || { color: 'var(--text-muted)', bg: 'var(--bg-input)' };

  return (
    <div
      className="rounded-xl border overflow-hidden transition-all"
      style={{
        background: 'var(--bg-card)',
        borderColor: expanded ? 'var(--accent)' : 'var(--border)',
        boxShadow: 'var(--card-shadow)',
        backdropFilter: 'var(--card-backdrop)',
        WebkitBackdropFilter: 'var(--card-backdrop)',
        opacity: pack.enabled ? 1 : 0.6,
      }}
    >
      {/* Collapsed header row */}
      <div
        className="flex items-center gap-3 px-4 py-3 cursor-pointer select-none"
        onClick={onToggleExpand}
      >
        {/* Toggle switch */}
        <button
          onClick={(e) => { e.stopPropagation(); onToggleEnabled(); }}
          className="w-9 h-5 rounded-full relative flex-shrink-0 transition-colors"
          style={{
            background: pack.enabled ? 'var(--accent)' : 'var(--bg-input)',
            border: pack.enabled ? 'none' : '1px solid var(--border)',
          }}
        >
          <div
            className="w-3.5 h-3.5 rounded-full absolute transition-all"
            style={{
              background: pack.enabled ? 'white' : 'var(--text-muted)',
              top: '3px',
              left: pack.enabled ? '19px' : '3px',
            }}
          />
        </button>

        {/* Name */}
        <span className="font-semibold text-sm whitespace-nowrap" style={{ color: 'var(--text-primary)' }}>
          {pack.name}
        </span>

        {/* Category badge */}
        <span
          className="text-xs px-2 py-0.5 rounded whitespace-nowrap flex-shrink-0"
          style={{ color: catStyle.color, background: catStyle.bg }}
        >
          {pack.category}
        </span>

        {/* Param summary */}
        <span
          className="text-xs whitespace-nowrap flex-shrink-0 hidden sm:inline"
          style={{ color: 'var(--text-muted)' }}
        >
          {paramSummary(pack.params)}
        </span>

        {/* Trigger count */}
        <span
          className="text-xs whitespace-nowrap flex-shrink-0"
          style={{ color: 'var(--text-muted)' }}
        >
          {pack.triggers.length} triggers
        </span>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Edit/Delete always visible */}
        <button
          onClick={(e) => { e.stopPropagation(); onEdit(); }}
          className="px-2 py-1 rounded text-xs transition-colors flex-shrink-0"
          style={{ color: 'var(--accent)', background: 'var(--accent-muted)' }}
        >
          Edit
        </button>
        <button
          onClick={(e) => { e.stopPropagation(); onDelete(); }}
          className="px-2 py-1 rounded text-xs transition-colors flex-shrink-0"
          style={{ color: 'var(--red)', background: 'var(--red-muted)' }}
        >
          Delete
        </button>

        {/* Expand chevron */}
        <span
          className="text-sm flex-shrink-0 transition-transform"
          style={{
            color: 'var(--text-muted)',
            transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
          }}
        >
          &#9662;
        </span>

        {/* Overflow for rename */}
        <OverflowMenu
          onEdit={onEdit}
          onDelete={onDelete}
        />
      </div>

      {/* Expanded content */}
      {expanded && (
        <div
          className="px-4 pb-4 border-t"
          style={{ borderColor: 'var(--border)' }}
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 pt-4">
            {/* Parameters panel */}
            <div
              className="rounded-lg border p-4"
              style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border)' }}
            >
              <h4 className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-muted)' }}>
                Parameters
              </h4>
              <div className="space-y-3">
                {pack.params.map((param) => (
                  <div key={param.key} className="flex items-center gap-3">
                    <label className="text-sm w-32 flex-shrink-0" style={{ color: 'var(--text-secondary)' }}>
                      {param.label}
                    </label>
                    <input
                      type="number"
                      className="w-20 px-2 py-1.5 rounded text-sm text-center font-mono"
                      style={{
                        background: 'var(--bg-input)',
                        border: '1px solid var(--border)',
                        color: 'var(--text-primary)',
                      }}
                      value={param.value}
                      min={param.min}
                      max={param.max}
                      onChange={(e) => onParamChange(param.key, parseFloat(e.target.value) || param.min)}
                      onClick={(e) => e.stopPropagation()}
                    />
                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                      ({param.min}-{param.max})
                    </span>
                  </div>
                ))}
              </div>
              <div className="flex gap-2 mt-4">
                <button
                  className="px-3 py-1.5 rounded text-xs font-medium"
                  style={{ background: 'var(--accent)', color: 'white' }}
                  onClick={(e) => { e.stopPropagation(); onSaveParams(); }}
                >
                  Save
                </button>
              </div>
            </div>

            {/* Quick Reference panel */}
            <div
              className="rounded-lg border p-4"
              style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border)' }}
            >
              <h4 className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-muted)' }}>
                Quick Reference
              </h4>

              {/* Outputs */}
              <div className="mb-3">
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Outputs</span>
                <div className="flex flex-wrap gap-1 mt-1">
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
              </div>

              {/* Triggers */}
              <div>
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Triggers</span>
                <div className="space-y-1 mt-1">
                  {pack.triggers.map((trigger) => {
                    const dirStyle = directionColors[trigger.direction] || { color: 'var(--text-muted)', bg: 'var(--bg-input)' };
                    return (
                      <div
                        key={trigger.name}
                        className="flex items-center gap-2 px-2 py-1 rounded text-xs"
                        style={{ background: 'var(--bg-input)' }}
                      >
                        <span
                          className="font-mono px-1 py-0.5 rounded flex-shrink-0"
                          style={{ color: 'var(--green)', background: 'var(--green-muted)' }}
                        >
                          {trigger.exec}
                        </span>
                        <span className="flex-1" style={{ color: 'var(--text-primary)' }}>
                          {trigger.name}
                        </span>
                        <span
                          className="px-1.5 py-0.5 rounded font-medium flex-shrink-0"
                          style={{ color: dirStyle.color, background: dirStyle.bg, fontSize: '10px' }}
                        >
                          {trigger.direction}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Created date */}
              <p className="text-xs mt-3" style={{ color: 'var(--text-muted)' }}>
                Created {pack.createdAt}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ---------- Main Component ---------- */
export default function UserPacksV3() {
  const [packs, setPacks] = useState<UserPack[]>(mockPacks);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showRenameModal, setShowRenameModal] = useState<string | null>(null);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const [newName, setNewName] = useState('');
  const [newCategory, setNewCategory] = useState('Momentum');
  const [newTemplate, setNewTemplate] = useState('');

  const enabledCount = packs.filter((p) => p.enabled).length;

  function toggleEnabled(id: string) {
    setPacks((prev) => prev.map((p) => p.id === id ? { ...p, enabled: !p.enabled } : p));
  }

  function updateParam(packId: string, paramKey: string, value: number) {
    setPacks((prev) =>
      prev.map((p) =>
        p.id === packId
          ? { ...p, params: p.params.map((pr) => pr.key === paramKey ? { ...pr, value } : pr) }
          : p,
      ),
    );
  }

  function deletePack(packId: string) {
    setPacks((prev) => prev.filter((p) => p.id !== packId));
    setShowDeleteConfirm(null);
    if (expandedId === packId) setExpandedId(null);
  }

  function renamePack(packId: string, newName: string) {
    setPacks((prev) =>
      prev.map((p) => p.id === packId ? { ...p, name: newName } : p),
    );
    setShowRenameModal(null);
  }

  function createPack() {
    if (!newName.trim()) return;
    const id = `user-${newName.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}`;
    const newPack: UserPack = {
      id,
      name: newName,
      category: newCategory,
      enabled: true,
      params: newTemplate === 'rsi'
        ? [
          { key: 'period', label: 'Period', value: 14, min: 2, max: 100 },
          { key: 'overbought', label: 'Overbought', value: 70, min: 50, max: 95 },
          { key: 'oversold', label: 'Oversold', value: 30, min: 5, max: 50 },
        ]
        : [],
      outputs: newTemplate === 'rsi' ? ['OVERBOUGHT', 'NEUTRAL', 'OVERSOLD'] : [],
      triggers: [],
      createdAt: new Date().toISOString().split('T')[0],
    };
    setPacks((prev) => [...prev, newPack]);
    setExpandedId(id);
    setShowCreateModal(false);
    setNewName('');
    setNewCategory('Momentum');
    setNewTemplate('');
  }

  const renamePack_ = showRenameModal ? packs.find((p) => p.id === showRenameModal) : null;
  const deletePack_ = showDeleteConfirm ? packs.find((p) => p.id === showDeleteConfirm) : null;

  return (
    <div>
      <PageHeader
        title="User Packs"
        subtitle={packs.length > 0 ? `${packs.length} packs \u00b7 ${enabledCount} enabled` : 'Custom indicator packs you\'ve created'}
        actions={
          <button
            onClick={() => setShowCreateModal(true)}
            className="px-4 py-2 rounded-lg text-sm font-medium transition-colors"
            style={{ background: 'var(--accent)', color: 'white' }}
          >
            + Create Pack
          </button>
        }
      />

      {/* Empty state */}
      {packs.length === 0 && (
        <Card>
          <div className="text-center py-16">
            <p className="text-lg font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
              Create your first pack
            </p>
            <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>
              Build a custom indicator pack with your own parameters, outputs, and triggers.
            </p>
            <button
              onClick={() => setShowCreateModal(true)}
              className="px-5 py-2.5 rounded-lg text-sm font-medium"
              style={{ background: 'var(--accent)', color: 'white' }}
            >
              + Create Pack
            </button>
          </div>
        </Card>
      )}

      {/* Pack list */}
      {packs.length > 0 && (
        <div className="space-y-2">
          {packs.map((pack) => (
            <PackRow
              key={pack.id}
              pack={pack}
              expanded={expandedId === pack.id}
              onToggleExpand={() => setExpandedId(expandedId === pack.id ? null : pack.id)}
              onToggleEnabled={() => toggleEnabled(pack.id)}
              onParamChange={(key, val) => updateParam(pack.id, key, val)}
              onSaveParams={() => { /* toast: saved */ }}
              onEdit={() => {
                setRenameValue(pack.name);
                setShowRenameModal(pack.id);
              }}
              onDelete={() => setShowDeleteConfirm(pack.id)}
            />
          ))}
        </div>
      )}

      {/* Create Pack Modal */}
      <Modal title="Create Custom Pack" isOpen={showCreateModal} onClose={() => setShowCreateModal(false)} width="440px">
        <div className="space-y-4">
          <div>
            <label className="text-sm mb-1 block" style={{ color: 'var(--text-secondary)' }}>Pack Name</label>
            <input
              className="w-full px-3 py-2 rounded-lg text-sm"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
              placeholder="e.g. My RSI Pack"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              autoFocus
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
            <label className="text-sm mb-1 block" style={{ color: 'var(--text-secondary)' }}>Start from Template</label>
            <select
              className="w-full px-3 py-2 rounded-lg text-sm"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
              value={newTemplate}
              onChange={(e) => setNewTemplate(e.target.value)}
            >
              {templateOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>
          <div className="flex gap-3 pt-2">
            <button
              className="px-4 py-2 rounded-lg text-sm font-medium flex-1"
              style={{
                background: newName.trim() ? 'var(--accent)' : 'var(--bg-input)',
                color: newName.trim() ? 'white' : 'var(--text-muted)',
                cursor: newName.trim() ? 'pointer' : 'not-allowed',
              }}
              onClick={createPack}
              disabled={!newName.trim()}
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

      {/* Rename Modal */}
      <Modal
        title={renamePack_ ? `Rename - ${renamePack_.name}` : 'Rename Pack'}
        isOpen={!!showRenameModal}
        onClose={() => setShowRenameModal(null)}
        width="400px"
      >
        <div className="space-y-4">
          <div>
            <label className="text-sm mb-1 block" style={{ color: 'var(--text-secondary)' }}>Pack Name</label>
            <input
              className="w-full px-3 py-2 rounded-lg text-sm"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
              value={renameValue}
              onChange={(e) => setRenameValue(e.target.value)}
              autoFocus
            />
          </div>
          <div className="flex gap-3 pt-2">
            <button
              className="px-4 py-2 rounded-lg text-sm font-medium flex-1"
              style={{ background: 'var(--accent)', color: 'white' }}
              onClick={() => showRenameModal && renamePack(showRenameModal, renameValue)}
            >
              Rename
            </button>
            <button
              className="px-4 py-2 rounded-lg text-sm flex-1"
              style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
              onClick={() => setShowRenameModal(null)}
            >
              Cancel
            </button>
          </div>
        </div>
      </Modal>

      {/* Delete Confirmation Modal */}
      <Modal
        title="Delete Pack"
        isOpen={!!showDeleteConfirm}
        onClose={() => setShowDeleteConfirm(null)}
        width="400px"
      >
        <div className="space-y-4">
          <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
            Are you sure you want to delete <strong style={{ color: 'var(--text-primary)' }}>{deletePack_?.name}</strong>?
            This action cannot be undone.
          </p>
          <div className="flex gap-3 pt-2">
            <button
              className="px-4 py-2 rounded-lg text-sm font-medium flex-1"
              style={{ background: 'var(--red)', color: 'white' }}
              onClick={() => showDeleteConfirm && deletePack(showDeleteConfirm)}
            >
              Delete
            </button>
            <button
              className="px-4 py-2 rounded-lg text-sm flex-1"
              style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
              onClick={() => setShowDeleteConfirm(null)}
            >
              Cancel
            </button>
          </div>
        </div>
      </Modal>
    </div>
  );
}
