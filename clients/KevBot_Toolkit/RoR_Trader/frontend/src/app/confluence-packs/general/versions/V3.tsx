'use client';

import { useState, useRef, useEffect } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import Modal from '@/components/Modal';

/* ---------- Types ---------- */
interface PackParam {
  key: string;
  label: string;
  type: 'int' | 'float' | 'bool' | 'select';
  value: number | boolean | string;
  min?: number;
  max?: number;
  options?: string[];
}

interface PackTrigger {
  name: string;
  direction: 'BOTH';
  type: 'ENTRY' | 'EXIT';
  exec: string;
}

interface GeneralPack {
  id: string;
  name: string;
  category: 'Session' | 'Calendar';
  enabled: boolean;
  isDefault: boolean;
  params: PackParam[];
  outputs: string[];
  triggers: PackTrigger[];
}

/* ---------- Mock Data ---------- */
const mockPacks: GeneralPack[] = [
  {
    id: 'tod-default',
    name: 'Time of Day',
    category: 'Session',
    enabled: true,
    isDefault: true,
    params: [
      { key: 'start_hour', label: 'Start Hour', type: 'int', value: 9, min: 0, max: 23 },
      { key: 'start_minute', label: 'Start Minute', type: 'int', value: 30, min: 0, max: 59 },
      { key: 'end_hour', label: 'End Hour', type: 'int', value: 12, min: 0, max: 23 },
      { key: 'end_minute', label: 'End Minute', type: 'int', value: 0, min: 0, max: 59 },
    ],
    outputs: ['IN_WINDOW', 'OUT_OF_WINDOW'],
    triggers: [
      { name: 'Window Opens', direction: 'BOTH', type: 'ENTRY', exec: '[C]' },
      { name: 'Window Closes', direction: 'BOTH', type: 'EXIT', exec: '[C]' },
    ],
  },
  {
    id: 'session-default',
    name: 'Trading Session',
    category: 'Session',
    enabled: true,
    isDefault: true,
    params: [
      { key: 'session', label: 'Session', type: 'select', value: 'regular', options: ['pre_market', 'regular', 'after_hours', 'extended'] },
    ],
    outputs: ['IN_SESSION', 'OUT_OF_SESSION'],
    triggers: [
      { name: 'Session Opens', direction: 'BOTH', type: 'ENTRY', exec: '[C]' },
      { name: 'Session Closes', direction: 'BOTH', type: 'EXIT', exec: '[C]' },
    ],
  },
  {
    id: 'dow-default',
    name: 'Day of Week',
    category: 'Calendar',
    enabled: true,
    isDefault: true,
    params: [
      { key: 'monday', label: 'Monday', type: 'bool', value: true },
      { key: 'tuesday', label: 'Tuesday', type: 'bool', value: true },
      { key: 'wednesday', label: 'Wednesday', type: 'bool', value: true },
      { key: 'thursday', label: 'Thursday', type: 'bool', value: true },
      { key: 'friday', label: 'Friday', type: 'bool', value: true },
    ],
    outputs: ['ALLOWED_DAY', 'BLOCKED_DAY'],
    triggers: [],
  },
  {
    id: 'cal-default',
    name: 'Calendar Filter',
    category: 'Calendar',
    enabled: false,
    isDefault: true,
    params: [
      { key: 'avoid_fomc', label: 'Avoid FOMC', type: 'bool', value: true },
      { key: 'avoid_opex', label: 'Avoid OpEx', type: 'bool', value: false },
      { key: 'avoid_nfp', label: 'Avoid NFP', type: 'bool', value: true },
      { key: 'buffer_minutes', label: 'Buffer (min)', type: 'int', value: 30, min: 0, max: 120 },
    ],
    outputs: ['CLEAR', 'BLOCKED'],
    triggers: [
      { name: 'Event Block Starts', direction: 'BOTH', type: 'EXIT', exec: '[C]' },
      { name: 'Event Block Clears', direction: 'BOTH', type: 'ENTRY', exec: '[C]' },
    ],
  },
];

const categoryFilters = ['All', 'Session', 'Calendar'];

const categoryColors: Record<string, { color: string; bg: string }> = {
  'Session': { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  'Calendar': { color: 'var(--orange)', bg: 'var(--orange-muted)' },
};

/* ---------- Helpers ---------- */
function paramSummary(params: PackParam[]): string {
  return params
    .filter((p) => p.type === 'int' || p.type === 'float')
    .map((p) => {
      const short = p.label.replace('Start ', 'S').replace('End ', 'E').replace('Buffer (min)', 'Buf');
      return `${short}: ${p.value}`;
    })
    .join(', ');
}

function boolSummary(params: PackParam[]): string {
  const boolParams = params.filter((p) => p.type === 'bool');
  if (boolParams.length === 0) return '';
  const on = boolParams.filter((p) => p.value === true).map((p) => p.label);
  return on.length === boolParams.length ? 'All enabled' : on.join(', ') || 'None';
}

function selectSummary(params: PackParam[]): string {
  const selects = params.filter((p) => p.type === 'select');
  return selects.map((p) => String(p.value)).join(', ');
}

function getParamDisplay(params: PackParam[]): string {
  const parts: string[] = [];
  const sel = selectSummary(params);
  if (sel) parts.push(sel);
  const ps = paramSummary(params);
  if (ps) parts.push(ps);
  const bs = boolSummary(params);
  if (bs) parts.push(bs);
  return parts.join(' | ') || 'No params';
}

/* ---------- Overflow Menu ---------- */
function OverflowMenu({
  pack,
  onRename,
  onCopy,
  onDelete,
}: {
  pack: GeneralPack;
  onRename: () => void;
  onCopy: () => void;
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
          className="absolute right-0 top-8 z-20 rounded-lg border py-1 min-w-[160px]"
          style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border)', boxShadow: '0 8px 24px rgba(0,0,0,0.3)' }}
        >
          <button
            onClick={(e) => { e.stopPropagation(); setOpen(false); onRename(); }}
            className="w-full text-left px-3 py-2 text-sm transition-colors"
            style={{ color: pack.isDefault ? 'var(--text-muted)' : 'var(--text-primary)' }}
            onMouseEnter={(e) => { if (!pack.isDefault) e.currentTarget.style.background = 'var(--bg-card)'; }}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
            disabled={pack.isDefault}
          >
            Rename
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); setOpen(false); onCopy(); }}
            className="w-full text-left px-3 py-2 text-sm transition-colors"
            style={{ color: 'var(--text-primary)' }}
            onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-card)')}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
          >
            Duplicate
          </button>
          <div className="my-1 border-t" style={{ borderColor: 'var(--border)' }} />
          <button
            onClick={(e) => { e.stopPropagation(); setOpen(false); onDelete(); }}
            className="w-full text-left px-3 py-2 text-sm transition-colors"
            style={{ color: pack.isDefault ? 'var(--text-muted)' : 'var(--red)' }}
            onMouseEnter={(e) => { if (!pack.isDefault) e.currentTarget.style.background = 'var(--bg-card)'; }}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
            disabled={pack.isDefault}
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
  onRename,
  onCopy,
  onDelete,
}: {
  pack: GeneralPack;
  expanded: boolean;
  onToggleExpand: () => void;
  onToggleEnabled: () => void;
  onParamChange: (key: string, value: number | boolean | string) => void;
  onSaveParams: () => void;
  onRename: () => void;
  onCopy: () => void;
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
          {getParamDisplay(pack.params)}
        </span>

        {/* Spacer */}
        <div className="flex-1" />

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

        {/* Overflow menu */}
        <OverflowMenu
          pack={pack}
          onRename={onRename}
          onCopy={onCopy}
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
                    {param.type === 'bool' ? (
                      <button
                        onClick={(e) => { e.stopPropagation(); onParamChange(param.key, !param.value); }}
                        className="w-9 h-5 rounded-full relative flex-shrink-0 transition-colors"
                        style={{
                          background: param.value ? 'var(--accent)' : 'var(--bg-input)',
                          border: param.value ? 'none' : '1px solid var(--border)',
                        }}
                      >
                        <div
                          className="w-3.5 h-3.5 rounded-full absolute transition-all"
                          style={{
                            background: param.value ? 'white' : 'var(--text-muted)',
                            top: '3px',
                            left: param.value ? '19px' : '3px',
                          }}
                        />
                      </button>
                    ) : param.type === 'select' ? (
                      <select
                        className="px-2 py-1.5 rounded text-sm"
                        style={{
                          background: 'var(--bg-input)',
                          border: '1px solid var(--border)',
                          color: 'var(--text-primary)',
                        }}
                        value={String(param.value)}
                        onChange={(e) => onParamChange(param.key, e.target.value)}
                        onClick={(e) => e.stopPropagation()}
                      >
                        {param.options?.map((opt) => (
                          <option key={opt} value={opt}>{opt}</option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="number"
                        className="w-20 px-2 py-1.5 rounded text-sm text-center font-mono"
                        style={{
                          background: 'var(--bg-input)',
                          border: '1px solid var(--border)',
                          color: 'var(--text-primary)',
                        }}
                        value={Number(param.value)}
                        min={param.min}
                        max={param.max}
                        onChange={(e) => onParamChange(param.key, parseFloat(e.target.value) || param.min || 0)}
                        onClick={(e) => e.stopPropagation()}
                      />
                    )}
                    {param.min !== undefined && param.max !== undefined && (
                      <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        ({param.min}-{param.max})
                      </span>
                    )}
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
              {pack.triggers.length > 0 && (
                <div>
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Triggers</span>
                  <div className="space-y-1 mt-1">
                    {pack.triggers.map((trigger) => (
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
                          style={{
                            color: trigger.type === 'ENTRY' ? 'var(--accent)' : 'var(--orange)',
                            background: trigger.type === 'ENTRY' ? 'var(--accent-muted)' : 'var(--orange-muted)',
                            fontSize: '10px',
                          }}
                        >
                          {trigger.type}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {pack.triggers.length === 0 && (
                <div className="mt-1">
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    Condition-only pack (no triggers)
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ---------- Main Component ---------- */
export default function GeneralPacksV3() {
  const [packs, setPacks] = useState<GeneralPack[]>(mockPacks);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [activeFilter, setActiveFilter] = useState('All');
  const [showRenameModal, setShowRenameModal] = useState<string | null>(null);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState('');

  const enabledCount = packs.filter((p) => p.enabled).length;
  const filteredPacks = activeFilter === 'All'
    ? packs
    : packs.filter((p) => p.category === activeFilter);

  function toggleEnabled(id: string) {
    setPacks((prev) => prev.map((p) => p.id === id ? { ...p, enabled: !p.enabled } : p));
  }

  function updateParam(packId: string, paramKey: string, value: number | boolean | string) {
    setPacks((prev) =>
      prev.map((p) =>
        p.id === packId
          ? { ...p, params: p.params.map((pr) => pr.key === paramKey ? { ...pr, value } : pr) }
          : p,
      ),
    );
  }

  function duplicatePack(packId: string) {
    const source = packs.find((p) => p.id === packId);
    if (!source) return;
    const newPack: GeneralPack = {
      ...source,
      id: `${source.id}-copy-${Date.now()}`,
      name: `${source.name} Copy`,
      isDefault: false,
      params: source.params.map((pr) => ({ ...pr })),
      triggers: source.triggers.map((tr) => ({ ...tr })),
      outputs: [...source.outputs],
    };
    setPacks((prev) => [...prev, newPack]);
    setExpandedId(newPack.id);
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

  const renamePack_ = showRenameModal ? packs.find((p) => p.id === showRenameModal) : null;
  const deletePack_ = showDeleteConfirm ? packs.find((p) => p.id === showDeleteConfirm) : null;

  return (
    <div>
      <PageHeader
        title="General Packs"
        subtitle={`${packs.length} packs \u00b7 ${enabledCount} enabled`}
      />

      {/* Category filter pills */}
      <div className="flex gap-2 mb-5">
        {categoryFilters.map((filter) => {
          const isActive = activeFilter === filter;
          const catStyle = filter !== 'All' ? categoryColors[filter] : null;
          return (
            <button
              key={filter}
              onClick={() => setActiveFilter(filter)}
              className="px-3 py-1.5 rounded-full text-xs font-medium transition-colors"
              style={{
                background: isActive
                  ? (catStyle?.bg || 'var(--accent-muted)')
                  : 'transparent',
                color: isActive
                  ? (catStyle?.color || 'var(--accent)')
                  : 'var(--text-muted)',
                border: isActive
                  ? `1px solid ${catStyle?.color || 'var(--accent)'}`
                  : '1px solid var(--border)',
              }}
            >
              {filter}
              {filter !== 'All' && (
                <span className="ml-1" style={{ opacity: 0.7 }}>
                  {packs.filter((p) => p.category === filter).length}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Pack list */}
      <div className="space-y-2">
        {filteredPacks.length === 0 && (
          <div className="text-center py-12">
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No packs in this category.</p>
          </div>
        )}
        {filteredPacks.map((pack) => (
          <PackRow
            key={pack.id}
            pack={pack}
            expanded={expandedId === pack.id}
            onToggleExpand={() => setExpandedId(expandedId === pack.id ? null : pack.id)}
            onToggleEnabled={() => toggleEnabled(pack.id)}
            onParamChange={(key, val) => updateParam(pack.id, key, val)}
            onSaveParams={() => { /* toast: saved */ }}
            onRename={() => {
              setRenameValue(pack.name);
              setShowRenameModal(pack.id);
            }}
            onCopy={() => duplicatePack(pack.id)}
            onDelete={() => setShowDeleteConfirm(pack.id)}
          />
        ))}
      </div>

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
