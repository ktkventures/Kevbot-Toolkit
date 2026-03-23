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

interface TfPack {
  id: string;
  name: string;
  version: string;
  category: string;
  enabled: boolean;
  isDefault: boolean;
  params: PackParam[];
  outputs: string[];
  triggers: PackTrigger[];
}

/* ---------- Mock Data ---------- */
const mockPacks: TfPack[] = [
  {
    id: 'ema-stack-default',
    name: 'EMA Stack',
    version: 'Default',
    category: 'Moving Averages',
    enabled: true,
    isDefault: true,
    params: [
      { key: 'short_period', label: 'Short Period', value: 9, min: 1, max: 200 },
      { key: 'mid_period', label: 'Mid Period', value: 21, min: 1, max: 200 },
      { key: 'long_period', label: 'Long Period', value: 200, min: 1, max: 500 },
    ],
    outputs: ['SML', 'SLM', 'MSL', 'MLS', 'LSM', 'LMS'],
    triggers: [
      { name: 'Short > Mid Cross', direction: 'LONG', exec: '[C]' },
      { name: 'Short < Mid Cross', direction: 'SHORT', exec: '[C]' },
      { name: 'Mid > Long Cross', direction: 'LONG', exec: '[C]' },
      { name: 'Mid < Long Cross', direction: 'SHORT', exec: '[C]' },
    ],
  },
  {
    id: 'macd-line-default',
    name: 'MACD Line',
    version: 'Default',
    category: 'Momentum',
    enabled: true,
    isDefault: true,
    params: [
      { key: 'fast_period', label: 'Fast Period', value: 12, min: 1, max: 100 },
      { key: 'slow_period', label: 'Slow Period', value: 26, min: 1, max: 200 },
      { key: 'signal_period', label: 'Signal Period', value: 9, min: 1, max: 50 },
    ],
    outputs: ['M>S+', 'M>S-', 'M<S-', 'M<S+'],
    triggers: [
      { name: 'MACD Bull Cross', direction: 'LONG', exec: '[C]' },
      { name: 'MACD Bear Cross', direction: 'SHORT', exec: '[C]' },
      { name: 'Zero Line Cross Up', direction: 'LONG', exec: '[C]' },
      { name: 'Zero Line Cross Down', direction: 'SHORT', exec: '[C]' },
    ],
  },
  {
    id: 'macd-hist-default',
    name: 'MACD Histogram',
    version: 'Default',
    category: 'Momentum',
    enabled: false,
    isDefault: true,
    params: [
      { key: 'fast_period', label: 'Fast Period', value: 12, min: 1, max: 100 },
      { key: 'slow_period', label: 'Slow Period', value: 26, min: 1, max: 200 },
      { key: 'signal_period', label: 'Signal Period', value: 9, min: 1, max: 50 },
    ],
    outputs: ['H+up', 'H+dn', 'H-dn', 'H-up'],
    triggers: [
      { name: 'Histogram Flip Positive', direction: 'LONG', exec: '[C]' },
      { name: 'Histogram Flip Negative', direction: 'SHORT', exec: '[C]' },
      { name: 'Histogram Peak', direction: 'SHORT', exec: '[C]' },
      { name: 'Histogram Trough', direction: 'LONG', exec: '[C]' },
    ],
  },
  {
    id: 'vwap-default',
    name: 'VWAP',
    version: 'Default',
    category: 'Volume',
    enabled: true,
    isDefault: true,
    params: [
      { key: 'sd1', label: 'Band 1 StdDev', value: 1.0, min: 0.1, max: 5.0 },
      { key: 'sd2', label: 'Band 2 StdDev', value: 2.0, min: 0.1, max: 5.0 },
    ],
    outputs: ['>+2\u03c3', '>+1\u03c3', '>V', '@V', '<V', '<-1\u03c3', '<-2\u03c3'],
    triggers: [
      { name: 'VWAP Reclaim', direction: 'LONG', exec: '[C]' },
      { name: 'VWAP Rejection', direction: 'SHORT', exec: '[C]' },
      { name: 'Upper Band Break', direction: 'LONG', exec: '[C]' },
      { name: 'Lower Band Break', direction: 'SHORT', exec: '[C]' },
    ],
  },
  {
    id: 'rvol-default',
    name: 'Relative Volume',
    version: 'Default',
    category: 'Volume',
    enabled: false,
    isDefault: true,
    params: [
      { key: 'sma_period', label: 'SMA Period', value: 20, min: 5, max: 200 },
      { key: 'high_threshold', label: 'High Threshold', value: 1.5, min: 0.5, max: 10.0 },
      { key: 'extreme_threshold', label: 'Extreme Threshold', value: 3.0, min: 1.0, max: 20.0 },
    ],
    outputs: ['EXTREME', 'HIGH', 'NORMAL', 'LOW', 'MINIMAL'],
    triggers: [
      { name: 'Volume Spike', direction: 'LONG', exec: '[C]' },
      { name: 'Volume Extreme', direction: 'LONG', exec: '[C]' },
      { name: 'Volume Dry Up', direction: 'SHORT', exec: '[C]' },
    ],
  },
  {
    id: 'ema-stack-scalping',
    name: 'EMA Stack',
    version: 'Scalping',
    category: 'Moving Averages',
    enabled: true,
    isDefault: false,
    params: [
      { key: 'short_period', label: 'Short Period', value: 5, min: 1, max: 200 },
      { key: 'mid_period', label: 'Mid Period', value: 13, min: 1, max: 200 },
      { key: 'long_period', label: 'Long Period', value: 50, min: 1, max: 500 },
    ],
    outputs: ['SML', 'SLM', 'MSL', 'MLS', 'LSM', 'LMS'],
    triggers: [
      { name: 'Short > Mid Cross', direction: 'LONG', exec: '[C]' },
      { name: 'Short < Mid Cross', direction: 'SHORT', exec: '[C]' },
      { name: 'Mid > Long Cross', direction: 'LONG', exec: '[C]' },
      { name: 'Mid < Long Cross', direction: 'SHORT', exec: '[C]' },
    ],
  },
];

const categoryFilters = ['All', 'Moving Averages', 'Momentum', 'Volume'];

const categoryColors: Record<string, { color: string; bg: string }> = {
  'Moving Averages': { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  'Momentum': { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  'Volume': { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  'Volatility': { color: 'var(--red)', bg: 'var(--red-muted)' },
  'Trend': { color: 'var(--green)', bg: 'var(--green-muted)' },
};

const directionColors: Record<string, { color: string; bg: string }> = {
  'LONG': { color: 'var(--green)', bg: 'var(--green-muted)' },
  'SHORT': { color: 'var(--red)', bg: 'var(--red-muted)' },
};

const templateCategories = ['Moving Averages', 'Momentum', 'Volume', 'Volatility', 'Trend'];

/* ---------- Helpers ---------- */
function paramSummary(params: PackParam[]): string {
  return params.map((p) => {
    const short = p.label.replace('Period', '').replace('Threshold', '').replace('StdDev', 'SD').trim();
    const initial = short.charAt(0);
    return `${initial}:${p.value}`;
  }).join(' ');
}

/* ---------- Overflow Menu ---------- */
function OverflowMenu({
  pack,
  onRename,
  onCopy,
  onPlotSettings,
  onDelete,
}: {
  pack: TfPack;
  onRename: () => void;
  onCopy: () => void;
  onPlotSettings: () => void;
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
          <button
            onClick={(e) => { e.stopPropagation(); setOpen(false); onPlotSettings(); }}
            className="w-full text-left px-3 py-2 text-sm transition-colors"
            style={{ color: 'var(--text-primary)' }}
            onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-card)')}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
          >
            Plot Settings
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
  onResetParams,
  onRename,
  onCopy,
  onPlotSettings,
  onDelete,
}: {
  pack: TfPack;
  expanded: boolean;
  onToggleExpand: () => void;
  onToggleEnabled: () => void;
  onParamChange: (key: string, value: number) => void;
  onSaveParams: () => void;
  onResetParams: () => void;
  onRename: () => void;
  onCopy: () => void;
  onPlotSettings: () => void;
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

        {/* Name + version */}
        <div className="flex items-center gap-2 min-w-0">
          <span className="font-semibold text-sm whitespace-nowrap" style={{ color: 'var(--text-primary)' }}>
            {pack.name}
          </span>
          <span
            className="text-xs px-1.5 py-0.5 rounded whitespace-nowrap"
            style={{
              color: pack.isDefault ? 'var(--text-muted)' : 'var(--text-secondary)',
              background: 'var(--bg-input)',
            }}
          >
            {pack.version}
          </span>
        </div>

        {/* Category badge */}
        <span
          className="text-xs px-2 py-0.5 rounded whitespace-nowrap flex-shrink-0"
          style={{ color: catStyle.color, background: catStyle.bg }}
        >
          {pack.category}
        </span>

        {/* Param summary */}
        <span
          className="text-xs font-mono whitespace-nowrap flex-shrink-0 hidden sm:inline"
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
          onPlotSettings={onPlotSettings}
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
                      step={param.max <= 5 ? 0.1 : 1}
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
                <button
                  className="px-3 py-1.5 rounded text-xs"
                  style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                  onClick={(e) => { e.stopPropagation(); onResetParams(); }}
                >
                  Reset
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
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ---------- Main Component ---------- */
export default function TfConfluenceV3() {
  const [packs, setPacks] = useState<TfPack[]>(mockPacks);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [activeFilter, setActiveFilter] = useState('All');
  const [showNewModal, setShowNewModal] = useState(false);
  const [showPlotModal, setShowPlotModal] = useState<string | null>(null);
  const [showRenameModal, setShowRenameModal] = useState<string | null>(null);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState<string | null>(null);
  const [newTemplate, setNewTemplate] = useState(templateCategories[0]);
  const [newName, setNewName] = useState('');
  const [renameValue, setRenameValue] = useState('');

  const enabledCount = packs.filter((p) => p.enabled).length;
  const filteredPacks = activeFilter === 'All'
    ? packs
    : packs.filter((p) => p.category === activeFilter);

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

  function resetParams(packId: string) {
    const original = mockPacks.find((p) => p.id === packId);
    if (!original) return;
    setPacks((prev) =>
      prev.map((p) =>
        p.id === packId ? { ...p, params: original.params.map((pr) => ({ ...pr })) } : p,
      ),
    );
  }

  function duplicatePack(packId: string) {
    const source = packs.find((p) => p.id === packId);
    if (!source) return;
    const newPack: TfPack = {
      ...source,
      id: `${source.id}-copy-${Date.now()}`,
      version: `${source.version} Copy`,
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

  function renamePack(packId: string, newVersion: string) {
    setPacks((prev) =>
      prev.map((p) => p.id === packId ? { ...p, version: newVersion } : p),
    );
    setShowRenameModal(null);
  }

  function createPack() {
    const basePack = packs.find((p) => p.category === newTemplate && p.isDefault);
    const id = `${newTemplate.toLowerCase().replace(/\s+/g, '-')}-${(newName || 'custom').toLowerCase().replace(/\s+/g, '-')}-${Date.now()}`;
    const newPack: TfPack = basePack
      ? {
        ...basePack,
        id,
        version: newName || 'Custom',
        isDefault: false,
        enabled: true,
        params: basePack.params.map((pr) => ({ ...pr })),
        triggers: basePack.triggers.map((tr) => ({ ...tr })),
        outputs: [...basePack.outputs],
      }
      : {
        id,
        name: newTemplate,
        version: newName || 'Custom',
        category: newTemplate,
        enabled: true,
        isDefault: false,
        params: [],
        outputs: [],
        triggers: [],
      };
    setPacks((prev) => [...prev, newPack]);
    setExpandedId(id);
    setShowNewModal(false);
    setNewName('');
  }

  const plotPack = showPlotModal ? packs.find((p) => p.id === showPlotModal) : null;
  const renamePack_ = showRenameModal ? packs.find((p) => p.id === showRenameModal) : null;
  const deletePack_ = showDeleteConfirm ? packs.find((p) => p.id === showDeleteConfirm) : null;

  return (
    <div>
      <PageHeader
        title="TF Confluence Packs"
        subtitle={`${packs.length} packs \u00b7 ${enabledCount} enabled`}
        actions={
          <button
            onClick={() => setShowNewModal(true)}
            className="px-4 py-2 rounded-lg text-sm font-medium transition-colors"
            style={{ background: 'var(--accent)', color: 'white' }}
          >
            + New Pack
          </button>
        }
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
            onResetParams={() => resetParams(pack.id)}
            onRename={() => {
              setRenameValue(pack.version);
              setShowRenameModal(pack.id);
            }}
            onCopy={() => duplicatePack(pack.id)}
            onPlotSettings={() => setShowPlotModal(pack.id)}
            onDelete={() => setShowDeleteConfirm(pack.id)}
          />
        ))}
      </div>

      {/* New Pack Modal */}
      <Modal title="Create New Pack" isOpen={showNewModal} onClose={() => setShowNewModal(false)} width="440px">
        <div className="space-y-4">
          <div>
            <label className="text-sm mb-1 block" style={{ color: 'var(--text-secondary)' }}>Template</label>
            <select
              className="w-full px-3 py-2 rounded-lg text-sm"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
              value={newTemplate}
              onChange={(e) => setNewTemplate(e.target.value)}
            >
              {templateCategories.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-sm mb-1 block" style={{ color: 'var(--text-secondary)' }}>Version Name</label>
            <input
              className="w-full px-3 py-2 rounded-lg text-sm"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
              placeholder="e.g. Scalping, Swing, Conservative..."
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
            />
          </div>
          <div>
            <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Pack ID Preview</label>
            <p className="text-sm font-mono px-3 py-2 rounded-lg" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
              {newTemplate.toLowerCase().replace(/\s+/g, '-')}-{newName ? newName.toLowerCase().replace(/\s+/g, '-') : 'custom'}
            </p>
          </div>
          <div className="flex gap-3 pt-2">
            <button
              className="px-4 py-2 rounded-lg text-sm font-medium flex-1"
              style={{ background: 'var(--accent)', color: 'white' }}
              onClick={createPack}
            >
              Create Pack
            </button>
            <button
              className="px-4 py-2 rounded-lg text-sm flex-1"
              style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
              onClick={() => setShowNewModal(false)}
            >
              Cancel
            </button>
          </div>
        </div>
      </Modal>

      {/* Plot Settings Modal */}
      <Modal
        title={plotPack ? `Plot Settings - ${plotPack.name} (${plotPack.version})` : 'Plot Settings'}
        isOpen={!!showPlotModal}
        onClose={() => setShowPlotModal(null)}
        width="480px"
      >
        <div className="space-y-4">
          {[
            { label: 'Primary Line Color', value: '#3b82f6' },
            { label: 'Secondary Line Color', value: '#f59e0b' },
            { label: 'Fill Color', value: '#3b82f620' },
          ].map((setting) => (
            <div key={setting.label} className="flex items-center gap-3">
              <label className="text-sm w-44 flex-shrink-0" style={{ color: 'var(--text-secondary)' }}>
                {setting.label}
              </label>
              <div className="flex items-center gap-2">
                <div
                  className="w-7 h-7 rounded border flex-shrink-0"
                  style={{ background: setting.value, borderColor: 'var(--border)' }}
                />
                <input
                  className="px-2 py-1.5 rounded text-sm font-mono w-24"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  defaultValue={setting.value}
                />
              </div>
            </div>
          ))}
          <div className="flex items-center gap-3">
            <label className="text-sm w-44 flex-shrink-0" style={{ color: 'var(--text-secondary)' }}>Line Width</label>
            <input type="range" min="1" max="5" defaultValue="2" className="flex-1" />
            <span className="text-sm w-8 text-center" style={{ color: 'var(--text-muted)' }}>2px</span>
          </div>
          <div className="flex items-center gap-3">
            <label className="text-sm w-44 flex-shrink-0" style={{ color: 'var(--text-secondary)' }}>Fill Opacity</label>
            <input type="range" min="0" max="100" defaultValue="20" className="flex-1" />
            <span className="text-sm w-8 text-center" style={{ color: 'var(--text-muted)' }}>20%</span>
          </div>
          <div className="flex gap-3 pt-2">
            <button
              className="px-4 py-2 rounded-lg text-sm font-medium flex-1"
              style={{ background: 'var(--accent)', color: 'white' }}
              onClick={() => setShowPlotModal(null)}
            >
              Save
            </button>
            <button
              className="px-4 py-2 rounded-lg text-sm flex-1"
              style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
              onClick={() => setShowPlotModal(null)}
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
            <label className="text-sm mb-1 block" style={{ color: 'var(--text-secondary)' }}>Version Name</label>
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
            Are you sure you want to delete <strong style={{ color: 'var(--text-primary)' }}>{deletePack_?.name} ({deletePack_?.version})</strong>?
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
