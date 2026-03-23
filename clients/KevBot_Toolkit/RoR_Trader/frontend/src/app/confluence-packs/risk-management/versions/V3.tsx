'use client';

import { useState, useRef, useEffect } from 'react';
import PageHeader from '@/components/PageHeader';
import Modal from '@/components/Modal';

/* ---------- Types ---------- */
interface PackParam {
  key: string;
  label: string;
  type: 'int' | 'float' | 'select';
  value: number | string;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  help?: string;
  visibleWhen?: { field: string; value: string };
}

interface RiskPack {
  id: string;
  name: string;
  category: 'Volatility' | 'Fixed' | 'Structure' | 'Trailing';
  stopType: string;
  enabled: boolean;
  isDefault: boolean;
  params: PackParam[];
  stopSummary: string;
  targetSummary: string;
  contextualHelp: string[];
}

/* ---------- Mock Data ---------- */
const mockPacks: RiskPack[] = [
  {
    id: 'atr-default',
    name: 'ATR-Based (Default)',
    category: 'Volatility',
    stopType: 'ATR',
    enabled: true,
    isDefault: true,
    params: [
      { key: 'stop_atr_mult', label: 'Stop ATR Mult', type: 'float', value: 1.5, min: 0.5, max: 5.0, step: 0.1, help: 'ATR multiplier for stop distance from entry' },
      { key: 'target_atr_mult', label: 'Target ATR Mult', type: 'float', value: 3.0, min: 0.0, max: 10.0, step: 0.1, help: 'Set to 0 for no ATR target' },
    ],
    stopSummary: 'ATR x1.5',
    targetSummary: 'ATR x3.0',
    contextualHelp: [
      'ATR measures volatility over the last 14 bars by default.',
      'Higher multipliers give more room but increase risk per trade.',
      'ATR stops automatically adapt to market conditions.',
    ],
  },
  {
    id: 'atr-tight',
    name: 'ATR-Based (Tight)',
    category: 'Volatility',
    stopType: 'ATR',
    enabled: true,
    isDefault: false,
    params: [
      { key: 'stop_atr_mult', label: 'Stop ATR Mult', type: 'float', value: 1.0, min: 0.5, max: 5.0, step: 0.1, help: 'ATR multiplier for stop distance from entry' },
      { key: 'target_atr_mult', label: 'Target ATR Mult', type: 'float', value: 2.0, min: 0.0, max: 10.0, step: 0.1, help: 'Set to 0 for no ATR target' },
    ],
    stopSummary: 'ATR x1.0',
    targetSummary: 'ATR x2.0',
    contextualHelp: [
      'Tighter ATR stops for scalping or low-volatility setups.',
      '1:2 risk-reward provides a more aggressive profile.',
    ],
  },
  {
    id: 'fixed-default',
    name: 'Fixed Dollar',
    category: 'Fixed',
    stopType: 'Fixed $',
    enabled: true,
    isDefault: true,
    params: [
      { key: 'stop_amount', label: 'Stop ($)', type: 'float', value: 1.0, min: 0.01, max: 100.0, step: 0.01, help: 'Fixed dollar distance from entry to stop' },
      { key: 'target_amount', label: 'Target ($)', type: 'float', value: 2.0, min: 0.0, max: 100.0, step: 0.01, help: 'Fixed dollar distance to target (0 = no target)' },
    ],
    stopSummary: '$1.00',
    targetSummary: '$2.00',
    contextualHelp: [
      'Fixed dollar stops use a constant distance regardless of volatility.',
      'Best for instruments with stable price ranges (e.g., SPY 1-min scalps).',
      'Warning: may be too tight in volatile conditions.',
    ],
  },
  {
    id: 'pct-default',
    name: 'Percentage',
    category: 'Fixed',
    stopType: 'Pct',
    enabled: false,
    isDefault: true,
    params: [
      { key: 'stop_pct', label: 'Stop (%)', type: 'float', value: 0.5, min: 0.01, max: 10.0, step: 0.01, help: 'Percentage distance from entry to stop' },
      { key: 'target_pct', label: 'Target (%)', type: 'float', value: 1.0, min: 0.0, max: 20.0, step: 0.01, help: 'Percentage distance to target (0 = no target)' },
    ],
    stopSummary: '0.5%',
    targetSummary: '1.0%',
    contextualHelp: [
      'Percentage stops scale with instrument price.',
      'Works well across different price levels without manual adjustment.',
    ],
  },
  {
    id: 'swing-default',
    name: 'Swing Stop',
    category: 'Structure',
    stopType: 'Swing',
    enabled: true,
    isDefault: true,
    params: [
      { key: 'lookback', label: 'Lookback', type: 'int', value: 5, min: 2, max: 50, help: 'Bars to search for swing high/low' },
      { key: 'padding', label: 'Padding ($)', type: 'float', value: 0.05, min: 0.0, max: 10.0, step: 0.01, help: 'Buffer beyond the swing point' },
      { key: 'rr_ratio', label: 'R:R Ratio', type: 'float', value: 2.0, min: 0.0, max: 10.0, step: 0.1, help: 'Target as a multiple of stop distance (0 = no target)' },
    ],
    stopSummary: 'Swing (5 bar, $0.05 pad)',
    targetSummary: '2.0R',
    contextualHelp: [
      'Places the stop at the recent swing low (longs) or swing high (shorts).',
      'Respects market structure at logical support/resistance levels.',
      'Stop-validity guard: entries rejected when swing stop >= entry price.',
    ],
  },
  {
    id: 'atr-trailing',
    name: 'ATR Trailing',
    category: 'Trailing',
    stopType: 'Trail',
    enabled: false,
    isDefault: true,
    params: [
      { key: 'stop_atr_mult', label: 'Initial Stop ATR', type: 'float', value: 1.5, min: 0.5, max: 5.0, step: 0.1, help: 'Initial stop distance from entry' },
      { key: 'trail_atr_mult', label: 'Trail ATR', type: 'float', value: 1.0, min: 0.3, max: 5.0, step: 0.1, help: 'Trailing stop ATR distance (typically tighter)' },
      { key: 'trail_activation_r', label: 'Trail Activation (R)', type: 'float', value: 0.5, min: 0.0, max: 5.0, step: 0.1, help: 'Profit in R-multiples before trailing begins' },
      { key: 'target_atr_mult', label: 'Target ATR (0=none)', type: 'float', value: 0.0, min: 0.0, max: 10.0, step: 0.1, help: 'Optional fixed ATR target; 0 = let the trail ride' },
    ],
    stopSummary: 'ATR x1.5 -> Trail x1.0',
    targetSummary: 'None (trail)',
    contextualHelp: [
      'Starts with a standard ATR stop, then transitions to a trailing stop.',
      'The trailing stop ratchets in your favor -- it never moves backward.',
      'Set target to 0 for a pure trend-following approach.',
    ],
  },
  {
    id: 'breakeven-default',
    name: 'Breakeven Stop',
    category: 'Trailing',
    stopType: 'BE',
    enabled: false,
    isDefault: true,
    params: [
      { key: 'stop_atr_mult', label: 'Initial Stop ATR', type: 'float', value: 1.5, min: 0.5, max: 5.0, step: 0.1, help: 'Initial stop distance from entry' },
      { key: 'be_activation_r', label: 'BE Activation (R)', type: 'float', value: 1.0, min: 0.1, max: 5.0, step: 0.1, help: 'Profit in R-multiples before stop moves to breakeven' },
      { key: 'be_offset', label: 'BE Offset ($)', type: 'float', value: 0.0, min: 0.0, max: 2.0, step: 0.01, help: 'Small offset above entry for commissions' },
      { key: 'target_atr_mult', label: 'Target ATR (0=none)', type: 'float', value: 3.0, min: 0.0, max: 10.0, step: 0.1, help: 'ATR target multiplier; 0 = exit via signal only' },
    ],
    stopSummary: 'ATR x1.5 -> BE at 1.0R',
    targetSummary: 'ATR x3.0',
    contextualHelp: [
      'Standard ATR stop that jumps to breakeven once profit threshold is reached.',
      'Once at breakeven, worst outcome is a scratch trade.',
      'BE offset covers commissions and slippage.',
    ],
  },
];

const categoryFilters = ['All', 'Volatility', 'Fixed', 'Structure', 'Trailing'];

const categoryColors: Record<string, { color: string; bg: string }> = {
  'Volatility': { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  'Fixed': { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  'Structure': { color: 'var(--green)', bg: 'var(--green-muted)' },
  'Trailing': { color: 'var(--accent)', bg: 'var(--accent-muted)' },
};

const stopTypeColors: Record<string, { color: string; bg: string }> = {
  'ATR': { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  'Fixed $': { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  'Pct': { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  'Swing': { color: 'var(--green)', bg: 'var(--green-muted)' },
  'Trail': { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  'BE': { color: 'var(--accent)', bg: 'var(--accent-muted)' },
};

/* ---------- Helpers ---------- */
function keyParamDisplay(pack: RiskPack): string {
  return `Stop: ${pack.stopSummary} | Target: ${pack.targetSummary}`;
}

/* ---------- Overflow Menu ---------- */
function OverflowMenu({
  pack,
  onRename,
  onCopy,
  onDelete,
}: {
  pack: RiskPack;
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
  pack: RiskPack;
  expanded: boolean;
  onToggleExpand: () => void;
  onToggleEnabled: () => void;
  onParamChange: (key: string, value: number | string) => void;
  onSaveParams: () => void;
  onRename: () => void;
  onCopy: () => void;
  onDelete: () => void;
}) {
  const stStyle = stopTypeColors[pack.stopType] || { color: 'var(--text-muted)', bg: 'var(--bg-input)' };

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

        {/* Stop type badge */}
        <span
          className="text-xs px-2 py-0.5 rounded whitespace-nowrap flex-shrink-0"
          style={{ color: stStyle.color, background: stStyle.bg }}
        >
          {pack.stopType}
        </span>

        {/* Key param display */}
        <span
          className="text-xs whitespace-nowrap flex-shrink-0 hidden sm:inline"
          style={{ color: 'var(--text-muted)' }}
        >
          {keyParamDisplay(pack)}
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
                  <div key={param.key}>
                    <div className="flex items-center gap-3">
                      <label className="text-sm w-36 flex-shrink-0" style={{ color: 'var(--text-secondary)' }}>
                        {param.label}
                      </label>
                      {param.type === 'select' ? (
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
                          step={param.step || (param.type === 'float' ? 0.1 : 1)}
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
                    {param.help && (
                      <p className="text-xs mt-1 ml-[156px]" style={{ color: 'var(--text-muted)' }}>
                        {param.help}
                      </p>
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

            {/* Stop Placement Explanation panel */}
            <div
              className="rounded-lg border p-4"
              style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border)' }}
            >
              <h4 className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-muted)' }}>
                Stop Placement
              </h4>

              {/* Stop/Target summary */}
              <div className="grid grid-cols-2 gap-3 mb-4">
                <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Stop</p>
                  <p className="text-sm font-semibold" style={{ color: 'var(--red)' }}>{pack.stopSummary}</p>
                </div>
                <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Target</p>
                  <p className="text-sm font-semibold" style={{ color: 'var(--green)' }}>{pack.targetSummary}</p>
                </div>
              </div>

              {/* Contextual help */}
              <div className="space-y-2">
                {pack.contextualHelp.map((help, i) => (
                  <div key={i} className="flex items-start gap-2">
                    <span className="text-xs mt-0.5 flex-shrink-0" style={{ color: 'var(--accent)' }}>{i + 1}.</span>
                    <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>{help}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ---------- Main Component ---------- */
export default function RiskManagementV3() {
  const [packs, setPacks] = useState<RiskPack[]>(mockPacks);
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

  function updateParam(packId: string, paramKey: string, value: number | string) {
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
    const newPack: RiskPack = {
      ...source,
      id: `${source.id}-copy-${Date.now()}`,
      name: `${source.name} Copy`,
      isDefault: false,
      params: source.params.map((pr) => ({ ...pr })),
      contextualHelp: [...source.contextualHelp],
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
        title="Risk Management"
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
