'use client';

import { useState, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import MetricCard from '@/components/MetricCard';
import Modal from '@/components/Modal';
import TabBar from '@/components/TabBar';

/* ========================================================================= */
/* TYPES                                                                      */
/* ========================================================================= */

interface Rule {
  id: string;
  name: string;
  type: string;
  value: number;
  threshold: number | null;
  currentValue: number; // mock current value for progress bars
}

interface ComplianceResult {
  rule: string;
  status: 'pass' | 'fail' | 'warning';
  current: string;
  limit: string;
  pct: number;
}

interface RequirementSet {
  id: string;
  name: string;
  isBuiltIn: boolean;
  rules: Rule[];
  usedBy: string[]; // portfolio names
}

interface PropFirmPreset {
  id: string;
  name: string;
  firm: string;
  accountSize: string;
  rules: Omit<Rule, 'currentValue'>[];
}

/* ========================================================================= */
/* CONSTANTS                                                                  */
/* ========================================================================= */

const RULE_TYPE_OPTIONS = [
  { value: 'max_daily_loss', label: 'Max Daily Loss ($)' },
  { value: 'max_total_drawdown', label: 'Max Total Drawdown ($)' },
  { value: 'min_trading_days', label: 'Min Trading Days' },
  { value: 'max_daily_trades', label: 'Max Daily Trades' },
  { value: 'min_profitable_days', label: 'Min Profitable Days (%)' },
  { value: 'max_position_size', label: 'Max Position Size' },
  { value: 'max_daily_risk_pct', label: 'Max Daily Risk (%)' },
  { value: 'profit_target', label: 'Profit Target ($)' },
];

const RULE_TYPE_LABELS: Record<string, string> = {};
RULE_TYPE_OPTIONS.forEach((opt) => {
  RULE_TYPE_LABELS[opt.value] = opt.label;
});

const RULE_DESCRIPTIONS: Record<string, string> = {
  max_daily_loss: 'Maximum dollar loss allowed in a single trading day before account lockout.',
  max_total_drawdown: 'Maximum cumulative loss from account peak equity before account termination.',
  min_trading_days: 'Minimum number of days with at least one trade to qualify for evaluation.',
  max_daily_trades: 'Maximum number of round-trip trades allowed per day.',
  min_profitable_days: 'Percentage of trading days that must be net profitable.',
  max_position_size: 'Maximum number of contracts/shares per position.',
  max_daily_risk_pct: 'Maximum percentage of account balance that can be risked in a single day.',
  profit_target: 'Dollar profit target that must be reached to pass evaluation.',
};

/* ========================================================================= */
/* MOCK DATA                                                                  */
/* ========================================================================= */

const initialRequirementSets: RequirementSet[] = [
  {
    id: 'ttp-50k',
    name: 'TTP 50k Evaluation',
    isBuiltIn: true,
    usedBy: ['My Portfolio', 'Scalping Portfolio'],
    rules: [
      { id: 'r1', name: 'Max Daily Loss', type: 'max_daily_loss', value: 1000, threshold: null, currentValue: 342 },
      { id: 'r2', name: 'Max Total Drawdown', type: 'max_total_drawdown', value: 2500, threshold: null, currentValue: 875 },
      { id: 'r3', name: 'Min Trading Days', type: 'min_trading_days', value: 5, threshold: null, currentValue: 3 },
      { id: 'r4', name: 'Max Daily Trades', type: 'max_daily_trades', value: 15, threshold: null, currentValue: 7 },
      { id: 'r5', name: 'Profit Target', type: 'profit_target', value: 3000, threshold: null, currentValue: 1850 },
    ],
  },
  {
    id: 'ftmo-100k',
    name: 'FTMO 100k Challenge',
    isBuiltIn: true,
    usedBy: ['Swing Portfolio'],
    rules: [
      { id: 'r1', name: 'Max Daily Loss', type: 'max_daily_loss', value: 5000, threshold: null, currentValue: 1200 },
      { id: 'r2', name: 'Max Total Drawdown', type: 'max_total_drawdown', value: 10000, threshold: null, currentValue: 3400 },
      { id: 'r3', name: 'Min Trading Days', type: 'min_trading_days', value: 4, threshold: null, currentValue: 4 },
      { id: 'r4', name: 'Min Profitable Days', type: 'min_profitable_days', value: 60, threshold: 50, currentValue: 72 },
      { id: 'r5', name: 'Profit Target', type: 'profit_target', value: 10000, threshold: null, currentValue: 6200 },
      { id: 'r6', name: 'Max Position Size', type: 'max_position_size', value: 10, threshold: null, currentValue: 5 },
    ],
  },
  {
    id: 'topstep-150k',
    name: 'Topstep 150k Combine',
    isBuiltIn: true,
    usedBy: [],
    rules: [
      { id: 'r1', name: 'Max Daily Loss', type: 'max_daily_loss', value: 4500, threshold: null, currentValue: 0 },
      { id: 'r2', name: 'Max Total Drawdown', type: 'max_total_drawdown', value: 5000, threshold: null, currentValue: 0 },
      { id: 'r3', name: 'Min Trading Days', type: 'min_trading_days', value: 5, threshold: null, currentValue: 0 },
      { id: 'r4', name: 'Profit Target', type: 'profit_target', value: 9000, threshold: null, currentValue: 0 },
    ],
  },
  {
    id: 'custom-1',
    name: 'My Custom Rules',
    isBuiltIn: false,
    usedBy: ['Personal Account'],
    rules: [
      { id: 'r1', name: 'Max Daily Loss', type: 'max_daily_loss', value: 500, threshold: null, currentValue: 125 },
      { id: 'r2', name: 'Max Daily Risk %', type: 'max_daily_risk_pct', value: 2.0, threshold: null, currentValue: 0.8 },
    ],
  },
];

const PROP_FIRM_PRESETS: PropFirmPreset[] = [
  {
    id: 'ttp-25k', name: 'TTP 25k Evaluation', firm: 'Trade The Pool', accountSize: '$25,000',
    rules: [
      { id: 'r1', name: 'Max Daily Loss', type: 'max_daily_loss', value: 500, threshold: null },
      { id: 'r2', name: 'Max Total Drawdown', type: 'max_total_drawdown', value: 1250, threshold: null },
      { id: 'r3', name: 'Min Trading Days', type: 'min_trading_days', value: 5, threshold: null },
      { id: 'r4', name: 'Max Daily Trades', type: 'max_daily_trades', value: 15, threshold: null },
      { id: 'r5', name: 'Profit Target', type: 'profit_target', value: 1500, threshold: null },
    ],
  },
  {
    id: 'ttp-100k', name: 'TTP 100k Evaluation', firm: 'Trade The Pool', accountSize: '$100,000',
    rules: [
      { id: 'r1', name: 'Max Daily Loss', type: 'max_daily_loss', value: 2000, threshold: null },
      { id: 'r2', name: 'Max Total Drawdown', type: 'max_total_drawdown', value: 5000, threshold: null },
      { id: 'r3', name: 'Min Trading Days', type: 'min_trading_days', value: 5, threshold: null },
      { id: 'r4', name: 'Max Daily Trades', type: 'max_daily_trades', value: 15, threshold: null },
      { id: 'r5', name: 'Profit Target', type: 'profit_target', value: 6000, threshold: null },
    ],
  },
  {
    id: 'ftmo-50k', name: 'FTMO 50k Challenge', firm: 'FTMO', accountSize: '$50,000',
    rules: [
      { id: 'r1', name: 'Max Daily Loss', type: 'max_daily_loss', value: 2500, threshold: null },
      { id: 'r2', name: 'Max Total Drawdown', type: 'max_total_drawdown', value: 5000, threshold: null },
      { id: 'r3', name: 'Min Trading Days', type: 'min_trading_days', value: 4, threshold: null },
      { id: 'r4', name: 'Min Profitable Days', type: 'min_profitable_days', value: 60, threshold: 50 },
      { id: 'r5', name: 'Profit Target', type: 'profit_target', value: 5000, threshold: null },
    ],
  },
  {
    id: 'ftmo-200k', name: 'FTMO 200k Challenge', firm: 'FTMO', accountSize: '$200,000',
    rules: [
      { id: 'r1', name: 'Max Daily Loss', type: 'max_daily_loss', value: 10000, threshold: null },
      { id: 'r2', name: 'Max Total Drawdown', type: 'max_total_drawdown', value: 20000, threshold: null },
      { id: 'r3', name: 'Min Trading Days', type: 'min_trading_days', value: 4, threshold: null },
      { id: 'r4', name: 'Min Profitable Days', type: 'min_profitable_days', value: 60, threshold: 50 },
      { id: 'r5', name: 'Profit Target', type: 'profit_target', value: 20000, threshold: null },
    ],
  },
  {
    id: 'topstep-50k', name: 'Topstep 50k Combine', firm: 'Topstep', accountSize: '$50,000',
    rules: [
      { id: 'r1', name: 'Max Daily Loss', type: 'max_daily_loss', value: 1000, threshold: null },
      { id: 'r2', name: 'Max Total Drawdown', type: 'max_total_drawdown', value: 2000, threshold: null },
      { id: 'r3', name: 'Min Trading Days', type: 'min_trading_days', value: 5, threshold: null },
      { id: 'r4', name: 'Profit Target', type: 'profit_target', value: 3000, threshold: null },
    ],
  },
  {
    id: 'topstep-100k', name: 'Topstep 100k Combine', firm: 'Topstep', accountSize: '$100,000',
    rules: [
      { id: 'r1', name: 'Max Daily Loss', type: 'max_daily_loss', value: 3000, threshold: null },
      { id: 'r2', name: 'Max Total Drawdown', type: 'max_total_drawdown', value: 3500, threshold: null },
      { id: 'r3', name: 'Min Trading Days', type: 'min_trading_days', value: 5, threshold: null },
      { id: 'r4', name: 'Profit Target', type: 'profit_target', value: 6000, threshold: null },
    ],
  },
];

/* ========================================================================= */
/* HELPERS                                                                    */
/* ========================================================================= */

function formatRuleValue(rule: { type: string; value: number }): string {
  const dollarTypes = ['max_daily_loss', 'max_total_drawdown', 'profit_target'];
  const pctTypes = ['min_profitable_days', 'max_daily_risk_pct'];
  if (dollarTypes.includes(rule.type)) return `$${rule.value.toLocaleString()}`;
  if (pctTypes.includes(rule.type)) return `${rule.value}%`;
  return rule.value.toString();
}

function formatCurrentValue(type: string, value: number): string {
  const dollarTypes = ['max_daily_loss', 'max_total_drawdown', 'profit_target'];
  const pctTypes = ['min_profitable_days', 'max_daily_risk_pct'];
  if (dollarTypes.includes(type)) return `$${value.toLocaleString()}`;
  if (pctTypes.includes(type)) return `${value}%`;
  return value.toString();
}

function generateId(): string {
  return 'id-' + Math.random().toString(36).substring(2, 9);
}

function getProgressPct(rule: Rule): number {
  if (rule.value === 0) return 0;
  // For "min" types, progress toward target
  if (rule.type === 'min_trading_days' || rule.type === 'min_profitable_days' || rule.type === 'profit_target') {
    return Math.min(100, (rule.currentValue / rule.value) * 100);
  }
  // For "max" types, usage against limit
  return Math.min(100, (rule.currentValue / rule.value) * 100);
}

function getProgressColor(rule: Rule): string {
  const pct = getProgressPct(rule);
  const isMaxType = rule.type.startsWith('max_');
  if (isMaxType) {
    // For max types: high usage = danger
    if (pct >= 80) return 'var(--red)';
    if (pct >= 60) return 'var(--yellow, #e5a813)';
    return 'var(--green)';
  } else {
    // For min/target types: progress toward goal
    if (pct >= 100) return 'var(--green)';
    if (pct >= 50) return 'var(--accent)';
    return 'var(--text-muted)';
  }
}

function getComplianceStatus(rule: Rule): 'pass' | 'fail' | 'warning' {
  const pct = getProgressPct(rule);
  const isMaxType = rule.type.startsWith('max_');
  if (isMaxType) {
    if (pct >= 100) return 'fail';
    if (pct >= 80) return 'warning';
    return 'pass';
  } else {
    if (pct >= 100) return 'pass';
    if (pct >= 50) return 'warning';
    return 'fail';
  }
}

function complianceStatusColor(status: 'pass' | 'fail' | 'warning'): string {
  if (status === 'pass') return 'var(--green)';
  if (status === 'fail') return 'var(--red)';
  return 'var(--yellow, #e5a813)';
}

function complianceStatusBg(status: 'pass' | 'fail' | 'warning'): string {
  if (status === 'pass') return 'var(--green-muted)';
  if (status === 'fail') return 'var(--red-muted)';
  return 'rgba(229, 168, 19, 0.15)';
}

/* ========================================================================= */
/* COMPONENT                                                                  */
/* ========================================================================= */

export default function PortfolioRequirementsV2() {
  const [sets, setSets] = useState<RequirementSet[]>(initialRequirementSets);
  const [view, setView] = useState<'list' | 'editor'>('list');
  const [editingSet, setEditingSet] = useState<RequirementSet | null>(null);
  const [isNewSet, setIsNewSet] = useState(false);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [complianceModalId, setComplianceModalId] = useState<string | null>(null);
  const [presetModalOpen, setPresetModalOpen] = useState(false);

  // Add Rule form state
  const [newRuleName, setNewRuleName] = useState('');
  const [newRuleType, setNewRuleType] = useState(RULE_TYPE_OPTIONS[0].value);
  const [newRuleValue, setNewRuleValue] = useState('');
  const [newRuleThreshold, setNewRuleThreshold] = useState('');
  const [addRuleExpanded, setAddRuleExpanded] = useState(false);

  const complianceSet = useMemo(
    () => sets.find((s) => s.id === complianceModalId) ?? null,
    [sets, complianceModalId]
  );

  // ── Handlers ──

  function handleNewSet() {
    const newSet: RequirementSet = {
      id: generateId(),
      name: '',
      isBuiltIn: false,
      usedBy: [],
      rules: [],
    };
    setEditingSet(newSet);
    setIsNewSet(true);
    setAddRuleExpanded(false);
    resetAddRuleForm();
    setView('editor');
  }

  function handleEdit(set: RequirementSet) {
    setEditingSet({ ...set, rules: set.rules.map((r) => ({ ...r })) });
    setIsNewSet(false);
    setAddRuleExpanded(false);
    resetAddRuleForm();
    setView('editor');
  }

  function handleClone(set: RequirementSet) {
    const cloned: RequirementSet = {
      id: generateId(),
      name: `${set.name} (Copy)`,
      isBuiltIn: false,
      usedBy: [],
      rules: set.rules.map((r) => ({ ...r, id: generateId() })),
    };
    setEditingSet(cloned);
    setIsNewSet(true);
    setAddRuleExpanded(false);
    resetAddRuleForm();
    setView('editor');
  }

  function handleSave() {
    if (!editingSet || !editingSet.name.trim()) return;
    setSets((prev) => {
      const exists = prev.find((s) => s.id === editingSet.id);
      if (exists) return prev.map((s) => (s.id === editingSet.id ? editingSet : s));
      return [...prev, editingSet];
    });
    setView('list');
    setEditingSet(null);
    setIsNewSet(false);
  }

  function handleCancel() {
    setView('list');
    setEditingSet(null);
    setIsNewSet(false);
  }

  function handleDelete(id: string) {
    setSets((prev) => prev.filter((s) => s.id !== id));
    setDeleteConfirmId(null);
  }

  function handleRemoveRule(ruleId: string) {
    if (!editingSet) return;
    setEditingSet({
      ...editingSet,
      rules: editingSet.rules.filter((r) => r.id !== ruleId),
    });
  }

  function resetAddRuleForm() {
    setNewRuleName('');
    setNewRuleType(RULE_TYPE_OPTIONS[0].value);
    setNewRuleValue('');
    setNewRuleThreshold('');
  }

  function handleAddRule() {
    if (!editingSet || !newRuleName.trim() || !newRuleValue) return;
    const rule: Rule = {
      id: generateId(),
      name: newRuleName.trim(),
      type: newRuleType,
      value: parseFloat(newRuleValue),
      threshold: newRuleType === 'min_profitable_days' && newRuleThreshold
        ? parseFloat(newRuleThreshold)
        : null,
      currentValue: 0,
    };
    setEditingSet({
      ...editingSet,
      rules: [...editingSet.rules, rule],
    });
    resetAddRuleForm();
  }

  function handleCreateFromPreset(preset: PropFirmPreset) {
    const newSet: RequirementSet = {
      id: generateId(),
      name: preset.name,
      isBuiltIn: false,
      usedBy: [],
      rules: preset.rules.map((r) => ({ ...r, id: generateId(), currentValue: 0 })),
    };
    setSets((prev) => [...prev, newSet]);
    setPresetModalOpen(false);
  }

  function handleExport(set: RequirementSet) {
    const exportData = {
      name: set.name,
      rules: set.rules.map(({ id: _id, currentValue: _cv, ...rest }) => rest),
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${set.name.toLowerCase().replace(/\s+/g, '_')}_rules.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // =====================
  // EDITOR VIEW
  // =====================
  if (view === 'editor' && editingSet) {
    const isBuiltIn = editingSet.isBuiltIn;
    return (
      <div>
        <PageHeader
          title={isNewSet ? 'New Requirement Set' : `Edit: ${editingSet.name}`}
          subtitle={isBuiltIn ? 'Built-in set (read-only name)' : undefined}
          backHref="/portfolio-requirements"
          actions={
            <div className="flex gap-2">
              <button
                onClick={handleCancel}
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
                  opacity: editingSet.name.trim() ? 1 : 0.5,
                  cursor: editingSet.name.trim() ? 'pointer' : 'not-allowed',
                }}
                disabled={!editingSet.name.trim()}
              >
                Save Requirement Set
              </button>
            </div>
          }
        />

        {/* Details Card */}
        <Card className="mb-6">
          <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Details</h3>
          <div>
            <label className="block text-xs mb-1.5" style={{ color: 'var(--text-muted)' }}>Set Name</label>
            <input
              type="text"
              value={editingSet.name}
              onChange={(e) => setEditingSet({ ...editingSet, name: e.target.value })}
              disabled={isBuiltIn}
              placeholder="e.g. My Prop Firm Rules"
              className="w-full px-3 py-2 rounded-lg text-sm"
              style={{
                background: 'var(--bg-input)',
                border: '1px solid var(--border)',
                color: 'var(--text-primary)',
                opacity: isBuiltIn ? 0.6 : 1,
              }}
            />
          </div>
        </Card>

        {/* Rules Card */}
        <Card className="mb-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Rules</h3>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
              {editingSet.rules.length} rule{editingSet.rules.length !== 1 ? 's' : ''}
            </span>
          </div>

          {editingSet.rules.length === 0 ? (
            <div className="text-center py-8 rounded-lg" style={{ background: 'var(--bg-input)' }}>
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                No rules yet. Add a rule below to get started.
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {editingSet.rules.map((rule) => (
                <div
                  key={rule.id}
                  className="py-3 px-3 rounded-lg"
                  style={{ background: 'var(--bg-input)' }}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <div className="flex items-center gap-3 flex-1 min-w-0">
                      <span className="text-sm font-medium truncate">{rule.name}</span>
                      <span
                        className="text-xs px-2 py-0.5 rounded shrink-0"
                        style={{ background: 'var(--bg-card)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}
                      >
                        {RULE_TYPE_LABELS[rule.type] || rule.type}
                      </span>
                      <span className="text-sm font-medium shrink-0" style={{ color: 'var(--accent)' }}>
                        {formatRuleValue(rule)}
                      </span>
                      {rule.threshold !== null && (
                        <span className="text-xs shrink-0" style={{ color: 'var(--text-muted)' }}>
                          ({rule.threshold}% threshold)
                        </span>
                      )}
                    </div>
                    {!isBuiltIn && (
                      <button
                        onClick={() => handleRemoveRule(rule.id)}
                        className="ml-3 w-6 h-6 rounded flex items-center justify-center text-xs shrink-0"
                        style={{ color: 'var(--red)', background: 'transparent' }}
                        onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--red-muted)')}
                        onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                      >
                        x
                      </button>
                    )}
                  </div>
                  {/* Validation hint */}
                  <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                    {RULE_DESCRIPTIONS[rule.type] || 'Custom rule.'}
                  </p>
                </div>
              ))}
            </div>
          )}
        </Card>

        {/* Add Rule Card */}
        {!isBuiltIn && (
          <Card>
            <button
              onClick={() => setAddRuleExpanded(!addRuleExpanded)}
              className="flex items-center justify-between w-full"
            >
              <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
                Add Rule
              </h3>
              <span
                className="text-sm transition-transform"
                style={{
                  color: 'var(--text-muted)',
                  transform: addRuleExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
                  display: 'inline-block',
                }}
              >
                v
              </span>
            </button>

            {addRuleExpanded && (
              <div className="mt-4">
                {/* Rule type hint */}
                {newRuleType && (
                  <div
                    className="px-3 py-2 rounded-lg mb-4 text-xs"
                    style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                  >
                    {RULE_DESCRIPTIONS[newRuleType] || 'Select a rule type to see its description.'}
                  </div>
                )}

                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="space-y-3">
                    <div>
                      <label className="block text-xs mb-1.5" style={{ color: 'var(--text-muted)' }}>Rule Name</label>
                      <input
                        type="text"
                        value={newRuleName}
                        onChange={(e) => setNewRuleName(e.target.value)}
                        placeholder="e.g. Max Daily Loss"
                        className="w-full px-3 py-2 rounded-lg text-sm"
                        style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                      />
                    </div>
                    <div>
                      <label className="block text-xs mb-1.5" style={{ color: 'var(--text-muted)' }}>Rule Type</label>
                      <select
                        value={newRuleType}
                        onChange={(e) => setNewRuleType(e.target.value)}
                        className="w-full px-3 py-2 rounded-lg text-sm"
                        style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                      >
                        {RULE_TYPE_OPTIONS.map((opt) => (
                          <option key={opt.value} value={opt.value}>{opt.label}</option>
                        ))}
                      </select>
                    </div>
                  </div>
                  <div className="space-y-3">
                    <div>
                      <label className="block text-xs mb-1.5" style={{ color: 'var(--text-muted)' }}>Value</label>
                      <input
                        type="number"
                        value={newRuleValue}
                        onChange={(e) => setNewRuleValue(e.target.value)}
                        placeholder="e.g. 1000"
                        className="w-full px-3 py-2 rounded-lg text-sm"
                        style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                      />
                    </div>
                    {newRuleType === 'min_profitable_days' && (
                      <div>
                        <label className="block text-xs mb-1.5" style={{ color: 'var(--text-muted)' }}>Threshold (%)</label>
                        <input
                          type="number"
                          value={newRuleThreshold}
                          onChange={(e) => setNewRuleThreshold(e.target.value)}
                          placeholder="e.g. 50"
                          className="w-full px-3 py-2 rounded-lg text-sm"
                          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                        />
                      </div>
                    )}
                  </div>
                </div>

                <button
                  onClick={handleAddRule}
                  className="px-4 py-2 rounded-lg text-sm font-medium"
                  style={{
                    background: 'var(--accent-muted)',
                    color: 'var(--accent)',
                    opacity: newRuleName.trim() && newRuleValue ? 1 : 0.5,
                    cursor: newRuleName.trim() && newRuleValue ? 'pointer' : 'not-allowed',
                  }}
                  disabled={!newRuleName.trim() || !newRuleValue}
                >
                  + Add Rule
                </button>
              </div>
            )}
          </Card>
        )}

        {/* Bottom action bar */}
        <div className="flex justify-end gap-3 mt-6 pt-4 border-t" style={{ borderColor: 'var(--border)' }}>
          <button
            onClick={handleCancel}
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
              opacity: editingSet.name.trim() ? 1 : 0.5,
              cursor: editingSet.name.trim() ? 'pointer' : 'not-allowed',
            }}
            disabled={!editingSet.name.trim()}
          >
            Save Requirement Set
          </button>
        </div>
      </div>
    );
  }

  // =====================
  // LIST VIEW (default)
  // =====================

  const totalRules = sets.reduce((acc, s) => acc + s.rules.length, 0);
  const builtInCount = sets.filter((s) => s.isBuiltIn).length;
  const customCount = sets.filter((s) => !s.isBuiltIn).length;

  return (
    <div>
      <PageHeader
        title="Portfolio Requirements"
        subtitle="Prop firm rules and custom requirement sets"
        actions={
          <div className="flex gap-2">
            <button
              onClick={() => setPresetModalOpen(true)}
              className="px-4 py-2 rounded-lg text-sm"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
            >
              Prop Firm Presets
            </button>
            <button
              onClick={handleNewSet}
              className="px-4 py-2 rounded-lg text-sm font-medium"
              style={{ background: 'var(--accent)', color: 'white' }}
            >
              + New Requirement Set
            </button>
          </div>
        }
      />

      {/* Summary metrics */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <MetricCard label="Total Sets" value={sets.length.toString()} />
        <MetricCard label="Built-in" value={builtInCount.toString()} />
        <MetricCard label="Custom" value={customCount.toString()} />
        <MetricCard label="Total Rules" value={totalRules.toString()} />
      </div>

      {/* Requirement sets */}
      <TabBar tabs={['All Sets', 'Built-in', 'Custom']}>
        {(activeTab) => {
          const filtered = sets.filter((s) => {
            if (activeTab === 'Built-in') return s.isBuiltIn;
            if (activeTab === 'Custom') return !s.isBuiltIn;
            return true;
          });

          return (
            <div className="space-y-4">
              {filtered.length === 0 ? (
                <Card>
                  <div className="text-center py-8" style={{ color: 'var(--text-muted)' }}>
                    <p className="text-sm">No requirement sets in this category.</p>
                  </div>
                </Card>
              ) : (
                filtered.map((set) => (
                  <Card key={set.id}>
                    <div className="flex items-start justify-between gap-4">
                      {/* Left side */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1 flex-wrap">
                          <h3 className="font-semibold text-base truncate">{set.name}</h3>
                          {set.isBuiltIn && (
                            <span
                              className="text-xs px-2 py-0.5 rounded-full shrink-0"
                              style={{ background: 'var(--blue-muted)', color: 'var(--blue)' }}
                            >
                              Built-in
                            </span>
                          )}
                        </div>

                        {/* Usage + count */}
                        <div className="flex items-center gap-3 mb-3">
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                            {set.rules.length} rule{set.rules.length !== 1 ? 's' : ''}
                          </span>
                          {set.usedBy.length > 0 && (
                            <span className="text-xs" style={{ color: 'var(--accent)' }}>
                              Used by: {set.usedBy.join(', ')}
                            </span>
                          )}
                        </div>

                        {/* Rules with progress bars */}
                        <div className="space-y-2.5">
                          {set.rules.slice(0, 5).map((rule) => {
                            const pct = getProgressPct(rule);
                            const color = getProgressColor(rule);
                            return (
                              <div key={rule.id}>
                                <div className="flex items-center justify-between mb-1">
                                  <div className="flex items-center gap-2">
                                    <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                                      {rule.name}
                                    </span>
                                  </div>
                                  <div className="flex items-center gap-2">
                                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                                      {formatCurrentValue(rule.type, rule.currentValue)}
                                    </span>
                                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>/</span>
                                    <span className="text-xs font-medium">
                                      {formatRuleValue(rule)}
                                    </span>
                                  </div>
                                </div>
                                <div className="w-full h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--bg-input)' }}>
                                  <div
                                    className="h-full rounded-full transition-all"
                                    style={{ width: `${pct}%`, background: color }}
                                  />
                                </div>
                              </div>
                            );
                          })}
                          {set.rules.length > 5 && (
                            <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                              ... and {set.rules.length - 5} more
                            </p>
                          )}
                        </div>
                      </div>

                      {/* Right side: action buttons */}
                      <div className="flex flex-col gap-2 shrink-0">
                        <button
                          onClick={() => handleEdit(set)}
                          className="px-3 py-1.5 rounded text-xs"
                          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                        >
                          Edit
                        </button>
                        <button
                          onClick={() => handleClone(set)}
                          className="px-3 py-1.5 rounded text-xs"
                          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                        >
                          Clone
                        </button>
                        <button
                          onClick={() => setComplianceModalId(set.id)}
                          className="px-3 py-1.5 rounded text-xs"
                          style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                        >
                          Compliance
                        </button>
                        <button
                          onClick={() => handleExport(set)}
                          className="px-3 py-1.5 rounded text-xs"
                          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                        >
                          Export
                        </button>
                        {set.isBuiltIn ? (
                          <span
                            className="px-3 py-1.5 rounded text-xs text-center"
                            style={{
                              background: 'var(--bg-input)',
                              border: '1px solid var(--border)',
                              color: 'var(--text-muted)',
                              opacity: 0.5,
                              cursor: 'not-allowed',
                            }}
                            title="Built-in sets cannot be deleted"
                          >
                            Delete
                          </span>
                        ) : (
                          <button
                            onClick={() => setDeleteConfirmId(set.id)}
                            className="px-3 py-1.5 rounded text-xs"
                            style={{ background: 'var(--red-muted)', border: '1px solid transparent', color: 'var(--red)' }}
                          >
                            Delete
                          </button>
                        )}
                      </div>
                    </div>
                  </Card>
                ))
              )}
            </div>
          );
        }}
      </TabBar>

      {/* ── Compliance Preview Modal ── */}
      <Modal
        title={complianceSet ? `Compliance: ${complianceSet.name}` : 'Compliance Check'}
        isOpen={complianceModalId !== null}
        onClose={() => setComplianceModalId(null)}
        width="600px"
      >
        {complianceSet && (() => {
          const results = complianceSet.rules.map((rule) => ({
            rule: rule.name,
            status: getComplianceStatus(rule),
            current: formatCurrentValue(rule.type, rule.currentValue),
            limit: formatRuleValue(rule),
            pct: getProgressPct(rule),
          }));
          const passCount = results.filter((r) => r.status === 'pass').length;
          const failCount = results.filter((r) => r.status === 'fail').length;
          const warnCount = results.filter((r) => r.status === 'warning').length;

          return (
            <div className="space-y-4">
              {/* Summary */}
              <div className="grid grid-cols-3 gap-3">
                <div className="text-center py-3 rounded-lg" style={{ background: 'var(--green-muted)' }}>
                  <p className="text-lg font-bold" style={{ color: 'var(--green)' }}>{passCount}</p>
                  <p className="text-xs" style={{ color: 'var(--green)' }}>Passing</p>
                </div>
                <div className="text-center py-3 rounded-lg" style={{ background: 'rgba(229, 168, 19, 0.15)' }}>
                  <p className="text-lg font-bold" style={{ color: 'var(--yellow, #e5a813)' }}>{warnCount}</p>
                  <p className="text-xs" style={{ color: 'var(--yellow, #e5a813)' }}>Warning</p>
                </div>
                <div className="text-center py-3 rounded-lg" style={{ background: 'var(--red-muted)' }}>
                  <p className="text-lg font-bold" style={{ color: 'var(--red)' }}>{failCount}</p>
                  <p className="text-xs" style={{ color: 'var(--red)' }}>Failing</p>
                </div>
              </div>

              {/* Rule-by-rule */}
              <div className="space-y-2">
                {results.map((r, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between py-2.5 px-3 rounded-lg"
                    style={{ background: complianceStatusBg(r.status) }}
                  >
                    <div className="flex items-center gap-3 flex-1 min-w-0">
                      <span
                        className="w-2 h-2 rounded-full shrink-0"
                        style={{ background: complianceStatusColor(r.status) }}
                      />
                      <span className="text-sm truncate">{r.rule}</span>
                    </div>
                    <div className="flex items-center gap-3 shrink-0">
                      <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        {r.current} / {r.limit}
                      </span>
                      <span
                        className="text-xs px-2 py-0.5 rounded font-medium uppercase"
                        style={{ color: complianceStatusColor(r.status) }}
                      >
                        {r.status}
                      </span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Portfolio usage */}
              {complianceSet.usedBy.length > 0 && (
                <div className="pt-3 border-t" style={{ borderColor: 'var(--border)' }}>
                  <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Applied to portfolios:</p>
                  <div className="flex flex-wrap gap-1.5">
                    {complianceSet.usedBy.map((name) => (
                      <span
                        key={name}
                        className="text-xs px-2 py-0.5 rounded"
                        style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                      >
                        {name}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              <div className="flex justify-end pt-2">
                <button
                  onClick={() => setComplianceModalId(null)}
                  className="px-4 py-2 rounded-lg text-sm font-medium"
                  style={{ background: 'var(--accent)', color: 'white' }}
                >
                  Close
                </button>
              </div>
            </div>
          );
        })()}
      </Modal>

      {/* ── Prop Firm Presets Modal ── */}
      <Modal
        title="Prop Firm Presets"
        isOpen={presetModalOpen}
        onClose={() => setPresetModalOpen(false)}
        width="680px"
      >
        <div>
          <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
            Quick-create a requirement set from a built-in prop firm template.
          </p>
          <div className="space-y-3">
            {PROP_FIRM_PRESETS.map((preset) => (
              <div
                key={preset.id}
                className="flex items-center justify-between py-3 px-4 rounded-lg"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className="text-sm font-medium truncate">{preset.name}</span>
                    <span
                      className="text-xs px-2 py-0.5 rounded shrink-0"
                      style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                    >
                      {preset.accountSize}
                    </span>
                  </div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    {preset.firm} - {preset.rules.length} rules
                  </p>
                </div>
                <button
                  onClick={() => handleCreateFromPreset(preset)}
                  className="px-3 py-1.5 rounded text-xs font-medium shrink-0"
                  style={{ background: 'var(--accent)', color: 'white' }}
                >
                  Create
                </button>
              </div>
            ))}
          </div>
        </div>
      </Modal>

      {/* ── Delete Confirmation Modal ── */}
      <Modal
        title="Delete Requirement Set"
        isOpen={deleteConfirmId !== null}
        onClose={() => setDeleteConfirmId(null)}
        width="420px"
      >
        {deleteConfirmId && (() => {
          const target = sets.find((s) => s.id === deleteConfirmId);
          if (!target) return null;
          return (
            <div>
              <p className="text-sm mb-2" style={{ color: 'var(--text-secondary)' }}>
                Delete &ldquo;{target.name}&rdquo;? This cannot be undone.
              </p>
              {target.usedBy.length > 0 && (
                <p className="text-xs mb-4 px-3 py-2 rounded-lg" style={{ background: 'var(--red-muted)', color: 'var(--red)' }}>
                  Warning: This set is used by {target.usedBy.join(', ')}. Deleting it will remove requirement checks from those portfolios.
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
