'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

/* ========================================================================= */
/* TYPES                                                                      */
/* ========================================================================= */

interface TradeQualRule {
  id: string;
  name: string;
  type: string;
  value: number;
  unit: string;
  appliesTo: 'wins' | 'losses' | 'all';
  description: string;
}

interface Rule {
  id: string;
  name: string;
  type: string;
  value: number;
  currentValue: number;
  description?: string;
  thresholdPct?: number;
}

interface RequirementSet {
  id: string;
  name: string;
  firmKey?: string;
  isBuiltIn: boolean;
  rules: Rule[];
  tradeQualRules: TradeQualRule[];
  usedBy: string[];
}

/* ========================================================================= */
/* CONSTANTS                                                                  */
/* ========================================================================= */

const RULE_TYPE_OPTIONS = [
  { value: 'max_daily_loss_pct', label: 'Max Daily Loss (%)' },
  { value: 'max_total_drawdown_pct', label: 'Max Total Drawdown (%)' },
  { value: 'daily_pause_pct', label: 'Daily Pause Threshold (%)' },
  { value: 'min_profit_pct', label: 'Profit Target (%)' },
  { value: 'min_trading_days', label: 'Min Trading Days' },
  { value: 'min_profitable_days', label: 'Min Profitable Days (%)' },
  { value: 'max_position_size', label: 'Max Position Size' },
];

const RULE_TYPE_LABELS: Record<string, string> = {};
RULE_TYPE_OPTIONS.forEach((opt) => { RULE_TYPE_LABELS[opt.value] = opt.label; });

const TRADE_QUAL_OPTIONS = [
  { value: 'min_hold_time', label: 'Min Hold Time', unit: 'seconds' },
  { value: 'min_price_move', label: 'Min Price Move', unit: '$' },
  { value: 'min_profit_threshold', label: 'Min Profit Threshold', unit: '$' },
];

/* ========================================================================= */
/* MOCK DATA                                                                  */
/* ========================================================================= */

const initialSets: RequirementSet[] = [
  {
    id: 'ttp-50k', name: 'Trade The Pool — $50k', firmKey: 'ttp', isBuiltIn: true,
    usedBy: ['Main Portfolio', 'Scalping Portfolio'],
    rules: [
      { id: 'r1', name: 'Max Daily Loss', type: 'max_daily_loss_pct', value: 2, currentValue: 0.7, description: 'Maximum 2% loss in a single trading day' },
      { id: 'r2', name: 'Max Total Drawdown', type: 'max_total_drawdown_pct', value: 5, currentValue: 1.8, description: 'Maximum 5% total account drawdown' },
      { id: 'r3', name: 'Daily Pause', type: 'daily_pause_pct', value: 1.5, currentValue: 0.7, description: 'Pause trading when daily loss reaches 1.5%' },
      { id: 'r4', name: 'Min Trading Days', type: 'min_trading_days', value: 5, currentValue: 3, description: 'Minimum 5 trading days required' },
      { id: 'r5', name: 'Profit Target', type: 'min_profit_pct', value: 6, currentValue: 3.7, description: 'Achieve 6% profit target' },
    ],
    tradeQualRules: [
      { id: 'tq1', name: 'Min Hold Time', type: 'min_hold_time', value: 30, unit: 'seconds', appliesTo: 'wins', description: 'Position must be held for at least 30 seconds for profit to count' },
      { id: 'tq2', name: 'Min Price Move', type: 'min_price_move', value: 0.10, unit: '$', appliesTo: 'wins', description: 'Trade must move at least $0.10 in profit to count toward prop firm P&L' },
    ],
  },
  {
    id: 'ftmo-100k', name: 'FTMO — $100k Challenge', firmKey: 'ftmo', isBuiltIn: true,
    usedBy: ['Swing Portfolio'],
    rules: [
      { id: 'r1', name: 'Max Daily Loss', type: 'max_daily_loss_pct', value: 5, currentValue: 1.2, description: 'Maximum 5% loss in a single trading day' },
      { id: 'r2', name: 'Max Total Drawdown', type: 'max_total_drawdown_pct', value: 10, currentValue: 3.4, description: 'Maximum 10% total account drawdown' },
      { id: 'r3', name: 'Min Trading Days', type: 'min_trading_days', value: 4, currentValue: 4, description: 'Minimum 4 trading days required' },
      { id: 'r4', name: 'Min Profitable Days', type: 'min_profitable_days', value: 60, currentValue: 72, description: 'At least 60% of trading days must be profitable', thresholdPct: 0.5 },
      { id: 'r5', name: 'Profit Target', type: 'min_profit_pct', value: 10, currentValue: 6.2, description: 'Achieve 10% profit target' },
    ],
    tradeQualRules: [],
  },
  {
    id: 'topstep-150k', name: 'Topstep — $150k Combine', firmKey: 'topstep', isBuiltIn: true,
    usedBy: [],
    rules: [
      { id: 'r1', name: 'Max Daily Loss', type: 'max_daily_loss_pct', value: 3, currentValue: 0, description: 'Maximum 3% loss in a single trading day' },
      { id: 'r2', name: 'Max Total Drawdown', type: 'max_total_drawdown_pct', value: 3.3, currentValue: 0, description: 'Maximum 3.3% trailing drawdown' },
      { id: 'r3', name: 'Min Trading Days', type: 'min_trading_days', value: 5, currentValue: 0, description: 'Minimum 5 trading days required' },
      { id: 'r4', name: 'Profit Target', type: 'min_profit_pct', value: 6, currentValue: 0, description: 'Achieve 6% profit target' },
    ],
    tradeQualRules: [],
  },
  {
    id: 'custom-1', name: 'My Custom Rules', isBuiltIn: false,
    usedBy: ['Personal Account'],
    rules: [
      { id: 'r1', name: 'Max Daily Loss', type: 'max_daily_loss_pct', value: 2, currentValue: 0.8, description: 'Personal max daily loss limit' },
      { id: 'r2', name: 'Daily Pause', type: 'daily_pause_pct', value: 1, currentValue: 0.8, description: 'Pause at 1% daily loss' },
    ],
    tradeQualRules: [
      { id: 'tq1', name: 'Min Hold Time', type: 'min_hold_time', value: 15, unit: 'seconds', appliesTo: 'all', description: 'Custom minimum hold time for my scalping strategies' },
    ],
  },
];

/* ========================================================================= */
/* HELPERS                                                                    */
/* ========================================================================= */

function generateId(): string {
  return 'id-' + Math.random().toString(36).substring(2, 9);
}

function formatRuleValue(type: string, value: number): string {
  const pctTypes = ['max_daily_loss_pct', 'max_total_drawdown_pct', 'daily_pause_pct', 'min_profit_pct', 'min_profitable_days'];
  if (pctTypes.includes(type)) return `${value}%`;
  return value.toString();
}

function getRuleStatus(rule: Rule): 'pass' | 'fail' | 'warning' {
  if (rule.value === 0) return 'pass';
  const pct = (rule.currentValue / rule.value) * 100;
  const isMaxType = rule.type.startsWith('max_');
  if (isMaxType) {
    if (pct >= 100) return 'fail';
    if (pct >= 80) return 'warning';
    return 'pass';
  }
  if (pct >= 100) return 'pass';
  if (pct >= 50) return 'warning';
  return 'fail';
}

function statusDot(status: 'pass' | 'fail' | 'warning'): string {
  if (status === 'pass') return 'var(--green)';
  if (status === 'fail') return 'var(--red)';
  return 'var(--yellow, #e5a813)';
}

/* ========================================================================= */
/* COMPONENT                                                                  */
/* ========================================================================= */

export default function PortfolioRequirementsV5() {
  const [sets, setSets] = useState<RequirementSet[]>(initialSets);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  // Inline edit state
  const [editingRuleId, setEditingRuleId] = useState<string | null>(null);
  const [editRuleValue, setEditRuleValue] = useState('');

  // Inline add rule state
  const [addingToSetId, setAddingToSetId] = useState<string | null>(null);
  const [newRuleType, setNewRuleType] = useState(RULE_TYPE_OPTIONS[0].value);
  const [newRuleValue, setNewRuleValue] = useState('');

  // Inline add trade qual rule state
  const [addingTQToSetId, setAddingTQToSetId] = useState<string | null>(null);
  const [newTQType, setNewTQType] = useState(TRADE_QUAL_OPTIONS[0].value);
  const [newTQValue, setNewTQValue] = useState('');
  const [newTQAppliesTo, setNewTQAppliesTo] = useState<'wins' | 'losses' | 'all'>('wins');

  // Inline set name edit
  const [editingNameId, setEditingNameId] = useState<string | null>(null);
  const [editNameValue, setEditNameValue] = useState('');

  function handleNewSet() {
    const newSet: RequirementSet = {
      id: generateId(),
      name: 'New Requirement Set',
      isBuiltIn: false,
      usedBy: [],
      rules: [],
      tradeQualRules: [],
    };
    setSets((prev) => [...prev, newSet]);
    setExpandedId(newSet.id);
    setEditingNameId(newSet.id);
    setEditNameValue(newSet.name);
  }

  function handleSaveName(setId: string) {
    if (!editNameValue.trim()) return;
    setSets((prev) => prev.map((s) => s.id === setId ? { ...s, name: editNameValue.trim() } : s));
    setEditingNameId(null);
  }

  function handleDelete(id: string) {
    setSets((prev) => prev.filter((s) => s.id !== id));
    setDeleteConfirmId(null);
    if (expandedId === id) setExpandedId(null);
  }

  function handleStartEditRule(rule: Rule) {
    setEditingRuleId(rule.id);
    setEditRuleValue(rule.value.toString());
  }

  function handleSaveRule(setId: string, ruleId: string) {
    const val = parseFloat(editRuleValue);
    if (isNaN(val)) return;
    setSets((prev) => prev.map((s) =>
      s.id === setId
        ? { ...s, rules: s.rules.map((r) => r.id === ruleId ? { ...r, value: val } : r) }
        : s
    ));
    setEditingRuleId(null);
  }

  function handleDeleteRule(setId: string, ruleId: string) {
    setSets((prev) => prev.map((s) =>
      s.id === setId ? { ...s, rules: s.rules.filter((r) => r.id !== ruleId) } : s
    ));
  }

  function handleAddRule(setId: string) {
    if (!newRuleValue) return;
    const val = parseFloat(newRuleValue);
    if (isNaN(val)) return;
    const label = RULE_TYPE_LABELS[newRuleType] || newRuleType;
    const rule: Rule = {
      id: generateId(),
      name: label.replace(/\s*\(.*\)/, ''),
      type: newRuleType,
      value: val,
      currentValue: 0,
    };
    setSets((prev) => prev.map((s) =>
      s.id === setId ? { ...s, rules: [...s.rules, rule] } : s
    ));
    setAddingToSetId(null);
    setNewRuleType(RULE_TYPE_OPTIONS[0].value);
    setNewRuleValue('');
  }

  function handleCloneSet(set: RequirementSet) {
    const cloned: RequirementSet = {
      id: generateId(),
      name: `${set.name} (Copy)`,
      isBuiltIn: false,
      usedBy: [],
      rules: set.rules.map((r) => ({ ...r, id: generateId(), currentValue: 0 })),
      tradeQualRules: set.tradeQualRules.map((r) => ({ ...r, id: generateId() })),
    };
    setSets((prev) => [...prev, cloned]);
    setExpandedId(cloned.id);
  }

  function handleAddTQRule(setId: string) {
    if (!newTQValue) return;
    const val = parseFloat(newTQValue);
    if (isNaN(val)) return;
    const opt = TRADE_QUAL_OPTIONS.find((o) => o.value === newTQType);
    const rule: TradeQualRule = {
      id: generateId(),
      name: opt?.label || newTQType,
      type: newTQType,
      value: val,
      unit: opt?.unit || '',
      appliesTo: newTQAppliesTo,
      description: `${opt?.label || newTQType}: ${val} ${opt?.unit || ''} (${newTQAppliesTo === 'wins' ? 'wins only' : newTQAppliesTo === 'losses' ? 'losses only' : 'all trades'})`,
    };
    setSets((prev) => prev.map((s) =>
      s.id === setId ? { ...s, tradeQualRules: [...s.tradeQualRules, rule] } : s
    ));
    setAddingTQToSetId(null);
    setNewTQType(TRADE_QUAL_OPTIONS[0].value);
    setNewTQValue('');
  }

  function handleDeleteTQRule(setId: string, ruleId: string) {
    setSets((prev) => prev.map((s) =>
      s.id === setId ? { ...s, tradeQualRules: s.tradeQualRules.filter((r) => r.id !== ruleId) } : s
    ));
  }

  function getSetSummary(set: RequirementSet): { pass: number; fail: number; warn: number } {
    let pass = 0, fail = 0, warn = 0;
    set.rules.forEach((r) => {
      const s = getRuleStatus(r);
      if (s === 'pass') pass++;
      else if (s === 'fail') fail++;
      else warn++;
    });
    return { pass, fail, warn };
  }

  return (
    <div>
      <PageHeader
        title="Portfolio Requirements"
        subtitle="Manage requirement sets and rules"
        actions={
          <button
            onClick={handleNewSet}
            className="px-4 py-2 rounded-lg text-sm font-medium"
            style={{ background: 'var(--accent)', color: 'white' }}
          >
            + New Set
          </button>
        }
      />

      <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
        {sets.length} requirement set{sets.length !== 1 ? 's' : ''}
      </p>

      <div className="space-y-3">
        {sets.map((set) => {
          const isExpanded = expandedId === set.id;
          const summary = getSetSummary(set);

          return (
            <Card key={set.id}>
              {/* Set header */}
              <div
                className="flex items-center justify-between cursor-pointer"
                onClick={() => setExpandedId(isExpanded ? null : set.id)}
              >
                <div className="flex items-center gap-3 flex-1 min-w-0">
                  {/* Expand indicator */}
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    {isExpanded ? '▼' : '▶'}
                  </span>

                  {/* Name (inline editable) */}
                  {editingNameId === set.id ? (
                    <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
                      <input
                        type="text"
                        value={editNameValue}
                        onChange={(e) => setEditNameValue(e.target.value)}
                        onKeyDown={(e) => { if (e.key === 'Enter') handleSaveName(set.id); if (e.key === 'Escape') setEditingNameId(null); }}
                        autoFocus
                        className="px-2 py-1 rounded text-sm"
                        style={{ background: 'var(--bg-input)', border: '1px solid var(--accent)', color: 'var(--text-primary)', width: '200px' }}
                      />
                      <button
                        onClick={() => handleSaveName(set.id)}
                        className="text-xs px-2 py-1 rounded"
                        style={{ background: 'var(--accent)', color: 'white' }}
                      >
                        Save
                      </button>
                    </div>
                  ) : (
                    <h3
                      className="font-semibold text-sm truncate"
                      onDoubleClick={(e) => {
                        e.stopPropagation();
                        if (!set.isBuiltIn) {
                          setEditingNameId(set.id);
                          setEditNameValue(set.name);
                        }
                      }}
                    >
                      {set.name}
                    </h3>
                  )}

                  {set.isBuiltIn && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded shrink-0" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                      Built-in
                    </span>
                  )}
                  {set.isBuiltIn && (
                    <span className="text-[10px]" style={{ color: 'var(--text-muted)' }} title="Built-in sets are read-only. Clone to customize.">🔒</span>
                  )}
                  {set.tradeQualRules.length > 0 && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded shrink-0" style={{ background: 'rgba(255,152,0,0.12)', color: 'var(--orange)' }}>
                      {set.tradeQualRules.length} TQ rule{set.tradeQualRules.length !== 1 ? 's' : ''}
                    </span>
                  )}
                </div>

                {/* Summary badges */}
                <div className="flex items-center gap-3 shrink-0">
                  {set.rules.length > 0 && (
                    <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-muted)' }}>
                      {summary.pass > 0 && (
                        <span className="flex items-center gap-1">
                          <span className="w-2 h-2 rounded-full inline-block" style={{ background: 'var(--green)' }} />
                          {summary.pass}
                        </span>
                      )}
                      {summary.warn > 0 && (
                        <span className="flex items-center gap-1">
                          <span className="w-2 h-2 rounded-full inline-block" style={{ background: 'var(--yellow, #e5a813)' }} />
                          {summary.warn}
                        </span>
                      )}
                      {summary.fail > 0 && (
                        <span className="flex items-center gap-1">
                          <span className="w-2 h-2 rounded-full inline-block" style={{ background: 'var(--red)' }} />
                          {summary.fail}
                        </span>
                      )}
                    </div>
                  )}
                  {set.usedBy.length > 0 && (
                    <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                      {set.usedBy.length} portfolio{set.usedBy.length !== 1 ? 's' : ''}
                    </span>
                  )}
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    {set.rules.length} rule{set.rules.length !== 1 ? 's' : ''}
                  </span>
                </div>
              </div>

              {/* Expanded content: rules as badges + inline actions */}
              {isExpanded && (
                <div className="mt-4 pt-3 border-t" style={{ borderColor: 'var(--border)' }}>
                  {/* Rule badges */}
                  <div className="flex flex-wrap gap-2 mb-3">
                    {set.rules.map((rule) => {
                      const status = getRuleStatus(rule);
                      const isEditing = editingRuleId === rule.id;

                      return (
                        <div
                          key={rule.id}
                          className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs"
                          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
                        >
                          {/* Status dot */}
                          <span
                            className="w-2 h-2 rounded-full shrink-0"
                            style={{ background: statusDot(status) }}
                          />

                          <span style={{ color: 'var(--text-secondary)' }}>{rule.name}:</span>

                          {isEditing ? (
                            <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
                              <input
                                type="number"
                                value={editRuleValue}
                                onChange={(e) => setEditRuleValue(e.target.value)}
                                onKeyDown={(e) => { if (e.key === 'Enter') handleSaveRule(set.id, rule.id); if (e.key === 'Escape') setEditingRuleId(null); }}
                                autoFocus
                                className="w-20 px-1.5 py-0.5 rounded text-xs"
                                style={{ background: 'var(--bg-card)', border: '1px solid var(--accent)', color: 'var(--text-primary)' }}
                              />
                              <button
                                onClick={() => handleSaveRule(set.id, rule.id)}
                                className="text-[10px] px-1.5 py-0.5 rounded"
                                style={{ background: 'var(--accent)', color: 'white' }}
                              >
                                OK
                              </button>
                            </div>
                          ) : (
                            <span
                              className="font-medium cursor-pointer"
                              style={{ color: 'var(--text-primary)' }}
                              onClick={(e) => { e.stopPropagation(); handleStartEditRule(rule); }}
                              title="Click to edit"
                            >
                              {formatRuleValue(rule.type, rule.value)}
                            </span>
                          )}

                          {/* Delete rule button (not on built-in sets) */}
                          {!set.isBuiltIn && (
                            <button
                              onClick={(e) => { e.stopPropagation(); handleDeleteRule(set.id, rule.id); }}
                              className="w-4 h-4 flex items-center justify-center rounded text-[10px] shrink-0 transition-colors"
                              style={{ color: 'var(--text-muted)' }}
                              onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--red)'; e.currentTarget.style.background = 'var(--red-muted)'; }}
                              onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-muted)'; e.currentTarget.style.background = 'transparent'; }}
                              title="Remove rule"
                            >
                              x
                            </button>
                          )}
                          {/* Description tooltip */}
                          {rule.description && (
                            <span className="text-[10px] hidden sm:inline" style={{ color: 'var(--text-muted)' }} title={rule.description}>ⓘ</span>
                          )}
                        </div>
                      );
                    })}

                    {set.rules.length === 0 && (
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>No rules yet.</p>
                    )}
                  </div>

                  {/* Add rule inline form (not on built-in sets) */}
                  {!set.isBuiltIn && addingToSetId === set.id ? (
                    <div className="flex items-end gap-2 flex-wrap mb-3" onClick={(e) => e.stopPropagation()}>
                      <div>
                        <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Type</label>
                        <select value={newRuleType} onChange={(e) => setNewRuleType(e.target.value)}
                          className="px-2 py-1.5 rounded text-xs"
                          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}>
                          {RULE_TYPE_OPTIONS.map((opt) => (
                            <option key={opt.value} value={opt.value}>{opt.label}</option>
                          ))}
                        </select>
                      </div>
                      <div>
                        <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Value</label>
                        <input type="number" value={newRuleValue} onChange={(e) => setNewRuleValue(e.target.value)}
                          onKeyDown={(e) => { if (e.key === 'Enter') handleAddRule(set.id); }}
                          placeholder="e.g. 5" autoFocus className="w-24 px-2 py-1.5 rounded text-xs"
                          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} />
                      </div>
                      <button onClick={() => handleAddRule(set.id)} className="px-3 py-1.5 rounded text-xs font-medium"
                        style={{ background: 'var(--accent)', color: 'white', opacity: newRuleValue ? 1 : 0.5, cursor: 'pointer' }} disabled={!newRuleValue}>
                        Add
                      </button>
                      <button onClick={() => setAddingToSetId(null)} className="px-3 py-1.5 rounded text-xs"
                        style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-muted)', cursor: 'pointer' }}>
                        Cancel
                      </button>
                    </div>
                  ) : !set.isBuiltIn ? (
                    <div className="mb-3">
                      <button onClick={(e) => { e.stopPropagation(); setAddingToSetId(set.id); setNewRuleValue(''); }}
                        className="text-xs px-3 py-1.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)', cursor: 'pointer' }}>
                        + Add Compliance Rule
                      </button>
                    </div>
                  ) : null}

                  {/* ---- Trade Qualification Rules ---- */}
                  <div className="mt-3 pt-3 border-t" style={{ borderColor: 'var(--border)' }}>
                    <div className="flex items-center gap-2 mb-2">
                      <h4 className="text-xs font-medium" style={{ color: 'var(--orange)' }}>Trade Qualification Rules</h4>
                      <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                        Trades not meeting these criteria may not count toward prop firm P&L
                      </span>
                    </div>

                    {set.tradeQualRules.length > 0 ? (
                      <div className="flex flex-wrap gap-2 mb-2">
                        {set.tradeQualRules.map((tq) => {
                          const appliesToColors = { wins: { color: 'var(--green)', label: 'Wins only' }, losses: { color: 'var(--red)', label: 'Losses only' }, all: { color: 'var(--text-muted)', label: 'All trades' } };
                          const at = appliesToColors[tq.appliesTo];
                          return (
                          <div key={tq.id} className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs"
                            style={{ background: 'rgba(255,152,0,0.08)', border: '1px solid rgba(255,152,0,0.2)' }}>
                            <span style={{ color: 'var(--orange)' }}>{tq.name}:</span>
                            <span className="font-medium" style={{ color: 'var(--text-primary)' }}>
                              {tq.unit === '$' ? `$${tq.value}` : `${tq.value} ${tq.unit}`}
                            </span>
                            <span className="text-[10px] px-1.5 py-0.5 rounded-full" style={{ color: at.color, background: at.color + '18' }}>
                              {at.label}
                            </span>
                            {!set.isBuiltIn && (
                              <button onClick={(e) => { e.stopPropagation(); handleDeleteTQRule(set.id, tq.id); }}
                                className="w-4 h-4 flex items-center justify-center rounded text-[10px] shrink-0"
                                style={{ color: 'var(--text-muted)' }}
                                onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--red)'; }}
                                onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-muted)'; }}>
                                x
                              </button>
                            )}
                            <span className="text-[10px]" style={{ color: 'var(--text-muted)' }} title={tq.description}>ⓘ</span>
                          </div>
                          );
                        })}
                      </div>
                    ) : (
                      <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>No trade qualification rules. All trades count.</p>
                    )}

                    {/* Add TQ rule form (not on built-in sets) */}
                    {!set.isBuiltIn && addingTQToSetId === set.id ? (
                      <div className="flex items-end gap-2 flex-wrap" onClick={(e) => e.stopPropagation()}>
                        <div>
                          <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Type</label>
                          <select value={newTQType} onChange={(e) => setNewTQType(e.target.value)}
                            className="px-2 py-1.5 rounded text-xs"
                            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}>
                            {TRADE_QUAL_OPTIONS.map((opt) => (
                              <option key={opt.value} value={opt.value}>{opt.label} ({opt.unit})</option>
                            ))}
                          </select>
                        </div>
                        <div>
                          <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Value</label>
                          <input type="number" value={newTQValue} onChange={(e) => setNewTQValue(e.target.value)}
                            onKeyDown={(e) => { if (e.key === 'Enter') handleAddTQRule(set.id); }}
                            placeholder="e.g. 30" autoFocus className="w-24 px-2 py-1.5 rounded text-xs"
                            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} />
                        </div>
                        <div>
                          <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Applies To</label>
                          <select value={newTQAppliesTo} onChange={(e) => setNewTQAppliesTo(e.target.value as 'wins' | 'losses' | 'all')}
                            className="px-2 py-1.5 rounded text-xs"
                            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}>
                            <option value="wins">Wins only</option>
                            <option value="losses">Losses only</option>
                            <option value="all">All trades</option>
                          </select>
                        </div>
                        <button onClick={() => handleAddTQRule(set.id)} className="px-3 py-1.5 rounded text-xs font-medium"
                          style={{ background: 'var(--orange)', color: 'white', opacity: newTQValue ? 1 : 0.5, cursor: 'pointer' }} disabled={!newTQValue}>
                          Add
                        </button>
                        <button onClick={() => setAddingTQToSetId(null)} className="px-3 py-1.5 rounded text-xs"
                          style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-muted)', cursor: 'pointer' }}>
                          Cancel
                        </button>
                      </div>
                    ) : !set.isBuiltIn ? (
                      <button onClick={(e) => { e.stopPropagation(); setAddingTQToSetId(set.id); setNewTQValue(''); }}
                        className="text-xs px-3 py-1.5 rounded"
                        style={{ background: 'rgba(255,152,0,0.12)', color: 'var(--orange)', cursor: 'pointer' }}>
                        + Add Qualification Rule
                      </button>
                    ) : null}
                  </div>

                  {/* ---- Action row ---- */}
                  <div className="flex items-center gap-2 mt-3 pt-3 border-t" style={{ borderColor: 'var(--border)' }}>
                    <button onClick={(e) => { e.stopPropagation(); handleCloneSet(set); }}
                      className="text-xs px-3 py-1.5 rounded"
                      style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                      Clone
                    </button>

                    {/* Delete set (not built-in) */}
                    {!set.isBuiltIn && (
                      <>
                        {deleteConfirmId === set.id ? (
                          <div className="flex items-center gap-2 ml-auto">
                            <span className="text-xs" style={{ color: 'var(--red)' }}>Delete this set?</span>
                            <button onClick={(e) => { e.stopPropagation(); handleDelete(set.id); }}
                              className="text-xs px-2 py-1 rounded font-medium" style={{ background: 'var(--red)', color: 'white', cursor: 'pointer' }}>
                              Yes
                            </button>
                            <button onClick={(e) => { e.stopPropagation(); setDeleteConfirmId(null); }}
                              className="text-xs px-2 py-1 rounded" style={{ border: '1px solid var(--border)', color: 'var(--text-muted)', cursor: 'pointer' }}>
                              No
                            </button>
                          </div>
                        ) : (
                          <button onClick={(e) => { e.stopPropagation(); setDeleteConfirmId(set.id); }}
                            className="text-xs px-3 py-1.5 rounded ml-auto" style={{ color: 'var(--red)', cursor: 'pointer' }}
                            onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--red-muted)')}
                            onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}>
                            Delete Set
                          </button>
                        )}
                      </>
                    )}
                    {set.isBuiltIn && (
                      <span className="text-[10px] ml-auto" style={{ color: 'var(--text-muted)' }}>
                        Built-in sets are read-only. Clone to customize.
                      </span>
                    )}
                  </div>
                </div>
              )}
            </Card>
          );
        })}
      </div>
    </div>
  );
}
