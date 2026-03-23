'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

/* ========================================================================= */
/* TYPES                                                                      */
/* ========================================================================= */

interface Rule {
  id: string;
  name: string;
  type: string;
  value: number;
  currentValue: number;
}

interface RequirementSet {
  id: string;
  name: string;
  isBuiltIn: boolean;
  rules: Rule[];
  usedBy: string[];
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
RULE_TYPE_OPTIONS.forEach((opt) => { RULE_TYPE_LABELS[opt.value] = opt.label; });

/* ========================================================================= */
/* MOCK DATA                                                                  */
/* ========================================================================= */

const initialSets: RequirementSet[] = [
  {
    id: 'ttp-50k', name: 'TTP 50k Evaluation', isBuiltIn: true,
    usedBy: ['My Portfolio', 'Scalping Portfolio'],
    rules: [
      { id: 'r1', name: 'Max Daily Loss', type: 'max_daily_loss', value: 1000, currentValue: 342 },
      { id: 'r2', name: 'Max Total Drawdown', type: 'max_total_drawdown', value: 2500, currentValue: 875 },
      { id: 'r3', name: 'Min Trading Days', type: 'min_trading_days', value: 5, currentValue: 3 },
      { id: 'r4', name: 'Max Daily Trades', type: 'max_daily_trades', value: 15, currentValue: 7 },
      { id: 'r5', name: 'Profit Target', type: 'profit_target', value: 3000, currentValue: 1850 },
    ],
  },
  {
    id: 'ftmo-100k', name: 'FTMO 100k Challenge', isBuiltIn: true,
    usedBy: ['Swing Portfolio'],
    rules: [
      { id: 'r1', name: 'Max Daily Loss', type: 'max_daily_loss', value: 5000, currentValue: 1200 },
      { id: 'r2', name: 'Max Total Drawdown', type: 'max_total_drawdown', value: 10000, currentValue: 3400 },
      { id: 'r3', name: 'Min Trading Days', type: 'min_trading_days', value: 4, currentValue: 4 },
      { id: 'r4', name: 'Min Profitable Days', type: 'min_profitable_days', value: 60, currentValue: 72 },
      { id: 'r5', name: 'Profit Target', type: 'profit_target', value: 10000, currentValue: 6200 },
      { id: 'r6', name: 'Max Position Size', type: 'max_position_size', value: 10, currentValue: 5 },
    ],
  },
  {
    id: 'topstep-150k', name: 'Topstep 150k Combine', isBuiltIn: true,
    usedBy: [],
    rules: [
      { id: 'r1', name: 'Max Daily Loss', type: 'max_daily_loss', value: 4500, currentValue: 0 },
      { id: 'r2', name: 'Max Total Drawdown', type: 'max_total_drawdown', value: 5000, currentValue: 0 },
      { id: 'r3', name: 'Min Trading Days', type: 'min_trading_days', value: 5, currentValue: 0 },
      { id: 'r4', name: 'Profit Target', type: 'profit_target', value: 9000, currentValue: 0 },
    ],
  },
  {
    id: 'custom-1', name: 'My Custom Rules', isBuiltIn: false,
    usedBy: ['Personal Account'],
    rules: [
      { id: 'r1', name: 'Max Daily Loss', type: 'max_daily_loss', value: 500, currentValue: 125 },
      { id: 'r2', name: 'Max Daily Risk %', type: 'max_daily_risk_pct', value: 2.0, currentValue: 0.8 },
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
  const dollarTypes = ['max_daily_loss', 'max_total_drawdown', 'profit_target'];
  const pctTypes = ['min_profitable_days', 'max_daily_risk_pct'];
  if (dollarTypes.includes(type)) return `$${value.toLocaleString()}`;
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

export default function PortfolioRequirementsV3() {
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

                          {/* Delete rule button */}
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
                        </div>
                      );
                    })}

                    {set.rules.length === 0 && (
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>No rules yet.</p>
                    )}
                  </div>

                  {/* Add rule inline form */}
                  {addingToSetId === set.id ? (
                    <div className="flex items-end gap-2 flex-wrap" onClick={(e) => e.stopPropagation()}>
                      <div>
                        <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Type</label>
                        <select
                          value={newRuleType}
                          onChange={(e) => setNewRuleType(e.target.value)}
                          className="px-2 py-1.5 rounded text-xs"
                          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                        >
                          {RULE_TYPE_OPTIONS.map((opt) => (
                            <option key={opt.value} value={opt.value}>{opt.label}</option>
                          ))}
                        </select>
                      </div>
                      <div>
                        <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Value</label>
                        <input
                          type="number"
                          value={newRuleValue}
                          onChange={(e) => setNewRuleValue(e.target.value)}
                          onKeyDown={(e) => { if (e.key === 'Enter') handleAddRule(set.id); }}
                          placeholder="e.g. 1000"
                          autoFocus
                          className="w-24 px-2 py-1.5 rounded text-xs"
                          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                        />
                      </div>
                      <button
                        onClick={() => handleAddRule(set.id)}
                        className="px-3 py-1.5 rounded text-xs font-medium"
                        style={{ background: 'var(--accent)', color: 'white', opacity: newRuleValue ? 1 : 0.5 }}
                        disabled={!newRuleValue}
                      >
                        Add
                      </button>
                      <button
                        onClick={() => setAddingToSetId(null)}
                        className="px-3 py-1.5 rounded text-xs"
                        style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-muted)' }}
                      >
                        Cancel
                      </button>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2">
                      <button
                        onClick={(e) => { e.stopPropagation(); setAddingToSetId(set.id); setNewRuleValue(''); }}
                        className="text-xs px-3 py-1.5 rounded"
                        style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                      >
                        + Add Rule
                      </button>

                      {/* Delete set */}
                      {deleteConfirmId === set.id ? (
                        <div className="flex items-center gap-2 ml-auto">
                          <span className="text-xs" style={{ color: 'var(--red)' }}>Delete this set?</span>
                          <button
                            onClick={(e) => { e.stopPropagation(); handleDelete(set.id); }}
                            className="text-xs px-2 py-1 rounded font-medium"
                            style={{ background: 'var(--red)', color: 'white' }}
                          >
                            Yes
                          </button>
                          <button
                            onClick={(e) => { e.stopPropagation(); setDeleteConfirmId(null); }}
                            className="text-xs px-2 py-1 rounded"
                            style={{ border: '1px solid var(--border)', color: 'var(--text-muted)' }}
                          >
                            No
                          </button>
                        </div>
                      ) : (
                        <button
                          onClick={(e) => { e.stopPropagation(); setDeleteConfirmId(set.id); }}
                          className="text-xs px-3 py-1.5 rounded ml-auto"
                          style={{ color: 'var(--red)' }}
                          onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--red-muted)')}
                          onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                        >
                          Delete Set
                        </button>
                      )}
                    </div>
                  )}
                </div>
              )}
            </Card>
          );
        })}
      </div>
    </div>
  );
}
