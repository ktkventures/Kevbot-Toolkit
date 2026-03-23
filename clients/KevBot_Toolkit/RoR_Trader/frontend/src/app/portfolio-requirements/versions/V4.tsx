'use client';

import { useState, useMemo, useCallback } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
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
  currentValue: number;
}

interface RequirementSet {
  id: string;
  name: string;
  isBuiltIn: boolean;
  rules: Rule[];
  usedBy: string[];
  description: string;
}

/* ========================================================================= */
/* CONSTANTS                                                                  */
/* ========================================================================= */

const RULE_TYPE_OPTIONS = [
  { value: 'max_daily_loss', label: 'Max Daily Loss ($)', icon: 'DL', category: 'loss' },
  { value: 'max_total_drawdown', label: 'Max Total Drawdown ($)', icon: 'DD', category: 'loss' },
  { value: 'min_trading_days', label: 'Min Trading Days', icon: 'TD', category: 'activity' },
  { value: 'max_daily_trades', label: 'Max Daily Trades', icon: 'DT', category: 'activity' },
  { value: 'min_profitable_days', label: 'Min Profitable Days (%)', icon: 'PD', category: 'profit' },
  { value: 'max_position_size', label: 'Max Position Size', icon: 'PS', category: 'risk' },
  { value: 'max_daily_risk_pct', label: 'Max Daily Risk (%)', icon: 'DR', category: 'risk' },
  { value: 'profit_target', label: 'Profit Target ($)', icon: 'PT', category: 'profit' },
];

const RULE_TYPE_LABELS: Record<string, string> = {};
RULE_TYPE_OPTIONS.forEach((opt) => { RULE_TYPE_LABELS[opt.value] = opt.label; });

const RULE_CATEGORIES: Record<string, { label: string; color: string }> = {
  loss: { label: 'Loss Limits', color: 'var(--red)' },
  activity: { label: 'Activity', color: 'var(--blue)' },
  risk: { label: 'Risk Control', color: 'var(--orange)' },
  profit: { label: 'Targets', color: 'var(--green)' },
};

/* ========================================================================= */
/* PROP FIRM TEMPLATES                                                        */
/* ========================================================================= */

interface PropFirmTemplate {
  id: string;
  name: string;
  firm: string;
  accountSize: string;
  description: string;
  rules: Omit<Rule, 'currentValue'>[];
  popularity: number; // 1-5 stars
}

const PROP_FIRM_TEMPLATES: PropFirmTemplate[] = [
  {
    id: 'ttp-50k', firm: 'TTP', name: 'TTP 50k Evaluation', accountSize: '$50,000',
    description: 'Trade The Pool evaluation account with standard rules.',
    popularity: 5,
    rules: [
      { id: 'r1', name: 'Max Daily Loss', type: 'max_daily_loss', value: 1000 },
      { id: 'r2', name: 'Max Total Drawdown', type: 'max_total_drawdown', value: 2500 },
      { id: 'r3', name: 'Min Trading Days', type: 'min_trading_days', value: 5 },
      { id: 'r4', name: 'Max Daily Trades', type: 'max_daily_trades', value: 15 },
      { id: 'r5', name: 'Profit Target', type: 'profit_target', value: 3000 },
    ],
  },
  {
    id: 'ftmo-100k', firm: 'FTMO', name: 'FTMO 100k Challenge', accountSize: '$100,000',
    description: 'FTMO challenge phase with standard evaluation rules.',
    popularity: 5,
    rules: [
      { id: 'r1', name: 'Max Daily Loss', type: 'max_daily_loss', value: 5000 },
      { id: 'r2', name: 'Max Total Drawdown', type: 'max_total_drawdown', value: 10000 },
      { id: 'r3', name: 'Min Trading Days', type: 'min_trading_days', value: 4 },
      { id: 'r4', name: 'Min Profitable Days', type: 'min_profitable_days', value: 60 },
      { id: 'r5', name: 'Profit Target', type: 'profit_target', value: 10000 },
      { id: 'r6', name: 'Max Position Size', type: 'max_position_size', value: 10 },
    ],
  },
  {
    id: 'topstep-150k', firm: 'Topstep', name: 'Topstep 150k Combine', accountSize: '$150,000',
    description: 'Topstep Trading Combine with profit target and trailing drawdown.',
    popularity: 4,
    rules: [
      { id: 'r1', name: 'Max Daily Loss', type: 'max_daily_loss', value: 4500 },
      { id: 'r2', name: 'Max Total Drawdown', type: 'max_total_drawdown', value: 5000 },
      { id: 'r3', name: 'Min Trading Days', type: 'min_trading_days', value: 5 },
      { id: 'r4', name: 'Profit Target', type: 'profit_target', value: 9000 },
    ],
  },
  {
    id: 'apex-250k', firm: 'Apex', name: 'Apex 250k Evaluation', accountSize: '$250,000',
    description: 'Apex Trader Funding with generous drawdown limits.',
    popularity: 4,
    rules: [
      { id: 'r1', name: 'Max Daily Loss', type: 'max_daily_loss', value: 6250 },
      { id: 'r2', name: 'Max Total Drawdown', type: 'max_total_drawdown', value: 15000 },
      { id: 'r3', name: 'Min Trading Days', type: 'min_trading_days', value: 7 },
      { id: 'r4', name: 'Profit Target', type: 'profit_target', value: 15000 },
    ],
  },
];

/* ========================================================================= */
/* MOCK DATA                                                                  */
/* ========================================================================= */

const initialSets: RequirementSet[] = [
  {
    id: 'ttp-50k', name: 'TTP 50k Evaluation', isBuiltIn: true,
    description: 'Trade The Pool evaluation rules',
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
    description: 'FTMO challenge evaluation rules',
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
    id: 'custom-1', name: 'My Custom Rules', isBuiltIn: false,
    description: 'Personal account risk management',
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

function statusColor(status: 'pass' | 'fail' | 'warning'): string {
  if (status === 'pass') return 'var(--green)';
  if (status === 'fail') return 'var(--red)';
  return 'var(--yellow, #e5a813)';
}

function getRuleCategory(type: string): string {
  return RULE_TYPE_OPTIONS.find(o => o.value === type)?.category || 'risk';
}

function getRuleProgress(rule: Rule): number {
  if (rule.value === 0) return 0;
  return Math.min((rule.currentValue / rule.value) * 100, 100);
}

/* ========================================================================= */
/* COMPONENT                                                                  */
/* ========================================================================= */

export default function PortfolioRequirementsV4() {
  const [sets, setSets] = useState<RequirementSet[]>(initialSets);
  const [selectedSetId, setSelectedSetId] = useState<string | null>(null);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  // Template marketplace modal
  const [showTemplates, setShowTemplates] = useState(false);
  const [templateFilter, setTemplateFilter] = useState('');

  // Compliance simulator
  const [simulatorSetId, setSimulatorSetId] = useState<string | null>(null);
  const [simRunning, setSimRunning] = useState(false);
  const [simResults, setSimResults] = useState<Record<string, 'pass' | 'fail' | 'warning'> | null>(null);

  // Inline editing
  const [editingRuleId, setEditingRuleId] = useState<string | null>(null);
  const [editRuleValue, setEditRuleValue] = useState('');

  // Add rule state
  const [addingToSetId, setAddingToSetId] = useState<string | null>(null);
  const [newRuleType, setNewRuleType] = useState(RULE_TYPE_OPTIONS[0].value);
  const [newRuleValue, setNewRuleValue] = useState('');

  // Editing set name
  const [editingNameId, setEditingNameId] = useState<string | null>(null);
  const [editNameValue, setEditNameValue] = useState('');

  // --- Computed ---
  const selectedSet = useMemo(
    () => sets.find(s => s.id === selectedSetId) || null,
    [sets, selectedSetId]
  );

  const filteredTemplates = useMemo(
    () => PROP_FIRM_TEMPLATES.filter(t =>
      templateFilter === '' ||
      t.name.toLowerCase().includes(templateFilter.toLowerCase()) ||
      t.firm.toLowerCase().includes(templateFilter.toLowerCase())
    ),
    [templateFilter]
  );

  // --- Handlers ---
  function handleNewSet() {
    const newSet: RequirementSet = {
      id: generateId(),
      name: 'New Requirement Set',
      isBuiltIn: false,
      description: '',
      usedBy: [],
      rules: [],
    };
    setSets((prev) => [...prev, newSet]);
    setSelectedSetId(newSet.id);
    setEditingNameId(newSet.id);
    setEditNameValue(newSet.name);
  }

  function handleImportTemplate(template: PropFirmTemplate) {
    const exists = sets.find(s => s.id === template.id);
    if (exists) return;
    const newSet: RequirementSet = {
      id: template.id,
      name: template.name,
      isBuiltIn: true,
      description: template.description,
      usedBy: [],
      rules: template.rules.map(r => ({ ...r, currentValue: 0 })),
    };
    setSets((prev) => [...prev, newSet]);
    setShowTemplates(false);
    setSelectedSetId(newSet.id);
  }

  function handleSaveName(setId: string) {
    if (!editNameValue.trim()) return;
    setSets((prev) => prev.map((s) => s.id === setId ? { ...s, name: editNameValue.trim() } : s));
    setEditingNameId(null);
  }

  function handleDelete(id: string) {
    setSets((prev) => prev.filter((s) => s.id !== id));
    setDeleteConfirmId(null);
    if (selectedSetId === id) setSelectedSetId(null);
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

  function runComplianceSimulator(setId: string) {
    setSimulatorSetId(setId);
    setSimRunning(true);
    setSimResults(null);
    // Simulate a 1.5s check
    setTimeout(() => {
      const set = sets.find(s => s.id === setId);
      if (!set) { setSimRunning(false); return; }
      const results: Record<string, 'pass' | 'fail' | 'warning'> = {};
      set.rules.forEach(r => {
        results[r.id] = getRuleStatus(r);
      });
      setSimResults(results);
      setSimRunning(false);
    }, 1500);
  }

  function getSetSummary(set: RequirementSet): { pass: number; fail: number; warn: number; total: number } {
    let pass = 0, fail = 0, warn = 0;
    set.rules.forEach((r) => {
      const s = getRuleStatus(r);
      if (s === 'pass') pass++;
      else if (s === 'fail') fail++;
      else warn++;
    });
    return { pass, fail, warn, total: set.rules.length };
  }

  // Overall compliance score
  function getComplianceScore(set: RequirementSet): number {
    if (set.rules.length === 0) return 100;
    const summary = getSetSummary(set);
    return Math.round((summary.pass / summary.total) * 100);
  }

  return (
    <div>
      <PageHeader
        title="Portfolio Requirements"
        subtitle="Visual rule builder with compliance simulation"
        actions={
          <div className="flex gap-2">
            <button
              onClick={() => setShowTemplates(true)}
              className="px-4 py-2 rounded-lg text-sm font-medium"
              style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
            >
              Browse Templates
            </button>
            <button
              onClick={handleNewSet}
              className="px-4 py-2 rounded-lg text-sm font-medium"
              style={{ background: 'var(--accent)', color: 'white' }}
            >
              + New Set
            </button>
          </div>
        }
      />

      <TabBar tabs={['My Rule Sets', 'Compare Firms']}>
        {(activeTab) => {
          if (activeTab === 'Compare Firms') {
            return (
              <div>
                {/* Prop firm comparison table */}
                <Card>
                  <h3 className="text-sm font-semibold mb-4" style={{ color: 'var(--text-secondary)' }}>
                    Prop Firm Rule Comparison
                  </h3>
                  <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8125rem' }}>
                      <thead>
                        <tr>
                          <th
                            className="text-left py-2 px-3 text-xs font-semibold"
                            style={{ color: 'var(--text-muted)', borderBottom: '1px solid var(--border)' }}
                          >
                            Rule
                          </th>
                          {PROP_FIRM_TEMPLATES.map(t => (
                            <th
                              key={t.id}
                              className="text-center py-2 px-3 text-xs font-semibold"
                              style={{ color: 'var(--text-muted)', borderBottom: '1px solid var(--border)', minWidth: 120 }}
                            >
                              <div>{t.firm}</div>
                              <div className="text-[10px] font-normal" style={{ color: 'var(--text-muted)' }}>{t.accountSize}</div>
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {RULE_TYPE_OPTIONS.map(ruleType => {
                          const cat = RULE_CATEGORIES[ruleType.category];
                          return (
                            <tr key={ruleType.value}>
                              <td
                                className="py-2 px-3"
                                style={{ borderBottom: '1px solid var(--border)' }}
                              >
                                <div className="flex items-center gap-2">
                                  <span
                                    className="text-[10px] px-1.5 py-0.5 rounded font-semibold"
                                    style={{ background: `${cat.color}20`, color: cat.color }}
                                  >
                                    {ruleType.icon}
                                  </span>
                                  <span className="text-xs">{ruleType.label}</span>
                                </div>
                              </td>
                              {PROP_FIRM_TEMPLATES.map(template => {
                                const rule = template.rules.find(r => r.type === ruleType.value);
                                return (
                                  <td
                                    key={template.id}
                                    className="text-center py-2 px-3"
                                    style={{ borderBottom: '1px solid var(--border)' }}
                                  >
                                    {rule ? (
                                      <span className="text-xs font-semibold">
                                        {formatRuleValue(ruleType.value, rule.value)}
                                      </span>
                                    ) : (
                                      <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>--</span>
                                    )}
                                  </td>
                                );
                              })}
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </Card>
              </div>
            );
          }

          // My Rule Sets tab
          return (
            <div>
              <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
                {sets.length} requirement set{sets.length !== 1 ? 's' : ''}
              </p>

              <div className="grid grid-cols-3 gap-4 mb-6">
                {sets.map((set) => {
                  const summary = getSetSummary(set);
                  const score = getComplianceScore(set);
                  const isActive = selectedSetId === set.id;

                  return (
                    <div
                      key={set.id}
                      onClick={() => setSelectedSetId(isActive ? null : set.id)}
                      className="rounded-xl border p-4 cursor-pointer transition-all"
                      style={{
                        background: 'var(--bg-card)',
                        borderColor: isActive ? 'var(--accent)' : 'var(--border)',
                        boxShadow: isActive ? '0 0 0 2px var(--accent-muted)' : 'var(--card-shadow)',
                        transform: isActive ? 'translateY(-2px)' : 'translateY(0)',
                      }}
                    >
                      {/* Header */}
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-0.5">
                            <h3 className="font-semibold text-sm truncate">{set.name}</h3>
                            {set.isBuiltIn && (
                              <span className="text-[9px] px-1.5 py-0.5 rounded shrink-0" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                                Built-in
                              </span>
                            )}
                          </div>
                          <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                            {set.rules.length} rule{set.rules.length !== 1 ? 's' : ''}
                            {set.usedBy.length > 0 && ` | ${set.usedBy.length} portfolio${set.usedBy.length !== 1 ? 's' : ''}`}
                          </p>
                        </div>

                        {/* Compliance ring */}
                        <div style={{ position: 'relative', width: 44, height: 44, flexShrink: 0 }}>
                          <svg width="44" height="44" viewBox="0 0 44 44">
                            <circle cx="22" cy="22" r="18" fill="none" stroke="var(--border)" strokeWidth="3" />
                            <circle
                              cx="22" cy="22" r="18"
                              fill="none"
                              stroke={score >= 80 ? 'var(--green)' : score >= 50 ? 'var(--yellow, #e5a813)' : 'var(--red)'}
                              strokeWidth="3"
                              strokeLinecap="round"
                              strokeDasharray={`${(score / 100) * 113} 113`}
                              transform="rotate(-90 22 22)"
                            />
                            <text x="22" y="24" textAnchor="middle" fontSize="10" fontWeight="700"
                              fill={score >= 80 ? 'var(--green)' : score >= 50 ? 'var(--yellow, #e5a813)' : 'var(--red)'}
                            >
                              {score}%
                            </text>
                          </svg>
                        </div>
                      </div>

                      {/* Traffic light rule summary */}
                      <div className="flex gap-2">
                        {summary.pass > 0 && (
                          <div
                            className="flex items-center gap-1 text-[10px] px-2 py-1 rounded"
                            style={{ background: 'rgba(74,222,128,0.1)', color: 'var(--green)' }}
                          >
                            <span className="w-2 h-2 rounded-full inline-block" style={{ background: 'var(--green)' }} />
                            {summary.pass} pass
                          </div>
                        )}
                        {summary.warn > 0 && (
                          <div
                            className="flex items-center gap-1 text-[10px] px-2 py-1 rounded"
                            style={{ background: 'rgba(229,168,19,0.1)', color: 'var(--yellow, #e5a813)' }}
                          >
                            <span className="w-2 h-2 rounded-full inline-block" style={{ background: 'var(--yellow, #e5a813)' }} />
                            {summary.warn} warn
                          </div>
                        )}
                        {summary.fail > 0 && (
                          <div
                            className="flex items-center gap-1 text-[10px] px-2 py-1 rounded"
                            style={{ background: 'rgba(248,113,113,0.1)', color: 'var(--red)' }}
                          >
                            <span className="w-2 h-2 rounded-full inline-block" style={{ background: 'var(--red)' }} />
                            {summary.fail} fail
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* ============ Selected Set Detail ============ */}
              {selectedSet && (
                <Card>
                  {/* Set header */}
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3 flex-1 min-w-0">
                      {editingNameId === selectedSet.id ? (
                        <div className="flex items-center gap-2">
                          <input
                            type="text"
                            value={editNameValue}
                            onChange={(e) => setEditNameValue(e.target.value)}
                            onKeyDown={(e) => { if (e.key === 'Enter') handleSaveName(selectedSet.id); if (e.key === 'Escape') setEditingNameId(null); }}
                            autoFocus
                            className="px-2 py-1 rounded text-sm"
                            style={{ background: 'var(--bg-input)', border: '1px solid var(--accent)', color: 'var(--text-primary)', width: '220px' }}
                          />
                          <button
                            onClick={() => handleSaveName(selectedSet.id)}
                            className="text-xs px-2.5 py-1 rounded"
                            style={{ background: 'var(--accent)', color: 'white' }}
                          >
                            Save
                          </button>
                        </div>
                      ) : (
                        <h3
                          className="font-semibold text-base truncate cursor-pointer"
                          onDoubleClick={() => {
                            if (!selectedSet.isBuiltIn) {
                              setEditingNameId(selectedSet.id);
                              setEditNameValue(selectedSet.name);
                            }
                          }}
                          title={selectedSet.isBuiltIn ? undefined : 'Double-click to rename'}
                        >
                          {selectedSet.name}
                        </h3>
                      )}
                      {selectedSet.isBuiltIn && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded shrink-0" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                          Built-in
                        </span>
                      )}
                    </div>

                    <div className="flex items-center gap-2 shrink-0">
                      <button
                        onClick={() => runComplianceSimulator(selectedSet.id)}
                        className="px-3 py-1.5 rounded text-xs font-medium"
                        style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                        disabled={simRunning}
                      >
                        {simRunning && simulatorSetId === selectedSet.id ? 'Simulating...' : 'Run Compliance Check'}
                      </button>
                      {!selectedSet.isBuiltIn && (
                        <>
                          {deleteConfirmId === selectedSet.id ? (
                            <div className="flex items-center gap-2">
                              <span className="text-xs" style={{ color: 'var(--red)' }}>Delete?</span>
                              <button
                                onClick={() => handleDelete(selectedSet.id)}
                                className="text-xs px-2 py-1 rounded font-medium"
                                style={{ background: 'var(--red)', color: 'white' }}
                              >
                                Yes
                              </button>
                              <button
                                onClick={() => setDeleteConfirmId(null)}
                                className="text-xs px-2 py-1 rounded"
                                style={{ border: '1px solid var(--border)', color: 'var(--text-muted)' }}
                              >
                                No
                              </button>
                            </div>
                          ) : (
                            <button
                              onClick={() => setDeleteConfirmId(selectedSet.id)}
                              className="text-xs px-3 py-1.5 rounded"
                              style={{ color: 'var(--red)' }}
                              onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--red-muted)')}
                              onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                            >
                              Delete
                            </button>
                          )}
                        </>
                      )}
                    </div>
                  </div>

                  {/* Rules with visual progress bars */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                    {selectedSet.rules.map((rule) => {
                      const status = getRuleStatus(rule);
                      const progress = getRuleProgress(rule);
                      const category = getRuleCategory(rule.type);
                      const catInfo = RULE_CATEGORIES[category];
                      const isEditing = editingRuleId === rule.id;
                      const simResult = simResults && simulatorSetId === selectedSet.id ? simResults[rule.id] : null;

                      return (
                        <div
                          key={rule.id}
                          className="rounded-lg p-4"
                          style={{
                            background: 'var(--bg-input)',
                            border: simResult
                              ? `1px solid ${statusColor(simResult)}`
                              : '1px solid var(--border)',
                            transition: 'border-color 0.3s ease',
                          }}
                        >
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              {/* Category badge */}
                              <span
                                className="text-[10px] px-1.5 py-0.5 rounded font-semibold"
                                style={{ background: `${catInfo.color}20`, color: catInfo.color }}
                              >
                                {RULE_TYPE_OPTIONS.find(o => o.value === rule.type)?.icon || '?'}
                              </span>

                              {/* Status traffic light */}
                              <div
                                className="w-2.5 h-2.5 rounded-full"
                                style={{ background: statusColor(status) }}
                              />

                              <span className="text-sm font-medium">{rule.name}</span>
                            </div>

                            <div className="flex items-center gap-2">
                              {/* Simulator result badge */}
                              {simResult && (
                                <span
                                  className="text-[10px] px-2 py-0.5 rounded-full font-semibold uppercase"
                                  style={{
                                    background: `${statusColor(simResult)}20`,
                                    color: statusColor(simResult),
                                  }}
                                >
                                  {simResult}
                                </span>
                              )}

                              {/* Value (editable) */}
                              {isEditing ? (
                                <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
                                  <input
                                    type="number"
                                    value={editRuleValue}
                                    onChange={(e) => setEditRuleValue(e.target.value)}
                                    onKeyDown={(e) => { if (e.key === 'Enter') handleSaveRule(selectedSet.id, rule.id); if (e.key === 'Escape') setEditingRuleId(null); }}
                                    autoFocus
                                    className="w-20 px-2 py-1 rounded text-xs"
                                    style={{ background: 'var(--bg-card)', border: '1px solid var(--accent)', color: 'var(--text-primary)' }}
                                  />
                                  <button
                                    onClick={() => handleSaveRule(selectedSet.id, rule.id)}
                                    className="text-[10px] px-2 py-1 rounded"
                                    style={{ background: 'var(--accent)', color: 'white' }}
                                  >
                                    OK
                                  </button>
                                </div>
                              ) : (
                                <span
                                  className="text-sm font-semibold cursor-pointer"
                                  style={{ color: 'var(--text-primary)' }}
                                  onClick={() => handleStartEditRule(rule)}
                                  title="Click to edit limit"
                                >
                                  {formatRuleValue(rule.type, rule.value)}
                                </span>
                              )}

                              {/* Delete rule */}
                              <button
                                onClick={() => handleDeleteRule(selectedSet.id, rule.id)}
                                className="w-5 h-5 flex items-center justify-center rounded text-[10px] shrink-0 transition-colors"
                                style={{ color: 'var(--text-muted)' }}
                                onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--red)'; e.currentTarget.style.background = 'var(--red-muted)'; }}
                                onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-muted)'; e.currentTarget.style.background = 'transparent'; }}
                                title="Remove rule"
                              >
                                x
                              </button>
                            </div>
                          </div>

                          {/* Progress bar */}
                          <div className="flex items-center gap-3">
                            <div
                              className="flex-1 rounded-full"
                              style={{ height: 6, background: 'var(--bg-card)', overflow: 'hidden' }}
                            >
                              <div
                                className="rounded-full"
                                style={{
                                  height: '100%',
                                  width: `${progress}%`,
                                  background: statusColor(status),
                                  transition: 'width 0.5s ease, background 0.3s ease',
                                }}
                              />
                            </div>
                            <span className="text-[10px] shrink-0" style={{ color: 'var(--text-muted)', minWidth: 80, textAlign: 'right' }}>
                              {formatRuleValue(rule.type, rule.currentValue)} / {formatRuleValue(rule.type, rule.value)}
                            </span>
                          </div>

                          {/* Severity indicator */}
                          {status === 'warning' && (
                            <p className="text-[10px] mt-1.5" style={{ color: 'var(--yellow, #e5a813)' }}>
                              Approaching limit ({progress.toFixed(0)}% used)
                            </p>
                          )}
                          {status === 'fail' && (
                            <p className="text-[10px] mt-1.5" style={{ color: 'var(--red)' }}>
                              Limit exceeded! Current value has breached the rule.
                            </p>
                          )}
                        </div>
                      );
                    })}

                    {selectedSet.rules.length === 0 && (
                      <div className="text-center py-8 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No rules yet. Add a rule below.</p>
                      </div>
                    )}
                  </div>

                  {/* Add rule inline form */}
                  <div className="mt-4 pt-3 border-t" style={{ borderColor: 'var(--border)' }}>
                    {addingToSetId === selectedSet.id ? (
                      <div className="flex items-end gap-2 flex-wrap">
                        <div>
                          <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Rule Type</label>
                          <select
                            value={newRuleType}
                            onChange={(e) => setNewRuleType(e.target.value)}
                            className="px-2 py-1.5 rounded text-xs"
                            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                          >
                            {RULE_TYPE_OPTIONS.map((opt) => {
                              const cat = RULE_CATEGORIES[opt.category];
                              return (
                                <option key={opt.value} value={opt.value}>[{opt.icon}] {opt.label}</option>
                              );
                            })}
                          </select>
                        </div>
                        <div>
                          <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Limit Value</label>
                          <input
                            type="number"
                            value={newRuleValue}
                            onChange={(e) => setNewRuleValue(e.target.value)}
                            onKeyDown={(e) => { if (e.key === 'Enter') handleAddRule(selectedSet.id); }}
                            placeholder="e.g. 1000"
                            autoFocus
                            className="w-28 px-2 py-1.5 rounded text-xs"
                            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                          />
                        </div>
                        <button
                          onClick={() => handleAddRule(selectedSet.id)}
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
                      <button
                        onClick={() => { setAddingToSetId(selectedSet.id); setNewRuleValue(''); }}
                        className="text-xs px-3 py-1.5 rounded"
                        style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                      >
                        + Add Rule
                      </button>
                    )}
                  </div>
                </Card>
              )}

              {!selectedSet && sets.length > 0 && (
                <Card>
                  <div className="text-center py-12">
                    <p className="text-sm mb-1" style={{ color: 'var(--text-muted)' }}>Select a requirement set above to view and edit its rules.</p>
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Or browse templates to import a pre-built prop firm rule set.</p>
                  </div>
                </Card>
              )}
            </div>
          );
        }}
      </TabBar>

      {/* ============ Template Marketplace Modal ============ */}
      <Modal
        title="Rule Set Templates"
        isOpen={showTemplates}
        onClose={() => { setShowTemplates(false); setTemplateFilter(''); }}
        width="700px"
      >
        <div>
          {/* Search */}
          <input
            type="text"
            placeholder="Search templates (e.g. FTMO, TTP, Topstep...)"
            value={templateFilter}
            onChange={(e) => setTemplateFilter(e.target.value)}
            className="w-full px-3 py-2 rounded-lg text-sm mb-4"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', outline: 'none' }}
            autoFocus
          />

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {filteredTemplates.map(template => {
              const alreadyImported = sets.some(s => s.id === template.id);
              return (
                <div
                  key={template.id}
                  className="rounded-xl p-4"
                  style={{
                    background: 'var(--bg-input)',
                    border: '1px solid var(--border)',
                    opacity: alreadyImported ? 0.5 : 1,
                  }}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <div className="flex items-center gap-2 mb-0.5">
                        <h4 className="text-sm font-semibold">{template.name}</h4>
                        <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                          {template.accountSize}
                        </span>
                      </div>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{template.description}</p>
                    </div>
                    <button
                      onClick={() => handleImportTemplate(template)}
                      className="px-3 py-1.5 rounded text-xs font-medium shrink-0"
                      style={{
                        background: alreadyImported ? 'var(--bg-card)' : 'var(--accent)',
                        color: alreadyImported ? 'var(--text-muted)' : 'white',
                        cursor: alreadyImported ? 'not-allowed' : 'pointer',
                      }}
                      disabled={alreadyImported}
                    >
                      {alreadyImported ? 'Imported' : 'Import'}
                    </button>
                  </div>

                  {/* Popularity stars */}
                  <div className="flex items-center gap-2 mb-2">
                    <div className="flex gap-0.5">
                      {Array.from({ length: 5 }).map((_, i) => (
                        <span
                          key={i}
                          className="text-xs"
                          style={{ color: i < template.popularity ? 'var(--accent)' : 'var(--border)' }}
                        >
                          *
                        </span>
                      ))}
                    </div>
                    <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                      {template.rules.length} rules
                    </span>
                  </div>

                  {/* Rule preview chips */}
                  <div className="flex flex-wrap gap-1.5">
                    {template.rules.map(rule => {
                      const cat = getRuleCategory(rule.type);
                      const catInfo = RULE_CATEGORIES[cat];
                      return (
                        <span
                          key={rule.id}
                          className="text-[10px] px-2 py-0.5 rounded-full"
                          style={{ background: `${catInfo.color}15`, color: catInfo.color, border: `1px solid ${catInfo.color}30` }}
                        >
                          {rule.name}: {formatRuleValue(rule.type, rule.value)}
                        </span>
                      );
                    })}
                  </div>
                </div>
              );
            })}

            {filteredTemplates.length === 0 && (
              <p className="text-sm text-center py-4" style={{ color: 'var(--text-muted)' }}>
                No templates match your search.
              </p>
            )}
          </div>
        </div>
      </Modal>
    </div>
  );
}
