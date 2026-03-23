'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import Modal from '@/components/Modal';

// --- Types ---
interface Rule {
  id: string;
  name: string;
  type: string;
  value: number;
  threshold: number | null;
}

interface RequirementSet {
  id: string;
  name: string;
  isBuiltIn: boolean;
  rules: Rule[];
}

// --- Rule type options ---
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

// --- Mock data ---
const initialRequirementSets: RequirementSet[] = [
  {
    id: 'ttp-50k',
    name: 'TTP 50k Evaluation',
    isBuiltIn: true,
    rules: [
      { id: 'r1', name: 'Max Daily Loss', type: 'max_daily_loss', value: 1000, threshold: null },
      { id: 'r2', name: 'Max Total Drawdown', type: 'max_total_drawdown', value: 2500, threshold: null },
      { id: 'r3', name: 'Min Trading Days', type: 'min_trading_days', value: 5, threshold: null },
      { id: 'r4', name: 'Max Daily Trades', type: 'max_daily_trades', value: 15, threshold: null },
      { id: 'r5', name: 'Profit Target', type: 'profit_target', value: 3000, threshold: null },
    ],
  },
  {
    id: 'ftmo-100k',
    name: 'FTMO 100k Challenge',
    isBuiltIn: true,
    rules: [
      { id: 'r1', name: 'Max Daily Loss', type: 'max_daily_loss', value: 5000, threshold: null },
      { id: 'r2', name: 'Max Total Drawdown', type: 'max_total_drawdown', value: 10000, threshold: null },
      { id: 'r3', name: 'Min Trading Days', type: 'min_trading_days', value: 4, threshold: null },
      { id: 'r4', name: 'Min Profitable Days', type: 'min_profitable_days', value: 60, threshold: 50 },
      { id: 'r5', name: 'Profit Target', type: 'profit_target', value: 10000, threshold: null },
      { id: 'r6', name: 'Max Position Size', type: 'max_position_size', value: 10, threshold: null },
    ],
  },
  {
    id: 'custom-1',
    name: 'My Custom Rules',
    isBuiltIn: false,
    rules: [
      { id: 'r1', name: 'Max Daily Loss', type: 'max_daily_loss', value: 500, threshold: null },
      { id: 'r2', name: 'Max Daily Risk %', type: 'max_daily_risk_pct', value: 2.0, threshold: null },
    ],
  },
];

// --- Helpers ---
function formatRuleValue(rule: Rule): string {
  const dollarTypes = ['max_daily_loss', 'max_total_drawdown', 'profit_target'];
  const pctTypes = ['min_profitable_days', 'max_daily_risk_pct'];
  if (dollarTypes.includes(rule.type)) {
    return `$${rule.value.toLocaleString()}`;
  }
  if (pctTypes.includes(rule.type)) {
    return `${rule.value}%`;
  }
  return rule.value.toString();
}

function generateId(): string {
  return 'id-' + Math.random().toString(36).substring(2, 9);
}

// --- Component ---
export default function PortfolioRequirementsV1() {
  const [sets, setSets] = useState<RequirementSet[]>(initialRequirementSets);
  const [view, setView] = useState<'list' | 'editor'>('list');
  const [editingSet, setEditingSet] = useState<RequirementSet | null>(null);
  const [isNewSet, setIsNewSet] = useState(false);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  // Add Rule form state
  const [newRuleName, setNewRuleName] = useState('');
  const [newRuleType, setNewRuleType] = useState(RULE_TYPE_OPTIONS[0].value);
  const [newRuleValue, setNewRuleValue] = useState('');
  const [newRuleThreshold, setNewRuleThreshold] = useState('');
  const [addRuleExpanded, setAddRuleExpanded] = useState(false);

  // --- Handlers ---
  function handleNewSet() {
    const newSet: RequirementSet = {
      id: generateId(),
      name: '',
      isBuiltIn: false,
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
      if (exists) {
        return prev.map((s) => (s.id === editingSet.id ? editingSet : s));
      }
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
    };
    setEditingSet({
      ...editingSet,
      rules: [...editingSet.rules, rule],
    });
    resetAddRuleForm();
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
                style={{
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border)',
                  color: 'var(--text-secondary)',
                }}
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
          <h3
            className="text-sm font-medium mb-4"
            style={{ color: 'var(--text-secondary)' }}
          >
            Details
          </h3>
          <div>
            <label
              className="block text-xs mb-1.5"
              style={{ color: 'var(--text-muted)' }}
            >
              Set Name
            </label>
            <input
              type="text"
              value={editingSet.name}
              onChange={(e) =>
                setEditingSet({ ...editingSet, name: e.target.value })
              }
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
            <h3
              className="text-sm font-medium"
              style={{ color: 'var(--text-secondary)' }}
            >
              Rules
            </h3>
            <span
              className="text-xs"
              style={{ color: 'var(--text-muted)' }}
            >
              {editingSet.rules.length} rule{editingSet.rules.length !== 1 ? 's' : ''}
            </span>
          </div>

          {editingSet.rules.length === 0 ? (
            <div
              className="text-center py-8 rounded-lg"
              style={{ background: 'var(--bg-input)' }}
            >
              <p
                className="text-sm"
                style={{ color: 'var(--text-muted)' }}
              >
                No rules yet. Add a rule below to get started.
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {editingSet.rules.map((rule) => (
                <div
                  key={rule.id}
                  className="flex items-center justify-between py-2.5 px-3 rounded-lg"
                  style={{ background: 'var(--bg-input)' }}
                >
                  <div className="flex items-center gap-3 flex-1 min-w-0">
                    <span className="text-sm font-medium truncate">
                      {rule.name}
                    </span>
                    <span
                      className="text-xs px-2 py-0.5 rounded shrink-0"
                      style={{
                        background: 'var(--bg-card)',
                        color: 'var(--text-muted)',
                        border: '1px solid var(--border)',
                      }}
                    >
                      {RULE_TYPE_LABELS[rule.type] || rule.type}
                    </span>
                    <span
                      className="text-sm font-medium shrink-0"
                      style={{ color: 'var(--accent)' }}
                    >
                      {formatRuleValue(rule)}
                    </span>
                    {rule.threshold !== null && (
                      <span
                        className="text-xs shrink-0"
                        style={{ color: 'var(--text-muted)' }}
                      >
                        ({rule.threshold}% threshold)
                      </span>
                    )}
                  </div>
                  {!isBuiltIn && (
                    <button
                      onClick={() => handleRemoveRule(rule.id)}
                      className="ml-3 w-6 h-6 rounded flex items-center justify-center text-xs shrink-0"
                      style={{
                        color: 'var(--red)',
                        background: 'transparent',
                      }}
                      onMouseEnter={(e) =>
                        (e.currentTarget.style.background = 'var(--red-muted)')
                      }
                      onMouseLeave={(e) =>
                        (e.currentTarget.style.background = 'transparent')
                      }
                    >
                      x
                    </button>
                  )}
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
              <h3
                className="text-sm font-medium"
                style={{ color: 'var(--text-secondary)' }}
              >
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
                <div className="grid grid-cols-2 gap-4 mb-4">
                  {/* Left column */}
                  <div className="space-y-3">
                    <div>
                      <label
                        className="block text-xs mb-1.5"
                        style={{ color: 'var(--text-muted)' }}
                      >
                        Rule Name
                      </label>
                      <input
                        type="text"
                        value={newRuleName}
                        onChange={(e) => setNewRuleName(e.target.value)}
                        placeholder="e.g. Max Daily Loss"
                        className="w-full px-3 py-2 rounded-lg text-sm"
                        style={{
                          background: 'var(--bg-input)',
                          border: '1px solid var(--border)',
                          color: 'var(--text-primary)',
                        }}
                      />
                    </div>
                    <div>
                      <label
                        className="block text-xs mb-1.5"
                        style={{ color: 'var(--text-muted)' }}
                      >
                        Rule Type
                      </label>
                      <select
                        value={newRuleType}
                        onChange={(e) => setNewRuleType(e.target.value)}
                        className="w-full px-3 py-2 rounded-lg text-sm"
                        style={{
                          background: 'var(--bg-input)',
                          border: '1px solid var(--border)',
                          color: 'var(--text-primary)',
                        }}
                      >
                        {RULE_TYPE_OPTIONS.map((opt) => (
                          <option key={opt.value} value={opt.value}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>

                  {/* Right column */}
                  <div className="space-y-3">
                    <div>
                      <label
                        className="block text-xs mb-1.5"
                        style={{ color: 'var(--text-muted)' }}
                      >
                        Value
                      </label>
                      <input
                        type="number"
                        value={newRuleValue}
                        onChange={(e) => setNewRuleValue(e.target.value)}
                        placeholder="e.g. 1000"
                        className="w-full px-3 py-2 rounded-lg text-sm"
                        style={{
                          background: 'var(--bg-input)',
                          border: '1px solid var(--border)',
                          color: 'var(--text-primary)',
                        }}
                      />
                    </div>
                    {newRuleType === 'min_profitable_days' && (
                      <div>
                        <label
                          className="block text-xs mb-1.5"
                          style={{ color: 'var(--text-muted)' }}
                        >
                          Threshold (%)
                        </label>
                        <input
                          type="number"
                          value={newRuleThreshold}
                          onChange={(e) => setNewRuleThreshold(e.target.value)}
                          placeholder="e.g. 50"
                          className="w-full px-3 py-2 rounded-lg text-sm"
                          style={{
                            background: 'var(--bg-input)',
                            border: '1px solid var(--border)',
                            color: 'var(--text-primary)',
                          }}
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
                    opacity:
                      newRuleName.trim() && newRuleValue ? 1 : 0.5,
                    cursor:
                      newRuleName.trim() && newRuleValue
                        ? 'pointer'
                        : 'not-allowed',
                  }}
                  disabled={!newRuleName.trim() || !newRuleValue}
                >
                  + Add Rule
                </button>
              </div>
            )}
          </Card>
        )}

        {/* Bottom action bar (mobile-friendly duplicate) */}
        <div
          className="flex justify-end gap-3 mt-6 pt-4 border-t"
          style={{ borderColor: 'var(--border)' }}
        >
          <button
            onClick={handleCancel}
            className="px-4 py-2 rounded-lg text-sm"
            style={{
              background: 'var(--bg-card)',
              border: '1px solid var(--border)',
              color: 'var(--text-secondary)',
            }}
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
  return (
    <div>
      <PageHeader
        title="Portfolio Requirements"
        subtitle="Prop firm rules and custom requirement sets"
        actions={
          <button
            onClick={handleNewSet}
            className="px-4 py-2 rounded-lg text-sm font-medium"
            style={{ background: 'var(--accent)', color: 'white' }}
          >
            + New Requirement Set
          </button>
        }
      />

      <p
        className="text-sm mb-4"
        style={{ color: 'var(--text-muted)' }}
      >
        {sets.length} requirement set{sets.length !== 1 ? 's' : ''}
      </p>

      <div className="space-y-4">
        {sets.map((set) => (
          <Card key={set.id}>
            <div className="flex items-start justify-between gap-4">
              {/* Left side */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <h3 className="font-semibold text-base truncate">
                    {set.name}
                  </h3>
                  {set.isBuiltIn && (
                    <span
                      className="text-xs px-2 py-0.5 rounded-full shrink-0"
                      style={{
                        background: 'var(--blue-muted)',
                        color: 'var(--blue)',
                      }}
                    >
                      Built-in
                    </span>
                  )}
                </div>
                <p
                  className="text-xs mb-3"
                  style={{ color: 'var(--text-muted)' }}
                >
                  {set.rules.length} rule{set.rules.length !== 1 ? 's' : ''}
                </p>

                {/* Rules preview */}
                <div className="space-y-1.5">
                  {set.rules.slice(0, 4).map((rule) => (
                    <div
                      key={rule.id}
                      className="flex items-center gap-2 text-sm"
                    >
                      <span
                        className="w-1.5 h-1.5 rounded-full shrink-0"
                        style={{ background: 'var(--accent)' }}
                      />
                      <span style={{ color: 'var(--text-secondary)' }}>
                        {rule.name}
                      </span>
                      <span
                        style={{ color: 'var(--text-muted)' }}
                      >
                        —
                      </span>
                      <span className="font-medium">
                        {formatRuleValue(rule)}
                      </span>
                      {rule.threshold !== null && (
                        <span
                          className="text-xs"
                          style={{ color: 'var(--text-muted)' }}
                        >
                          ({rule.threshold}% threshold)
                        </span>
                      )}
                    </div>
                  ))}
                  {set.rules.length > 4 && (
                    <p
                      className="text-xs mt-1"
                      style={{ color: 'var(--text-muted)' }}
                    >
                      ... and {set.rules.length - 4} more
                    </p>
                  )}
                </div>
              </div>

              {/* Right side: action buttons */}
              <div className="flex flex-col gap-2 shrink-0">
                <button
                  onClick={() => handleEdit(set)}
                  className="px-3 py-1.5 rounded text-xs"
                  style={{
                    background: 'var(--bg-input)',
                    border: '1px solid var(--border)',
                    color: 'var(--text-secondary)',
                  }}
                >
                  Edit
                </button>
                <button
                  onClick={() => handleClone(set)}
                  className="px-3 py-1.5 rounded text-xs"
                  style={{
                    background: 'var(--bg-input)',
                    border: '1px solid var(--border)',
                    color: 'var(--text-secondary)',
                  }}
                >
                  Clone
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
                    style={{
                      background: 'var(--red-muted)',
                      border: '1px solid transparent',
                      color: 'var(--red)',
                    }}
                  >
                    Delete
                  </button>
                )}
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* Delete Confirmation Modal */}
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
              <p className="text-sm mb-6" style={{ color: 'var(--text-secondary)' }}>
                Delete &ldquo;{target.name}&rdquo;? This cannot be undone.
              </p>
              <div className="flex justify-end gap-3">
                <button
                  onClick={() => setDeleteConfirmId(null)}
                  className="px-4 py-2 rounded-lg text-sm"
                  style={{
                    background: 'var(--bg-card)',
                    border: '1px solid var(--border)',
                    color: 'var(--text-secondary)',
                  }}
                >
                  Cancel
                </button>
                <button
                  onClick={() => handleDelete(deleteConfirmId)}
                  className="px-4 py-2 rounded-lg text-sm font-medium"
                  style={{
                    background: 'var(--red)',
                    color: 'white',
                  }}
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
