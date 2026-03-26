'use client';
/**
 * Portfolio Requirements — V5 locked design. Expand/collapse sets with rule badges,
 * TQ rules (orange accent), Clone/Delete actions, + New Set.
 */
import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';


interface Rule {
  id: string;
  name: string;
  type: string;
  value: number;
  current_value?: number;
  description?: string;
}

interface TradeQualRule {
  id: string;
  name: string;
  type: string;
  value: number;
  unit: string;
  applies_to: string;
  description?: string;
}

interface RequirementSet {
  id: string;
  name: string;
  firm_key?: string;
  is_built_in: boolean;
  rules: Rule[];
  trade_qual_rules?: TradeQualRule[];
  used_by?: string[];
}


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


function useRequirements() {
  return useQuery({
    queryKey: ['requirements'],
    queryFn: () => apiFetch<RequirementSet[]>('/api/requirements'),
  });
}

function useSaveRequirements() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (sets: RequirementSet[]) =>
      apiFetch('/api/requirements', { method: 'PUT', body: JSON.stringify(sets) }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['requirements'] }); },
  });
}


function getRuleStatus(rule: Rule): 'pass' | 'fail' | 'warning' {
  if (rule.current_value === undefined || rule.value === 0) return 'pass';
  const pct = (rule.current_value / rule.value) * 100;
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

function formatRuleValue(type: string, value: number): string {
  const pctTypes = ['max_daily_loss_pct', 'max_total_drawdown_pct', 'daily_pause_pct', 'min_profit_pct', 'min_profitable_days'];
  if (pctTypes.includes(type)) return `${value}%`;
  return value.toString();
}

function getSetSummary(set: RequirementSet): { pass: number; fail: number; warn: number } {
  let pass = 0, fail = 0, warn = 0;
  (set.rules || []).forEach((r) => {
    const s = getRuleStatus(r);
    if (s === 'pass') pass++;
    else if (s === 'fail') fail++;
    else warn++;
  });
  return { pass, fail, warn };
}


export default function RequirementsPage() {
  const { data: sets, isLoading, error } = useRequirements();
  const saveMutation = useSaveRequirements();

  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [editingRuleId, setEditingRuleId] = useState<string | null>(null);
  const [editRuleValue, setEditRuleValue] = useState('');
  const [addingToSetId, setAddingToSetId] = useState<string | null>(null);
  const [newRuleType, setNewRuleType] = useState(RULE_TYPE_OPTIONS[0].value);
  const [newRuleValue, setNewRuleValue] = useState('');
  const [addingTQToSetId, setAddingTQToSetId] = useState<string | null>(null);
  const [newTQType, setNewTQType] = useState(TRADE_QUAL_OPTIONS[0].value);
  const [newTQValue, setNewTQValue] = useState('');
  const [newTQAppliesTo, setNewTQAppliesTo] = useState<string>('wins');
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);


  if (isLoading) {
    return (
      <div>
        <PageHeader title="Portfolio Requirements" subtitle="Loading..." />
        <div className="space-y-3 mt-4">
          {[1, 2, 3].map((i) => (
            <Card key={i}>
              <div className="animate-pulse space-y-3">
                <div className="h-4 rounded w-1/4" style={{ background: 'var(--border)' }} />
                <div className="h-3 rounded w-1/2" style={{ background: 'var(--border)' }} />
                <div className="flex gap-2">
                  {[1, 2, 3].map((j) => (
                    <div key={j} className="h-6 w-12 rounded" style={{ background: 'var(--border)' }} />
                  ))}
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div>
        <PageHeader title="Portfolio Requirements" subtitle="Error" />
        <Card>
          <div className="text-center py-8" style={{ color: 'var(--red)' }}>
            Failed to load requirement sets. Check your connection and try again.
          </div>
        </Card>
      </div>
    );
  }

  const requirementSets = sets || [];

  return (
    <div>
      <PageHeader
        title="Portfolio Requirements"
        subtitle="Manage requirement sets and rules"
        actions={
          <div className="flex items-center gap-2">
            <span
              className="text-xs px-2 py-1 rounded-full flex items-center gap-1.5"
              style={{ background: 'var(--green)' + '15', color: 'var(--green)' }}
            >
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--green)' }} />
              Live
            </span>
            <button
              className="px-4 py-2 rounded-lg text-sm font-medium"
              style={{ background: 'var(--accent)', color: 'white', cursor: 'pointer' }}
            >
              + New Set
            </button>
          </div>
        }
      />

      <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
        {requirementSets.length} requirement set{requirementSets.length !== 1 ? 's' : ''}
      </p>

      {/* Empty state */}
      {requirementSets.length === 0 && (
        <Card>
          <div className="text-center py-12">
            <p className="text-lg mb-2" style={{ color: 'var(--text-secondary)' }}>
              No requirement sets found.
            </p>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              Requirement sets define prop firm rules and compliance criteria for your portfolios.
            </p>
          </div>
        </Card>
      )}

      {/* Requirement set cards */}
      <div className="space-y-3">
        {requirementSets.map((set) => {
          const summary = getSetSummary(set);
          const isExpanded = expandedId === set.id;
          const totalRules = (set.rules || []).length;
          const tqRules = set.trade_qual_rules || [];
          const tqCount = tqRules.length;
          const usedByList = set.used_by || [];

          return (
            <Card key={set.id}>
              {/* Set header — V5 style with triangle expand indicator */}
              <div
                className="flex items-center justify-between cursor-pointer"
                onClick={() => setExpandedId(isExpanded ? null : set.id)}
              >
                <div className="flex items-center gap-3 flex-1 min-w-0">
                  {/* Expand indicator */}
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    {isExpanded ? '\u25BC' : '\u25B6'}
                  </span>

                  <h3 className="font-semibold text-sm truncate">{set.name}</h3>

                  {set.is_built_in && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded shrink-0" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                      Built-in
                    </span>
                  )}
                  {set.firm_key && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded shrink-0" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                      {set.firm_key.toUpperCase()}
                    </span>
                  )}
                  {tqCount > 0 && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded shrink-0" style={{ background: 'rgba(255,152,0,0.12)', color: 'var(--orange)' }}>
                      {tqCount} TQ rule{tqCount !== 1 ? 's' : ''}
                    </span>
                  )}
                </div>

                {/* Summary badges */}
                <div className="flex items-center gap-3 shrink-0">
                  {totalRules > 0 && (
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
                  {usedByList.length > 0 && (
                    <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                      {usedByList.length} portfolio{usedByList.length !== 1 ? 's' : ''}
                    </span>
                  )}
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    {totalRules} rule{totalRules !== 1 ? 's' : ''}
                  </span>
                </div>
              </div>

              {/* Expanded content — V5 style: rule badges, TQ rules, actions */}
              {isExpanded && (
                <div className="mt-4 pt-3 border-t" style={{ borderColor: 'var(--border)' }}>
                  {/* Rule badges (V5 inline pill display) */}
                  <div className="flex flex-wrap gap-2 mb-3">
                    {(set.rules || []).map((rule) => {
                      const rStatus = getRuleStatus(rule);
                      const isEditing = editingRuleId === rule.id;

                      return (
                        <div
                          key={rule.id}
                          className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs"
                          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
                        >
                          <span
                            className="w-2 h-2 rounded-full shrink-0"
                            style={{ background: statusDot(rStatus) }}
                          />
                          <span style={{ color: 'var(--text-secondary)' }}>{rule.name}:</span>

                          {isEditing ? (
                            <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
                              <input
                                type="number"
                                value={editRuleValue}
                                onChange={(e) => setEditRuleValue(e.target.value)}
                                onKeyDown={(e) => {
                                  if (e.key === 'Enter') setEditingRuleId(null);
                                  if (e.key === 'Escape') setEditingRuleId(null);
                                }}
                                autoFocus
                                className="w-20 px-1.5 py-0.5 rounded text-xs"
                                style={{ background: 'var(--bg-card)', border: '1px solid var(--accent)', color: 'var(--text-primary)' }}
                              />
                              <button
                                onClick={() => setEditingRuleId(null)}
                                className="text-[10px] px-1.5 py-0.5 rounded"
                                style={{ background: 'var(--accent)', color: 'white', cursor: 'pointer' }}
                              >
                                OK
                              </button>
                            </div>
                          ) : (
                            <span
                              className="font-medium"
                              style={{ color: 'var(--text-primary)', cursor: set.is_built_in ? 'default' : 'pointer' }}
                              onClick={(e) => {
                                if (set.is_built_in) return;
                                e.stopPropagation();
                                setEditingRuleId(rule.id);
                                setEditRuleValue(rule.value.toString());
                              }}
                              title={set.is_built_in ? 'Clone to edit' : 'Click to edit'}
                            >
                              {rule.current_value !== undefined
                                ? `${rule.current_value} / ${formatRuleValue(rule.type, rule.value)}`
                                : formatRuleValue(rule.type, rule.value)
                              }
                            </span>
                          )}

                          {rule.description && (
                            <span className="text-[10px] hidden sm:inline" style={{ color: 'var(--text-muted)' }} title={rule.description}>i</span>
                          )}
                        </div>
                      );
                    })}

                    {(set.rules || []).length === 0 && (
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>No rules yet.</p>
                    )}
                  </div>

                  {/* Add rule (V5 — not on built-in sets) */}
                  {!set.is_built_in && addingToSetId === set.id ? (
                    <div className="flex items-end gap-2 flex-wrap mb-3" onClick={(e) => e.stopPropagation()}>
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
                          placeholder="e.g. 5"
                          autoFocus
                          className="w-24 px-2 py-1.5 rounded text-xs"
                          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                        />
                      </div>
                      <button
                        className="px-3 py-1.5 rounded text-xs font-medium"
                        style={{ background: 'var(--accent)', color: 'white', opacity: newRuleValue ? 1 : 0.5, cursor: 'pointer' }}
                        disabled={!newRuleValue}
                      >
                        Add
                      </button>
                      <button
                        onClick={() => setAddingToSetId(null)}
                        className="px-3 py-1.5 rounded text-xs"
                        style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-muted)', cursor: 'pointer' }}
                      >
                        Cancel
                      </button>
                    </div>
                  ) : !set.is_built_in ? (
                    <div className="mb-3">
                      <button
                        onClick={(e) => { e.stopPropagation(); setAddingToSetId(set.id); setNewRuleValue(''); }}
                        className="text-xs px-3 py-1.5 rounded"
                        style={{ background: 'var(--accent-muted)', color: 'var(--accent)', cursor: 'pointer' }}
                      >
                        + Add Compliance Rule
                      </button>
                    </div>
                  ) : null}

                  {/* Trade Qualification Rules (V5 — orange accent section) */}
                  <div className="mt-3 pt-3 border-t" style={{ borderColor: 'var(--border)' }}>
                    <div className="flex items-center gap-2 mb-2">
                      <h4 className="text-xs font-medium" style={{ color: 'var(--orange)' }}>Trade Qualification Rules</h4>
                      <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                        Trades not meeting these criteria may not count toward prop firm P&L
                      </span>
                    </div>

                    {tqRules.length > 0 ? (
                      <div className="flex flex-wrap gap-2 mb-2">
                        {tqRules.map((tq) => {
                          const appliesToColors: Record<string, { color: string; label: string }> = {
                            wins: { color: 'var(--green)', label: 'Wins only' },
                            losses: { color: 'var(--red)', label: 'Losses only' },
                            all: { color: 'var(--text-muted)', label: 'All trades' },
                          };
                          const at = appliesToColors[tq.applies_to] || appliesToColors.all;
                          return (
                            <div
                              key={tq.id}
                              className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs"
                              style={{ background: 'rgba(255,152,0,0.08)', border: '1px solid rgba(255,152,0,0.2)' }}
                            >
                              <span style={{ color: 'var(--orange)' }}>{tq.name}:</span>
                              <span className="font-medium" style={{ color: 'var(--text-primary)' }}>
                                {tq.unit === '$' ? `$${tq.value}` : `${tq.value} ${tq.unit}`}
                              </span>
                              <span
                                className="text-[10px] px-1.5 py-0.5 rounded-full"
                                style={{ color: at.color, background: at.color + '18' }}
                              >
                                {at.label}
                              </span>
                              {tq.description && (
                                <span className="text-[10px]" style={{ color: 'var(--text-muted)' }} title={tq.description}>i</span>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>
                        No trade qualification rules. All trades count.
                      </p>
                    )}

                    {/* Add TQ rule (V5 — not on built-in sets) */}
                    {!set.is_built_in && addingTQToSetId === set.id ? (
                      <div className="flex items-end gap-2 flex-wrap" onClick={(e) => e.stopPropagation()}>
                        <div>
                          <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Type</label>
                          <select
                            value={newTQType}
                            onChange={(e) => setNewTQType(e.target.value)}
                            className="px-2 py-1.5 rounded text-xs"
                            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                          >
                            {TRADE_QUAL_OPTIONS.map((opt) => (
                              <option key={opt.value} value={opt.value}>{opt.label} ({opt.unit})</option>
                            ))}
                          </select>
                        </div>
                        <div>
                          <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Value</label>
                          <input
                            type="number"
                            value={newTQValue}
                            onChange={(e) => setNewTQValue(e.target.value)}
                            placeholder="e.g. 30"
                            autoFocus
                            className="w-24 px-2 py-1.5 rounded text-xs"
                            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                          />
                        </div>
                        <div>
                          <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Applies To</label>
                          <select
                            value={newTQAppliesTo}
                            onChange={(e) => setNewTQAppliesTo(e.target.value)}
                            className="px-2 py-1.5 rounded text-xs"
                            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                          >
                            <option value="wins">Wins only</option>
                            <option value="losses">Losses only</option>
                            <option value="all">All trades</option>
                          </select>
                        </div>
                        <button
                          className="px-3 py-1.5 rounded text-xs font-medium"
                          style={{ background: 'var(--orange)', color: 'white', opacity: newTQValue ? 1 : 0.5, cursor: 'pointer' }}
                          disabled={!newTQValue}
                        >
                          Add
                        </button>
                        <button
                          onClick={() => setAddingTQToSetId(null)}
                          className="px-3 py-1.5 rounded text-xs"
                          style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-muted)', cursor: 'pointer' }}
                        >
                          Cancel
                        </button>
                      </div>
                    ) : !set.is_built_in ? (
                      <button
                        onClick={(e) => { e.stopPropagation(); setAddingTQToSetId(set.id); setNewTQValue(''); }}
                        className="text-xs px-3 py-1.5 rounded"
                        style={{ background: 'rgba(255,152,0,0.12)', color: 'var(--orange)', cursor: 'pointer' }}
                      >
                        + Add Qualification Rule
                      </button>
                    ) : null}
                  </div>

                  {/* Action row (V5 — Clone / Delete) */}
                  <div className="flex items-center gap-2 mt-3 pt-3 border-t" style={{ borderColor: 'var(--border)' }}>
                    <button
                      className="text-xs px-3 py-1.5 rounded"
                      style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)', cursor: 'pointer' }}
                    >
                      Clone
                    </button>

                    {!set.is_built_in && (
                      <>
                        {deleteConfirmId === set.id ? (
                          <div className="flex items-center gap-2 ml-auto">
                            <span className="text-xs" style={{ color: 'var(--red)' }}>Delete this set?</span>
                            <button
                              onClick={(e) => { e.stopPropagation(); setDeleteConfirmId(null); }}
                              className="text-xs px-2 py-1 rounded font-medium"
                              style={{ background: 'var(--red)', color: 'white', cursor: 'pointer' }}
                            >
                              Yes
                            </button>
                            <button
                              onClick={(e) => { e.stopPropagation(); setDeleteConfirmId(null); }}
                              className="text-xs px-2 py-1 rounded"
                              style={{ border: '1px solid var(--border)', color: 'var(--text-muted)', cursor: 'pointer' }}
                            >
                              No
                            </button>
                          </div>
                        ) : (
                          <button
                            onClick={(e) => { e.stopPropagation(); setDeleteConfirmId(set.id); }}
                            className="text-xs px-3 py-1.5 rounded ml-auto"
                            style={{ color: 'var(--red)', cursor: 'pointer' }}
                          >
                            Delete Set
                          </button>
                        )}
                      </>
                    )}

                    {set.is_built_in && (
                      <span className="text-[10px] ml-auto" style={{ color: 'var(--text-muted)' }}>
                        Built-in sets are read-only. Clone to customize.
                      </span>
                    )}
                  </div>

                  {/* Used by */}
                  {usedByList.length > 0 && (
                    <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
                      Used by: {usedByList.join(', ')}
                    </p>
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
