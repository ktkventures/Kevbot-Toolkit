'use client';

/**
 * Portfolio Requirements — Clean API-first page.
 *
 * Visual design derived from V5 (versions/V5.tsx), data layer built
 * around the requirements API endpoint. No mock data.
 */

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

// ---------------------------------------------------------------------------
// Types — match expected API shapes
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Inline hooks (no dedicated hook file yet)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

function statusColor(status: 'pass' | 'fail' | 'warning'): string {
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

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function RequirementsPage() {
  const { data: sets, isLoading, error } = useRequirements();
  const [expandedId, setExpandedId] = useState<string | null>(null);

  // ---------------------------------------------------------------------------
  // Loading / Error / Empty states
  // ---------------------------------------------------------------------------

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
        subtitle={`${requirementSets.length} requirement set${requirementSets.length === 1 ? '' : 's'} — prop firm rules and custom compliance`}
        actions={
          <div className="flex items-center gap-2">
            <span
              className="text-xs px-2 py-1 rounded-full flex items-center gap-1.5"
              style={{ background: 'var(--green)' + '15', color: 'var(--green)' }}
            >
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--green)' }} />
              Live
            </span>
          </div>
        }
      />

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
      <div className="space-y-3 mt-4">
        {requirementSets.map((set) => {
          const summary = getSetSummary(set);
          const isExpanded = expandedId === set.id;
          const totalRules = (set.rules || []).length;
          const tqCount = (set.trade_qual_rules || []).length;
          const usedByCount = (set.used_by || []).length;

          return (
            <Card key={set.id}>
              {/* Header row */}
              <div
                className="flex items-center justify-between cursor-pointer"
                onClick={() => setExpandedId(isExpanded ? null : set.id)}
              >
                <div className="flex items-center gap-3">
                  <h3 className="font-semibold text-sm">{set.name}</h3>
                  {set.is_built_in && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}>
                      Built-in
                    </span>
                  )}
                  {set.firm_key && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                      {set.firm_key.toUpperCase()}
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-4">
                  {/* Pass/fail summary */}
                  <div className="flex items-center gap-2">
                    {summary.pass > 0 && (
                      <span className="flex items-center gap-1 text-xs">
                        <span className="w-2 h-2 rounded-full" style={{ background: 'var(--green)' }} />
                        {summary.pass}
                      </span>
                    )}
                    {summary.warn > 0 && (
                      <span className="flex items-center gap-1 text-xs">
                        <span className="w-2 h-2 rounded-full" style={{ background: 'var(--yellow, #e5a813)' }} />
                        {summary.warn}
                      </span>
                    )}
                    {summary.fail > 0 && (
                      <span className="flex items-center gap-1 text-xs">
                        <span className="w-2 h-2 rounded-full" style={{ background: 'var(--red)' }} />
                        {summary.fail}
                      </span>
                    )}
                  </div>

                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    {totalRules} rule{totalRules !== 1 ? 's' : ''}
                    {tqCount > 0 && ` + ${tqCount} TQ`}
                    {usedByCount > 0 && ` | Used by ${usedByCount}`}
                  </span>

                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    {isExpanded ? 'v' : '>'}
                  </span>
                </div>
              </div>

              {/* Expanded: rule list */}
              {isExpanded && (
                <div className="mt-4 space-y-2">
                  {(set.rules || []).map((rule) => {
                    const rStatus = getRuleStatus(rule);
                    return (
                      <div
                        key={rule.id}
                        className="flex items-center justify-between py-2 px-3 rounded-lg"
                        style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
                      >
                        <div className="flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: statusColor(rStatus) }} />
                          <span className="text-sm">{rule.name}</span>
                          {rule.description && (
                            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                              — {rule.description}
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-3 flex-shrink-0">
                          {rule.current_value !== undefined && (
                            <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
                              {rule.current_value} / {formatRuleValue(rule.type, rule.value)}
                            </span>
                          )}
                          {rule.current_value === undefined && (
                            <span className="text-xs font-mono font-medium" style={{ color: 'var(--text-primary)' }}>
                              {formatRuleValue(rule.type, rule.value)}
                            </span>
                          )}
                        </div>
                      </div>
                    );
                  })}

                  {/* Trade qualification rules */}
                  {(set.trade_qual_rules || []).length > 0 && (
                    <>
                      <div className="mt-3 mb-1">
                        <span className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>
                          Trade Qualification Rules
                        </span>
                      </div>
                      {set.trade_qual_rules!.map((tq) => (
                        <div
                          key={tq.id}
                          className="flex items-center justify-between py-2 px-3 rounded-lg"
                          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
                        >
                          <div className="flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: 'var(--accent)' }} />
                            <span className="text-sm">{tq.name}</span>
                            <span className="text-xs px-1.5 py-0.5 rounded" style={{ background: 'var(--bg-secondary)', color: 'var(--text-muted)' }}>
                              {tq.applies_to}
                            </span>
                          </div>
                          <span className="text-xs font-mono" style={{ color: 'var(--text-primary)' }}>
                            {tq.value} {tq.unit}
                          </span>
                        </div>
                      ))}
                    </>
                  )}

                  {/* Used by */}
                  {usedByCount > 0 && (
                    <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
                      Used by: {(set.used_by || []).join(', ')}
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
