'use client';

/**
 * Mass Results — Clean API-first page.
 *
 * Visual design derived from V5 (versions/V5.tsx), data layer built
 * around the mass builder results API endpoint. No mock data.
 */

import { useState, useMemo } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import { useMassResults, useDeleteMassResult, useCancelMassSearch } from '@/hooks/queries/useMassBuilder';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function StatusBadge({ status }: { status: string }) {
  const s = (status || '').toLowerCase();
  let color = 'var(--text-muted)';
  let bg = 'var(--bg-input)';
  let label = status || 'Unknown';

  if (s === 'completed') { color = 'var(--green)'; bg = 'var(--green-muted, rgba(76,175,80,0.15))'; }
  else if (s === 'running') { color = 'var(--accent)'; bg = 'var(--accent-muted)'; }
  else if (s === 'queued') { color = 'var(--text-muted)'; bg = 'var(--bg-input)'; }
  else if (s === 'failed') { color = 'var(--red)'; bg = 'var(--red-muted, rgba(239,83,80,0.15))'; }

  return (
    <span className="text-xs px-2 py-0.5 rounded-full font-medium" style={{ color, background: bg }}>
      {label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function MassResultsPage() {
  const { data: results, isLoading, error } = useMassResults();
  const deleteMutation = useDeleteMassResult();
  const cancelMutation = useCancelMassSearch();

  const [sortBy, setSortBy] = useState('Newest');

  const sorted = useMemo(() => {
    if (!results) return [];
    const arr = [...results] as any[];
    if (sortBy === 'Newest') arr.sort((a, b) => (b.created_at || b.date || '').localeCompare(a.created_at || a.date || ''));
    else if (sortBy === 'Results') arr.sort((a, b) => (b.result_count || 0) - (a.result_count || 0));
    return arr;
  }, [results, sortBy]);

  // ---------------------------------------------------------------------------
  // Loading / Error
  // ---------------------------------------------------------------------------

  if (isLoading) {
    return (
      <div>
        <PageHeader title="Mass Results" subtitle="Loading..." />
        <div className="space-y-3 mt-4">
          {[1, 2, 3].map((i) => (
            <Card key={i}>
              <div className="animate-pulse space-y-3">
                <div className="h-4 rounded w-1/3" style={{ background: 'var(--border)' }} />
                <div className="h-3 rounded w-2/3" style={{ background: 'var(--border)' }} />
                <div className="h-3 rounded w-1/2" style={{ background: 'var(--border)' }} />
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
        <PageHeader title="Mass Results" subtitle="Error" />
        <Card>
          <div className="text-center py-8" style={{ color: 'var(--red)' }}>
            Failed to load mass results. Check your connection and try again.
          </div>
        </Card>
      </div>
    );
  }

  const runningCount = sorted.filter((s: any) => (s.status || '').toLowerCase() === 'running').length;

  return (
    <div>
      <PageHeader
        title="Mass Results"
        subtitle="Browse and manage saved mass builder searches"
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

      {/* Controls */}
      <div className="flex items-center justify-between mb-4 mt-4">
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
          {sorted.length} saved search{sorted.length !== 1 ? 'es' : ''}
          {runningCount > 0 && ` \u00b7 ${runningCount} running`}
        </p>
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value)}
          className="px-2 py-1 rounded text-xs"
          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
        >
          <option value="Newest">Newest First</option>
          <option value="Results">Most Results</option>
        </select>
      </div>

      {/* Empty state */}
      {sorted.length === 0 && (
        <Card>
          <div className="text-center py-12">
            <p className="text-lg mb-2" style={{ color: 'var(--text-secondary)' }}>
              No saved searches yet.
            </p>
            <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
              Run a mass search from the Mass Builder to see results here.
            </p>
            <Link href="/mass-builder">
              <button className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: '#fff' }}>
                Go to Mass Builder
              </button>
            </Link>
          </div>
        </Card>
      )}

      {/* Search cards */}
      <div className="space-y-3">
        {sorted.map((search: any) => {
          const searchId = search.id || search.search_id;
          const isRunning = (search.status || '').toLowerCase() === 'running';
          const isQueued = (search.status || '').toLowerCase() === 'queued';
          const isComplete = (search.status || '').toLowerCase() === 'completed';

          return (
            <Card key={searchId}>
              <div className="flex items-start justify-between">
                {/* Left: info */}
                <div className="flex-1 min-w-0">
                  {/* Row 1: Name + status + date */}
                  <div className="flex items-center gap-2 mb-1 flex-wrap">
                    <h3 className="font-semibold text-sm">{search.name || `Search #${searchId}`}</h3>
                    <StatusBadge status={search.status} />
                    <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
                      {search.created_at || search.date || '--'}
                    </span>
                  </div>

                  {/* Row 2: Config summary */}
                  <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>
                    {search.ticker_count || search.symbols?.length || '?'} ticker{(search.ticker_count || 0) !== 1 ? 's' : ''}
                    {' \u00b7 '}{search.tf_count || search.timeframes?.length || '?'} TFs
                    {' \u00b7 '}{search.total_evaluations?.toLocaleString() || '?'} evaluations
                    {isComplete && search.result_count != null && ` \u00b7 ${search.result_count} results`}
                  </p>

                  {/* Running: progress bar */}
                  {isRunning && search.progress != null && (
                    <div className="mb-2">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs" style={{ color: 'var(--accent)' }}>
                          {search.current_label || search.current_step || 'Processing...'}
                        </span>
                        <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
                          {Math.round((search.progress / (search.total || 1)) * 100)}%
                        </span>
                      </div>
                      <div className="w-full h-2 rounded-full overflow-hidden" style={{ background: 'var(--bg-input)' }}>
                        <div
                          className="h-full rounded-full transition-all"
                          style={{
                            width: `${Math.round((search.progress / (search.total || 1)) * 100)}%`,
                            background: 'var(--accent)',
                          }}
                        />
                      </div>
                    </div>
                  )}

                  {/* Queued: status message */}
                  {isQueued && (
                    <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>
                      Queued — waiting for active search to complete
                    </p>
                  )}

                  {/* Completed: best KPIs */}
                  {isComplete && search.best_kpis && (
                    <div className="flex gap-6">
                      {[
                        { label: 'Best Daily R', value: search.best_kpis.daily_r != null ? `+${search.best_kpis.daily_r.toFixed(2)}` : '--' },
                        { label: 'Best WR', value: search.best_kpis.win_rate != null ? `${search.best_kpis.win_rate.toFixed(1)}%` : '--' },
                        { label: 'Best PF', value: search.best_kpis.profit_factor != null ? search.best_kpis.profit_factor.toFixed(2) : '--' },
                      ].map((kpi) => (
                        <div key={kpi.label}>
                          <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
                          <p className="text-xs font-bold">{kpi.value}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Right: actions */}
                <div className="flex items-center gap-2 flex-shrink-0 ml-4">
                  {isComplete && (
                    <Link
                      href={`/mass-builder?load=${searchId}`}
                      className="text-xs px-3 py-1.5 rounded"
                      style={{
                        background: 'var(--bg-input)',
                        color: 'var(--text-secondary)',
                        border: '1px solid var(--border)',
                        textDecoration: 'none',
                      }}
                    >
                      View
                    </Link>
                  )}
                  {isRunning && (
                    <button
                      className="text-xs px-3 py-1.5 rounded"
                      style={{ background: 'var(--bg-input)', color: 'var(--orange)', border: '1px solid var(--orange)', cursor: 'pointer' }}
                      onClick={() => cancelMutation.mutate(searchId)}
                    >
                      Cancel
                    </button>
                  )}
                  <button
                    className="text-xs px-3 py-1.5 rounded"
                    style={{ background: 'var(--red)15', color: 'var(--red)', border: '1px solid var(--red)30', cursor: 'pointer' }}
                    onClick={() => {
                      if (confirm('Delete this search?')) {
                        deleteMutation.mutate(searchId);
                      }
                    }}
                  >
                    Delete
                  </button>
                </div>
              </div>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
