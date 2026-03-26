'use client';

import { useState, useMemo } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import { useMassResults, useDeleteMassResult, useCancelMassSearch } from '@/hooks/queries/useMassBuilder';

/* ========================================================================= */
/* TYPES                                                                      */
/* ========================================================================= */

interface SavedSearch {
  id: string;
  name: string;
  date: string;
  status: 'completed' | 'running' | 'queued';
  // Config summary
  tickerCount: number;
  tfCount: number;
  dirCount: number;
  entryCount: number;
  exitCount: number;
  confluenceCount: number;
  stopCount: number;
  targetCount: number;
  totalEvaluations: number;
  // Results (if completed)
  resultCount: number;
  bestDailyR: number | null;
  bestWR: number | null;
  bestPF: number | null;
  bestR2: number | null;
  // Progress (if running)
  progress: number | null;
  elapsed: string | null;
  eta: string | null;
  currentStep: string | null;
}

/* ========================================================================= */
/* HELPERS                                                                     */
/* ========================================================================= */

/** Map a snake_case API record to the component's SavedSearch shape */
function mapApiSearch(raw: any): SavedSearch {
  return {
    id: String(raw.id ?? raw.search_id ?? ''),
    name: raw.name ?? '--',
    date: raw.created_at ?? raw.date ?? '--',
    status: raw.status === 'completed' ? 'completed'
      : raw.status === 'running' ? 'running'
      : raw.status === 'queued' ? 'queued'
      : 'completed',
    tickerCount: raw.ticker_count ?? raw.config?.tickers?.length ?? 0,
    tfCount: raw.tf_count ?? raw.config?.timeframes?.length ?? 0,
    dirCount: raw.dir_count ?? raw.config?.directions?.length ?? 0,
    entryCount: raw.entry_count ?? raw.config?.entry_triggers?.length ?? 0,
    exitCount: raw.exit_count ?? raw.config?.exit_triggers?.length ?? 0,
    confluenceCount: raw.confluence_count ?? 0,
    stopCount: raw.stop_count ?? 0,
    targetCount: raw.target_count ?? 0,
    totalEvaluations: raw.total_evaluations ?? 0,
    resultCount: raw.result_count ?? 0,
    bestDailyR: raw.best_daily_r ?? null,
    bestWR: raw.best_win_rate ?? raw.best_wr ?? null,
    bestPF: raw.best_profit_factor ?? raw.best_pf ?? null,
    bestR2: raw.best_r_squared ?? raw.best_r2 ?? null,
    progress: raw.progress ?? null,
    elapsed: raw.elapsed ?? null,
    eta: raw.eta ?? null,
    currentStep: raw.current_step ?? null,
  };
}

/* ========================================================================= */
/* STYLES                                                                      */
/* ========================================================================= */

const btnSecondary: React.CSSProperties = {
  background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)',
  padding: '4px 12px', borderRadius: '8px', fontSize: '0.75rem', cursor: 'pointer',
};

/* ========================================================================= */
/* COMPONENT                                                                   */
/* ========================================================================= */

export default function MassResultsPage() {
  // ---- API hooks (MUST come before any early returns) ----
  const { data: apiResults, isLoading, error } = useMassResults();
  const deleteMut = useDeleteMassResult();
  const cancelMut = useCancelMassSearch();

  // ---- Local UI state ----
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState('Newest');

  // ---- Map API data ----
  const searches: SavedSearch[] = useMemo(() => {
    if (!apiResults) return [];
    return apiResults.map(mapApiSearch);
  }, [apiResults]);

  const sorted = useMemo(() => {
    return [...searches].sort((a, b) => {
      if (sortBy === 'Newest') return b.date.localeCompare(a.date);
      if (sortBy === 'Results') return (b.resultCount || 0) - (a.resultCount || 0);
      if (sortBy === 'Best Daily R') return (b.bestDailyR || 0) - (a.bestDailyR || 0);
      return 0;
    });
  }, [searches, sortBy]);

  // ---- Loading / Error states ----

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
                <div className="h-8 rounded" style={{ background: 'var(--bg-input)' }} />
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
        <PageHeader title="Mass Results" subtitle="Error loading results" />
        <Card>
          <div className="text-center py-8" style={{ color: 'var(--red)' }}>
            Failed to load mass builder results. Check your connection and try again.
          </div>
        </Card>
      </div>
    );
  }

  // ---- Action handlers ----
  function handleDelete(searchId: string) {
    deleteMut.mutate(Number(searchId));
    setDeleteConfirmId(null);
  }

  function handleCancel(searchId: string) {
    cancelMut.mutate(Number(searchId));
  }

  return (
    <div>
      <PageHeader
        title="Mass Results"
        subtitle="Browse and manage saved mass builder searches"
      />

      {/* Controls */}
      <div className="flex items-center justify-between mb-4">
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
          {searches.length} saved searches &middot; {searches.filter((s) => s.status === 'running').length} running
        </p>
        <div className="flex items-center gap-2">
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="px-2 py-1 rounded text-xs"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
          >
            <option value="Newest">Newest First</option>
            <option value="Results">Most Results</option>
            <option value="Best Daily R">Best Daily R</option>
          </select>
        </div>
      </div>

      {/* Empty state */}
      {searches.length === 0 && (
        <Card>
          <div className="text-center py-8">
            <p className="text-sm mb-1" style={{ color: 'var(--text-secondary)' }}>
              No saved searches yet
            </p>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
              Run a mass builder search to see results here.
            </p>
          </div>
        </Card>
      )}

      {/* Search cards */}
      <div className="space-y-3">
        {sorted.map((search) => {
          const isRunning = search.status === 'running';
          const isQueued = search.status === 'queued';
          const isComplete = search.status === 'completed';

          return (
            <Card key={search.id}>
              <div className="flex items-start justify-between">
                {/* Left: info */}
                <div className="flex-1 min-w-0">
                  {/* Row 1: Name + status + date */}
                  <div className="flex items-center gap-2 mb-1 flex-wrap">
                    <h3 className="font-semibold text-sm">{search.name}</h3>
                    <span className="text-xs px-2 py-0.5 rounded-full font-medium" style={{
                      color: isComplete ? 'var(--green)' : isRunning ? 'var(--accent)' : 'var(--text-muted)',
                      background: isComplete ? 'var(--green-muted)' : isRunning ? 'var(--accent-muted)' : 'var(--bg-input)',
                    }}>
                      {isComplete ? `${search.resultCount} results` : isRunning ? 'Running' : 'Queued'}
                    </span>
                    <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>{search.date}</span>
                  </div>

                  {/* Row 2: Config summary */}
                  <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>
                    {search.tickerCount} ticker{search.tickerCount !== 1 ? 's' : ''}
                    {' '}&middot; {search.tfCount} TF{search.tfCount !== 1 ? 's' : ''}
                    {' '}&middot; {search.dirCount} dir
                    {' '}&middot; {search.entryCount} entries
                    {' '}&middot; {search.exitCount} exits
                    {' '}&middot; {search.confluenceCount} confluences
                    {' '}&middot; {search.stopCount} stops
                    {' '}&middot; {search.targetCount} targets
                    {' '}&middot; {search.totalEvaluations.toLocaleString()} evaluations
                  </p>

                  {/* Running: progress bar */}
                  {isRunning && search.progress !== null && (
                    <div className="mb-2">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs" style={{ color: 'var(--accent)' }}>{search.currentStep}</span>
                        <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
                          {Math.round(search.progress * 100)}% &middot; Elapsed: {search.elapsed ?? '--'} &middot; ETA: {search.eta ?? '--'}
                        </span>
                      </div>
                      <div className="w-full h-2 rounded-full overflow-hidden" style={{ background: 'var(--bg-input)' }}>
                        <div className="h-full rounded-full transition-all" style={{ width: `${search.progress * 100}%`, background: 'var(--accent)' }} />
                      </div>
                    </div>
                  )}

                  {/* Queued: status message */}
                  {isQueued && (
                    <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>
                      {search.currentStep ?? 'Queued'}
                    </p>
                  )}

                  {/* Completed: best KPIs */}
                  {isComplete && (
                    <div className="flex gap-6">
                      {[
                        { label: 'Best Daily R', value: search.bestDailyR != null ? `+${search.bestDailyR.toFixed(2)}` : '--' },
                        { label: 'Best WR', value: search.bestWR != null ? `${search.bestWR.toFixed(1)}%` : '--' },
                        { label: 'Best PF', value: search.bestPF != null ? search.bestPF.toFixed(2) : '--' },
                        { label: 'Best R\u00B2', value: search.bestR2 != null ? search.bestR2.toFixed(2) : '--' },
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
                    <Link href={`/mass-builder?load=${search.id}`} style={{ ...btnSecondary, textDecoration: 'none' }}>
                      View
                    </Link>
                  )}
                  {isComplete && (
                    <button style={btnSecondary}>Load</button>
                  )}
                  <button style={btnSecondary}>Copy</button>
                  {isRunning && (
                    <button
                      style={{ ...btnSecondary, color: 'var(--orange)', borderColor: 'var(--orange)' }}
                      onClick={() => handleCancel(search.id)}
                    >
                      Cancel
                    </button>
                  )}
                  {deleteConfirmId === search.id ? (
                    <div className="flex items-center gap-1">
                      <button style={{ ...btnSecondary, background: 'var(--red)', color: 'white', border: 'none' }}
                        onClick={() => handleDelete(search.id)}>Yes</button>
                      <button style={btnSecondary} onClick={() => setDeleteConfirmId(null)}>No</button>
                    </div>
                  ) : (
                    <button style={{ ...btnSecondary, color: 'var(--red)' }} onClick={() => setDeleteConfirmId(search.id)}>Delete</button>
                  )}
                </div>
              </div>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
