'use client';

import { useState, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import { useMassResults, useDeleteMassResult, useCancelMassSearch, useMassProgress, useResumeMassSearch } from '@/hooks/queries/useMassBuilder';
import MassSearchResultsPanel from '@/views/MassSearchResultsPanel';

/* ========================================================================= */
/* TYPES                                                                      */
/* ========================================================================= */

interface SavedSearch {
  id: string;
  name: string;
  date: string;
  status: 'completed' | 'running' | 'queued' | 'orphaned';
  // Config summary
  tickers: string[];
  timeframes: string[];
  directions: string[];
  tickerCount: number;
  tfCount: number;
  dirCount: number;
  entryCount: number;
  exitCount: number;
  confluenceCount: number;
  stopCount: number;
  targetCount: number;
  timeExitCount: number;
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
  // Checkpoint (if orphaned + resumable)
  checkpointGroupCount: number;
  plannedGroupCount: number;
  // Diagnostics — populated on completion
  wallTotalSec: number | null;
  searchFidelity: string;
}

/* ========================================================================= */
/* HELPERS                                                                     */
/* ========================================================================= */

/** Map a snake_case API record to the component's SavedSearch shape */
function mapApiSearch(raw: any): SavedSearch {
  const cfg = raw.config || raw.config_data || {};
  const summary = raw.summary || {};
  const resultsArr: any[] = Array.isArray(raw.results) ? raw.results : [];
  // Result count: prefer summary.results_stored, then actual array length.
  const resultCount = raw.result_count ?? summary.results_stored ?? resultsArr.length ?? 0;
  // Best KPIs: derived from results array if not pre-summarized.
  const bestOf = (key: string) =>
    resultsArr.length > 0
      ? Math.max(...resultsArr.map((r) => r?.kpis?.[key] ?? -Infinity))
      : null;
  const bestDailyR = raw.best_daily_r ?? summary.best_daily_r ?? bestOf('daily_r');
  return {
    id: String(raw.id ?? raw.search_id ?? ''),
    name: raw.name ?? cfg.name ?? '--',
    date: raw.created_at ?? raw.date ?? '--',
    status: raw.status === 'completed' ? 'completed'
      : raw.status === 'running' ? 'running'
      : raw.status === 'queued' ? 'queued'
      : raw.status === 'orphaned' ? 'orphaned'
      : raw.status === 'failed' ? 'orphaned'  // surface failed as interrupted
      : raw.status === 'cancelled' ? 'completed'  // collapse cancelled into a terminal bucket
      : 'completed',
    tickers: Array.isArray(cfg.tickers) ? cfg.tickers : [],
    timeframes: Array.isArray(cfg.timeframes) ? cfg.timeframes : [],
    directions: Array.isArray(cfg.directions) ? cfg.directions : [],
    tickerCount: raw.ticker_count ?? cfg.tickers?.length ?? 0,
    tfCount: raw.tf_count ?? cfg.timeframes?.length ?? 0,
    dirCount: raw.dir_count ?? cfg.directions?.length ?? 0,
    entryCount: raw.entry_count ?? cfg.entry_triggers?.length ?? 0,
    exitCount: raw.exit_count ?? cfg.exit_triggers?.length ?? 0,
    confluenceCount: raw.confluence_count ?? (cfg.tf_confluences?.length ?? 0) + (cfg.general_confluences?.length ?? 0),
    stopCount: raw.stop_count ?? cfg.stop_packs?.length ?? 0,
    targetCount: raw.target_count ?? cfg.target_packs?.length ?? 0,
    timeExitCount: cfg.time_exit_packs?.length ?? 0,
    // Approximate total evaluations: trigger backtests × confluence combos.
    // Mirrors the Mass Builder preview formula (simplified — doesn't account
    // for TF expansion multiplier). Good enough for an at-a-glance chip.
    totalEvaluations: raw.total_evaluations
      ?? ((cfg.tickers?.length ?? 1)
          * (cfg.timeframes?.length ?? 1)
          * (cfg.directions?.length ?? 1)
          * (cfg.entry_triggers?.length ?? 1)
          * (cfg.exit_triggers?.length ?? 1)
          * Math.max(cfg.stop_packs?.length ?? 1, 1)
          * Math.max(cfg.target_packs?.length ?? 1, 1)),
    resultCount,
    bestDailyR: bestDailyR !== -Infinity ? bestDailyR : null,
    bestWR: raw.best_win_rate ?? raw.best_wr ?? (resultsArr.length ? bestOf('win_rate') : null),
    bestPF: raw.best_profit_factor ?? raw.best_pf ?? (resultsArr.length ? bestOf('profit_factor') : null),
    bestR2: raw.best_r_squared ?? raw.best_r2 ?? (resultsArr.length ? bestOf('r_squared') : null),
    progress: raw.progress ?? null,
    elapsed: raw.elapsed ?? null,
    eta: raw.eta ?? null,
    currentStep: raw.current_step ?? null,
    // Total wall-clock from the search's diagnostics (mass_builder.py:1272).
    // Falls back to null for in-progress or pre-diagnostics searches.
    wallTotalSec: (summary?.diagnostics?.wall_total_sec ?? null) as number | null,
    searchFidelity: (cfg.search_fidelity ?? cfg.searchFidelity ?? 'Rapid') as string,
    // Checkpoint for resume UX — count completed (symbol, tf) groups and
    // compare against the planned total. 'checkpoint' surfaces at top
    // level because get_mass_search merges config_data with the row.
    checkpointGroupCount: Array.isArray(raw.checkpoint?.completed_symbol_tfs)
      ? raw.checkpoint.completed_symbol_tfs.length : 0,
    plannedGroupCount:
      (cfg.tickers?.length ?? 0) * (cfg.timeframes?.length ?? 0),
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
  const router = useRouter();
  const { data: apiResults, isLoading, error } = useMassResults();
  const deleteMut = useDeleteMassResult();
  const cancelMut = useCancelMassSearch();
  const resumeMut = useResumeMassSearch();

  // ---- Local UI state ----
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState('Newest');
  const [expandedId, setExpandedId] = useState<string | null>(null);

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

  function handleEdit(searchId: string) {
    // Navigate to Mass Builder with ?edit={id}. Builder loads the saved
    // config AND populates the results panel, so the user can inspect,
    // sort, save strategies, and tweak the search without leaving the
    // single Mass Builder page. Re-running creates a new search; the
    // original saved record is never overwritten.
    router.push(`/mass-builder?edit=${searchId}`);
  }

  function handleResume(searchId: string) {
    // Resume an orphaned search from its last checkpoint (Roadmap 9s).
    // Backend loads the saved config + checkpoint and relaunches the
    // worker skipping already-completed (symbol, tf) groups. Same
    // search_id stays in use so results list stays stable.
    resumeMut.mutate(searchId);
  }

  return (
    <div>
      <PageHeader
        title="Mass Results"
        subtitle="Browse and manage saved mass builder searches"
      />

      {/* Preview-KPIs warning — shown alongside any Mass Builder output. */}
      <div
        className="mb-4 px-3 py-2 rounded flex items-start gap-2"
        style={{ background: 'rgba(255, 152, 0, 0.12)', border: '1px solid var(--orange)' }}
      >
        <span style={{ color: 'var(--orange)', fontSize: '1rem', lineHeight: 1 }}>⚠</span>
        <p className="text-xs" style={{ color: 'var(--text-secondary)', lineHeight: 1.5 }}>
          <span className="font-semibold" style={{ color: 'var(--orange)' }}>Preview KPIs.</span>{' '}
          Results below rank candidates via a post-filter on the base backtest.
          Real engine performance can differ when confluence filters out trades whose
          position-state blocked other candidates. Save interesting strategies, then run{' '}
          <strong>Update All Data</strong> on each to confirm with engine-accurate KPIs.
        </p>
      </div>

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
          const isOrphaned = search.status === 'orphaned';
          const isExpanded = expandedId === search.id;
          const canExpand = isComplete || isRunning || isOrphaned;

          const badgeColor = isComplete ? 'var(--green)'
            : isRunning ? 'var(--accent)'
            : isOrphaned ? 'var(--orange)'
            : 'var(--text-muted)';
          const badgeBg = isComplete ? 'var(--green-muted)'
            : isRunning ? 'var(--accent-muted)'
            : isOrphaned ? 'rgba(255, 152, 0, 0.15)'
            : 'var(--bg-input)';
          const badgeLabel = isComplete ? `${search.resultCount} results`
            : isRunning ? 'Running'
            : isOrphaned ? 'Interrupted'
            : 'Queued';

          return (
            <Card key={search.id}>
              <div className="flex items-start justify-between">
                {/* Left: info */}
                <div className="flex-1 min-w-0">
                  {/* Row 1: Name + status + date */}
                  <div className="flex items-center gap-2 mb-1 flex-wrap">
                    <h3 className="font-semibold text-sm">{search.name}</h3>
                    <span className="text-xs px-2 py-0.5 rounded-full font-medium" style={{
                      color: badgeColor, background: badgeBg,
                    }}>
                      {badgeLabel}
                    </span>
                    <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>{search.date}</span>
                    {/* Search Mode chip — visible when HighFidelity to signal
                        that this search used the new bit-matched path. */}
                    {search.searchFidelity === 'HighFidelity' && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded font-medium"
                            style={{ background: 'var(--accent)' + '20',
                                     color: 'var(--accent)' }}>
                        Hi-Fi
                      </span>
                    )}
                    {/* Wall-clock duration — populated on completion. */}
                    {search.wallTotalSec != null && (
                      <span className="text-xs font-mono"
                            style={{ color: 'var(--text-muted)' }}>
                        ⏱ {search.wallTotalSec < 60
                            ? `${search.wallTotalSec.toFixed(1)}s`
                            : search.wallTotalSec < 3600
                              ? `${Math.floor(search.wallTotalSec / 60)}m ${Math.round(search.wallTotalSec % 60)}s`
                              : `${(search.wallTotalSec / 3600).toFixed(2)}h`}
                      </span>
                    )}
                  </div>

                  {/* Interrupted: explain + point to Resume (or Edit fallback) */}
                  {isOrphaned && (
                    <p className="text-xs mb-2" style={{ color: 'var(--orange)' }}>
                      Search stopped before completing — the API restarted mid-run.{' '}
                      {search.checkpointGroupCount > 0 ? (
                        <>
                          <strong>{search.checkpointGroupCount}</strong>
                          {search.plannedGroupCount > 0 && <> of <strong>{search.plannedGroupCount}</strong></>}{' '}
                          (symbol × timeframe) group{search.checkpointGroupCount === 1 ? '' : 's'} completed —
                          click <strong>Resume</strong> to pick up from the next group.
                        </>
                      ) : (
                        <>No checkpoint yet — click <strong>Resume</strong> to restart from scratch, or <strong>Edit</strong> to tweak the config first.</>
                      )}
                    </p>
                  )}

                  {/* Row 2: Config summary — counts matching the Mass Builder tab badges */}
                  <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>
                    {search.tickerCount} ticker{search.tickerCount !== 1 ? 's' : ''}
                    {' '}&middot; {search.tfCount} TF{search.tfCount !== 1 ? 's' : ''}
                    {' '}&middot; {search.dirCount} dir
                    {' '}&middot; {search.entryCount} {search.entryCount === 1 ? 'entry' : 'entries'}
                    {' '}&middot; {search.exitCount} {search.exitCount === 1 ? 'exit' : 'exits'}
                    {' '}&middot; {search.confluenceCount} confluences
                    {' '}&middot; {search.stopCount} stops
                    {' '}&middot; {search.targetCount} targets
                    {' '}&middot; {search.timeExitCount} time exits
                    {' '}&middot; {search.totalEvaluations.toLocaleString()} trigger bts
                  </p>

                  {/* Running: live progress via useMassProgress (polls /progress/{id}) */}
                  {isRunning && <LiveProgress searchId={search.id} />}

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
                  {canExpand && (
                    <button
                      style={{ ...btnSecondary, color: isExpanded ? 'var(--accent)' : 'var(--text-secondary)', borderColor: isExpanded ? 'var(--accent)' : 'var(--border)' }}
                      onClick={() => setExpandedId(isExpanded ? null : search.id)}
                      title="Show per-dough results"
                    >
                      {isExpanded ? '▼ Results' : '▶ Results'}
                    </button>
                  )}
                  {isComplete && (
                    <button
                      style={{ ...btnSecondary, background: 'var(--accent)', color: 'white', border: 'none' }}
                      onClick={() => handleEdit(search.id)}
                    >
                      Edit
                    </button>
                  )}
                  {isOrphaned && (
                    <>
                      <button
                        style={{
                          ...btnSecondary,
                          background: 'var(--accent)',
                          color: 'white',
                          border: 'none',
                          opacity: resumeMut.isPending ? 0.6 : 1,
                        }}
                        disabled={resumeMut.isPending}
                        onClick={() => handleResume(search.id)}
                        title={search.checkpointGroupCount > 0
                          ? `Resume from group ${search.checkpointGroupCount + 1}`
                          : 'Restart from scratch (no checkpoint)'}
                      >
                        {resumeMut.isPending ? 'Resuming…' : 'Resume'}
                      </button>
                      <button
                        style={btnSecondary}
                        onClick={() => handleEdit(search.id)}
                        title="Edit the saved config (creates a new search on Analyze)"
                      >
                        Edit
                      </button>
                    </>
                  )}
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
              {isExpanded && (
                <MassSearchResultsPanel searchId={search.id} isRunning={isRunning} />
              )}
            </Card>
          );
        })}
      </div>
    </div>
  );
}

const PHASE_LABELS: Record<string, string> = {
  load: 'Loading',
  prep: 'Preparing',
  backtest: 'Backtesting',
  confluence: 'Searching confluences',
  save: 'Saving',
};

function formatLiveInner(phase: string | undefined | null, step: number, total: number): string {
  const pct = Math.round((step / total) * 100);
  const s = step.toLocaleString();
  const t = total.toLocaleString();
  switch (phase) {
    case 'load': return `${s}/${t} bars loaded (${pct}%)`;
    case 'backtest': return `Bar ${s}/${t} (${pct}%)`;
    case 'confluence': return `Combo ${s}/${t} (${pct}%)`;
    default: return `${s}/${t}`;
  }
}

/** Live progress row for a running search. Uses useMassProgress (250ms polling
 *  when running). Mirrors the two-tier design used on the Mass Builder page. */
function LiveProgress({ searchId }: { searchId: string }) {
  const { data } = useMassProgress(searchId);
  if (!data) {
    return (
      <div className="mb-2">
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Connecting to search…</p>
      </div>
    );
  }
  const d = data as any;
  const pct = d.total > 0 ? (d.progress / d.total) : 0;
  const phase = d.phase as string | undefined;
  const phaseDetail = d.phase_detail || d.current_label || 'Running...';
  const innerStep = d.inner_step;
  const innerTotal = d.inner_total;
  const hasInner = innerStep != null && innerTotal > 0;
  return (
    <div className="mb-2">
      <div className="flex items-center justify-between mb-1 gap-2">
        <div className="flex items-center gap-1.5 min-w-0 flex-1">
          {phase && (
            <span className="text-[9px] font-semibold px-1 py-0.5 rounded flex-shrink-0"
                  style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
              {PHASE_LABELS[phase] || phase}
            </span>
          )}
          <span className="text-xs truncate" style={{ color: 'var(--accent)' }}>{phaseDetail}</span>
        </div>
        <span className="text-xs font-mono flex-shrink-0" style={{ color: 'var(--text-muted)' }}>
          {d.progress}/{d.total} &middot; {Math.round(pct * 100)}%
        </span>
      </div>
      <div className="w-full h-2 rounded-full overflow-hidden" style={{ background: 'var(--bg-input)' }}>
        <div className="h-full rounded-full transition-all" style={{ width: `${pct * 100}%`, background: 'var(--accent)' }} />
      </div>
      {hasInner && (
        <div className="mt-1">
          <div className="flex items-center justify-between mb-0.5">
            <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
              {formatLiveInner(phase, innerStep, innerTotal)}
            </span>
          </div>
          <div className="w-full h-1 rounded-full overflow-hidden" style={{ background: 'var(--bg-input)' }}>
            <div className="h-full rounded-full transition-all duration-150" style={{ width: `${(innerStep / innerTotal) * 100}%`, background: 'var(--text-muted)' }} />
          </div>
        </div>
      )}
    </div>
  );
}
