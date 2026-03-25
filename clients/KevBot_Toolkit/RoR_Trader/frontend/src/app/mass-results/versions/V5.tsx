'use client';

import { useState } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import Modal from '@/components/Modal';

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
/* MOCK DATA                                                                  */
/* ========================================================================= */

const mockSearches: SavedSearch[] = [
  {
    id: 's1', name: 'NVDA Scalping Search', date: '2026-03-24 14:32', status: 'completed',
    tickerCount: 1, tfCount: 1, dirCount: 1, entryCount: 4, exitCount: 2, confluenceCount: 8, stopCount: 2, targetCount: 2,
    totalEvaluations: 1152,
    resultCount: 47, bestDailyR: 2.41, bestWR: 62.5, bestPF: 3.12, bestR2: 0.87,
    progress: null, elapsed: null, eta: null, currentStep: null,
  },
  {
    id: 's2', name: 'Multi-Ticker Swing', date: '2026-03-23 10:15', status: 'completed',
    tickerCount: 3, tfCount: 2, dirCount: 2, entryCount: 6, exitCount: 3, confluenceCount: 12, stopCount: 3, targetCount: 2,
    totalEvaluations: 15552,
    resultCount: 128, bestDailyR: 1.95, bestWR: 58.3, bestPF: 2.45, bestR2: 0.82,
    progress: null, elapsed: null, eta: null, currentStep: null,
  },
  {
    id: 's3', name: 'Full Universe Scan', date: '2026-03-22 09:02', status: 'running',
    tickerCount: 7, tfCount: 3, dirCount: 2, entryCount: 8, exitCount: 4, confluenceCount: 15, stopCount: 4, targetCount: 3,
    totalEvaluations: 48384,
    resultCount: 0, bestDailyR: null, bestWR: null, bestPF: null, bestR2: null,
    progress: 0.62, elapsed: '4m 18s', eta: '~2m 40s', currentStep: 'Testing combinations... 30,198 / 48,384',
  },
  {
    id: 's4', name: 'Crypto Weekend Test', date: '2026-03-21 20:45', status: 'queued',
    tickerCount: 2, tfCount: 2, dirCount: 1, entryCount: 3, exitCount: 2, confluenceCount: 6, stopCount: 2, targetCount: 1,
    totalEvaluations: 864,
    resultCount: 0, bestDailyR: null, bestWR: null, bestPF: null, bestR2: null,
    progress: null, elapsed: null, eta: null, currentStep: 'Queued — waiting for Full Universe Scan to complete',
  },
  {
    id: 's5', name: 'ETF Momentum', date: '2026-03-20 11:30', status: 'completed',
    tickerCount: 4, tfCount: 2, dirCount: 1, entryCount: 5, exitCount: 3, confluenceCount: 10, stopCount: 2, targetCount: 2,
    totalEvaluations: 8640,
    resultCount: 85, bestDailyR: 1.78, bestWR: 55.4, bestPF: 2.08, bestR2: 0.79,
    progress: null, elapsed: null, eta: null, currentStep: null,
  },
];

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

export default function MassResultsV5() {
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState('Newest');

  const sorted = [...mockSearches].sort((a, b) => {
    if (sortBy === 'Newest') return b.date.localeCompare(a.date);
    if (sortBy === 'Results') return (b.resultCount || 0) - (a.resultCount || 0);
    if (sortBy === 'Best Daily R') return (b.bestDailyR || 0) - (a.bestDailyR || 0);
    return 0;
  });

  return (
    <div>
      <PageHeader
        title="Mass Results"
        subtitle="Browse and manage saved mass builder searches"
      />

      {/* Controls */}
      <div className="flex items-center justify-between mb-4">
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
          {mockSearches.length} saved searches &middot; {mockSearches.filter((s) => s.status === 'running').length} running
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
                          {Math.round(search.progress * 100)}% &middot; Elapsed: {search.elapsed} &middot; ETA: {search.eta}
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
                      {search.currentStep}
                    </p>
                  )}

                  {/* Completed: best KPIs */}
                  {isComplete && (
                    <div className="flex gap-6">
                      {[
                        { label: 'Best Daily R', value: `+${search.bestDailyR?.toFixed(2)}` },
                        { label: 'Best WR', value: `${search.bestWR?.toFixed(1)}%` },
                        { label: 'Best PF', value: search.bestPF?.toFixed(2) },
                        { label: 'Best R²', value: search.bestR2?.toFixed(2) },
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
                    <button style={{ ...btnSecondary, color: 'var(--orange)', borderColor: 'var(--orange)' }}>Cancel</button>
                  )}
                  {deleteConfirmId === search.id ? (
                    <div className="flex items-center gap-1">
                      <button style={{ ...btnSecondary, background: 'var(--red)', color: 'white', border: 'none' }}
                        onClick={() => setDeleteConfirmId(null)}>Yes</button>
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
