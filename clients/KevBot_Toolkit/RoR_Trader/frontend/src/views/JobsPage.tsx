/**
 * Jobs page — lists recent recompute jobs with live progress.
 *
 * Backend: /api/jobs (in-memory, no DB persistence). State is per-API-
 * process; jobs lost on worker restart. Per spec (M8.7 Phase 3 plan).
 */
'use client';

import React, { useState } from 'react';
import { useSearchParams } from 'next/navigation';
import Card from '@/components/Card';
import {
  useJobsList,
  useJobStatus,
  useCancelJob,
  type RecomputeJob,
  type JobStatus,
} from '@/hooks/queries/useJobs';

const btnSecondary: React.CSSProperties = {
  padding: '6px 12px',
  borderRadius: '6px',
  background: 'var(--bg-input)',
  color: 'var(--text-secondary)',
  border: '1px solid var(--border)',
  fontSize: '13px',
  cursor: 'pointer',
};

const btnDanger: React.CSSProperties = {
  ...btnSecondary,
  background: 'var(--red-muted)',
  color: 'var(--red)',
  border: 'none',
};

const STATUS_COLORS: Record<JobStatus, { bg: string; fg: string }> = {
  queued: { bg: 'var(--bg-input)', fg: 'var(--text-muted)' },
  running: { bg: 'var(--orange-muted)', fg: 'var(--orange)' },
  completed: { bg: 'var(--green-muted)', fg: 'var(--green)' },
  failed: { bg: 'var(--red-muted)', fg: 'var(--red)' },
  cancelled: { bg: 'var(--bg-input)', fg: 'var(--text-muted)' },
};

function fmtAge(iso: string | null): string {
  if (!iso) return '—';
  const ms = Date.now() - new Date(iso).getTime();
  if (ms < 60_000) return `${Math.round(ms / 1000)}s ago`;
  if (ms < 3_600_000) return `${Math.round(ms / 60_000)}m ago`;
  if (ms < 86_400_000) return `${Math.round(ms / 3_600_000)}h ago`;
  return `${Math.round(ms / 86_400_000)}d ago`;
}

function fmtJobType(jt: string): string {
  if (jt === 'append_recent') return 'Update New Data';
  if (jt === 'full_recompute') return 'Update All Data';
  return jt;
}

function ProgressBar({ pct }: { pct: number }) {
  const clamped = Math.max(0, Math.min(100, pct));
  return (
    <div
      style={{
        width: '100%',
        height: '8px',
        background: 'var(--bg-input)',
        borderRadius: '4px',
        overflow: 'hidden',
        marginTop: '6px',
      }}
    >
      <div
        style={{
          width: `${clamped}%`,
          height: '100%',
          background: 'var(--blue, #3b82f6)',
          transition: 'width 300ms ease',
        }}
      />
    </div>
  );
}

function JobCard({
  job,
  expanded,
  onToggle,
}: {
  job: RecomputeJob;
  expanded: boolean;
  onToggle: () => void;
}) {
  const cancelMut = useCancelJob();
  const total = job.summary.total_strategies || job.strategy_ids.length;
  const done = job.current_strategy_idx + (job.status === 'completed' ? 1 : 0);
  const pct = total > 0 ? (done / total) * 100 : 0;
  const statusColor = STATUS_COLORS[job.status] || STATUS_COLORS.queued;

  return (
    <Card>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          gap: '16px',
        }}
      >
        <div style={{ flex: 1, minWidth: 0 }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              flexWrap: 'wrap',
            }}
          >
            <span style={{ fontSize: '15px', fontWeight: 600 }}>
              {fmtJobType(job.job_type)}
            </span>
            <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
              {total} {total === 1 ? 'strategy' : 'strategies'}
            </span>
            <span
              style={{
                background: statusColor.bg,
                color: statusColor.fg,
                fontSize: '11px',
                padding: '2px 8px',
                borderRadius: '4px',
                fontWeight: 600,
                textTransform: 'uppercase',
              }}
            >
              {job.status}
            </span>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              created {fmtAge(job.created_at)}
            </span>
          </div>
          {(job.status === 'running' || job.status === 'queued') && (
            <>
              <div
                style={{
                  marginTop: '6px',
                  fontSize: '12px',
                  color: 'var(--text-secondary)',
                }}
              >
                {job.current_strategy_name ? (
                  <>
                    Current: {job.current_strategy_name}{' '}
                    <span style={{ color: 'var(--text-muted)' }}>
                      (sid {job.current_strategy_id})
                    </span>
                  </>
                ) : (
                  job.progress_label
                )}
              </div>
              <ProgressBar pct={pct} />
              <div
                style={{
                  marginTop: '4px',
                  fontSize: '11px',
                  color: 'var(--text-muted)',
                }}
              >
                {job.current_strategy_idx} / {total} ·{' '}
                {job.progress_label} · {job.summary.total_inserted} new trades
              </div>
            </>
          )}
          {(job.status === 'completed' ||
            job.status === 'failed' ||
            job.status === 'cancelled') && (
            <div
              style={{
                marginTop: '6px',
                fontSize: '12px',
                color: 'var(--text-secondary)',
              }}
            >
              Inserted {job.summary.total_inserted} · Skipped{' '}
              {job.summary.total_skipped} · Failed {job.summary.total_failed}
              {job.completed_at && (
                <span style={{ color: 'var(--text-muted)' }}>
                  {' '}
                  · finished {fmtAge(job.completed_at)}
                </span>
              )}
              {job.error && (
                <div style={{ color: 'var(--red)', marginTop: '4px' }}>
                  Error: {job.error}
                </div>
              )}
            </div>
          )}
        </div>
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '6px',
            alignItems: 'flex-end',
          }}
        >
          {(job.status === 'running' || job.status === 'queued') && (
            <button
              style={btnDanger}
              disabled={cancelMut.isPending}
              onClick={() => cancelMut.mutate(job.id)}
            >
              {cancelMut.isPending ? 'Cancelling…' : 'Cancel'}
            </button>
          )}
          <button style={btnSecondary} onClick={onToggle}>
            {expanded ? 'Hide details' : 'Show details'}
          </button>
        </div>
      </div>
      {expanded && job.per_strategy_results.length > 0 && (
        <div
          style={{
            marginTop: '12px',
            padding: '10px',
            background: 'var(--bg-input)',
            borderRadius: '6px',
            fontSize: '12px',
            fontFamily: 'monospace',
          }}
        >
          {job.per_strategy_results.map((r, i) => {
            // Build the engine-scope string. Skipped jobs don't have
            // bars_processed (engine never ran). Show "scanned X bars
            // over Y days" when present so user knows how far the
            // windowed run looked.
            const scope = (r.bars_processed != null && r.window_days != null)
              ? `scanned ${r.bars_processed.toLocaleString()} bars · ${r.window_days}d window`
              : (r.engine_path === 'full' ? 'full backtest' : null);
            // Skip reason — surface 'recently_processed' / 'no_recent_exits'
            // so the user knows why a row says skipped (was opaque before).
            const skipDetail = r.status === 'skipped' && r.reason
              ? r.reason.replace(/_/g, ' ')
              : null;
            return (
              <div
                key={i}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  padding: '6px 0',
                  borderBottom: i < job.per_strategy_results.length - 1
                    ? '1px solid var(--border)'
                    : 'none',
                  gap: '2px',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ flex: 1, minWidth: 0 }}>
                    sid {r.strategy_id} · {r.strategy_name}
                  </span>
                  <span style={{ color: 'var(--text-muted)', marginRight: '8px' }}>
                    {r.engine_path && `(${r.engine_path}) `}
                    {r.elapsed_s}s
                  </span>
                  <span
                    style={{
                      color:
                        r.status === 'appended' || r.status === 'refreshed'
                          ? 'var(--green)'
                          : r.status === 'error' || r.status === 'failed'
                          ? 'var(--red)'
                          : 'var(--text-muted)',
                      fontWeight: 600,
                      minWidth: '90px',
                      textAlign: 'right',
                    }}
                  >
                    {r.status} {r.inserted ? `(+${r.inserted})` : ''}
                  </span>
                </div>
                {(scope || skipDetail || r.error) && (
                  <div
                    style={{
                      fontSize: '11px',
                      color: 'var(--text-muted)',
                      paddingLeft: '8px',
                    }}
                  >
                    {scope && <span>{scope}</span>}
                    {scope && (skipDetail || r.error) && <span> · </span>}
                    {skipDetail && <span>reason: {skipDetail}</span>}
                    {r.error && (
                      <span style={{ color: 'var(--red)' }}>error: {r.error}</span>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </Card>
  );
}

export default function JobsPage() {
  const searchParams = useSearchParams();
  const highlightId = searchParams.get('highlight');
  const [filter, setFilter] = useState<JobStatus | 'all'>('all');
  const [expandedIds, setExpandedIds] = useState<Set<string>>(
    new Set(highlightId ? [highlightId] : []),
  );

  const { data, error, isLoading } = useJobsList(
    filter === 'all' ? null : filter,
  );

  const toggleExpand = (jobId: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(jobId)) next.delete(jobId);
      else next.add(jobId);
      return next;
    });
  };

  const filterButtonStyle = (active: boolean): React.CSSProperties => ({
    ...btnSecondary,
    background: active ? 'var(--blue-muted, #1e40af44)' : 'var(--bg-input)',
    color: active ? 'var(--blue, #3b82f6)' : 'var(--text-secondary)',
    fontWeight: active ? 600 : 400,
  });

  return (
    <div style={{ padding: '32px', maxWidth: '1100px', margin: '0 auto' }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '20px',
        }}
      >
        <div>
          <h1 style={{ fontSize: '24px', fontWeight: 700, marginBottom: '4px' }}>
            Jobs
          </h1>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
            Background recompute jobs from My Strategies bulk actions. State
            is in-memory; jobs are lost on a worker restart.
          </p>
        </div>
      </div>

      <div
        style={{
          display: 'flex',
          gap: '6px',
          marginBottom: '16px',
          flexWrap: 'wrap',
        }}
      >
        {(['all', 'running', 'queued', 'completed', 'failed', 'cancelled'] as const).map(
          (s) => (
            <button
              key={s}
              style={filterButtonStyle(filter === s)}
              onClick={() => setFilter(s)}
            >
              {s.charAt(0).toUpperCase() + s.slice(1)}
            </button>
          ),
        )}
      </div>

      {isLoading && (
        <div style={{ color: 'var(--text-muted)', padding: '24px' }}>
          Loading…
        </div>
      )}
      {error && (
        <Card>
          <div style={{ color: 'var(--red)', padding: '12px' }}>
            Failed to load jobs: {(error as any)?.message || String(error)}
          </div>
        </Card>
      )}
      {data && data.jobs.length === 0 && (
        <Card>
          <div
            style={{
              padding: '24px',
              textAlign: 'center',
              color: 'var(--text-muted)',
            }}
          >
            No {filter === 'all' ? '' : filter} jobs yet. Trigger one from
            My Strategies → bulk select → Update New Data / Update All Data.
          </div>
        </Card>
      )}
      {data && data.jobs.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {data.jobs.map((job) => (
            <JobCard
              key={job.id}
              job={job}
              expanded={expandedIds.has(job.id)}
              onToggle={() => toggleExpand(job.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
