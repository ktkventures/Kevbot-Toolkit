/**
 * Update Jobs admin page — DB-persisted async UAD jobs.
 *
 * Parallel to /jobs (in-memory). The endpoints `/api/update-jobs/*`
 * survive worker restarts via orphan cleanup on api boot, so this is
 * the page to use when you want a job to survive a redeploy.
 *
 * Triggered from My Strategies bulk actions (Update New / Update All
 * Data) once Phase 2 frontend swap-over lands.
 */
'use client';

import React, { useState } from 'react';
import { useSearchParams } from 'next/navigation';
import Card from '@/components/Card';
import {
  useUpdateJobsList,
  useStartUpdateJob,
  useCancelUpdateJob,
  useDeleteUpdateJob,
  useCleanupOrphans,
  useSystemSettings,
  useSetSystemSetting,
  type UpdateJobListItem,
  type UpdateJobStatus,
} from '@/hooks/queries/useUpdateJobs';

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

const STATUS_COLORS: Record<UpdateJobStatus, { bg: string; fg: string }> = {
  queued: { bg: 'var(--bg-input)', fg: 'var(--text-muted)' },
  running: { bg: 'var(--orange-muted)', fg: 'var(--orange)' },
  completed: { bg: 'var(--green-muted)', fg: 'var(--green)' },
  failed: { bg: 'var(--red-muted)', fg: 'var(--red)' },
  cancelled: { bg: 'var(--bg-input)', fg: 'var(--text-muted)' },
  orphaned: { bg: 'var(--red-muted)', fg: 'var(--red)' },
};

function fmtAge(iso: string | null | undefined): string {
  if (!iso) return '—';
  const ms = Date.now() - new Date(iso).getTime();
  if (ms < 60_000) return `${Math.round(ms / 1000)}s ago`;
  if (ms < 3_600_000) return `${Math.round(ms / 60_000)}m ago`;
  if (ms < 86_400_000) return `${Math.round(ms / 3_600_000)}h ago`;
  return `${Math.round(ms / 86_400_000)}d ago`;
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
  job: UpdateJobListItem;
  expanded: boolean;
  onToggle: () => void;
}) {
  const cancelMut = useCancelUpdateJob();
  const deleteMut = useDeleteUpdateJob();
  const progress = job.progress;
  const total = progress?.total_steps || job.strategy_ids.length;
  const done = progress?.current_step ?? 0;
  const pct = total > 0 ? (done / total) * 100 : 0;
  const statusColor = STATUS_COLORS[job.status] || STATUS_COLORS.queued;
  const per = progress?.per_strategy || {};
  const perRows = Object.entries(per);

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
              {job.name || `Update job #${job.id}`}
            </span>
            <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
              {total} {total === 1 ? 'strategy' : 'strategies'} ·{' '}
              {job.scope}/{job.mode}
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
              #{job.id} · created {fmtAge(job.created_at)}
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
                {progress?.current_sid != null
                  ? `Current: sid ${progress.current_sid} (${progress.current_phase ?? ''})`
                  : (progress?.current_phase ?? 'starting')}
              </div>
              <ProgressBar pct={pct} />
              <div
                style={{
                  marginTop: '4px',
                  fontSize: '11px',
                  color: 'var(--text-muted)',
                }}
              >
                {done} / {total}
                {job.summary?.total_trades_inserted != null
                  ? ` · ${job.summary.total_trades_inserted} trades inserted so far`
                  : ''}
              </div>
            </>
          )}

          {(job.status === 'completed' ||
            job.status === 'failed' ||
            job.status === 'cancelled' ||
            job.status === 'orphaned') && (
            <div
              style={{
                marginTop: '6px',
                fontSize: '12px',
                color: 'var(--text-secondary)',
              }}
            >
              Succeeded {job.summary?.succeeded ?? 0} · Failed{' '}
              {job.summary?.failed ?? 0} · Trades inserted{' '}
              {job.summary?.total_trades_inserted ?? 0}
              {job.summary?.total_elapsed_s != null && (
                <span style={{ color: 'var(--text-muted)' }}>
                  {' '}· elapsed {job.summary.total_elapsed_s}s
                </span>
              )}
              {job.updated_at && (
                <span style={{ color: 'var(--text-muted)' }}>
                  {' '}· finished {fmtAge(job.updated_at)}
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
          {job.status !== 'running' && (
            <button
              style={btnSecondary}
              disabled={deleteMut.isPending}
              onClick={() => {
                if (confirm(`Delete update job #${job.id}?`)) {
                  deleteMut.mutate(job.id);
                }
              }}
            >
              {deleteMut.isPending ? 'Deleting…' : 'Delete'}
            </button>
          )}
        </div>
      </div>

      {expanded && perRows.length > 0 && (
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
          {perRows.map(([sid, info], i) => {
            const btCount = info.backtest?.count;
            const algoCount = info.algo?.count;
            const inserted = (btCount ?? 0) + (algoCount ?? 0);
            return (
              <div
                key={sid}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  padding: '6px 0',
                  borderBottom:
                    i < perRows.length - 1 ? '1px solid var(--border)' : 'none',
                  gap: '2px',
                }}
              >
                <div
                  style={{ display: 'flex', justifyContent: 'space-between' }}
                >
                  <span style={{ flex: 1, minWidth: 0 }}>sid {sid}</span>
                  <span
                    style={{ color: 'var(--text-muted)', marginRight: '8px' }}
                  >
                    {info.elapsed_s != null ? `${info.elapsed_s}s` : ''}
                  </span>
                  <span
                    style={{
                      color:
                        info.status === 'ok'
                          ? 'var(--green)'
                          : info.status === 'error'
                          ? 'var(--red)'
                          : 'var(--text-muted)',
                      fontWeight: 600,
                      minWidth: '90px',
                      textAlign: 'right',
                    }}
                  >
                    {info.status}
                    {inserted ? ` (+${inserted})` : ''}
                  </span>
                </div>
                {(info.backtest || info.algo || info.error) && (
                  <div
                    style={{
                      fontSize: '11px',
                      color: 'var(--text-muted)',
                      paddingLeft: '8px',
                    }}
                  >
                    {info.backtest && (
                      <span>
                        backtest: {info.backtest.status ?? '?'} (+
                        {info.backtest.count ?? 0})
                      </span>
                    )}
                    {info.backtest && info.algo && <span> · </span>}
                    {info.algo && (
                      <span>
                        algo: {info.algo.status ?? '?'} (+
                        {info.algo.count ?? 0})
                      </span>
                    )}
                    {info.error && (
                      <div style={{ color: 'var(--red)' }}>
                        error: {info.error}
                      </div>
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

/**
 * Convert a `<input type="datetime-local">` value (no timezone, e.g.
 * "2026-06-05T13:30") into an ISO UTC string. The input is interpreted
 * as UTC, NOT local — the form label makes this explicit.
 */
function localInputToUtcIso(v: string): string | null {
  if (!v) return null;
  // Already has seconds? Use as-is. Otherwise add :00.
  const withSeconds = /T\d{2}:\d{2}:\d{2}$/.test(v) ? v : `${v}:00`;
  return `${withSeconds}Z`;
}

/**
 * Format a Date as a `datetime-local` value in UTC (YYYY-MM-DDTHH:mm).
 * Used by the "Today RTH" / "Last 1h" preset buttons.
 */
function dateToUtcLocalInput(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, '0');
  return (
    `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}` +
    `T${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`
  );
}

function WindowBackfillCard() {
  const startMut = useStartUpdateJob();
  const [sidsRaw, setSidsRaw] = useState('');
  const [windowStart, setWindowStart] = useState('');
  const [windowEnd, setWindowEnd] = useState('');
  const [lastResult, setLastResult] = useState<{
    job_id: number;
    strategy_count: number;
  } | null>(null);

  const parsedSids = sidsRaw
    .split(/[,\s]+/)
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => Number(s))
    .filter((n) => Number.isInteger(n) && n > 0);

  const startIso = localInputToUtcIso(windowStart);
  const endIso = localInputToUtcIso(windowEnd);
  const validRange =
    startIso && endIso && new Date(endIso) > new Date(startIso);
  const canSubmit =
    parsedSids.length > 0 && validRange && !startMut.isPending;

  const presetTodayRTH = () => {
    // RTH = 13:30-20:00 UTC. Use today's date, RTH open → now (or RTH close).
    const now = new Date();
    const open = new Date(
      Date.UTC(
        now.getUTCFullYear(),
        now.getUTCMonth(),
        now.getUTCDate(),
        13,
        30,
        0,
      ),
    );
    const close = new Date(
      Date.UTC(
        now.getUTCFullYear(),
        now.getUTCMonth(),
        now.getUTCDate(),
        20,
        0,
        0,
      ),
    );
    const end = now < close ? now : close;
    setWindowStart(dateToUtcLocalInput(open));
    setWindowEnd(dateToUtcLocalInput(end));
  };

  const presetLastNHours = (hours: number) => {
    const now = new Date();
    const start = new Date(now.getTime() - hours * 3_600_000);
    setWindowStart(dateToUtcLocalInput(start));
    setWindowEnd(dateToUtcLocalInput(now));
  };

  const onSubmit = () => {
    if (!canSubmit || !startIso || !endIso) return;
    const label =
      parsedSids.length === 1
        ? `Window backfill sid ${parsedSids[0]}`
        : `Window backfill ${parsedSids.length} strategies`;
    startMut.mutate(
      {
        mode: 'window',
        strategy_ids: parsedSids,
        window_start: startIso,
        window_end: endIso,
        name: label,
      },
      {
        onSuccess: (res) => {
          setLastResult({
            job_id: res.job_id,
            strategy_count: res.strategy_count,
          });
        },
      },
    );
  };

  const inputStyle: React.CSSProperties = {
    padding: '6px 10px',
    borderRadius: '6px',
    border: '1px solid var(--border)',
    background: 'var(--bg-input)',
    color: 'var(--text)',
    fontSize: '13px',
    fontFamily: 'monospace',
  };

  return (
    <Card>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          flexWrap: 'wrap',
          marginBottom: '10px',
        }}
      >
        <span style={{ fontSize: '15px', fontWeight: 600 }}>
          Manual Window Backfill (BT lane)
        </span>
        <span
          style={{
            background: 'var(--red-muted)',
            color: 'var(--red)',
            fontSize: '11px',
            padding: '2px 8px',
            borderRadius: '4px',
            fontWeight: 700,
            textTransform: 'uppercase',
          }}
          title="Over-produces ~17% extra trades vs UAD on canary sid 296 (2026-06-05). Do not trust output until cooldown/position-state serialization is fixed."
        >
          WIP — do not trust
        </span>
      </div>
      <div
        style={{
          background: 'var(--red-muted)',
          color: 'var(--red)',
          border: '1px solid var(--red)',
          borderRadius: '6px',
          padding: '8px 12px',
          fontSize: '12px',
          lineHeight: 1.5,
          marginBottom: '14px',
        }}
      >
        <strong>⚠ WORK IN PROGRESS — parity bug.</strong> On the
        2026-06-05 sid 296 canary, this produced 433 in-window trades
        while Update All Data produced 359 — ~17% over-production.
        Suspected cause: snapshot-resume re-initializes the post-exit
        cooldown counter to 0, allowing rapid re-entries UAD would
        suppress. Use <em>Update All Data</em> for trustworthy
        backfills until this is fixed.
      </div>
      <p
        style={{
          fontSize: '12px',
          color: 'var(--text-secondary)',
          margin: '0 0 14px 0',
          lineHeight: 1.5,
        }}
      >
        Re-run the backtest for one or more strategies over an explicit
        time window. Inserts BT trades into the {'{[start, end)}'} range
        only — does NOT touch the algo lane, KPIs, snapshot, or
        last_recompute_until_ts.
      </p>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr',
          gap: '12px',
          marginBottom: '12px',
        }}
      >
        <div>
          <label
            style={{
              display: 'block',
              fontSize: '12px',
              color: 'var(--text-muted)',
              marginBottom: '4px',
            }}
          >
            Strategy IDs (comma- or space-separated)
          </label>
          <input
            style={{ ...inputStyle, width: '100%' }}
            placeholder="e.g. 302  or  263, 267, 290"
            value={sidsRaw}
            onChange={(e) => setSidsRaw(e.target.value)}
          />
          {sidsRaw && (
            <div
              style={{
                fontSize: '11px',
                color: 'var(--text-muted)',
                marginTop: '3px',
              }}
            >
              parsed: {parsedSids.length} sid
              {parsedSids.length === 1 ? '' : 's'}
              {parsedSids.length > 0 && ` → [${parsedSids.join(', ')}]`}
            </div>
          )}
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '12px',
          }}
        >
          <div>
            <label
              style={{
                display: 'block',
                fontSize: '12px',
                color: 'var(--text-muted)',
                marginBottom: '4px',
              }}
            >
              Window start (UTC)
            </label>
            <input
              type="datetime-local"
              style={{ ...inputStyle, width: '100%' }}
              value={windowStart}
              onChange={(e) => setWindowStart(e.target.value)}
            />
          </div>
          <div>
            <label
              style={{
                display: 'block',
                fontSize: '12px',
                color: 'var(--text-muted)',
                marginBottom: '4px',
              }}
            >
              Window end (UTC)
            </label>
            <input
              type="datetime-local"
              style={{ ...inputStyle, width: '100%' }}
              value={windowEnd}
              onChange={(e) => setWindowEnd(e.target.value)}
            />
          </div>
        </div>

        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
          <button
            style={{ ...btnSecondary, fontSize: '11px' }}
            onClick={presetTodayRTH}
            type="button"
          >
            Today RTH (13:30 → now)
          </button>
          <button
            style={{ ...btnSecondary, fontSize: '11px' }}
            onClick={() => presetLastNHours(1)}
            type="button"
          >
            Last 1h
          </button>
          <button
            style={{ ...btnSecondary, fontSize: '11px' }}
            onClick={() => presetLastNHours(4)}
            type="button"
          >
            Last 4h
          </button>
          <button
            style={{ ...btnSecondary, fontSize: '11px' }}
            onClick={() => presetLastNHours(24)}
            type="button"
          >
            Last 24h
          </button>
        </div>
      </div>

      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          flexWrap: 'wrap',
        }}
      >
        <button
          style={{
            ...btnSecondary,
            background: canSubmit
              ? 'var(--green-muted)'
              : 'var(--bg-input)',
            color: canSubmit ? 'var(--green)' : 'var(--text-muted)',
            border: 'none',
            fontWeight: 600,
            opacity: startMut.isPending ? 0.6 : 1,
            cursor: canSubmit ? 'pointer' : 'not-allowed',
          }}
          disabled={!canSubmit}
          onClick={onSubmit}
        >
          {startMut.isPending ? 'Submitting…' : 'Run Window Backfill'}
        </button>
        {!validRange && (windowStart || windowEnd) && (
          <span style={{ fontSize: '11px', color: 'var(--red)' }}>
            window end must be after window start
          </span>
        )}
        {startMut.isError && (
          <span style={{ fontSize: '11px', color: 'var(--red)' }}>
            {(startMut.error as any)?.message || 'request failed'}
          </span>
        )}
        {lastResult && (
          <span style={{ fontSize: '11px', color: 'var(--green)' }}>
            queued job #{lastResult.job_id} ({lastResult.strategy_count}{' '}
            sid
            {lastResult.strategy_count === 1 ? '' : 's'}) — watch progress
            below
          </span>
        )}
      </div>
    </Card>
  );
}

function StreamingToggleCard() {
  const { data, isLoading } = useSystemSettings();
  const setMut = useSetSystemSetting();
  const setting = data?.settings.find(
    (s) => s.key === 'data_worker_streaming_enabled',
  );
  const enabled = !!setting?.value;
  const source = setting?.source;
  const updatedAt = setting?.updated_at;

  if (isLoading || !setting) {
    return (
      <Card>
        <div style={{ padding: '12px', color: 'var(--text-muted)' }}>
          Loading streaming mode…
        </div>
      </Card>
    );
  }

  return (
    <Card>
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
          gap: '16px',
          flexWrap: 'wrap',
        }}
      >
        <div style={{ flex: 1, minWidth: 0 }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              flexWrap: 'wrap',
              marginBottom: '6px',
            }}
          >
            <span style={{ fontSize: '15px', fontWeight: 600 }}>
              Streaming Mode (Data Worker)
            </span>
            <span
              style={{
                background: enabled ? 'var(--green-muted)' : 'var(--bg-input)',
                color: enabled ? 'var(--green)' : 'var(--text-muted)',
                fontSize: '11px',
                padding: '2px 8px',
                borderRadius: '4px',
                fontWeight: 600,
                textTransform: 'uppercase',
              }}
            >
              {enabled ? 'AUTO (streaming on)' : 'MANUAL (on-demand only)'}
            </span>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              source: {source}
              {updatedAt && ` · updated ${fmtAge(updatedAt)}`}
            </span>
          </div>
          <p
            style={{
              fontSize: '12px',
              color: 'var(--text-secondary)',
              margin: 0,
              lineHeight: 1.5,
            }}
          >
            {setting.description}
          </p>
          <p
            style={{
              fontSize: '11px',
              color: 'var(--text-muted)',
              marginTop: '6px',
              marginBottom: 0,
            }}
          >
            Data Worker re-reads this every ~30s. Changes take effect within a
            minute without a Railway redeploy.
          </p>
        </div>
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '6px',
            alignItems: 'flex-end',
          }}
        >
          <button
            style={{
              ...btnSecondary,
              background: enabled ? 'var(--red-muted)' : 'var(--green-muted)',
              color: enabled ? 'var(--red)' : 'var(--green)',
              border: 'none',
              fontWeight: 600,
              opacity: setMut.isPending ? 0.6 : 1,
              minWidth: '180px',
            }}
            disabled={setMut.isPending}
            onClick={() => {
              const next = !enabled;
              const verb = next ? 'ENABLE' : 'DISABLE';
              if (
                !confirm(
                  `${verb} streaming mode?\n\n` +
                    (next
                      ? 'Data Worker will resume continuous backtest catchup, snapshot flush, and KPI recompute. Strategies will keep KPIs fresh automatically.'
                      : 'Data Worker will stop continuous catchup. Trades + KPIs will only update when you trigger an /admin/update-jobs run. Bar ingest + live alerts are unaffected.'),
                )
              ) {
                return;
              }
              setMut.mutate({
                key: 'data_worker_streaming_enabled',
                value: next,
              });
            }}
          >
            {setMut.isPending
              ? 'Saving…'
              : enabled
              ? 'Switch to MANUAL'
              : 'Switch to AUTO'}
          </button>
        </div>
      </div>
    </Card>
  );
}

export default function UpdateJobsAdminPage() {
  const searchParams = useSearchParams();
  const highlightId = searchParams.get('highlight');
  const [filter, setFilter] = useState<UpdateJobStatus | 'all'>('all');
  const [expandedIds, setExpandedIds] = useState<Set<number>>(
    new Set(highlightId ? [Number(highlightId)] : []),
  );

  const { data, error, isLoading } = useUpdateJobsList();
  const cleanupMut = useCleanupOrphans();

  const filtered = (data || []).filter((j) =>
    filter === 'all' ? true : j.status === filter,
  );

  const toggleExpand = (jobId: number) => {
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
          <h1
            style={{ fontSize: '24px', fontWeight: 700, marginBottom: '4px' }}
          >
            Update Jobs (DB-persisted)
          </h1>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
            Background UAD jobs that survive worker restarts. Use this page
            instead of /jobs when running long-tail bulk updates you want to
            walk away from. Orphan cleanup runs on api boot.
          </p>
        </div>
        <button
          style={btnSecondary}
          disabled={cleanupMut.isPending}
          onClick={() => cleanupMut.mutate()}
          title="Mark any zombie 'running' rows as orphaned (already runs at api boot — this is a manual override)"
        >
          {cleanupMut.isPending ? 'Cleaning…' : 'Clean orphans'}
        </button>
      </div>

      <div style={{ marginBottom: '16px' }}>
        <StreamingToggleCard />
      </div>

      <div style={{ marginBottom: '16px' }}>
        <WindowBackfillCard />
      </div>

      <div
        style={{
          display: 'flex',
          gap: '6px',
          marginBottom: '16px',
          flexWrap: 'wrap',
        }}
      >
        {(
          [
            'all',
            'running',
            'queued',
            'completed',
            'failed',
            'cancelled',
            'orphaned',
          ] as const
        ).map((s) => (
          <button
            key={s}
            style={filterButtonStyle(filter === s)}
            onClick={() => setFilter(s)}
          >
            {s.charAt(0).toUpperCase() + s.slice(1)}
          </button>
        ))}
      </div>

      {isLoading && (
        <div style={{ color: 'var(--text-muted)', padding: '24px' }}>
          Loading…
        </div>
      )}
      {error && (
        <Card>
          <div style={{ color: 'var(--red)', padding: '12px' }}>
            Failed to load update jobs:{' '}
            {(error as any)?.message || String(error)}
          </div>
        </Card>
      )}
      {data && filtered.length === 0 && (
        <Card>
          <div
            style={{
              padding: '24px',
              textAlign: 'center',
              color: 'var(--text-muted)',
            }}
          >
            No {filter === 'all' ? '' : filter} update jobs yet. Trigger one
            from My Strategies → bulk select → Update New Data / Update All
            Data.
          </div>
        </Card>
      )}
      {filtered.length > 0 && (
        <div
          style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}
        >
          {filtered.map((job) => (
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
