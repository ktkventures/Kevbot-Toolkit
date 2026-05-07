/**
 * Recompute-jobs query + mutation hooks.
 *
 * Mirrors the Mass Builder pattern (useMassBuilder.ts) but scoped to
 * the simpler recompute case (no checkpoint/resume, no orphan recovery).
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

export type JobType = 'append_recent' | 'full_recompute';
export type JobStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface PerStrategyResult {
  strategy_id: number;
  strategy_name: string;
  status: string | null;
  inserted: number;
  engine_path: 'windowed' | 'full' | null;
  bars_processed: number | null;
  window_days: number | null;
  reason: string | null;  // skip reason: 'recently_processed' | 'no_recent_exits' | etc
  elapsed_s: number;
  error: string | null;
}

export interface RecomputeJob {
  id: string;
  user_id: string;
  strategy_ids: number[];
  job_type: JobType;
  status: JobStatus;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  cancelled: boolean;
  current_strategy_idx: number;
  current_strategy_id: number | null;
  current_strategy_name: string | null;
  progress_label: string;
  per_strategy_results: PerStrategyResult[];
  summary: {
    total_strategies: number;
    total_inserted: number;
    total_skipped: number;
    total_failed: number;
    total_appended: number;
  };
  error: string | null;
}

export function useJobsList(status?: JobStatus | null) {
  return useQuery({
    queryKey: ['recompute-jobs', status ?? 'all'],
    queryFn: () =>
      apiFetch<{ jobs: RecomputeJob[]; total: number | null }>(
        `/api/jobs${status ? `?status=${status}` : ''}`,
      ),
    // Poll the list every 2s while any job is running, so list-page
    // status transitions show without manual refresh. Stop when idle.
    refetchInterval: (query) => {
      const data = query.state.data;
      if (data && Array.isArray(data.jobs) &&
          data.jobs.some((j) => j.status === 'running' || j.status === 'queued')) {
        return 2000;
      }
      return false;
    },
    placeholderData: (prev) => prev,
  });
}

export function useJobStatus(jobId: string | null) {
  return useQuery({
    queryKey: ['recompute-job', jobId],
    queryFn: () => apiFetch<RecomputeJob>(`/api/jobs/${jobId}`),
    enabled: jobId !== null,
    // 1s poll while running. Per-strategy progress updates come through
    // here so the user sees "now processing sid X of N".
    refetchInterval: (query) =>
      query.state.data?.status === 'running' ||
      query.state.data?.status === 'queued'
        ? 1000
        : false,
  });
}

export function useQueueRecomputeJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (input: { strategy_ids: number[]; job_type: JobType }) =>
      apiFetch<{
        job_id: string;
        status: string;
        job_type: JobType;
        strategy_count: number;
      }>('/api/jobs/recompute', {
        method: 'POST',
        body: JSON.stringify(input),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['recompute-jobs'] });
    },
  });
}

export function useCancelJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) =>
      apiFetch(`/api/jobs/${jobId}/cancel`, { method: 'POST' }),
    onSuccess: (_d, jobId) => {
      qc.invalidateQueries({ queryKey: ['recompute-job', jobId] });
      qc.invalidateQueries({ queryKey: ['recompute-jobs'] });
    },
  });
}
