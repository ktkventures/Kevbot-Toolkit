/**
 * Update-jobs query + mutation hooks.
 *
 * Parallel to useJobs.ts (in-memory `/api/jobs/*`), but for the
 * DB-persisted `/api/update-jobs/*` system. New since Phase 2 — survives
 * worker restart via orphan cleanup on api boot.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

export type UpdateJobMode = 'all' | 'new';
export type UpdateJobScope = 'single' | 'bulk';
export type UpdateJobStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'orphaned';

export interface UpdateJobPerStrategy {
  status: 'ok' | 'error' | string;
  elapsed_s?: number | null;
  backtest?: { status?: string | null; count?: number | null };
  algo?: { status?: string | null; count?: number | null };
  error?: string | null;
}

export interface UpdateJobSummary {
  succeeded?: number;
  failed?: number;
  cancelled?: boolean;
  total_trades_inserted?: number;
  total_elapsed_s?: number;
}

export interface UpdateJobProgress {
  current_step: number;
  total_steps: number;
  current_sid: number | null;
  current_phase: string | null;
  per_strategy: Record<string, UpdateJobPerStrategy>;
}

export interface UpdateJobListItem {
  id: number;
  user_id: string;
  name: string;
  mode: UpdateJobMode;
  scope: UpdateJobScope;
  strategy_ids: number[];
  status: UpdateJobStatus;
  progress?: UpdateJobProgress;
  summary?: UpdateJobSummary;
  error?: string | null;
  created_at: string;
  updated_at: string;
}

export interface UpdateJobProgressResponse {
  job_id: number;
  status: UpdateJobStatus;
  mode: UpdateJobMode;
  scope: UpdateJobScope;
  strategy_ids: number[];
  current_step: number;
  total_steps: number;
  current_sid: number | null;
  current_phase: string | null;
  per_strategy: Record<string, UpdateJobPerStrategy>;
  summary?: UpdateJobSummary;
  error?: string | null;
  created_at: string;
  updated_at: string;
}

export function useUpdateJobsList() {
  return useQuery({
    queryKey: ['update-jobs', 'list'],
    queryFn: () => apiFetch<UpdateJobListItem[]>('/api/update-jobs'),
    refetchInterval: (query) => {
      const data = query.state.data;
      if (Array.isArray(data) && data.some((j) =>
          j.status === 'running' || j.status === 'queued')) {
        return 2000;
      }
      return false;
    },
    placeholderData: (prev) => prev,
  });
}

export function useUpdateJobProgress(jobId: number | null) {
  return useQuery({
    queryKey: ['update-job', jobId],
    queryFn: () =>
      apiFetch<UpdateJobProgressResponse>(`/api/update-jobs/progress/${jobId}`),
    enabled: jobId !== null,
    refetchInterval: (query) =>
      query.state.data?.status === 'running' ||
      query.state.data?.status === 'queued'
        ? 1000
        : false,
  });
}

export function useStartUpdateJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (input: {
      mode: UpdateJobMode;
      strategy_ids: number[];
      name?: string;
    }) =>
      apiFetch<{
        job_id: number;
        status: string;
        mode: UpdateJobMode;
        scope: UpdateJobScope;
        strategy_count: number;
      }>('/api/update-jobs/run', {
        method: 'POST',
        body: JSON.stringify(input),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['update-jobs'] });
    },
  });
}

export function useCancelUpdateJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: number) =>
      apiFetch<{ status: string }>(`/api/update-jobs/cancel/${jobId}`, {
        method: 'POST',
      }),
    onSuccess: (_d, jobId) => {
      qc.invalidateQueries({ queryKey: ['update-job', jobId] });
      qc.invalidateQueries({ queryKey: ['update-jobs'] });
    },
  });
}

export function useDeleteUpdateJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: number) =>
      apiFetch<{ status: string }>(`/api/update-jobs/${jobId}`, {
        method: 'DELETE',
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['update-jobs'] });
    },
  });
}

export function useCleanupOrphans() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () =>
      apiFetch<{ orphaned: number; ids: number[] }>(
        '/api/update-jobs/cleanup-orphans',
        { method: 'POST' },
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['update-jobs'] });
    },
  });
}
