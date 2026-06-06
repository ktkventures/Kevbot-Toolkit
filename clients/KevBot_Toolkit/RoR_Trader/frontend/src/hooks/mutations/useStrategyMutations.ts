/**
 * Strategy mutation hooks — create, update, delete, duplicate, bulk-delete.
 */

import { useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

export function useCreateStrategy() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (strategy: Record<string, any>) =>
      apiFetch('/api/strategies', {
        method: 'POST',
        body: JSON.stringify(strategy),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['strategies'] });
    },
  });
}

export function useUpdateStrategy() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, strategy }: { id: number; strategy: Record<string, any> }) =>
      apiFetch(`/api/strategies/${id}`, {
        method: 'PUT',
        body: JSON.stringify(strategy),
      }),
    onSuccess: (_data, { id }) => {
      queryClient.invalidateQueries({ queryKey: ['strategies'] });
      queryClient.invalidateQueries({ queryKey: ['strategy', id] });
    },
  });
}

export function useDeleteStrategy() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: number) =>
      apiFetch(`/api/strategies/${id}`, { method: 'DELETE' }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['strategies'] });
    },
  });
}

export function useDuplicateStrategy() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: number) =>
      apiFetch(`/api/strategies/${id}/duplicate`, { method: 'POST' }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['strategies'] });
    },
  });
}

export function useRefreshStrategy() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: number) =>
      apiFetch<{ status: string; trades: number; kpis: Record<string, number> }>(
        `/api/strategies/${id}/refresh`,
        { method: 'POST' }
      ),
    onSuccess: (_data, id) => {
      queryClient.invalidateQueries({ queryKey: ['strategies'] });
      queryClient.invalidateQueries({ queryKey: ['strategy', id] });
      queryClient.invalidateQueries({ queryKey: ['strategy-trades', id] });
      queryClient.invalidateQueries({ queryKey: ['strategy-forward-test', id] });
      queryClient.invalidateQueries({ queryKey: ['strategy-kpis', id] });
    },
  });
}

export function useBulkDeleteStrategies() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (ids: number[]) =>
      apiFetch('/api/strategies/bulk-delete', {
        method: 'POST',
        body: JSON.stringify(ids),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['strategies'] });
    },
  });
}

/**
 * Admin override for a strategy's ``forward_test_start``. Unblocks the
 * "recreate old strategy under current schema, then restore original start
 * date" workflow.
 *
 * Pass `forwardTestStart: null` (or empty string) to clear. No refresh is
 * triggered — caller should invoke useRefreshStrategy afterward if they
 * want stored_trades + equity_curve_data regenerated against the new
 * boundary.
 */
/**
 * Toggle whether the data-worker maintains an auto-snapshot for this
 * strategy. Setting `enabled=false` parks the strategy in the snapshot
 * lane — useful for known-broken strategies (no baseline trades,
 * persistent errors) that were bogging the worker down. Live alerts
 * are unaffected; only the snapshot lane is gated. 2026-05-27.
 */
export function useSetSnapshotSubscription() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, enabled }: { id: number; enabled: boolean }) =>
      apiFetch(`/api/strategies/${id}/snapshot-subscription`, {
        method: 'PATCH',
        body: JSON.stringify({ enabled }),
      }),
    onSuccess: (_data, { id }) => {
      queryClient.invalidateQueries({ queryKey: ['admin', 'strategy-health'] });
      queryClient.invalidateQueries({ queryKey: ['strategy', id] });
    },
  });
}


export function useSetForwardTestStart() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, forwardTestStart }: { id: number; forwardTestStart: string | null }) =>
      apiFetch(`/api/strategies/${id}/forward-test-start`, {
        method: 'PATCH',
        body: JSON.stringify({ forward_test_start: forwardTestStart }),
      }),
    onSuccess: (_data, { id }) => {
      queryClient.invalidateQueries({ queryKey: ['strategies'] });
      queryClient.invalidateQueries({ queryKey: ['strategy', id] });
      queryClient.invalidateQueries({ queryKey: ['strategy-forward-test', id] });
    },
  });
}

/**
 * Update strategy data on backtest + algo lanes.
 *
 * 2026-06-06: migrated from the sync `/api/strategies/{id}/update` endpoint
 * to the async `/api/update-jobs/run` path (mirrors the /strategies bulk
 * page). The sync endpoint blocked the HTTP request for the entire UAD,
 * which Railway's load balancer terminates at ~60-90s with "Failed to
 * fetch" — making the detail-page Update buttons useless for any strategy
 * that takes longer than that to recompute. The async path:
 *   1. POST /api/update-jobs/run — returns job_id immediately
 *   2. Poll /api/update-jobs/progress/{job_id} every 1.5s
 *   3. Resolve when status='completed' (or surface failure)
 * The hook returns the same `{ backtest, algo }` shape as before so
 * callers (StrategyDetailPage `handleUpdate`) don't have to change.
 */
export function useUpdateStrategyLanes() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ id, mode }: { id: number; mode: 'all' | 'new' }) => {
      // 1. Kick off the job.
      const kickoff = await apiFetch<{
        job_id: number;
        status: string;
      }>('/api/update-jobs/run', {
        method: 'POST',
        body: JSON.stringify({
          mode,
          strategy_ids: [id],
          name: `Update ${mode} (sid ${id})`,
        }),
      });
      const jobId = kickoff.job_id;
      if (!jobId) {
        throw new Error(
          `update-jobs/run returned no job_id: ${JSON.stringify(kickoff)}`,
        );
      }
      // 2. Poll for completion. 10-min hard cap to prevent infinite
      // hang if the worker thread dies silently. The DB-side orphan
      // cleanup will mark such jobs 'orphaned' eventually but we don't
      // want the UI hung waiting.
      const POLL_INTERVAL_MS = 1500;
      const MAX_WAIT_MS = 10 * 60 * 1000;
      const start = Date.now();
      // eslint-disable-next-line no-constant-condition
      while (true) {
        if (Date.now() - start > MAX_WAIT_MS) {
          throw new Error(
            `Update job ${jobId} still running after 10 min — check /admin/update-jobs`,
          );
        }
        await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
        const progress = await apiFetch<{
          status: string;
          per_strategy: Record<string, any>;
          error?: string | null;
        }>(`/api/update-jobs/progress/${jobId}`);
        const status = progress.status;
        if (
          status === 'completed' ||
          status === 'failed' ||
          status === 'cancelled' ||
          status === 'orphaned'
        ) {
          // 3. Extract per-strategy result + reshape to legacy
          // `{ backtest, algo }` shape so the existing handleUpdate in
          // StrategyDetailPage doesn't need to change.
          const per = progress.per_strategy?.[String(id)] || {};
          const bt = per.backtest || null;
          const algo = per.algo || null;
          // The per_strategy shape from update_jobs is
          //   { status, elapsed_s, backtest: { status, count }, algo: { status, count } }
          // while legacy callers expect the FULL recompute_and_persist_*
          // dict shape (status, reason, inserted, etc). Translate:
          //   count → inserted
          // The other fields the legacy formatter checks (`reason`,
          // `inserted`) gracefully fall through to undefined which the
          // formatter handles.
          const translate = (r: any) =>
            r
              ? {
                  status: r.status,
                  inserted: r.count ?? 0,
                  reason: r.reason,
                }
              : null;
          if (status !== 'completed') {
            // Surface the failure in the per-lane result so the
            // existing error formatter renders it.
            const err = progress.error || `job ${status}`;
            return {
              backtest: bt
                ? translate(bt)
                : { status: 'error', reason: err, inserted: 0 },
              algo: algo
                ? translate(algo)
                : { status: 'error', reason: err, inserted: 0 },
            };
          }
          return { backtest: translate(bt), algo: translate(algo) };
        }
        // else: still running/queued — keep polling
      }
    },
    onSuccess: (_data, { id }) => {
      queryClient.invalidateQueries({ queryKey: ['strategy', id] });
      queryClient.invalidateQueries({ queryKey: ['strategy-trades', id] });
      queryClient.invalidateQueries({ queryKey: ['strategy-forward-test', id] });
      queryClient.invalidateQueries({ queryKey: ['strategy-divergence', id] });
      queryClient.invalidateQueries({ queryKey: ['strategy-kpis', id] });
      // Refresh the update-jobs list too so the admin page shows the
      // new job if the user navigates there.
      queryClient.invalidateQueries({ queryKey: ['update-jobs'] });
    },
  });
}
