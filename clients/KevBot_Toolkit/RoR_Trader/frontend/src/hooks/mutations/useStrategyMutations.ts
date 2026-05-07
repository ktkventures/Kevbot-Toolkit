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
 * Update strategy data on backtest + algo lanes (algo_model split 2026-05-07).
 * mode='all'  -> full backtest recompute + forward algo append
 * mode='new'  -> forward algo only (forward backtest append deferred)
 */
export function useUpdateStrategyLanes() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, mode }: { id: number; mode: 'all' | 'new' }) =>
      apiFetch(`/api/strategies/${id}/update?mode=${mode}`, {
        method: 'POST',
      }),
    onSuccess: (_data, { id }) => {
      queryClient.invalidateQueries({ queryKey: ['strategy', id] });
      queryClient.invalidateQueries({ queryKey: ['strategy-trades', id] });
      queryClient.invalidateQueries({ queryKey: ['strategy-forward-test', id] });
      queryClient.invalidateQueries({ queryKey: ['strategy-divergence', id] });
      queryClient.invalidateQueries({ queryKey: ['strategy-kpis', id] });
    },
  });
}
