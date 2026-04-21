/**
 * Mass builder query + mutation hooks.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

export function useMassResults() {
  return useQuery({
    queryKey: ['mass-results'],
    queryFn: () => apiFetch<any[]>('/api/mass-builder/results'),
    // Refetch the list every 3s when any search is running, so status
    // transitions (running → completed) show up without a manual refresh.
    refetchInterval: (query) => {
      const data = query.state.data;
      if (Array.isArray(data) && data.some((s) => s?.status === 'running')) {
        return 3000;
      }
      return false;
    },
  });
}

export function useMassResult(id: string | number | null) {
  return useQuery({
    queryKey: ['mass-result', id],
    queryFn: () => apiFetch<any>(`/api/mass-builder/results/${id}`),
    enabled: id !== null,
  });
}

export function useMassProgress(searchId: string | number | null) {
  return useQuery({
    queryKey: ['mass-progress', searchId],
    queryFn: () => apiFetch<{
      search_id: string | number;
      status: string;
      progress: number;
      total: number;
      current_label: string;
      phase?: 'load' | 'prep' | 'backtest' | 'confluence' | 'save' | null;
      phase_detail?: string | null;
      inner_step?: number | null;
      inner_total?: number | null;
      conf_step?: number;
      conf_total?: number;
    }>(`/api/mass-builder/progress/${searchId}`),
    enabled: searchId !== null,
    // Poll 250ms while running for sub-second phase/bar updates; stop when
    // done. Terminal states: completed, cancelled, failed, orphaned (added
    // by Tier 1 — an API pod restart killed the worker thread). Anything
    // not 'running' or 'queued' stops the poll loop.
    refetchInterval: (query) =>
      query.state.data?.status === 'running' ? 250 : false,
  });
}

export function useRunMassSearch() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (config: Record<string, any>) =>
      apiFetch<{ search_id: number; status: string }>('/api/mass-builder/run', {
        method: 'POST',
        body: JSON.stringify(config),
      }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['mass-results'] }); },
  });
}

export function useCancelMassSearch() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (searchId: number) =>
      apiFetch(`/api/mass-builder/cancel/${searchId}`, { method: 'POST' }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['mass-results'] }); },
  });
}

export function useDeleteMassResult() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (searchId: number) =>
      apiFetch(`/api/mass-builder/results/${searchId}`, { method: 'DELETE' }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['mass-results'] }); },
  });
}
