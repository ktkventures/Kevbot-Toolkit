/**
 * Mass builder query + mutation hooks.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

export function useMassResults() {
  return useQuery({
    queryKey: ['mass-results'],
    queryFn: () => apiFetch<any[]>('/api/mass-builder/results'),
  });
}

export function useMassResult(id: number | null) {
  return useQuery({
    queryKey: ['mass-result', id],
    queryFn: () => apiFetch<any>(`/api/mass-builder/results/${id}`),
    enabled: id !== null,
  });
}

export function useMassProgress(searchId: number | null) {
  return useQuery({
    queryKey: ['mass-progress', searchId],
    queryFn: () => apiFetch<{
      search_id: number;
      status: string;
      progress: number;
      total: number;
      current_label: string;
    }>(`/api/mass-builder/progress/${searchId}`),
    enabled: searchId !== null,
    refetchInterval: (query) =>
      query.state.data?.status === 'running' ? 2000 : false,
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
