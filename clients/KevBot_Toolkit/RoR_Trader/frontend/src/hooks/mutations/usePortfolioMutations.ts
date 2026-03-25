/**
 * Portfolio mutation hooks.
 */

import { useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

export function useCreatePortfolio() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (portfolio: Record<string, any>) =>
      apiFetch('/api/portfolios', { method: 'POST', body: JSON.stringify(portfolio) }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['portfolios'] }); },
  });
}

export function useUpdatePortfolio() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, portfolio }: { id: number; portfolio: Record<string, any> }) =>
      apiFetch(`/api/portfolios/${id}`, { method: 'PUT', body: JSON.stringify(portfolio) }),
    onSuccess: (_d, { id }) => {
      qc.invalidateQueries({ queryKey: ['portfolios'] });
      qc.invalidateQueries({ queryKey: ['portfolio', id] });
    },
  });
}

export function useDeletePortfolio() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => apiFetch(`/api/portfolios/${id}`, { method: 'DELETE' }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['portfolios'] }); },
  });
}

export function useDuplicatePortfolio() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => apiFetch(`/api/portfolios/${id}/duplicate`, { method: 'POST' }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['portfolios'] }); },
  });
}

export function useAddLedgerEntry() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, entry }: { id: number; entry: Record<string, any> }) =>
      apiFetch(`/api/portfolios/${id}/account/deposit`, { method: 'POST', body: JSON.stringify(entry) }),
    onSuccess: (_d, { id }) => { qc.invalidateQueries({ queryKey: ['portfolio-account', id] }); },
  });
}

export function useRemoveLedgerEntry() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ portfolioId, entryId }: { portfolioId: number; entryId: string }) =>
      apiFetch(`/api/portfolios/${portfolioId}/account/ledger/${entryId}`, { method: 'DELETE' }),
    onSuccess: (_d, { portfolioId }) => { qc.invalidateQueries({ queryKey: ['portfolio-account', portfolioId] }); },
  });
}
