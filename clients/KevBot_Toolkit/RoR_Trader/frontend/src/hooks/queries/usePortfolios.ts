/**
 * Portfolio query hooks.
 */

import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

export interface PortfolioDTO {
  id: number;
  name: string;
  strategies?: any[];
  requirement_set_id?: number;
  account?: any;
  kpis?: Record<string, number>;
  equity_curve_data?: any;
  tags?: string[];
  enabled?: boolean;
  [key: string]: any;
}

export function usePortfolios(opts?: { preview?: boolean }) {
  const preview = opts?.preview ?? false;
  return useQuery({
    queryKey: ['portfolios', { preview }],
    queryFn: () =>
      apiFetch<PortfolioDTO[]>(`/api/portfolios${preview ? '?preview=true' : ''}`),
    retry: 2, // Retry on 401 if token not yet in localStorage
  });
}

export function usePortfolio(id: number | null) {
  return useQuery({
    queryKey: ['portfolio', id],
    queryFn: () => apiFetch<PortfolioDTO>(`/api/portfolios/${id}`),
    enabled: id !== null,
  });
}

export function usePortfolioCompute(id: number | null, include: string[] = ['kpis']) {
  return useQuery({
    queryKey: ['portfolio-compute', id, include],
    queryFn: () =>
      apiFetch<Record<string, any>>(`/api/portfolios/${id}/compute`, {
        method: 'POST',
        body: JSON.stringify({ include }),
      }),
    enabled: id !== null,
  });
}

export function usePortfolioTrades(id: number | null) {
  return useQuery({
    queryKey: ['portfolio-trades', id],
    queryFn: () => apiFetch<any[]>(`/api/portfolios/${id}/trades`),
    enabled: id !== null,
  });
}

export function usePortfolioAnomalies(id: number | null) {
  return useQuery({
    queryKey: ['portfolio-anomalies', id],
    queryFn: () => apiFetch<any>(`/api/portfolios/${id}/anomalies`),
    enabled: id !== null,
  });
}

export function usePortfolioAccount(id: number | null) {
  return useQuery({
    queryKey: ['portfolio-account', id],
    queryFn: () => apiFetch<any>(`/api/portfolios/${id}/account`),
    enabled: id !== null,
  });
}

export function usePortfolioWorstCase(id: number | null) {
  return useQuery({
    queryKey: ['portfolio-worst-case', id],
    queryFn: () =>
      apiFetch<any>(`/api/portfolios/${id}/worst-case`, {
        method: 'POST',
        body: JSON.stringify({}),
      }),
    enabled: id !== null,
  });
}

export function usePortfolioCapitalUtilization(
  id: number | null,
  source: 'backtest' | 'alerts' = 'backtest',
) {
  return useQuery({
    queryKey: ['portfolio-capital-utilization', id, source],
    queryFn: () =>
      apiFetch<any>(`/api/portfolios/${id}/capital-utilization`, {
        method: 'POST',
        body: JSON.stringify({ source }),
      }),
    enabled: id !== null,
  });
}

export function usePortfolioOpenPositions(id: number | null) {
  return useQuery({
    queryKey: ['portfolio-open-positions', id],
    queryFn: () => apiFetch<any>(`/api/portfolios/${id}/open-positions`),
    enabled: id !== null,
  });
}

export function usePortfolioPreview(payload: Record<string, any> | null) {
  return useQuery({
    queryKey: ['portfolio-preview', payload],
    queryFn: () =>
      apiFetch<any>('/api/portfolios/preview', {
        method: 'POST',
        body: JSON.stringify(payload),
      }),
    enabled: payload !== null && Array.isArray(payload.strategies) && payload.strategies.length > 0,
  });
}

export function usePortfolioRecommendations(selectedIds: number[], max: number = 5) {
  return useQuery({
    queryKey: ['portfolio-recommendations', selectedIds.slice().sort(), max],
    queryFn: () =>
      apiFetch<any>('/api/portfolios/recommendations', {
        method: 'POST',
        body: JSON.stringify({ selected_strategy_ids: selectedIds, max }),
      }),
    enabled: Array.isArray(selectedIds),
  });
}

export function usePortfolioRequirementsCheck(id: number | null) {
  return useQuery({
    queryKey: ['portfolio-requirements-check', id],
    queryFn: () =>
      apiFetch<any>(`/api/portfolios/${id}/requirements/check`, {
        method: 'POST',
        body: JSON.stringify({}),
      }),
    enabled: id !== null,
  });
}
