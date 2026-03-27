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

export function usePortfolios() {
  return useQuery({
    queryKey: ['portfolios'],
    queryFn: () => apiFetch<PortfolioDTO[]>('/api/portfolios'),
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
