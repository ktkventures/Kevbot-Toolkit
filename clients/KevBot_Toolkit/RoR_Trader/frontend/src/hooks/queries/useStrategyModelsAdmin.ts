/**
 * Hooks for the /api/admin/strategy-models/* endpoints.
 *
 * Used by /admin/live-models and /admin/backtest-models pages.
 * Read-only data — refetch on focus, modest stale time.
 */

import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

export type ModelKind = 'live' | 'backtest';
export type ModelStatus = 'available' | 'coming_soon' | 'unknown';

export interface StrategyModelRow {
  id: string;
  label: string;
  available: boolean;
  default: boolean;
  status: ModelStatus;
  description: string;
  strategies_using: number;
  sample_strategy_ids: number[];
  // backtest kind only — the registry doubles as the algo_model registry
  algo_default?: boolean;
  algo_strategies_using?: number;
}

export interface StrategyModelsListResponse {
  kind: ModelKind;
  default_id: string;
  algo_default_id?: string | null;
  rows: StrategyModelRow[];
}

export interface StrategyModelDetail {
  id: string;
  kind: ModelKind;
  label: string;
  available: boolean;
  default: boolean;
  status: ModelStatus;
  description: string;
  strategies: { id: number; name: string }[];
  strategies_using: number;
  // backtest kind only
  algo_default?: boolean;
  algo_strategies?: { id: number; name: string }[];
  algo_strategies_using?: number;
}

export function useStrategyModelsList(kind: ModelKind) {
  return useQuery<StrategyModelsListResponse>({
    queryKey: ['admin', 'strategy-models', kind],
    queryFn: () => apiFetch<StrategyModelsListResponse>(
      `/api/admin/strategy-models/${kind}`
    ),
    staleTime: 30_000,
  });
}

export function useStrategyModelDetail(kind: ModelKind, id: string | null) {
  return useQuery<StrategyModelDetail>({
    queryKey: ['admin', 'strategy-models', kind, id],
    queryFn: () => apiFetch<StrategyModelDetail>(
      `/api/admin/strategy-models/${kind}/${id}`
    ),
    enabled: !!id,
    staleTime: 30_000,
  });
}
