/**
 * React Query hooks for strategies.
 */

import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

export interface StrategyDTO {
  id: number;
  name: string;
  symbol: string;
  direction: 'LONG' | 'SHORT';
  timeframe: string;
  tags?: string[];
  forward_testing?: boolean;
  forward_test_start?: string;
  alert_tracking_enabled?: boolean;
  kpis?: Record<string, number>;
  stored_trades?: any[];
  equity_curve_data?: {
    exit_times: string[];
    cumulative_r: number[];
    boundary_index?: number | null;
  };
  created_at?: string;
  updated_at?: string;
  // Config fields (flattened from JSONB)
  entry_trigger_confluence_id?: string;
  exit_trigger_confluence_ids?: string[];
  confluence?: string[];
  stop_config?: Record<string, any>;
  target_config?: Record<string, any>;
  trading_session?: string;
  data_days?: number;
  [key: string]: any;
}

export interface TradeDTO {
  entry_time?: string;
  exit_time?: string;
  direction?: string;
  entry_price?: number;
  exit_price?: number;
  stop_price?: number;
  target_price?: number;
  r_multiple: number;
  win: boolean;
  exit_reason?: string;
  exec_type?: string;
  bars_held?: number;
  entry_trigger?: string;
}

export interface ForwardTestDTO {
  backtest_trades: TradeDTO[];
  forward_trades: TradeDTO[];
  forward_test_start: string | null;
}

export function useStrategies() {
  return useQuery({
    queryKey: ['strategies'],
    queryFn: () => apiFetch<StrategyDTO[]>('/api/strategies'),
  });
}

export function useStrategy(id: number | null) {
  return useQuery({
    queryKey: ['strategy', id],
    queryFn: () => apiFetch<StrategyDTO>(`/api/strategies/${id}`),
    enabled: id !== null,
  });
}

export function useStrategyTrades(id: number | null, useStored = true) {
  return useQuery({
    queryKey: ['strategy-trades', id, useStored],
    queryFn: () =>
      apiFetch<TradeDTO[]>(`/api/strategies/${id}/trades?use_stored=${useStored}`),
    enabled: id !== null,
  });
}

export function useStrategyForwardTest(id: number | null) {
  return useQuery({
    queryKey: ['strategy-forward-test', id],
    queryFn: () => apiFetch<ForwardTestDTO>(`/api/strategies/${id}/forward-test`),
    enabled: id !== null,
  });
}

export function useStrategyKPIs(id: number | null) {
  return useQuery({
    queryKey: ['strategy-kpis', id],
    queryFn: () =>
      apiFetch<{ kpis: Record<string, number>; secondary_kpis: Record<string, any> }>(
        `/api/strategies/${id}/kpis`
      ),
    enabled: id !== null,
  });
}
