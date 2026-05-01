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

export function useStrategy(id: number | null, dateRange?: string) {
  const params = dateRange && dateRange !== 'Strategy Default'
    ? `?date_range=${encodeURIComponent(dateRange)}`
    : '';
  return useQuery({
    queryKey: ['strategy', id, dateRange || 'Strategy Default'],
    queryFn: () => apiFetch<StrategyDTO>(`/api/strategies/${id}${params}`),
    enabled: id !== null,
    // M8.5 B+: poll so newly-persisted algo trades (from Ralph's exit
    // signals) surface on the Chart & Trades tab without a manual
    // "Update All Data" click.
    refetchInterval: 60_000,
  });
}

export function useStrategyTrades(id: number | null, useStored = true) {
  return useQuery({
    queryKey: ['strategy-trades', id, useStored],
    queryFn: () =>
      apiFetch<TradeDTO[]>(`/api/strategies/${id}/trades?use_stored=${useStored}`),
    enabled: id !== null,
    retry: 1,
    retryDelay: 2000,
    refetchInterval: 60_000,
  });
}

export function useStrategyForwardTest(id: number | null) {
  return useQuery({
    queryKey: ['strategy-forward-test', id],
    queryFn: () => apiFetch<ForwardTestDTO>(`/api/strategies/${id}/forward-test`),
    enabled: id !== null,
    retry: 1,
    retryDelay: 2000,
    refetchInterval: 60_000,
  });
}

export function useStrategyKPIs(id: number | null, dateRange?: string) {
  const params = dateRange && dateRange !== 'Strategy Default'
    ? `?date_range=${encodeURIComponent(dateRange)}`
    : '';
  return useQuery({
    queryKey: ['strategy-kpis', id, dateRange || 'Strategy Default'],
    queryFn: () =>
      apiFetch<{ kpis: Record<string, number>; secondary_kpis: Record<string, any> }>(
        `/api/strategies/${id}/kpis${params}`
      ),
    enabled: id !== null,
  });
}

export interface TriggerAnalysis {
  confluence_groups: { id: string; name: string; pack: string }[];
  entry_trigger: string;
  exit_triggers: string[];
  exit_breakdown: {
    exit_reason: string;
    trades: number;
    wins: number;
    losses: number;
    win_rate: number;
    total_r: number;
    avg_r: number;
    best_trade: number;
    worst_trade: number;
  }[];
  trade_distribution: { exit_reason: string; wins: number; losses: number }[];
}

export interface ChartDataResponse {
  chart_data: Record<string, any>[];
  indicators: string[];
}

export function useStrategyChartData(id: number | null) {
  return useQuery({
    queryKey: ['strategy-chart-data', id],
    queryFn: () => apiFetch<ChartDataResponse>(`/api/strategies/${id}/chart-data`),
    enabled: id !== null,
    staleTime: 300000, // 5 min cache — this is a slow endpoint
    retry: 1,
  });
}

export interface CacheBar {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  source: string;
}
export interface CacheBarsResponse {
  chart_data: CacheBar[];
  value_type: string;
  symbol?: string;
  timeframe?: string;
  tf_seconds?: number;
  window_start?: string;
  window_end?: string;
  row_count: number;
  notes: string[];
}

// M8.7: pulls OHLCV from live_bars (cache) for the Lab tab data-source
// toggle. value_type='latest' = post-rebroadcast WS values; 'first' =
// decision-time values (what live engine actually saw).
export function useStrategyCacheBars(
  id: number | null,
  valueType: 'latest' | 'first',
  enabled: boolean = true,
) {
  return useQuery({
    queryKey: ['strategy-cache-bars', id, valueType],
    queryFn: () => apiFetch<CacheBarsResponse>(
      `/api/strategies/${id}/cache-bars?value_type=${valueType}`
    ),
    enabled: enabled && id !== null,
    staleTime: 30000, // 30s — bars age into cache; refetch occasionally
    retry: 1,
  });
}

export interface ConfluenceChartData {
  bars: Record<string, any>[];
  indicator_columns: string[];
  state_column: string | null;
  needed_state: string;
  timeframe: string;
  condition: string;
}

export function useConfluenceChart(strategyId: number | null, condition: string | null) {
  return useQuery({
    queryKey: ['confluence-chart', strategyId, condition],
    queryFn: () => apiFetch<ConfluenceChartData>(
      `/api/strategies/${strategyId}/confluence-chart?condition=${encodeURIComponent(condition!)}`
    ),
    enabled: strategyId !== null && !!condition,
    staleTime: 300000,
    retry: 1,
  });
}

// =============================================================================
// TRADE DRILL-DOWN (1-second zoom)
// =============================================================================

export interface TradeZoomResponse {
  bars_1s: { time: string; open: number; high: number; low: number; close: number; volume: number }[];
  trade: Record<string, any>;
  indicators?: Record<string, { time: string; value: number }[]>;
  cb_confluence_timeline?: Record<string, { time: string; states: Record<string, string> }[]>;
  pb_states?: Record<string, string>;
  side: 'entry' | 'exit';
  timeframe: string;
  symbol: string;
}

export function useTradeZoom(strategyId: number | null, tradeIdx: number | null, side: 'entry' | 'exit') {
  return useQuery({
    queryKey: ['trade-zoom', strategyId, tradeIdx, side],
    queryFn: () => apiFetch<TradeZoomResponse>(
      `/api/strategies/${strategyId}/trade-zoom?trade_idx=${tradeIdx}&side=${side}`
    ),
    enabled: strategyId !== null && tradeIdx !== null,
    staleTime: 300000, // Cache for 5 min — 1s bars don't change
  });
}

export function useTriggerAnalysis(id: number | null) {
  return useQuery({
    queryKey: ['trigger-analysis', id],
    queryFn: () => apiFetch<TriggerAnalysis>(`/api/strategies/${id}/trigger-analysis`),
    enabled: id !== null,
  });
}
