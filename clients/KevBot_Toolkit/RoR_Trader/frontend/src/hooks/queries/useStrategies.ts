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
  // Canonical post-Phase 40 timestamp columns; entry_time/exit_time are
  // legacy aliases _row_to_trade emits for backwards compat. Many code
  // paths read both — keep them all.
  entry_fill_ts?: string;
  exit_fill_ts?: string;
  entry_trigger_ts?: string;
  exit_trigger_ts?: string;
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
  hold_time_seconds?: number;
  pnl?: number;
  data_source?: string;
  hifi_resolved?: boolean;
  behavior?: string;
  entry_trigger?: string;
  exit_trigger?: string;
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

// Algo-lane trades (data_source LIKE 'cache_%') for a strategy.
// Distinct from useStrategyTrades which returns the backtest lane
// (stored_trades JSONB, filtered to backtest_% post-Phase 41 hydration).
// Used by Chart & Trades "Algo History" + "Price Divergence (Algo vs
// Alert)" modules so they show real algo-lane data instead of backtest.
//
// Optional `since` ISO timestamp filter for date-windowed views.
export function useStrategyAlgoTrades(
  id: number | null,
  since?: string | null,
) {
  const params = since ? `?since=${encodeURIComponent(since)}` : '';
  return useQuery({
    queryKey: ['strategy-algo-trades', id, since ?? null],
    queryFn: () =>
      apiFetch<TradeDTO[]>(`/api/strategies/${id}/algo-trades${params}`),
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

// M8.7 (2026-05-02): list of available backtest_model and live_model
// values for the strategy edit/detail page. Static-ish data; long stale time.
export interface StrategyModel {
  label: string;
  available: boolean;
  default: boolean;
  description: string;
}
export interface StrategyModelsResponse {
  backtest_models: Record<string, StrategyModel>;
  algo_models: Record<string, StrategyModel>;
  live_models: Record<string, StrategyModel>;
  defaults: { backtest_model: string; algo_model: string; live_model: string };
}
export function useStrategyModels() {
  return useQuery({
    queryKey: ['strategy-models'],
    queryFn: () => apiFetch<StrategyModelsResponse>(`/api/strategies/models`),
    staleTime: 600000, // 10 min — values change rarely
    retry: 1,
  });
}

// M8.7 Phase 2 (2026-05-02): full chart-data response (OHLCV + indicators
// + heatmap) computed from live_bars cache. Used by the Alert Lens on the
// Lab tab so the right-side chart's heatmap and indicators reflect what
// the live engine actually saw, not REST-derived values.
export function useStrategyChartDataCache(
  id: number | null,
  valueType: 'latest' | 'first',
  enabled: boolean = true,
) {
  return useQuery({
    queryKey: ['strategy-chart-data-cache', id, valueType],
    queryFn: () => apiFetch<ChartDataResponse>(
      `/api/strategies/${id}/chart-data-cache?value_type=${valueType}`
    ),
    enabled: enabled && id !== null,
    staleTime: 30000,
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

// ============================================================
// Divergence tab (M8.7 — 2026-05-07)
// ============================================================

export interface DivergenceLaneTrade {
  entry_fill_ts: string | null;
  exit_fill_ts: string | null;
  entry_price: number | null;
  exit_price: number | null;
  direction: string | null;
  exit_reason: string | null;
  data_source: string | null;
  live_model: string | null;
}

export type LaneComposition =
  | '3way'
  | 'rest_live'
  | 'cache_live'
  | 'rest_cache'
  | 'rest_only'
  | 'cache_only'
  | 'live_only'
  | 'empty';

export interface DivergenceRow {
  row_id: number;
  rest: DivergenceLaneTrade | null;   // = backtest lane
  cache: DivergenceLaneTrade | null;  // = algo lane
  live: DivergenceLaneTrade | null;
  lane_composition: LaneComposition;
  anchor_entry_fill_ts: string | null;
  direction: string | null;
  drift_rest_live_entry_s: number | null;
  drift_cache_live_entry_s: number | null;
  drift_rest_cache_entry_s: number | null;
  drift_rest_live_exit_s: number | null;
  drift_cache_live_exit_s: number | null;
}

export interface DivergenceDriftStats {
  count: number;
  median_s: number | null;
  p95_s: number | null;
  max_s: number | null;
}

export interface DivergenceData {
  strategy_id: number;
  forward_test_start: string | null;
  backtest_model: string | null;
  algo_model: string | null;
  live_model: string | null;
  backtest: { count: number; shown?: number; last_trade_ts: string | null; available: boolean };
  algo: { count: number; shown?: number; last_trade_ts: string | null; available: boolean };
  live: { count: number; last_alert_ts: string | null };
  max_per_lane?: number;
  rows: DivergenceRow[];
  kpis: {
    matched_3way: number;
    matched_rest_live: number;
    matched_cache_live: number;
    matched_rest_cache: number;
    rest_only: number;
    cache_only: number;
    live_only: number;
    drift_rest_live_entry: DivergenceDriftStats;
    drift_cache_live_entry: DivergenceDriftStats;
    drift_rest_cache_entry: DivergenceDriftStats;
    drift_rest_live_exit: DivergenceDriftStats;
    drift_cache_live_exit: DivergenceDriftStats;
  };
  lane_counts: { rest: number; cache: number; live: number };
  forward_test_only: boolean;
  direction_filter: string | null;
}

export interface UpdateLaneResult {
  status: string;
  reason?: string;
  inserted?: number;
  elapsed_s?: number;
}

export interface UpdateLanesResponse {
  mode: 'all' | 'new';
  backtest: UpdateLaneResult | null;
  algo: UpdateLaneResult | null;
}

export function useStrategyDivergence(
  id: number | null,
  opts: {
    forward_test_only?: boolean;
    direction?: 'LONG' | 'SHORT' | null;
    tolerance_seconds?: number;
    start?: string | null;
    end?: string | null;
    enabled?: boolean;
  } = {},
) {
  const params = new URLSearchParams();
  if (opts.forward_test_only !== undefined)
    params.set('forward_test_only', String(opts.forward_test_only));
  if (opts.direction) params.set('direction', opts.direction);
  if (opts.tolerance_seconds !== undefined)
    params.set('tolerance_seconds', String(opts.tolerance_seconds));
  if (opts.start) params.set('start', opts.start);
  if (opts.end) params.set('end', opts.end);
  const qs = params.toString();

  return useQuery({
    queryKey: ['strategy-divergence', id, opts.forward_test_only, opts.direction, opts.tolerance_seconds, opts.start, opts.end],
    queryFn: () =>
      apiFetch<DivergenceData>(
        `/api/strategies/${id}/divergence-data${qs ? '?' + qs : ''}`,
      ),
    enabled: (opts.enabled ?? true) && id !== null,
    staleTime: 30_000,
    placeholderData: (prev) => prev,
  });
}

// ============================================================
// Admin: cross-strategy divergence summary
// ============================================================

export interface AdminDivergenceRow {
  strategy_id: number;
  name: string;
  symbol: string;
  timeframe: string;
  backtest_model?: string | null;
  algo_model?: string | null;
  live_model?: string | null;
  backtest_count?: number;
  algo_count?: number;
  live_count?: number;
  matched_3way?: number;
  matched_rest_cache?: number;
  matched_cache_live?: number;
  entry_rc_median_s?: number | null;
  entry_rc_p95_s?: number | null;
  entry_rc_max_s?: number | null;
  entry_cl_median_s?: number | null;
  entry_cl_p95_s?: number | null;
  exit_cl_median_s?: number | null;
  exit_cl_p95_s?: number | null;
  exit_cl_max_s?: number | null;
  no_activity?: boolean;
  error?: string;
}

export interface AdminDivergenceSummary {
  start: string;
  end: string;
  strategy_count: number;
  rows: AdminDivergenceRow[];
}

export function useAdminDivergenceSummary(
  start: string | null,
  end: string | null,
  enabled = true,
) {
  return useQuery({
    queryKey: ['admin-divergence-summary', start, end],
    queryFn: () => {
      const params = new URLSearchParams();
      if (start) params.set('start', start);
      if (end) params.set('end', end);
      return apiFetch<AdminDivergenceSummary>(
        `/api/strategies/admin/divergence-summary?${params.toString()}`,
      );
    },
    enabled: enabled && start !== null,
    staleTime: 30_000,
    placeholderData: (prev) => prev,
  });
}
