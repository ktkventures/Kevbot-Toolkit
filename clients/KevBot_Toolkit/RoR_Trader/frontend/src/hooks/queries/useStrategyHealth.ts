/**
 * Hook for the /api/admin/strategy-health endpoint.  Polls every 30s so
 * the dashboard reflects fresh data-worker writes and recompute runs
 * without manual refresh.  Stale-time set so multiple subscribers in the
 * same session share the result.
 */

import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

/** Red-flag codes emitted by the backend.  Keep in sync with
 *  src/api/routers/strategy_health.py. */
export type StrategyHealthFlag =
  | 'legacy_no_confluence_id'
  | 'no_baseline'
  | 'snapshot_missing'
  | 'snapshot_stale'
  | 'kpis_stale'
  | 'kpis_marked_stale'
  | 'data_refresh_stale'
  | 'no_recent_trades'
  | 'parity_fail'
  | 'has_discrepancies';

export interface StrategyHealthRow {
  strategy_id: number;
  user_id: string;
  name: string | null;
  symbol: string | null;
  timeframe: string | null;
  direction: string | null;
  strategy_origin: string | null;
  forward_testing: boolean;
  forward_test_start: string | null;
  streaming_eligible: boolean;
  data_source: string | null;
  backtest_model: string | null;

  snapshot_at: string | null;
  snapshot_age_sec: number | null;
  last_recompute_until_ts: string | null;

  kpis_computed_at: string | null;
  kpis_age_sec: number | null;
  kpis_stale_since: string | null;

  data_refreshed_at: string | null;
  data_refreshed_age_sec: number | null;

  last_entry_ts: string | null;
  last_entry_age_sec: number | null;
  last_exit_ts: string | null;
  last_exit_age_sec: number | null;
  trade_count_backtest: number;

  parity_status: Record<string, unknown> | null;
  parity_verdict: string | null;
  discrepancies_count: number;

  red_flags: StrategyHealthFlag[];

  updated_at: string | null;
  created_at: string | null;
}

export interface StrategyHealthResponse {
  now: string;
  rows: StrategyHealthRow[];
}

export function useStrategyHealth() {
  return useQuery<StrategyHealthResponse>({
    queryKey: ['admin', 'strategy-health'],
    queryFn: () => apiFetch<StrategyHealthResponse>('/api/admin/strategy-health'),
    staleTime: 25_000,
    refetchInterval: 30_000,
  });
}
