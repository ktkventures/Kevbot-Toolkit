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
  | 'has_discrepancies'
  | 'phantom_alerts'
  | 'missed_alerts';

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
  /** When the most-recent backtest_% trade row was inserted into the
   *  trades table (vs last_entry_ts which is its execution time).
   *  Better freshness signal for "is the backtest history current". */
  last_backtest_created_at: string | null;
  last_backtest_created_age_sec: number | null;
  /** When the most-recent alert was received for this strategy
   *  (un-windowed, capped to last 30 days). Null = no alerts in 30d. */
  last_alert_at: string | null;
  last_alert_age_sec: number | null;
  /** True when both last_backtest_created_at and last_alert_at exist
   *  but are >1h apart. Phantom/missed counts aren't trustworthy in
   *  this state — one source is stale relative to the other. */
  timestamps_upside_down: boolean;
  upside_down_delta_sec: number | null;
  /** Whether the data-worker should maintain a backtest snapshot for
   *  this strategy. False = parked (worker skips entirely). Toggle via
   *  PATCH /strategies/{id}/snapshot-subscription. Defaults to true.
   *  Live alerts are unaffected — only the snapshot lane is gated. */
  snapshot_subscribe_enabled: boolean;
  trade_count_backtest: number;

  parity_status: Record<string, unknown> | null;
  parity_verdict: string | null;
  discrepancies_count: number;
  /** Alerts in the configurable lookback window with no matching
   *  backtest trade edge (entry or exit) within ±60s. Kevin's
   *  "phantom" — alert fired without a corresponding backtest trade. */
  phantom_count: number;
  /** Backtest trade edges in the configurable lookback window with no
   *  matching alert within ±60s. Kevin's "missed" — backtest fired but
   *  algo didn't. */
  missed_count: number;
  /** Successful (alert, edge) matches within ±60s. Counted in edges
   *  (each trade has up to two: entry + exit), so this can be up to
   *  ~2× trade_count_backtest. */
  paired_count: number;

  red_flags: StrategyHealthFlag[];

  updated_at: string | null;
  created_at: string | null;
}

export interface StrategyHealthResponse {
  now: string;
  /** Echoed back from the server so the UI can display "Phantom Nh"
   *  consistent with what was actually computed. */
  window_hours: number;
  /** Resolved window bounds — present in BOTH rolling + custom modes.
   *  In rolling mode, window_end == now; in custom mode, frozen. */
  window_start?: string;
  window_end?: string;
  mode?: 'rolling' | 'custom';
  rows: StrategyHealthRow[];
}

export interface StrategyHealthQueryArgs {
  /** Rolling-mode lookback in hours. Backend clamps to [1, 168].
   *  Default 24. Ignored when both `start` and `end` are provided. */
  windowHours?: number;
  /** Custom mode: ISO 8601 or Unix-sec start. When set together with
   *  `end`, overrides windowHours. */
  start?: string | null;
  /** Custom mode end. Stays frozen — no auto-tick to "now". */
  end?: string | null;
}

export function useStrategyHealth(args: StrategyHealthQueryArgs = {}) {
  const { start, end } = args;
  const windowHours = args.windowHours ?? 24;
  const isCustom = !!(start && end);
  const qs = isCustom
    ? `?start=${encodeURIComponent(start!)}&end=${encodeURIComponent(end!)}`
    : `?window_hours=${windowHours}`;
  return useQuery<StrategyHealthResponse>({
    queryKey: ['admin', 'strategy-health',
                isCustom ? 'custom' : windowHours,
                start ?? null, end ?? null],
    queryFn: () =>
      apiFetch<StrategyHealthResponse>(`/api/admin/strategy-health${qs}`),
    staleTime: 25_000,
    // Custom mode = frozen end → no auto-refetch. Rolling mode polls
    // every 30s for live updates.
    refetchInterval: isCustom ? false : 30_000,
  });
}
