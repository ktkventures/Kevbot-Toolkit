/**
 * Hook for /api/admin/strategy-health/by-hour endpoint.
 *
 * Returns per-hour KPIs (avg/max/min combined %) + cross-hour ranks
 * + per-strategy per-hour drill-down rows for the "By Hour" tab on
 * the Strategy Health admin page.
 */

import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

export interface ByHourKpiRow {
  hour_utc: string;
  n_active: number;
  n_ranked: number;
  avg_combined_pct: number | null;
  max_combined_pct: number | null;
  min_combined_pct: number | null;
  avg_rank: number | null;
  max_rank: number | null;
  min_rank: number | null;
  ranked_total: number;
}

export interface ByHourStrategyRow {
  strategy_id: number;
  name: string | null;
  symbol: string | null;
  timeframe: string | null;
  alerts: number;
  bt_events: number;
  paired: number;
  phantom: number;
  missed: number;
  tbd: number;
  combined_pct: number;
  alert_pair_pct: number;
  rankable: boolean;
  rank: number | null;
  rank_total: number;
}

export interface ByHourCohortEntry {
  strategy_id: number;
  name: string | null;
  symbol: string | null;
  timeframe: string | null;
}

export interface StrategyHealthByHourResponse {
  now: string;
  since: string;
  until: string;
  pair_window_s: number;
  stability_tail_minutes: number;
  cohort_size: number;
  hours_count: number;
  hours_with_data: number;
  hours: ByHourKpiRow[];
  strategies_by_hour: Record<string, ByHourStrategyRow[]>;
  cohort: ByHourCohortEntry[];
  strategy_metadata: import('@/components/StrategyFilters').StrategyMeta[];
}

interface UseByHourOpts {
  since?: string;
  until?: string;
  pairWindowS?: number;
  stabilityTailMinutes?: number;
}

export function useStrategyHealthByHour(opts: UseByHourOpts = {}) {
  const {
    since,
    until,
    pairWindowS = 5,
    stabilityTailMinutes = 15,
  } = opts;
  const params = new URLSearchParams();
  if (since) params.set('since', since);
  if (until) params.set('until', until);
  params.set('pair_window_s', String(pairWindowS));
  params.set('stability_tail_minutes', String(stabilityTailMinutes));
  return useQuery({
    queryKey: [
      'strategy-health-by-hour',
      since,
      until,
      pairWindowS,
      stabilityTailMinutes,
    ],
    queryFn: () =>
      apiFetch<StrategyHealthByHourResponse>(
        `/api/admin/strategy-health/by-hour?${params.toString()}`,
      ),
    staleTime: 60_000,
  });
}
