/**
 * Hook for /api/admin/strategy-health/by-deploy.
 *
 * Returns per-deploy-window pair-rate stats. Each row = one commit's
 * active window (from its timestamp until the next deploy or now).
 */

import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

export interface DeployWindowRow {
  sha: string;
  subject: string;
  timestamp_iso: string;
  window_start: string;
  window_end: string;
  duration_seconds: number;
  stable_duration_seconds: number;
  raw_duration_seconds: number;
  low_confidence: boolean;
  alerts: number;
  bt_events: number;
  paired: number;
  phantom: number;
  missed: number;
  combined_pct: number | null;
  avg_strategy_combined_pct: number | null;
  n_active_strategies: number;
  n_ranked_strategies: number;
}

export interface DeployStrategyRow {
  strategy_id: number;
  alerts: number;
  bt_events: number;
  paired: number;
  phantom: number;
  missed: number;
  combined_pct: number;
  rankable: boolean;
}

export interface StrategyHealthByDeployResponse {
  now: string;
  since: string;
  until: string;
  pair_window_s: number;
  stability_tail_minutes: number;
  warmup_minutes: number;
  deploy_count: number;
  deploys: DeployWindowRow[];
  strategies_by_deploy: Record<string, DeployStrategyRow[]>;
  cohort_size: number;
  strategy_metadata: import('@/components/StrategyFilters').StrategyMeta[];
  note?: string;
}

interface UseByDeployOpts {
  since?: string;
  until?: string;
  pairWindowS?: number;
  stabilityTailMinutes?: number;
  warmupMinutes?: number;
}

export function useStrategyHealthByDeploy(opts: UseByDeployOpts = {}) {
  const {
    since,
    until,
    pairWindowS = 5,
    stabilityTailMinutes = 15,
    warmupMinutes = 30,
  } = opts;
  const params = new URLSearchParams();
  if (since) params.set('since', since);
  if (until) params.set('until', until);
  params.set('pair_window_s', String(pairWindowS));
  params.set('stability_tail_minutes', String(stabilityTailMinutes));
  params.set('warmup_minutes', String(warmupMinutes));
  return useQuery({
    queryKey: [
      'strategy-health-by-deploy',
      since,
      until,
      pairWindowS,
      stabilityTailMinutes,
      warmupMinutes,
    ],
    queryFn: () =>
      apiFetch<StrategyHealthByDeployResponse>(
        `/api/admin/strategy-health/by-deploy?${params.toString()}`,
      ),
    staleTime: 60_000,
  });
}
