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

export interface StrategyHealthByDeployResponse {
  now: string;
  since: string;
  until: string;
  pair_window_s: number;
  deploy_count: number;
  deploys: DeployWindowRow[];
  cohort_size: number;
  note?: string;
}

interface UseByDeployOpts {
  since?: string;
  until?: string;
  pairWindowS?: number;
}

export function useStrategyHealthByDeploy(opts: UseByDeployOpts = {}) {
  const { since, until, pairWindowS = 5 } = opts;
  const params = new URLSearchParams();
  if (since) params.set('since', since);
  if (until) params.set('until', until);
  params.set('pair_window_s', String(pairWindowS));
  return useQuery({
    queryKey: ['strategy-health-by-deploy', since, until, pairWindowS],
    queryFn: () =>
      apiFetch<StrategyHealthByDeployResponse>(
        `/api/admin/strategy-health/by-deploy?${params.toString()}`,
      ),
    staleTime: 60_000,
  });
}
