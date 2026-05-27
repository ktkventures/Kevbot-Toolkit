/**
 * Hook for /api/admin/strategy-health/backlog — per-event divergence
 * with auto-classification. Companion to useStrategyHealth (which
 * aggregates per strategy).
 *
 * Polls every 30s in rolling mode; custom mode (fixed start/end) is
 * frozen and never refetches.
 */
import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

export type DivergenceClassification =
  | 'needs_investigation'
  | 'timestamps_out_of_sync'
  | 'phase2_signal_exit'
  | 'non_fill_event'
  | 'legacy_strategy';

export interface DivergenceBacklogRow {
  event_type: 'phantom' | 'missed';
  strategy_id: number;
  strategy_name: string | null;
  symbol: string | null;
  timeframe: string | null;
  direction: string | null;
  /** For missed events: 'entry' or 'exit'. Null for phantom. */
  edge: 'entry' | 'exit' | null;
  /** ISO timestamp of the unpaired event. */
  timestamp: string;
  trade_id: number | null;
  alert_id: number | null;
  exit_reason: string | null;
  exit_trigger: string | null;
  classification: DivergenceClassification;
  classification_reason: string;
}

export interface DivergenceBacklogResponse {
  now: string;
  window_hours: number;
  window_start: string;
  window_end: string;
  mode: 'rolling' | 'custom';
  row_count: number;
  rows: DivergenceBacklogRow[];
}

export interface BacklogQueryArgs {
  windowHours?: number;
  start?: string | null;
  end?: string | null;
  onlyNeedsInvestigation?: boolean;
}

export function useStrategyHealthBacklog(args: BacklogQueryArgs = {}) {
  const { start, end, onlyNeedsInvestigation } = args;
  const windowHours = args.windowHours ?? 24;
  const isCustom = !!(start && end);
  const params = new URLSearchParams();
  if (isCustom) {
    params.set('start', start!);
    params.set('end', end!);
  } else {
    params.set('window_hours', String(windowHours));
  }
  if (onlyNeedsInvestigation) {
    params.set('only_needs_investigation', 'true');
  }
  return useQuery<DivergenceBacklogResponse>({
    queryKey: ['admin', 'strategy-health-backlog',
               isCustom ? 'custom' : windowHours,
               start ?? null, end ?? null,
               !!onlyNeedsInvestigation],
    queryFn: () => apiFetch<DivergenceBacklogResponse>(
      `/api/admin/strategy-health/backlog?${params.toString()}`),
    staleTime: 25_000,
    refetchInterval: isCustom ? false : 30_000,
  });
}
