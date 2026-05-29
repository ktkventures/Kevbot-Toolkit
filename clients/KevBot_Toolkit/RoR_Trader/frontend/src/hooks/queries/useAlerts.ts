/**
 * Alert + monitor query hooks.
 */

import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

export function useAlerts() {
  return useQuery({
    queryKey: ['alerts'],
    queryFn: () => apiFetch<any[]>('/api/alerts'),
    refetchInterval: 5000,
  });
}

// 2026-05-29: bumped default limit from API default (50) to 1000 so
// Strategy Detail's Alert History matches Algo History's "all in one
// query, paginate client-side" pattern. 50 capped the visible alert
// table at ~25 paired entry/exit rows even on strategies with thousands
// of alerts in DB (Kevin caught this comparing alert deltas vs backtest).
// 1000 is the API max. UI applies its own display cap (default 100)
// with a "Show all N" toggle, matching the algo-history pattern.
export function useStrategyAlerts(
  strategyId: number | null,
  limit: number = 1000,
) {
  return useQuery({
    queryKey: ['alerts', 'strategy', strategyId, limit],
    queryFn: () =>
      apiFetch<any[]>(`/api/alerts/strategy/${strategyId}?limit=${limit}`),
    enabled: strategyId !== null,
    refetchInterval: 5000,
  });
}

// Single alert lookup (GET /api/alerts/{id}) — returns the full alert row
// including its webhook_deliveries JSONB array. Powers the Portfolio Trade
// History Details drawer (roadmap 9ad).
export interface WebhookDelivery {
  webhook_id?: string;
  webhook_name?: string;
  portfolio_id?: number;
  sent_at?: string;
  success?: boolean;
  status_code?: number | null;
  payload_sent?: string | null;
  error?: string | null;
}

export interface AlertRow {
  id: number;
  type?: string;
  strategy_id?: number;
  strategy_name?: string;
  symbol?: string;
  direction?: string;
  trigger_id?: string;
  price?: number | null;
  actual_price?: number | null;
  timestamp?: string;
  trigger_ts?: string;
  fill_ts?: string;
  webhook_deliveries?: WebhookDelivery[];
  [key: string]: any;
}

export function useAlertDetails(alertId: number | null | undefined) {
  return useQuery({
    queryKey: ['alert-details', alertId],
    queryFn: () => apiFetch<AlertRow>(`/api/alerts/${alertId}`),
    enabled: alertId != null,
    staleTime: 30_000, // deliveries don't retroactively change
  });
}

export function useAlertConfig() {
  return useQuery({
    queryKey: ['alert-config'],
    queryFn: () => apiFetch<Record<string, any>>('/api/alerts/config'),
  });
}

export function useMonitorStatus() {
  return useQuery({
    queryKey: ['monitor-status'],
    queryFn: () => apiFetch<Record<string, any>>('/api/monitor/status'),
    refetchInterval: 5000,
  });
}

export function useEngineState() {
  return useQuery({
    queryKey: ['engine-state'],
    queryFn: () => apiFetch<Record<string, any>>('/api/monitor/engine-state'),
    refetchInterval: 5000,
  });
}
