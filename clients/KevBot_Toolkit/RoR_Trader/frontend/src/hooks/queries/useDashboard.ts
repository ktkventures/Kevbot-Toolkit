/**
 * Dashboard query hooks.
 */

import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

export interface DashboardSummary {
  strategy_count: number;
  portfolio_count: number;
  monitored_count: number;
  total_trades: number;
  total_r: number;
  avg_win_rate: number;
  strategies: Array<{
    id: number;
    name: string;
    symbol: string;
    direction: string;
    kpis: Record<string, number>;
    alert_tracking_enabled: boolean;
  }>;
}

export function useDashboardSummary() {
  return useQuery({
    queryKey: ['dashboard-summary'],
    queryFn: () => apiFetch<DashboardSummary>('/api/dashboard/summary'),
    refetchInterval: 30000, // 30s refresh
  });
}
