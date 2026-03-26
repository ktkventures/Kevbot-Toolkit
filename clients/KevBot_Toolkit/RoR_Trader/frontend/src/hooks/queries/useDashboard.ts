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

export interface DashboardHealthItem {
  id: number;
  name: string;
  symbol: string;
  deviation_sd: number | null;
  status: string;
  alert_trades: number;
  expected_r: number;
  actual_r: number;
}

export function useDashboardHealth() {
  return useQuery({
    queryKey: ['dashboard-health'],
    queryFn: () => apiFetch<{ strategies: DashboardHealthItem[] }>('/api/dashboard/health'),
    refetchInterval: 30000,
  });
}

export function useDashboardPositions() {
  return useQuery({
    queryKey: ['dashboard-positions'],
    queryFn: () => apiFetch<{ positions: any[] }>('/api/dashboard/positions'),
    refetchInterval: 5000, // 5s for live positions
  });
}

export function useDashboardActivity() {
  return useQuery({
    queryKey: ['dashboard-activity'],
    queryFn: () => apiFetch<{ items: { time: string; type: string; message: string }[] }>('/api/dashboard/activity'),
    refetchInterval: 10000,
  });
}
