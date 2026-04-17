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

export function useStrategyAlerts(strategyId: number | null) {
  return useQuery({
    queryKey: ['alerts', 'strategy', strategyId],
    queryFn: () => apiFetch<any[]>(`/api/alerts/strategy/${strategyId}`),
    enabled: strategyId !== null,
    refetchInterval: 5000,
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
