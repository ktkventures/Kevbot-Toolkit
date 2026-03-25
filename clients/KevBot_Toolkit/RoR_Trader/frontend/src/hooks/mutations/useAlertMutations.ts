/**
 * Alert + monitor mutation hooks.
 */

import { useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

export function useAcknowledgeAlert() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (alertId: number) =>
      apiFetch(`/api/alerts/${alertId}/acknowledge`, { method: 'PUT' }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['alerts'] }); },
  });
}

export function useClearAlerts() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => apiFetch('/api/alerts/clear', { method: 'POST' }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['alerts'] }); },
  });
}

export function useSaveAlertConfig() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (config: Record<string, any>) =>
      apiFetch('/api/alerts/config', { method: 'PUT', body: JSON.stringify(config) }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['alert-config'] }); },
  });
}

export function useStartMonitor() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => apiFetch('/api/monitor/start', { method: 'POST' }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['monitor-status'] }); },
  });
}

export function useStopMonitor() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => apiFetch('/api/monitor/stop', { method: 'POST' }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['monitor-status'] }); },
  });
}
