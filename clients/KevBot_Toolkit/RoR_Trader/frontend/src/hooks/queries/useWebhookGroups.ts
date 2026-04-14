/**
 * Webhook Group query + mutation hooks.
 * Account-based webhook groups — 11 events per group, each with optional URL+payload.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

export interface WebhookEventConfig {
  url: string;
  payload: string;
}

export type UrlMode = 'per_event' | 'shared';

export interface WebhookGroup {
  id: string;
  name: string;
  service?: string;
  url_mode?: UrlMode;
  shared_url?: string;
  events: Record<string, WebhookEventConfig>;
  created_at?: string;
  updated_at?: string;
}

export function useWebhookGroups() {
  return useQuery({
    queryKey: ['webhook-groups'],
    queryFn: () => apiFetch<WebhookGroup[]>('/api/webhook-groups'),
  });
}

export function useWebhookGroup(id: string | null) {
  return useQuery({
    queryKey: ['webhook-group', id],
    queryFn: () => apiFetch<WebhookGroup>(`/api/webhook-groups/${id}`),
    enabled: id !== null,
  });
}

export function useWebhookEventTypes() {
  return useQuery({
    queryKey: ['webhook-event-types'],
    queryFn: () => apiFetch<{ event_types: string[] }>('/api/webhook-groups/meta/event-types'),
    staleTime: 1000 * 60 * 10,
  });
}

export function useCreateWebhookGroup() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (group: Partial<WebhookGroup>) =>
      apiFetch<WebhookGroup>('/api/webhook-groups', {
        method: 'POST',
        body: JSON.stringify(group),
      }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['webhook-groups'] }); },
  });
}

export function useUpdateWebhookGroup() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, updates }: { id: string; updates: Partial<WebhookGroup> }) =>
      apiFetch(`/api/webhook-groups/${id}`, {
        method: 'PUT',
        body: JSON.stringify(updates),
      }),
    onSuccess: (_d, { id }) => {
      qc.invalidateQueries({ queryKey: ['webhook-groups'] });
      qc.invalidateQueries({ queryKey: ['webhook-group', id] });
    },
  });
}

export function useDeleteWebhookGroup() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch(`/api/webhook-groups/${id}`, { method: 'DELETE' }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['webhook-groups'] }); },
  });
}

export function useDuplicateWebhookGroup() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<WebhookGroup>(`/api/webhook-groups/${id}/duplicate`, { method: 'POST' }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['webhook-groups'] }); },
  });
}

export function useTestWebhookGroupEvent() {
  return useMutation({
    mutationFn: ({ id, eventType, alert }: { id: string; eventType: string; alert?: Record<string, any> }) =>
      apiFetch(`/api/webhook-groups/${id}/test/${eventType}`, {
        method: 'POST',
        body: JSON.stringify(alert ? { alert } : {}),
      }),
  });
}
