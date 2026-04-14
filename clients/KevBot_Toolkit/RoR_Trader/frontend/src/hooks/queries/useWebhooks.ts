/**
 * Webhook template query hooks.
 */

import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

export function useWebhookTemplates() {
  return useQuery({
    queryKey: ['webhook-templates'],
    queryFn: () => apiFetch<any[]>('/api/webhooks/templates'),
  });
}

export function useWebhookTemplate(id: string | null) {
  return useQuery({
    queryKey: ['webhook-template', id],
    queryFn: () => apiFetch<any>(`/api/webhooks/templates/${id}`),
    enabled: id !== null,
  });
}

export function useWebhookDeliveryLog(opts?: { templateId?: string | null; limit?: number }) {
  const templateId = opts?.templateId ?? null;
  const limit = opts?.limit ?? 100;
  const qs = new URLSearchParams();
  qs.set('limit', String(limit));
  if (templateId) qs.set('template_id', templateId);
  return useQuery({
    queryKey: ['webhook-delivery-log', { templateId, limit }],
    queryFn: () => apiFetch<any[]>(`/api/webhooks/delivery-log?${qs.toString()}`),
  });
}
