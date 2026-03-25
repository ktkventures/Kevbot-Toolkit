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

export function useWebhookDeliveryLog() {
  return useQuery({
    queryKey: ['webhook-delivery-log'],
    queryFn: () => apiFetch<any[]>('/api/webhooks/delivery-log'),
  });
}
