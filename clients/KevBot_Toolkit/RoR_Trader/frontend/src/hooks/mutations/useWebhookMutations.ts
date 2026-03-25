/**
 * Webhook template mutation hooks.
 */

import { useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

export function useCreateWebhookTemplate() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (template: Record<string, any>) =>
      apiFetch('/api/webhooks/templates', { method: 'POST', body: JSON.stringify(template) }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['webhook-templates'] }); },
  });
}

export function useUpdateWebhookTemplate() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, template }: { id: string; template: Record<string, any> }) =>
      apiFetch(`/api/webhooks/templates/${id}`, { method: 'PUT', body: JSON.stringify(template) }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['webhook-templates'] }); },
  });
}

export function useDeleteWebhookTemplate() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch(`/api/webhooks/templates/${id}`, { method: 'DELETE' }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['webhook-templates'] }); },
  });
}

export function useTestWebhook() {
  return useMutation({
    mutationFn: (config: Record<string, any>) =>
      apiFetch('/api/webhooks/test', { method: 'POST', body: JSON.stringify(config) }),
  });
}
