/**
 * Hooks for execution type modules and user configuration.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

export interface ExecTypeModule {
  slug: string;
  name: string;
  description: string;
  display_code: string;
  exec_type_codes: string[];
  contexts: string[];
  enabled: boolean;
  user_params: Record<string, any>;
  steps: Array<{
    action: string;
    label: string;
    [key: string]: any;
  }>;
  parameters_schema: Record<string, {
    type: string;
    default: any;
    options?: any[];
    label: string;
    min?: number;
  }>;
}

export function useExecutionTypes() {
  return useQuery({
    queryKey: ['execution-types'],
    queryFn: () => apiFetch<ExecTypeModule[]>('/api/execution-types'),
  });
}

export function useExecutionType(slug: string | null) {
  return useQuery({
    queryKey: ['execution-type', slug],
    queryFn: () => apiFetch<ExecTypeModule>(`/api/execution-types/${slug}`),
    enabled: !!slug,
  });
}

export function useToggleExecutionType() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (slug: string) =>
      apiFetch(`/api/execution-types/${slug}/toggle`, { method: 'PUT' }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['execution-types'] });
    },
  });
}

export function useUpdateExecTypeParams() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ slug, params }: { slug: string; params: Record<string, any> }) =>
      apiFetch(`/api/execution-types/${slug}/params`, {
        method: 'PUT',
        body: JSON.stringify(params),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['execution-types'] });
    },
  });
}
