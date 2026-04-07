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
  is_default: boolean;
  user_params: Record<string, any>;
  variations: ExecTypeVariation[];
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

export interface SimulationResult {
  slug: string;
  display_code: string;
  symbol: string;
  timeframe: string;
  direction: string;
  trigger_bar: number;
  trigger_time: string;
  fill_price: number;
  stop_price: number | null;
  target_price: number | null;
  confirmed: boolean;
  steps: Array<{
    bar: number;
    time: string;
    action: string;
    label: string;
    price?: number;
    event?: string;
    confirmed?: boolean;
    reason?: string;
    r_multiple?: number;
    stop_price?: number;
    target_price?: number;
  }>;
  chart_bars?: Array<{ timestamp: string; open: number; high: number; low: number; close: number }>;
  chart_markers?: Array<{ time: string; position: string; shape: string; color: string; text: string; size: number }>;
}

export function useSimulateExecType() {
  return useMutation({
    mutationFn: ({ slug, params }: { slug: string; params: { symbol: string; timeframe: string; direction: string; days: number } }) =>
      apiFetch<SimulationResult>(`/api/execution-types/${slug}/simulate`, {
        method: 'POST',
        body: JSON.stringify(params),
      }),
  });
}

export interface ExecTypeVariation {
  id: string;
  parent_slug: string;
  name: string;
  badge: string;
  params: Record<string, any>;
  enabled: boolean;
}

export function useCreateVariation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ slug, name, params, badge }: { slug: string; name: string; params: Record<string, any>; badge?: string }) =>
      apiFetch<ExecTypeVariation>(`/api/execution-types/${slug}/variations`, {
        method: 'POST',
        body: JSON.stringify({ name, params, badge }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['execution-types'] });
    },
  });
}

export function useDeleteVariation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (varId: string) =>
      apiFetch(`/api/execution-types/variations/${varId}`, { method: 'DELETE' }),
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

// --- Dynamic scenario generation ---

export interface GenerateScenariosRequest {
  symbol: string;
  timeframe: string;
  direction: 'LONG' | 'SHORT';
  days: number;
  entry_trigger_confluence_id: string;
  exit_trigger_confluence_ids: string[];
  stop_atr_mult: number;
  target_r_multiple: number;
  bar_count_exit: number | null;
}

export function useGenerateScenarios(slug: string) {
  return useMutation({
    mutationFn: (req: GenerateScenariosRequest) =>
      apiFetch<any>(`/api/execution-types/${slug}/generate-scenarios`, {
        method: 'POST',
        body: JSON.stringify(req),
      }),
  });
}
