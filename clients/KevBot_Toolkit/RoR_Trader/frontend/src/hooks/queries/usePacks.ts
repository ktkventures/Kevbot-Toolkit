/**
 * React Query hooks for all pack types:
 * - Confluence Groups (TF packs)
 * - General Packs
 * - Risk Management Packs
 */

import { useQuery, useMutation } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

// =============================================================================
// CONFLUENCE GROUPS
// =============================================================================

export function useConfluenceGroups() {
  return useQuery({
    queryKey: ['confluence-groups'],
    queryFn: () => apiFetch<ConfluenceGroupDTO[]>('/api/packs/confluence-groups'),
  });
}

export function useConfluenceTemplates() {
  return useQuery({
    queryKey: ['confluence-templates'],
    queryFn: () => apiFetch<Record<string, ConfluenceTemplateDTO>>('/api/packs/confluence-groups/templates'),
    staleTime: Infinity, // Templates are static
  });
}

export function useConfluenceTriggers(direction: 'LONG' | 'SHORT' | 'EXIT') {
  return useQuery({
    queryKey: ['confluence-triggers', direction],
    queryFn: () => apiFetch<Record<string, string>>(`/api/packs/confluence-groups/triggers/${direction}`),
  });
}

// =============================================================================
// GENERAL PACKS
// =============================================================================

export function useGeneralPacks() {
  return useQuery({
    queryKey: ['general-packs'],
    queryFn: () => apiFetch<GeneralPackDTO[]>('/api/packs/general'),
  });
}

export function useGeneralTemplates() {
  return useQuery({
    queryKey: ['general-templates'],
    queryFn: () => apiFetch<Record<string, GeneralTemplateDTO>>('/api/packs/general/templates'),
    staleTime: Infinity,
  });
}

// =============================================================================
// RISK MANAGEMENT PACKS
// =============================================================================

export function useRiskManagementPacks() {
  return useQuery({
    queryKey: ['risk-management-packs'],
    queryFn: () => apiFetch<RiskManagementPackDTO[]>('/api/packs/risk-management'),
  });
}

export function useRiskManagementTemplates() {
  return useQuery({
    queryKey: ['risk-management-templates'],
    queryFn: () => apiFetch<Record<string, RMTemplateDTO>>('/api/packs/risk-management/templates'),
    staleTime: Infinity,
  });
}

// =============================================================================
// STOP LOSS PACKS (dedicated filtered view)
// =============================================================================

export function useStopLossPacks() {
  return useQuery({
    queryKey: ['stop-loss-packs'],
    queryFn: () => apiFetch<RiskManagementPackDTO[]>('/api/packs/stop-loss'),
  });
}

export function useStopLossTemplates() {
  return useQuery({
    queryKey: ['stop-loss-templates'],
    queryFn: () => apiFetch<Record<string, RMTemplateDTO>>('/api/packs/stop-loss/templates'),
    staleTime: Infinity,
  });
}

// =============================================================================
// TAKE PROFIT PACKS (dedicated filtered view)
// =============================================================================

export function useTakeProfitPacks() {
  return useQuery({
    queryKey: ['take-profit-packs'],
    queryFn: () => apiFetch<RiskManagementPackDTO[]>('/api/packs/take-profit'),
  });
}

export function useTakeProfitTemplates() {
  return useQuery({
    queryKey: ['take-profit-templates'],
    queryFn: () => apiFetch<Record<string, RMTemplateDTO>>('/api/packs/take-profit/templates'),
    staleTime: Infinity,
  });
}

// =============================================================================
// TIME EXIT PACKS
// =============================================================================

export function useTimeExitPacks() {
  return useQuery({
    queryKey: ['time-exit-packs'],
    queryFn: () => apiFetch<TimeExitPackDTO[]>('/api/packs/time-exit'),
  });
}

export function useTimeExitTemplates() {
  return useQuery({
    queryKey: ['time-exit-templates'],
    queryFn: () => apiFetch<Record<string, TimeExitTemplateDTO>>('/api/packs/time-exit/templates'),
    staleTime: Infinity,
  });
}

// =============================================================================
// USER PACKS
// =============================================================================

export interface UserPackCodeResponse {
  manifest: Record<string, unknown>;
  indicator_code: string;
  interpreter_code: string;
}

export function useUserPackCode(slug: string | null) {
  return useQuery({
    queryKey: ['user-pack-code', slug],
    queryFn: () => apiFetch<UserPackCodeResponse>(`/api/packs/builder/user-packs/${slug}/code`),
    enabled: !!slug,
  });
}

// =============================================================================
// TYPES — Mirror the Python dataclass shapes
// =============================================================================

export interface ConfluenceGroupDTO {
  id: string;
  base_template: string;
  version: string;
  description: string;
  enabled: boolean;
  is_default: boolean;
  parameters: Record<string, unknown>;
  plot_settings: {
    colors: Record<string, string>;
    line_width: number;
    visible: boolean;
  };
}

export interface ConfluenceTemplateDTO {
  name: string;
  category: string;
  description: string;
  interpreters: string[];
  trigger_prefix: string;
  /** True when the template was injected from the user-pack registry
   *  (pack_registry._inject_into_templates). Absent on the built-in
   *  templates hardcoded in confluence_groups.py. */
  _user_pack?: boolean;
  parameters_schema: Record<string, {
    type: string;
    default: number | string;
    min?: number;
    max?: number;
    label: string;
  }>;
  plot_schema?: Record<string, {
    type: string;
    default: string;
    label: string;
  }>;
  outputs: Record<string, string>;
  triggers: Array<{
    id: string;
    name: string;
    direction: string;
    trigger_type: string;
    execution: string;
    exec_variants?: Record<string, {
      enabled?: boolean;
      reference_bar?: number;
      order_type?: string;
      hold_seconds?: number;
      confirm_bar_offset?: number;
      bail_action?: string;
    }>;
  }>;
}

export interface GeneralPackDTO {
  id: string;
  base_template: string;
  version: string;
  description: string;
  enabled: boolean;
  is_default: boolean;
  parameters: Record<string, unknown>;
}

export interface GeneralTemplateDTO {
  name: string;
  category: string;
  description: string;
  parameters_schema: Record<string, {
    type: string;
    default: unknown;
    label: string;
  }>;
  outputs: string[];
  triggers?: Array<{
    id: string;
    name: string;
    direction: string;
    trigger_type: string;
  }>;
}

export interface RiskManagementPackDTO {
  id: string;
  base_template: string;
  template_name?: string;
  version: string;
  description: string;
  enabled: boolean;
  is_default: boolean;
  parameters: Record<string, unknown>;
  stop_summary?: string;
  target_summary?: string;
  supported_exec_types?: Array<'C' | 'L' | 'LC' | 'CC'>;
  stop_config?: Record<string, any>;
  target_config?: Record<string, any>;
}

export interface RMTemplateDTO {
  name: string;
  category: string;
  description: string;
  parameters_schema: Record<string, {
    type: string;
    default: unknown;
    label: string;
  }>;
  stop_method: string;
  target_method?: string;
}

export interface TimeExitPackDTO {
  id: string;
  base_template: string;
  template_name?: string;
  version: string;
  description: string;
  enabled: boolean;
  is_default: boolean;
  parameters: Record<string, unknown>;
  exit_summary?: string;
  exit_config?: Record<string, any>;
}

export interface TimeExitTemplateDTO {
  name: string;
  category: string;
  description: string;
  applies_to_24_7: boolean;
  parameters_schema: Record<string, {
    type: string;
    default: unknown;
    min?: number;
    max?: number;
    label: string;
  }>;
  exit_reason: string;
  exit_summary: string;
}

// =============================================================================
// PACK PREVIEW (Chart Preview tab for User Packs)
// =============================================================================

export interface PackPreviewBar {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  state: string | null;
  [key: string]: any; // indicator columns
}

export interface PackPreviewTrigger {
  timestamp: string;
  trigger_id: string;
  trigger_name: string;
  direction: string;
  type: string;
}

export interface PackPreviewResponse {
  bars: PackPreviewBar[];
  triggers: PackPreviewTrigger[];
  states: string[];
  indicator_columns: string[];
  display_type: string;
}

export function usePackPreview() {
  return useMutation({
    mutationFn: ({ slug, ...params }: { slug: string; symbol: string; timeframe: string; days: number; session: string }) =>
      apiFetch<PackPreviewResponse>(`/api/packs/builder/user-packs/${slug}/preview`, {
        method: 'POST',
        body: JSON.stringify(params),
      }),
  });
}
