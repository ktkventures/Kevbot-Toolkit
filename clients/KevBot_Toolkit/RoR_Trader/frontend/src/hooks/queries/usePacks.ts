/**
 * React Query hooks for all pack types:
 * - Confluence Groups (TF packs)
 * - General Packs
 * - Risk Management Packs
 */

import { useQuery } from '@tanstack/react-query';
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
  version: string;
  description: string;
  enabled: boolean;
  is_default: boolean;
  parameters: Record<string, unknown>;
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
