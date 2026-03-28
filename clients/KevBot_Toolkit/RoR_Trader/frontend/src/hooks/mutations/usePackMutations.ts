/**
 * Mutation hooks for saving pack data.
 *
 * All three pack types use the "save full list" pattern — the frontend
 * modifies local state and sends the complete updated list on save.
 */

import { useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

// Inline DTO types to avoid circular import with usePacks.ts
interface ConfluenceGroupDTO { id: string; base_template: string; version: string; description: string; enabled: boolean; is_default: boolean; parameters: Record<string, unknown>; plot_settings: any; }
interface GeneralPackDTO { id: string; base_template: string; version: string; description: string; enabled: boolean; is_default: boolean; parameters: Record<string, unknown>; }
interface RiskManagementPackDTO { id: string; base_template: string; version: string; description: string; enabled: boolean; is_default: boolean; parameters: Record<string, unknown>; }

export function useSaveConfluenceGroups() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (groups: ConfluenceGroupDTO[]) =>
      apiFetch('/api/packs/confluence-groups', {
        method: 'PUT',
        body: JSON.stringify(groups),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['confluence-groups'] });
      queryClient.invalidateQueries({ queryKey: ['confluence-triggers'] });
    },
  });
}

export function useSaveGeneralPacks() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (packs: GeneralPackDTO[]) =>
      apiFetch('/api/packs/general', {
        method: 'PUT',
        body: JSON.stringify(packs),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['general-packs'] });
    },
  });
}

export function useSaveRiskManagementPacks() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (packs: RiskManagementPackDTO[]) =>
      apiFetch('/api/packs/risk-management', {
        method: 'PUT',
        body: JSON.stringify(packs),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['risk-management-packs'] });
    },
  });
}
