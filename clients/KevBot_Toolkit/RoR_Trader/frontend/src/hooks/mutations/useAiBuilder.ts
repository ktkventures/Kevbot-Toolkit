/**
 * Mutation hooks for the Pack Builder AI wizard.
 *
 * Five hooks matching the five backend endpoints in ai_builder.py:
 *   1. useGenerateStructure — Step 2: AI proposes params/outputs/triggers
 *   2. useGenerateCode     — Step 4: AI generates manifest + code
 *   3. useRequestFix       — Step 4/5: AI fixes validation errors
 *   4. useValidatePack     — Re-validate after manual edits (no AI)
 *   5. useInstallPack      — Write to user_packs/ and register
 */

import { useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

export interface GenerateStructureRequest {
  pack_name: string;
  pack_type: string;
  category: string;
  display_type: string;
  description: string;
  pine_script?: string;
  ai_model: string;
}

export interface GenerateCodeRequest {
  pack_name: string;
  slug: string;
  pack_type: string;
  category: string;
  display_type: string;
  description: string;
  parameters: Record<string, unknown>[];
  outputs: Record<string, unknown>[];
  triggers: Record<string, unknown>[];
  pine_script?: string;
  ai_model: string;
}

export interface FixRequest {
  manifest: Record<string, unknown>;
  indicator_code: string;
  interpreter_code: string;
  validation_errors: string[];
  user_description?: string;
  ai_model: string;
}

export interface ValidateRequest {
  manifest: Record<string, unknown>;
  indicator_code: string;
  interpreter_code: string;
}

export interface InstallRequest {
  manifest: Record<string, unknown>;
  indicator_code: string;
  interpreter_code: string;
  pine_script_code?: string | null;
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

export interface GenerateStructureResponse {
  parameters: Array<{ name: string; type: string; default: unknown; min?: number; max?: number; label: string }>;
  outputs: Array<{ code: string; description: string }>;
  triggers: Array<{ name: string; base: string; sentiment: string; direction: string; type: string; execution: string }>;
  ai_message: string;
}

export interface ValidationItemDTO {
  id: string;
  label: string;
  category: string;
  status: string;
  message: string;
}

export interface GenerateCodeResponse {
  manifest: Record<string, unknown>;
  indicator_code: string;
  interpreter_code: string;
  pine_script_code: string | null;
  validation: ValidationItemDTO[];
  ai_message: string;
}

export interface FixResponse {
  manifest: Record<string, unknown>;
  indicator_code: string;
  interpreter_code: string;
  pine_script_code: string | null;
  validation: ValidationItemDTO[];
  ai_message: string;
}

export interface ValidateResponse {
  is_valid: boolean;
  validation: ValidationItemDTO[];
}

export interface InstallResponse {
  success: boolean;
  slug: string;
  errors: string[];
}

// ---------------------------------------------------------------------------
// Mutation hooks
// ---------------------------------------------------------------------------

export function useGenerateStructure() {
  return useMutation({
    mutationFn: (params: GenerateStructureRequest) =>
      apiFetch<GenerateStructureResponse>('/api/packs/builder/generate-structure', {
        method: 'POST',
        body: JSON.stringify(params),
      }),
  });
}

export function useGenerateCode() {
  return useMutation({
    mutationFn: (params: GenerateCodeRequest) =>
      apiFetch<GenerateCodeResponse>('/api/packs/builder/generate-code', {
        method: 'POST',
        body: JSON.stringify(params),
      }),
  });
}

export function useRequestFix() {
  return useMutation({
    mutationFn: (params: FixRequest) =>
      apiFetch<FixResponse>('/api/packs/builder/fix', {
        method: 'POST',
        body: JSON.stringify(params),
      }),
  });
}

export function useValidatePack() {
  return useMutation({
    mutationFn: (params: ValidateRequest) =>
      apiFetch<ValidateResponse>('/api/packs/builder/validate', {
        method: 'POST',
        body: JSON.stringify(params),
      }),
  });
}

export function useInstallPack() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (params: InstallRequest) =>
      apiFetch<InstallResponse>('/api/packs/builder/install', {
        method: 'POST',
        body: JSON.stringify(params),
      }),
    onSuccess: (data) => {
      if (data.success) {
        // Invalidate pack queries so newly installed pack appears
        queryClient.invalidateQueries({ queryKey: ['confluence-groups'] });
        queryClient.invalidateQueries({ queryKey: ['confluence-triggers'] });
        queryClient.invalidateQueries({ queryKey: ['confluence-templates'] });
      }
    },
  });
}
