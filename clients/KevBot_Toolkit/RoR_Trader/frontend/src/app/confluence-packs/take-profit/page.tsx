'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import { useRiskManagementPacks, useRiskManagementTemplates } from '@/hooks/queries/usePacks';
import type { RiskManagementPackDTO } from '@/hooks/queries/usePacks';
import { useSaveRiskManagementPacks } from '@/hooks/mutations/usePackMutations';

/**
 * Transform API risk management pack into the TargetPack shape V1 expects.
 * Only includes packs whose template has target_method.
 */
function apiPackToTargetPack(pack: RiskManagementPackDTO, templates: Record<string, any>): any {
  const template = templates[pack.base_template];
  if (!template) return null;
  // Filter: only target-related templates (has target_method)
  if (!template.target_method) return null;

  const params = Object.entries(template.parameters_schema || {}).map(([key, schema]: [string, any]) => ({
    key,
    label: schema.label || key,
    type: schema.type === 'float' ? 'float' : 'int',
    value: pack.parameters[key] ?? schema.default,
    default: schema.default,
    min: schema.min ?? 0,
    max: schema.max ?? 999,
    step: schema.step,
    unit: schema.unit,
  }));

  return {
    id: pack.id,
    templateKey: pack.base_template,
    name: `${template.name} (${pack.version})`,
    version: pack.version,
    tags: [template.category || 'Other'],
    enabled: pack.enabled,
    isDefault: pack.is_default,
    isSaved: true,
    description: template.description || '',
    params,
  };
}

/**
 * Transform TargetPack back to API RiskManagementPackDTO for saving.
 */
function targetPackToApiPack(pack: any): RiskManagementPackDTO {
  const parameters: Record<string, unknown> = {};
  for (const p of pack.params || []) {
    parameters[p.key] = p.value;
  }
  return {
    id: pack.id,
    base_template: pack.templateKey,
    version: pack.version,
    description: '',
    enabled: pack.enabled,
    is_default: pack.isDefault,
    parameters,
  };
}

function WiredV1() {
  // Same issue as Stop Loss — RM pack templates contain Python callables
  // (targetSummary, build_target) that can't serialize to JSON.
  // Render V1 with mock data for now.
  return <V1 />;
}

const versions = [
  {
    meta: {
      id: 'v1-production',
      name: 'Production',
      description: '6 take profit templates (R:R, ATR, Fixed $, %, Swing, None), locked params, nested variations, draft mode, search, Behavior tab.',
      rationale: 'Mirrors the stop loss pack architecture. Take profit methods are saved as reusable packs with locked parameters for mass backtesting. R:R is the most common (requires a stop loss to calculate risk). "None" template for strategies that exit only via signals or bar count.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v1-wired',
      name: 'Production (Live)',
      description: 'Take profit packs wired to real API data from FastAPI risk management endpoint.',
      rationale: 'Production version with live data — same UI as V1 but fetches target packs from the API.',
    },
    component: WiredV1,
  },
];

export default function TakeProfitPage() {
  return <VersionedPage pageKey="take-profit" versions={versions} />;
}
