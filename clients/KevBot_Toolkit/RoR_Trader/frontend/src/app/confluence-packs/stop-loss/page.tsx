'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import { useRiskManagementPacks, useRiskManagementTemplates } from '@/hooks/queries/usePacks';
import type { RiskManagementPackDTO } from '@/hooks/queries/usePacks';
import { useSaveRiskManagementPacks } from '@/hooks/mutations/usePackMutations';

/**
 * Transform API risk management pack into the StopPack shape V1 expects.
 * Only includes packs whose template has stop_method (not target_method).
 */
function apiPackToStopPack(pack: RiskManagementPackDTO, templates: Record<string, any>): any {
  const template = templates[pack.base_template];
  if (!template) return null;
  // Filter: only stop-related templates (has stop_method, no target_method)
  if (template.target_method && !template.stop_method) return null;

  const params = Object.entries(template.parameters_schema || {}).map(([key, schema]: [string, any]) => ({
    key,
    label: schema.label || key,
    type: schema.type === 'float' ? 'float' : schema.type === 'bool' ? 'bool' : 'int',
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
 * Transform StopPack back to API RiskManagementPackDTO for saving.
 */
function stopPackToApiPack(pack: any): RiskManagementPackDTO {
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
  const { data: packs, isLoading: packsLoading } = useRiskManagementPacks();
  const { data: templates, isLoading: templatesLoading } = useRiskManagementTemplates();
  const saveMutation = useSaveRiskManagementPacks();

  const isLoading = packsLoading || templatesLoading;

  if (isLoading) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>
        Loading stop loss packs...
      </div>
    );
  }

  let apiPacks: any[] | undefined;
  if (packs && templates) {
    apiPacks = packs
      .map((p) => apiPackToStopPack(p, templates))
      .filter(Boolean);
  }

  const handleSave = (packList: any[]) => {
    const apiGroups = packList.map(stopPackToApiPack);
    saveMutation.mutate(apiGroups);
  };

  return <V1 apiPacks={apiPacks} onSave={handleSave} />;
}

const versions = [
  {
    meta: {
      id: 'v1-production',
      name: 'Production',
      description: '6 stop loss templates (ATR, Fixed $, %, Swing, ATR Trailing, Breakeven), locked params, nested variations, draft mode, search, Behavior tab.',
      rationale: 'Follows the same pack architecture as TF Confluence and General Packs. Stop loss methods are saved as reusable packs with locked parameters so the mass backtester can iterate through multiple stop variations. Templates cover volatility-based (ATR), fixed (dollar/percentage), structural (swing), and trailing (ATR trailing, breakeven) approaches.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v1-wired',
      name: 'Production (Live)',
      description: 'Stop loss packs wired to real API data from FastAPI risk management endpoint.',
      rationale: 'Production version with live data — same UI as V1 but fetches stop packs from the API.',
    },
    component: WiredV1,
  },
];

export default function StopLossPage() {
  return <VersionedPage pageKey="stop-loss" versions={versions} />;
}
