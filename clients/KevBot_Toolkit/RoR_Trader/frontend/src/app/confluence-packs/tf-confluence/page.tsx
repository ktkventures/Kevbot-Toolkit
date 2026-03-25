'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';
import V5 from './versions/V5';
import { useConfluenceGroups, useConfluenceTemplates } from '@/hooks/queries/usePacks';
import { useSaveConfluenceGroups } from '@/hooks/mutations/usePackMutations';
import type { ConfluenceGroupDTO } from '@/hooks/queries/usePacks';

/**
 * Transform API confluence groups into the TfPack shape V5 expects.
 * Maps from Python ConfluenceGroup dataclass → V5 TfPack interface.
 */
function apiGroupToTfPack(group: ConfluenceGroupDTO, templates: Record<string, any>): any {
  const template = templates[group.base_template];
  if (!template) return null;

  // Build params array from template schema + group's stored values
  const params = Object.entries(template.parameters_schema || {}).map(([key, schema]: [string, any]) => ({
    key,
    label: schema.label || key,
    type: schema.type === 'float' ? 'float' : 'int',
    value: group.parameters[key] ?? schema.default,
    default: schema.default,
    min: schema.min ?? 0,
    max: schema.max ?? 999,
  }));

  // Build plot settings
  const plotSettings = Object.entries(template.plot_schema || {}).map(([key, schema]: [string, any]) => ({
    key,
    label: schema.label || key,
    value: group.plot_settings?.colors?.[key] || schema.default || '#ffffff',
  }));

  return {
    id: group.id,
    templateKey: group.base_template,
    name: `${template.name} (${group.version})`,
    version: group.version,
    tags: [template.category || 'Other'],
    enabled: group.enabled,
    isDefault: group.is_default,
    isSaved: true,
    params,
    plotSettings,
    outputs: Object.entries(template.outputs || {}).map(([code, desc]: [string, any]) => ({
      code,
      description: typeof desc === 'string' ? desc : code,
    })),
    triggers: (template.triggers || []).map((t: any) => ({
      id: t.id,
      name: t.name,
      sentiment: t.direction === 'LONG' ? 'bullish' : t.direction === 'SHORT' ? 'bearish' : 'neutral',
      description: t.name,
      execVariants: {
        C: { enabled: true, referenceBar: 0, orderType: 'market' },
        L: { enabled: true, referenceBar: -1, orderType: 'market', holdSeconds: 0 },
        LC: { enabled: true, referenceBar: -1, orderType: 'market', confirmBarOffset: 0, bailAction: 'exit_market' },
        CC: { enabled: true, referenceBar: 0, orderType: 'market', confirmBarOffset: 1, bailAction: 'exit_market' },
      },
    })),
  };
}

/**
 * Transform TfPack back to API ConfluenceGroupDTO for saving.
 */
function tfPackToApiGroup(pack: any): ConfluenceGroupDTO {
  const parameters: Record<string, unknown> = {};
  for (const p of pack.params || []) {
    parameters[p.key] = p.value;
  }
  const colors: Record<string, string> = {};
  for (const ps of pack.plotSettings || []) {
    colors[ps.key] = ps.value;
  }
  return {
    id: pack.id,
    base_template: pack.templateKey,
    version: pack.version,
    description: '',
    enabled: pack.enabled,
    is_default: pack.isDefault,
    parameters,
    plot_settings: { colors, line_width: 1, visible: true },
  };
}

function WiredV5() {
  const { data: groups, isLoading: groupsLoading } = useConfluenceGroups();
  const { data: templates, isLoading: templatesLoading } = useConfluenceTemplates();
  const saveMutation = useSaveConfluenceGroups();

  const isLoading = groupsLoading || templatesLoading;

  if (isLoading) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>
        Loading confluence groups...
      </div>
    );
  }

  // Transform API data to V5 TfPack format
  let apiPacks: any[] | undefined;
  if (groups && templates) {
    apiPacks = groups
      .map((g) => apiGroupToTfPack(g, templates))
      .filter(Boolean);
  }

  const handleSave = (packs: any[]) => {
    const apiGroups = packs.map(tfPackToApiGroup);
    saveMutation.mutate(apiGroups);
  };

  return <V5 apiPacks={apiPacks} onSave={handleSave} />;
}

const versions = [
  {
    meta: {
      id: 'v1-initial',
      name: 'Initial Scaffold',
      description: 'Clean card layout with toggle switches, category tags, and 5-tab detail view.',
      rationale: 'First pass — focused on getting the core structure right.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Parity',
      description: 'Everything from Streamlit, re-designed.',
      rationale: 'Direct port of every feature from the Streamlit app.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'Accordion-style inline editing.',
      rationale: 'Removed friction.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Live dashboard with SVG previews.',
      rationale: 'Reimagined as live dashboard.',
    },
    component: V4,
  },
  {
    meta: {
      id: 'v5-production',
      name: 'Production (Live)',
      description: 'All 8 templates, locked params, nested variations — wired to real API data.',
      rationale: 'Production version with live data from FastAPI.',
    },
    component: WiredV5,
  },
];

export default function TfConfluencePage() {
  return <VersionedPage pageKey="tf-confluence" versions={versions} />;
}
