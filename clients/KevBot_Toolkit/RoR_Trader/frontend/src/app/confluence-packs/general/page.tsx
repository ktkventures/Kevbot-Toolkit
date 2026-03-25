'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';
import V5 from './versions/V5';
import { useGeneralPacks, useGeneralTemplates } from '@/hooks/queries/usePacks';
import type { GeneralPackDTO } from '@/hooks/queries/usePacks';
import { useSaveGeneralPacks } from '@/hooks/mutations/usePackMutations';

/**
 * Transform API general pack into the GeneralPack shape V5 expects.
 */
function apiPackToGeneralPack(pack: GeneralPackDTO, templates: Record<string, any>): any {
  const template = templates[pack.base_template];
  if (!template) return null;

  const params = Object.entries(template.parameters_schema || {}).map(([key, schema]: [string, any]) => ({
    key,
    label: schema.label || key,
    type: schema.type === 'bool' ? 'bool' : schema.type === 'select' ? 'select' : schema.type === 'float' ? 'float' : 'int',
    value: pack.parameters[key] ?? schema.default,
    default: schema.default,
    min: schema.min ?? 0,
    max: schema.max ?? 999,
    options: schema.options,
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
    conditionLogic: template.condition_logic || '',
    params,
    outputs: (template.outputs || []).map((code: string) => ({
      code,
      description: code,
    })),
    triggers: (template.triggers || []).map((t: any) => ({
      id: t.id,
      name: t.name,
      sentiment: t.direction === 'LONG' ? 'bullish' : t.direction === 'SHORT' ? 'bearish' : 'neutral',
      description: t.name,
    })),
  };
}

/**
 * Transform GeneralPack back to API GeneralPackDTO for saving.
 */
function generalPackToApiPack(pack: any): GeneralPackDTO {
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

function WiredV5() {
  const { data: packs, isLoading: packsLoading } = useGeneralPacks();
  const { data: templates, isLoading: templatesLoading } = useGeneralTemplates();
  const saveMutation = useSaveGeneralPacks();

  const isLoading = packsLoading || templatesLoading;

  if (isLoading) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>
        Loading general packs...
      </div>
    );
  }

  let apiPacks: any[] | undefined;
  if (packs && templates) {
    apiPacks = packs
      .map((p) => apiPackToGeneralPack(p, templates))
      .filter(Boolean);
  }

  const handleSave = (packList: any[]) => {
    const apiGroups = packList.map(generalPackToApiPack);
    saveMutation.mutate(apiGroups);
  };

  return <V5 apiPacks={apiPacks} onSave={handleSave} />;
}

const versions = [
  {
    meta: {
      id: 'v1-initial',
      name: 'Initial Scaffold',
      description: 'Toggle-switch pack list with category tags, 4-tab detail view (Parameters, Outputs & Triggers, Preview, Danger Zone), and create modal.',
      rationale: 'First pass — covers the Session and Calendar pack types with enable/disable toggles, output state previews, and pack CRUD.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Streamlit Parity',
      description: 'Template-driven architecture with real general_packs.py data: Time of Day, Trading Session, Day of Week, Calendar Filter. 5-tab detail view (Parameters with template-specific forms, Outputs & Triggers with condition/trigger split, Preview with state timeline and trigger events, Code with evaluator functions and JSON config, Danger Zone). Category-grouped list with Time and Calendar headers, contextual parameter previews (time windows, session hours, day toggles), and create modal with template selector.',
      rationale: 'Mirrors the Streamlit render_general_packs() page — real template schemas, bool/select/int/float parameter types, condition-only packs (Day of Week), trigger-equipped packs (Time of Day, Calendar Filter), and template-specific contextual summaries.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'Accordion-style inline editing with category filter pills (All | Session | Calendar), overflow menu for rare actions (Rename, Copy, Delete). No page navigation.',
      rationale: 'Removed friction: packs expand inline to show parameters (bool toggles, selects, number inputs) + quick reference (outputs, triggers) side-by-side. All editing happens on one page. Overflow menu hides rare actions. Category pills replace tab-based grouping.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: '24-hour activity timeline showing session, time window, and overlap bands with live "now" marker. Live ticking clock with session detection. Week-at-a-glance calendar grid with FOMC/NFP event badges and blocked-day highlights. Combined gate status (TRADE/WAIT). Inline SVG icons per template (clock, session bars, day-of-week columns, calendar grid). Pack cards with state pulse animation, detail modal with slider-based parameter editing, and visual output state badges.',
      rationale: 'General packs are inherently time-based, so the V4 creative hook is temporal visualization: a timeline that shows when each condition is active throughout the day, plus a weekly calendar that combines day-of-week and calendar filter into one view. The live clock reinforces that these packs gate real-time trading decisions.',
    },
    component: V4,
  },
  {
    meta: {
      id: 'v5-production',
      name: 'Production',
      description: 'All 4 templates, locked params after save, nested variations, search, draft mode with "Save as Variation", tooltips on state/trigger counts.',
      rationale: 'Adapted from TF Confluence V5 for general packs. Same patterns: locked params, nested variations, search, draft mode, "Create New Template" → Pack Builder. Simplified for scalar conditions: no plot settings tab, no trigger parameters tab (bar-close only), no execution type variants, no fidelity badges. Binary output states, 0-2 triggers per template, template-specific parameter inputs (time ranges, session selects, day toggles, event toggles).',
    },
    component: V5,
  },
  {
    meta: {
      id: 'v5-wired',
      name: 'Production (Live)',
      description: 'All 4 templates wired to real API data from FastAPI general packs endpoint.',
      rationale: 'Production version with live data — same UI as V5 but fetches packs from the API.',
    },
    component: WiredV5,
  },
];

export default function GeneralPacksPage() {
  return <VersionedPage pageKey="general-packs" versions={versions} />;
}
