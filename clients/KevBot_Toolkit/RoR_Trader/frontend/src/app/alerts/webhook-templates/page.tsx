'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';
import V5 from './versions/V5';
import { useWebhookTemplates } from '@/hooks/queries/useWebhooks';
import { useCreateWebhookTemplate, useDeleteWebhookTemplate } from '@/hooks/mutations/useWebhookMutations';

/**
 * Transform API webhook template to V5 AccountTemplate shape.
 */
function apiToAccountTemplate(t: any): any {
  return {
    id: t.id || t.template_id || '',
    name: t.name || 'Untitled',
    exchange: t.exchange || t.service || 'Custom',
    url: t.url || t.webhook_url || '',
    isDefault: t.is_default ?? false,
    usedByPortfolios: t.used_by_portfolios ?? 0,
    eventCount: t.event_count ?? 11,
    lastDelivery: t.last_delivery ? { time: t.last_delivery.time, status: t.last_delivery.status } : undefined,
    entryCfg: t.entry_cfg || 'Market',
    exitCfg: t.exit_cfg || 'Market',
  };
}

function WiredV5() {
  const { data: templates, isLoading } = useWebhookTemplates();
  const createMutation = useCreateWebhookTemplate();
  const deleteMutation = useDeleteWebhookTemplate();

  if (isLoading) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>
        Loading webhook templates...
      </div>
    );
  }

  const v5Templates = templates
    ? templates.map(apiToAccountTemplate)
    : undefined;

  return (
    <V5
      apiTemplates={v5Templates}
      onDelete={(id) => deleteMutation.mutate(id)}
      onCreate={(template) => createMutation.mutate(template)}
    />
  );
}

const versions = [
  {
    meta: {
      id: 'v1-initial',
      name: 'Initial Scaffold',
      description: 'Simple card list of webhook templates with Edit and Test action buttons.',
      rationale: 'First pass — minimal layout showing template names with inline actions. Establishes the structure for payload template management.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Parity',
      description: 'Complete template management: rich cards with URL/type/usage, full editor modal with payload textarea, placeholder picker, custom headers, rendered preview, test delivery with response display, and clone/delete with confirmation.',
      rationale: 'Matches Streamlit webhook template system — CRUD, placeholder-based payload editing, live preview, and test delivery workflow.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'Compact table-like rows with name, masked URL, usage count, and inline Edit/Test/Delete icon buttons. Simple modal for edit/create with name, URL, and payload textarea. Inline test feedback, inline delete confirmation. No placeholder picker, no headers editor, no response display.',
      rationale: 'Maximum density — all templates visible at once in a scannable list. Test sends inline with status feedback. Editing uses a minimal modal without advanced features.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Webhook Studio with rich template cards showing status indicators, delivery stats, and expandable detail panels. Features payload diff view (template vs rendered), delivery timeline with status dots, response inspector with headers/body, placeholder palette organized by category (signal/strategy/position/meta), live payload preview in editor, and aggregate delivery metrics strip.',
      rationale: 'Elevates webhook management into a full development studio. The payload diff view shows template vs rendered side-by-side. The delivery timeline visualizes webhook health at a glance. The response inspector shows exactly what came back from test sends. The placeholder palette makes it easy to build payloads without memorizing variable names.',
    },
    component: V4,
  },
  {
    meta: {
      id: 'v5-account-based',
      name: 'Account-Based Templates (Phase 39)',
      description: 'Account-based webhook templates where each template defines JSON payloads for all 11 event types (entry/exit market/limit, cancel, compliance). Templates are tied to an exchange/service. Portfolios select a template instead of manually wiring individual webhooks. Category-filtered event list, inline payload editor with resolved preview, placeholder reference, all-events overview table, and delivery history.',
      rationale: 'Phase 39 Webhook Event System redesign. Shifts from individual payload templates to account-based templates that cover all event types. One-time setup per exchange, portfolios just pick a template. Driven by the new execution type parameters that require market/limit/cancel differentiation.',
    },
    component: V5,
  },
  {
    meta: {
      id: 'v5-wired',
      name: 'Production (Live)',
      description: 'Webhook templates wired to real API data from FastAPI webhooks endpoint.',
      rationale: 'Production version with live data.',
    },
    component: WiredV5,
  },
];

export default function WebhookTemplatesPage() {
  return <VersionedPage pageKey="webhook-templates" versions={versions} />;
}
