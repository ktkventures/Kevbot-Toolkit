'use client';

import VersionedPage from '@/components/VersionedPage';
import V5 from './versions/V5';

const versions = [
  {
    meta: {
      id: 'v5-account-detail',
      name: 'Account-Based Template Detail (Phase 39)',
      description: 'Detail view for account-based webhook templates with 4 tabs: Event Payloads (category-filtered editor with resolved preview), Placeholders (grouped reference with copy), Delivery History (metrics + timeline + log), Settings (config + portfolio usage + danger zone).',
      rationale: 'Phase 39 detail page matching the strategy detail page pattern. Tabs for each concern, category-filtered event payloads, delivery analytics.',
    },
    component: V5,
  },
];

export default function WebhookTemplateDetailPage() {
  return <VersionedPage pageKey="webhook-template-detail" versions={versions} />;
}
