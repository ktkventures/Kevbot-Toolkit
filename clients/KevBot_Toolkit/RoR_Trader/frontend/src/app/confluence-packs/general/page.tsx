'use client';

import PageSwitch from '@/components/PageSwitch';
import GeneralPacksPage from '@/views/GeneralPacksPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';
import V5 from './versions/V5';

const designVersions = [
  { meta: { id: 'v1', name: 'V1 Initial', description: 'Toggle-switch pack list with category tags.', rationale: '' }, component: V1 },
  { meta: { id: 'v2', name: 'V2 Full Parity', description: 'Template-driven architecture with real data shapes.', rationale: '' }, component: V2 },
  { meta: { id: 'v3', name: 'V3 Streamlined', description: 'Accordion-style inline editing with category pills.', rationale: '' }, component: V3 },
  { meta: { id: 'v4', name: 'V4 Creative', description: '24-hour activity timeline with session visualization.', rationale: '' }, component: V4 },
  { meta: { id: 'v5', name: 'V5 Production', description: 'All 4 templates, locked params, nested variations.', rationale: '' }, component: V5 },
];

export default function Page() {
  return <PageSwitch live={GeneralPacksPage} versions={designVersions} pageKey="general-packs" />;
}
