'use client';

import PageSwitch from '@/components/PageSwitch';
import MassBuilderPage from '@/views/MassBuilderPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';
import V5 from './versions/V5';
import V6 from './versions/V6';

const designVersions = [
  { meta: { id: 'v1', name: 'V1 Initial', description: 'Two-panel layout with config sidebar.', rationale: '' }, component: V1 },
  { meta: { id: 'v2', name: 'V2 Full Parity', description: 'Complete mass builder with 9-tab config.', rationale: '' }, component: V2 },
  { meta: { id: 'v3', name: 'V3 Streamlined', description: 'Two-column layout with results table.', rationale: '' }, component: V3 },
  { meta: { id: 'v4', name: 'V4 Creative', description: 'Strategy Discovery Lab.', rationale: '' }, component: V4 },
  { meta: { id: 'v5', name: 'V5 Production', description: 'V2 refined with pack selectors.', rationale: '' }, component: V5 },
  { meta: { id: 'v6', name: 'V6 Strategy Cards', description: 'Strategy-style result cards.', rationale: '' }, component: V6 },
];

export default function Page() {
  return <PageSwitch live={MassBuilderPage} versions={designVersions} pageKey="mass-builder" />;
}
