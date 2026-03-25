'use client';

import PageSwitch from '@/components/PageSwitch';
import MassResultsPage from '@/views/MassResultsPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';
import V5 from './versions/V5';

const designVersions = [
  { meta: { id: 'v1', name: 'V1 Initial', description: 'Simple card placeholder.', rationale: '' }, component: V1 },
  { meta: { id: 'v2', name: 'V2 Full Parity', description: 'Saved search list with expand/collapse.', rationale: '' }, component: V2 },
  { meta: { id: 'v3', name: 'V3 Streamlined', description: 'Expandable list with results table.', rationale: '' }, component: V3 },
  { meta: { id: 'v4', name: 'V4 Creative', description: 'Results Explorer with portfolio builder slots.', rationale: '' }, component: V4 },
  { meta: { id: 'v5', name: 'V5 Production', description: 'Simple cards with worker progress.', rationale: '' }, component: V5 },
];

export default function Page() {
  return <PageSwitch live={MassResultsPage} versions={designVersions} pageKey="mass-results" />;
}
