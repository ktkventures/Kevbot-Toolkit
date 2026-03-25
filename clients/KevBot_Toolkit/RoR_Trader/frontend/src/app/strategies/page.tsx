'use client';

import PageSwitch from '@/components/PageSwitch';
import StrategiesPage from '@/views/StrategiesPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';
import V5 from './versions/V5';

const designVersions = [
  { meta: { id: 'v1', name: 'V1 Initial', description: 'Strategy list with cards.', rationale: '' }, component: V1 },
  { meta: { id: 'v2', name: 'V2 Full Parity', description: 'All Streamlit features.', rationale: '' }, component: V2 },
  { meta: { id: 'v3', name: 'V3 Streamlined', description: 'Compact table view.', rationale: '' }, component: V3 },
  { meta: { id: 'v4', name: 'V4 Creative', description: 'Command center.', rationale: '' }, component: V4 },
  { meta: { id: 'v5', name: 'V5 Refined', description: 'Production design with mock data.', rationale: '' }, component: V5 },
];

export default function Page() {
  return <PageSwitch live={StrategiesPage} versions={designVersions} pageKey="strategies" />;
}
