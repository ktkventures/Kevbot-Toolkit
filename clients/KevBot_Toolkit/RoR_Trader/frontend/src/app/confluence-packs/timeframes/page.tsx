'use client';

import PageSwitch from '@/components/PageSwitch';
import TimeframesPage from '@/views/TimeframesPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';
import V5 from './versions/V5';

const designVersions = [
  { meta: { id: 'v1', name: 'V1 Initial', description: 'Timeframe grid with toggles.', rationale: '' }, component: V1 },
  { meta: { id: 'v2', name: 'V2 Full Parity', description: 'Rich TF cards with usage stats.', rationale: '' }, component: V2 },
  { meta: { id: 'v3', name: 'V3 Streamlined', description: 'Compact horizontal rows.', rationale: '' }, component: V3 },
  { meta: { id: 'v4', name: 'V4 Creative', description: 'Timeframe Explorer with hierarchy tree.', rationale: '' }, component: V4 },
  { meta: { id: 'v5', name: 'V5 Use Case Grid', description: 'Grid table with 4 use case columns.', rationale: '' }, component: V5 },
];

export default function Page() {
  return <PageSwitch live={TimeframesPage} versions={designVersions} pageKey="timeframes" />;
}
