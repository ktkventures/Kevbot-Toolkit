'use client';

import PageSwitch from '@/components/PageSwitch';
import PortfoliosPage from '@/views/PortfoliosPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V5 from './versions/V5';

const designVersions = [
  { meta: { id: 'v1', name: 'V1 Initial', description: 'Two-column grid of portfolio cards.', rationale: '' }, component: V1 },
  { meta: { id: 'v2', name: 'V2 Full Parity', description: 'Rich portfolio cards with deploy status.', rationale: '' }, component: V2 },
  { meta: { id: 'v5', name: 'V5 Refined', description: 'Production design with mock data.', rationale: '' }, component: V5 },
];

export default function Page() {
  return <PageSwitch live={PortfoliosPage} versions={designVersions} pageKey="portfolios" />;
}
