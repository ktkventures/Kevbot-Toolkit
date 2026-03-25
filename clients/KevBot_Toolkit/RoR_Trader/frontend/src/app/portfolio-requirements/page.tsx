'use client';

import PageSwitch from '@/components/PageSwitch';
import RequirementsPage from '@/views/RequirementsPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';
import V5 from './versions/V5';

const designVersions = [
  { meta: { id: 'v1', name: 'V1 Initial', description: 'List/editor view for requirement sets.', rationale: '' }, component: V1 },
  { meta: { id: 'v2', name: 'V2 Full Parity', description: 'Enhanced with progress bars and presets.', rationale: '' }, component: V2 },
  { meta: { id: 'v3', name: 'V3 Streamlined', description: 'Inline editing, no modals.', rationale: '' }, component: V3 },
  { meta: { id: 'v4', name: 'V4 Creative', description: 'Visual rule builder with compliance rings.', rationale: '' }, component: V4 },
  { meta: { id: 'v5', name: 'V5 Trade Qualification', description: 'Production design with TQ rules.', rationale: '' }, component: V5 },
];

export default function Page() {
  return <PageSwitch live={RequirementsPage} versions={designVersions} pageKey="portfolio-requirements" />;
}
