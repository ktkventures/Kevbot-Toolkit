'use client';

import PageSwitch from '@/components/PageSwitch';
import TfConfluencePage from '@/views/TfConfluencePage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';
import V5 from './versions/V5';

const designVersions = [
  { meta: { id: 'v1', name: 'V1 Initial', description: 'Clean card layout with toggle switches.', rationale: '' }, component: V1 },
  { meta: { id: 'v2', name: 'V2 Full Parity', description: 'Everything from Streamlit, re-designed.', rationale: '' }, component: V2 },
  { meta: { id: 'v3', name: 'V3 Streamlined', description: 'Accordion-style inline editing.', rationale: '' }, component: V3 },
  { meta: { id: 'v4', name: 'V4 Creative', description: 'Live dashboard with SVG previews.', rationale: '' }, component: V4 },
  { meta: { id: 'v5', name: 'V5 Production', description: 'All 8 templates, locked params, nested variations.', rationale: '' }, component: V5 },
];

export default function Page() {
  return <PageSwitch live={TfConfluencePage} versions={designVersions} pageKey="tf-confluence" />;
}
