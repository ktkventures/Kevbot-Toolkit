'use client';

import PageSwitch from '@/components/PageSwitch';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';

const designVersions = [
  { meta: { id: 'v1', name: 'V1 Theme Switcher', description: 'Tiered theme grid.', rationale: '' }, component: V1 },
  { meta: { id: 'v2', name: 'V2 Full Parity', description: 'Preview cards with swatches.', rationale: '' }, component: V2 },
  { meta: { id: 'v3', name: 'V3 Streamlined', description: 'Simple radio list.', rationale: '' }, component: V3 },
  { meta: { id: 'v4', name: 'V4 Creative', description: 'Live mini-dashboard previews.', rationale: '' }, component: V4 },
];

// V1 is the working theme switcher — use it directly as the live component
export default function Page() {
  return <PageSwitch live={V1} versions={designVersions} pageKey="settings-themes" />;
}
