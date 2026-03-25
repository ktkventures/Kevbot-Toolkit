'use client';

import PageSwitch from '@/components/PageSwitch';
import SettingsDisplayPage from '@/views/SettingsDisplayPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';
import V5 from './versions/V5';

const designVersions = [
  { meta: { id: 'v1', name: 'V1 Initial', description: 'Display preferences form.', rationale: '' }, component: V1 },
  { meta: { id: 'v2', name: 'V2 Full Parity', description: 'Complete display settings with live preview.', rationale: '' }, component: V2 },
  { meta: { id: 'v3', name: 'V3 Streamlined', description: 'Just timezone and date format.', rationale: '' }, component: V3 },
  { meta: { id: 'v4', name: 'V4 Creative', description: 'Visual preference builder with live preview.', rationale: '' }, component: V4 },
  { meta: { id: 'v5', name: 'V5 Tabbed', description: 'Tab navigation: Charts, Formatting, Components, Tables.', rationale: '' }, component: V5 },
];

export default function Page() {
  return <PageSwitch live={SettingsDisplayPage} versions={designVersions} pageKey="settings-display" />;
}
