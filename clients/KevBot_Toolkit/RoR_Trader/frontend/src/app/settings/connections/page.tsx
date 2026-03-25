'use client';

import PageSwitch from '@/components/PageSwitch';
import SettingsConnectionsPage from '@/views/SettingsConnectionsPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';
import V5 from './versions/V5';

const designVersions = [
  { meta: { id: 'v1', name: 'V1 Initial', description: 'Connection status card.', rationale: '' }, component: V1 },
  { meta: { id: 'v2', name: 'V2 Full Parity', description: 'Full connection management.', rationale: '' }, component: V2 },
  { meta: { id: 'v3', name: 'V3 Streamlined', description: 'Status dots only.', rationale: '' }, component: V3 },
  { meta: { id: 'v4', name: 'V4 Creative', description: 'Network topology diagram.', rationale: '' }, component: V4 },
  { meta: { id: 'v5', name: 'V5 Production', description: 'Topology + Alert Engine admin.', rationale: '' }, component: V5 },
];

export default function Page() {
  return <PageSwitch live={SettingsConnectionsPage} versions={designVersions} pageKey="settings-connections" />;
}
