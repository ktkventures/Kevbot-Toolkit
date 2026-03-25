'use client';

import PageSwitch from '@/components/PageSwitch';
import AlertsPage from '@/views/AlertsPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';
import V5 from './versions/V5';

const designVersions = [
  { meta: { id: 'v1', name: 'V1 Initial', description: 'Engine status, strategy monitors, alert feed.', rationale: '' }, component: V1 },
  { meta: { id: 'v2', name: 'V2 Full Parity', description: 'Complete alert system with config.', rationale: '' }, component: V2 },
  { meta: { id: 'v3', name: 'V3 Streamlined', description: 'Two-panel inline layout.', rationale: '' }, component: V3 },
  { meta: { id: 'v4', name: 'V4 Creative', description: 'Alert command center with radar.', rationale: '' }, component: V4 },
  { meta: { id: 'v5', name: 'V5 Production', description: '4-tab alert management with webhook delivery log.', rationale: '' }, component: V5 },
];

export default function Page() {
  return <PageSwitch live={AlertsPage} versions={designVersions} pageKey="alerts" />;
}
