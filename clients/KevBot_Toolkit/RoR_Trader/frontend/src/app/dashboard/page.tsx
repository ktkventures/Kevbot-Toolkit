'use client';
import PageSwitch from '@/components/PageSwitch';
import DashboardPage from '@/views/DashboardPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';
import V5 from './versions/V5';
import V6 from './versions/V6';
import V7 from './versions/V7';

const designVersions = [
  { meta: { id: 'v1', name: 'V1', description: '', rationale: '' }, component: V1 },
  { meta: { id: 'v2', name: 'V2', description: '', rationale: '' }, component: V2 },
  { meta: { id: 'v3', name: 'V3', description: '', rationale: '' }, component: V3 },
  { meta: { id: 'v4', name: 'V4', description: '', rationale: '' }, component: V4 },
  { meta: { id: 'v5', name: 'V5', description: '', rationale: '' }, component: V5 },
  { meta: { id: 'v6', name: 'V6 Refined', description: 'Locked design.', rationale: '' }, component: V6 },
  { meta: { id: 'v7', name: 'V7 Detailed', description: 'Detail modals.', rationale: '' }, component: V7 },
];

export default function Page() {
  return <PageSwitch live={DashboardPage} versions={designVersions} pageKey="dashboard" />;
}
