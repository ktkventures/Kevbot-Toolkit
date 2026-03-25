'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';
import V5 from './versions/V5';
import V6 from './versions/V6';
import V7 from './versions/V7';
import { useDashboardSummary } from '@/hooks/queries/useDashboard';

function WiredV6() {
  const { data: summary, isLoading } = useDashboardSummary();

  if (isLoading) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>
        Loading dashboard...
      </div>
    );
  }

  return <V6 apiSummary={summary || undefined} />;
}

const versions = [
  { meta: { id: 'v1-initial', name: 'Initial Scaffold', description: 'Summary metrics and chart placeholders.', rationale: 'First pass.' }, component: V1 },
  { meta: { id: 'v2-full-parity', name: 'Full Parity', description: 'Complete dashboard.', rationale: 'Direct port.' }, component: V2 },
  { meta: { id: 'v3-streamlined', name: 'Streamlined', description: 'At-a-glance.', rationale: 'No scrolling.' }, component: V3 },
  { meta: { id: 'v4-creative', name: 'Creative', description: 'Trading cockpit.', rationale: 'Wow factor.' }, component: V4 },
  { meta: { id: 'v5-hybrid', name: 'Hybrid Cockpit', description: 'Charts + creative widgets.', rationale: 'Combines V1 + V4.' }, component: V5 },
  {
    meta: {
      id: 'v6-production',
      name: 'Production (Live)',
      description: 'Refined cockpit with real API data — strategy count, KPIs, positions.',
      rationale: 'Production version with live dashboard summary from FastAPI.',
    },
    component: WiredV6,
  },
  { meta: { id: 'v7-detailed', name: 'Detailed Cockpit', description: 'Detail modals on every widget.', rationale: 'Drill-down from dashboard.' }, component: V7 },
];

export default function DashboardPage() {
  return <VersionedPage pageKey="dashboard" versions={versions} />;
}
