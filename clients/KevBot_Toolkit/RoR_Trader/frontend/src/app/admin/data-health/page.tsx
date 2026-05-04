'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';

const versions = [
  {
    meta: {
      id: 'v1-coverage-table',
      name: 'Coverage Table',
      description: 'Per-(symbol, timeframe) coverage table with rolling-window stats, gap counts, freshness, and source split.',
      rationale: 'Diagnostic dashboard for the live_bars cache. Color-coded coverage cells make missing-AM-event patterns visible at a glance. Backs the question "is the data I\'m looking at actually being collected?"',
    },
    component: V1,
  },
];

export default function AdminDataHealthPage() {
  return <VersionedPage pageKey="admin-data-health" versions={versions} />;
}
