'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';

const versions = [
  {
    meta: {
      id: 'v1-overview-table',
      name: 'Health Overview',
      description: 'Per-strategy freshness + red-flag table. Snapshot age, KPI age, latest trade, parity, discrepancies — sortable and filterable.',
      rationale: 'Operational view across the whole strategy fleet. Answers "is the data-worker actually keeping every strategy current, or are some quietly stale?" Click red-flag chips to filter.',
    },
    component: V1,
  },
];

export default function AdminStrategyHealthPage() {
  return <VersionedPage pageKey="admin-strategy-health" versions={versions} />;
}
