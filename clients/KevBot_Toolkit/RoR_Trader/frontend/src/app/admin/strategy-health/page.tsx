'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';

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
  {
    meta: {
      id: 'v2-divergence-backlog',
      name: 'Divergence Backlog',
      description: 'Per-event divergence list — one row per phantom alert or missed backtest edge, with auto-classification. Surface the real mysteries vs known-cause divergence.',
      rationale: 'Operational view at the trade level. The Health Overview tells you which strategies are diverging; this tells you which SPECIFIC events to dig into. "Needs investigation only" hides phase-2 gaps + legacy strategies + non-fill alerts so the list is just the bugs.',
    },
    component: V2,
  },
];

export default function AdminStrategyHealthPage() {
  return <VersionedPage pageKey="admin-strategy-health" versions={versions} />;
}
