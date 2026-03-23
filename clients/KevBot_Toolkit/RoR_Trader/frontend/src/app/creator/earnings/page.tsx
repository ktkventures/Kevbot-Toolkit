'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';

const versions = [
  {
    meta: {
      id: 'v1-basic',
      name: 'Basic',
      description: 'Monthly earnings summary table with gross/fee/net columns, payout history table with status badges.',
      rationale: 'Straightforward earnings view — MetricCards for total/pending/next/fees, then two tables: monthly breakdown and payout history. No charts, no frills, just the numbers.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full',
      name: 'Full Earnings',
      description: 'Revenue donut chart, per-item breakdown, revenue share chain details, payout settings, and tax documents.',
      rationale: 'Five-tab layout covering every earnings concern: Revenue tab shows donut chart and monthly table, Revenue Shares shows the creator chain split per portfolio, Payouts lists history, Settings configures method/threshold/frequency, and Tax Documents has 1099 download stubs.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-compact',
      name: 'Compact',
      description: 'KPI strip, earnings table, inline payout list — everything on one page, no tabs.',
      rationale: 'Dense, scannable layout. Horizontal KPI strip at top, earnings table with just the essentials, payout list as rounded rows instead of a table. Ideal for quick reference.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Animated money flow showing revenue chain, earnings forecast projections, milestone badges with progress bars, pulsing counters.',
      rationale: 'Revenue as a journey. Animated flow diagram shows how subscriber payments split through the creator chain. Forecast panel projects future earnings at current growth rates. Milestone system with shimmer badges tracks achievements like "$5K Earned" and "500 Subscribers."',
    },
    component: V4,
  },
];

export default function CreatorEarningsPage() {
  return <VersionedPage pageKey="creator-earnings" versions={versions} />;
}
