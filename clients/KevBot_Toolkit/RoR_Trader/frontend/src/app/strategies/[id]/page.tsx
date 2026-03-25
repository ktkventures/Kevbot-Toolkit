'use client';

import { useParams } from 'next/navigation';
import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';
import V5 from './versions/V5';
import { useStrategy } from '@/hooks/queries/useStrategies';

function WiredV5() {
  const params = useParams();
  const strategyId = params?.id ? Number(params.id) : null;
  const { data: strategy, isLoading } = useStrategy(strategyId);

  if (isLoading) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>
        Loading strategy...
      </div>
    );
  }

  return <V5 apiStrategy={strategy || undefined} />;
}

const versions = [
  { meta: { id: 'v1-initial', name: 'Initial Scaffold', description: '9-tab detail view.', rationale: 'First pass.' }, component: V1 },
  { meta: { id: 'v2-full-parity', name: 'Full Parity', description: 'All 9 tabs fully populated.', rationale: 'Complete port.' }, component: V2 },
  { meta: { id: 'v3-streamlined', name: 'Streamlined', description: '3 tabs.', rationale: 'Focused.' }, component: V3 },
  { meta: { id: 'v4-creative', name: 'Creative', description: 'Visual analytics dashboard.', rationale: 'Wow factor.' }, component: V4 },
  {
    meta: {
      id: 'v5-production',
      name: 'Production (Live)',
      description: '6-tab strategy detail with real API data — KPIs, charts, trades, alerts.',
      rationale: 'Production version fetching strategy data from FastAPI.',
    },
    component: WiredV5,
  },
];

export default function StrategyDetailPage() {
  return <VersionedPage pageKey="strategy-detail" versions={versions} />;
}
