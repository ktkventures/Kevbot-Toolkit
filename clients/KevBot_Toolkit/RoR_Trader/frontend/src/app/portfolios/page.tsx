'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';

const versions = [
  {
    meta: {
      id: 'v1-initial',
      name: 'Initial Scaffold',
      description: 'Two-column grid of portfolio cards with mini equity curves, P&L/DD/WR/strategy count KPIs, and active status badges.',
      rationale: 'First pass establishing the portfolio list with clickable cards that link to detail views.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Parity',
      description: 'Rich portfolio cards with deploy status, compliance checks, requirement set badges, and full KPIs.',
      rationale: 'Every Streamlit portfolio list feature: richer cards with live/paper/paused status, deployment indicator, requirement set badge, compliance check (passing/violations), full 6-KPI row, mini equity curves, and Clone/Delete actions alongside View/Edit.',
    },
    component: V2,
  },
];

export default function PortfoliosPage() {
  return <VersionedPage pageKey="portfolios" versions={versions} />;
}
