'use client';

import PageSwitch from '@/components/PageSwitch';
import PortfolioNewPage from '@/views/PortfolioNewPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';
import V5 from './versions/V5';

const designVersions = [
  {
    meta: {
      id: 'v1-initial',
      name: 'Initial Scaffold',
      description: 'Portfolio creation form with settings, live metrics, combined equity curve with per-strategy lines, strategy management panel, recommendations, and drawdown analysis.',
      rationale: 'First pass with interactive strategy builder including add/remove strategies, risk-per-trade editing, and portfolio recommendation engine.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Parity',
      description: 'Full builder: risk scaling, position sizing calcs, strategy recommendations, risk summary, chart legend.',
      rationale: 'Every Streamlit portfolio builder feature.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'Single-column builder: compact settings row, inline search-to-add strategies, simple equity preview, dynamic KPI strip.',
      rationale: 'Removed two-column layout, recommendations engine, risk summary card.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Wizard-style builder with 4-step flow (Setup, Strategies, Risk, Review).',
      rationale: 'Guided creation experience.',
    },
    component: V4,
  },
  {
    meta: {
      id: 'v5-production',
      name: 'Production (Design Ref)',
      description: 'V2 Full Parity + webhook template selector, capital utilization chart, worst case analysis, Monte Carlo.',
      rationale: 'Design reference with mock data.',
    },
    component: V5,
  },
];

export default function Page() {
  return <PageSwitch live={PortfolioNewPage} versions={designVersions} pageKey="portfolio-new" />;
}
