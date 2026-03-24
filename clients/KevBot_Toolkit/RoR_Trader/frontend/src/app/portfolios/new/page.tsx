'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';
import V5 from './versions/V5';

const versions = [
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
      rationale: 'Every Streamlit portfolio builder feature. Adds: risk scaling with explanation, position sizing calculation per strategy, strategy recommendations with P&L/DD/correlation analysis, risk summary card with daily exposure and capital utilization, chart legend mapping colors to strategies, drag-to-reorder.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'Single-column builder: compact settings row, inline search-to-add strategies, simple equity preview, dynamic KPI strip.',
      rationale: 'Removed two-column layout, recommendations engine, risk summary card, position sizing calculations, reorder controls, and drawdown chart. Focus on adding strategies with risk and seeing the combined result.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Wizard-style builder with 4-step flow (Setup, Strategies, Risk, Review), interactive strategy card picker with sparkline equity curves, live risk gauge, donut composition chart, smart defaults from risk tolerance, and what-if stress scenarios.',
      rationale: 'Guided creation experience that breaks portfolio building into digestible steps. Visual strategy picker replaces search/dropdown. Risk gauge and stress scenarios add portfolio-level insight before committing.',
    },
    component: V4,
  },
  {
    meta: {
      id: 'v5-production',
      name: 'Production (Phase 39 Aligned)',
      description: 'V2 Full Parity + webhook template selector (defaults to paper), capital utilization chart, worst case analysis, Monte Carlo risk simulation.',
      rationale: 'Adds the three missing risk analytics modules from Streamlit and the Phase 39 webhook template selector. All Streamlit portfolio builder features present.',
    },
    component: V5,
  },
];

export default function PortfolioNewPage() {
  return <VersionedPage pageKey="portfolio-new" versions={versions} />;
}
