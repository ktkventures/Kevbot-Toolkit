'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V5 from './versions/V5';

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
  {
    meta: {
      id: 'v5-refined',
      name: 'Refined (My Strategies Style)',
      description: 'Portfolio cards matching My Strategies V5 style: pulsing enabled dot, status badge, final balance, meta line (strategies/balance/scaling/risk/trades per day/webhook template), strategy pills, 3-segment equity curve with HWM/Edge Check/Confidence Bands, 6-KPI row or comparison table (Overall/BT vs FWD/FWD vs Alerts/BT vs Alerts), requirement set compliance badge, action row with View/Edit/Clone/Delete + enabled toggle + bulk select checkbox.',
      rationale: 'Consistent visual language across strategies and portfolios. Same equity curve controls, KPI comparison modes, bulk actions, and card structure. Portfolio-specific KPIs: P&L ($), Win Rate, PF, Max DD (%), Avg Daily P&L ($), Trades. Enabled toggle replaces alert tracking toggle.',
    },
    component: V5,
  },
];

export default function PortfoliosPage() {
  return <VersionedPage pageKey="portfolios" versions={versions} />;
}
