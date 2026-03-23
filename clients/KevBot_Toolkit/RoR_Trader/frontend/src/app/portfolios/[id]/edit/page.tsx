'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';

const versions = [
  {
    meta: {
      id: 'v1-initial',
      name: 'Initial Scaffold',
      description: 'Portfolio edit form pre-filled with existing data, combined equity curve with 4 strategy lines, strategy management, recommendations, reset changes and update actions.',
      rationale: 'First pass mirroring the new-portfolio form but pre-populated with existing portfolio data and including reset functionality.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Parity',
      description: 'Edit mode with change tracking, pre-filled data, reset changes, position sizing calculations.',
      rationale: 'Same layout as New Portfolio V2 but with pre-filled mock data, change tracking indicators, Reset Changes button, and Update Portfolio action. Shows risk scaling, recommendations, and risk summary.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'Single-column edit with change tracking, inline search-to-add, reset changes, compact equity preview.',
      rationale: 'Same streamlined layout as New V3 with pre-filled data, unsaved changes indicator, and Reset Changes button. Removed recommendations, risk summary, position sizing, reorder, and drawdown chart.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Wizard-style editor with 4-step flow, pre-filled from existing portfolio. Change indicators on wizard steps and fields (orange highlights), NEW badges on added strategies, per-field change tracking, reset changes, and visual risk gauge with stress scenarios.',
      rationale: 'Same guided wizard as New V4 but adapted for editing: pre-populated state, change tracking with per-step orange dots, field-level change labels, NEW/removed strategy badges, and update-only-when-changed validation.',
    },
    component: V4,
  },
];

export default function PortfolioEditPage() {
  return <VersionedPage pageKey="portfolio-edit" versions={versions} />;
}
