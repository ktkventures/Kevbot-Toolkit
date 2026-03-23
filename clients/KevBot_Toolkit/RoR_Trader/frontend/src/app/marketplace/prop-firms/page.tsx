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
      description: 'Grid of prop firm cards with logo placeholder, name, evaluation price, profit target, max DD, daily loss limit. Sign Up affiliate button per firm. Click to view details with compatible portfolios.',
      rationale: 'First pass — establish the prop firm hub. Cards surface the key evaluation rules traders care about at a glance. Click-through to detail view shows compatible portfolios. Sign Up button with affiliate link is the monetization hook.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Parity',
      description: 'Full firm cards with all rules, account sizes, compatible portfolios with metrics. Comparison table tab for side-by-side. Account calculator tab with Daily R projections and evaluation timeline.',
      rationale: 'Everything a trader needs to choose a firm and understand what it takes to pass. Firm cards show compatible portfolios with their stats — instant credibility. Comparison table lets users see all firms side-by-side on every dimension. Account calculator answers the key question: "how long will it take me to pass this evaluation?" with projected timelines based on account size and risk.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'Comparison table only — firms as columns, rules as rows. Sign Up link per firm. Compatible portfolio count per firm shown below.',
      rationale: 'Pure data. One table, all firms, all rules, instant comparison. No clicking around, no detail views. A trader who already knows the landscape can pick their firm in 10 seconds. Compatible portfolio counts below give the final decision factor.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Interactive firm comparison tool with select-to-compare. Evaluation simulator (enter strategy stats, see pass/fail per firm). Funding calculator with profit projections. Success rate gauges per firm.',
      rationale: 'Reimagined as a decision tool. Select two firms to compare side-by-side with visual VS layout. Evaluation simulator takes your real strategy stats and tells you which firms you would pass. Funding calculator projects earnings across time horizons. Success rate gauges show platform-wide pass rates. The goal: help a trader go from "which firm?" to "signed up" in one page.',
    },
    component: V4,
  },
];

export default function PropFirmsPage() {
  return <VersionedPage pageKey="prop-firms" versions={versions} />;
}
