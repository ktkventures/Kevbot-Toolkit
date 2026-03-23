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
      description: 'Search bar, 3-tab filter (Portfolios/Strategies/Packs), grid of cards with name, creator, price, rating stars, subscriber count. Category filter pills.',
      rationale: 'First pass to establish the marketplace layout. Cards show the essential info a subscriber needs at a glance: what it is, who made it, what it costs, and how popular it is. Category pills let users narrow down quickly without overwhelming filter UIs.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Parity',
      description: 'Verification badges (gray/blue/gold), performance metrics per card (WR, PF, Daily R, Max DD), mini equity curves, Free Rotation section, advanced filters, sort options, prop firm compatibility tags.',
      rationale: 'Everything a subscriber needs to make an informed decision. Verification badges build trust. Performance metrics let users compare quantitatively. Mini equity curves give an instant visual of trajectory. Free Rotation section drives adoption. Advanced filters (min WR, price range, prop firm) let serious traders find exactly what they need. Creator profile links start building the community layer.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'Compact list/table view with all items in one sortable list. Verification dots, inline metrics, quick search + sort dropdown. No featured section, no categories.',
      rationale: 'For power users who want to scan fast. The table format shows everything in one row — verification status, key metrics, price — without the visual overhead of cards. Sort by any column. A trader can evaluate 20 items in the time it takes to scroll through 6 cards.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Hero showcase of top 3, Rising Stars section, performance leaderboard table, interactive risk-vs-return scatter plot, creator spotlight cards, personalized recommendations, category browsing with icons.',
      rationale: 'Reimagined as a discovery experience. The hero section draws attention to proven winners. Rising Stars highlights new talent. The scatter plot lets users visually compare risk/return tradeoffs. Creator spotlights build community. Recommendations use portfolio stats to suggest good fits. Category icons make browsing feel intuitive rather than transactional.',
    },
    component: V4,
  },
];

export default function MarketplacePage() {
  return <VersionedPage pageKey="marketplace" versions={versions} />;
}
