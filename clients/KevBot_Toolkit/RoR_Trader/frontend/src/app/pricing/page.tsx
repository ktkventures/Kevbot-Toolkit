'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';

const versions = [
  {
    meta: {
      id: 'v1-tier-cards',
      name: '3-Column Tiers',
      description: 'Three tier cards (Free, Creator, Pro Creator) with feature bullet lists and CTA buttons.',
      rationale: 'Simple starting point with 3 clear tiers. Quick visual scan of what each plan offers. Clean layout, no extras.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-pricing',
      name: 'Full Pricing Page',
      description: '4 tiers with feature comparison matrix, billing toggle, FAQ section, and money-back guarantee.',
      rationale: 'Complete pricing page with all the details a potential customer needs: annual/monthly toggle with 20% discount, full feature comparison table, expandable FAQ covering revenue sharing, upgrades, and guarantees. Current Plan badge on active tier.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-compact',
      name: 'Compact',
      description: '4 tier cards in a row, 5 key features each, minimal design.',
      rationale: 'Stripped down to essentials. No comparison table, no FAQ, no toggle. Just the tiers, key features, and CTA. For users who want to decide quickly without scrolling through details.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Interactive tier explorer with hover effects, revenue calculator, testimonials, animated transitions.',
      rationale: 'Designed to sell. Hover effects highlight each tier with its signature color. Revenue calculator shows potential earnings for portfolio creators ("50 subscribers at $30/mo = $X after fees"). Testimonial cards add social proof. Everything animates smoothly to create a premium feel.',
    },
    component: V4,
  },
];

export default function PricingPage() {
  return <VersionedPage pageKey="pricing" versions={versions} />;
}
