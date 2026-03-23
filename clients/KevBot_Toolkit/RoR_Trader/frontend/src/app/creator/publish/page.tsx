'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';

const versions = [
  {
    meta: {
      id: 'v1-simple',
      name: 'Simple Form',
      description: 'Choose type, select item, set price, publish. Two-column layout with published items list.',
      rationale: 'Minimal friction: type selector buttons, radio-style item picker, price input with fee preview, and one publish button. Right column shows already-published items with edit and unpublish options.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-flow',
      name: 'Full Publish Flow',
      description: 'Five-step wizard: Type, Select (with KPIs), Configure (title/description/tags/price/preview), Verification status, Review & Publish with terms acceptance.',
      rationale: 'Guided multi-step flow that walks creators through listing setup. Step 3 includes category and prop firm tags, description editor, and chart preview. Step 4 explains the verification tier system (Backtest Only / Forward Tested / Live Verified) and pricing implications. Step 5 shows a final review summary with terms acceptance checkbox.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'One card, three inputs (type pills, dropdown, price), publish button. Published items as compact rows.',
      rationale: 'Fastest path to publish. Type selector as small pills, item as a native dropdown, price as a number input. Fee preview inline. Published items listed with single-letter type badges and x button to unpublish.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative Wizard',
      description: 'Animated step indicator, visual type cards, slider pricing with live revenue projection, competitive analysis, listing preview mockup.',
      rationale: 'Publishing as an experience. Animated step dots scale and glow, type cards have colored shadows, price slider shows live revenue projections (monthly and annual with animated counters and green glow), competitive analysis shows how your price compares to similar items, and a collapsible listing preview shows how subscribers will see it.',
    },
    component: V4,
  },
];

export default function CreatorPublishPage() {
  return <VersionedPage pageKey="creator-publish" versions={versions} />;
}
