'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';

const versions = [
  {
    meta: {
      id: 'v1-production',
      name: 'Production',
      description: '6 take profit templates (R:R, ATR, Fixed $, %, Swing, None), locked params, nested variations, draft mode, search, Behavior tab.',
      rationale: 'Mirrors the stop loss pack architecture. Take profit methods are saved as reusable packs with locked parameters for mass backtesting. R:R is the most common (requires a stop loss to calculate risk). "None" template for strategies that exit only via signals or bar count.',
    },
    component: V1,
  },
];

export default function TakeProfitPage() {
  return <VersionedPage pageKey="take-profit" versions={versions} />;
}
