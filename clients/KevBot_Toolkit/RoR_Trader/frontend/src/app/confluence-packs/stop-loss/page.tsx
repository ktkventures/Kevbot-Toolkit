'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';

const versions = [
  {
    meta: {
      id: 'v1-production',
      name: 'Production',
      description: '6 stop loss templates (ATR, Fixed $, %, Swing, ATR Trailing, Breakeven), locked params, nested variations, draft mode, search, Behavior tab.',
      rationale: 'Follows the same pack architecture as TF Confluence and General Packs. Stop loss methods are saved as reusable packs with locked parameters so the mass backtester can iterate through multiple stop variations. Templates cover volatility-based (ATR), fixed (dollar/percentage), structural (swing), and trailing (ATR trailing, breakeven) approaches.',
    },
    component: V1,
  },
];

export default function StopLossPage() {
  return <VersionedPage pageKey="stop-loss" versions={versions} />;
}
