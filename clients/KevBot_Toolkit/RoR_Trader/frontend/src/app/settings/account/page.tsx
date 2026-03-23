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
      description: 'Account info card with email, membership date, and sign out button.',
      rationale: 'Minimal account overview for the settings sub-page.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Parity',
      description: 'Complete account management: profile info, subscription tier, usage statistics (strategies, backtests, alerts), API usage bars with limits, password change form, 2FA toggle, and data export.',
      rationale: 'Full account page with usage metrics, security settings (password change with form reveal, 2FA enable/disable), API consumption bars, and data export button.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'Email display and Sign Out button. That is it.',
      rationale: 'Absolute minimum account page — just shows who you are and lets you leave. No stats, no security, no extras.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'User profile card with avatar, GitHub-style activity heatmap (26 weeks), achievement badges with icons (earned/locked), and usage analytics with bar charts for backtests and horizontal bars for alert types.',
      rationale: 'Gamified account page. Large avatar with initials, activity heatmap showing daily engagement intensity, collectible achievement badges (First Strategy, 100 Backtests, Night Trader, etc.), and visual usage analytics with mini charts.',
    },
    component: V4,
  },
];

export default function SettingsAccountPage() {
  return <VersionedPage pageKey="settings-account" versions={versions} />;
}
