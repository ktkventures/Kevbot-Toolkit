'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';

const versions = [
  {
    meta: {
      id: 'v1-plan-badge',
      name: 'Plan & Badge',
      description: 'Current plan display, role badge, upgrade button.',
      rationale: 'Simplest profile view. Avatar, name, role badge, current plan details, and a single upgrade CTA. Quick snapshot of where you stand.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-profile',
      name: 'Full Profile',
      description: 'Tabbed profile: Profile (avatar, bio, socials), Plan & Features (access list, upgrade CTA), Achievements (badge grid), Security (2FA, sessions, password).',
      rationale: 'Complete account management. Profile tab shows avatar, usage stats, and public creator settings (bio, social links). Plan tab shows feature access list with checkmarks and upgrade comparison. Achievements tab displays earned/locked badges. Security tab covers 2FA, active sessions, and password change.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-compact',
      name: 'Compact',
      description: 'Single card with avatar, plan badge, key stats, and action buttons.',
      rationale: 'Everything on one screen without tabs. Avatar and role badge at top, key account details in a simple list (plan, member since, subscriptions, alerts), upgrade and edit buttons at bottom.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Visual profile card with gradient header, level/XP progress bar, gamification badges with rarity tiers, activity timeline with color-coded events.',
      rationale: 'Gamified profile experience. Profile card has a gradient decorative header with glowing avatar. Level and XP system with animated progress bar. Badges grid with rarity tiers (Common to Legendary) and hover tooltips. Activity timeline with color-coded event dots. Gradient upgrade CTA at bottom.',
    },
    component: V4,
  },
];

export default function SettingsProfilePage() {
  return <VersionedPage pageKey="settings-profile" versions={versions} />;
}
