'use client';

import PageSwitch from '@/components/PageSwitch';
import SettingsAccountPage from '@/views/SettingsAccountPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';

const designVersions = [
  { meta: { id: 'v1', name: 'V1 Initial', description: 'Account info card with email, membership date, and sign out button.', rationale: '' }, component: V1 },
  { meta: { id: 'v2', name: 'V2 Full Parity', description: 'Complete account management: profile info, subscription tier, usage statistics, API usage bars, password change, 2FA, data export.', rationale: '' }, component: V2 },
  { meta: { id: 'v3', name: 'V3 Streamlined', description: 'Email display and Sign Out button.', rationale: '' }, component: V3 },
  { meta: { id: 'v4', name: 'V4 Creative', description: 'User profile card with avatar, activity heatmap, achievement badges, usage analytics.', rationale: '' }, component: V4 },
];

export default function Page() {
  return <PageSwitch live={SettingsAccountPage} versions={designVersions} pageKey="settings-account" />;
}
