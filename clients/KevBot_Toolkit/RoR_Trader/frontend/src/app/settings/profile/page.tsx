'use client';

import PageSwitch from '@/components/PageSwitch';
import SettingsProfilePage from '@/views/SettingsProfilePage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';

const designVersions = [
  { meta: { id: 'v1', name: 'V1 Plan & Badge', description: 'Current plan display, role badge, upgrade button.', rationale: '' }, component: V1 },
  { meta: { id: 'v2', name: 'V2 Full Profile', description: 'Tabbed profile: Profile, Plan & Features, Achievements, Security.', rationale: '' }, component: V2 },
  { meta: { id: 'v3', name: 'V3 Compact', description: 'Single card with avatar, plan badge, key stats, and action buttons.', rationale: '' }, component: V3 },
  { meta: { id: 'v4', name: 'V4 Creative', description: 'Visual profile card with gradient header, level/XP, gamification badges, activity timeline.', rationale: '' }, component: V4 },
];

export default function Page() {
  return <PageSwitch live={SettingsProfilePage} versions={designVersions} pageKey="settings-profile" />;
}
