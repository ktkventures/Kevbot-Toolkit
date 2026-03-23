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
      description: 'Theme switcher card with tiered theme grid and animated background previews.',
      rationale: 'Wraps the ThemeSwitcher component in a settings sub-page layout.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Parity',
      description: 'Theme grid with preview cards showing color swatches, current theme banner with animated preview background, tier sections with mini UI mockups, and active theme highlighting.',
      rationale: 'Full theme management page with visual preview of each theme\'s color palette, grouped by performance tier, with a prominent current-theme display.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'Simple radio list of all themes with instant switching. No previews, no tiers, no cards.',
      rationale: 'Minimal theme picker — one click to switch, active theme marked, nothing else. Fastest possible theme selection.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Visual theme showcase with live mini-dashboard previews rendered in each theme\'s colors, side-by-side comparison mode, and a custom theme builder concept with color pickers.',
      rationale: 'Each theme card shows a miniature dashboard (sidebar, metric cards, candlestick chart) rendered with that theme\'s exact colors. Compare any theme against the current one side-by-side before switching. Custom theme builder section previews what a user-defined palette would look like.',
    },
    component: V4,
  },
];

export default function SettingsThemesPage() {
  return <VersionedPage pageKey="settings-themes" versions={versions} />;
}
