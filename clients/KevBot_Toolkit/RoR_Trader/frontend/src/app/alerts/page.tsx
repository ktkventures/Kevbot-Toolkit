'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';
import V5 from './versions/V5';

const versions = [
  {
    meta: {
      id: 'v1-initial',
      name: 'Initial Scaffold',
      description: 'Engine status card, monitored strategies list with position state badges, and recent alerts feed with entry/exit type tags.',
      rationale: 'First pass — focused on the three core sections: engine health, active strategy monitors, and chronological alert stream.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Parity',
      description: 'Complete alert system: monitor status with metrics, per-strategy config modal, filtered alert feed, open positions table with health indicators, and searchable history with export.',
      rationale: 'Matches Streamlit Alerts page — engine toggle, strategy-level alert/webhook config, position tracking, and tabbed alert management.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'Two-panel layout: left has monitor toggle + strategy toggles, right has position card + one-line alert feed.',
      rationale: 'No tabs, no config modal, no separate history or positions pages. All controls inline: toggle switches for entry/exit/webhook per strategy. Alert feed uses compact one-line format with color-coded ENTRY/EXIT. Open positions shown as small card above the feed.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Alert command center: signal radar, live position cards with stop-to-target bar, strategy signal matrix, alert frequency sparkline, terminal-style feed with animations, and alert statistics dashboard.',
      rationale: 'Visual-first alert management. Dense cockpit layout with animated SVG radar showing symbol signal directions, position cards with real-time R tracking and stop/target progress, terminal-style feed with scanline and flash effects, and strategy config sidebar. No tabs — everything visible at once.',
    },
    component: V4,
  },
  {
    meta: {
      id: 'v5-production',
      name: 'Production (Phase 39)',
      description: '4-tab alert management: Strategy Alerts (filtered feed + entry/exit pair history with webhook status), Portfolio Alerts (per-portfolio with matched/phantom status), Outbound Webhooks (delivery log with payload inspection, success metrics), Inbound Webhooks (endpoint reference). Engine status strip (non-admin). Admin controls moved to Settings > Connections.',
      rationale: 'Focuses on what users care about: what alerts fired, what webhooks were sent, and whether they were delivered correctly. Admin engine controls relocated to Connections page. Entry/exit pair history shows the full lifecycle of each trade with webhook delivery status for validation.',
    },
    component: V5,
  },
];

export default function AlertsPage() {
  return <VersionedPage pageKey="alerts" versions={versions} />;
}
