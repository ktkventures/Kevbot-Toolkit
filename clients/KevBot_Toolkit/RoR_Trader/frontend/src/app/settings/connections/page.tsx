'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';
import V5 from './versions/V5';
import { useMonitorStatus, useEngineState } from '@/hooks/queries/useAlerts';
import { useStartMonitor, useStopMonitor } from '@/hooks/mutations/useAlertMutations';

function WiredV5() {
  const { data: status } = useMonitorStatus();
  const { data: engineState } = useEngineState();
  const startMutation = useStartMonitor();
  const stopMutation = useStopMonitor();
  // V5 will receive engine status + control callbacks when props are added
  return <V5 />;
}

const versions = [
  {
    meta: {
      id: 'v1-initial',
      name: 'Initial Scaffold',
      description: 'Connection status card showing market data provider and broker with connection indicators.',
      rationale: 'Displays Polygon.io and Alpaca integration status in a clean settings layout.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Parity',
      description: 'Full connection management: Polygon.io and Alpaca with API key display, plan tier, rate limits, plus Supabase and Railway status. Health check button, connect/test actions per service.',
      rationale: 'Complete infrastructure overview with masked API keys, plan details, rate limit info, and per-service action buttons for testing and updating connections.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'Status dots only — green/red per service with an "All systems operational" summary line.',
      rationale: 'Absolute minimum connection status. Four services, four dots, one summary. No API keys, no actions, no details.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Network topology diagram showing RoR Trader hub connected to all services via SVG lines with latency labels. Click a node for detailed stats, connection history timeline bars, and auto-diagnosis terminal output on failures.',
      rationale: 'Visual-first connection management. SVG network map with the app at center and services as clickable nodes. Each connection line shows latency. Node detail panel has uptime stats, hourly health bars, and a terminal-style auto-diagnosis feature for disconnected services.',
    },
    component: V4,
  },
  {
    meta: {
      id: 'v5-production',
      name: 'Production (Engine Status)',
      description: 'V4 topology diagram + Alert Engine (Ralph) admin module with status, uptime, ticks, symbols, active strategies, restart/disable controls, and monitoring logs terminal.',
      rationale: 'Moves engine admin controls from Alerts & Signals to Settings > Connections where admin/infrastructure concerns belong. Regular users see engine status on the Alerts page but admin controls live here.',
    },
    component: V5,
  },
  {
    meta: { id: 'v5-wired', name: 'Production (Live)', description: 'Live engine status + controls.', rationale: 'Polls engine status from API.' },
    component: WiredV5,
  },
];

export default function SettingsConnectionsPage() {
  return <VersionedPage pageKey="settings-connections" versions={versions} />;
}
