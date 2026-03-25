'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';
import V5 from './versions/V5';
import { useSettings, useSaveSettings } from '@/hooks/queries/useSettings';

function WiredV5() {
  const { data: settings, isLoading } = useSettings();
  const saveMutation = useSaveSettings();
  if (isLoading) return <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>Loading settings...</div>;
  // V5 will receive settings + onSave when props are added
  return <V5 />;
}

const versions = [
  {
    meta: {
      id: 'v1-initial',
      name: 'Initial Scaffold',
      description: 'Display preferences form with timezone and date format selectors.',
      rationale: 'Basic settings form layout for user display customization options.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Parity',
      description: 'Complete display settings: timezone, date format, number format (1,000 vs 1.000), currency symbol, chart defaults (candle style, timeframe, grid lines), and table density selector with live preview.',
      rationale: 'All display preferences in organized sections with a table density preview showing how compact/comfortable/spacious affects row height.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'Just timezone and date format with inline selectors on a single card.',
      rationale: 'Bare minimum display settings — two dropdowns, no save button, no extra options. Instant and minimal.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Visual preference builder with two-column layout: controls on the left, live preview on the right showing how dates, prices, P&L, and tables render with current settings.',
      rationale: 'Every setting change instantly updates the live preview panel — formatted dates, currency values, number separators, and a sample table with the selected density. See exactly how your data will look before saving.',
    },
    component: V4,
  },
  {
    meta: {
      id: 'v5-tabbed',
      name: 'Tabbed',
      description: 'V4 reorganized with tab navigation: Charts, Formatting, Components, Tables. Each tab pairs its controls with a contextual preview.',
      rationale: 'Addresses V4 feeling disjointed — now when you\'re adjusting chart settings, only the chart preview shows. When adjusting exec type styling, only the component preview shows. Same settings, better focus. Added chart library selector, candle color themes, and richer trade history table preview.',
    },
    component: V5,
  },
  {
    meta: { id: 'v5-wired', name: 'Production (Live)', description: 'Display settings from API.', rationale: 'Persists to Supabase.' },
    component: WiredV5,
  },
];

export default function SettingsDisplayPage() {
  return <VersionedPage pageKey="settings-display" versions={versions} />;
}
