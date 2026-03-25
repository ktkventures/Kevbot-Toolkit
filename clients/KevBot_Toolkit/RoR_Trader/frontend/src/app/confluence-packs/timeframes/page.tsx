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
  if (isLoading) return <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>Loading timeframes...</div>;
  return <V5 />;
}

const versions = [
  {
    meta: {
      id: 'v1-initial',
      name: 'Initial Scaffold',
      description: 'Timeframe grid with enable/disable toggles, primary TF selection modal, and cross-TF confluence summary.',
      rationale: 'Mirrors the Streamlit timeframe configuration UI with interactive toggles and primary designation.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Streamlit Parity',
      description: 'Rich timeframe cards with usage stats, pack lists, cross-TF visual diagram, custom TF modal, and dependency warnings on disable.',
      rationale: 'Full feature parity with the Streamlit timeframe configuration — adds bar counts, pack usage detail, visual cross-TF confluence diagram, and custom timeframe creation.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'Compact horizontal rows: each TF shows label, toggle, and primary badge in a single row. Set Primary via click (no confirmation modal). No cross-TF explanation section, no custom TF button.',
      rationale: 'Stripped to essentials for experienced users. Each timeframe is a single compact row instead of a card. Set Primary is a direct click — no modal confirmation because it is a safe, reversible action. Removed the cross-TF explanation and custom TF features (noise for power users who already understand the system).',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Timeframe Explorer with visual hierarchy tree, session bar density visualization, cross-TF animated data flow diagram, TF recommendation engine by strategy type, and metrics strip with bar counts and volatility.',
      rationale: 'Creative experience that helps users understand timeframe relationships visually. The hierarchy tree shows primary/confluence/available groups. The bar density chart makes it intuitive how many bars each TF produces. The recommendation engine suggests optimal TF combos. The data flow diagram shows how confluence data feeds into the primary decision TF.',
    },
    component: V4,
  },
  {
    meta: {
      id: 'v5-production',
      name: 'Production (Use Case Grid)',
      description: 'Grid table with 17 timeframes × 4 use case columns (Strategy Primary, TF Confluence, Mass Builder, Chart Display). Checkbox toggles per cell. Default TF radio selector. Bars/day column. Provider support badges (All/Polygon/Stream). Sub-minute timeframes dimmed with badge. Summary cards showing enabled counts per use case.',
      rationale: 'Clean, functional grid that answers "where is each timeframe used?" at a glance. Toggle per use case rather than a single enable/disable. Default TF selection via radio buttons. Provider badges show data feed requirements.',
    },
    component: V5,
  },
  {
    meta: { id: 'v5-wired', name: 'Production (Live)', description: 'Timeframe config from API settings.', rationale: 'Persists to user settings.' },
    component: WiredV5,
  },
];

export default function TimeframesPage() {
  return <VersionedPage pageKey="timeframes" versions={versions} />;
}
