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
      description: 'Strategy list with status badges, mini equity curve placeholders, KPI rows, ticker/direction/tag filters, and view/edit actions.',
      rationale: 'First pass focused on card-based strategy list with forward-test status indicators and quick-access KPIs.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Parity',
      description: 'All Streamlit features: tags, bulk actions, forward test KPIs, status filters, multi-select mode.',
      rationale: 'Direct port of every My Strategies feature. Added: tag pills on each card, bulk action bar with multi-select, forward test KPI row per strategy, full status/ticker/direction/tag filtering, duplicate and delete actions. Strategy cards show both backtest and forward test metrics side by side.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'Compact table view with sortable columns, text search, status filter, and inline icon actions.',
      rationale: 'Maximum information density: table layout replaces cards, sortable columns replace dropdown sort, inline icon buttons replace action bars. Removed tags, bulk actions, equity curves, forward test KPI rows, and multi-select to minimize clicks and visual noise.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Portfolio command center with overview strip, live status animations, SVG sparklines, mini DNA radars, comparison mode with overlay radar chart and KPI diff table, and visual search.',
      rationale: 'The "wow factor" version. Portfolio overview strip aggregates total R, avg WR/PF, best/worst performers across all strategies. Cards feature real SVG equity sparklines with animated draw, mini DNA radar charts showing strategy character, and pulsing live dots for monitored strategies. Comparison mode lets traders select 2 strategies for side-by-side radar overlay and KPI delta table. 6 strategies with richer mock data including R-squared and total R.',
    },
    component: V4,
  },
  {
    meta: {
      id: 'v5-refined',
      name: 'Refined',
      description: 'Inline checkbox on each card action row, no select mode toggle, bulk actions bar with Delete Selected + Create Portfolio + Update Portfolio.',
      rationale: 'Removes the janky select mode toggle. Checkbox is always available on the action button row of each card. Checking any card surfaces the bulk action bar with Delete Selected, Create Portfolio, Update Portfolio, Select All, and Clear. Cleaner workflow for bulk operations.',
    },
    component: V5,
  },
];

export default function StrategiesPage() {
  return <VersionedPage pageKey="strategies" versions={versions} />;
}
