'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';

const versions = [
  {
    meta: {
      id: 'v1-basic',
      name: 'Basic',
      description: 'MetricCards for key stats, sortable published items list with subscriber counts and trend indicators.',
      rationale: 'Simple command center — four KPIs at the top, items sorted by subscribers/revenue/rating. Gets you the essentials without any noise.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full',
      name: 'Full Dashboard',
      description: 'Revenue and subscriber growth SVG charts, per-item performance table, top performer spotlight, activity feed, payout history, revenue breakdown by type.',
      rationale: 'Complete creator overview with tabs for Overview, Performance, and Payouts. Revenue charts show monthly trends, activity feed keeps you connected to subscriber events, and the revenue-by-type breakdown shows where your income comes from.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-compact',
      name: 'Compact',
      description: 'KPI strip, revenue sparkline, compact sortable table — single-page, no tabs.',
      rationale: 'Everything on one screen. Horizontal KPI strip replaces cards, inline sparklines show trends, sortable table with single-letter type badges. For creators who want data density over decoration.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Animated revenue counter, subscriber growth pulse, earnings heatmap calendar, item health gauges, AI insights panel.',
      rationale: 'Dashboard as a living system. Animated counters tick up on load, green pulse dot shows live status, heatmap calendar visualizes daily earnings patterns, circular health gauges score each item, and AI insights surface actionable observations like "Your VWAP Scalper gained 12 subscribers this week."',
    },
    component: V4,
  },
];

export default function CreatorDashboardPage() {
  return <VersionedPage pageKey="creator-dashboard" versions={versions} />;
}
