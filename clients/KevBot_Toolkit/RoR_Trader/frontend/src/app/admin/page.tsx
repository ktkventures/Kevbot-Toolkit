'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';

const versions = [
  {
    meta: {
      id: 'v1-metric-cards',
      name: 'Metric Cards',
      description: 'Four key MetricCards: Total Users, Monthly Revenue, Active Portfolios, Alert Volume.',
      rationale: 'Minimal admin overview. Four numbers that answer "is the platform healthy?" at a glance. No charts, no tables, just the essentials.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-dashboard',
      name: 'Full Dashboard',
      description: 'Tabbed dashboard with KPIs, revenue/growth charts, top creators leaderboard, activity feed, system health.',
      rationale: 'Complete admin command center. Tabs for Overview, Revenue, Content, and System. Overview shows grouped KPIs (users, revenue, content), charts for MRR and user growth, top creators leaderboard with earnings, and real-time activity feed. System tab covers uptime, webhook delivery, and error rates.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-compact',
      name: 'Compact',
      description: 'Dense KPI grid plus chronological activity feed.',
      rationale: 'Two rows of 4 MetricCards covering all key stats, plus a compact activity feed. Everything visible without scrolling on a desktop. No tabs, no charts — just data density.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Real-time dashboard with animated counters, geographic distribution bars, alert volume heatmap, platform pulse indicators.',
      rationale: 'Dashboard that feels alive. Animated counters tick up on page load. Geographic user distribution with progress bars. Alert volume heatmap shows trading patterns by hour and day. Platform pulse shows system health with glowing status indicators. Live badge pulses through colors.',
    },
    component: V4,
  },
];

export default function AdminPage() {
  return <VersionedPage pageKey="admin" versions={versions} />;
}
