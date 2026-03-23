'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';
import V5 from './versions/V5';
import V6 from './versions/V6';
import V7 from './versions/V7';

const versions = [
  {
    meta: {
      id: 'v1-initial',
      name: 'Initial Scaffold',
      description: 'Summary metrics row, portfolio equity curve and daily P&L chart placeholders, recent alerts list.',
      rationale: 'First pass establishing the dashboard overview layout with key portfolio-level metrics and chart slots.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Parity',
      description: 'Full dashboard: overview KPIs, top strategy showcase, quick actions, activity feed, market status.',
      rationale: 'Complete Streamlit dashboard with all sections: overview metrics, best strategy highlight with mini equity curve, quick action grid for navigation, chronological activity feed (alerts, saves, monitor events), and market session/data status. Includes empty state with onboarding CTA when no strategies exist.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'At-a-glance dashboard: 4 KPIs, best strategy + portfolio health cards, compact activity feed.',
      rationale: 'Everything on one screen, no scrolling. Removed quick actions grid (sidebar handles navigation), market overview (sidebar footer), system status section, equity curves, and empty state tutorial. Just zeros when no data.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Trading cockpit: daily briefing, strategy leaderboard with sparklines, P&L calendar heatmap, portfolio pulse heartbeat, active positions tracker, market regime gauge, monthly goal progress, notification center.',
      rationale: 'The wow-factor Bloomberg-style trading cockpit. Personalized daily briefing card summarizes the day. Strategy leaderboard ranks all strategies with trend sparklines and live performance. P&L calendar heatmap shows every trading day color-coded by profit/loss. Portfolio pulse shows animated equity heartbeat. Active positions widget tracks live positions with unrealized P&L. Market regime gauge synthesizes VIX and market breadth. Monthly goal tracker with progress bar. Notification center with bell icon and unread count.',
    },
    component: V4,
  },
  {
    meta: {
      id: 'v5-hybrid',
      name: 'Hybrid Cockpit',
      description: 'V1 chart focus + V4 creative widgets: equity curve, daily P&L, active positions with match status, monthly goal, market regime, P&L calendar, issues panel, customizable layout.',
      rationale: 'Combines V1\'s clean chart-centric layout (equity curve + daily P&L as hero content) with V4\'s best widgets: active positions with close-early and match status, market regime with VIX, monthly goal tracker, P&L calendar heatmap. Added: issues/warnings panel for anomaly detection, position match status (backtest vs live), close-early button, and customizable widget toggle.',
    },
    component: V5,
  },
  {
    meta: {
      id: 'v6-refined',
      name: 'Refined Cockpit',
      description: 'V5 with layout cleanup: charts + monthly goal in left 2/3, positions + widgets in right 1/3, portfolio filter, customizable KPIs.',
      rationale: 'Addresses V5 feedback: moved monthly goal under the equity/P&L charts for visual consistency. Added portfolio multi-select filter to exclude test portfolios. Made KPI strip customizable — users choose which 4-6 metrics matter to them. System status moved to header as a compact indicator. Right column focuses on real-time info: positions, regime, calendar, issues, activity.',
    },
    component: V6,
  },
  {
    meta: {
      id: 'v7-detailed',
      name: 'Detailed Cockpit',
      description: 'V6 with detail modals on every widget: equity breakdown, P&L analysis, position context, health deep-dive, market analysis, goal tracking, full activity log.',
      rationale: 'Every widget now has a detail button that opens a rich modal with deeper context, actionable insights, and links to related pages. Traders can drill into any metric without leaving the dashboard — then navigate to the full page when they need to take action.',
    },
    component: V7,
  },
];

export default function DashboardPage() {
  return <VersionedPage pageKey="dashboard" versions={versions} />;
}
