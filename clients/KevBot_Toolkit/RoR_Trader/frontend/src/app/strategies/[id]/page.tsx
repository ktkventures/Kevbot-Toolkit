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
      description: '9-tab detail view with equity curve, KPI metrics, price/live charts, trade history, confluence analysis, configuration, alerts, and alert analysis.',
      rationale: 'First pass covering all strategy detail tabs with placeholder content for charts and tables.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Parity',
      description: 'All 9 tabs fully populated: KPIs, extended analysis, charts, trade history, confluence analysis, config, alerts.',
      rationale: 'Complete Streamlit strategy detail. Every tab has real content: primary + secondary KPIs, backtest vs forward equity curves, sortable trade history with exec type badges, confluence state analysis, full read-only config display, alert configuration panel, and alert accuracy analysis.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: '3 tabs instead of 9: Overview (KPIs + equity + chart + config), Trades (history + confluence/ToD/DoW analysis as collapsibles), Monitoring (alerts + live chart + accuracy). Inline KPI strip in header.',
      rationale: 'Answers the 3 trader questions: "Is this strategy good?" (Overview), "What are the trades doing?" (Trades), "Is it running correctly?" (Monitoring). Extended KPIs, price chart, and config merged into Overview. Confluence analysis, time-of-day, and day-of-week analysis collapsed by default under Trades. Alert config and accuracy merged into Monitoring.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Visual analytics dashboard with health score ring, strategy DNA radar, trade timeline, DoW x ToD heatmap, equity curve with market regime overlay, risk/reward scatter, confluence effectiveness bars, live position widget, and AI-style smart summary.',
      rationale: 'The "wow factor" version. Circular health score (0-100) with segmented ring for WR/DD/consistency/risk. Interactive trade timeline with hover tooltips. Performance heatmap reveals best trading windows. Equity curve with bull/bear/neutral regime bands. Strategy DNA radar chart shows character fingerprint. Live position widget with animated status and risk bar. Confluence effectiveness bars rank condition contributions. Risk/reward scatter plots trade distribution. Smart summary provides actionable AI-style narrative.',
    },
    component: V4,
  },
  {
    meta: {
      id: 'v5-production',
      name: 'Production',
      description: '6-tab detail view: Equity & KPIs (comparison modes, extended KPIs, advanced analysis), Chart & Trades (merged), Confluence Analysis (per-group), Configuration (pack-aware), Alerts (event types + mapping), Alert Analysis (discrepancies, timing, trade-by-trade).',
      rationale: 'Forward test and alerts always on. KPI comparison modes (Overall/BT vs FWD/FWD vs Alerts/BT vs Alerts) with Daily ROI and TPD. 3-segment equity curve per display settings V5. Pack-aware variable display matching My Strategies V5. Advanced analysis (Rolling Metrics, Return Distribution, Markov Motor). Full alert analysis with discrepancies, position health, trigger timing, and trade-by-trade slippage.',
    },
    component: V5,
  },
];

export default function StrategyDetailPage() {
  return <VersionedPage pageKey="strategy-detail" versions={versions} />;
}
