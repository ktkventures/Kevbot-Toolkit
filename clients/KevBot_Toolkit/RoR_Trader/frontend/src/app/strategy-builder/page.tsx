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
      description: 'Three-column layout with config inputs, entry trigger, confluence conditions on the left; price chart and equity curve with KPIs on the right.',
      rationale: 'First pass focused on establishing the core strategy builder layout with all major sections represented as placeholders.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Parity',
      description: 'Complete Streamlit feature set: trigger search, exit config, confluence scoring, advanced KPIs, trade history.',
      rationale: 'Includes every feature from the Streamlit Strategy Builder: searchable trigger picker grouped by pack, full exit strategy config (swing/fixed stops, TP, opposite signal, bar count), confluence scoring with weights, two-row KPI dashboard, trade history table, and advanced analysis section. Nothing removed from Streamlit.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'Single scrollable page: compact config bar, trigger chip selector, collapsed exit config, inline confluence chips, stacked KPIs + equity + price chart + trade history. No tabs, no analysis sidebar.',
      rationale: 'Removes build-time friction. Trigger/exit drill-down analysis moved to Strategy Detail (post-save). Confluence scoring, asset type selector, lookback mode, and Markov motor removed from builder. Config-to-results feedback loop is now: configure -> run -> scroll to see everything. Exit config shows a one-line summary, expandable on demand.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Visual pipeline builder with interactive node graph, Strategy DNA radar chart, confidence gauge, AI insight cards, risk calculator, interactive equity curve with hover tooltips, quick presets, and live indicator sparkline previews.',
      rationale: 'The wow-factor version. Strategy building visualized as an interactive data pipeline. Strategy DNA radar chart shows the character of the strategy (aggression, selectivity, speed, consistency, risk control, trend alignment). Confidence gauge scores the strategy based on trade count, win rate, profit factor, and confluence depth. Risk calculator projects daily/monthly returns at a given account size. AI insight card provides mock suggestions. Quick presets auto-fill entire strategies with one click.',
    },
    component: V4,
  },
  {
    meta: {
      id: 'v5-pack-integrated',
      name: 'Pack Integrated',
      description: 'Updated for new pack architecture: stop/target via pack selectors, triggers with [C]/[L]/[LC]/[CC] exec types, all 8 TF templates.',
      rationale: 'Based on V2 with key changes: stop loss and take profit are now selected from saved packs (not configured inline) — enabling mass backtester iteration. Trigger exec types updated to [C]/[L]/[LC]/[CC] naming. Exec badges use display settings V5 uniform blue. Links to Stop Loss and Take Profit pack management pages.',
    },
    component: V5,
  },
];

export default function StrategyBuilderPage() {
  return <VersionedPage pageKey="strategy-builder" versions={versions} />;
}
