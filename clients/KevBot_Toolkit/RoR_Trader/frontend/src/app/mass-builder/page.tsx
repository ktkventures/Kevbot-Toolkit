'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';
import V5 from './versions/V5';
import V6 from './versions/V6';

const versions = [
  {
    meta: {
      id: 'v1-initial',
      name: 'Initial Scaffold',
      description: 'Two-panel layout with search configuration sidebar and results area placeholder.',
      rationale: 'Establishes the split-pane structure for bulk strategy discovery and optimization.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Streamlit Parity',
      description: 'Complete mass builder with 9-tab config panel (tickers, TFs, direction, entry/exit triggers, TF/general confluence, stop loss, take profit), combination estimator, progress simulation, and ranked result cards with equity curves, KPIs, save/pass actions.',
      rationale: 'Full feature parity with Streamlit render_mass_strategy_builder — all configuration tabs, preset tickers, required performance filters, post-analysis sorting, and interactive result cards with mini equity curves.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'Two-column layout: left column has compact config form with ticker chips, TF chips, direction toggle, trigger multiselect, and confluence multiselect with inline min criteria row. Right column shows a sortable results table with rank, ticker, TF, trigger, KPIs, and save button. No progress bar, no search name, no stop/target config.',
      rationale: 'Everything on one screen — config and results side by side. Table view replaces cards for density. Only essential configuration exposed, uses defaults for stop/target.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Strategy Discovery Lab with visual config cards, animated progress ticker showing current evaluation, interactive scatter plot (x=WR, y=PF, size=trades), ticker/TF heatmap, AI-style insight cards, top performer spotlight, and mini equity sparklines per result.',
      rationale: 'Transforms bulk strategy discovery into an immersive lab experience. The scatter plot makes it instant to spot quality strategies in the "sweet zone". The animated ticker during analysis creates engagement. The heatmap reveals which ticker+TF combos produce the best results. Insight cards surface key findings automatically.',
    },
    component: V4,
  },
  {
    meta: {
      id: 'v5-production',
      name: 'Production',
      description: 'V2 Full Parity refined: updated exec types ([C], [L], [LC], [CC]), stop loss and take profit as pack selectors (matching Strategy Builder V5), Trade Qualification filter in post-analysis filters, consistent styling.',
      rationale: 'Aligns Mass Builder with recent design decisions: pack-based stop/target selection, correct execution type naming, TQ filter for prop firm compliance preview on bulk results.',
    },
    component: V5,
  },
  {
    meta: {
      id: 'v6-strategy-cards',
      name: 'Strategy-Style Cards',
      description: 'V5 config panel + My Strategies-style result cards: 1/2/3 column toggle, equity curve with chart height + HWM controls, pack-aware variable display (entry/exit/stop/target/confluence badges), search name in strategy title. Same KPIs but in My Strategies layout.',
      rationale: 'Result cards should feel familiar — same card structure as My Strategies. Column toggle lets users choose density. Equity curve replaces the small sparkline. Search name flows into strategy name for identification after saving.',
    },
    component: V6,
  },
];

export default function MassBuilderPage() {
  return <VersionedPage pageKey="mass-builder" versions={versions} />;
}
