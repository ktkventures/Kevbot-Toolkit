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
      description: 'Simple card with description of saved mass search results browsing.',
      rationale: 'Placeholder page for future mass results listing with filtering capabilities.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Streamlit Parity',
      description: 'Saved search list with expand/collapse, result browsing with sort/filter by ticker and KPI, multi-select for batch actions (save to strategies, create portfolio, export CSV), side-by-side compare modal, and delete/copy/load actions per search.',
      rationale: 'Full feature parity with Streamlit render_mass_strategy_results — expandable search cards with config summaries, per-result KPI grids, equity curves, compare mode, and batch operations.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'Simple expandable list of saved searches showing date and result count. Click to expand into a sortable results table (same format as Mass Builder V3). Batch actions limited to Save All and Export. Inline delete confirmation per search. No compare mode, no portfolio creation.',
      rationale: 'Focused browsing — each search expands into a clean data table. Only the two most common batch actions retained. Inline confirmations avoid modal interruptions.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Results Explorer with portfolio builder slots, radar chart comparison for up to 4 strategies, "Best Of" highlight badges (best WR, PF, R, trades), rich result cards with sparklines and compare/add-to-portfolio actions, tabbed search navigation, and CSV export.',
      rationale: 'The portfolio builder lets users visually assemble a portfolio from results by clicking "Add to Portfolio" — slots fill with live aggregate stats. The radar comparison overlays multiple strategies on the same chart for instant visual comparison. Best-of badges surface category winners automatically. Each result card is a self-contained unit with all KPIs, sparkline, and action buttons.',
    },
    component: V4,
  },
  {
    meta: {
      id: 'v5-production',
      name: 'Production (Simple + Worker Status)',
      description: 'Simple non-expandable search cards: name, status badge (completed/running/queued), date, config summary (tickers × TFs × dirs × entries × exits × confluences × stops × targets = evaluations), best KPIs for completed, progress bar with elapsed/ETA for running, queue message for queued. Actions: View, Load, Copy, Cancel (running), Delete.',
      rationale: 'Clean list focused on search management. Background worker status visible with progress, elapsed time, and ETA. No expanding into results — View navigates to mass builder with results loaded.',
    },
    component: V5,
  },
];

export default function MassResultsPage() {
  return <VersionedPage pageKey="mass-results" versions={versions} />;
}
