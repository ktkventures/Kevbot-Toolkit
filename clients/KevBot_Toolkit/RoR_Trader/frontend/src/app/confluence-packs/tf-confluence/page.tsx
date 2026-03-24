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
      description: 'Clean card layout with toggle switches, category tags, and 5-tab detail view.',
      rationale: 'First pass — focused on getting the core structure right: pack list with enable/disable, basic detail view with Parameters, Plot Settings, Outputs & Triggers, Preview, and Danger Zone tabs. Kept it simple to establish the layout pattern.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Parity',
      description: 'Everything from Streamlit, re-designed. 6 tabs including Code, grouped by category, full output descriptions, all trigger metadata.',
      rationale: 'Direct port of every feature from the Streamlit app. Added: packs grouped by category headers, output descriptions (e.g. "SML = Full Bull Stack"), all trigger details with direction/type/execution/ID, template-specific color pickers, Code tab with parameter config and source preview, interpreter state timeline and trigger events tables in Preview. Nothing removed — if it exists in Streamlit, it\'s here.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'No page navigation. Accordion-style inline editing, category filter pills, overflow menu for rare actions.',
      rationale: 'Removed friction: no separate detail page, no tabs. Packs expand inline to show parameters + quick reference side-by-side. Plot Settings and Danger Zone moved behind a "..." overflow menu since they\'re rarely used. Category filter pills replace grouping. A trader can scan all packs, tweak a parameter, and move on without ever leaving the page.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Inline SVG indicator previews, live state pulses, state distribution donuts, drag-to-reorder, comparison mode, preset buttons.',
      rationale: 'Reimagined the page as a live dashboard. Each pack shows its current state with an animated pulse, a donut chart of state distribution, and usage stats (strategies using it, last triggered). Added: drag-to-reorder priority, compare-two-packs mode, quick-configure presets (Scalping/Swing/Position), compact vs rich view toggle, smart grouping by usage. Visual parameter tuning with sliders. The goal: make a trader feel like they\'re looking at a living system, not a config page.',
    },
    component: V4,
  },
  {
    meta: {
      id: 'v5-production',
      name: 'Production',
      description: 'All 8 templates, locked params after save, nested variations, search, updated trigger naming [C]/[L]/[LC]/[CC], trigger params, "Create New Template" → Pack Builder.',
      rationale: 'Based on V2 structure with key production changes: parameters lock after save to protect live strategies/portfolios, variations nest under their default template (created via Copy), search bar for discovery at scale, tags on cards instead of category sections, updated execution type naming from display settings, trigger-level parameters shown in detail view. "New Pack" replaced with link to Pack Builder for template creation.',
    },
    component: V5,
  },
];

export default function TfConfluencePage() {
  return <VersionedPage pageKey="tf-confluence" versions={versions} />;
}
