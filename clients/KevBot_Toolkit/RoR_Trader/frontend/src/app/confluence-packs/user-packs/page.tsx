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
      description: 'Custom pack list with toggle switches, 5-tab detail view (Parameters, Plot Settings, Outputs & Triggers, Preview, Danger Zone), create and delete modals, empty state.',
      rationale: 'First pass — establishes the user-created pack workflow with category color coding, template selection on create, and plot customization (colors, line width).',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Streamlit Parity',
      description: 'Pack cards with usage stats and last modified date. Full CRUD (create, edit, clone, delete). 6-tab detail view (Parameters, Plot Settings, Outputs & Triggers, Preview, Code, Danger Zone) with type-appropriate inputs, state timeline, trigger events table. Create from scratch, from template, or import JSON. Empty state with two CTAs.',
      rationale: 'Full parity with Streamlit — rich pack management with structured data models, code preview, strategy usage tracking, and JSON import/export workflow.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'Accordion-style inline editing with Edit/Delete always visible (no default protection). Empty state with "Create your first pack" CTA. Create modal with name + category + template option. No import/export.',
      rationale: 'Same accordion pattern as other V3s. Key difference: Edit and Delete buttons are always visible in the collapsed row since user packs have no default protection. Create modal is simplified (name, category, optional template). No import/export — that is a V2 feature. Empty state guides new users.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Three-tab experience: My Packs, Marketplace, and Leaderboard. Community marketplace with pack cards showing author, star ratings, download counts, win rate badges, and Install button. Performance leaderboard ranked by win rate with gold/silver/bronze medals. Per-pack indicator preview SVGs (RSI oscillator, Bollinger bands, volume divergence, Ichimoku cloud). Detail modal with 4 tabs (Overview, Params, Triggers, Performance) including win/loss bar chart, signal frequency histogram, and slider-based parameter editing. Create modal with scratch/template/clone workflow and category selector.',
      rationale: 'User packs are the user\'s creative expression. V4 adds a marketplace concept to inspire discovery (browse, install, compare) and a leaderboard to motivate quality. Category-specific indicator preview SVGs help users visually identify what each pack does at a glance. Performance tracking with win rate and signal history turns packs from static configs into living, measurable tools.',
    },
    component: V4,
  },
  {
    meta: {
      id: 'v5-production',
      name: 'Production (Validation + Parity)',
      description: 'V2 base updated: "Outputs" → "States", 8-tab detail view (Parameters, Plot Settings, States & Triggers, Chart Preview with confluence visualization, Signal Validation, Parity Simulator, Code, Danger Zone). Added: visibility toggle (private/public), pack type badge (TF/General), validation status badge, parity score badge. Chart Preview shows confluence state shading + trigger markers. Signal Validation and Parity Simulator match Pack Builder V7 for ongoing health checks.',
      rationale: 'Users need to verify packs continue working correctly after changes. Same Signal Validation and Parity Simulator from Pack Builder available as ongoing health checks. Visibility toggle enables admin-controlled public packs (temporary until marketplace). Consistent naming and badge conventions across all pack-related pages.',
    },
    component: V5,
  },
  {
    meta: { id: 'v5-wired', name: 'Production (Live)', description: 'User packs from API.', rationale: 'Future: load installed user packs.' },
    component: V5, // Same as V5 for now — user packs API is placeholder
  },
];

export default function UserPacksPage() {
  return <VersionedPage pageKey="user-packs" versions={versions} />;
}
