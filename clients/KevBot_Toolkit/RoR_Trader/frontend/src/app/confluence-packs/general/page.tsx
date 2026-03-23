'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';

const versions = [
  {
    meta: {
      id: 'v1-initial',
      name: 'Initial Scaffold',
      description: 'Toggle-switch pack list with category tags, 4-tab detail view (Parameters, Outputs & Triggers, Preview, Danger Zone), and create modal.',
      rationale: 'First pass — covers the Session and Calendar pack types with enable/disable toggles, output state previews, and pack CRUD.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Streamlit Parity',
      description: 'Template-driven architecture with real general_packs.py data: Time of Day, Trading Session, Day of Week, Calendar Filter. 5-tab detail view (Parameters with template-specific forms, Outputs & Triggers with condition/trigger split, Preview with state timeline and trigger events, Code with evaluator functions and JSON config, Danger Zone). Category-grouped list with Time and Calendar headers, contextual parameter previews (time windows, session hours, day toggles), and create modal with template selector.',
      rationale: 'Mirrors the Streamlit render_general_packs() page — real template schemas, bool/select/int/float parameter types, condition-only packs (Day of Week), trigger-equipped packs (Time of Day, Calendar Filter), and template-specific contextual summaries.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'Accordion-style inline editing with category filter pills (All | Session | Calendar), overflow menu for rare actions (Rename, Copy, Delete). No page navigation.',
      rationale: 'Removed friction: packs expand inline to show parameters (bool toggles, selects, number inputs) + quick reference (outputs, triggers) side-by-side. All editing happens on one page. Overflow menu hides rare actions. Category pills replace tab-based grouping.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: '24-hour activity timeline showing session, time window, and overlap bands with live "now" marker. Live ticking clock with session detection. Week-at-a-glance calendar grid with FOMC/NFP event badges and blocked-day highlights. Combined gate status (TRADE/WAIT). Inline SVG icons per template (clock, session bars, day-of-week columns, calendar grid). Pack cards with state pulse animation, detail modal with slider-based parameter editing, and visual output state badges.',
      rationale: 'General packs are inherently time-based, so the V4 creative hook is temporal visualization: a timeline that shows when each condition is active throughout the day, plus a weekly calendar that combines day-of-week and calendar filter into one view. The live clock reinforces that these packs gate real-time trading decisions.',
    },
    component: V4,
  },
];

export default function GeneralPacksPage() {
  return <VersionedPage pageKey="general-packs" versions={versions} />;
}
