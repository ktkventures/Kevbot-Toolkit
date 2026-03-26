'use client';

import PageSwitch from '@/components/PageSwitch';
import PackBuilderPage from '@/views/PackBuilderPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';
import V5 from './versions/V5';
import V6 from './versions/V6';
import V7 from './versions/V7';

const designVersions = [
  {
    meta: {
      id: 'v1-initial',
      name: 'Initial Scaffold',
      description: 'Two-column builder layout: left side has pack info, indicator/interpreter code blocks, and trigger editor; right side has live preview chart, detected outputs, and test results with per-trigger breakdown.',
      rationale: 'First pass — establishes the pack authoring workflow with mock code blocks, state transition trigger definitions, execution type selection, and test statistics.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Streamlit Parity',
      description: 'Complete builder with pack info (name, category, description, version), indicator logic card with dark-themed code preview and parameter schema builder (add/remove with name, type, default, min, max), interpreter logic card with code preview and output state definer, trigger definitions with direction/type/execution and live preview badges. Right column: live preview chart with symbol/TF/session selectors, detected outputs, test results (trigger fires, signal accuracy, avg bars between, state coverage, per-trigger breakdown), and validation panel (syntax, output coverage, parameter types, trigger states, naming conventions).',
      rationale: 'Full parity with Streamlit — complete pack authoring experience with structured parameter schema, output state definitions, trigger configuration with all execution types, and multi-panel validation feedback.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'Single-column stacked layout with 5 collapsible step sections: Info, Parameters, Outputs, Triggers, Preview & Validation. No code editor (V4 territory). Inline validation per step. Save at bottom.',
      rationale: 'Replaced the 2-column layout with a focused step-by-step flow. Each step is a collapsible section with a numbered badge and pass/fail indicator. Parameters, outputs, and triggers are inline-editable with add/remove. Preview step shows a pack summary and validation checklist. No code editor placeholder — that is deliberately deferred to V4.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Two-column builder: left side has tabbed editor (Info, Params, Outputs, Triggers) with inline add/remove, category pills, and state transition diagram. Right side has 4 live preview panels: Logic Flow, Code Preview, Indicator Chart, Test Results.',
      rationale: 'Pack building is a creative authoring experience. V4 makes it visual and immediate with logic flow diagrams, code previews, and instant visual feedback.',
    },
    component: V4,
  },
  {
    meta: {
      id: 'v5-prompt-wizard',
      name: 'Prompt Wizard',
      description: '5-step wizard: Pack Type & Info, Define Structure, Generate & Copy Prompt, Paste & Validate, Review & Install.',
      rationale: 'Prompt-based approach for users without AI API. Clean wizard guides through full pack creation with comprehensive validation.',
    },
    component: V5,
  },
  {
    meta: {
      id: 'v6-revised-wizard',
      name: 'Revised Wizard (Describe, Generate, Refine)',
      description: '5-step wizard with revised flow: Pack Info, Generate Structure, Refine Structure, Generate & Validate Code, Review & Install.',
      rationale: 'Users describe first, AI proposes structure, user refines. Chart Preview and Parity Simulator for verification.',
    },
    component: V6,
  },
  {
    meta: {
      id: 'v7-api-connected',
      name: 'API-Connected (AI-Powered)',
      description: 'Same 5-step wizard as V6 but with built-in AI API integration. Auto-fix loop, model selector, conversation history.',
      rationale: 'Seamless AI-powered flow eliminates manual copy/paste. Auto-fix loop reduces iteration time.',
    },
    component: V7,
  },
];

export default function Page() {
  return <PageSwitch live={PackBuilderPage} versions={designVersions} pageKey="pack-builder" />;
}
