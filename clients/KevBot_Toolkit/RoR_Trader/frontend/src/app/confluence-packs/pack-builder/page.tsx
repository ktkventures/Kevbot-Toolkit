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
      description: 'Two-column builder: left side has tabbed editor (Info, Params, Outputs, Triggers) with inline add/remove, category pills, and state transition diagram. Right side has 4 live preview panels: Logic Flow (animated node-based diagram with indicator/condition/output/trigger nodes, bezier connections, animated flow dots), Code Preview (syntax-highlighted Python with dark terminal theme), Indicator Chart (mock RSI with OB/OS zones and signal markers that update with params), Test Results (mock signal fires, accuracy, per-trigger breakdown). Top validation bar with progress ring and per-check status badges. State transition diagram auto-generates from outputs + triggers.',
      rationale: 'Pack building is a creative authoring experience. V4 makes it visual and immediate: the logic flow diagram shows how data moves from indicator to condition to output to trigger, the code preview shows what will be generated, and the indicator chart provides instant visual feedback as parameters change. The validation bar gives continuous feedback so builders catch issues before saving.',
    },
    component: V4,
  },
];

export default function PackBuilderPage() {
  return <VersionedPage pageKey="pack-builder" versions={versions} />;
}
