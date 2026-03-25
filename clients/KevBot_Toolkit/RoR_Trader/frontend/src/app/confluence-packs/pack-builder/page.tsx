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
  {
    meta: {
      id: 'v5-prompt-wizard',
      name: 'Prompt Wizard',
      description: '5-step wizard: Pack Type & Info → Define Structure → Generate & Copy Prompt → Paste & Validate → Review & Install. Supports TF Confluence and General pack types. Updated exec types [C]/[L]/[LC]/[CC], fidelity badges [PB]/[CB]. 16-point validation checklist (schema, safety, functions, execution, backtest parity). Horizontal stepper, dark code preview, sentiment badges, state transition tracking.',
      rationale: 'Prompt-based approach for users without AI API. Clean wizard guides through full pack creation with comprehensive validation. Pack type selector adapts the entire wizard (TF = 3 files + all exec types + fidelity; General = 2 files + [C] only + binary outputs).',
    },
    component: V5,
  },
  {
    meta: {
      id: 'v6-revised-wizard',
      name: 'Revised Wizard (Describe → Generate → Refine)',
      description: '5-step wizard with revised flow: Pack Info (describe in plain language) → Generate Structure (AI proposes params/outputs/triggers) → Refine Structure (user tweaks) → Generate & Validate Code (paste LLM response, 16-point validation) → Review & Install (Chart Preview with confluence state shading + trigger markers, Parity Simulator for backtest↔live verification, Code preview). Structure generated FROM description rather than defined manually.',
      rationale: 'Flipped Steps 2+3: users describe first, AI proposes structure, user refines. Much less intimidating than blank-canvas parameter/output/trigger definition. Chart Preview shows real confluence visualization. Parity Simulator replays historical data through both engine paths to verify triggers match between backtest and live.',
    },
    component: V6,
  },
  {
    meta: {
      id: 'v7-api-connected',
      name: 'API-Connected (AI-Powered)',
      description: 'Same 5-step wizard as V6 but with built-in AI API integration. Steps 2 and 4 use direct AI calls instead of copy/paste. Three-column layout on generation steps: AI conversation panel (left), code preview (center), validation (right). Auto-fix loop: up to 3 automatic correction attempts when validation fails. Model selector (Claude Sonnet/Opus, GPT-4/4o). Conversation history shows all AI interactions. Same Review & Install with Signal Validation, Parity Simulator, and Request Fix.',
      rationale: 'Seamless AI-powered flow eliminates manual copy/paste. Auto-fix loop reduces iteration time — system sends validation errors back to AI for surgical correction without regenerating from scratch. Conversation panel provides transparency into what the AI is doing and why.',
    },
    component: V7,
  },
];

export default function PackBuilderPage() {
  return <VersionedPage pageKey="pack-builder" versions={versions} />;
}
