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
      description: 'Toggle-switch pack list for Stops and Sizing categories, 4-tab detail view with contextual risk explanations and output state descriptions.',
      rationale: 'First pass — focused on ATR stops, swing stops, and fixed risk packs with volatility-aware output labels and strategy impact notes.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Streamlit Parity',
      description: 'Template-driven architecture with all 7 risk_management_packs.py templates: ATR-Based, Fixed Dollar, Percentage, Swing, Risk:Reward composite, ATR Trailing, Breakeven Stop. 5-tab detail view (Parameters with conditional visibility for composite template, Outputs with stop/target config detail and exit priority order, Preview with stop placement examples and lifecycle diagrams, Code with builder functions and JSON config, Danger Zone). Category-grouped list (Volatility, Fixed, Structure, Composite, Trailing) with stop/target summaries, contextual help per template, and create modal.',
      rationale: 'Mirrors the Streamlit render_risk_management_packs() page — all 7 templates with full parameter schemas, conditional parameter visibility (rr_ratio template), stop/target summary formatting, trailing and breakeven stop lifecycle visualization, and risk-specific contextual explanations.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'Accordion-style inline editing with category filter pills (All | Volatility | Fixed | Structure | Trailing), stop type badges, contextual help + stop/target summary in expanded view.',
      rationale: 'Same accordion pattern as TF Confluence V3. Each pack expands to show parameters with per-field help text on the left, and stop placement explanation (stop/target summary cards + numbered contextual tips) on the right. No separate detail page, no tabs. Overflow menu for Rename/Copy/Delete.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Visual stop placement diagrams on mock price charts per pack (entry/stop/target with R:R brackets, trailing stop staircase, breakeven level). Interactive Risk/Reward calculator with entry/stop/target/position inputs and dollar risk, breakeven win rate. Position sizing visualizer with account size, risk %, stop distance inputs and risk bar. Stop type comparison matrix (ATR/Fixed/Pct/Swing/Trail/BE). Pack cards show inline mini stop diagrams. Detail modal has live-updating stop placement as parameter sliders change, stop lifecycle flowcharts for Trail/BE types.',
      rationale: 'Risk management is fundamentally about spatial relationships between entry, stop, and target on a price chart. The V4 creative hook makes these relationships visual: every pack shows a stop placement diagram rather than just numbers. The calculator and position sizer are interactive tools that bridge the gap between pack config and real trading decisions.',
    },
    component: V4,
  },
];

export default function RiskManagementPage() {
  return <VersionedPage pageKey="risk-management" versions={versions} />;
}
