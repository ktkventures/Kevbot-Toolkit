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
      description: 'List/editor view for requirement sets with built-in prop firm presets, custom rules, clone, and delete confirmation modal.',
      rationale: 'First pass — establishes the CRUD pattern for requirement sets with rule type options, value formatting, and built-in vs custom distinction.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Parity',
      description: 'Enhanced requirements: progress bars per rule, prop firm preset quick-create modal (TTP/FTMO/Topstep), portfolio usage indicators, rule validation hints, compliance preview modal, export as JSON, and tabbed filtering.',
      rationale: 'Matches Streamlit requirements page plus visual enhancements — progress tracking, one-click presets, and compliance checking.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'Single-view inline editing: expandable requirement set cards with compact rule badges (pass/fail dots), click-to-edit values, inline add-rule form, and inline delete confirmation. No page transitions, no modals for editing.',
      rationale: 'Minimal friction — all editing happens in place without navigating to a separate editor view. Removes presets, compliance preview, and import/export for a focused experience.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Visual rule builder with compliance ring scores, traffic light severity indicators, progress bars per rule, compliance simulator, template marketplace modal, prop firm comparison table, and categorized rule badges (Loss/Activity/Risk/Targets).',
      rationale: 'Combines card-based set selection with detailed rule editing. Compliance ring gives at-a-glance health. Simulator runs mock checks. Template marketplace allows one-click import of prop firm presets. Comparison table shows side-by-side firm differences.',
    },
    component: V4,
  },
];

export default function PortfolioRequirementsPage() {
  return <VersionedPage pageKey="portfolio-requirements" versions={versions} />;
}
