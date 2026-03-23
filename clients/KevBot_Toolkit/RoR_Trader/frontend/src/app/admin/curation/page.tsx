'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';

const versions = [
  {
    meta: {
      id: 'v1-toggle-list',
      name: 'Toggle List',
      description: 'Simple portfolio list with "Add to Free Rotation" / "Remove" toggle buttons.',
      rationale: 'Minimal curation. Each portfolio shows name, creator, subscriber count, and win rate. One button toggles free rotation membership. No categories, no review queue, no featured system.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-curation',
      name: 'Full Curation',
      description: 'Tabbed interface: Free Rotation, Featured, Review Queue, All Content. Content review modal with approve/reject. Quality metrics per item.',
      rationale: 'Complete content management. Tabs separate free rotation, featured items, and review queue. Each item shows type badge, review status, and quality metrics (WR, PF, DD). Review modal shows detailed stats with approve/reject actions. Feature toggle for spotlight items.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-drag-drop',
      name: 'Drag & Drop',
      description: 'Two-column layout with drag-to-reorder free rotation list and available items pool.',
      rationale: 'Visual curation. Left column is the ordered free rotation list — drag handles let you reorder priority. Right column shows available items with an "Add" button. Numbered positions make rotation order explicit.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Visual content board with quality scores, performance metrics, A/B test badges, and performance-based auto-suggestions view.',
      rationale: 'Smart curation. Each item gets a quality score (0-100) displayed as a colored circle. Board view shows two columns with full metric cards. Suggestions view ranks available items by quality score for data-driven curation decisions. A/B variant badges indicate items being tested.',
    },
    component: V4,
  },
];

export default function AdminCurationPage() {
  return <VersionedPage pageKey="admin-curation" versions={versions} />;
}
