'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';

const versions = [
  {
    meta: {
      id: 'v1-user-table',
      name: 'User Table',
      description: 'Simple table with name, email, role, join date, and status columns.',
      rationale: 'Basic user list. Just the facts in a table. No filtering, no actions, no modals. Quick reference for who is on the platform.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-management',
      name: 'Full Management',
      description: 'Searchable/filterable user table with detail modal showing activity, subscriptions, earnings. Role change and suspend/ban actions.',
      rationale: 'Complete user management. Search by name or email, filter by role and status. Click a row to open a detail modal with subscription count, earnings, strategies created, and last active. Admin actions: change role, suspend/unsuspend.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-compact-search',
      name: 'Compact Searchable',
      description: 'Compact searchable table with inline status badges.',
      rationale: 'Stripped-down user list with search. No modals, no filters, no admin actions. Just search and scan.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'User segment visualization with clickable segment cards, cohort retention heatmap, retention curves, LTV display.',
      rationale: 'Analytics-first user view. Segment cards (Power Creator, Growing Creator, Engaged Subscriber, New User, At Risk, Churned) filter the user list on click. Cohort retention table with color-coded percentages. Retention curve chart. Each user shows their segment badge and lifetime value.',
    },
    component: V4,
  },
];

export default function AdminUsersPage() {
  return <VersionedPage pageKey="admin-users" versions={versions} />;
}
