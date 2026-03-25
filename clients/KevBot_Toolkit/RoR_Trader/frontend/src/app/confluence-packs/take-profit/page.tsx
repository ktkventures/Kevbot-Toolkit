'use client';

import PageSwitch from '@/components/PageSwitch';
import TakeProfitPage from '@/views/TakeProfitPage';
import V1 from './versions/V1';

const designVersions = [
  { meta: { id: 'v1', name: 'V1 Production', description: '6 take profit templates with locked params and nested variations.', rationale: '' }, component: V1 },
];

export default function Page() {
  return <PageSwitch live={TakeProfitPage} versions={designVersions} pageKey="take-profit" />;
}
