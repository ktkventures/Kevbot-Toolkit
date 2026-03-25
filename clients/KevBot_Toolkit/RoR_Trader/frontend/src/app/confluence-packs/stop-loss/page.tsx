'use client';

import PageSwitch from '@/components/PageSwitch';
import StopLossPage from '@/views/StopLossPage';
import V1 from './versions/V1';

const designVersions = [
  { meta: { id: 'v1', name: 'V1 Production', description: '6 stop loss templates with locked params and nested variations.', rationale: '' }, component: V1 },
];

export default function Page() {
  return <PageSwitch live={StopLossPage} versions={designVersions} pageKey="stop-loss" />;
}
