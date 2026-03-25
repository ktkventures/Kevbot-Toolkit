'use client';

import { useParams } from 'next/navigation';
import PageSwitch from '@/components/PageSwitch';
import WebhookTemplateDetailPage from '@/views/WebhookTemplateDetailPage';
import V5 from './versions/V5';

function LiveDetail() {
  const params = useParams();
  const id = params?.id ? String(params.id) : '';
  return <WebhookTemplateDetailPage templateId={id} />;
}

const designVersions = [
  { meta: { id: 'v5', name: 'V5 Account-Based Detail', description: 'Phase 39 detail page.', rationale: '' }, component: V5 },
];

export default function Page() {
  return <PageSwitch live={LiveDetail} versions={designVersions} pageKey="webhook-template-detail" />;
}
