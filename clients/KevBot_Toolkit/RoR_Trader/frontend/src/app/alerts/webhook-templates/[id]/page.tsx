'use client';

import { useParams } from 'next/navigation';
import dynamic from 'next/dynamic';

const WebhookTemplateDetailPage = dynamic(() => import('@/views/WebhookTemplateDetailPage'), { ssr: false });

export default function Page() {
  const params = useParams();
  const id = params?.id ? String(params.id) : '';
  return <WebhookTemplateDetailPage templateId={id} />;
}
