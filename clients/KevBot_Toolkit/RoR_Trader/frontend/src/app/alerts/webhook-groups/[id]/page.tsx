'use client';

import { useParams } from 'next/navigation';
import dynamic from 'next/dynamic';

const WebhookGroupDetailPage = dynamic(() => import('@/views/WebhookGroupDetailPage'), { ssr: false });

export default function Page() {
  const params = useParams();
  const id = params?.id ? String(params.id) : '';
  return <WebhookGroupDetailPage groupId={id} />;
}
