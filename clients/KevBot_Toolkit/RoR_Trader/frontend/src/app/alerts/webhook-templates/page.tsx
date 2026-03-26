'use client';

import dynamic from 'next/dynamic';

const WebhookTemplatesPage = dynamic(() => import('@/views/WebhookTemplatesPage'), { ssr: false });

export default function Page() {
  return <WebhookTemplatesPage />;
}
