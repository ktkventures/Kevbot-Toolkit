'use client';

import dynamic from 'next/dynamic';

const WebhookGroupsPage = dynamic(() => import('@/views/WebhookGroupsPage'), { ssr: false });

export default function Page() {
  return <WebhookGroupsPage />;
}
