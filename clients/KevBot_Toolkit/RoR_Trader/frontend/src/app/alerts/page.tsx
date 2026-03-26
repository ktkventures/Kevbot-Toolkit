'use client';

import dynamic from 'next/dynamic';

const AlertsPage = dynamic(() => import('@/views/AlertsPage'), { ssr: false });

export default function Page() {
  return <AlertsPage />;
}
