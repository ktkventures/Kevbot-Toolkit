'use client';

import dynamic from 'next/dynamic';

const SettingsConnectionsPage = dynamic(() => import('@/views/SettingsConnectionsPage'), { ssr: false });

export default function Page() {
  return <SettingsConnectionsPage />;
}
