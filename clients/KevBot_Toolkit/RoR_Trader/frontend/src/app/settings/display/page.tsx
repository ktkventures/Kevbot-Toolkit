'use client';

import dynamic from 'next/dynamic';

const SettingsDisplayPage = dynamic(() => import('@/views/SettingsDisplayPage'), { ssr: false });

export default function Page() {
  return <SettingsDisplayPage />;
}
