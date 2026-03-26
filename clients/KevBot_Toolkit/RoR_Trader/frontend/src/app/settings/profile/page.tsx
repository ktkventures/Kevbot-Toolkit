'use client';

import dynamic from 'next/dynamic';

const SettingsProfilePage = dynamic(() => import('@/views/SettingsProfilePage'), { ssr: false });

export default function Page() {
  return <SettingsProfilePage />;
}
