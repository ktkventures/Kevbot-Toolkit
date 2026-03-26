'use client';

import dynamic from 'next/dynamic';

const SettingsAccountPage = dynamic(() => import('@/views/SettingsAccountPage'), { ssr: false });

export default function Page() {
  return <SettingsAccountPage />;
}
