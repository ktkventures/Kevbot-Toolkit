'use client';

import dynamic from 'next/dynamic';

const UserPacksPage = dynamic(() => import('@/views/UserPacksPage'), { ssr: false });

export default function Page() {
  return <UserPacksPage />;
}
