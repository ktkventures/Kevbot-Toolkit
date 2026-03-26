'use client';

import dynamic from 'next/dynamic';

const GeneralPacksPage = dynamic(() => import('@/views/GeneralPacksPage'), { ssr: false });

export default function Page() {
  return <GeneralPacksPage />;
}
