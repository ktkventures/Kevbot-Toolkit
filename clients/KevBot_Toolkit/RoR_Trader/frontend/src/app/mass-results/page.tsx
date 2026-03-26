'use client';

import dynamic from 'next/dynamic';

const MassResultsPage = dynamic(() => import('@/views/MassResultsPage'), { ssr: false });

export default function Page() {
  return <MassResultsPage />;
}
