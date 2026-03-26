'use client';

import dynamic from 'next/dynamic';

const MassBuilderPage = dynamic(() => import('@/views/MassBuilderPage'), { ssr: false });

export default function Page() {
  return <MassBuilderPage />;
}
