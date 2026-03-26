'use client';

import dynamic from 'next/dynamic';

const PackBuilderPage = dynamic(() => import('@/views/PackBuilderPage'), { ssr: false });

export default function Page() {
  return <PackBuilderPage />;
}
