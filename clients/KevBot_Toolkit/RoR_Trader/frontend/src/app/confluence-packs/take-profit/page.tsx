'use client';

import dynamic from 'next/dynamic';

const TakeProfitPage = dynamic(() => import('@/views/TakeProfitPage'), { ssr: false });

export default function Page() {
  return <TakeProfitPage />;
}
