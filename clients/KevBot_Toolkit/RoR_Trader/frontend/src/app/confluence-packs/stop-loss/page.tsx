'use client';

import dynamic from 'next/dynamic';

const StopLossPage = dynamic(() => import('@/views/StopLossPage'), { ssr: false });

export default function Page() {
  return <StopLossPage />;
}
