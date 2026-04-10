'use client';

import dynamic from 'next/dynamic';

const TimeExitPage = dynamic(() => import('@/views/TimeExitPage'), { ssr: false });

export default function Page() {
  return <TimeExitPage />;
}
