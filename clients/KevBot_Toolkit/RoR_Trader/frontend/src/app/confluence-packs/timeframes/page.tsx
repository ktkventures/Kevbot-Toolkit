'use client';

import dynamic from 'next/dynamic';

const TimeframesPage = dynamic(() => import('@/views/TimeframesPage'), { ssr: false });

export default function Page() {
  return <TimeframesPage />;
}
