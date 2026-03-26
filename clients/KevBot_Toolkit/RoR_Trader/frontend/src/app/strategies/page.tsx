'use client';

import dynamic from 'next/dynamic';

// Dynamic import to prevent SSR prerender errors
const StrategiesPage = dynamic(() => import('@/views/StrategiesPage'), { ssr: false });

export default function Page() {
  return <StrategiesPage />;
}
