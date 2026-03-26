'use client';

import dynamic from 'next/dynamic';

const StrategyBuilderPage = dynamic(() => import('@/views/StrategyBuilderPage'), { ssr: false });

export default function Page() {
  return <StrategyBuilderPage />;
}
