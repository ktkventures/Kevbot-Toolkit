'use client';

import dynamic from 'next/dynamic';

const PortfoliosPage = dynamic(() => import('@/views/PortfoliosPage'), { ssr: false });

export default function Page() {
  return <PortfoliosPage />;
}
