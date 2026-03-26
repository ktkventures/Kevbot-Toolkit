'use client';

import dynamic from 'next/dynamic';

const PortfolioNewPage = dynamic(() => import('@/views/PortfolioNewPage'), { ssr: false });

export default function Page() {
  return <PortfolioNewPage />;
}
