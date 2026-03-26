'use client';

import { useParams } from 'next/navigation';
import dynamic from 'next/dynamic';

const PortfolioDetailPage = dynamic(() => import('@/views/PortfolioDetailPage'), { ssr: false });

export default function Page() {
  const params = useParams();
  const id = params?.id ? Number(params.id) : 0;
  return <PortfolioDetailPage portfolioId={id} />;
}
