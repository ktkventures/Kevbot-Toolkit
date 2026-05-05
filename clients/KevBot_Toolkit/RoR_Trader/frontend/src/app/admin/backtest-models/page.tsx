'use client';

import dynamic from 'next/dynamic';

const BacktestModelsAdminPage = dynamic(
  () => import('@/views/StrategyModelsAdminPage').then(m => m.BacktestModelsAdminPage),
  { ssr: false }
);

export default function Page() {
  return <BacktestModelsAdminPage />;
}
