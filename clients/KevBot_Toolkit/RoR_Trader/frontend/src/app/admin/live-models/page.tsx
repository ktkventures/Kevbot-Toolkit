'use client';

import dynamic from 'next/dynamic';

const LiveModelsAdminPage = dynamic(
  () => import('@/views/StrategyModelsAdminPage').then(m => m.LiveModelsAdminPage),
  { ssr: false }
);

export default function Page() {
  return <LiveModelsAdminPage />;
}
