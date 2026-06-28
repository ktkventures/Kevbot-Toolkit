'use client';

import dynamic from 'next/dynamic';

const TradeSnapshotsAdminPage = dynamic(
  () => import('@/views/TradeSnapshotsAdminPage'),
  { ssr: false },
);

export default function Page() {
  return <TradeSnapshotsAdminPage />;
}
