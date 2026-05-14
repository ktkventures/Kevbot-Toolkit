'use client';

import dynamic from 'next/dynamic';

const AdminParityPage = dynamic(
  () => import('@/views/AdminParityPage'),
  { ssr: false }
);

export default function Page() {
  return <AdminParityPage />;
}
