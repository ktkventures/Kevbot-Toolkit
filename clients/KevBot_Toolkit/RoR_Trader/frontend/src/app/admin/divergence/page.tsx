'use client';

import dynamic from 'next/dynamic';

const AdminDivergencePage = dynamic(
  () => import('@/views/AdminDivergencePage'),
  { ssr: false }
);

export default function Page() {
  return <AdminDivergencePage />;
}
