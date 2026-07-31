'use client';

import dynamic from 'next/dynamic';

const AdminKevinPage = dynamic(() => import('@/views/AdminKevinPage'), {
  ssr: false,
});

export default function Page() {
  return <AdminKevinPage />;
}
