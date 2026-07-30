'use client';

import dynamic from 'next/dynamic';

const AdminDispatchPage = dynamic(() => import('@/views/AdminDispatchPage'), {
  ssr: false,
});

export default function Page() {
  return <AdminDispatchPage />;
}
