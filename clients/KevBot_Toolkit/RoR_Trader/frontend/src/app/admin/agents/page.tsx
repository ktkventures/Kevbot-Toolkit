'use client';

import dynamic from 'next/dynamic';

const AdminAgentsPage = dynamic(() => import('@/views/AdminAgentsPage'), {
  ssr: false,
});

export default function Page() {
  return <AdminAgentsPage />;
}
