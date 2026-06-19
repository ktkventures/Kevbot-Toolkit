'use client';

import dynamic from 'next/dynamic';

const AdminTasksPage = dynamic(() => import('@/views/AdminTasksPage'), {
  ssr: false,
});

export default function Page() {
  return <AdminTasksPage />;
}
