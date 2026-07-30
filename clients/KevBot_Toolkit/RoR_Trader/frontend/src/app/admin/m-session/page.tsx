'use client';

import dynamic from 'next/dynamic';

const AdminMSessionPage = dynamic(() => import('@/views/AdminMSessionPage'), {
  ssr: false,
});

export default function Page() {
  return <AdminMSessionPage />;
}
