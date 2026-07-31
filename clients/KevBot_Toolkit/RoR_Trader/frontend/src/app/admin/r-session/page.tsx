'use client';

import dynamic from 'next/dynamic';

const AdminRSessionPage = dynamic(() => import('@/views/AdminRSessionPage'), {
  ssr: false,
});

export default function Page() {
  return <AdminRSessionPage />;
}
