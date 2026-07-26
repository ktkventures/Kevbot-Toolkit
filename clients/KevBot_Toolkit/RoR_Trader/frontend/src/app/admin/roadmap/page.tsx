'use client';

import dynamic from 'next/dynamic';

const AdminRoadmapPage = dynamic(() => import('@/views/AdminRoadmapPage'), {
  ssr: false,
});

export default function Page() {
  return <AdminRoadmapPage />;
}
