'use client';

import dynamic from 'next/dynamic';

const BarCacheAdminPage = dynamic(() => import('@/views/BarCacheAdminPage'), {
  ssr: false,
});

export default function Page() {
  return <BarCacheAdminPage />;
}
