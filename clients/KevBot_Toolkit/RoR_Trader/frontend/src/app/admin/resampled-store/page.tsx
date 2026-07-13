'use client';

import dynamic from 'next/dynamic';

const ResampledStoreAdminPage = dynamic(() => import('@/views/ResampledStoreAdminPage'), {
  ssr: false,
});

export default function Page() {
  return <ResampledStoreAdminPage />;
}
