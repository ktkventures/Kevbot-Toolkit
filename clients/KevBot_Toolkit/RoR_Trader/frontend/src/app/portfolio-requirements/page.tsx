'use client';

import dynamic from 'next/dynamic';

const RequirementsPage = dynamic(() => import('@/views/RequirementsPage'), { ssr: false });

export default function Page() {
  return <RequirementsPage />;
}
