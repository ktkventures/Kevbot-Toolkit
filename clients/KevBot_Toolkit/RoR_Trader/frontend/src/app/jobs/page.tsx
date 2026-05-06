'use client';

import dynamic from 'next/dynamic';

const JobsPage = dynamic(() => import('@/views/JobsPage'), { ssr: false });

export default function Page() {
  return <JobsPage />;
}
