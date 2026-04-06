'use client';

import dynamic from 'next/dynamic';

const ExecutionTypesPage = dynamic(() => import('@/views/ExecutionTypesPage'), { ssr: false });

export default function Page() {
  return <ExecutionTypesPage />;
}
