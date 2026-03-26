'use client';

import dynamic from 'next/dynamic';

const TfConfluencePage = dynamic(() => import('@/views/TfConfluencePage'), { ssr: false });

export default function Page() {
  return <TfConfluencePage />;
}
