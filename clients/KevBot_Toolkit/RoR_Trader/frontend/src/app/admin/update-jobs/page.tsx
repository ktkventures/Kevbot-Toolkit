'use client';

import dynamic from 'next/dynamic';

const UpdateJobsAdminPage = dynamic(
  () => import('@/views/UpdateJobsAdminPage'),
  { ssr: false },
);

export default function Page() {
  return <UpdateJobsAdminPage />;
}
