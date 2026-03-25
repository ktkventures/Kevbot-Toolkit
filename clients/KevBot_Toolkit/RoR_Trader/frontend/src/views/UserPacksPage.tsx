'use client';

import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import Link from 'next/link';

export default function UserPacksPage() {
  return (
    <div>
      <PageHeader
        title="User Packs"
        subtitle="Custom indicator packs"
        actions={
          <span className="text-xs px-2 py-1 rounded-full flex items-center gap-1.5"
            style={{ background: 'var(--green)15', color: 'var(--green)' }}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--green)' }} />
            Live
          </span>
        }
      />
      <Card>
        <div className="text-center py-16">
          <div className="text-4xl mb-4" style={{ opacity: 0.3 }}>&#9881;</div>
          <h3 className="text-lg font-medium mb-2" style={{ color: 'var(--text-primary)' }}>
            No Custom Packs Installed
          </h3>
          <p className="text-sm mb-6" style={{ color: 'var(--text-muted)', maxWidth: 400, margin: '0 auto' }}>
            User packs are custom indicator packs created with the Pack Builder.
            Build your own indicators, interpreters, and triggers — then install them here.
          </p>
          <Link href="/confluence-packs/pack-builder">
            <button className="px-4 py-2 rounded-lg text-sm font-medium"
              style={{ background: 'var(--accent)', color: '#fff' }}>
              Go to Pack Builder
            </button>
          </Link>
        </div>
      </Card>
    </div>
  );
}
