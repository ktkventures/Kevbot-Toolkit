'use client';

import { useRouter } from 'next/navigation';

interface PageHeaderProps {
  title: string;
  subtitle?: string;
  backHref?: string;
  actions?: React.ReactNode;
}

export default function PageHeader({ title, subtitle, backHref, actions }: PageHeaderProps) {
  const router = useRouter();

  return (
    <div className="flex items-start justify-between mb-6">
      <div>
        {backHref && (
          <button
            onClick={() => router.push(backHref)}
            className="text-sm mb-2 flex items-center gap-1"
            style={{ color: 'var(--text-muted)' }}
          >
            ← Back
          </button>
        )}
        <h1 className="text-2xl font-bold">{title}</h1>
        {subtitle && (
          <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>{subtitle}</p>
        )}
      </div>
      {actions && <div className="flex gap-2">{actions}</div>}
    </div>
  );
}
