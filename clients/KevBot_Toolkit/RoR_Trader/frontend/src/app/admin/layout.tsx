'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

/**
 * Shared admin nav. Renders the page content first, then a strip of
 * cross-links at the bottom so you can jump between admin sub-pages
 * without typing URLs.
 */

const ADMIN_LINKS: { href: string; label: string }[] = [
  { href: '/admin',                  label: 'Dashboard' },
  { href: '/admin/strategy-health',  label: 'Strategy Health' },
  { href: '/admin/data-health',      label: 'Data Health' },
  { href: '/admin/parity',           label: 'Parity' },
  { href: '/admin/divergence',       label: 'Divergence' },
  { href: '/admin/backtest-models',  label: 'Backtest Models' },
  { href: '/admin/live-models',      label: 'Live Models' },
  { href: '/admin/users',            label: 'Users' },
  { href: '/admin/curation',         label: 'Curation' },
];

export default function AdminLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <>
      {children}
      <nav
        aria-label="Admin navigation"
        style={{
          marginTop: 24,
          padding: '12px 16px',
          borderTop: '1px solid var(--border)',
          display: 'flex',
          flexWrap: 'wrap',
          gap: 8,
          fontSize: 13,
        }}
      >
        <span
          style={{
            color: 'var(--text-muted)',
            marginRight: 4,
            alignSelf: 'center',
          }}
        >
          Admin:
        </span>
        {ADMIN_LINKS.map(({ href, label }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              prefetch={false}
              style={{
                padding: '4px 10px',
                borderRadius: 4,
                border: '1px solid var(--border)',
                background: active ? 'var(--accent)' : 'var(--bg-input)',
                color: active ? 'white' : 'var(--text-muted)',
                textDecoration: 'none',
                whiteSpace: 'nowrap',
              }}
            >
              {label}
            </Link>
          );
        })}
      </nav>
    </>
  );
}
