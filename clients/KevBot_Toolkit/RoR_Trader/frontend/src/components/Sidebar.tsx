'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';
import { useViewMode } from '@/providers/StoreProvider';

function ViewModeToggle() {
  const { mode, toggle } = useViewMode();
  return (
    <button
      onClick={toggle}
      className="w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs transition-colors"
      style={{
        background: mode === 'design' ? '#FF980020' : 'var(--green)10',
        border: `1px solid ${mode === 'design' ? '#FF980040' : 'var(--green)30'}`,
      }}
    >
      <div className="flex items-center gap-2">
        <span
          className="w-2 h-2 rounded-full"
          style={{ background: mode === 'live' ? 'var(--green)' : '#FF9800' }}
        />
        <span style={{ color: mode === 'live' ? 'var(--green)' : '#FF9800', fontWeight: 600 }}>
          {mode === 'live' ? 'Live' : 'Design Ref'}
        </span>
      </div>
      <span style={{ color: 'var(--text-muted)', fontSize: '10px' }}>
        {mode === 'live' ? 'API Data' : 'Mock Data'}
      </span>
    </button>
  );
}

interface NavItem {
  href: string;
  label: string;
  icon: string;
  children?: { href: string; label: string }[];
}

const navItems: NavItem[] = [
  { href: '/dashboard', label: 'Dashboard', icon: '◈' },
  {
    href: '/confluence-packs', label: 'Confluence Packs', icon: '◇',
    children: [
      { href: '/confluence-packs/tf-confluence', label: 'TF Confluence' },
      { href: '/confluence-packs/general', label: 'General' },
      { href: '/confluence-packs/stop-loss', label: 'Stop Loss' },
      { href: '/confluence-packs/take-profit', label: 'Take Profit' },
      { href: '/confluence-packs/time-exit', label: 'Time Exit' },
      { href: '/confluence-packs/execution-types', label: 'Execution Types' },
      { href: '/confluence-packs/user-packs', label: 'User Packs' },
      { href: '/confluence-packs/pack-builder', label: 'Pack Builder' },
      { href: '/confluence-packs/timeframes', label: 'Timeframes' },
    ],
  },
  {
    href: '/strategies', label: 'Strategies', icon: '⚡',
    children: [
      { href: '/strategy-builder', label: 'Strategy Builder' },
      { href: '/strategies', label: 'My Strategies' },
      { href: '/mass-builder', label: 'Mass Builder' },
      { href: '/mass-results', label: 'Mass Results' },
      { href: '/jobs', label: 'Jobs' },
    ],
  },
  {
    href: '/portfolios', label: 'Portfolios', icon: '▦',
    children: [
      { href: '/portfolios', label: 'My Portfolios' },
      { href: '/portfolio-requirements', label: 'Requirements' },
    ],
  },
  {
    href: '/alerts', label: 'Alerts', icon: '◉',
    children: [
      { href: '/alerts', label: 'Alerts & Signals' },
      { href: '/alerts/webhook-groups', label: 'Webhook Groups' },
      { href: '/alerts/webhook-templates', label: 'Webhook Templates' },
    ],
  },
  {
    href: '/marketplace', label: 'Marketplace', icon: '◆',
    children: [
      { href: '/marketplace', label: 'Browse' },
      { href: '/marketplace/subscriptions', label: 'My Subscriptions' },
      { href: '/marketplace/prop-firms', label: 'Prop Firm Hub' },
    ],
  },
  {
    href: '/creator', label: 'Creator', icon: '★',
    children: [
      { href: '/creator/dashboard', label: 'Dashboard' },
      { href: '/creator/earnings', label: 'Earnings & Payouts' },
      { href: '/creator/publish', label: 'Publish' },
    ],
  },
  { href: '/pricing', label: 'Pricing & Plans', icon: '◎' },
  {
    href: '/admin', label: 'Admin', icon: '▣',
    children: [
      { href: '/admin', label: 'Platform Overview' },
      { href: '/admin/users', label: 'Users' },
      { href: '/admin/curation', label: 'Curation' },
      { href: '/admin/data-health', label: 'Data Health' },
      { href: '/admin/divergence', label: 'Divergence Summary' },
      { href: '/admin/parity', label: 'Parity Comparison' },
      { href: '/admin/live-models', label: 'Live Models' },
      { href: '/admin/backtest-models', label: 'Backtest / Algo Models' },
    ],
  },
  {
    href: '/settings', label: 'Settings', icon: '⚙',
    children: [
      { href: '/settings/themes', label: 'Themes' },
      { href: '/settings/display', label: 'Display' },
      { href: '/settings/connections', label: 'Connections' },
      { href: '/settings/account', label: 'Account' },
      { href: '/settings/profile', label: 'Profile & Roles' },
    ],
  },
];

export default function Sidebar() {
  const pathname = usePathname();
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  const isActive = (href: string) => {
    if (pathname === href) return true;
    // Only match prefix for paths that aren't also a parent of other nav items
    // e.g. /alerts shouldn't match /alerts/webhook-templates
    if (href !== '/' && pathname?.startsWith(href + '/')) {
      // Check if another child nav item is a better (more specific) match
      const allChildHrefs = navItems.flatMap((item) => item.children?.map((c) => c.href) ?? []);
      const hasBetterMatch = allChildHrefs.some((h) => h !== href && pathname?.startsWith(h) && h.startsWith(href));
      return !hasBetterMatch;
    }
    return false;
  };

  const isSectionActive = (item: NavItem) =>
    isActive(item.href) || item.children?.some((c) => isActive(c.href));

  const toggleExpand = (label: string) =>
    setExpanded((prev) => ({ ...prev, [label]: !prev[label] }));

  return (
    <aside
      className="fixed left-0 top-0 h-full flex flex-col border-r overflow-y-auto"
      style={{
        width: 'var(--sidebar-width)',
        background: 'var(--sidebar-bg)',
        borderColor: 'var(--border)',
        backdropFilter: 'var(--card-backdrop)',
        WebkitBackdropFilter: 'var(--card-backdrop)',
        zIndex: 2,
      }}
    >
      <div className="px-5 py-4 border-b" style={{ borderColor: 'var(--border)' }}>
        <h1 className="text-lg font-bold" style={{ color: 'var(--logo-color)' }}>
          RoR Trader
        </h1>
        <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
          Rate of Return Trading System
        </p>
      </div>

      <nav className="flex-1 py-3 px-2">
        {navItems.map((item) => {
          const active = isSectionActive(item);
          const isOpen = expanded[item.label] ?? active;

          if (!item.children) {
            return (
              <Link
                key={item.href}
                href={item.href}
                className="flex items-center gap-3 px-3 py-2.5 rounded-lg mb-0.5 text-sm transition-colors"
                style={{
                  background: isActive(item.href) ? 'var(--accent-muted)' : 'transparent',
                  color: isActive(item.href) ? 'var(--accent)' : 'var(--text-secondary)',
                }}
              >
                <span className="text-base">{item.icon}</span>
                {item.label}
              </Link>
            );
          }

          return (
            <div key={item.label} className="mb-0.5">
              <button
                onClick={() => toggleExpand(item.label)}
                className="w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-sm transition-colors"
                style={{
                  color: active ? 'var(--accent)' : 'var(--text-secondary)',
                }}
              >
                <span className="flex items-center gap-3">
                  <span className="text-base">{item.icon}</span>
                  {item.label}
                </span>
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  {isOpen ? '▾' : '▸'}
                </span>
              </button>
              {isOpen && (
                <div className="ml-8 mt-0.5 space-y-0.5">
                  {item.children.map((child) => (
                    <Link
                      key={child.href}
                      href={child.href}
                      className="block px-3 py-1.5 rounded text-xs transition-colors"
                      style={{
                        color: isActive(child.href) ? 'var(--accent)' : 'var(--text-muted)',
                        background: isActive(child.href) ? 'var(--accent-muted)' : 'transparent',
                      }}
                    >
                      {child.label}
                    </Link>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </nav>

      {/* View mode toggle */}
      <div className="px-4 py-2 border-t" style={{ borderColor: 'var(--border)' }}>
        <ViewModeToggle />
      </div>

      <div className="px-4 py-3 border-t" style={{ borderColor: 'var(--border)' }}>
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full" style={{ background: 'var(--green)' }} />
          <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
            Polygon.io — Live
          </span>
        </div>
      </div>
    </aside>
  );
}
