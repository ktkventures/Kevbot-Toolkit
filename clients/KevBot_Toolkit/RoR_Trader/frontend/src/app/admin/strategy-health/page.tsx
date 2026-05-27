'use client';

/**
 * Strategy Health page — two tabs (Health Overview + Divergence Backlog).
 *
 * Bypasses VersionedPage's right-edge slide-out because that pattern is
 * too easy to miss for a primary navigation. The two views are
 * functionally distinct enough to warrant explicit horizontal tabs at
 * the top. Active tab persists to localStorage.
 */
import { useEffect, useState } from 'react';
import V1 from './versions/V1';
import V2 from './versions/V2';

const TABS = [
  { id: 'overview', name: 'Health Overview', Component: V1 },
  { id: 'backlog', name: 'Divergence Backlog', Component: V2 },
] as const;

type TabId = (typeof TABS)[number]['id'];
const STORAGE_KEY = 'admin-strategy-health-tab';

export default function AdminStrategyHealthPage() {
  const [active, setActive] = useState<TabId>('overview');

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved && TABS.some(t => t.id === saved)) {
      setActive(saved as TabId);
    }
  }, []);

  const handleClick = (id: TabId) => {
    setActive(id);
    localStorage.setItem(STORAGE_KEY, id);
  };

  const ActiveComponent =
    TABS.find(t => t.id === active)?.Component ?? V1;

  return (
    <div>
      {/* Tab bar */}
      <div
        style={{
          display: 'flex',
          gap: 0,
          borderBottom: '1px solid var(--border)',
          padding: '0 16px',
          background: 'var(--bg-card)',
        }}
      >
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => handleClick(t.id)}
            style={{
              padding: '12px 18px',
              background: 'transparent',
              border: 'none',
              borderBottom: active === t.id
                ? '2px solid var(--accent, #4f8cf6)'
                : '2px solid transparent',
              marginBottom: -1,
              color: active === t.id ? 'var(--text-primary)' : 'var(--text-muted)',
              fontSize: 13,
              fontWeight: active === t.id ? 600 : 500,
              cursor: 'pointer',
            }}
          >
            {t.name}
          </button>
        ))}
      </div>
      <ActiveComponent />
    </div>
  );
}
