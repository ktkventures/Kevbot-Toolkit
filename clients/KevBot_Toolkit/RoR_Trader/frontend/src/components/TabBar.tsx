'use client';

import { useState } from 'react';

interface TabBarProps {
  tabs: string[];
  defaultTab?: number;
  children: (activeTab: string) => React.ReactNode;
}

export default function TabBar({ tabs, defaultTab = 0, children }: TabBarProps) {
  const [active, setActive] = useState(tabs[defaultTab]);

  return (
    <div>
      <div className="flex gap-1 border-b mb-6" style={{ borderColor: 'var(--border)' }}>
        {tabs.map((tab) => (
          <button
            key={tab}
            onClick={() => setActive(tab)}
            className="px-4 py-2.5 text-sm transition-colors relative"
            style={{
              color: active === tab ? 'var(--accent)' : 'var(--text-muted)',
              borderBottom: active === tab ? '2px solid var(--accent)' : '2px solid transparent',
              marginBottom: '-1px',
            }}
          >
            {tab}
          </button>
        ))}
      </div>
      {children(active)}
    </div>
  );
}
