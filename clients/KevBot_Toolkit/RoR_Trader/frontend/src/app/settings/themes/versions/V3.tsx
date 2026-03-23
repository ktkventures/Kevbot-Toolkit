'use client';

import { useState, useEffect } from 'react';
import Card from '@/components/Card';

/* ========================================================================= */
/* DATA                                                                       */
/* ========================================================================= */

const themes = [
  { id: 'dark', name: 'Dark' },
  { id: 'pipboy', name: 'Pip-Boy' },
  { id: 'cyberpunk-lion', name: 'Cyberpunk Lion' },
  { id: 'crystal', name: 'Crystal Glass' },
  { id: 'fortress', name: 'Fortress of Solitude' },
  { id: 'neon-tokyo', name: 'Neon Tokyo' },
  { id: 'lightning', name: 'Lightning Storm' },
  { id: 'kawaii', name: 'Kawaii Trader' },
  { id: 'tron', name: 'Tron Grid' },
];

/* ========================================================================= */
/* COMPONENT                                                                  */
/* ========================================================================= */

export default function SettingsThemesV3() {
  const [current, setCurrent] = useState('dark');

  useEffect(() => {
    const saved = localStorage.getItem('ror-theme') || 'dark';
    setCurrent(saved);
  }, []);

  const setTheme = (themeId: string) => {
    setCurrent(themeId);
    document.documentElement.setAttribute('data-theme', themeId);
    localStorage.setItem('ror-theme', themeId);
  };

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Themes</h1>

      <div className="max-w-md">
        <Card>
          <div className="space-y-0">
            {themes.map((theme, i) => (
              <label
                key={theme.id}
                className="flex items-center gap-3 py-3 cursor-pointer border-b last:border-0"
                style={{ borderColor: 'var(--border)' }}
              >
                <input
                  type="radio"
                  name="theme"
                  checked={current === theme.id}
                  onChange={() => setTheme(theme.id)}
                  className="accent-[var(--accent)]"
                  style={{ width: 16, height: 16 }}
                />
                <span
                  className="text-sm font-medium"
                  style={{
                    color: current === theme.id ? 'var(--accent)' : 'var(--text-primary)',
                  }}
                >
                  {theme.name}
                </span>
                {current === theme.id && (
                  <span
                    className="text-xs px-2 py-0.5 rounded-full ml-auto"
                    style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                  >
                    Active
                  </span>
                )}
              </label>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}
