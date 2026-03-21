'use client';

import { useState, useEffect } from 'react';

interface ThemeOption {
  id: string;
  name: string;
  description: string;
  preview: string[];
  tier: 'light' | 'standard' | 'immersive';
}

const tierLabels = {
  light: { label: 'Light', color: 'var(--green)', description: 'Minimal effects, best performance' },
  standard: { label: 'Standard', color: 'var(--accent)', description: 'Subtle animations, smooth performance' },
  immersive: { label: 'Immersive', color: 'var(--purple)', description: 'Rich effects, may use more GPU' },
};

const themes: ThemeOption[] = [
  {
    id: 'dark',
    name: 'Dark',
    description: 'Clean, minimal dark theme',
    preview: ['#0a0a12', '#1a1a2e', '#2196F3'],
    tier: 'light',
  },
  {
    id: 'pipboy',
    name: 'Pip-Boy',
    description: 'Vault-Tec approved — terminal green with scanlines',
    preview: ['#0a0c08', '#10180c', '#30e868'],
    tier: 'standard',
  },
  {
    id: 'cyberpunk-lion',
    name: 'Cyberpunk Lion',
    description: 'Amber glow, electric accents — the lion hunts at night',
    preview: ['#0d0a08', '#261c0e', '#ff8c00'],
    tier: 'standard',
  },
  {
    id: 'crystal',
    name: 'Crystal Glass',
    description: 'Translucent panels, soft gradients — crystalline future',
    preview: ['#080818', 'rgba(25,25,55,0.6)', '#7c9df5'],
    tier: 'immersive',
  },
  {
    id: 'fortress',
    name: 'Fortress of Solitude',
    description: 'Brilliant ice, glass panels, arctic aurora — Kryptonian command center',
    preview: ['#e8f4ff', 'rgba(255,255,255,0.55)', '#1890ff'],
    tier: 'immersive',
  },
];

export default function ThemeSwitcher() {
  const [current, setCurrent] = useState('dark');

  useEffect(() => {
    const saved = localStorage.getItem('ror-theme') || 'dark';
    setCurrent(saved);
    document.documentElement.setAttribute('data-theme', saved);
  }, []);

  const setTheme = (themeId: string) => {
    setCurrent(themeId);
    document.documentElement.setAttribute('data-theme', themeId);
    localStorage.setItem('ror-theme', themeId);
  };

  const grouped = {
    light: themes.filter(t => t.tier === 'light'),
    standard: themes.filter(t => t.tier === 'standard'),
    immersive: themes.filter(t => t.tier === 'immersive'),
  };

  return (
    <div>
      {(['light', 'standard', 'immersive'] as const).map((tier) => (
        <div key={tier} className="mb-8 last:mb-0">
          <div className="flex items-center gap-2 mb-3">
            <span
              className="text-xs font-medium px-2 py-0.5 rounded-full"
              style={{ color: tierLabels[tier].color, background: tierLabels[tier].color + '18' }}
            >
              {tierLabels[tier].label}
            </span>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
              {tierLabels[tier].description}
            </span>
          </div>

          <div className="grid grid-cols-3 gap-4">
            {grouped[tier].map((theme) => (
              <button
                key={theme.id}
                onClick={() => setTheme(theme.id)}
                className="rounded-xl border p-4 text-left transition-all"
                style={{
                  borderColor: current === theme.id ? 'var(--accent)' : 'var(--border)',
                  background: current === theme.id ? 'var(--accent-muted)' : 'var(--bg-input)',
                  boxShadow: current === theme.id ? '0 0 0 1px var(--accent)' : 'none',
                }}
              >
                <div className="flex gap-2 mb-3">
                  {theme.preview.map((color, i) => (
                    <div
                      key={i}
                      className="w-6 h-6 rounded-full border"
                      style={{ background: color, borderColor: 'rgba(255,255,255,0.1)' }}
                    />
                  ))}
                </div>
                <p className="text-sm font-medium mb-0.5">{theme.name}</p>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{theme.description}</p>
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
