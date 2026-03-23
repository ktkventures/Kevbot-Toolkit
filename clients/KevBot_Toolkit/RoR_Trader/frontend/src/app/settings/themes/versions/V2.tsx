'use client';

import { useState, useEffect } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

/* ========================================================================= */
/* TYPES & DATA                                                               */
/* ========================================================================= */

interface ThemeOption {
  id: string;
  name: string;
  description: string;
  swatches: string[];
  tier: 'light' | 'standard' | 'immersive';
  bgPreview: string;
  cardPreview: string;
  accentPreview: string;
}

const tierLabels: Record<string, { label: string; color: string; description: string }> = {
  light: { label: 'Light', color: 'var(--green)', description: 'Minimal effects, best performance' },
  standard: { label: 'Standard', color: 'var(--accent)', description: 'Subtle animations, smooth performance' },
  immersive: { label: 'Immersive', color: 'var(--purple, #a855f7)', description: 'Rich effects, may use more GPU' },
};

const themes: ThemeOption[] = [
  { id: 'dark', name: 'Dark', description: 'Clean, minimal dark theme', swatches: ['#0a0a12', '#1a1a2e', '#2196F3', '#16a34a', '#ef4444'], tier: 'light', bgPreview: '#0a0a12', cardPreview: '#1a1a2e', accentPreview: '#2196F3' },
  { id: 'pipboy', name: 'Pip-Boy', description: 'Vault-Tec approved — terminal green with scanlines', swatches: ['#0a0c08', '#10180c', '#30e868', '#1a3018', '#ff4444'], tier: 'standard', bgPreview: '#0a0c08', cardPreview: '#10180c', accentPreview: '#30e868' },
  { id: 'cyberpunk-lion', name: 'Cyberpunk Lion', description: 'Amber glow, electric accents — the lion hunts at night', swatches: ['#0d0a08', '#261c0e', '#ff8c00', '#e5a813', '#ff4444'], tier: 'standard', bgPreview: '#0d0a08', cardPreview: '#261c0e', accentPreview: '#ff8c00' },
  { id: 'crystal', name: 'Crystal Glass', description: 'Translucent panels, soft gradients — crystalline future', swatches: ['#080818', '#191937', '#7c9df5', '#4ade80', '#f87171'], tier: 'immersive', bgPreview: '#080818', cardPreview: '#191937', accentPreview: '#7c9df5' },
  { id: 'fortress', name: 'Fortress of Solitude', description: 'Brilliant ice, glass panels, arctic aurora — Kryptonian command center', swatches: ['#e8f4ff', '#ffffff', '#1890ff', '#10b981', '#ef4444'], tier: 'immersive', bgPreview: '#e8f4ff', cardPreview: '#ffffff', accentPreview: '#1890ff' },
  { id: 'neon-tokyo', name: 'Neon Tokyo', description: 'Rain-soaked streets, neon signs, hot pink — trading from Akihabara', swatches: ['#0a0010', '#1e0a32', '#ff3090', '#00ff88', '#ff4060'], tier: 'immersive', bgPreview: '#0a0010', cardPreview: '#1e0a32', accentPreview: '#ff3090' },
  { id: 'lightning', name: 'Lightning Storm', description: 'Random strikes, electric flashes — Raiden approves this terminal', swatches: ['#06050f', '#14102a', '#6080ff', '#40e0d0', '#ff6060'], tier: 'immersive', bgPreview: '#06050f', cardPreview: '#14102a', accentPreview: '#6080ff' },
  { id: 'kawaii', name: 'Kawaii Trader', description: 'Pastel pink, floating bubbles, sparkles — aggressively cute', swatches: ['#fff0f8', '#ffffff', '#ff60a0', '#80d080', '#ff6060'], tier: 'immersive', bgPreview: '#fff0f8', cardPreview: '#ffffff', accentPreview: '#ff60a0' },
  { id: 'tron', name: 'Tron Grid', description: 'Perspective grid, cyan glow, scan lines — inside the Grid', swatches: ['#000008', '#000a1e', '#00dcff', '#00ff80', '#ff4040'], tier: 'immersive', bgPreview: '#000008', cardPreview: '#000a1e', accentPreview: '#00dcff' },
];

/* ========================================================================= */
/* COMPONENT                                                                  */
/* ========================================================================= */

export default function SettingsThemesV2() {
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

  const grouped = {
    light: themes.filter((t) => t.tier === 'light'),
    standard: themes.filter((t) => t.tier === 'standard'),
    immersive: themes.filter((t) => t.tier === 'immersive'),
  };

  const currentTheme = themes.find((t) => t.id === current) ?? themes[0];

  return (
    <div>
      <PageHeader title="Themes" subtitle="Choose a visual theme for your trading dashboard" />

      {/* Current Theme Banner */}
      <Card className="mb-6">
        <div className="flex items-center gap-4">
          {/* Animated preview background */}
          <div
            className="w-24 h-24 rounded-xl flex-shrink-0 relative overflow-hidden"
            style={{
              background: `linear-gradient(135deg, ${currentTheme.bgPreview}, ${currentTheme.cardPreview}, ${currentTheme.accentPreview}20)`,
              border: `2px solid ${currentTheme.accentPreview}`,
            }}
          >
            {/* Mini card preview */}
            <div
              className="absolute top-3 left-3 right-3 h-6 rounded-md"
              style={{ background: currentTheme.cardPreview, border: `1px solid ${currentTheme.accentPreview}40` }}
            />
            {/* Mini bar chart */}
            <div className="absolute bottom-2 left-3 right-3 flex items-end gap-1 h-8">
              {[4, 7, 3, 8, 5, 6].map((h, i) => (
                <div
                  key={i}
                  className="flex-1 rounded-t-sm"
                  style={{
                    height: `${h * 3.5}px`,
                    background: i % 2 === 0 ? currentTheme.accentPreview : `${currentTheme.accentPreview}60`,
                  }}
                />
              ))}
            </div>
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-1">
              <h2 className="text-lg font-semibold">{currentTheme.name}</h2>
              <span
                className="text-xs px-2 py-0.5 rounded-full font-medium"
                style={{
                  color: tierLabels[currentTheme.tier].color,
                  background: `${tierLabels[currentTheme.tier].color}18`,
                }}
              >
                {tierLabels[currentTheme.tier].label}
              </span>
            </div>
            <p className="text-sm mb-2" style={{ color: 'var(--text-muted)' }}>
              {currentTheme.description}
            </p>
            <div className="flex gap-1.5">
              {currentTheme.swatches.map((color, i) => (
                <div
                  key={i}
                  className="w-5 h-5 rounded-full border"
                  style={{ background: color, borderColor: 'rgba(128,128,128,0.3)' }}
                />
              ))}
            </div>
          </div>
        </div>
      </Card>

      {/* Tier Sections */}
      {(['light', 'standard', 'immersive'] as const).map((tier) => (
        <div key={tier} className="mb-8 last:mb-0">
          <div className="flex items-center gap-2 mb-4">
            <span
              className="text-xs font-medium px-2 py-0.5 rounded-full"
              style={{ color: tierLabels[tier].color, background: `${tierLabels[tier].color}18` }}
            >
              {tierLabels[tier].label}
            </span>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
              {tierLabels[tier].description}
            </span>
          </div>

          <div className="grid grid-cols-3 gap-4">
            {grouped[tier].map((theme) => {
              const isActive = current === theme.id;
              return (
                <button
                  key={theme.id}
                  onClick={() => setTheme(theme.id)}
                  className="rounded-xl border text-left transition-all overflow-hidden"
                  style={{
                    borderColor: isActive ? 'var(--accent)' : 'var(--border)',
                    background: isActive ? 'var(--accent-muted)' : 'var(--bg-card)',
                    boxShadow: isActive ? '0 0 0 1px var(--accent), 0 0 20px var(--accent)30' : 'none',
                  }}
                >
                  {/* Preview area */}
                  <div
                    className="h-20 relative"
                    style={{
                      background: `linear-gradient(135deg, ${theme.bgPreview}, ${theme.cardPreview})`,
                    }}
                  >
                    {/* Mini UI mockup */}
                    <div className="absolute inset-2 flex gap-1.5">
                      <div className="w-4 rounded-sm" style={{ background: `${theme.accentPreview}30` }} />
                      <div className="flex-1 flex flex-col gap-1">
                        <div className="h-3 rounded-sm" style={{ background: theme.cardPreview, border: `1px solid ${theme.accentPreview}20` }} />
                        <div className="flex-1 rounded-sm" style={{ background: theme.cardPreview, border: `1px solid ${theme.accentPreview}20` }} />
                      </div>
                    </div>
                    {/* Active indicator */}
                    {isActive && (
                      <div
                        className="absolute top-2 right-2 w-5 h-5 rounded-full flex items-center justify-center text-xs"
                        style={{ background: 'var(--accent)', color: 'white' }}
                      >
                        &#10003;
                      </div>
                    )}
                  </div>

                  {/* Info */}
                  <div className="p-3">
                    <div className="flex gap-1.5 mb-2">
                      {theme.swatches.slice(0, 5).map((color, i) => (
                        <div
                          key={i}
                          className="w-4 h-4 rounded-full border"
                          style={{ background: color, borderColor: 'rgba(128,128,128,0.2)' }}
                        />
                      ))}
                    </div>
                    <p className="text-sm font-medium mb-0.5">{theme.name}</p>
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                      {theme.description}
                    </p>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
