'use client';

import { useState, useEffect } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

/* ========================================================================= */
/* TYPES & DATA                                                               */
/* ========================================================================= */

interface ThemeDef {
  id: string;
  name: string;
  description: string;
  tier: 'light' | 'standard' | 'immersive';
  bg: string;
  card: string;
  sidebar: string;
  accent: string;
  green: string;
  red: string;
  textPrimary: string;
  textMuted: string;
}

const themes: ThemeDef[] = [
  { id: 'dark', name: 'Dark', description: 'Clean, minimal dark theme', tier: 'light', bg: '#0a0a12', card: '#1a1a2e', sidebar: '#12121e', accent: '#2196F3', green: '#16a34a', red: '#ef4444', textPrimary: '#e8e8f0', textMuted: '#6b7280' },
  { id: 'pipboy', name: 'Pip-Boy', description: 'Vault-Tec terminal green', tier: 'standard', bg: '#0a0c08', card: '#10180c', sidebar: '#0c1208', accent: '#30e868', green: '#30e868', red: '#ff4444', textPrimary: '#c0f0c8', textMuted: '#507050' },
  { id: 'cyberpunk-lion', name: 'Cyberpunk Lion', description: 'Amber glow, electric accents', tier: 'standard', bg: '#0d0a08', card: '#261c0e', sidebar: '#1a1208', accent: '#ff8c00', green: '#e5a813', red: '#ff4444', textPrimary: '#f0e0c0', textMuted: '#806840' },
  { id: 'crystal', name: 'Crystal Glass', description: 'Translucent panels, soft gradients', tier: 'immersive', bg: '#080818', card: '#191937', sidebar: '#101028', accent: '#7c9df5', green: '#4ade80', red: '#f87171', textPrimary: '#d0d8f0', textMuted: '#606880' },
  { id: 'fortress', name: 'Fortress of Solitude', description: 'Brilliant ice, glass panels', tier: 'immersive', bg: '#e8f4ff', card: '#ffffff', sidebar: '#d0e8f8', accent: '#1890ff', green: '#10b981', red: '#ef4444', textPrimary: '#1a1a2e', textMuted: '#6b7280' },
  { id: 'neon-tokyo', name: 'Neon Tokyo', description: 'Rain-soaked streets, neon signs', tier: 'immersive', bg: '#0a0010', card: '#1e0a32', sidebar: '#140820', accent: '#ff3090', green: '#00ff88', red: '#ff4060', textPrimary: '#e8d0f0', textMuted: '#806090' },
  { id: 'lightning', name: 'Lightning Storm', description: 'Electric flashes, Raiden terminal', tier: 'immersive', bg: '#06050f', card: '#14102a', sidebar: '#0e0c1e', accent: '#6080ff', green: '#40e0d0', red: '#ff6060', textPrimary: '#c8d0f0', textMuted: '#505880' },
  { id: 'kawaii', name: 'Kawaii Trader', description: 'Pastel pink, floating bubbles', tier: 'immersive', bg: '#fff0f8', card: '#ffffff', sidebar: '#ffe8f0', accent: '#ff60a0', green: '#80d080', red: '#ff6060', textPrimary: '#3a2030', textMuted: '#a080a0' },
  { id: 'tron', name: 'Tron Grid', description: 'Perspective grid, cyan glow', tier: 'immersive', bg: '#000008', card: '#000a1e', sidebar: '#000610', accent: '#00dcff', green: '#00ff80', red: '#ff4040', textPrimary: '#c0f0ff', textMuted: '#406070' },
];

/* ========================================================================= */
/* MINI DASHBOARD PREVIEW                                                     */
/* ========================================================================= */

function MiniDashboard({ theme, isCompare }: { theme: ThemeDef; isCompare?: boolean }) {
  return (
    <div
      className="rounded-xl overflow-hidden border"
      style={{
        background: theme.bg,
        borderColor: isCompare ? theme.accent : 'transparent',
        boxShadow: isCompare ? `0 0 20px ${theme.accent}30` : 'none',
        height: isCompare ? 200 : 160,
      }}
    >
      <div className="flex h-full">
        {/* Mini sidebar */}
        <div className="flex flex-col gap-1.5 p-2" style={{ background: theme.sidebar, width: 32 }}>
          {[0, 1, 2, 3, 4].map((i) => (
            <div
              key={i}
              className="rounded-sm"
              style={{
                height: 6,
                background: i === 0 ? theme.accent : `${theme.textMuted}40`,
              }}
            />
          ))}
        </div>

        {/* Main area */}
        <div className="flex-1 p-2 flex flex-col gap-1.5">
          {/* Header bar */}
          <div className="flex items-center gap-2">
            <div className="h-2.5 w-16 rounded-sm" style={{ background: theme.textPrimary }} />
            <div className="flex-1" />
            <div className="h-2 w-10 rounded-sm" style={{ background: `${theme.textMuted}60` }} />
          </div>

          {/* Metric cards row */}
          <div className="flex gap-1.5">
            {[theme.accent, theme.green, theme.red, theme.accent].map((color, i) => (
              <div
                key={i}
                className="flex-1 rounded-md p-1.5"
                style={{ background: theme.card, border: `1px solid ${theme.textMuted}20` }}
              >
                <div className="h-1.5 w-6 rounded-sm mb-1" style={{ background: `${theme.textMuted}60` }} />
                <div className="h-2 w-8 rounded-sm" style={{ background: color }} />
              </div>
            ))}
          </div>

          {/* Chart area */}
          <div
            className="flex-1 rounded-md relative overflow-hidden"
            style={{ background: theme.card, border: `1px solid ${theme.textMuted}20` }}
          >
            {/* Candlestick preview */}
            <svg className="w-full h-full" viewBox="0 0 120 40" preserveAspectRatio="none">
              {[
                { x: 8, o: 28, c: 18, h: 12, l: 32 },
                { x: 18, o: 18, c: 14, h: 10, l: 22 },
                { x: 28, o: 14, c: 22, h: 8, l: 26 },
                { x: 38, o: 22, c: 16, h: 12, l: 28 },
                { x: 48, o: 16, c: 10, h: 6, l: 20 },
                { x: 58, o: 10, c: 18, h: 6, l: 22 },
                { x: 68, o: 18, c: 24, h: 14, l: 28 },
                { x: 78, o: 24, c: 20, h: 16, l: 28 },
                { x: 88, o: 20, c: 12, h: 8, l: 24 },
                { x: 98, o: 12, c: 8, h: 4, l: 16 },
                { x: 108, o: 8, c: 14, h: 4, l: 18 },
              ].map((c, i) => {
                const bullish = c.c < c.o;
                const color = bullish ? theme.green : theme.red;
                const top = Math.min(c.o, c.c);
                const bodyH = Math.abs(c.o - c.c) || 1;
                return (
                  <g key={i}>
                    <line x1={c.x} y1={c.h} x2={c.x} y2={c.l} stroke={color} strokeWidth="0.8" />
                    <rect x={c.x - 3} y={top} width={6} height={bodyH} fill={color} />
                  </g>
                );
              })}
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ========================================================================= */
/* COMPONENT                                                                  */
/* ========================================================================= */

export default function SettingsThemesV4() {
  const [current, setCurrent] = useState('dark');
  const [compareId, setCompareId] = useState<string | null>(null);

  useEffect(() => {
    const saved = localStorage.getItem('ror-theme') || 'dark';
    setCurrent(saved);
  }, []);

  const setTheme = (themeId: string) => {
    setCurrent(themeId);
    document.documentElement.setAttribute('data-theme', themeId);
    localStorage.setItem('ror-theme', themeId);
  };

  const currentTheme = themes.find((t) => t.id === current) ?? themes[0];
  const compareTheme = compareId ? themes.find((t) => t.id === compareId) ?? null : null;

  return (
    <div>
      <PageHeader
        title="Theme Showcase"
        subtitle="Preview each theme with a live mini-dashboard, compare side-by-side, or dream up a custom theme"
      />

      {/* Side-by-side comparison (only when compare is selected) */}
      {compareTheme && (
        <Card className="mb-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold">Side-by-Side Comparison</h3>
            <button
              onClick={() => setCompareId(null)}
              className="text-xs px-2 py-1 rounded"
              style={{ background: 'var(--bg-input)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}
            >
              Close Comparison
            </button>
          </div>
          <div className="grid grid-cols-2 gap-6">
            <div>
              <p className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>
                Current: {currentTheme.name}
              </p>
              <MiniDashboard theme={currentTheme} isCompare />
            </div>
            <div>
              <p className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>
                Compare: {compareTheme.name}
              </p>
              <MiniDashboard theme={compareTheme} isCompare />
              <button
                onClick={() => { setTheme(compareTheme.id); setCompareId(null); }}
                className="mt-3 w-full px-3 py-2 rounded-lg text-sm font-medium"
                style={{ background: 'var(--accent)', color: 'white' }}
              >
                Switch to {compareTheme.name}
              </button>
            </div>
          </div>
        </Card>
      )}

      {/* Theme Gallery */}
      <div className="grid grid-cols-3 gap-5 mb-8">
        {themes.map((theme) => {
          const isActive = current === theme.id;
          return (
            <div key={theme.id} className="flex flex-col">
              <button
                onClick={() => setTheme(theme.id)}
                className="rounded-xl border text-left transition-all overflow-hidden"
                style={{
                  borderColor: isActive ? 'var(--accent)' : 'var(--border)',
                  background: 'var(--bg-card)',
                  boxShadow: isActive ? '0 0 0 2px var(--accent)' : 'none',
                }}
              >
                {/* Live mini dashboard preview */}
                <MiniDashboard theme={theme} />

                {/* Info */}
                <div className="p-3">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-semibold">{theme.name}</span>
                    {isActive && (
                      <span
                        className="text-xs px-2 py-0.5 rounded-full"
                        style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                      >
                        Active
                      </span>
                    )}
                  </div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    {theme.description}
                  </p>
                </div>
              </button>

              {/* Compare button */}
              {!isActive && (
                <button
                  onClick={() => setCompareId(theme.id)}
                  className="mt-2 text-xs py-1.5 rounded-lg"
                  style={{ background: 'var(--bg-input)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}
                >
                  Compare with current
                </button>
              )}
            </div>
          );
        })}
      </div>

      {/* Custom Theme Builder (Concept) */}
      <Card>
        <h3 className="text-sm font-semibold mb-1">Custom Theme Builder</h3>
        <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
          Design your own theme (coming soon)
        </p>
        <div className="grid grid-cols-2 gap-6">
          {/* Color pickers */}
          <div className="space-y-3">
            {[
              { label: 'Background', key: 'bg' },
              { label: 'Card', key: 'card' },
              { label: 'Sidebar', key: 'sidebar' },
              { label: 'Accent', key: 'accent' },
              { label: 'Bullish (Green)', key: 'green' },
              { label: 'Bearish (Red)', key: 'red' },
            ].map((field) => (
              <div key={field.key} className="flex items-center justify-between">
                <label className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                  {field.label}
                </label>
                <div className="flex items-center gap-2">
                  <input
                    type="color"
                    defaultValue={currentTheme[field.key as keyof ThemeDef] as string}
                    disabled
                    className="w-8 h-6 rounded border cursor-not-allowed opacity-50"
                    style={{ borderColor: 'var(--border)' }}
                  />
                  <span className="text-xs font-mono w-16" style={{ color: 'var(--text-muted)' }}>
                    {currentTheme[field.key as keyof ThemeDef] as string}
                  </span>
                </div>
              </div>
            ))}
          </div>

          {/* Live preview of custom colors */}
          <div>
            <p className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>
              Preview (based on current theme)
            </p>
            <MiniDashboard theme={currentTheme} />
          </div>
        </div>
      </Card>
    </div>
  );
}
