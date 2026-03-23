'use client';

import { useState, useEffect } from 'react';

export interface VersionMeta {
  id: string;
  name: string;
  description: string;
  rationale: string;
}

interface VersionPanelProps {
  versions: VersionMeta[];
  activeVersion: string;
  onVersionChange: (id: string) => void;
}

const PANEL_WIDTH = 280;

export default function VersionPanel({ versions, activeVersion, onVersionChange }: VersionPanelProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [hoveredVersion, setHoveredVersion] = useState<string | null>(null);

  // Close on Escape
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setIsOpen(false);
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, []);

  return (
    <>
      {/* Tab handle — always visible on right edge */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="fixed top-1/2 -translate-y-1/2 flex items-center justify-center"
        style={{
          right: isOpen ? PANEL_WIDTH : 0,
          width: 28,
          height: 100,
          background: 'var(--bg-card)',
          border: '1px solid var(--border)',
          borderRight: isOpen ? '1px solid var(--border)' : 'none',
          borderTopLeftRadius: 8,
          borderBottomLeftRadius: 8,
          borderTopRightRadius: isOpen ? 0 : 0,
          borderBottomRightRadius: isOpen ? 0 : 0,
          color: 'var(--accent)',
          zIndex: 40,
          transition: 'right 0.25s ease',
          cursor: 'pointer',
          writingMode: 'vertical-lr',
          fontSize: 11,
          fontWeight: 600,
          letterSpacing: '0.05em',
        }}
      >
        {isOpen ? '>' : 'Versions'}
      </button>

      {/* Panel */}
      <aside
        className="fixed top-0 h-full border-l overflow-y-auto flex flex-col"
        style={{
          right: isOpen ? 0 : -PANEL_WIDTH,
          width: PANEL_WIDTH,
          background: 'var(--bg-secondary)',
          borderColor: 'var(--border)',
          zIndex: 39,
          transition: 'right 0.25s ease',
        }}
      >
        {/* Header */}
        <div className="px-4 py-4 border-b" style={{ borderColor: 'var(--border)' }}>
          <h2 className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>
            Page Versions
          </h2>
          <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
            Click to switch, hover for rationale
          </p>
        </div>

        {/* Version list */}
        <div className="flex-1 py-3 px-3 space-y-2">
          {versions.map((v, i) => {
            const isActive = v.id === activeVersion;
            const isHovered = v.id === hoveredVersion;

            return (
              <div key={v.id} className="relative">
                <button
                  onClick={() => onVersionChange(v.id)}
                  onMouseEnter={() => setHoveredVersion(v.id)}
                  onMouseLeave={() => setHoveredVersion(null)}
                  className="w-full text-left rounded-lg p-3 transition-all"
                  style={{
                    background: isActive ? 'var(--accent-muted)' : 'var(--bg-card)',
                    border: isActive
                      ? '1px solid var(--accent)'
                      : '1px solid var(--border)',
                    cursor: 'pointer',
                  }}
                >
                  {/* Version number + name */}
                  <div className="flex items-center gap-2 mb-1">
                    <span
                      className="text-xs font-bold px-1.5 py-0.5 rounded"
                      style={{
                        background: isActive ? 'var(--accent)' : 'var(--bg-input)',
                        color: isActive ? 'white' : 'var(--text-muted)',
                        minWidth: 22,
                        textAlign: 'center',
                      }}
                    >
                      V{i + 1}
                    </span>
                    <span
                      className="text-sm font-medium"
                      style={{ color: isActive ? 'var(--accent)' : 'var(--text-primary)' }}
                    >
                      {v.name}
                    </span>
                  </div>

                  {/* Short description */}
                  <p className="text-xs leading-relaxed" style={{ color: 'var(--text-muted)' }}>
                    {v.description}
                  </p>

                  {/* Active indicator */}
                  {isActive && (
                    <div className="flex items-center gap-1 mt-2">
                      <span
                        className="w-1.5 h-1.5 rounded-full"
                        style={{ background: 'var(--green)' }}
                      />
                      <span className="text-xs" style={{ color: 'var(--green)' }}>
                        Active
                      </span>
                    </div>
                  )}
                </button>

                {/* Rationale tooltip — shows on hover, positioned to the left */}
                {isHovered && !isActive && v.rationale && (
                  <div
                    className="absolute top-0 rounded-lg p-3 border shadow-lg"
                    style={{
                      right: '100%',
                      marginRight: 8,
                      width: 260,
                      background: 'var(--bg-card)',
                      borderColor: 'var(--border)',
                      zIndex: 50,
                    }}
                  >
                    <p className="text-xs font-semibold mb-1" style={{ color: 'var(--accent)' }}>
                      Design Rationale
                    </p>
                    <p className="text-xs leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                      {v.rationale}
                    </p>
                  </div>
                )}

                {/* Rationale always visible for active version */}
                {isActive && v.rationale && (
                  <div
                    className="mt-2 rounded-lg p-2.5"
                    style={{ background: 'var(--bg-input)' }}
                  >
                    <p className="text-xs font-semibold mb-1" style={{ color: 'var(--accent)' }}>
                      Rationale
                    </p>
                    <p className="text-xs leading-relaxed" style={{ color: 'var(--text-muted)' }}>
                      {v.rationale}
                    </p>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Footer */}
        <div className="px-4 py-3 border-t" style={{ borderColor: 'var(--border)' }}>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Locked versions are preserved. Active version is saved per page.
          </p>
        </div>
      </aside>
    </>
  );
}
