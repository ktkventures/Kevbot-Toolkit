'use client';

import { useState } from 'react';
import PageHeader from '@/components/PageHeader';

/* ---------- Types ---------- */
interface TimeframeConfig {
  tf: string;
  label: string;
  enabled: boolean;
  primary: boolean;
}

/* ---------- Mock Data ---------- */
const mockTimeframes: TimeframeConfig[] = [
  { tf: '1Min', label: '1 Minute', enabled: true, primary: true },
  { tf: '5Min', label: '5 Minutes', enabled: true, primary: false },
  { tf: '15Min', label: '15 Minutes', enabled: true, primary: false },
  { tf: '1H', label: '1 Hour', enabled: false, primary: false },
  { tf: '4H', label: '4 Hours', enabled: false, primary: false },
  { tf: '1D', label: '1 Day', enabled: true, primary: false },
];

/* ---------- Main Component ---------- */
export default function TimeframesV3() {
  const [timeframes, setTimeframes] = useState<TimeframeConfig[]>(mockTimeframes);

  const enabledCount = timeframes.filter((t) => t.enabled).length;
  const primaryTf = timeframes.find((t) => t.primary);

  function toggleTimeframe(tf: string) {
    setTimeframes((prev) =>
      prev.map((t) => {
        if (t.tf === tf) {
          if (t.primary) return t; // Cannot disable primary
          return { ...t, enabled: !t.enabled };
        }
        return t;
      }),
    );
  }

  function setPrimary(tf: string) {
    setTimeframes((prev) =>
      prev.map((t) => ({
        ...t,
        primary: t.tf === tf,
        enabled: t.tf === tf ? true : t.enabled,
      })),
    );
  }

  return (
    <div>
      <PageHeader
        title="Timeframes"
        subtitle={`${enabledCount} of ${timeframes.length} enabled${primaryTf ? ` \u00b7 Primary: ${primaryTf.tf}` : ''}`}
      />

      {/* Compact horizontal rows */}
      <div className="space-y-1">
        {timeframes.map((tf) => (
          <div
            key={tf.tf}
            className="flex items-center gap-3 px-4 py-2.5 rounded-xl border transition-all"
            style={{
              background: 'var(--bg-card)',
              borderColor: tf.primary ? 'var(--accent)' : 'var(--border)',
              boxShadow: 'var(--card-shadow)',
              backdropFilter: 'var(--card-backdrop)',
              WebkitBackdropFilter: 'var(--card-backdrop)',
              opacity: tf.enabled ? 1 : 0.5,
            }}
          >
            {/* TF label */}
            <span
              className="text-sm font-bold font-mono w-14 flex-shrink-0"
              style={{ color: tf.enabled ? 'var(--text-primary)' : 'var(--text-muted)' }}
            >
              {tf.tf}
            </span>

            {/* Full label */}
            <span className="text-xs flex-1" style={{ color: 'var(--text-muted)' }}>
              {tf.label}
            </span>

            {/* Toggle switch */}
            <button
              onClick={() => toggleTimeframe(tf.tf)}
              className="w-9 h-5 rounded-full relative flex-shrink-0 transition-colors"
              style={{
                background: tf.enabled ? 'var(--accent)' : 'var(--bg-input)',
                border: tf.enabled ? 'none' : '1px solid var(--border)',
                cursor: tf.primary ? 'not-allowed' : 'pointer',
                opacity: tf.primary ? 0.7 : 1,
              }}
              title={tf.primary ? 'Primary TF cannot be disabled' : undefined}
            >
              <div
                className="w-3.5 h-3.5 rounded-full absolute transition-all"
                style={{
                  background: tf.enabled ? 'white' : 'var(--text-muted)',
                  top: '3px',
                  left: tf.enabled ? '19px' : '3px',
                }}
              />
            </button>

            {/* Primary badge or Set Primary button */}
            {tf.primary ? (
              <span
                className="text-xs px-2 py-0.5 rounded font-medium flex-shrink-0"
                style={{ color: 'var(--accent)', background: 'var(--accent-muted)' }}
              >
                Primary
              </span>
            ) : tf.enabled ? (
              <button
                onClick={() => setPrimary(tf.tf)}
                className="text-xs px-2 py-0.5 rounded flex-shrink-0 transition-colors"
                style={{ color: 'var(--text-muted)', background: 'transparent', border: '1px solid var(--border)' }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = 'var(--accent)';
                  e.currentTarget.style.color = 'var(--accent)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = 'var(--border)';
                  e.currentTarget.style.color = 'var(--text-muted)';
                }}
              >
                Set Primary
              </button>
            ) : (
              <span className="w-[86px] flex-shrink-0" />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
