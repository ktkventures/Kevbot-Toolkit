'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import Modal from '@/components/Modal';

interface TimeframeConfig {
  tf: string;
  label: string;
  enabled: boolean;
  primary: boolean;
  packsUsing: number;
}

const mockTimeframes: TimeframeConfig[] = [
  { tf: '1Min', label: '1 Minute', enabled: true, primary: true, packsUsing: 4 },
  { tf: '5Min', label: '5 Minutes', enabled: true, primary: false, packsUsing: 3 },
  { tf: '15Min', label: '15 Minutes', enabled: true, primary: false, packsUsing: 2 },
  { tf: '1H', label: '1 Hour', enabled: false, primary: false, packsUsing: 0 },
  { tf: '4H', label: '4 Hours', enabled: false, primary: false, packsUsing: 0 },
  { tf: '1D', label: '1 Day', enabled: true, primary: false, packsUsing: 1 },
];

export default function TimeframesV1() {
  const [timeframes, setTimeframes] = useState<TimeframeConfig[]>(mockTimeframes);
  const [showPrimaryModal, setShowPrimaryModal] = useState<string | null>(null);

  const enabledCount = timeframes.filter((t) => t.enabled).length;

  function toggleTimeframe(tf: string) {
    setTimeframes((prev) =>
      prev.map((t) => {
        if (t.tf === tf) {
          // Don't allow disabling the primary
          if (t.primary) return t;
          return { ...t, enabled: !t.enabled };
        }
        return t;
      })
    );
  }

  function setPrimary(tf: string) {
    setTimeframes((prev) =>
      prev.map((t) => ({
        ...t,
        primary: t.tf === tf,
        enabled: t.tf === tf ? true : t.enabled,
      }))
    );
    setShowPrimaryModal(null);
  }

  return (
    <div>
      <PageHeader
        title="Timeframes"
        subtitle="Configure which timeframes are available for confluence analysis"
      />

      <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>
        {enabledCount} of {timeframes.length} timeframes enabled
      </p>

      {/* Timeframe grid */}
      <div className="grid grid-cols-3 gap-4">
        {timeframes.map((tf) => (
          <Card key={tf.tf}>
            <div className="flex items-start justify-between mb-3">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-lg font-bold" style={{ color: tf.enabled ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                    {tf.tf}
                  </span>
                  {tf.primary && (
                    <span
                      className="text-xs px-2 py-0.5 rounded font-medium"
                      style={{ color: 'var(--accent)', background: 'var(--accent-muted)' }}
                    >
                      Primary
                    </span>
                  )}
                </div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{tf.label}</p>
              </div>

              {/* Toggle */}
              <button
                onClick={() => toggleTimeframe(tf.tf)}
                className="w-10 h-6 rounded-full relative flex-shrink-0 transition-colors"
                style={{
                  background: tf.enabled ? 'var(--accent)' : 'var(--bg-input)',
                  border: tf.enabled ? 'none' : '1px solid var(--border)',
                  cursor: tf.primary ? 'not-allowed' : 'pointer',
                  opacity: tf.primary ? 0.7 : 1,
                }}
              >
                <div
                  className="w-4 h-4 rounded-full absolute top-1 transition-all"
                  style={{
                    background: tf.enabled ? 'white' : 'var(--text-muted)',
                    left: tf.enabled ? '22px' : '4px',
                  }}
                />
              </button>
            </div>

            {/* Pack usage info */}
            <div
              className="rounded-lg px-3 py-2 mb-3"
              style={{ background: 'var(--bg-input)' }}
            >
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                {tf.packsUsing > 0
                  ? `Used by ${tf.packsUsing} pack${tf.packsUsing > 1 ? 's' : ''}`
                  : 'Not used by any packs'}
              </p>
            </div>

            {/* Set as primary */}
            {!tf.primary && tf.enabled && (
              <button
                onClick={() => setShowPrimaryModal(tf.tf)}
                className="w-full px-3 py-1.5 rounded-lg text-xs"
                style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
              >
                Set as Primary
              </button>
            )}
            {tf.primary && (
              <p className="text-xs text-center py-1.5" style={{ color: 'var(--text-muted)' }}>
                Primary TF cannot be disabled
              </p>
            )}
            {!tf.primary && !tf.enabled && (
              <p className="text-xs text-center py-1.5" style={{ color: 'var(--text-muted)' }}>
                Enable to use in confluence
              </p>
            )}
          </Card>
        ))}
      </div>

      {/* Cross-TF confluence info */}
      <Card className="mt-6">
        <h3 className="text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Cross-Timeframe Confluence</h3>
        <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
          When multiple timeframes are enabled, secondary TF indicators are computed using the previous closed bar
          (not the forming bar) to avoid look-ahead bias. The primary timeframe drives trade execution.
        </p>
        <div className="flex gap-2">
          {timeframes.filter((t) => t.enabled).map((tf) => (
            <span
              key={tf.tf}
              className="text-xs px-2 py-1 rounded font-mono"
              style={{
                background: tf.primary ? 'var(--accent-muted)' : 'var(--bg-input)',
                color: tf.primary ? 'var(--accent)' : 'var(--text-secondary)',
                border: tf.primary ? '1px solid var(--accent)' : '1px solid var(--border)',
              }}
            >
              {tf.tf}{tf.primary ? ' (primary)' : ''}
            </span>
          ))}
        </div>
      </Card>

      {/* Set Primary Modal */}
      <Modal
        title="Change Primary Timeframe"
        isOpen={!!showPrimaryModal}
        onClose={() => setShowPrimaryModal(null)}
        width="420px"
      >
        <p className="text-sm mb-4" style={{ color: 'var(--text-secondary)' }}>
          Set <strong>{showPrimaryModal}</strong> as the primary timeframe? The primary timeframe drives trade execution
          and all other enabled timeframes become secondary confluence inputs.
        </p>
        <div className="flex gap-3">
          <button
            onClick={() => showPrimaryModal && setPrimary(showPrimaryModal)}
            className="px-4 py-2 rounded-lg text-sm font-medium flex-1"
            style={{ background: 'var(--accent)', color: 'white' }}
          >
            Set as Primary
          </button>
          <button
            onClick={() => setShowPrimaryModal(null)}
            className="px-4 py-2 rounded-lg text-sm flex-1"
            style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
          >
            Cancel
          </button>
        </div>
      </Modal>
    </div>
  );
}
