'use client';

import { useState, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import Modal from '@/components/Modal';

/* ========================================================================
   Types
   ======================================================================== */

interface PackUsage {
  packId: string;
  packName: string;
}

interface TimeframeConfig {
  tf: string;
  label: string;
  period: string;
  enabled: boolean;
  primary: boolean;
  packs: number;
  strategies: number;
  packList: PackUsage[];
}

/* ========================================================================
   Mock Data
   ======================================================================== */

const initialTimeframes: TimeframeConfig[] = [
  {
    tf: '1Min', label: '1 Minute', period: '390 bars/session', enabled: true, primary: true,
    packs: 4, strategies: 8,
    packList: [
      { packId: 'ema-1', packName: 'EMA Stack (Default)' },
      { packId: 'macd-1', packName: 'MACD Line (Default)' },
      { packId: 'vwap-1', packName: 'VWAP (Default)' },
      { packId: 'rvol-1', packName: 'RVOL (Default)' },
    ],
  },
  {
    tf: '5Min', label: '5 Minutes', period: '78 bars/session', enabled: true, primary: false,
    packs: 3, strategies: 5,
    packList: [
      { packId: 'ema-1', packName: 'EMA Stack (Default)' },
      { packId: 'macd-1', packName: 'MACD Line (Default)' },
      { packId: 'vwap-1', packName: 'VWAP (Default)' },
    ],
  },
  {
    tf: '15Min', label: '15 Minutes', period: '26 bars/session', enabled: true, primary: false,
    packs: 2, strategies: 3,
    packList: [
      { packId: 'ema-1', packName: 'EMA Stack (Default)' },
      { packId: 'macd-1', packName: 'MACD Line (Default)' },
    ],
  },
  {
    tf: '1H', label: '1 Hour', period: '6.5 bars/session', enabled: false, primary: false,
    packs: 0, strategies: 0, packList: [],
  },
  {
    tf: '4H', label: '4 Hours', period: '~1.6 bars/session', enabled: false, primary: false,
    packs: 0, strategies: 0, packList: [],
  },
  {
    tf: '1D', label: '1 Day', period: '1 bar/session', enabled: true, primary: false,
    packs: 2, strategies: 4,
    packList: [
      { packId: 'ema-1', packName: 'EMA Stack (Default)' },
      { packId: 'atr-1', packName: 'ATR (Default)' },
    ],
  },
];

/* ========================================================================
   Component
   ======================================================================== */

export default function TimeframesV2() {
  const [timeframes, setTimeframes] = useState<TimeframeConfig[]>(initialTimeframes);
  const [showPrimaryModal, setShowPrimaryModal] = useState<string | null>(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [showDisableWarning, setShowDisableWarning] = useState<string | null>(null);

  // Custom TF form state
  const [customTfValue, setCustomTfValue] = useState('');
  const [customTfUnit, setCustomTfUnit] = useState<'Min' | 'Hour'>('Min');

  const enabledTFs = useMemo(() => timeframes.filter((t) => t.enabled), [timeframes]);
  const primaryTF = useMemo(() => timeframes.find((t) => t.primary), [timeframes]);
  const secondaryTFs = useMemo(() => enabledTFs.filter((t) => !t.primary), [enabledTFs]);

  function attemptToggle(tf: string) {
    const target = timeframes.find((t) => t.tf === tf);
    if (!target) return;
    if (target.primary) return; // Cannot disable primary

    // If disabling and has strategies, show warning
    if (target.enabled && target.strategies > 0) {
      setShowDisableWarning(tf);
      return;
    }

    doToggle(tf);
  }

  function doToggle(tf: string) {
    setTimeframes((prev) =>
      prev.map((t) => (t.tf === tf && !t.primary ? { ...t, enabled: !t.enabled } : t))
    );
    setShowDisableWarning(null);
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

  function addCustomTF() {
    if (!customTfValue) return;
    const val = parseInt(customTfValue, 10);
    if (isNaN(val) || val <= 0) return;
    const tfCode = `${val}${customTfUnit}`;
    const labelUnit = customTfUnit === 'Min' ? 'Minute' : 'Hour';
    const label = `${val} ${labelUnit}${val > 1 ? 's' : ''}`;

    // Calculate bars per session (6.5h = 390 min)
    const totalMinutes = customTfUnit === 'Min' ? val : val * 60;
    const barsPerSession = 390 / totalMinutes;
    const period = barsPerSession >= 1
      ? `${Math.round(barsPerSession)} bars/session`
      : `~${barsPerSession.toFixed(1)} bars/session`;

    if (timeframes.find((t) => t.tf === tfCode)) return; // Already exists

    setTimeframes((prev) => [
      ...prev,
      {
        tf: tfCode, label, period,
        enabled: false, primary: false,
        packs: 0, strategies: 0, packList: [],
      },
    ]);
    setCustomTfValue('');
    setShowAddModal(false);
  }

  return (
    <div>
      <PageHeader
        title="Timeframes"
        subtitle="Configure which timeframes are available for confluence analysis and trade execution"
      />

      {/* Summary bar */}
      <div className="flex items-center justify-between mb-6">
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
          {enabledTFs.length} of {timeframes.length} timeframes enabled
          {primaryTF && <> &middot; Primary: <strong style={{ color: 'var(--accent)' }}>{primaryTF.tf}</strong></>}
        </p>
        <button
          onClick={() => setShowAddModal(true)}
          className="px-4 py-2 rounded-lg text-sm font-medium transition-opacity hover:opacity-80"
          style={{ background: 'var(--accent)', color: 'white' }}
        >
          + Add Custom Timeframe
        </button>
      </div>

      {/* ====== Timeframe Grid ====== */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        {timeframes.map((tf) => (
          <Card key={tf.tf}>
            {/* Header row: label + toggle */}
            <div className="flex items-start justify-between mb-3">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1 flex-wrap">
                  <span
                    className="text-lg font-bold font-mono"
                    style={{ color: tf.enabled ? 'var(--text-primary)' : 'var(--text-muted)' }}
                  >
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
                <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                  {tf.label}
                </p>
                <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
                  {tf.period}
                </p>
              </div>

              {/* Toggle switch */}
              <button
                onClick={() => attemptToggle(tf.tf)}
                className="w-11 h-6 rounded-full relative flex-shrink-0 transition-colors"
                style={{
                  background: tf.enabled ? 'var(--accent)' : 'var(--bg-input)',
                  border: tf.enabled ? 'none' : '1px solid var(--border)',
                  cursor: tf.primary ? 'not-allowed' : 'pointer',
                  opacity: tf.primary ? 0.7 : 1,
                }}
                title={tf.primary ? 'Primary TF cannot be disabled' : undefined}
              >
                <div
                  className="w-4 h-4 rounded-full absolute top-1 transition-all"
                  style={{
                    background: tf.enabled ? 'white' : 'var(--text-muted)',
                    left: tf.enabled ? '24px' : '4px',
                  }}
                />
              </button>
            </div>

            {/* Usage stats */}
            <div
              className="rounded-lg px-3 py-2.5 mb-3"
              style={{ background: 'var(--bg-input)' }}
            >
              {tf.packs > 0 || tf.strategies > 0 ? (
                <>
                  <p className="text-xs font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>
                    Used by {tf.packs} pack{tf.packs !== 1 ? 's' : ''}, {tf.strategies} strateg{tf.strategies !== 1 ? 'ies' : 'y'}
                  </p>
                  {/* Pack list */}
                  <div className="flex flex-wrap gap-1 mt-1.5">
                    {tf.packList.map((p) => (
                      <span
                        key={p.packId}
                        className="text-[10px] px-1.5 py-0.5 rounded"
                        style={{ background: 'var(--bg-card)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}
                      >
                        {p.packName}
                      </span>
                    ))}
                  </div>
                </>
              ) : (
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  Not used by any packs or strategies
                </p>
              )}
            </div>

            {/* Action area */}
            {!tf.primary && tf.enabled && (
              <button
                onClick={() => setShowPrimaryModal(tf.tf)}
                className="w-full px-3 py-2 rounded-lg text-xs font-medium transition-colors"
                style={{
                  background: 'transparent',
                  border: '1px solid var(--border)',
                  color: 'var(--text-secondary)',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = 'var(--accent)';
                  e.currentTarget.style.color = 'var(--accent)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = 'var(--border)';
                  e.currentTarget.style.color = 'var(--text-secondary)';
                }}
              >
                Set as Primary
              </button>
            )}
            {tf.primary && (
              <p className="text-xs text-center py-2" style={{ color: 'var(--text-muted)' }}>
                Primary TF drives trade execution
              </p>
            )}
            {!tf.primary && !tf.enabled && (
              <p className="text-xs text-center py-2" style={{ color: 'var(--text-muted)' }}>
                Enable to use in confluence analysis
              </p>
            )}
          </Card>
        ))}
      </div>

      {/* ====== Cross-TF Confluence Section ====== */}
      <Card>
        <h3 className="text-base font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
          Cross-Timeframe Confluence
        </h3>

        {/* Visual diagram */}
        <div
          className="rounded-lg p-5 mb-4"
          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
        >
          <div className="flex items-center justify-center gap-6 flex-wrap">
            {/* Primary TF */}
            {primaryTF && (
              <div className="text-center">
                <div
                  className="rounded-xl px-5 py-3 border-2 mb-1"
                  style={{ borderColor: 'var(--accent)', background: 'var(--accent-muted)' }}
                >
                  <p className="text-lg font-bold font-mono" style={{ color: 'var(--accent)' }}>
                    {primaryTF.tf}
                  </p>
                  <p className="text-[10px] font-medium" style={{ color: 'var(--accent)' }}>
                    PRIMARY
                  </p>
                </div>
                <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                  Drives execution
                </p>
              </div>
            )}

            {/* Arrow */}
            {secondaryTFs.length > 0 && (
              <div className="flex flex-col items-center">
                <svg width="60" height="24" viewBox="0 0 60 24" fill="none">
                  <path d="M0 12 H48 M48 12 L40 6 M48 12 L40 18" stroke="var(--text-muted)" strokeWidth="2" />
                </svg>
                <p className="text-[10px] mt-0.5" style={{ color: 'var(--text-muted)' }}>
                  confluence
                </p>
              </div>
            )}

            {/* Secondary TFs */}
            <div className="flex gap-2 flex-wrap">
              {secondaryTFs.map((tf) => (
                <div key={tf.tf} className="text-center">
                  <div
                    className="rounded-lg px-4 py-2.5 border mb-1"
                    style={{ borderColor: 'var(--border)', background: 'var(--bg-card)' }}
                  >
                    <p className="text-sm font-bold font-mono" style={{ color: 'var(--text-secondary)' }}>
                      {tf.tf}
                    </p>
                    <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                      Secondary
                    </p>
                  </div>
                </div>
              ))}
              {secondaryTFs.length === 0 && (
                <p className="text-xs py-3" style={{ color: 'var(--text-muted)' }}>
                  Enable additional timeframes for cross-TF confluence
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Explanation */}
        <div className="space-y-2">
          <div className="flex items-start gap-2">
            <span className="text-xs mt-0.5" style={{ color: 'var(--accent)' }}>1.</span>
            <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
              The <strong>primary timeframe</strong> drives trade execution and position management.
              All entries and exits are based on the primary TF bar schedule.
            </p>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-xs mt-0.5" style={{ color: 'var(--accent)' }}>2.</span>
            <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
              <strong>Secondary timeframes</strong> provide confluence signals. Their indicators are
              computed using the <em>previous closed bar</em> (shifted forward by one period) to
              avoid look-ahead bias.
            </p>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-xs mt-0.5" style={{ color: 'var(--accent)' }}>3.</span>
            <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
              Cross-TF confluence records follow the format: <code className="text-[10px] px-1 py-0.5 rounded" style={{ background: 'var(--bg-input)' }}>{'{TF}-{INTERPRETER}-{STATE}'}</code> (e.g., &quot;5MIN-EMA_STACK-SML&quot;).
            </p>
          </div>
        </div>

        {/* Active TF chips */}
        <div className="mt-4 pt-4 border-t" style={{ borderColor: 'var(--border)' }}>
          <p className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>
            Enabled Timeframes
          </p>
          <div className="flex gap-2 flex-wrap">
            {enabledTFs.map((tf) => (
              <span
                key={tf.tf}
                className="text-xs px-3 py-1.5 rounded-lg font-mono font-medium"
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
        </div>
      </Card>

      {/* ====== Set Primary Modal ====== */}
      <Modal
        title="Change Primary Timeframe"
        isOpen={!!showPrimaryModal}
        onClose={() => setShowPrimaryModal(null)}
        width="420px"
      >
        {showPrimaryModal && (() => {
          const target = timeframes.find((t) => t.tf === showPrimaryModal);
          return (
            <>
              <p className="text-sm mb-2" style={{ color: 'var(--text-secondary)' }}>
                Set <strong style={{ color: 'var(--text-primary)' }}>{showPrimaryModal}</strong> as the primary timeframe?
              </p>
              <div
                className="rounded-lg p-3 mb-4"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
              >
                <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>
                  This will change execution from:
                </p>
                <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                  <span style={{ color: 'var(--red)' }}>{primaryTF?.tf}</span>
                  {' '}&rarr;{' '}
                  <span style={{ color: 'var(--green)' }}>{showPrimaryModal}</span>
                </p>
                {target && target.strategies > 0 && (
                  <p className="text-xs mt-2" style={{ color: 'var(--yellow, #f59e0b)' }}>
                    Warning: {target.strategies} strategies use this timeframe and may need reconfiguration.
                  </p>
                )}
              </div>
              <div className="flex gap-3">
                <button
                  onClick={() => setPrimary(showPrimaryModal)}
                  className="px-4 py-2 rounded-lg text-sm font-medium flex-1 transition-opacity hover:opacity-80"
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
            </>
          );
        })()}
      </Modal>

      {/* ====== Add Custom Timeframe Modal ====== */}
      <Modal
        title="Add Custom Timeframe"
        isOpen={showAddModal}
        onClose={() => setShowAddModal(false)}
        width="420px"
      >
        <p className="text-sm mb-4" style={{ color: 'var(--text-secondary)' }}>
          Create a non-standard timeframe for confluence analysis.
        </p>

        <div className="flex gap-3 mb-4">
          <div className="flex-1">
            <label className="text-xs font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>
              Value
            </label>
            <input
              type="number"
              min={1}
              max={240}
              value={customTfValue}
              onChange={(e) => setCustomTfValue(e.target.value)}
              className="w-full px-3 py-2 rounded-lg text-sm"
              style={{
                background: 'var(--bg-input)',
                border: '1px solid var(--border)',
                color: 'var(--text-primary)',
              }}
              placeholder="e.g. 2"
            />
          </div>
          <div className="flex-1">
            <label className="text-xs font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>
              Unit
            </label>
            <div className="flex rounded-lg overflow-hidden border" style={{ borderColor: 'var(--border)' }}>
              {(['Min', 'Hour'] as const).map((unit) => (
                <button
                  key={unit}
                  onClick={() => setCustomTfUnit(unit)}
                  className="flex-1 px-3 py-2 text-sm font-medium transition-colors"
                  style={{
                    background: customTfUnit === unit ? 'var(--accent)' : 'var(--bg-input)',
                    color: customTfUnit === unit ? 'white' : 'var(--text-muted)',
                  }}
                >
                  {unit === 'Min' ? 'Minutes' : 'Hours'}
                </button>
              ))}
            </div>
          </div>
        </div>

        {customTfValue && (
          <div
            className="rounded-lg p-3 mb-4"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
          >
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
              Preview: <strong style={{ color: 'var(--text-primary)' }}>{customTfValue}{customTfUnit}</strong>
              {' '}&mdash;{' '}
              {(() => {
                const val = parseInt(customTfValue, 10);
                if (isNaN(val) || val <= 0) return 'Invalid';
                const totalMin = customTfUnit === 'Min' ? val : val * 60;
                const bps = 390 / totalMin;
                return bps >= 1
                  ? `${Math.round(bps)} bars per session`
                  : `~${bps.toFixed(1)} bars per session`;
              })()}
            </p>
            {timeframes.find((t) => t.tf === `${customTfValue}${customTfUnit}`) && (
              <p className="text-xs mt-1" style={{ color: 'var(--red)' }}>
                This timeframe already exists.
              </p>
            )}
          </div>
        )}

        <div className="flex gap-3">
          <button
            onClick={addCustomTF}
            disabled={!customTfValue || !!timeframes.find((t) => t.tf === `${customTfValue}${customTfUnit}`)}
            className="px-4 py-2 rounded-lg text-sm font-medium flex-1 transition-opacity hover:opacity-80 disabled:opacity-40"
            style={{ background: 'var(--accent)', color: 'white' }}
          >
            Add Timeframe
          </button>
          <button
            onClick={() => setShowAddModal(false)}
            className="px-4 py-2 rounded-lg text-sm flex-1"
            style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
          >
            Cancel
          </button>
        </div>
      </Modal>

      {/* ====== Disable Warning Modal ====== */}
      <Modal
        title="Disable Timeframe"
        isOpen={!!showDisableWarning}
        onClose={() => setShowDisableWarning(null)}
        width="420px"
      >
        {showDisableWarning && (() => {
          const target = timeframes.find((t) => t.tf === showDisableWarning);
          if (!target) return null;
          return (
            <>
              <div
                className="rounded-lg p-3 mb-4"
                style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)' }}
              >
                <p className="text-sm font-medium mb-1" style={{ color: 'var(--red)' }}>
                  Dependency Warning
                </p>
                <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                  Disabling <strong>{showDisableWarning}</strong> will affect:
                </p>
                <ul className="text-sm mt-2 space-y-1" style={{ color: 'var(--text-secondary)' }}>
                  <li>&bull; {target.strategies} strateg{target.strategies !== 1 ? 'ies' : 'y'} using this timeframe</li>
                  <li>&bull; {target.packs} confluence pack{target.packs !== 1 ? 's' : ''} configured for this TF</li>
                </ul>
                <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
                  Strategies using this TF will no longer receive confluence signals from it.
                </p>
              </div>
              <div className="flex gap-3">
                <button
                  onClick={() => doToggle(showDisableWarning)}
                  className="px-4 py-2 rounded-lg text-sm font-medium flex-1"
                  style={{ background: 'var(--red)', color: 'white' }}
                >
                  Disable Anyway
                </button>
                <button
                  onClick={() => setShowDisableWarning(null)}
                  className="px-4 py-2 rounded-lg text-sm flex-1"
                  style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                >
                  Cancel
                </button>
              </div>
            </>
          );
        })()}
      </Modal>
    </div>
  );
}
