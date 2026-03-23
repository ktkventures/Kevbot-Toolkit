'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import ChartPlaceholder from '@/components/ChartPlaceholder';

interface RiskPack {
  id: string;
  name: string;
  category: string;
  enabled: boolean;
  isDefault: boolean;
  params: string;
  outputs: string[];
}

const mockPacks: RiskPack[] = [
  { id: 'atr-stops', name: 'ATR Stops (Default)', category: 'Stops', enabled: true, isDefault: true, params: 'ATR Period: 14, Multiplier: 2.0', outputs: ['TIGHT','NORMAL','WIDE'] },
  { id: 'swing-stops', name: 'Swing Stops (Default)', category: 'Stops', enabled: true, isDefault: true, params: 'Lookback: 5', outputs: ['SWING_HIGH','SWING_LOW'] },
  { id: 'fixed-risk', name: 'Fixed Risk (Default)', category: 'Sizing', enabled: true, isDefault: true, params: 'Max Risk: 1.0%', outputs: [] },
];

const categoryColors: Record<string, { color: string; bg: string }> = {
  'Stops': { color: 'var(--red)', bg: 'var(--red-muted)' },
  'Sizing': { color: 'var(--green)', bg: 'var(--green-muted)' },
};

export default function RiskManagementV1() {
  const [packs, setPacks] = useState<RiskPack[]>(mockPacks);
  const [detailPack, setDetailPack] = useState<RiskPack | null>(null);

  const enabledCount = packs.filter((p) => p.enabled).length;

  function togglePack(id: string) {
    setPacks((prev) => prev.map((p) => p.id === id ? { ...p, enabled: !p.enabled } : p));
  }

  // Detail view
  if (detailPack) {
    return (
      <div>
        <PageHeader
          title={detailPack.name}
          subtitle={`${detailPack.category} pack`}
          backHref="#"
          actions={
            <button
              onClick={() => setDetailPack(null)}
              className="px-4 py-2 rounded-lg text-sm"
              style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
            >
              Back to Packs
            </button>
          }
        />

        <TabBar tabs={['Parameters', 'Outputs', 'Preview', 'Danger Zone']}>
          {(tab) => (
            <div>
              {tab === 'Parameters' && (
                <Card>
                  <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Risk Parameters</h3>
                  <div className="space-y-4">
                    {detailPack.params.split(', ').map((param) => {
                      const [label, value] = param.split(': ');
                      return (
                        <div key={label} className="flex items-center gap-4">
                          <label className="text-sm w-40" style={{ color: 'var(--text-secondary)' }}>{label.trim()}</label>
                          <input
                            className="px-3 py-2 rounded-lg text-sm flex-1"
                            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                            defaultValue={value?.trim() || ''}
                          />
                        </div>
                      );
                    })}
                  </div>

                  {/* Additional context for risk packs */}
                  <div className="mt-6 p-4 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                    <p className="text-xs font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>How this affects your strategies</p>
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                      {detailPack.category === 'Stops'
                        ? 'Stop placement determines exit levels for each trade. The output state (TIGHT/NORMAL/WIDE) is based on current volatility relative to historical range.'
                        : 'Position sizing controls how much capital is risked per trade. The max risk percentage is applied to your portfolio balance.'}
                    </p>
                  </div>

                  <div className="flex gap-3 mt-6">
                    <button className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: 'white' }}>
                      Save Changes
                    </button>
                    <button className="px-4 py-2 rounded-lg text-sm" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>
                      Reset to Default
                    </button>
                  </div>
                </Card>
              )}

              {tab === 'Outputs' && (
                <Card>
                  <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Output States</h3>
                  {detailPack.outputs.length > 0 ? (
                    <div className="space-y-2">
                      {detailPack.outputs.map((output) => (
                        <div
                          key={output}
                          className="flex items-center justify-between px-3 py-2 rounded-lg"
                          style={{ background: 'var(--bg-input)' }}
                        >
                          <span className="text-sm font-mono" style={{ color: 'var(--text-primary)' }}>{output}</span>
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                            {output === 'TIGHT' ? 'Low volatility' :
                             output === 'NORMAL' ? 'Average volatility' :
                             output === 'WIDE' ? 'High volatility' :
                             output === 'SWING_HIGH' ? 'Recent swing high level' :
                             output === 'SWING_LOW' ? 'Recent swing low level' : 'State'}
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                      This pack computes sizing values directly without discrete output states.
                    </p>
                  )}
                </Card>
              )}

              {tab === 'Preview' && (
                <div>
                  <div className="flex gap-3 mb-4">
                    <select
                      className="px-3 py-2 rounded-lg text-sm"
                      style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                    >
                      <option>NVDA</option>
                      <option>SPY</option>
                      <option>AAPL</option>
                    </select>
                    <select
                      className="px-3 py-2 rounded-lg text-sm"
                      style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                    >
                      <option>1Min</option>
                      <option>5Min</option>
                    </select>
                  </div>
                  <Card>
                    <ChartPlaceholder label={`Price chart with ${detailPack.name} levels overlay`} height={400} />
                  </Card>
                </div>
              )}

              {tab === 'Danger Zone' && (
                <Card>
                  <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--red)' }}>Danger Zone</h3>
                  <div className="space-y-6">
                    <div>
                      <label className="text-sm mb-2 block" style={{ color: 'var(--text-secondary)' }}>Rename Version</label>
                      <div className="flex gap-3">
                        <input
                          className="px-3 py-2 rounded-lg text-sm flex-1"
                          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                          defaultValue={detailPack.name}
                          disabled={detailPack.isDefault}
                        />
                        <button
                          className="px-4 py-2 rounded-lg text-sm"
                          style={{
                            background: detailPack.isDefault ? 'var(--bg-input)' : 'var(--bg-card)',
                            border: '1px solid var(--border)',
                            color: detailPack.isDefault ? 'var(--text-muted)' : 'var(--text-secondary)',
                            cursor: detailPack.isDefault ? 'not-allowed' : 'pointer',
                          }}
                          disabled={detailPack.isDefault}
                        >
                          Rename
                        </button>
                      </div>
                      {detailPack.isDefault && (
                        <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>Default packs cannot be renamed.</p>
                      )}
                    </div>

                    <div className="pt-4 border-t" style={{ borderColor: 'var(--border)' }}>
                      <p className="text-sm mb-2" style={{ color: 'var(--text-secondary)' }}>Delete this pack permanently.</p>
                      <button
                        className="px-4 py-2 rounded-lg text-sm font-medium"
                        style={{
                          background: detailPack.isDefault ? 'var(--bg-input)' : 'var(--red)',
                          color: detailPack.isDefault ? 'var(--text-muted)' : 'white',
                          cursor: detailPack.isDefault ? 'not-allowed' : 'pointer',
                        }}
                        disabled={detailPack.isDefault}
                      >
                        Delete Pack
                      </button>
                      {detailPack.isDefault && (
                        <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>Default packs cannot be deleted.</p>
                      )}
                    </div>
                  </div>
                </Card>
              )}
            </div>
          )}
        </TabBar>
      </div>
    );
  }

  // Main list view
  return (
    <div>
      <PageHeader
        title="Risk Management"
        subtitle="Position sizing, stop placement, and exposure rules"
      />

      <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>
        {packs.length} packs, {enabledCount} enabled
      </p>

      <div className="space-y-3">
        {packs.map((pack) => (
          <Card key={pack.id}>
            <div className="flex items-center gap-4">
              {/* Toggle */}
              <button
                onClick={() => togglePack(pack.id)}
                className="w-10 h-6 rounded-full relative flex-shrink-0 transition-colors"
                style={{
                  background: pack.enabled ? 'var(--accent)' : 'var(--bg-input)',
                  border: pack.enabled ? 'none' : '1px solid var(--border)',
                }}
              >
                <div
                  className="w-4 h-4 rounded-full absolute top-1 transition-all"
                  style={{
                    background: pack.enabled ? 'white' : 'var(--text-muted)',
                    left: pack.enabled ? '22px' : '4px',
                  }}
                />
              </button>

              {/* Info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-semibold text-sm">{pack.name}</span>
                  {pack.isDefault && (
                    <span className="text-xs px-1.5 py-0.5 rounded" style={{ color: 'var(--text-muted)', background: 'var(--bg-input)' }}>
                      default
                    </span>
                  )}
                  <span
                    className="text-xs px-2 py-0.5 rounded"
                    style={{
                      color: categoryColors[pack.category]?.color || 'var(--text-muted)',
                      background: categoryColors[pack.category]?.bg || 'var(--bg-input)',
                    }}
                  >
                    {pack.category}
                  </span>
                </div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  {pack.params}
                </p>
              </div>

              {/* Outputs preview */}
              <div className="flex items-center gap-1 flex-shrink-0">
                {pack.outputs.length > 0 ? (
                  pack.outputs.map((output) => (
                    <span
                      key={output}
                      className="text-xs font-mono px-1.5 py-0.5 rounded"
                      style={{ background: 'var(--bg-input)', color: 'var(--text-secondary)' }}
                    >
                      {output}
                    </span>
                  ))
                ) : (
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Scalar output</span>
                )}
              </div>

              {/* Actions */}
              <div className="flex gap-2 flex-shrink-0">
                <button
                  onClick={() => setDetailPack(pack)}
                  className="px-3 py-1.5 rounded text-xs"
                  style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                >
                  Details
                </button>
                <button
                  className="px-3 py-1.5 rounded text-xs"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                >
                  Copy
                </button>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
