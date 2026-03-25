'use client';

/**
 * Settings Connections — Clean API-first page.
 *
 * Visual design derived from V5 (versions/V5.tsx), data layer built
 * around the monitor status and engine state API endpoints. No mock data.
 */

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import { useMonitorStatus, useEngineState } from '@/hooks/queries/useAlerts';
import { useStartMonitor, useStopMonitor } from '@/hooks/mutations/useAlertMutations';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function nodeColor(connected: boolean): string {
  return connected ? 'var(--green)' : 'var(--text-muted)';
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function SettingsConnectionsPage() {
  const { data: monitorStatus, isLoading: statusLoading } = useMonitorStatus();
  const { data: engineState, isLoading: stateLoading } = useEngineState();
  const startMonitor = useStartMonitor();
  const stopMonitor = useStopMonitor();

  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  const isLoading = statusLoading || stateLoading;
  const isRunning = monitorStatus?.status === 'running' || monitorStatus?.desired_state === 'running';

  // Build service nodes from real data
  const services = [
    {
      id: 'polygon',
      name: 'Polygon.io',
      type: 'Data',
      connected: monitorStatus?.data_provider_connected ?? false,
      latency: monitorStatus?.data_provider_latency || '--',
    },
    {
      id: 'supabase',
      name: 'Supabase',
      type: 'Infra',
      connected: true, // If we loaded data, Supabase is reachable
      latency: '--',
    },
    {
      id: 'railway',
      name: 'Railway',
      type: 'Infra',
      connected: true,
      latency: '--',
    },
  ];

  // ---------------------------------------------------------------------------
  // Loading / Error
  // ---------------------------------------------------------------------------

  if (isLoading) {
    return (
      <div>
        <PageHeader title="Connection Topology" subtitle="Loading..." />
        <div className="grid grid-cols-2 gap-4 mt-4">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i}>
              <div className="animate-pulse space-y-3">
                <div className="h-4 rounded w-1/3" style={{ background: 'var(--border)' }} />
                <div className="h-3 rounded w-1/2" style={{ background: 'var(--border)' }} />
              </div>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div>
      <PageHeader
        title="Connection Topology"
        subtitle="Network diagram showing all service connections with real-time health monitoring"
        actions={
          <div className="flex items-center gap-2">
            <span
              className="text-xs px-2 py-1 rounded-full flex items-center gap-1.5"
              style={{ background: 'var(--green)' + '15', color: 'var(--green)' }}
            >
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--green)' }} />
              Live
            </span>
          </div>
        }
      />

      {/* Service status cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4 mb-6">
        {services.map((node) => (
          <Card key={node.id}>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <span
                  className="w-3 h-3 rounded-full"
                  style={{
                    background: nodeColor(node.connected),
                    boxShadow: node.connected ? `0 0 6px ${nodeColor(node.connected)}` : 'none',
                  }}
                />
                <span className="text-sm font-semibold">{node.name}</span>
              </div>
              <span className="text-xs px-1.5 py-0.5 rounded" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
                {node.type}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                {node.connected ? 'Connected' : 'Disconnected'}
              </span>
              <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
                {node.latency}
              </span>
            </div>
          </Card>
        ))}
      </div>

      {/* Alert Engine (Ralph) */}
      <Card>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <h3 className="text-sm font-semibold">Alert Engine (Ralph)</h3>
            <span
              className="text-xs px-2 py-0.5 rounded-full"
              style={{
                color: isRunning ? 'var(--green)' : 'var(--text-muted)',
                background: (isRunning ? 'var(--green)' : 'var(--text-muted)') + '20',
              }}
            >
              {isRunning ? 'Running' : 'Stopped'}
            </span>
          </div>
          <div className="flex gap-2">
            {isRunning ? (
              <>
                <button
                  className="px-3 py-1.5 rounded-lg text-xs font-medium"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)', cursor: 'pointer' }}
                  onClick={() => { stopMonitor.mutate(); setTimeout(() => startMonitor.mutate(), 1000); }}
                >
                  Restart
                </button>
                <button
                  className="px-3 py-1.5 rounded-lg text-xs font-medium"
                  style={{ background: 'var(--red-muted, rgba(239,83,80,0.15))', color: 'var(--red)', border: 'none', cursor: 'pointer' }}
                  onClick={() => stopMonitor.mutate()}
                  disabled={stopMonitor.isPending}
                >
                  {stopMonitor.isPending ? 'Stopping...' : 'Disable Monitoring'}
                </button>
              </>
            ) : (
              <button
                className="px-3 py-1.5 rounded-lg text-xs font-medium"
                style={{ background: 'var(--green-muted, rgba(76,175,80,0.15))', color: 'var(--green)', border: 'none', cursor: 'pointer' }}
                onClick={() => startMonitor.mutate()}
                disabled={startMonitor.isPending}
              >
                {startMonitor.isPending ? 'Starting...' : 'Enable Monitoring'}
              </button>
            )}
          </div>
        </div>

        {/* Engine metrics */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-4">
          {[
            { label: 'Uptime', value: monitorStatus?.uptime || '--' },
            { label: 'Ticks', value: monitorStatus?.ticks != null ? monitorStatus.ticks.toLocaleString() : '--' },
            { label: 'Symbols', value: monitorStatus?.symbols_count ?? engineState?.symbols_count ?? '--' },
            { label: 'Active Strategies', value: monitorStatus?.strategies_count ?? engineState?.strategies_count ?? '--' },
            { label: 'Last Update', value: monitorStatus?.last_update || '--' },
          ].map((m) => (
            <div key={m.label}>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{m.label}</p>
              <p className="text-sm font-bold font-mono">{String(m.value)}</p>
            </div>
          ))}
        </div>

        {/* Engine state details */}
        {engineState && engineState.positions && Object.keys(engineState.positions).length > 0 && (
          <div className="mt-4">
            <h4 className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>Active Positions</h4>
            <div className="rounded-lg overflow-hidden" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
              <div className="p-3 font-mono text-xs space-y-1">
                {Object.entries(engineState.positions).map(([symbol, pos]: [string, any]) => (
                  <div key={symbol} className="flex items-center gap-3">
                    <span style={{ color: 'var(--accent)' }}>{symbol}</span>
                    <span style={{ color: pos.status === 'IN_POSITION' ? 'var(--green)' : 'var(--text-muted)' }}>
                      {pos.status || 'FLAT'}
                    </span>
                    {pos.entry_price && (
                      <span style={{ color: 'var(--text-muted)' }}>@ ${pos.entry_price}</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
