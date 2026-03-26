'use client';

/**
 * Connection Topology — Faithful copy of V5 design with mock data replaced by API hooks.
 * Source: src/app/settings/connections/versions/V5.tsx (426 lines)
 *
 * Data convention:
 *   Real value = live from API
 *   -- = wired but no data yet
 *   {{field}} = not wired, needs backend work
 */

import { useState, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import { useMonitorStatus, useEngineState } from '@/hooks/queries/useAlerts';
import { useStartMonitor, useStopMonitor } from '@/hooks/mutations/useAlertMutations';

/* ========================================================================= */
/* TYPES & DATA                                                               */
/* ========================================================================= */

interface ServiceNode {
  id: string;
  name: string;
  type: 'app' | 'data' | 'broker' | 'infra';
  status: 'connected' | 'disconnected' | 'error';
  latency: string;
  uptime: string;
  lastChecked: string;
  history: { time: string; status: 'ok' | 'error' | 'slow' }[];
}

/** Placeholder history — no dedicated health-check endpoints yet */
const placeholderHistory: { time: string; status: 'ok' | 'error' | 'slow' }[] = Array.from({ length: 12 }, (_, i) => ({
  time: `${String(Math.floor(i / 12 + 10)).padStart(2, '0')}:${String((i * 5) % 60).padStart(2, '0')}`,
  status: 'ok' as const,
}));

/** Build service nodes — Polygon/Supabase/Railway use {{status}} placeholders; Alpaca is always disconnected (no integration) */
function buildNodes(): ServiceNode[] {
  return [
    {
      id: 'polygon',
      name: 'Polygon.io',
      type: 'data',
      status: 'connected',   // Placeholder — no health endpoint
      latency: '{{latency}}',
      uptime: '{{uptime}}',
      lastChecked: '{{lastChecked}}',
      history: placeholderHistory,
    },
    {
      id: 'alpaca',
      name: 'Alpaca',
      type: 'broker',
      status: 'disconnected',  // Not integrated
      latency: '--',
      uptime: '--',
      lastChecked: '--',
      history: placeholderHistory.map((h) => ({ ...h, status: 'error' as const })),
    },
    {
      id: 'supabase',
      name: 'Supabase',
      type: 'infra',
      status: 'connected',   // Placeholder — no health endpoint
      latency: '{{latency}}',
      uptime: '{{uptime}}',
      lastChecked: '{{lastChecked}}',
      history: placeholderHistory,
    },
    {
      id: 'railway',
      name: 'Railway',
      type: 'infra',
      status: 'connected',   // Placeholder — no health endpoint
      latency: '{{latency}}',
      uptime: '{{uptime}}',
      lastChecked: '{{lastChecked}}',
      history: placeholderHistory,
    },
  ];
}

/* ========================================================================= */
/* HELPERS                                                                    */
/* ========================================================================= */

function nodeColor(status: ServiceNode['status']): string {
  if (status === 'connected') return 'var(--green)';
  if (status === 'error') return 'var(--red)';
  return 'var(--text-muted)';
}

function historyDotColor(status: 'ok' | 'error' | 'slow'): string {
  if (status === 'ok') return 'var(--green)';
  if (status === 'slow') return 'var(--yellow, #e5a813)';
  return 'var(--red)';
}

function typeIcon(type: ServiceNode['type']): string {
  if (type === 'data') return 'D';
  if (type === 'broker') return 'B';
  if (type === 'infra') return 'I';
  return 'A';
}

/* ========================================================================= */
/* COMPONENT                                                                  */
/* ========================================================================= */

export default function SettingsConnectionsPage() {
  /* ---- API hooks (must be before any early return) ---- */
  const { data: monitorStatus } = useMonitorStatus();
  const { data: engineState } = useEngineState();
  const startMonitor = useStartMonitor();
  const stopMonitor = useStopMonitor();

  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [diagnosing, setDiagnosing] = useState(false);

  const nodes = useMemo(() => buildNodes(), []);
  const selected = nodes.find((n) => n.id === selectedNode) ?? null;

  /* ---- Derive engine status from API ---- */
  const engineRunning = monitorStatus?.desired_state === 'running' || monitorStatus?.status === 'running';
  const engineStatusLabel = engineRunning ? 'Running' : 'Stopped';
  const engineStatusColor = engineRunning ? 'var(--green)' : 'var(--text-muted)';
  const engineUptime = monitorStatus?.uptime ?? '--';
  const engineSymbols = engineState?.symbols ? Object.keys(engineState.symbols).length : '--';
  const engineStrategies = monitorStatus?.active_strategies ?? '--';
  const engineLastUpdate = monitorStatus?.last_update
    ? new Date(monitorStatus.last_update).toLocaleTimeString()
    : '--';
  const engineTicks = monitorStatus?.ticks ?? '--';

  function handleDiagnose() {
    setDiagnosing(true);
    setTimeout(() => setDiagnosing(false), 3000);
  }

  return (
    <div>
      <PageHeader
        title="Connection Topology"
        subtitle="Network diagram showing all service connections with real-time health monitoring"
      />

      {/* Network Topology Visualization */}
      <Card className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold">Network Map</h3>
          <div className="flex items-center gap-4 text-xs" style={{ color: 'var(--text-muted)' }}>
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full" style={{ background: 'var(--green)' }} /> Connected
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full" style={{ background: 'var(--yellow, #e5a813)' }} /> Slow
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full" style={{ background: 'var(--red)' }} /> Error
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full" style={{ background: 'var(--text-muted)' }} /> Disconnected
            </span>
          </div>
        </div>

        {/* SVG topology diagram */}
        <div className="relative" style={{ height: 200 }}>
          <svg className="w-full h-full" viewBox="0 0 800 200">
            {/* Center hub: RoR Trader */}
            <circle cx="400" cy="100" r="35" fill="var(--bg-input)" stroke="var(--accent)" strokeWidth="2" />
            <text x="400" y="96" textAnchor="middle" fill="var(--accent)" fontSize="10" fontWeight="bold">RoR</text>
            <text x="400" y="110" textAnchor="middle" fill="var(--accent)" fontSize="8">Trader</text>

            {/* Connection lines to nodes */}
            {[
              { node: nodes[0], x: 120, y: 60 },
              { node: nodes[1], x: 120, y: 150 },
              { node: nodes[2], x: 680, y: 60 },
              { node: nodes[3], x: 680, y: 150 },
            ].map(({ node, x, y }) => (
              <g key={node.id}>
                {/* Connection line */}
                <line
                  x1="400"
                  y1="100"
                  x2={x}
                  y2={y}
                  stroke={nodeColor(node.status)}
                  strokeWidth="1.5"
                  strokeDasharray={node.status === 'disconnected' ? '4,4' : 'none'}
                  opacity={node.status === 'disconnected' ? 0.4 : 0.6}
                />

                {/* Latency label on line */}
                {node.latency !== '--' && (
                  <text
                    x={(400 + x) / 2}
                    y={(100 + y) / 2 - 6}
                    textAnchor="middle"
                    fill="var(--text-muted)"
                    fontSize="9"
                  >
                    {node.latency}
                  </text>
                )}

                {/* Node circle */}
                <circle
                  cx={x}
                  cy={y}
                  r="28"
                  fill="var(--bg-input)"
                  stroke={nodeColor(node.status)}
                  strokeWidth="2"
                  style={{ cursor: 'pointer' }}
                  onClick={() => setSelectedNode(node.id === selectedNode ? null : node.id)}
                />

                {/* Type badge */}
                <circle cx={x + 20} cy={y - 20} r="8" fill={nodeColor(node.status)} />
                <text x={x + 20} y={y - 17} textAnchor="middle" fill="white" fontSize="8" fontWeight="bold">
                  {typeIcon(node.type)}
                </text>

                {/* Node label */}
                <text
                  x={x}
                  y={y + 4}
                  textAnchor="middle"
                  fill="var(--text-primary)"
                  fontSize="10"
                  fontWeight="600"
                  style={{ cursor: 'pointer' }}
                  onClick={() => setSelectedNode(node.id === selectedNode ? null : node.id)}
                >
                  {node.name}
                </text>
              </g>
            ))}
          </svg>
        </div>
      </Card>

      {/* Selected node detail + history timeline */}
      {selected && (
        <Card className="mb-6">
          <div className="flex items-start justify-between mb-4">
            <div>
              <div className="flex items-center gap-3 mb-1">
                <h3 className="font-semibold">{selected.name}</h3>
                <span
                  className="text-xs px-2 py-0.5 rounded-full"
                  style={{ color: nodeColor(selected.status), background: `${nodeColor(selected.status)}18` }}
                >
                  {selected.status === 'connected' ? 'Connected' : selected.status === 'error' ? 'Error' : 'Disconnected'}
                </span>
              </div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Last checked: {selected.lastChecked}</p>
            </div>
            <div className="flex gap-2">
              {selected.status === 'disconnected' && (
                <button
                  onClick={handleDiagnose}
                  className="px-3 py-1.5 rounded-lg text-xs font-medium"
                  style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                  disabled={diagnosing}
                >
                  {diagnosing ? 'Diagnosing...' : 'Auto-Diagnose'}
                </button>
              )}
              <button
                onClick={() => setSelectedNode(null)}
                className="px-3 py-1.5 rounded-lg text-xs"
                style={{ background: 'var(--bg-input)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}
              >
                Close
              </button>
            </div>
          </div>

          {/* Stats row */}
          <div className="grid grid-cols-3 gap-4 mb-4">
            <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Latency</p>
              <p className="text-lg font-semibold">{selected.latency}</p>
            </div>
            <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Uptime</p>
              <p className="text-lg font-semibold">{selected.uptime}</p>
            </div>
            <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Errors (1h)</p>
              <p className="text-lg font-semibold">{selected.history.filter((h) => h.status === 'error').length}</p>
            </div>
          </div>

          {/* Connection history timeline */}
          <div>
            <h4 className="text-xs font-medium mb-3" style={{ color: 'var(--text-muted)' }}>
              Connection History (last hour)
            </h4>
            <div className="flex items-end gap-1">
              {selected.history.map((h, i) => (
                <div key={i} className="flex flex-col items-center flex-1">
                  <div
                    className="w-full rounded-sm"
                    style={{
                      height: h.status === 'ok' ? 24 : h.status === 'slow' ? 16 : 8,
                      background: historyDotColor(h.status),
                      opacity: 0.8,
                    }}
                  />
                  <span className="text-xs mt-1 font-mono" style={{ color: 'var(--text-muted)', fontSize: '0.55rem' }}>
                    {h.time.split(':')[1]}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Diagnosis output */}
          {diagnosing && (
            <div
              className="mt-4 rounded-lg p-3 font-mono text-xs space-y-1"
              style={{ background: '#000', color: 'var(--green)', border: '1px solid var(--border)' }}
            >
              <p>$ diagnosing {selected.name.toLowerCase()}...</p>
              <p>  DNS resolution: OK (2ms)</p>
              <p>  TCP handshake: OK (8ms)</p>
              <p>  TLS negotiation: OK (15ms)</p>
              <p style={{ color: 'var(--red)' }}>  API authentication: FAILED</p>
              <p style={{ color: 'var(--yellow, #e5a813)' }}>  Suggestion: Check API key validity and permissions</p>
            </div>
          )}
        </Card>
      )}

      {/* Quick status grid (when no node selected) */}
      {!selected && (
        <div className="grid grid-cols-2 gap-4">
          {nodes.map((node) => (
            <button key={node.id} onClick={() => setSelectedNode(node.id)} className="text-left w-full">
              <Card className="cursor-pointer">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span
                      className="w-3 h-3 rounded-full"
                      style={{
                        background: nodeColor(node.status),
                        boxShadow: node.status === 'connected' ? `0 0 6px ${nodeColor(node.status)}` : 'none',
                      }}
                    />
                    <span className="text-sm font-semibold">{node.name}</span>
                  </div>
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{node.latency}</span>
                </div>
                {/* Mini history bar */}
                <div className="flex gap-0.5">
                  {node.history.map((h, i) => (
                    <div
                      key={i}
                      className="flex-1 h-1.5 rounded-full"
                      style={{ background: historyDotColor(h.status), opacity: 0.7 }}
                    />
                  ))}
                </div>
                <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
                  Uptime: {node.uptime}
                </p>
              </Card>
            </button>
          ))}
        </div>
      )}

      {/* ---- Alert Engine Status (Admin) ---- */}
      <Card className="mt-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <h3 className="text-sm font-semibold">Alert Engine (Ralph)</h3>
            <span className="text-xs px-2 py-0.5 rounded-full" style={{ color: engineStatusColor, background: engineStatusColor + '20' }}>
              {engineStatusLabel}
            </span>
          </div>
          <div className="flex gap-2">
            {!engineRunning ? (
              <button
                onClick={() => startMonitor.mutate()}
                disabled={startMonitor.isPending}
                className="px-3 py-1.5 rounded-lg text-xs font-medium"
                style={{ background: 'var(--green-muted)', color: 'var(--green)' }}
              >
                {startMonitor.isPending ? 'Starting...' : 'Enable Monitoring'}
              </button>
            ) : (
              <button
                onClick={() => stopMonitor.mutate()}
                disabled={stopMonitor.isPending}
                className="px-3 py-1.5 rounded-lg text-xs font-medium"
                style={{ background: 'var(--red-muted)', color: 'var(--red)' }}
              >
                {stopMonitor.isPending ? 'Stopping...' : 'Disable Monitoring'}
              </button>
            )}
          </div>
        </div>

        <div className="grid grid-cols-5 gap-4 mb-4">
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Uptime</p>
            <p className="text-sm font-bold">{engineUptime}</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Ticks</p>
            <p className="text-sm font-bold font-mono">{engineTicks}</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Symbols</p>
            <p className="text-sm font-bold">{engineSymbols}</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Active Strategies</p>
            <p className="text-sm font-bold">{engineStrategies}</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Last Update</p>
            <p className="text-sm font-bold font-mono">{engineLastUpdate}</p>
          </div>
        </div>

        {/* Monitoring Logs — no log endpoint yet, placeholder */}
        <div className="rounded-lg overflow-hidden" style={{ background: '#0d1117', maxHeight: 200, overflowY: 'auto' }}>
          <div className="p-3 font-mono text-xs space-y-1">
            {engineRunning ? (
              <div className="flex gap-2">
                <span style={{ color: '#484f58' }}>--:--:--</span>
                <span style={{ color: '#58a6ff', fontWeight: 600 }}>[INFO]</span>
                <span style={{ color: '#c9d1d9' }}>Engine is running. Log streaming not yet wired.</span>
              </div>
            ) : (
              <div className="flex gap-2">
                <span style={{ color: '#484f58' }}>--:--:--</span>
                <span style={{ color: '#484f58', fontWeight: 600 }}>[INFO]</span>
                <span style={{ color: '#8b949e' }}>Engine is stopped. Enable monitoring to start.</span>
              </div>
            )}
          </div>
        </div>
      </Card>
    </div>
  );
}
