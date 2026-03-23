'use client';

import { useState, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import MetricCard from '@/components/MetricCard';
import Modal from '@/components/Modal';

/* ========================================================================= */
/* TYPES                                                                      */
/* ========================================================================= */

interface MonitorStatus {
  status: 'running' | 'stopped' | 'error';
  uptime: string;
  ticks: number;
  symbols: number;
  lastUpdate: string;
}

interface LastSignal {
  type: 'ENTRY' | 'EXIT';
  time: string;
  price: string;
}

interface MonitoredStrategy {
  id: string;
  name: string;
  symbol: string;
  tf: string;
  direction: 'LONG' | 'SHORT';
  alertEntry: boolean;
  alertExit: boolean;
  webhook: boolean;
  totalAlerts: number;
  todayAlerts: number;
  position: 'FLAT' | 'IN_POSITION';
  lastSignal: LastSignal;
}

interface AlertRecord {
  time: string;
  type: 'ENTRY' | 'EXIT';
  strategy: string;
  price: string;
  execType: string;
  pnl: string | null;
}

interface OpenPosition {
  strategy: string;
  symbol: string;
  entryPrice: string;
  currentPrice: string;
  pnl: string;
  pnlPct: string;
  duration: string;
  health: 'Healthy' | 'Warning' | 'Critical';
}

/* ========================================================================= */
/* MOCK DATA                                                                  */
/* ========================================================================= */

const mockMonitorStatus: MonitorStatus = {
  status: 'running',
  uptime: '4h 32m',
  ticks: 12847,
  symbols: 4,
  lastUpdate: '10:32:15 AM',
};

const mockMonitoredStrategies: MonitoredStrategy[] = [
  { id: '1', name: 'NVDA LONG - Mass #2', symbol: 'NVDA', tf: '1Min', direction: 'LONG', alertEntry: true, alertExit: true, webhook: true, totalAlerts: 47, todayAlerts: 3, position: 'FLAT', lastSignal: { type: 'EXIT', time: '10:15 AM', price: '$487.23' } },
  { id: '2', name: 'SPY LONG - Mass #1', symbol: 'SPY', tf: '5Min', direction: 'LONG', alertEntry: true, alertExit: false, webhook: true, totalAlerts: 23, todayAlerts: 1, position: 'IN_POSITION', lastSignal: { type: 'ENTRY', time: '09:45 AM', price: '$562.10' } },
  { id: '3', name: 'AAPL LONG - Mass #5', symbol: 'AAPL', tf: '1Min', direction: 'LONG', alertEntry: true, alertExit: true, webhook: false, totalAlerts: 12, todayAlerts: 0, position: 'FLAT', lastSignal: { type: 'EXIT', time: 'Yesterday', price: '$192.45' } },
  { id: '4', name: 'TSLA LONG - Mass #5', symbol: 'TSLA', tf: '1Min', direction: 'LONG', alertEntry: true, alertExit: true, webhook: true, totalAlerts: 31, todayAlerts: 2, position: 'FLAT', lastSignal: { type: 'ENTRY', time: '10:30 AM', price: '$178.90' } },
];

const mockAlerts: AlertRecord[] = [
  { time: '10:32 AM', type: 'ENTRY', strategy: 'NVDA LONG - Mass #2', price: '$487.50', execType: '[C]', pnl: null },
  { time: '10:15 AM', type: 'EXIT', strategy: 'NVDA LONG - Mass #2', price: '$487.23', execType: '[C]', pnl: '+1.2R' },
  { time: '09:45 AM', type: 'ENTRY', strategy: 'SPY LONG - Mass #1', price: '$562.10', execType: '[L0]', pnl: null },
  { time: '09:31 AM', type: 'ENTRY', strategy: 'TSLA LONG - Mass #5', price: '$178.90', execType: '[HM]', pnl: null },
  { time: 'Yesterday', type: 'EXIT', strategy: 'AAPL LONG - Mass #5', price: '$192.45', execType: '[C]', pnl: '-0.8R' },
  { time: 'Yesterday', type: 'ENTRY', strategy: 'AAPL LONG - Mass #5', price: '$191.80', execType: '[L1]', pnl: null },
  { time: 'Yesterday', type: 'EXIT', strategy: 'TSLA LONG - Mass #5', price: '$176.50', execType: '[HL]', pnl: '+2.1R' },
  { time: '2 days ago', type: 'ENTRY', strategy: 'NVDA LONG - Mass #2', price: '$482.10', execType: '[C]', pnl: null },
  { time: '2 days ago', type: 'EXIT', strategy: 'SPY LONG - Mass #1', price: '$558.90', execType: '[C]', pnl: '+0.5R' },
];

const mockOpenPositions: OpenPosition[] = [
  { strategy: 'SPY LONG - Mass #1', symbol: 'SPY', entryPrice: '$562.10', currentPrice: '$563.45', pnl: '+$1.35', pnlPct: '+0.24%', duration: '47m', health: 'Healthy' },
];

/* ========================================================================= */
/* HELPERS                                                                    */
/* ========================================================================= */

function statusColor(status: MonitorStatus['status']): string {
  if (status === 'running') return 'var(--green)';
  if (status === 'error') return 'var(--red)';
  return 'var(--text-muted)';
}

function statusLabel(status: MonitorStatus['status']): string {
  if (status === 'running') return 'Running';
  if (status === 'error') return 'Error';
  return 'Stopped';
}

function healthColor(health: OpenPosition['health']): string {
  if (health === 'Healthy') return 'var(--green)';
  if (health === 'Warning') return 'var(--yellow, #e5a813)';
  return 'var(--red)';
}

/* ========================================================================= */
/* COMPONENT                                                                  */
/* ========================================================================= */

export default function AlertsV2() {
  const [monitor, setMonitor] = useState<MonitorStatus>(mockMonitorStatus);
  const [strategies, setStrategies] = useState<MonitoredStrategy[]>(mockMonitoredStrategies);
  const [configModalId, setConfigModalId] = useState<string | null>(null);
  const [filterStrategy, setFilterStrategy] = useState('All');
  const [filterSignalType, setFilterSignalType] = useState<'All' | 'ENTRY' | 'EXIT'>('All');
  const [searchQuery, setSearchQuery] = useState('');

  const configStrategy = useMemo(
    () => strategies.find((s) => s.id === configModalId) ?? null,
    [strategies, configModalId]
  );

  // Toggle monitor
  function handleToggleMonitor() {
    setMonitor((prev) => ({
      ...prev,
      status: prev.status === 'running' ? 'stopped' : 'running',
      ticks: prev.status === 'running' ? prev.ticks : 0,
      uptime: prev.status === 'running' ? '0m' : prev.uptime,
    }));
  }

  // Toggle strategy-level booleans
  function toggleField(id: string, field: 'alertEntry' | 'alertExit' | 'webhook') {
    setStrategies((prev) =>
      prev.map((s) => (s.id === id ? { ...s, [field]: !s[field] } : s))
    );
  }

  // Filtered alerts
  const filteredAlerts = useMemo(() => {
    return mockAlerts.filter((a) => {
      if (filterStrategy !== 'All' && a.strategy !== filterStrategy) return false;
      if (filterSignalType !== 'All' && a.type !== filterSignalType) return false;
      if (searchQuery && !a.strategy.toLowerCase().includes(searchQuery.toLowerCase())) return false;
      return true;
    });
  }, [filterStrategy, filterSignalType, searchQuery]);

  const todayCount = mockAlerts.filter((a) => a.time !== 'Yesterday' && !a.time.includes('days ago')).length;
  const entryCount = mockAlerts.filter((a) => a.type === 'ENTRY').length;
  const exitCount = mockAlerts.filter((a) => a.type === 'EXIT').length;

  return (
    <div>
      <PageHeader
        title="Alerts & Signals"
        subtitle="Real-time monitoring, position tracking, and alert management"
        actions={
          <button
            onClick={handleToggleMonitor}
            className="px-4 py-2 rounded-lg text-sm font-medium"
            style={{
              background: monitor.status === 'running' ? 'var(--red-muted)' : 'var(--green-muted)',
              color: monitor.status === 'running' ? 'var(--red)' : 'var(--green)',
            }}
          >
            {monitor.status === 'running' ? 'Disable Monitoring' : 'Enable Monitoring'}
          </button>
        }
      />

      {/* ── Monitor Status Card ── */}
      <Card className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <span
              className="w-3 h-3 rounded-full"
              style={{
                background: statusColor(monitor.status),
                boxShadow: monitor.status === 'running' ? `0 0 8px ${statusColor(monitor.status)}` : 'none',
              }}
            />
            <div>
              <p className="font-semibold text-base">
                Engine: {statusLabel(monitor.status)}
              </p>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                Last update: {monitor.lastUpdate}
              </p>
            </div>
          </div>
        </div>
        <div className="grid grid-cols-4 gap-4">
          <MetricCard label="Uptime" value={monitor.uptime} />
          <MetricCard label="Ticks Processed" value={monitor.ticks.toLocaleString()} />
          <MetricCard label="Symbols" value={monitor.symbols.toString()} />
          <MetricCard label="Active Strategies" value={strategies.length.toString()} />
        </div>
      </Card>

      {/* ── Tabbed Content ── */}
      <TabBar tabs={['Monitored Strategies', 'Recent Alerts', 'Open Positions', 'History']}>
        {(activeTab) => {
          /* ────────────────────── Monitored Strategies ────────────────────── */
          if (activeTab === 'Monitored Strategies') {
            return (
              <div className="space-y-3">
                {strategies.map((strat) => (
                  <Card key={strat.id}>
                    <div className="flex items-start justify-between gap-4">
                      {/* Left: info */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1 flex-wrap">
                          <h3 className="font-semibold text-sm truncate">{strat.name}</h3>
                          <span
                            className="text-xs px-2 py-0.5 rounded"
                            style={{ background: 'var(--bg-input)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}
                          >
                            {strat.symbol}
                          </span>
                          <span
                            className="text-xs px-2 py-0.5 rounded"
                            style={{ background: 'var(--bg-input)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}
                          >
                            {strat.tf}
                          </span>
                          <span
                            className="text-xs px-1.5 py-0.5 rounded font-medium"
                            style={{
                              background: strat.direction === 'LONG' ? 'var(--green-muted)' : 'var(--red-muted)',
                              color: strat.direction === 'LONG' ? 'var(--green)' : 'var(--red)',
                            }}
                          >
                            {strat.direction}
                          </span>
                        </div>

                        {/* Toggles row */}
                        <div className="flex items-center gap-4 mt-2 mb-2">
                          <label className="flex items-center gap-1.5 text-xs cursor-pointer" style={{ color: 'var(--text-secondary)' }}>
                            <input
                              type="checkbox"
                              checked={strat.alertEntry}
                              onChange={() => toggleField(strat.id, 'alertEntry')}
                              className="accent-[var(--accent)]"
                            />
                            Alert Entry
                          </label>
                          <label className="flex items-center gap-1.5 text-xs cursor-pointer" style={{ color: 'var(--text-secondary)' }}>
                            <input
                              type="checkbox"
                              checked={strat.alertExit}
                              onChange={() => toggleField(strat.id, 'alertExit')}
                              className="accent-[var(--accent)]"
                            />
                            Alert Exit
                          </label>
                          <label className="flex items-center gap-1.5 text-xs cursor-pointer" style={{ color: 'var(--text-secondary)' }}>
                            <input
                              type="checkbox"
                              checked={strat.webhook}
                              onChange={() => toggleField(strat.id, 'webhook')}
                              className="accent-[var(--accent)]"
                            />
                            Webhook
                          </label>
                        </div>

                        {/* Stats row */}
                        <div className="flex items-center gap-4 text-xs" style={{ color: 'var(--text-muted)' }}>
                          <span>{strat.totalAlerts} total alerts</span>
                          <span>{strat.todayAlerts} today</span>
                          <span>
                            Last: {strat.lastSignal.type} at {strat.lastSignal.price} ({strat.lastSignal.time})
                          </span>
                        </div>
                      </div>

                      {/* Right: position badge + configure */}
                      <div className="flex flex-col items-end gap-2 shrink-0">
                        <span
                          className="text-xs px-2.5 py-1 rounded font-medium"
                          style={{
                            background: strat.position === 'IN_POSITION' ? 'var(--green-muted)' : 'var(--bg-input)',
                            color: strat.position === 'IN_POSITION' ? 'var(--green)' : 'var(--text-muted)',
                            border: strat.position === 'IN_POSITION' ? 'none' : '1px solid var(--border)',
                          }}
                        >
                          {strat.position === 'IN_POSITION' ? 'In Position' : 'Flat'}
                        </span>
                        <button
                          onClick={() => setConfigModalId(strat.id)}
                          className="px-3 py-1.5 rounded text-xs font-medium"
                          style={{
                            background: 'var(--accent-muted)',
                            color: 'var(--accent)',
                          }}
                        >
                          Configure
                        </button>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            );
          }

          /* ────────────────────── Recent Alerts ────────────────────── */
          if (activeTab === 'Recent Alerts') {
            return (
              <div>
                {/* Summary row */}
                <div className="grid grid-cols-3 gap-4 mb-4">
                  <MetricCard label="Today" value={todayCount.toString()} />
                  <MetricCard label="Entries" value={entryCount.toString()} />
                  <MetricCard label="Exits" value={exitCount.toString()} />
                </div>

                {/* Filters */}
                <div className="flex gap-3 mb-4">
                  <select
                    value={filterStrategy}
                    onChange={(e) => setFilterStrategy(e.target.value)}
                    className="px-3 py-2 rounded-lg text-sm"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  >
                    <option value="All">All Strategies</option>
                    {strategies.map((s) => (
                      <option key={s.id} value={s.name}>{s.name}</option>
                    ))}
                  </select>
                  <select
                    value={filterSignalType}
                    onChange={(e) => setFilterSignalType(e.target.value as 'All' | 'ENTRY' | 'EXIT')}
                    className="px-3 py-2 rounded-lg text-sm"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  >
                    <option value="All">All Types</option>
                    <option value="ENTRY">Entry</option>
                    <option value="EXIT">Exit</option>
                  </select>
                </div>

                {/* Alert feed */}
                <Card>
                  <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>
                    {filteredAlerts.length} alert{filteredAlerts.length !== 1 ? 's' : ''}
                  </h3>
                  {filteredAlerts.length === 0 ? (
                    <div className="text-center py-8" style={{ color: 'var(--text-muted)' }}>
                      <p className="text-sm">No alerts match filters.</p>
                    </div>
                  ) : (
                    <div className="space-y-0">
                      {filteredAlerts.map((alert, i) => (
                        <div
                          key={i}
                          className="flex items-center justify-between py-2.5 border-b last:border-0"
                          style={{ borderColor: 'var(--border)' }}
                        >
                          <div className="flex items-center gap-3 flex-1 min-w-0">
                            <span className="text-xs w-20 shrink-0" style={{ color: 'var(--text-muted)' }}>
                              {alert.time}
                            </span>
                            <span
                              className="text-xs px-1.5 py-0.5 rounded font-medium shrink-0"
                              style={{
                                color: alert.type === 'ENTRY' ? 'var(--green)' : 'var(--red)',
                                background: alert.type === 'ENTRY' ? 'var(--green-muted)' : 'var(--red-muted)',
                              }}
                            >
                              {alert.type}
                            </span>
                            <span
                              className="text-xs px-1.5 py-0.5 rounded shrink-0"
                              style={{ background: 'var(--bg-input)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}
                            >
                              {alert.execType}
                            </span>
                            <span className="text-sm truncate">{alert.strategy}</span>
                          </div>
                          <div className="flex items-center gap-3 shrink-0">
                            <span className="text-sm font-medium">{alert.price}</span>
                            {alert.pnl && (
                              <span
                                className="text-xs font-medium px-1.5 py-0.5 rounded"
                                style={{
                                  color: alert.pnl.startsWith('+') ? 'var(--green)' : 'var(--red)',
                                  background: alert.pnl.startsWith('+') ? 'var(--green-muted)' : 'var(--red-muted)',
                                }}
                              >
                                {alert.pnl}
                              </span>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </Card>
              </div>
            );
          }

          /* ────────────────────── Open Positions ────────────────────── */
          if (activeTab === 'Open Positions') {
            return (
              <div>
                {mockOpenPositions.length === 0 ? (
                  <Card>
                    <div className="text-center py-8" style={{ color: 'var(--text-muted)' }}>
                      <p className="text-sm">No open positions. All strategies are flat.</p>
                    </div>
                  </Card>
                ) : (
                  <Card>
                    <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
                      {mockOpenPositions.length} Open Position{mockOpenPositions.length !== 1 ? 's' : ''}
                    </h3>
                    {/* Table header */}
                    <div
                      className="grid grid-cols-7 gap-2 pb-2 border-b mb-2 text-xs font-medium"
                      style={{ color: 'var(--text-muted)', borderColor: 'var(--border)' }}
                    >
                      <span>Strategy</span>
                      <span>Symbol</span>
                      <span>Entry</span>
                      <span>Current</span>
                      <span>P&L</span>
                      <span>Duration</span>
                      <span>Health</span>
                    </div>
                    {mockOpenPositions.map((pos, i) => (
                      <div key={i} className="grid grid-cols-7 gap-2 py-2.5 border-b last:border-0" style={{ borderColor: 'var(--border)' }}>
                        <span className="text-sm truncate">{pos.strategy}</span>
                        <span className="text-sm font-medium">{pos.symbol}</span>
                        <span className="text-sm">{pos.entryPrice}</span>
                        <span className="text-sm">{pos.currentPrice}</span>
                        <div>
                          <span
                            className="text-sm font-medium"
                            style={{ color: pos.pnl.startsWith('+') ? 'var(--green)' : 'var(--red)' }}
                          >
                            {pos.pnl}
                          </span>
                          <span className="text-xs ml-1" style={{ color: 'var(--text-muted)' }}>
                            ({pos.pnlPct})
                          </span>
                        </div>
                        <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>{pos.duration}</span>
                        <div className="flex items-center gap-1.5">
                          <span
                            className="w-2 h-2 rounded-full"
                            style={{ background: healthColor(pos.health) }}
                          />
                          <span className="text-xs" style={{ color: healthColor(pos.health) }}>
                            {pos.health}
                          </span>
                        </div>
                      </div>
                    ))}
                  </Card>
                )}
              </div>
            );
          }

          /* ────────────────────── History ────────────────────── */
          if (activeTab === 'History') {
            return (
              <div>
                {/* Search */}
                <div className="flex gap-3 mb-4">
                  <input
                    type="text"
                    placeholder="Search alert history..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="flex-1 px-3 py-2 rounded-lg text-sm"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  />
                  <button
                    className="px-4 py-2 rounded-lg text-sm"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                  >
                    Export CSV
                  </button>
                </div>

                <Card>
                  <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>
                    Full Alert History
                  </h3>
                  {/* Table header */}
                  <div
                    className="grid grid-cols-6 gap-2 pb-2 border-b mb-2 text-xs font-medium"
                    style={{ color: 'var(--text-muted)', borderColor: 'var(--border)' }}
                  >
                    <span>Time</span>
                    <span>Type</span>
                    <span>Exec</span>
                    <span>Strategy</span>
                    <span>Price</span>
                    <span>P&L</span>
                  </div>
                  {filteredAlerts.length === 0 ? (
                    <div className="text-center py-8" style={{ color: 'var(--text-muted)' }}>
                      <p className="text-sm">No alerts found.</p>
                    </div>
                  ) : (
                    filteredAlerts.map((alert, i) => (
                      <div
                        key={i}
                        className="grid grid-cols-6 gap-2 py-2 border-b last:border-0 text-sm"
                        style={{ borderColor: 'var(--border)' }}
                      >
                        <span style={{ color: 'var(--text-muted)' }}>{alert.time}</span>
                        <span>
                          <span
                            className="text-xs px-1.5 py-0.5 rounded font-medium"
                            style={{
                              color: alert.type === 'ENTRY' ? 'var(--green)' : 'var(--red)',
                              background: alert.type === 'ENTRY' ? 'var(--green-muted)' : 'var(--red-muted)',
                            }}
                          >
                            {alert.type}
                          </span>
                        </span>
                        <span
                          className="text-xs px-1.5 py-0.5 rounded self-center w-fit"
                          style={{ background: 'var(--bg-input)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}
                        >
                          {alert.execType}
                        </span>
                        <span className="truncate">{alert.strategy}</span>
                        <span className="font-medium">{alert.price}</span>
                        <span>
                          {alert.pnl ? (
                            <span
                              className="text-xs font-medium px-1.5 py-0.5 rounded"
                              style={{
                                color: alert.pnl.startsWith('+') ? 'var(--green)' : 'var(--red)',
                                background: alert.pnl.startsWith('+') ? 'var(--green-muted)' : 'var(--red-muted)',
                              }}
                            >
                              {alert.pnl}
                            </span>
                          ) : (
                            <span style={{ color: 'var(--text-muted)' }}>--</span>
                          )}
                        </span>
                      </div>
                    ))
                  )}
                </Card>
              </div>
            );
          }

          return null;
        }}
      </TabBar>

      {/* ── Alert Configuration Modal ── */}
      <Modal
        title={configStrategy ? `Configure: ${configStrategy.name}` : 'Configure Alert'}
        isOpen={configModalId !== null}
        onClose={() => setConfigModalId(null)}
        width="520px"
      >
        {configStrategy && (
          <div className="space-y-5">
            {/* Alert toggles */}
            <div>
              <h4 className="text-xs font-medium mb-3" style={{ color: 'var(--text-muted)' }}>
                Alert Triggers
              </h4>
              <div className="space-y-2">
                <label className="flex items-center justify-between py-2 px-3 rounded-lg cursor-pointer" style={{ background: 'var(--bg-input)' }}>
                  <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>Alert on Entry</span>
                  <input
                    type="checkbox"
                    checked={configStrategy.alertEntry}
                    onChange={() => toggleField(configStrategy.id, 'alertEntry')}
                    className="accent-[var(--accent)]"
                  />
                </label>
                <label className="flex items-center justify-between py-2 px-3 rounded-lg cursor-pointer" style={{ background: 'var(--bg-input)' }}>
                  <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>Alert on Exit</span>
                  <input
                    type="checkbox"
                    checked={configStrategy.alertExit}
                    onChange={() => toggleField(configStrategy.id, 'alertExit')}
                    className="accent-[var(--accent)]"
                  />
                </label>
              </div>
            </div>

            {/* Webhook config */}
            <div>
              <h4 className="text-xs font-medium mb-3" style={{ color: 'var(--text-muted)' }}>
                Webhook Delivery
              </h4>
              <label className="flex items-center justify-between py-2 px-3 rounded-lg cursor-pointer mb-3" style={{ background: 'var(--bg-input)' }}>
                <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>Enable Webhook</span>
                <input
                  type="checkbox"
                  checked={configStrategy.webhook}
                  onChange={() => toggleField(configStrategy.id, 'webhook')}
                  className="accent-[var(--accent)]"
                />
              </label>
              <div className="space-y-3">
                <div>
                  <label className="block text-xs mb-1.5" style={{ color: 'var(--text-muted)' }}>Webhook URL</label>
                  <input
                    type="url"
                    placeholder="https://hooks.example.com/..."
                    className="w-full px-3 py-2 rounded-lg text-sm"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  />
                </div>
                <div>
                  <label className="block text-xs mb-1.5" style={{ color: 'var(--text-muted)' }}>Webhook Template</label>
                  <select
                    className="w-full px-3 py-2 rounded-lg text-sm"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  >
                    <option>TradeThePool Market Buy</option>
                    <option>Discord Alert</option>
                    <option>Slack Notification</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Notification channels (future) */}
            <div>
              <h4 className="text-xs font-medium mb-3" style={{ color: 'var(--text-muted)' }}>
                Notification Channels (Coming Soon)
              </h4>
              <div className="space-y-2">
                {['Discord', 'Slack', 'Email'].map((channel) => (
                  <label key={channel} className="flex items-center justify-between py-2 px-3 rounded-lg" style={{ background: 'var(--bg-input)', opacity: 0.5 }}>
                    <span className="text-sm" style={{ color: 'var(--text-muted)' }}>{channel}</span>
                    <input type="checkbox" disabled className="accent-[var(--accent)]" />
                  </label>
                ))}
              </div>
            </div>

            {/* Actions */}
            <div className="flex justify-end gap-3 pt-3 border-t" style={{ borderColor: 'var(--border)' }}>
              <button
                className="px-4 py-2 rounded-lg text-sm"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
              >
                Test Delivery
              </button>
              <button
                onClick={() => setConfigModalId(null)}
                className="px-4 py-2 rounded-lg text-sm font-medium"
                style={{ background: 'var(--accent)', color: 'white' }}
              >
                Done
              </button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
