'use client';

import { useState, useMemo } from 'react';
import Card from '@/components/Card';

/* ========================================================================= */
/* TYPES                                                                      */
/* ========================================================================= */

interface MonitoredStrategy {
  id: string;
  name: string;
  symbol: string;
  alertEntry: boolean;
  alertExit: boolean;
  webhook: boolean;
  position: 'FLAT' | 'IN_POSITION';
}

interface AlertRecord {
  time: string;
  type: 'ENTRY' | 'EXIT';
  strategy: string;
  symbol: string;
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
  duration: string;
}

/* ========================================================================= */
/* MOCK DATA                                                                  */
/* ========================================================================= */

const mockMonitorRunning = true;

const mockStrategies: MonitoredStrategy[] = [
  { id: '1', name: 'NVDA LONG - Mass #2', symbol: 'NVDA', alertEntry: true, alertExit: true, webhook: true, position: 'FLAT' },
  { id: '2', name: 'SPY LONG - Mass #1', symbol: 'SPY', alertEntry: true, alertExit: false, webhook: true, position: 'IN_POSITION' },
  { id: '3', name: 'AAPL LONG - Mass #5', symbol: 'AAPL', alertEntry: true, alertExit: true, webhook: false, position: 'FLAT' },
  { id: '4', name: 'TSLA LONG - Mass #5', symbol: 'TSLA', alertEntry: true, alertExit: true, webhook: true, position: 'FLAT' },
];

const mockAlerts: AlertRecord[] = [
  { time: '10:32', type: 'ENTRY', strategy: 'NVDA LONG - Mass #2', symbol: 'NVDA', price: '$487.50', execType: '[C]', pnl: null },
  { time: '10:15', type: 'EXIT', strategy: 'NVDA LONG - Mass #2', symbol: 'NVDA', price: '$487.23', execType: '[C]', pnl: '+1.2R' },
  { time: '09:45', type: 'ENTRY', strategy: 'SPY LONG - Mass #1', symbol: 'SPY', price: '$562.10', execType: '[L0]', pnl: null },
  { time: '09:31', type: 'ENTRY', strategy: 'TSLA LONG - Mass #5', symbol: 'TSLA', price: '$178.90', execType: '[HM]', pnl: null },
  { time: '09:15', type: 'EXIT', strategy: 'AAPL LONG - Mass #5', symbol: 'AAPL', price: '$192.45', execType: '[C]', pnl: '-0.8R' },
  { time: '09:02', type: 'ENTRY', strategy: 'AAPL LONG - Mass #5', symbol: 'AAPL', price: '$191.80', execType: '[L1]', pnl: null },
  { time: '08:50', type: 'EXIT', strategy: 'TSLA LONG - Mass #5', symbol: 'TSLA', price: '$176.50', execType: '[HL]', pnl: '+2.1R' },
];

const mockOpenPositions: OpenPosition[] = [
  { strategy: 'SPY LONG - Mass #1', symbol: 'SPY', entryPrice: '$562.10', currentPrice: '$563.45', pnl: '+$1.35', duration: '47m' },
];

/* ========================================================================= */
/* HELPERS                                                                    */
/* ========================================================================= */

function ToggleSwitch({ on, onChange, size = 'sm' }: { on: boolean; onChange: () => void; size?: 'sm' | 'lg' }) {
  const w = size === 'lg' ? 44 : 28;
  const h = size === 'lg' ? 24 : 14;
  const dot = size === 'lg' ? 18 : 10;
  const offset = size === 'lg' ? 3 : 2;

  return (
    <button
      onClick={onChange}
      className="relative rounded-full transition-colors"
      style={{
        width: w,
        height: h,
        background: on ? 'var(--accent)' : 'var(--bg-input)',
        border: on ? 'none' : '1px solid var(--border)',
        cursor: 'pointer',
        flexShrink: 0,
      }}
    >
      <span
        className="absolute rounded-full transition-all"
        style={{
          width: dot,
          height: dot,
          top: offset,
          left: on ? w - dot - offset : offset,
          background: on ? 'white' : 'var(--text-muted)',
        }}
      />
    </button>
  );
}

/* ========================================================================= */
/* COMPONENT                                                                  */
/* ========================================================================= */

export default function AlertsV3() {
  const [monitorOn, setMonitorOn] = useState(mockMonitorRunning);
  const [strategies, setStrategies] = useState(mockStrategies);
  const [filterStrategy, setFilterStrategy] = useState('All');

  function toggleField(id: string, field: 'alertEntry' | 'alertExit' | 'webhook') {
    setStrategies((prev) =>
      prev.map((s) => (s.id === id ? { ...s, [field]: !s[field] } : s))
    );
  }

  const filteredAlerts = useMemo(() => {
    if (filterStrategy === 'All') return mockAlerts;
    return mockAlerts.filter((a) => a.strategy === filterStrategy);
  }, [filterStrategy]);

  return (
    <div>
      <h1 className="text-2xl font-bold mb-5">Alerts & Signals</h1>

      {/* Two-panel layout */}
      <div className="flex gap-5" style={{ minHeight: '70vh' }}>

        {/* ── Left panel: Monitor control + strategy toggles (40%) ── */}
        <div style={{ width: '40%', flexShrink: 0 }} className="flex flex-col gap-4">

          {/* Monitor toggle */}
          <Card>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span
                  className="w-3 h-3 rounded-full"
                  style={{
                    background: monitorOn ? 'var(--green)' : 'var(--text-muted)',
                    boxShadow: monitorOn ? '0 0 8px var(--green)' : 'none',
                  }}
                />
                <div>
                  <p className="font-semibold text-base">Monitor</p>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    {monitorOn ? 'Running' : 'Stopped'}
                  </p>
                </div>
              </div>
              <ToggleSwitch on={monitorOn} onChange={() => setMonitorOn(!monitorOn)} size="lg" />
            </div>
          </Card>

          {/* Strategy list with inline toggles */}
          <Card className="flex-1 overflow-y-auto">
            <h3 className="text-xs font-medium mb-3" style={{ color: 'var(--text-muted)' }}>
              Monitored Strategies ({strategies.length})
            </h3>
            <div className="space-y-0">
              {strategies.map((strat) => (
                <div
                  key={strat.id}
                  className="py-2.5 border-b last:border-0"
                  style={{ borderColor: 'var(--border)' }}
                >
                  {/* Strategy name + position badge */}
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-sm font-medium truncate mr-2">{strat.name}</span>
                    <span
                      className="text-xs px-1.5 py-0.5 rounded shrink-0"
                      style={{
                        background: strat.position === 'IN_POSITION' ? 'var(--green-muted)' : 'var(--bg-input)',
                        color: strat.position === 'IN_POSITION' ? 'var(--green)' : 'var(--text-muted)',
                        border: strat.position === 'FLAT' ? '1px solid var(--border)' : 'none',
                        fontSize: '0.65rem',
                      }}
                    >
                      {strat.position === 'IN_POSITION' ? 'IN POS' : 'FLAT'}
                    </span>
                  </div>

                  {/* Inline toggles */}
                  <div className="flex items-center gap-4">
                    <label className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--text-muted)' }}>
                      <ToggleSwitch on={strat.alertEntry} onChange={() => toggleField(strat.id, 'alertEntry')} />
                      Entry
                    </label>
                    <label className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--text-muted)' }}>
                      <ToggleSwitch on={strat.alertExit} onChange={() => toggleField(strat.id, 'alertExit')} />
                      Exit
                    </label>
                    <label className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--text-muted)' }}>
                      <ToggleSwitch on={strat.webhook} onChange={() => toggleField(strat.id, 'webhook')} />
                      Webhook
                    </label>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* ── Right panel: Alert feed (60%) ── */}
        <div style={{ width: '60%' }} className="flex flex-col gap-4">

          {/* Open positions card (compact, above feed) */}
          {mockOpenPositions.length > 0 && (
            <Card className="!py-3">
              <h3 className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>
                Open Positions ({mockOpenPositions.length})
              </h3>
              {mockOpenPositions.map((pos, i) => (
                <div key={i} className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-3">
                    <span className="font-medium">{pos.symbol}</span>
                    <span style={{ color: 'var(--text-muted)' }}>{pos.entryPrice}</span>
                    <span style={{ color: 'var(--text-muted)' }}>-&gt;</span>
                    <span>{pos.currentPrice}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span
                      className="font-medium"
                      style={{ color: pos.pnl.startsWith('+') ? 'var(--green)' : 'var(--red)' }}
                    >
                      {pos.pnl}
                    </span>
                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{pos.duration}</span>
                  </div>
                </div>
              ))}
            </Card>
          )}

          {/* Alert feed */}
          <Card className="flex-1 overflow-y-auto">
            {/* Filter */}
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>
                Alert Feed ({filteredAlerts.length})
              </h3>
              <select
                value={filterStrategy}
                onChange={(e) => setFilterStrategy(e.target.value)}
                className="text-xs px-2 py-1 rounded"
                style={{
                  background: 'var(--bg-input)',
                  border: '1px solid var(--border)',
                  color: 'var(--text-primary)',
                }}
              >
                <option value="All">All Strategies</option>
                {strategies.map((s) => (
                  <option key={s.id} value={s.name}>{s.name}</option>
                ))}
              </select>
            </div>

            {/* Feed items — compact one-line format */}
            <div className="space-y-0">
              {filteredAlerts.length === 0 ? (
                <div className="text-center py-8" style={{ color: 'var(--text-muted)' }}>
                  <p className="text-sm">No alerts match filter.</p>
                </div>
              ) : (
                filteredAlerts.map((alert, i) => (
                  <div
                    key={i}
                    className="flex items-center gap-2 py-2 border-b last:border-0"
                    style={{ borderColor: 'var(--border)' }}
                  >
                    {/* Time */}
                    <span className="text-xs w-10 shrink-0 font-mono" style={{ color: 'var(--text-muted)' }}>
                      {alert.time}
                    </span>

                    {/* Exec type badge */}
                    <span
                      className="text-xs px-1 py-0.5 rounded shrink-0 font-mono"
                      style={{
                        background: 'var(--bg-input)',
                        color: 'var(--text-muted)',
                        border: '1px solid var(--border)',
                        fontSize: '0.65rem',
                      }}
                    >
                      {alert.execType}
                    </span>

                    {/* ENTRY / EXIT */}
                    <span
                      className="text-xs font-semibold shrink-0"
                      style={{
                        color: alert.type === 'ENTRY' ? 'var(--green)' : 'var(--red)',
                        width: '38px',
                      }}
                    >
                      {alert.type}
                    </span>

                    {/* Symbol */}
                    <span className="text-sm font-medium shrink-0" style={{ width: '48px' }}>
                      {alert.symbol}
                    </span>

                    {/* Price */}
                    <span className="text-sm shrink-0">{alert.price}</span>

                    {/* P&L (if exit) */}
                    <span className="flex-1" />
                    {alert.pnl && (
                      <span
                        className="text-xs font-medium px-1.5 py-0.5 rounded shrink-0"
                        style={{
                          color: alert.pnl.startsWith('+') ? 'var(--green)' : 'var(--red)',
                          background: alert.pnl.startsWith('+') ? 'var(--green-muted)' : 'var(--red-muted)',
                        }}
                      >
                        {alert.pnl}
                      </span>
                    )}
                  </div>
                ))
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
