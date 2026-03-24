'use client';

import React, { useState, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';

/* ========================================================================= */
/* MOCK DATA                                                                  */
/* ========================================================================= */

const mockStrategyAlerts = [
  { time: '2026-03-24 10:32:15', type: 'ENTRY', strategy: 'NVDA LONG - Mass #2', symbol: 'NVDA', price: 142.35, execType: '[C]', event: 'entry_long_market', status: 'Delivered', pnl: null as number | null },
  { time: '2026-03-24 10:15:03', type: 'EXIT', strategy: 'NVDA LONG - Mass #2', symbol: 'NVDA', price: 143.10, execType: '[C]', event: 'exit_long_market', status: 'Delivered', pnl: 1.2 },
  { time: '2026-03-24 09:45:22', type: 'ENTRY', strategy: 'SPY LONG - Mass #1', symbol: 'SPY', price: 595.20, execType: '[L]', event: 'entry_long_market', status: 'Delivered', pnl: null },
  { time: '2026-03-24 09:31:08', type: 'ENTRY', strategy: 'TSLA LONG - Mass #5', symbol: 'TSLA', price: 178.90, execType: '[LC]', event: 'entry_long_limit', status: 'Delivered', pnl: null },
  { time: '2026-03-23 15:45:00', type: 'EXIT', strategy: 'AAPL LONG - Mass #5', symbol: 'AAPL', price: 248.30, execType: '[C]', event: 'exit_long_market', status: 'Delivered', pnl: -0.8 },
  { time: '2026-03-23 14:22:11', type: 'ENTRY', strategy: 'AAPL LONG - Mass #5', symbol: 'AAPL', price: 247.50, execType: '[C]', event: 'entry_long_market', status: 'Delivered', pnl: null },
  { time: '2026-03-23 10:05:33', type: 'EXIT', strategy: 'SPY LONG - Mass #1', symbol: 'SPY', price: 594.80, execType: '[C]', event: 'exit_long_market', status: 'Failed', pnl: 0.5 },
  { time: '2026-03-23 09:38:00', type: 'ENTRY', strategy: 'META LONG - Mass #13', symbol: 'META', price: 612.50, execType: '[C]', event: 'entry_long_market', status: 'Delivered', pnl: null },
];

const mockPortfolioAlerts = [
  { time: '2026-03-24 10:32:15', portfolio: 'Main Portfolio', strategy: 'NVDA LONG - Mass #2', event: 'entry_long_market', price: 142.35, qty: 12, status: 'Matched' as const },
  { time: '2026-03-24 10:15:03', portfolio: 'Main Portfolio', strategy: 'NVDA LONG - Mass #2', event: 'exit_long_market', price: 143.10, qty: 12, status: 'Matched' as const },
  { time: '2026-03-24 09:45:22', portfolio: 'Main Portfolio', strategy: 'SPY LONG - Mass #1', event: 'entry_long_market', price: 595.20, qty: 8, status: 'Matched' as const },
  { time: '2026-03-24 09:31:08', portfolio: 'Scalping Portfolio', strategy: 'TSLA LONG - Mass #5', event: 'entry_long_limit', price: 178.90, qty: 15, status: 'Phantom' as const },
  { time: '2026-03-23 15:45:00', portfolio: 'Main Portfolio', strategy: 'AAPL LONG - Mass #5', event: 'exit_long_market', price: 248.30, qty: 10, status: 'Matched' as const },
];

const mockOutboundWebhooks = [
  { time: '2026-03-24 10:32:15', template: 'SignalStack — Main', event: 'entry_long_market', strategy: 'NVDA LONG - Mass #2', status: 200, latency: 142, payload: '{"action":"buy","symbol":"NVDA","qty":12,"type":"market"}' },
  { time: '2026-03-24 10:15:03', template: 'SignalStack — Main', event: 'exit_long_market', strategy: 'NVDA LONG - Mass #2', status: 200, latency: 89, payload: '{"action":"sell","symbol":"NVDA","qty":12,"type":"market"}' },
  { time: '2026-03-24 09:45:22', template: 'SignalStack — Main', event: 'entry_long_market', strategy: 'SPY LONG - Mass #1', status: 200, latency: 201, payload: '{"action":"buy","symbol":"SPY","qty":8,"type":"market"}' },
  { time: '2026-03-24 09:31:08', template: 'SignalStack — Main', event: 'entry_long_limit', strategy: 'TSLA LONG - Mass #5', status: 429, latency: 55, payload: '{"action":"buy","symbol":"TSLA","qty":15,"type":"limit","price":178.90}' },
  { time: '2026-03-24 09:31:38', template: 'SignalStack — Main', event: 'entry_long_limit', strategy: 'TSLA LONG - Mass #5', status: 200, latency: 78, payload: '{"action":"buy","symbol":"TSLA","qty":15,"type":"limit","price":178.90}' },
  { time: '2026-03-23 15:45:00', template: 'Discord — #alerts', event: 'exit_long_market', strategy: 'AAPL LONG - Mass #5', status: 200, latency: 112, payload: '{"embeds":[{"title":"Exit Long","description":"AAPL @ $248.30"}]}' },
];

const mockAlertHistory = [
  { id: 1, strategy: 'NVDA LONG - Mass #2', symbol: 'NVDA', entryTime: '10:32:15', exitTime: '10:15:03', entryPrice: 142.35, exitPrice: 143.10, pnlR: 1.2, pnlDollar: 90, entryEvent: 'entry_long_market', exitEvent: 'exit_long_market', entryWebhook: 200, exitWebhook: 200, status: 'Closed' as const },
  { id: 2, strategy: 'SPY LONG - Mass #1', symbol: 'SPY', entryTime: '09:45:22', exitTime: null, entryPrice: 595.20, exitPrice: null, pnlR: null, pnlDollar: null, entryEvent: 'entry_long_market', exitEvent: null, entryWebhook: 200, exitWebhook: null, status: 'Open' as const },
  { id: 3, strategy: 'TSLA LONG - Mass #5', symbol: 'TSLA', entryTime: '09:31:08', exitTime: null, entryPrice: 178.90, exitPrice: null, pnlR: null, pnlDollar: null, entryEvent: 'entry_long_limit', exitEvent: null, entryWebhook: 429, exitWebhook: null, status: 'Open' as const },
  { id: 4, strategy: 'AAPL LONG - Mass #5', symbol: 'AAPL', entryTime: '14:22:11', exitTime: '15:45:00', entryPrice: 247.50, exitPrice: 248.30, pnlR: -0.8, pnlDollar: -40, entryEvent: 'entry_long_market', exitEvent: 'exit_long_market', entryWebhook: 200, exitWebhook: 200, status: 'Closed' as const },
];

const strategies = ['All', 'NVDA LONG - Mass #2', 'SPY LONG - Mass #1', 'AAPL LONG - Mass #5', 'TSLA LONG - Mass #5', 'META LONG - Mass #13'];
const portfolios = ['All', 'Main Portfolio', 'Scalping Portfolio', 'Swing Portfolio'];

/* ========================================================================= */
/* STYLES                                                                      */
/* ========================================================================= */

const thStyle: React.CSSProperties = {
  color: 'var(--text-muted)', background: 'var(--bg-secondary)', textAlign: 'left',
  padding: '8px 10px', fontSize: '0.7rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', whiteSpace: 'nowrap',
};

const tdStyle: React.CSSProperties = {
  padding: '8px 10px', fontSize: '0.8rem', borderBottom: '1px solid var(--border)', color: 'var(--text-secondary)',
};

const selectStyle: React.CSSProperties = {
  background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)',
  padding: '6px 10px', borderRadius: '8px', fontSize: '0.8rem',
};

/* ========================================================================= */
/* HELPERS                                                                     */
/* ========================================================================= */

function EventBadge({ event }: { event: string }) {
  const isEntry = event.includes('entry');
  const isCancel = event.includes('cancel');
  const color = isEntry ? 'var(--green)' : isCancel ? 'var(--orange)' : 'var(--red)';
  return (
    <span className="text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded-full" style={{ color, background: color + '20' }}>
      {event}
    </span>
  );
}

function StatusDot({ code }: { code: number }) {
  const ok = code >= 200 && code < 300;
  return (
    <span className="flex items-center gap-1">
      <span style={{ width: 6, height: 6, borderRadius: '50%', background: ok ? 'var(--green)' : 'var(--red)', display: 'inline-block' }} />
      <span className="text-xs font-mono" style={{ color: ok ? 'var(--green)' : 'var(--red)' }}>{code}</span>
    </span>
  );
}

/* ========================================================================= */
/* TABS                                                                        */
/* ========================================================================= */

const TABS = ['Strategy Alerts', 'Portfolio Alerts', 'Outbound Webhooks', 'Inbound Webhooks'];

/* ========================================================================= */
/* COMPONENT                                                                   */
/* ========================================================================= */

export default function AlertsSignalsV5() {
  const [stratFilter, setStratFilter] = useState('All');
  const [portFilter, setPortFilter] = useState('All');
  const [typeFilter, setTypeFilter] = useState('All');
  const [expandedPayload, setExpandedPayload] = useState<number | null>(null);

  const filteredStratAlerts = useMemo(() => {
    let result = mockStrategyAlerts;
    if (stratFilter !== 'All') result = result.filter((a) => a.strategy === stratFilter);
    if (typeFilter !== 'All') result = result.filter((a) => a.type === typeFilter);
    return result;
  }, [stratFilter, typeFilter]);

  const filteredPortAlerts = useMemo(() => {
    let result = mockPortfolioAlerts;
    if (portFilter !== 'All') result = result.filter((a) => a.portfolio === portFilter);
    return result;
  }, [portFilter]);

  return (
    <div>
      <PageHeader
        title="Alerts & Signals"
        subtitle="Monitor fired alerts, webhook deliveries, and signal activity across all strategies and portfolios."
      />

      {/* Engine status strip */}
      <div className="flex items-center gap-4 mb-5 px-3 py-2 rounded-lg" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
        <span className="flex items-center gap-1.5 text-xs">
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--green)', display: 'inline-block', boxShadow: '0 0 6px var(--green)' }} />
          <span style={{ color: 'var(--green)' }}>Engine Running</span>
        </span>
        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Uptime: 4h 32m</span>
        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Ticks: 12,847</span>
        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>4 symbols</span>
        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>5 strategies</span>
        <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>Last: 10:32:15</span>
      </div>

      <TabBar tabs={TABS}>
        {(tab) => (
          <div>
            {/* =========================================================== */}
            {/* TAB 1: STRATEGY ALERTS                                      */}
            {/* =========================================================== */}
            {tab === 'Strategy Alerts' && (
              <div>
                <div className="flex items-center gap-3 mb-4">
                  <select style={selectStyle} value={stratFilter} onChange={(e) => setStratFilter(e.target.value)}>
                    {strategies.map((s) => <option key={s} value={s}>{s === 'All' ? 'Strategy: All' : s}</option>)}
                  </select>
                  <select style={selectStyle} value={typeFilter} onChange={(e) => setTypeFilter(e.target.value)}>
                    <option value="All">Type: All</option>
                    <option value="ENTRY">Entry</option>
                    <option value="EXIT">Exit</option>
                  </select>
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{filteredStratAlerts.length} alerts</span>
                </div>

                <Card>
                  <h4 className="text-sm font-medium mb-3">Chronological Alert Feed</h4>
                  <div style={{ overflowX: 'auto' }}>
                    <table className="w-full text-sm" style={{ borderCollapse: 'collapse', minWidth: 800 }}>
                      <thead>
                        <tr>
                          {['Time', 'Type', 'Strategy', 'Symbol', 'Price', 'Exec', 'Event', 'P&L', 'Status'].map((h) => (
                            <th key={h} style={thStyle}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {filteredStratAlerts.map((a, i) => (
                          <tr key={i}>
                            <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{a.time}</td>
                            <td style={tdStyle}>
                              <span className="text-xs font-semibold px-2 py-0.5 rounded-full"
                                style={{ color: a.type === 'ENTRY' ? 'var(--green)' : 'var(--red)', background: a.type === 'ENTRY' ? 'var(--green-muted)' : 'var(--red-muted)' }}>
                                {a.type}
                              </span>
                            </td>
                            <td style={tdStyle}>{a.strategy}</td>
                            <td style={tdStyle}>{a.symbol}</td>
                            <td style={tdStyle}>${a.price.toFixed(2)}</td>
                            <td style={tdStyle}>
                              <span className="text-xs font-mono px-1.5 py-0.5 rounded-full" style={{ color: '#2196F3', background: '#2196F320' }}>{a.execType}</span>
                            </td>
                            <td style={tdStyle}><EventBadge event={a.event} /></td>
                            <td style={{ ...tdStyle, color: a.pnl === null ? 'var(--text-muted)' : a.pnl >= 0 ? 'var(--green)' : 'var(--red)', fontWeight: a.pnl !== null ? 600 : 400 }}>
                              {a.pnl !== null ? `${a.pnl >= 0 ? '+' : ''}${a.pnl.toFixed(1)}R` : '--'}
                            </td>
                            <td style={tdStyle}>
                              <span style={{ color: a.status === 'Delivered' ? 'var(--green)' : 'var(--red)' }}>{a.status}</span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>

                {/* Alert History — entry/exit pairs */}
                <Card className="mt-4">
                  <h4 className="text-sm font-medium mb-3">Alert History (Entry/Exit Pairs)</h4>
                  <div style={{ overflowX: 'auto' }}>
                    <table className="w-full text-sm" style={{ borderCollapse: 'collapse', minWidth: 900 }}>
                      <thead>
                        <tr>
                          {['#', 'Strategy', 'Symbol', 'Entry Time', 'Exit Time', 'Entry $', 'Exit $', 'R', 'P&L', 'Entry Event', 'Exit Event', 'Entry WH', 'Exit WH', 'Status'].map((h) => (
                            <th key={h} style={thStyle}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {mockAlertHistory.map((row) => (
                          <tr key={row.id}>
                            <td style={tdStyle}>{row.id}</td>
                            <td style={tdStyle}>{row.strategy.split(' - ')[0]}</td>
                            <td style={tdStyle}>{row.symbol}</td>
                            <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.7rem' }}>{row.entryTime}</td>
                            <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.7rem' }}>{row.exitTime || '--'}</td>
                            <td style={tdStyle}>${row.entryPrice.toFixed(2)}</td>
                            <td style={tdStyle}>{row.exitPrice ? `$${row.exitPrice.toFixed(2)}` : '--'}</td>
                            <td style={{ ...tdStyle, color: row.pnlR === null ? 'var(--text-muted)' : row.pnlR >= 0 ? 'var(--green)' : 'var(--red)', fontWeight: 600 }}>
                              {row.pnlR !== null ? `${row.pnlR >= 0 ? '+' : ''}${row.pnlR.toFixed(1)}R` : '--'}
                            </td>
                            <td style={{ ...tdStyle, color: row.pnlDollar === null ? 'var(--text-muted)' : row.pnlDollar >= 0 ? 'var(--green)' : 'var(--red)', fontWeight: 600 }}>
                              {row.pnlDollar !== null ? `${row.pnlDollar >= 0 ? '+' : ''}$${row.pnlDollar}` : '--'}
                            </td>
                            <td style={tdStyle}><EventBadge event={row.entryEvent} /></td>
                            <td style={tdStyle}>{row.exitEvent ? <EventBadge event={row.exitEvent} /> : '--'}</td>
                            <td style={tdStyle}>{row.entryWebhook ? <StatusDot code={row.entryWebhook} /> : '--'}</td>
                            <td style={tdStyle}>{row.exitWebhook ? <StatusDot code={row.exitWebhook} /> : '--'}</td>
                            <td style={tdStyle}>
                              <span className="text-xs px-1.5 py-0.5 rounded" style={{
                                background: row.status === 'Open' ? 'var(--accent-muted)' : 'var(--green-muted)',
                                color: row.status === 'Open' ? 'var(--accent)' : 'var(--green)',
                              }}>{row.status}</span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>
              </div>
            )}

            {/* =========================================================== */}
            {/* TAB 2: PORTFOLIO ALERTS                                     */}
            {/* =========================================================== */}
            {tab === 'Portfolio Alerts' && (
              <div>
                <div className="flex items-center gap-3 mb-4">
                  <select style={selectStyle} value={portFilter} onChange={(e) => setPortFilter(e.target.value)}>
                    {portfolios.map((p) => <option key={p} value={p}>{p === 'All' ? 'Portfolio: All' : p}</option>)}
                  </select>
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{filteredPortAlerts.length} alerts</span>
                </div>

                {/* Chronological Feed */}
                <Card className="mb-4">
                  <h4 className="text-sm font-medium mb-3">Chronological Alert Feed</h4>
                  <div style={{ overflowX: 'auto' }}>
                    <table className="w-full text-sm" style={{ borderCollapse: 'collapse', minWidth: 750 }}>
                      <thead>
                        <tr>
                          {['Time', 'Portfolio', 'Strategy', 'Event', 'Price', 'Qty', 'Status'].map((h) => (
                            <th key={h} style={thStyle}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {filteredPortAlerts.map((a, i) => (
                          <tr key={i}>
                            <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{a.time}</td>
                            <td style={tdStyle}>{a.portfolio}</td>
                            <td style={tdStyle}>{a.strategy.split(' - ')[0]}</td>
                            <td style={tdStyle}><EventBadge event={a.event} /></td>
                            <td style={tdStyle}>${a.price.toFixed(2)}</td>
                            <td style={tdStyle}>{a.qty}</td>
                            <td style={tdStyle}>
                              <span className="text-xs px-1.5 py-0.5 rounded" style={{
                                background: a.status === 'Matched' ? 'var(--green-muted)' : 'var(--orange)' + '20',
                                color: a.status === 'Matched' ? 'var(--green)' : 'var(--orange)',
                              }}>{a.status}</span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>

                {/* Entry/Exit Pairs */}
                <Card>
                  <h4 className="text-sm font-medium mb-3">Alert History (Entry/Exit Pairs)</h4>
                  <div style={{ overflowX: 'auto' }}>
                    <table className="w-full text-sm" style={{ borderCollapse: 'collapse', minWidth: 900 }}>
                      <thead>
                        <tr>
                          {['#', 'Portfolio', 'Strategy', 'Entry Time', 'Exit Time', 'Entry $', 'Exit $', 'R', 'P&L', 'Qty', 'Entry WH', 'Exit WH', 'Status'].map((h) => (
                            <th key={h} style={thStyle}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {[
                          { id: 1, portfolio: 'Main Portfolio', strategy: 'NVDA', entryTime: '10:32:15', exitTime: '10:15:03', entryPrice: 142.35, exitPrice: 143.10, pnlR: 1.2, pnlDollar: 90, qty: 12, entryWH: 200, exitWH: 200, status: 'Closed' as const },
                          { id: 2, portfolio: 'Main Portfolio', strategy: 'SPY', entryTime: '09:45:22', exitTime: null, entryPrice: 595.20, exitPrice: null, pnlR: null, pnlDollar: null, qty: 8, entryWH: 200, exitWH: null, status: 'Open' as const },
                          { id: 3, portfolio: 'Scalping Portfolio', strategy: 'TSLA', entryTime: '09:31:08', exitTime: null, entryPrice: 178.90, exitPrice: null, pnlR: null, pnlDollar: null, qty: 15, entryWH: 429, exitWH: null, status: 'Open' as const },
                          { id: 4, portfolio: 'Main Portfolio', strategy: 'AAPL', entryTime: '14:22:11', exitTime: '15:45:00', entryPrice: 247.50, exitPrice: 248.30, pnlR: -0.8, pnlDollar: -40, qty: 10, entryWH: 200, exitWH: 200, status: 'Closed' as const },
                        ].map((row) => (
                          <tr key={row.id}>
                            <td style={tdStyle}>{row.id}</td>
                            <td style={tdStyle}>{row.portfolio}</td>
                            <td style={tdStyle}>{row.strategy}</td>
                            <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.7rem' }}>{row.entryTime}</td>
                            <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.7rem' }}>{row.exitTime || '--'}</td>
                            <td style={tdStyle}>${row.entryPrice.toFixed(2)}</td>
                            <td style={tdStyle}>{row.exitPrice ? `$${row.exitPrice.toFixed(2)}` : '--'}</td>
                            <td style={{ ...tdStyle, color: row.pnlR === null ? 'var(--text-muted)' : row.pnlR >= 0 ? 'var(--green)' : 'var(--red)', fontWeight: 600 }}>
                              {row.pnlR !== null ? `${row.pnlR >= 0 ? '+' : ''}${row.pnlR.toFixed(1)}R` : '--'}
                            </td>
                            <td style={{ ...tdStyle, color: row.pnlDollar === null ? 'var(--text-muted)' : row.pnlDollar >= 0 ? 'var(--green)' : 'var(--red)', fontWeight: 600 }}>
                              {row.pnlDollar !== null ? `${row.pnlDollar >= 0 ? '+' : ''}$${row.pnlDollar}` : '--'}
                            </td>
                            <td style={tdStyle}>{row.qty}</td>
                            <td style={tdStyle}>{row.entryWH ? <StatusDot code={row.entryWH} /> : '--'}</td>
                            <td style={tdStyle}>{row.exitWH ? <StatusDot code={row.exitWH} /> : '--'}</td>
                            <td style={tdStyle}>
                              <span className="text-xs px-1.5 py-0.5 rounded" style={{
                                background: row.status === 'Open' ? 'var(--accent-muted)' : 'var(--green-muted)',
                                color: row.status === 'Open' ? 'var(--accent)' : 'var(--green)',
                              }}>{row.status}</span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>
              </div>
            )}

            {/* =========================================================== */}
            {/* TAB 3: OUTBOUND WEBHOOKS                                    */}
            {/* =========================================================== */}
            {tab === 'Outbound Webhooks' && (
              <div>
                {/* Summary metrics */}
                <div className="grid grid-cols-4 gap-3 mb-4">
                  {[
                    { label: 'Total Sent', value: String(mockOutboundWebhooks.length) },
                    { label: 'Success Rate', value: `${((mockOutboundWebhooks.filter((w) => w.status >= 200 && w.status < 300).length / mockOutboundWebhooks.length) * 100).toFixed(0)}%` },
                    { label: 'Avg Latency', value: `${Math.round(mockOutboundWebhooks.reduce((s, w) => s + w.latency, 0) / mockOutboundWebhooks.length)}ms` },
                    { label: 'Failed', value: String(mockOutboundWebhooks.filter((w) => w.status >= 400).length), color: 'var(--red)' },
                  ].map((m) => (
                    <Card key={m.label}>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{m.label}</p>
                      <p className="text-lg font-bold mt-1" style={(m as { color?: string }).color ? { color: (m as { color: string }).color } : undefined}>{m.value}</p>
                    </Card>
                  ))}
                </div>

                <Card>
                  <div style={{ overflowX: 'auto' }}>
                    <table className="w-full text-sm" style={{ borderCollapse: 'collapse', minWidth: 850 }}>
                      <thead>
                        <tr>
                          {['Time', 'Template', 'Event', 'Strategy', 'Status', 'Latency', 'Payload'].map((h) => (
                            <th key={h} style={thStyle}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {mockOutboundWebhooks.map((w, i) => (
                          <React.Fragment key={i}>
                            <tr>
                              <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{w.time}</td>
                              <td style={tdStyle}>{w.template}</td>
                              <td style={tdStyle}><EventBadge event={w.event} /></td>
                              <td style={tdStyle}>{w.strategy.split(' - ')[0]}</td>
                              <td style={tdStyle}><StatusDot code={w.status} /></td>
                              <td style={{ ...tdStyle, fontFamily: 'monospace' }}>{w.latency}ms</td>
                              <td style={tdStyle}>
                                <button
                                  className="text-xs px-2 py-1 rounded"
                                  style={{ background: expandedPayload === i ? 'var(--accent-muted)' : 'var(--bg-input)', color: expandedPayload === i ? 'var(--accent)' : 'var(--text-muted)', border: '1px solid var(--border)', cursor: 'pointer' }}
                                  onClick={() => setExpandedPayload(expandedPayload === i ? null : i)}
                                >
                                  {expandedPayload === i ? 'Hide' : 'View'}
                                </button>
                              </td>
                            </tr>
                            {expandedPayload === i && (
                              <tr>
                                <td colSpan={7} style={{ padding: 0, border: 'none' }}>
                                  <div className="px-4 py-3 font-mono text-xs" style={{ background: 'var(--bg-secondary)', borderBottom: '1px solid var(--border)', whiteSpace: 'pre-wrap', color: 'var(--text-secondary)' }}>
                                    {w.payload}
                                  </div>
                                </td>
                              </tr>
                            )}
                          </React.Fragment>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>
              </div>
            )}

            {/* =========================================================== */}
            {/* TAB 4: INBOUND WEBHOOKS                                     */}
            {/* =========================================================== */}
            {tab === 'Inbound Webhooks' && (
              <div>
                <Card>
                  <div className="text-center py-8">
                    <p className="text-sm font-medium mb-2">Inbound Webhooks</p>
                    <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
                      Receive signals from external sources (TradingView, LuxAlgo) via HTTP POST to your webhook endpoint. Used for Webhook-Based strategy methods.
                    </p>
                    <div className="inline-block rounded-lg p-4 font-mono text-xs text-left" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
                      <p style={{ color: 'var(--text-muted)' }}>POST https://your-domain.com/webhook/inbound/{'<strategy_id>'}</p>
                      <p style={{ color: 'var(--text-muted)' }}>Header: X-Webhook-Secret: {'<your_secret>'}</p>
                      <p style={{ color: 'var(--accent)' }}>{'{'}&quot;action&quot;: &quot;buy&quot;, &quot;price&quot;: 142.35{'}'}</p>
                    </div>
                    <p className="text-xs mt-4" style={{ color: 'var(--text-muted)' }}>
                      No inbound signals received yet. Configure a Webhook-Based strategy to get started.
                    </p>
                  </div>
                </Card>
              </div>
            )}
          </div>
        )}
      </TabBar>
    </div>
  );
}
