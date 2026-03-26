'use client';

/**
 * Alerts & Signals — Faithful copy of V5 design with mock data replaced by API hooks.
 * Source: V5 design file (450 lines)
 *
 * Data convention:
 *   Real value = live from API
 *   -- = wired but no data yet
 *   {{field}} = not wired, needs backend work
 */

import React, { useState, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import { useAlerts, useMonitorStatus, useAlertConfig } from '@/hooks/queries/useAlerts';
import { useStrategies } from '@/hooks/queries/useStrategies';
import { useWebhookDeliveryLog } from '@/hooks/queries/useWebhooks';
import { useStartMonitor, useStopMonitor, useAcknowledgeAlert } from '@/hooks/mutations/useAlertMutations';

/* ========================================================================= */
/* API → V5 ALERT MAPPING                                                     */
/* ========================================================================= */

interface StrategyAlert {
  time: string;
  type: string;
  strategy: string;
  symbol: string;
  price: number;
  execType: string;
  event: string;
  pnl: number | null;
  status: string;
}

interface PortfolioAlert {
  time: string;
  portfolio: string;
  strategy: string;
  event: string;
  price: number;
  qty: number;
  status: 'Matched' | 'Phantom';
}

interface OutboundWebhook {
  time: string;
  template: string;
  event: string;
  strategy: string;
  status: number;
  latency: number;
  payload: string;
}

interface AlertHistoryRow {
  id: number;
  strategy: string;
  symbol: string;
  entryTime: string;
  exitTime: string | null;
  entryPrice: number;
  exitPrice: number | null;
  pnlR: number | null;
  pnlDollar: number | null;
  entryEvent: string;
  exitEvent: string | null;
  entryWebhook: number | null;
  exitWebhook: number | null;
  status: 'Open' | 'Closed';
}

/** Map a raw API alert to the V5 strategy-alert shape */
function apiToStrategyAlert(a: any): StrategyAlert {
  return {
    time: a.time ?? a.timestamp ?? '--',
    type: a.type ?? (a.event?.includes('entry') ? 'ENTRY' : a.event?.includes('exit') ? 'EXIT' : '--'),
    strategy: a.strategy ?? a.strategy_name ?? '--',
    symbol: a.symbol ?? '--',
    price: a.price ?? 0,
    execType: a.exec_type ?? a.execType ?? '--',
    event: a.event ?? a.event_type ?? '--',
    pnl: a.pnl_r ?? a.pnl ?? null,
    status: a.status ?? a.delivery_status ?? '--',
  };
}

/** Map a raw API alert to the V5 portfolio-alert shape */
function apiToPortfolioAlert(a: any): PortfolioAlert {
  return {
    time: a.time ?? a.timestamp ?? '--',
    portfolio: a.portfolio ?? a.portfolio_name ?? '--',
    strategy: a.strategy ?? a.strategy_name ?? '--',
    event: a.event ?? a.event_type ?? '--',
    price: a.price ?? 0,
    qty: a.qty ?? a.quantity ?? 0,
    status: a.status === 'Phantom' ? 'Phantom' : 'Matched',
  };
}

/** Map raw API webhook delivery to the V5 outbound-webhook shape */
function apiToOutboundWebhook(w: any): OutboundWebhook {
  return {
    time: w.time ?? w.timestamp ?? '--',
    template: w.template ?? w.template_name ?? '--',
    event: w.event ?? w.event_type ?? '--',
    strategy: w.strategy ?? w.strategy_name ?? '--',
    status: w.status ?? w.status_code ?? 0,
    latency: w.latency ?? w.latency_ms ?? 0,
    payload: w.payload ?? '{}',
  };
}

/** Map raw API alert-history pair to the V5 history shape */
function apiToAlertHistory(h: any, idx: number): AlertHistoryRow {
  return {
    id: h.id ?? idx + 1,
    strategy: h.strategy ?? h.strategy_name ?? '--',
    symbol: h.symbol ?? '--',
    entryTime: h.entry_time ?? h.entryTime ?? '--',
    exitTime: h.exit_time ?? h.exitTime ?? null,
    entryPrice: h.entry_price ?? h.entryPrice ?? 0,
    exitPrice: h.exit_price ?? h.exitPrice ?? null,
    pnlR: h.pnl_r ?? h.pnlR ?? null,
    pnlDollar: h.pnl_dollar ?? h.pnlDollar ?? null,
    entryEvent: h.entry_event ?? h.entryEvent ?? '--',
    exitEvent: h.exit_event ?? h.exitEvent ?? null,
    entryWebhook: h.entry_webhook ?? h.entryWebhook ?? null,
    exitWebhook: h.exit_webhook ?? h.exitWebhook ?? null,
    status: h.exit_time || h.exitTime ? 'Closed' : 'Open',
  };
}

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

export default function AlertsPage() {
  /* ---- API hooks (all before any early return) ---- */
  const { data: rawAlerts, isLoading: alertsLoading, error: alertsError } = useAlerts();
  const { data: monitorStatus } = useMonitorStatus();
  const { data: alertConfig } = useAlertConfig();
  const { data: rawStrategies } = useStrategies();
  const { data: rawDeliveryLog } = useWebhookDeliveryLog();
  const startMonitor = useStartMonitor();
  const stopMonitor = useStopMonitor();
  const acknowledgeAlert = useAcknowledgeAlert();

  /* ---- Local UI state ---- */
  const [stratFilter, setStratFilter] = useState('All');
  const [portFilter, setPortFilter] = useState('All');
  const [typeFilter, setTypeFilter] = useState('All');
  const [expandedPayload, setExpandedPayload] = useState<number | null>(null);

  /* ---- Derived data from API ---- */
  const strategyAlerts: StrategyAlert[] = useMemo(() => {
    if (!rawAlerts) return [];
    return (rawAlerts as any[]).map(apiToStrategyAlert);
  }, [rawAlerts]);

  const portfolioAlerts: PortfolioAlert[] = useMemo(() => {
    if (!rawAlerts) return [];
    // Portfolio alerts are alerts that have portfolio context
    return (rawAlerts as any[]).filter((a: any) => a.portfolio || a.portfolio_name).map(apiToPortfolioAlert);
  }, [rawAlerts]);

  const outboundWebhooks: OutboundWebhook[] = useMemo(() => {
    if (!rawDeliveryLog) return [];
    return (rawDeliveryLog as any[]).map(apiToOutboundWebhook);
  }, [rawDeliveryLog]);

  const alertHistory: AlertHistoryRow[] = useMemo(() => {
    if (!rawAlerts) return [];
    // History entries are alerts with entry/exit pairing info
    return (rawAlerts as any[]).filter((a: any) => a.entry_time || a.entryTime || a.entry_price || a.entryPrice).map(apiToAlertHistory);
  }, [rawAlerts]);

  /** Strategy names for filter dropdown — derived from API strategies */
  const strategyNames = useMemo(() => {
    const names = ['All'];
    if (rawStrategies) {
      (rawStrategies as any[]).forEach((s) => { if (s.name) names.push(s.name); });
    }
    // Also add any strategy names that appear in alerts but not in strategies list
    strategyAlerts.forEach((a) => { if (a.strategy !== '--' && !names.includes(a.strategy)) names.push(a.strategy); });
    return names;
  }, [rawStrategies, strategyAlerts]);

  /** Portfolio names for filter dropdown — derived from portfolio alerts */
  const portfolioNames = useMemo(() => {
    const names = ['All'];
    portfolioAlerts.forEach((a) => { if (a.portfolio !== '--' && !names.includes(a.portfolio)) names.push(a.portfolio); });
    return names;
  }, [portfolioAlerts]);

  const filteredStratAlerts = useMemo(() => {
    let result = strategyAlerts;
    if (stratFilter !== 'All') result = result.filter((a) => a.strategy === stratFilter);
    if (typeFilter !== 'All') result = result.filter((a) => a.type === typeFilter);
    return result;
  }, [strategyAlerts, stratFilter, typeFilter]);

  const filteredPortAlerts = useMemo(() => {
    let result = portfolioAlerts;
    if (portFilter !== 'All') result = result.filter((a) => a.portfolio === portFilter);
    return result;
  }, [portfolioAlerts, portFilter]);

  /* ---- Engine status derived values ---- */
  const engineRunning = monitorStatus?.running === true || monitorStatus?.status === 'running';
  const engineUptime = monitorStatus?.uptime ?? '--';
  const engineTicks = monitorStatus?.ticks ?? '--';
  const engineSymbols = monitorStatus?.symbols ?? monitorStatus?.symbol_count ?? '--';
  const engineStrategies = monitorStatus?.strategy_count ?? monitorStatus?.strategies ?? '--';
  const engineLastTime = monitorStatus?.last_alert_time ?? monitorStatus?.last_tick ?? '--';

  /* ---- Loading / Error states ---- */
  if (alertsLoading) {
    return (
      <div>
        <PageHeader title="Alerts & Signals" subtitle="Loading..." />
        <div className="space-y-4 mt-4">
          {[1, 2, 3].map((i) => (
            <Card key={i}>
              <div className="animate-pulse space-y-3">
                <div className="h-4 rounded w-1/4" style={{ background: 'var(--border)' }} />
                <div className="h-20 rounded" style={{ background: 'var(--bg-input)' }} />
              </div>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  if (alertsError) {
    return (
      <div>
        <PageHeader title="Alerts & Signals" subtitle="Error loading alerts" />
        <Card>
          <div className="text-center py-8" style={{ color: 'var(--red)' }}>
            Failed to load alerts. Check your connection and try again.
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div>
      <PageHeader
        title="Alerts & Signals"
        subtitle="Monitor fired alerts, webhook deliveries, and signal activity across all strategies and portfolios."
      />

      {/* Engine status strip */}
      <div className="flex items-center gap-4 mb-5 px-3 py-2 rounded-lg" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
        <span className="flex items-center gap-1.5 text-xs">
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: engineRunning ? 'var(--green)' : 'var(--text-muted)', display: 'inline-block', boxShadow: engineRunning ? '0 0 6px var(--green)' : 'none' }} />
          <span style={{ color: engineRunning ? 'var(--green)' : 'var(--text-muted)' }}>{engineRunning ? 'Engine Running' : 'Engine Stopped'}</span>
        </span>
        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Uptime: {engineUptime}</span>
        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Ticks: {engineTicks}</span>
        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{engineSymbols} symbols</span>
        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{engineStrategies} strategies</span>
        <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>Last: {engineLastTime}</span>
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
                    {strategyNames.map((s) => <option key={s} value={s}>{s === 'All' ? 'Strategy: All' : s}</option>)}
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
                        {filteredStratAlerts.length === 0 ? (
                          <tr><td colSpan={9} style={{ ...tdStyle, textAlign: 'center', color: 'var(--text-muted)' }}>No alerts yet. Start the engine to begin monitoring.</td></tr>
                        ) : filteredStratAlerts.map((a, i) => (
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
                            <td style={tdStyle}>{a.price ? `$${a.price.toFixed(2)}` : '--'}</td>
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
                        {alertHistory.length === 0 ? (
                          <tr><td colSpan={14} style={{ ...tdStyle, textAlign: 'center', color: 'var(--text-muted)' }}>No alert history yet.</td></tr>
                        ) : alertHistory.map((row) => (
                          <tr key={row.id}>
                            <td style={tdStyle}>{row.id}</td>
                            <td style={tdStyle}>{row.strategy.split(' - ')[0]}</td>
                            <td style={tdStyle}>{row.symbol}</td>
                            <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.7rem' }}>{row.entryTime}</td>
                            <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.7rem' }}>{row.exitTime || '--'}</td>
                            <td style={tdStyle}>{row.entryPrice ? `$${row.entryPrice.toFixed(2)}` : '--'}</td>
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
                    {portfolioNames.map((p) => <option key={p} value={p}>{p === 'All' ? 'Portfolio: All' : p}</option>)}
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
                        {filteredPortAlerts.length === 0 ? (
                          <tr><td colSpan={7} style={{ ...tdStyle, textAlign: 'center', color: 'var(--text-muted)' }}>No portfolio alerts yet.</td></tr>
                        ) : filteredPortAlerts.map((a, i) => (
                          <tr key={i}>
                            <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{a.time}</td>
                            <td style={tdStyle}>{a.portfolio}</td>
                            <td style={tdStyle}>{a.strategy.split(' - ')[0]}</td>
                            <td style={tdStyle}><EventBadge event={a.event} /></td>
                            <td style={tdStyle}>{a.price ? `$${a.price.toFixed(2)}` : '--'}</td>
                            <td style={tdStyle}>{a.qty || '--'}</td>
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
                        {alertHistory.length === 0 ? (
                          <tr><td colSpan={13} style={{ ...tdStyle, textAlign: 'center', color: 'var(--text-muted)' }}>No portfolio alert history yet.</td></tr>
                        ) : alertHistory.filter((row) => portfolioAlerts.some((pa) => pa.strategy.includes(row.symbol) || pa.strategy.includes(row.strategy))).map((row) => (
                          <tr key={row.id}>
                            <td style={tdStyle}>{row.id}</td>
                            <td style={tdStyle}>{'{{portfolio}}'}</td>
                            <td style={tdStyle}>{row.strategy.split(' - ')[0]}</td>
                            <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.7rem' }}>{row.entryTime}</td>
                            <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.7rem' }}>{row.exitTime || '--'}</td>
                            <td style={tdStyle}>{row.entryPrice ? `$${row.entryPrice.toFixed(2)}` : '--'}</td>
                            <td style={tdStyle}>{row.exitPrice ? `$${row.exitPrice.toFixed(2)}` : '--'}</td>
                            <td style={{ ...tdStyle, color: row.pnlR === null ? 'var(--text-muted)' : row.pnlR >= 0 ? 'var(--green)' : 'var(--red)', fontWeight: 600 }}>
                              {row.pnlR !== null ? `${row.pnlR >= 0 ? '+' : ''}${row.pnlR.toFixed(1)}R` : '--'}
                            </td>
                            <td style={{ ...tdStyle, color: row.pnlDollar === null ? 'var(--text-muted)' : row.pnlDollar >= 0 ? 'var(--green)' : 'var(--red)', fontWeight: 600 }}>
                              {row.pnlDollar !== null ? `${row.pnlDollar >= 0 ? '+' : ''}$${row.pnlDollar}` : '--'}
                            </td>
                            <td style={tdStyle}>{'{{qty}}'}</td>
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
            {/* TAB 3: OUTBOUND WEBHOOKS                                    */}
            {/* =========================================================== */}
            {tab === 'Outbound Webhooks' && (
              <div>
                {/* Summary metrics */}
                <div className="grid grid-cols-4 gap-3 mb-4">
                  {[
                    { label: 'Total Sent', value: outboundWebhooks.length > 0 ? String(outboundWebhooks.length) : '--' },
                    { label: 'Success Rate', value: outboundWebhooks.length > 0 ? `${((outboundWebhooks.filter((w) => w.status >= 200 && w.status < 300).length / outboundWebhooks.length) * 100).toFixed(0)}%` : '--' },
                    { label: 'Avg Latency', value: outboundWebhooks.length > 0 ? `${Math.round(outboundWebhooks.reduce((s, w) => s + w.latency, 0) / outboundWebhooks.length)}ms` : '--' },
                    { label: 'Failed', value: outboundWebhooks.length > 0 ? String(outboundWebhooks.filter((w) => w.status >= 400).length) : '--', color: 'var(--red)' },
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
                        {outboundWebhooks.length === 0 ? (
                          <tr><td colSpan={7} style={{ ...tdStyle, textAlign: 'center', color: 'var(--text-muted)' }}>No webhook deliveries yet.</td></tr>
                        ) : outboundWebhooks.map((w, i) => (
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
