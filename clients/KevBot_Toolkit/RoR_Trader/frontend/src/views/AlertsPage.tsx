'use client';

/**
 * Alerts & Signals — Clean API-first page.
 *
 * Visual design derived from V5 (versions/V5.tsx), data layer built
 * around actual alert/monitor API endpoints. No mock data.
 */

import { useState, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import { useAlerts, useMonitorStatus, useAlertConfig } from '@/hooks/queries/useAlerts';
import { useStartMonitor, useStopMonitor, useAcknowledgeAlert } from '@/hooks/mutations/useAlertMutations';

// ---------------------------------------------------------------------------
// Style constants (from V5)
// ---------------------------------------------------------------------------

const thStyle: React.CSSProperties = {
  color: 'var(--text-muted)', background: 'var(--bg-secondary)', textAlign: 'left',
  padding: '8px 10px', fontSize: '0.7rem', fontWeight: 600, textTransform: 'uppercase',
  letterSpacing: '0.05em', whiteSpace: 'nowrap',
};

const tdStyle: React.CSSProperties = {
  padding: '8px 10px', fontSize: '0.8rem', borderBottom: '1px solid var(--border)',
  color: 'var(--text-secondary)',
};

const selectStyle: React.CSSProperties = {
  background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)',
  padding: '6px 10px', borderRadius: '8px', fontSize: '0.8rem',
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function EventBadge({ event }: { event: string }) {
  const isEntry = (event || '').includes('entry');
  const isCancel = (event || '').includes('cancel');
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

// ---------------------------------------------------------------------------
// Tabs (V5 4-tab layout)
// ---------------------------------------------------------------------------

const TABS = ['Strategy Alerts', 'Portfolio Alerts', 'Outbound Webhooks', 'Inbound Webhooks'];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function AlertsPage() {
  const { data: alerts, isLoading: alertsLoading, error: alertsError } = useAlerts();
  const { data: monitorStatus } = useMonitorStatus();
  const { data: alertConfig } = useAlertConfig();
  const startMonitor = useStartMonitor();
  const stopMonitor = useStopMonitor();
  const ackMutation = useAcknowledgeAlert();

  const [stratFilter, setStratFilter] = useState('All');
  const [typeFilter, setTypeFilter] = useState('All');
  const [portFilter, setPortFilter] = useState('All');
  const [expandedPayload, setExpandedPayload] = useState<number | null>(null);

  // Derive unique strategy names from alerts
  const strategyNames = useMemo(() => {
    if (!alerts) return [];
    return Array.from(new Set(alerts.map((a: any) => a.strategy || a.strategy_name || ''))).filter(Boolean).sort();
  }, [alerts]);

  // Derive unique portfolio names from alerts
  const portfolioNames = useMemo(() => {
    if (!alerts) return [];
    return Array.from(new Set(alerts.map((a: any) => a.portfolio || '').filter(Boolean))).sort();
  }, [alerts]);

  // Filter alerts for strategy tab
  const filteredAlerts = useMemo(() => {
    if (!alerts) return [];
    let result = [...alerts] as any[];
    if (stratFilter !== 'All') result = result.filter((a) => (a.strategy || a.strategy_name) === stratFilter);
    if (typeFilter !== 'All') result = result.filter((a) => (a.type || '').toUpperCase() === typeFilter);
    return result;
  }, [alerts, stratFilter, typeFilter]);

  // Filter alerts for portfolio tab
  const filteredPortfolioAlerts = useMemo(() => {
    if (!alerts) return [];
    let result = (alerts as any[]).filter((a) => a.portfolio);
    if (portFilter !== 'All') result = result.filter((a) => a.portfolio === portFilter);
    return result;
  }, [alerts, portFilter]);

  const webhookDeliveries = useMemo(() => !alerts ? [] : (alerts as any[]).filter((a) => a.webhook_status != null || a.webhook_template), [alerts]);

  // Monitor state
  const isRunning = monitorStatus?.status === 'running' || monitorStatus?.desired_state === 'running';

  // ---------------------------------------------------------------------------
  // Loading / Error
  // ---------------------------------------------------------------------------

  if (alertsLoading) {
    return (
      <div>
        <PageHeader title="Alerts & Signals" subtitle="Loading..." />
        <div className="space-y-3 mt-4">
          {[1, 2, 3].map((i) => (
            <Card key={i}>
              <div className="animate-pulse space-y-3">
                <div className="h-4 rounded w-1/3" style={{ background: 'var(--border)' }} />
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
        <PageHeader title="Alerts & Signals" subtitle="Error" />
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
        subtitle="Monitor fired alerts, webhook deliveries, and signal activity across all strategies and portfolios"
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

      {/* Engine status strip — V5: uptime, ticks, symbols, strategies, last tick */}
      <div className="flex items-center gap-4 mb-5 px-3 py-2 rounded-lg" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
        <span className="flex items-center gap-1.5 text-xs">
          <span
            style={{
              width: 8, height: 8, borderRadius: '50%', display: 'inline-block',
              background: isRunning ? 'var(--green)' : 'var(--text-muted)',
              boxShadow: isRunning ? '0 0 6px var(--green)' : 'none',
            }}
          />
          <span style={{ color: isRunning ? 'var(--green)' : 'var(--text-muted)' }}>
            {isRunning ? 'Engine Running' : 'Engine Stopped'}
          </span>
        </span>

        {monitorStatus?.uptime && (
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Uptime: {monitorStatus.uptime}
          </span>
        )}
        {monitorStatus?.ticks != null && (
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Ticks: {Number(monitorStatus.ticks).toLocaleString()}
          </span>
        )}
        {monitorStatus?.symbols_count != null && (
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {monitorStatus.symbols_count} symbols
          </span>
        )}
        {monitorStatus?.strategies_count != null && (
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {monitorStatus.strategies_count} strategies
          </span>
        )}
        {monitorStatus?.last_tick && (
          <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
            Last: {monitorStatus.last_tick}
          </span>
        )}

        <span className="flex-1" />

        <button
          className="text-xs px-3 py-1.5 rounded-lg font-medium"
          style={{
            background: isRunning ? 'var(--red-muted, rgba(239,83,80,0.15))' : 'var(--green-muted, rgba(76,175,80,0.15))',
            color: isRunning ? 'var(--red)' : 'var(--green)',
            border: 'none',
            cursor: 'pointer',
          }}
          onClick={() => {
            if (isRunning) stopMonitor.mutate();
            else startMonitor.mutate();
          }}
          disabled={startMonitor.isPending || stopMonitor.isPending}
        >
          {isRunning ? 'Stop Monitor' : 'Start Monitor'}
        </button>
      </div>

      <TabBar tabs={TABS}>
        {(tab) => (
          <div>
            {/* TAB 1: Strategy Alerts */}
            {tab === 'Strategy Alerts' && (
              <div>
                <div className="flex items-center gap-3 mb-4">
                  <select style={selectStyle} value={stratFilter} onChange={(e) => setStratFilter(e.target.value)}>
                    <option value="All">Strategy: All</option>
                    {strategyNames.map((s) => <option key={s} value={s}>{s}</option>)}
                  </select>
                  <select style={selectStyle} value={typeFilter} onChange={(e) => setTypeFilter(e.target.value)}>
                    <option value="All">Type: All</option>
                    <option value="ENTRY">Entry</option>
                    <option value="EXIT">Exit</option>
                  </select>
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    {filteredAlerts.length} alert{filteredAlerts.length !== 1 ? 's' : ''}
                  </span>
                </div>

                {filteredAlerts.length === 0 ? (
                  <Card>
                    <div className="text-center py-8" style={{ color: 'var(--text-muted)' }}>
                      No alerts found. Alerts will appear here when the monitor fires signals.
                    </div>
                  </Card>
                ) : (
                  <>
                    {/* Chronological Alert Feed */}
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
                            {filteredAlerts.map((a: any, i: number) => {
                              const alertType = (a.type || '').toUpperCase();
                              return (
                                <tr key={a.id || i}>
                                  <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>
                                    {a.timestamp || a.time || a.created_at || '--'}
                                  </td>
                                  <td style={tdStyle}>
                                    <span
                                      className="text-xs font-semibold px-2 py-0.5 rounded-full"
                                      style={{
                                        color: alertType === 'ENTRY' ? 'var(--green)' : 'var(--red)',
                                        background: alertType === 'ENTRY' ? 'var(--green-muted, rgba(76,175,80,0.15))' : 'var(--red-muted, rgba(239,83,80,0.15))',
                                      }}
                                    >
                                      {alertType || '--'}
                                    </span>
                                  </td>
                                  <td style={tdStyle}>{a.strategy || a.strategy_name || '--'}</td>
                                  <td style={tdStyle}>{a.symbol || '--'}</td>
                                  <td style={tdStyle}>
                                    {a.price != null ? `$${Number(a.price).toFixed(2)}` : '--'}
                                  </td>
                                  <td style={tdStyle}>
                                    {a.exec_type ? (
                                      <span className="text-xs font-mono px-1.5 py-0.5 rounded-full" style={{ color: '#2196F3', background: '#2196F320' }}>
                                        {a.exec_type}
                                      </span>
                                    ) : '--'}
                                  </td>
                                  <td style={tdStyle}>
                                    {a.event ? <EventBadge event={a.event} /> : '--'}
                                  </td>
                                  <td style={{
                                    ...tdStyle,
                                    color: a.pnl == null ? 'var(--text-muted)' : a.pnl >= 0 ? 'var(--green)' : 'var(--red)',
                                    fontWeight: a.pnl != null ? 600 : 400,
                                  }}>
                                    {a.pnl != null ? `${a.pnl >= 0 ? '+' : ''}${Number(a.pnl).toFixed(1)}R` : '--'}
                                  </td>
                                  <td style={tdStyle}>
                                    <span style={{ color: a.acknowledged ? 'var(--text-muted)' : 'var(--green)' }}>
                                      {a.acknowledged ? 'Ack' : a.status || 'New'}
                                    </span>
                                    {!a.acknowledged && a.id && (
                                      <button
                                        className="text-xs ml-2 px-1.5 py-0.5 rounded"
                                        style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-muted)', cursor: 'pointer' }}
                                        onClick={() => ackMutation.mutate(a.id)}
                                      >
                                        Ack
                                      </button>
                                    )}
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </Card>

                    {/* Alert History — entry/exit pairs */}
                    <Card className="mt-4">
                      <h4 className="text-sm font-medium mb-3">Alert History (Entry/Exit Pairs)</h4>
                      <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
                        Entry/exit pairs will be matched automatically from the alert feed. Pairs with webhook delivery status are shown when available.
                      </p>
                      <div className="text-center py-6" style={{ color: 'var(--text-muted)' }}>
                        <p className="text-xs">Alert pair matching requires active monitor data.</p>
                      </div>
                    </Card>
                  </>
                )}
              </div>
            )}

            {/* TAB 2: Portfolio Alerts */}
            {tab === 'Portfolio Alerts' && (
              <div>
                <div className="flex items-center gap-3 mb-4">
                  <select style={selectStyle} value={portFilter} onChange={(e) => setPortFilter(e.target.value)}>
                    <option value="All">Portfolio: All</option>
                    {portfolioNames.map((p) => <option key={p} value={p}>{p}</option>)}
                  </select>
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    {filteredPortfolioAlerts.length} alert{filteredPortfolioAlerts.length !== 1 ? 's' : ''}
                  </span>
                </div>

                {filteredPortfolioAlerts.length === 0 ? (
                  <Card>
                    <div className="text-center py-8" style={{ color: 'var(--text-muted)' }}>
                      No portfolio alerts yet. Portfolio-level alerts appear here when strategies within a portfolio fire signals.
                    </div>
                  </Card>
                ) : (
                  <Card>
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
                          {filteredPortfolioAlerts.map((a: any, i: number) => (
                            <tr key={a.id || i}>
                              <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>
                                {a.timestamp || a.time || a.created_at || '--'}
                              </td>
                              <td style={tdStyle}>{a.portfolio || '--'}</td>
                              <td style={tdStyle}>{a.strategy || a.strategy_name || '--'}</td>
                              <td style={tdStyle}>{a.event ? <EventBadge event={a.event} /> : '--'}</td>
                              <td style={tdStyle}>
                                {a.price != null ? `$${Number(a.price).toFixed(2)}` : '--'}
                              </td>
                              <td style={tdStyle}>{a.qty ?? '--'}</td>
                              <td style={tdStyle}>
                                <span className="text-xs px-1.5 py-0.5 rounded" style={{
                                  background: a.match_status === 'Matched' ? 'var(--green-muted, rgba(76,175,80,0.15))' : 'var(--orange)' + '20',
                                  color: a.match_status === 'Matched' ? 'var(--green)' : 'var(--orange)',
                                }}>
                                  {a.match_status || a.status || '--'}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </Card>
                )}
              </div>
            )}

            {/* TAB 3: Outbound Webhooks */}
            {tab === 'Outbound Webhooks' && (
              <div>
                {webhookDeliveries.length === 0 ? (
                  <Card>
                    <div className="text-center py-8">
                      <p className="text-sm font-medium mb-2">Outbound Webhook Delivery Log</p>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        View delivery history, response codes, and latency for all outbound webhook dispatches.
                        Configure webhook templates in Settings.
                      </p>
                    </div>
                  </Card>
                ) : (
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
                          {webhookDeliveries.map((w: any, i: number) => (
                            <tr key={w.id || i}>
                              <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>
                                {w.timestamp || w.time || w.created_at || '--'}
                              </td>
                              <td style={tdStyle}>{w.webhook_template || '--'}</td>
                              <td style={tdStyle}>{w.event ? <EventBadge event={w.event} /> : '--'}</td>
                              <td style={tdStyle}>{(w.strategy || w.strategy_name || '--').split(' - ')[0]}</td>
                              <td style={tdStyle}>
                                {w.webhook_status ? <StatusDot code={w.webhook_status} /> : '--'}
                              </td>
                              <td style={{ ...tdStyle, fontFamily: 'monospace' }}>
                                {w.webhook_latency != null ? `${w.webhook_latency}ms` : '--'}
                              </td>
                              <td style={tdStyle}>--</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </Card>
                )}
              </div>
            )}

            {/* TAB 4: Inbound Webhooks */}
            {tab === 'Inbound Webhooks' && (
              <Card>
                <div className="text-center py-8">
                  <p className="text-sm font-medium mb-2">Inbound Webhooks</p>
                  <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
                    Receive signals from external sources (TradingView, LuxAlgo) via HTTP POST.
                    Used for Webhook-Based strategy methods.
                  </p>
                  <div
                    className="inline-block rounded-lg p-4 font-mono text-xs text-left"
                    style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}
                  >
                    <p style={{ color: 'var(--text-muted)' }}>
                      POST https://your-domain.com/webhook/inbound/{'<strategy_id>'}
                    </p>
                    <p style={{ color: 'var(--text-muted)' }}>
                      Header: X-Webhook-Secret: {'<your_secret>'}
                    </p>
                    <p style={{ color: 'var(--accent)' }}>
                      {'{'}&#34;action&#34;: &#34;buy&#34;, &#34;price&#34;: 142.35{'}'}
                    </p>
                  </div>
                  <p className="text-xs mt-4" style={{ color: 'var(--text-muted)' }}>
                    No inbound signals received yet. Configure a Webhook-Based strategy to get started.
                  </p>
                </div>
              </Card>
            )}
          </div>
        )}
      </TabBar>
    </div>
  );
}
