'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import ChartPlaceholder from '@/components/ChartPlaceholder';

/* ========================================================================= */
/* CONSTANTS                                                                   */
/* ========================================================================= */

const EVENT_TYPES = [
  { event: 'entry_long_market', label: 'Entry Long (Market)', category: 'entry' },
  { event: 'entry_long_limit', label: 'Entry Long (Limit)', category: 'entry' },
  { event: 'entry_short_market', label: 'Entry Short (Market)', category: 'entry' },
  { event: 'entry_short_limit', label: 'Entry Short (Limit)', category: 'entry' },
  { event: 'exit_long_market', label: 'Exit Long (Market)', category: 'exit' },
  { event: 'exit_long_limit', label: 'Exit Long (Limit)', category: 'exit' },
  { event: 'exit_short_market', label: 'Exit Short (Market)', category: 'exit' },
  { event: 'exit_short_limit', label: 'Exit Short (Limit)', category: 'exit' },
  { event: 'cancel_long', label: 'Cancel Long', category: 'cancel' },
  { event: 'cancel_short', label: 'Cancel Short', category: 'cancel' },
  { event: 'compliance_breach', label: 'Compliance Breach', category: 'compliance' },
] as const;

const CATEGORY_COLORS: Record<string, { color: string; bg: string }> = {
  entry: { color: 'var(--green)', bg: 'var(--green-muted)' },
  exit: { color: 'var(--red)', bg: 'var(--red-muted)' },
  cancel: { color: 'var(--orange)', bg: 'rgba(255,152,0,0.12)' },
  compliance: { color: '#AB47BC', bg: 'rgba(171,71,188,0.12)' },
};

const PLACEHOLDERS = [
  { key: '{{symbol}}', example: 'NVDA', category: 'Signal' },
  { key: '{{direction}}', example: 'LONG', category: 'Signal' },
  { key: '{{event_type}}', example: 'entry_long_market', category: 'Signal' },
  { key: '{{order_action}}', example: 'buy', category: 'Order' },
  { key: '{{order_type}}', example: 'market', category: 'Order' },
  { key: '{{order_price}}', example: '142.35', category: 'Order' },
  { key: '{{stop_price}}', example: '141.50', category: 'Order' },
  { key: '{{quantity}}', example: '10', category: 'Order' },
  { key: '{{trigger_name}}', example: 'EMA_CROSS', category: 'Strategy' },
  { key: '{{strategy_name}}', example: 'NVDA LONG - Mass #2', category: 'Strategy' },
  { key: '{{timeframe}}', example: '1Min', category: 'Strategy' },
  { key: '{{confluence_met}}', example: '5M-RVOL-HIGH, 1D-MACD-BULL', category: 'Strategy' },
  { key: '{{portfolio_name}}', example: 'Main Portfolio', category: 'Portfolio' },
  { key: '{{risk_per_trade}}', example: '500.00', category: 'Portfolio' },
  { key: '{{atr}}', example: '1.42', category: 'Indicator' },
  { key: '{{timestamp}}', example: '2026-03-24T14:30:00Z', category: 'Meta' },
];

const TABS = ['Event Payloads', 'Placeholders', 'Delivery History', 'Settings'];

/* ========================================================================= */
/* MOCK DATA                                                                   */
/* ========================================================================= */

function makePayload(e: typeof EVENT_TYPES[number]): string {
  if (e.category === 'entry') {
    const action = e.event.includes('short') ? 'sell_short' : 'buy';
    const type = e.event.includes('limit') ? 'limit' : 'market';
    const priceField = e.event.includes('limit') ? ',\n  "price": {{order_price}}' : '';
    return `{\n  "action": "${action}",\n  "type": "${type}",\n  "symbol": "{{symbol}}",\n  "qty": {{quantity}}${priceField}\n}`;
  }
  if (e.category === 'exit') {
    const action = e.event.includes('short') ? 'buy_to_cover' : 'sell';
    const type = e.event.includes('limit') ? 'limit' : 'market';
    const priceField = e.event.includes('limit') ? ',\n  "price": {{order_price}}' : '';
    return `{\n  "action": "${action}",\n  "type": "${type}",\n  "symbol": "{{symbol}}",\n  "qty": {{quantity}}${priceField}\n}`;
  }
  if (e.category === 'cancel') {
    const side = e.event.includes('long') ? 'buy' : 'sell';
    return `{\n  "action": "cancel_all",\n  "symbol": "{{symbol}}",\n  "side": "${side}"\n}`;
  }
  return '{\n  "alert": "compliance_breach",\n  "portfolio": "{{portfolio_name}}",\n  "rule": "{{rule_name}}",\n  "value": "{{rule_value}}"\n}';
}

const template = {
  id: 'tpl_1',
  name: 'SignalStack — Main Account',
  exchange: 'SignalStack',
  url: 'https://app.signalstack.com/hook/abc123def456',
  isDefault: false,
  usedByPortfolios: 3,
  createdAt: '2026-03-10 14:30:00',
  updatedAt: '2026-03-24 09:00:00',
  portfolios: ['Main Portfolio', 'Scalping Portfolio', 'Swing Portfolio'],
  events: EVENT_TYPES.map((e) => ({ ...e, payload: makePayload(e) })),
  deliveries: [
    { time: '2026-03-24 09:42:15', event: 'entry_long_market', strategy: 'NVDA LONG - Mass #2', status: 200, latency: 142 },
    { time: '2026-03-24 09:46:32', event: 'exit_long_market', strategy: 'NVDA LONG - Mass #2', status: 200, latency: 89 },
    { time: '2026-03-23 14:15:03', event: 'entry_long_market', strategy: 'SPY Swing', status: 200, latency: 201 },
    { time: '2026-03-23 10:22:11', event: 'entry_long_limit', strategy: 'TSLA Momentum', status: 429, latency: 55 },
    { time: '2026-03-23 10:22:41', event: 'cancel_long', strategy: 'TSLA Momentum', status: 200, latency: 78 },
    { time: '2026-03-22 09:35:22', event: 'entry_long_market', strategy: 'NVDA LONG - Mass #2', status: 200, latency: 112 },
  ],
};

/* ========================================================================= */
/* STYLES                                                                      */
/* ========================================================================= */

const thStyle: React.CSSProperties = {
  color: 'var(--text-muted)', background: 'var(--bg-secondary)', textAlign: 'left',
  padding: '8px 12px', fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em',
};

const tdStyle: React.CSSProperties = {
  padding: '8px 12px', fontSize: '0.8125rem', borderBottom: '1px solid var(--border)', color: 'var(--text-secondary)',
};

const btnSecondary: React.CSSProperties = {
  background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)',
  padding: '6px 14px', borderRadius: '8px', fontSize: '0.875rem', cursor: 'pointer',
};

/* ========================================================================= */
/* SUB COMPONENTS                                                              */
/* ========================================================================= */

function EventBadge({ event, category }: { event: string; category: string }) {
  const c = CATEGORY_COLORS[category] || { color: 'var(--text-muted)', bg: 'var(--bg-input)' };
  return (
    <span className="text-xs font-mono font-semibold px-2 py-0.5 rounded-full" style={{ color: c.color, background: c.bg }}>
      {event}
    </span>
  );
}

function CategoryBadge({ category }: { category: string }) {
  const c = CATEGORY_COLORS[category] || { color: 'var(--text-muted)', bg: 'var(--bg-input)' };
  return (
    <span className="text-xs font-semibold px-2 py-0.5 rounded-full capitalize" style={{ color: c.color, background: c.bg }}>
      {category}
    </span>
  );
}

function resolvePayload(payload: string): string {
  return payload.replace(/\{\{(\w+)\}\}/g, (_, key) => {
    const ph = PLACEHOLDERS.find((p) => p.key === `{{${key}}}`);
    return ph ? ph.example : key;
  });
}

/* ========================================================================= */
/* COMPONENT                                                                   */
/* ========================================================================= */

export default function WebhookTemplateDetailV5() {
  const [activeCategory, setActiveCategory] = useState<string>('entry');
  const [selectedEventIdx, setSelectedEventIdx] = useState(0);

  const filteredEvents = template.events.filter((e) => e.category === activeCategory);
  const activeEvent = filteredEvents[selectedEventIdx] ?? filteredEvents[0] ?? null;

  return (
    <div>
      {/* Header */}
      <PageHeader
        title={template.name}
        backHref="/alerts/webhook-templates"
        actions={
          <>
            <button style={btnSecondary}>Edit</button>
            <button style={btnSecondary}>Duplicate</button>
            <button style={{ ...btnSecondary, color: 'var(--red)', borderColor: 'var(--red)' }}>Delete</button>
          </>
        }
      />

      {/* Meta line */}
      <div className="flex items-center gap-3 mb-2 flex-wrap">
        <span className="text-xs px-2 py-0.5 rounded-full" style={{ color: 'var(--accent)', background: 'var(--accent-muted)' }}>
          {template.exchange}
        </span>
        <span className="text-xs" style={{ color: 'var(--green)' }}>
          <span style={{ display: 'inline-block', width: 6, height: 6, borderRadius: '50%', background: 'var(--green)', marginRight: 4 }} />
          Healthy
        </span>
        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
          {template.usedByPortfolios} portfolio{template.usedByPortfolios !== 1 ? 's' : ''}
        </span>
      </div>
      <p className="text-xs font-mono mb-4" style={{ color: 'var(--text-muted)' }}>{template.url}</p>

      {/* Tabs */}
      <TabBar tabs={TABS}>
        {(tab) => (
          <div>
            {/* =========================================================== */}
            {/* TAB 1: EVENT PAYLOADS                                       */}
            {/* =========================================================== */}
            {tab === 'Event Payloads' && (
              <div>
                {/* Category pills */}
                <div className="flex gap-2 mb-4">
                  {['entry', 'exit', 'cancel', 'compliance'].map((cat) => {
                    const count = template.events.filter((e) => e.category === cat).length;
                    const c = CATEGORY_COLORS[cat];
                    return (
                      <button
                        key={cat}
                        className="text-xs px-3 py-1.5 rounded-full transition-colors capitalize"
                        style={{
                          background: activeCategory === cat ? c.bg : 'var(--bg-input)',
                          color: activeCategory === cat ? c.color : 'var(--text-muted)',
                          border: activeCategory === cat ? `1px solid ${c.color}40` : '1px solid var(--border)',
                          cursor: 'pointer',
                          fontWeight: activeCategory === cat ? 600 : 400,
                        }}
                        onClick={() => { setActiveCategory(cat); setSelectedEventIdx(0); }}
                      >
                        {cat} ({count})
                      </button>
                    );
                  })}
                </div>

                {/* Event selector + payload editor */}
                <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
                  {/* Left: event list */}
                  <Card>
                    <div className="flex flex-col gap-1">
                      {filteredEvents.map((evt, i) => (
                        <button
                          key={evt.event}
                          className="flex items-center gap-2 px-3 py-2.5 rounded-lg text-left transition-colors w-full"
                          style={{
                            background: selectedEventIdx === i ? 'var(--accent-muted)' : 'transparent',
                            border: 'none', cursor: 'pointer',
                          }}
                          onClick={() => setSelectedEventIdx(i)}
                        >
                          <span style={{
                            width: 6, height: 6, borderRadius: '50%',
                            background: CATEGORY_COLORS[evt.category]?.color || 'var(--text-muted)',
                            display: 'inline-block', flexShrink: 0,
                          }} />
                          <div className="min-w-0">
                            <p className="text-xs font-mono truncate" style={{
                              color: selectedEventIdx === i ? 'var(--text-primary)' : 'var(--text-secondary)',
                              fontWeight: selectedEventIdx === i ? 600 : 400,
                            }}>
                              {evt.event}
                            </p>
                            <p className="text-xs truncate" style={{ color: 'var(--text-muted)' }}>{evt.label}</p>
                          </div>
                        </button>
                      ))}
                    </div>
                  </Card>

                  {/* Right: payload editor */}
                  <div className="lg:col-span-3">
                    {activeEvent && (
                      <Card>
                        <div className="flex items-center justify-between mb-4">
                          <div className="flex items-center gap-2">
                            <EventBadge event={activeEvent.event} category={activeEvent.category} />
                            <span className="text-sm font-medium">{activeEvent.label}</span>
                          </div>
                          <button style={{ ...btnSecondary, fontSize: '0.75rem', padding: '4px 10px' }}>
                            Test Send
                          </button>
                        </div>

                        {/* Template payload */}
                        <div className="mb-4">
                          <p className="text-xs font-medium mb-1.5" style={{ color: 'var(--text-muted)' }}>Payload Template</p>
                          <div style={{
                            background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: '8px',
                            padding: '12px', fontFamily: 'monospace', fontSize: '0.8125rem', lineHeight: 1.6,
                            whiteSpace: 'pre-wrap', color: 'var(--text-primary)', minHeight: 140,
                          }}>
                            {activeEvent.payload}
                          </div>
                        </div>

                        {/* Resolved preview */}
                        <div>
                          <p className="text-xs font-medium mb-1.5" style={{ color: 'var(--text-muted)' }}>Resolved Preview</p>
                          <div style={{
                            background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: '8px',
                            padding: '12px', fontFamily: 'monospace', fontSize: '0.75rem', lineHeight: 1.6,
                            whiteSpace: 'pre-wrap', color: 'var(--text-muted)', minHeight: 100,
                          }}>
                            {resolvePayload(activeEvent.payload)}
                          </div>
                        </div>
                      </Card>
                    )}
                  </div>
                </div>

                {/* All events overview */}
                <Card className="mt-6">
                  <h4 className="text-sm font-medium mb-3">All Event Payloads Overview</h4>
                  <div style={{ overflowX: 'auto' }}>
                    <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                      <thead>
                        <tr>
                          {['Event Type', 'Category', 'Payload Preview', 'Placeholders'].map((h) => (
                            <th key={h} style={thStyle}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {template.events.map((evt) => {
                          const phCount = (evt.payload.match(/\{\{(\w+)\}\}/g) || []).length;
                          return (
                            <tr
                              key={evt.event}
                              className="cursor-pointer"
                              style={{ transition: 'background 0.1s' }}
                              onClick={() => {
                                setActiveCategory(evt.category);
                                const filtered = template.events.filter((e) => e.category === evt.category);
                                setSelectedEventIdx(Math.max(0, filtered.findIndex((e) => e.event === evt.event)));
                                window.scrollTo({ top: 0, behavior: 'smooth' });
                              }}
                            >
                              <td style={tdStyle}><EventBadge event={evt.event} category={evt.category} /></td>
                              <td style={tdStyle}><CategoryBadge category={evt.category} /></td>
                              <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.7rem', maxWidth: 320 }}>
                                <span className="truncate block" style={{ color: 'var(--text-muted)' }}>
                                  {evt.payload.replace(/\n/g, ' ').substring(0, 90)}...
                                </span>
                              </td>
                              <td style={{ ...tdStyle, textAlign: 'center' }}>{phCount}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </Card>
              </div>
            )}

            {/* =========================================================== */}
            {/* TAB 2: PLACEHOLDERS                                         */}
            {/* =========================================================== */}
            {tab === 'Placeholders' && (
              <div>
                <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
                  Use these placeholders in your payload templates. Values are resolved at delivery time from the strategy signal and portfolio context.
                </p>

                {/* Group by category */}
                {['Signal', 'Order', 'Strategy', 'Portfolio', 'Indicator', 'Meta'].map((cat) => {
                  const items = PLACEHOLDERS.filter((p) => p.category === cat);
                  if (items.length === 0) return null;
                  return (
                    <Card key={cat} className="mb-4">
                      <h4 className="text-sm font-medium mb-3">{cat}</h4>
                      <div style={{ overflowX: 'auto' }}>
                        <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                          <thead>
                            <tr>
                              {['Placeholder', 'Example Value', 'Click to Copy'].map((h) => (
                                <th key={h} style={thStyle}>{h}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {items.map((ph) => (
                              <tr key={ph.key}>
                                <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem', color: 'var(--accent)' }}>{ph.key}</td>
                                <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{ph.example}</td>
                                <td style={tdStyle}>
                                  <button
                                    className="text-xs px-2 py-1 rounded"
                                    style={{ ...btnSecondary, fontSize: '0.7rem', padding: '2px 8px' }}
                                    onClick={() => navigator.clipboard.writeText(ph.key)}
                                  >
                                    Copy
                                  </button>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </Card>
                  );
                })}
              </div>
            )}

            {/* =========================================================== */}
            {/* TAB 3: DELIVERY HISTORY                                     */}
            {/* =========================================================== */}
            {tab === 'Delivery History' && (
              <div>
                {/* Summary metrics */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
                  {[
                    { label: 'Total Deliveries', value: String(template.deliveries.length) },
                    { label: 'Success Rate', value: `${((template.deliveries.filter((d) => d.status >= 200 && d.status < 300).length / template.deliveries.length) * 100).toFixed(0)}%` },
                    { label: 'Avg Latency', value: `${Math.round(template.deliveries.reduce((s, d) => s + d.latency, 0) / template.deliveries.length)}ms` },
                    { label: 'Last Sent', value: template.deliveries[0]?.time.split(' ')[1] || '—' },
                  ].map((m) => (
                    <Card key={m.label}>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{m.label}</p>
                      <p className="text-lg font-bold mt-1">{m.value}</p>
                    </Card>
                  ))}
                </div>

                {/* Delivery timeline chart placeholder */}
                <Card className="mb-4">
                  <ChartPlaceholder label="Delivery timeline: success/failure dots, latency sparkline, event type breakdown" height={200} />
                </Card>

                {/* Delivery log table */}
                <Card>
                  <h4 className="text-sm font-medium mb-3">Recent Deliveries</h4>
                  <div style={{ overflowX: 'auto' }}>
                    <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                      <thead>
                        <tr>
                          {['Time', 'Event', 'Strategy', 'Status', 'Latency'].map((h) => (
                            <th key={h} style={thStyle}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {template.deliveries.map((d, i) => {
                          const ok = d.status >= 200 && d.status < 300;
                          const cat = EVENT_TYPES.find((e) => e.event === d.event)?.category || 'entry';
                          return (
                            <tr key={i}>
                              <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{d.time}</td>
                              <td style={tdStyle}><EventBadge event={d.event} category={cat} /></td>
                              <td style={tdStyle}>{d.strategy}</td>
                              <td style={tdStyle}>
                                <span className="flex items-center gap-1.5">
                                  <span style={{ width: 6, height: 6, borderRadius: '50%', background: ok ? 'var(--green)' : 'var(--red)', display: 'inline-block' }} />
                                  <span style={{ color: ok ? 'var(--green)' : 'var(--red)', fontWeight: 600 }}>{d.status}</span>
                                </span>
                              </td>
                              <td style={{ ...tdStyle, fontFamily: 'monospace' }}>{d.latency}ms</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </Card>
              </div>
            )}

            {/* =========================================================== */}
            {/* TAB 4: SETTINGS                                             */}
            {/* =========================================================== */}
            {tab === 'Settings' && (
              <div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                  {/* Template config */}
                  <Card>
                    <h4 className="text-sm font-medium mb-4">Template Configuration</h4>
                    <div className="flex flex-col gap-3">
                      {[
                        { label: 'Name', value: template.name },
                        { label: 'Exchange', value: template.exchange },
                        { label: 'Webhook URL', value: template.url },
                        { label: 'Created', value: template.createdAt },
                        { label: 'Last Updated', value: template.updatedAt },
                      ].map((row) => (
                        <div key={row.label} className="flex items-center justify-between">
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{row.label}</span>
                          <span className="text-sm font-medium truncate ml-4" style={{ maxWidth: 280 }}>{row.value}</span>
                        </div>
                      ))}
                    </div>
                  </Card>

                  {/* Usage */}
                  <Card>
                    <h4 className="text-sm font-medium mb-4">Used By Portfolios</h4>
                    {template.portfolios.length === 0 ? (
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Not used by any portfolios yet.</p>
                    ) : (
                      <div className="flex flex-col gap-2">
                        {template.portfolios.map((name) => (
                          <div key={name} className="flex items-center gap-2 px-3 py-2 rounded-lg" style={{ background: 'var(--bg-secondary)' }}>
                            <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--accent)', display: 'inline-block' }} />
                            <span className="text-sm">{name}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </Card>
                </div>

                {/* Danger zone */}
                <Card>
                  <h4 className="text-sm font-medium mb-3" style={{ color: 'var(--red)' }}>Danger Zone</h4>
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm">Delete this template</p>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        This will remove the template and disconnect it from {template.usedByPortfolios} portfolio{template.usedByPortfolios !== 1 ? 's' : ''}.
                      </p>
                    </div>
                    <button style={{ ...btnSecondary, color: 'var(--red)', borderColor: 'var(--red)' }}>Delete Template</button>
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
