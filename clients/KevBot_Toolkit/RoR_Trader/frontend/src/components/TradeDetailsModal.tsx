'use client';

/**
 * Trade Details Modal — Portfolio Trade History drawer (roadmap 9ad).
 *
 * Shows per-trade detail + the rendered webhook payload that dispatched
 * for entry and exit + the HTTP delivery result. Pre-live-trading
 * verification: Kevin clicks a row, sees exactly what was sent and
 * whether it was accepted, before pointing anything at real money.
 */

import { useMemo, useState } from 'react';
import Modal from '@/components/Modal';
import { useAlertDetails, type WebhookDelivery } from '@/hooks/queries/useAlerts';

type TradeRow = Record<string, any>;

interface TradeDetailsModalProps {
  trade: TradeRow | null;
  isOpen: boolean;
  onClose: () => void;
  tradeNumber?: number;
}

export default function TradeDetailsModal({
  trade,
  isOpen,
  onClose,
  tradeNumber,
}: TradeDetailsModalProps) {
  if (!trade) return null;

  const title =
    tradeNumber != null ? `Trade #${tradeNumber} — Details` : 'Trade Details';

  return (
    <Modal title={title} isOpen={isOpen} onClose={onClose} width="760px">
      <TradeSummary trade={trade} />
      <div style={{ height: 20 }} />
      <WebhookSection label="Entry Webhook" alertId={trade.entry_alert_id} />
      <div style={{ height: 20 }} />
      <WebhookSection label="Exit Webhook" alertId={trade.exit_alert_id} />
    </Modal>
  );
}

function TradeSummary({ trade }: { trade: TradeRow }) {
  const r = Number(trade.r_multiple ?? 0);
  const pnl = Number(trade.dollar_pnl ?? 0);
  const dir = (trade.direction || 'LONG').toUpperCase();
  const ds = trade.data_source || 'backtest';
  const statusLabel =
    trade.phantom === true
      ? 'Phantom'
      : ds === 'matched' || ds === 'live_executions'
        ? 'Matched'
        : ds === 'live' || ds === 'raw_alerts'
          ? 'Live'
          : ds === 'forward_test'
            ? 'Forward Test'
            : 'Backtest';

  const rows: Array<[string, React.ReactNode]> = [
    ['Strategy', trade.strategy_name ?? `Strategy ${trade.strategy_id}`],
    ['Symbol', <code key="sym" style={{ fontFamily: 'monospace' }}>{trade.symbol ?? '--'}</code>],
    [
      'Direction',
      <span key="dir" style={{ color: dir === 'LONG' ? 'var(--green)' : 'var(--red)', fontWeight: 600 }}>
        {dir}
      </span>,
    ],
    ['Status', statusLabel],
    [
      'Entry',
      <span key="entry" style={{ fontFamily: 'monospace' }}>
        {trade.entry_price != null ? `$${Number(trade.entry_price).toFixed(2)}` : '--'}
        {trade.entry_time ? (
          <span style={{ color: 'var(--text-muted)', marginLeft: 8 }}>
            @ {formatTs(trade.entry_time)}
          </span>
        ) : null}
      </span>,
    ],
    [
      'Exit',
      <span key="exit" style={{ fontFamily: 'monospace' }}>
        {trade.exit_price != null ? `$${Number(trade.exit_price).toFixed(2)}` : '--'}
        {trade.exit_time ? (
          <span style={{ color: 'var(--text-muted)', marginLeft: 8 }}>
            @ {formatTs(trade.exit_time)}
          </span>
        ) : null}
      </span>,
    ],
    ['Exit reason', trade.exit_reason || '--'],
    ['Exec type', trade.exec_type || '--'],
    [
      'RPT Qty',
      <span
        key="rpt"
        style={{ fontFamily: 'monospace' }}
        title="Risk-sized quantity (risk_per_trade / stop distance)"
      >
        {trade.rpt_quantity ?? trade.planned_quantity ?? trade.quantity ?? '--'}
      </span>,
    ],
    [
      'BP Qty',
      <span
        key="bp"
        style={{
          fontFamily: 'monospace',
          color: trade.bp_quantity == null ? 'var(--text-muted)' : 'inherit',
        }}
        title="Buying-power-sized quantity (available BP / entry price)"
      >
        {trade.bp_quantity ?? '--'}
      </span>,
    ],
    [
      'Executed Qty',
      <span
        key="exec"
        style={{ fontFamily: 'monospace' }}
        title="What the webhook carried — min(RPT, BP)"
      >
        {trade.executed_quantity ?? trade.quantity ?? '--'}
      </span>,
    ],
    [
      'R-multiple',
      <span
        key="r"
        style={{ fontFamily: 'monospace', color: r >= 0 ? 'var(--green)' : 'var(--red)' }}
      >
        {r >= 0 ? '+' : ''}
        {r.toFixed(2)}R
      </span>,
    ],
    [
      'Dollar P&L',
      <span
        key="pnl"
        style={{ fontFamily: 'monospace', color: pnl >= 0 ? 'var(--green)' : 'var(--red)' }}
      >
        {pnl >= 0 ? '+' : '-'}$
        {Math.abs(pnl).toLocaleString(undefined, { maximumFractionDigits: 2 })}
      </span>,
    ],
    [
      'Risk / Buying power',
      <span key="bp" style={{ fontFamily: 'monospace' }}>
        ${Number(trade.risk_per_trade ?? 0).toFixed(0)}
        {trade.buying_power_used != null
          ? ` / $${Number(trade.buying_power_used).toFixed(2)}`
          : ''}
      </span>,
    ],
  ];

  return (
    <section>
      <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-secondary)' }}>
        Trade Summary
      </h3>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(140px, max-content) 1fr',
          columnGap: 16,
          rowGap: 6,
          padding: '12px 14px',
          background: 'var(--bg-card)',
          border: '1px solid var(--border)',
          borderRadius: 8,
        }}
      >
        {rows.map(([k, v]) => (
          <SummaryRow key={k} label={k} value={v} />
        ))}
      </div>
    </section>
  );
}

function SummaryRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <>
      <div className="text-xs" style={{ color: 'var(--text-muted)', paddingTop: 2 }}>
        {label}
      </div>
      <div className="text-sm">{value}</div>
    </>
  );
}

function WebhookSection({
  label,
  alertId,
}: {
  label: string;
  alertId: number | null | undefined;
}) {
  const enabled = alertId != null;
  const { data, isLoading, error } = useAlertDetails(enabled ? alertId : null);

  return (
    <section>
      <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-secondary)' }}>
        {label}
      </h3>
      {!enabled ? (
        <EmptyNote>
          No live webhook — this row is from the engine&apos;s historical backtest.
        </EmptyNote>
      ) : isLoading ? (
        <EmptyNote>Loading webhook details…</EmptyNote>
      ) : error ? (
        <EmptyNote red>Failed to load webhook details — {String(error)}</EmptyNote>
      ) : !data?.webhook_deliveries || data.webhook_deliveries.length === 0 ? (
        <EmptyNote>
          Alert fired (id {alertId}) but no webhook deliveries recorded. Either
          the alert was dispatched before webhook groups were wired, or the
          strategy was not included in any portfolio at dispatch time. Newer
          alerts surface a diagnostic row here when a portfolio is missing a
          webhook group.
        </EmptyNote>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {data.webhook_deliveries.map((d, i) => (
            <DeliveryCard key={`${d.webhook_id ?? 'x'}-${i}`} delivery={d} />
          ))}
        </div>
      )}
    </section>
  );
}

function EmptyNote({
  children,
  red = false,
}: {
  children: React.ReactNode;
  red?: boolean;
}) {
  return (
    <div
      className="text-sm"
      style={{
        color: red ? 'var(--red)' : 'var(--text-muted)',
        padding: '10px 14px',
        background: 'var(--bg-input)',
        border: '1px solid var(--border)',
        borderRadius: 8,
      }}
    >
      {children}
    </div>
  );
}

function DeliveryCard({ delivery }: { delivery: WebhookDelivery }) {
  const [payloadOpen, setPayloadOpen] = useState(true);
  const [copied, setCopied] = useState(false);
  const success = !!delivery.success;
  const code = delivery.status_code;
  const prettyPayload = useMemo(() => {
    if (!delivery.payload_sent) return '';
    try {
      return JSON.stringify(JSON.parse(delivery.payload_sent), null, 2);
    } catch {
      return delivery.payload_sent;
    }
  }, [delivery.payload_sent]);

  const copy = async () => {
    if (!delivery.payload_sent) return;
    try {
      await navigator.clipboard.writeText(delivery.payload_sent);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* ignore */
    }
  };

  return (
    <div
      style={{
        padding: '10px 14px',
        background: 'var(--bg-card)',
        border: '1px solid var(--border)',
        borderRadius: 8,
      }}
    >
      {/* Header row */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          flexWrap: 'wrap',
          marginBottom: 6,
        }}
      >
        <span style={{ fontWeight: 600 }}>{delivery.webhook_name || 'Webhook'}</span>
        <span
          className="text-xs px-2 py-0.5 rounded-full font-medium"
          style={{
            color: success ? 'var(--green)' : 'var(--red)',
            background: success ? 'var(--green-muted)' : 'var(--red-muted)',
          }}
          title={code != null ? `HTTP ${code}` : 'No HTTP response'}
        >
          {success ? 'OK' : 'Failed'}
          {code != null ? ` · ${code}` : ''}
        </span>
        {delivery.sent_at ? (
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {formatTs(delivery.sent_at)}
          </span>
        ) : null}
        {delivery.portfolio_id ? (
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
            portfolio {delivery.portfolio_id}
          </span>
        ) : null}
      </div>

      {/* Error row */}
      {delivery.error ? (
        <div
          className="text-xs"
          style={{
            color: 'var(--red)',
            padding: '6px 8px',
            background: 'var(--red-muted)',
            borderRadius: 6,
            marginBottom: 8,
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}
        >
          {delivery.error}
        </div>
      ) : null}

      {/* Payload */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 4 }}>
        <button
          onClick={() => setPayloadOpen((o) => !o)}
          className="text-xs"
          style={{
            color: 'var(--text-muted)',
            background: 'transparent',
            border: 'none',
            cursor: 'pointer',
            padding: 0,
          }}
        >
          {payloadOpen ? '▾' : '▸'} Payload
          {delivery.payload_sent ? ` (${delivery.payload_sent.length} bytes)` : ''}
        </button>
        {delivery.payload_sent ? (
          <button
            onClick={copy}
            className="text-xs px-2 py-0.5 rounded"
            style={{
              color: 'var(--text-secondary)',
              background: 'var(--bg-input)',
              border: '1px solid var(--border)',
              cursor: 'pointer',
            }}
          >
            {copied ? 'Copied!' : 'Copy'}
          </button>
        ) : null}
      </div>
      {payloadOpen && prettyPayload ? (
        <pre
          className="text-xs"
          style={{
            margin: 0,
            padding: '8px 10px',
            background: 'var(--bg-input)',
            border: '1px solid var(--border)',
            borderRadius: 6,
            overflowX: 'auto',
            maxHeight: 280,
            whiteSpace: 'pre',
            fontFamily: 'monospace',
          }}
        >
          {prettyPayload}
        </pre>
      ) : null}
      {payloadOpen && !prettyPayload ? (
        <div className="text-xs" style={{ color: 'var(--text-muted)' }}>
          (no payload recorded)
        </div>
      ) : null}
    </div>
  );
}

function formatTs(ts: string | null | undefined): string {
  if (!ts) return '';
  try {
    const d = new Date(ts);
    if (Number.isNaN(d.getTime())) return ts;
    return d.toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  } catch {
    return ts;
  }
}
