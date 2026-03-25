'use client';

import { useState } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import { useWebhookTemplate, useWebhookDeliveryLog } from '@/hooks/queries/useWebhooks';
import { useDeleteWebhookTemplate } from '@/hooks/mutations/useWebhookMutations';

interface Props {
  templateId: string;
}

const EXCHANGE_COLORS: Record<string, { color: string; bg: string }> = {
  signalstack: { color: '#2196F3', bg: '#2196F320' },
  tradethepool: { color: '#FF9800', bg: '#FF980020' },
  discord: { color: '#7C3AED', bg: '#7C3AED20' },
  slack: { color: '#6B21A8', bg: '#6B21A820' },
  custom: { color: 'var(--text-muted)', bg: 'var(--bg-input)' },
};

export default function WebhookTemplateDetailPage({ templateId }: Props) {
  const { data: template, isLoading } = useWebhookTemplate(templateId);
  const { data: deliveryLog } = useWebhookDeliveryLog();
  const deleteMutation = useDeleteWebhookTemplate();
  const [activeTab, setActiveTab] = useState('Event Payloads');

  if (isLoading) {
    return (
      <div>
        <PageHeader title="Webhook Template" subtitle="Loading..." />
        <Card><div className="animate-pulse h-64" style={{ background: 'var(--bg-input)', borderRadius: 8 }} /></Card>
      </div>
    );
  }

  if (!template) {
    return (
      <div>
        <PageHeader title="Webhook Template" subtitle="Not found" />
        <Card>
          <div className="text-center py-12" style={{ color: 'var(--text-muted)' }}>
            Template not found. <Link href="/alerts/webhook-templates" style={{ color: 'var(--accent)' }}>Back to templates</Link>
          </div>
        </Card>
      </div>
    );
  }

  const exchange = (template.exchange || template.platform || 'custom').toLowerCase();
  const ec = EXCHANGE_COLORS[exchange] || EXCHANGE_COLORS.custom;

  return (
    <div>
      <PageHeader
        title={template.name || 'Untitled Template'}
        subtitle={
          <div className="flex items-center gap-2 mt-1">
            <span className="text-xs px-2 py-0.5 rounded-full font-medium" style={{ color: ec.color, background: ec.bg }}>
              {exchange}
            </span>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
              {template.url || template.config?.url_template || 'No URL configured'}
            </span>
          </div>
        }
        actions={
          <div className="flex items-center gap-2">
            <span className="text-xs px-2 py-1 rounded-full flex items-center gap-1.5"
              style={{ background: 'var(--green)15', color: 'var(--green)' }}>
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--green)' }} />
              Live
            </span>
            <Link href="/alerts/webhook-templates">
              <button className="text-xs px-3 py-1.5 rounded"
                style={{ background: 'var(--bg-input)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}>
                Back
              </button>
            </Link>
            <button className="text-xs px-3 py-1.5 rounded"
              style={{ background: 'var(--red)15', color: 'var(--red)', border: '1px solid var(--red)30' }}
              onClick={() => { if (confirm('Delete this template?')) deleteMutation.mutate(templateId); }}>
              Delete
            </button>
          </div>
        }
      />

      <div className="mt-4">
        <TabBar tabs={['Event Payloads', 'Placeholders', 'Delivery History', 'Settings']}>
          {(tab) => {
            if (tab === 'Event Payloads') return <EventPayloadsTab template={template} />;
            if (tab === 'Placeholders') return <PlaceholdersTab />;
            if (tab === 'Delivery History') return <DeliveryHistoryTab log={deliveryLog} />;
            return <SettingsTab template={template} />;
          }}
        </TabBar>
      </div>
    </div>
  );
}

function EventPayloadsTab({ template }: { template: any }) {
  const events = [
    'entry_long_market', 'entry_long_limit', 'entry_short_market', 'entry_short_limit',
    'exit_signal', 'exit_stop', 'exit_target', 'exit_bar_count',
    'cancel_entry', 'cancel_exit', 'compliance_alert',
  ];
  return (
    <div className="space-y-3">
      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
        Configure payload templates for each of the 11 webhook event types.
      </p>
      {events.map((evt) => (
        <Card key={evt}>
          <div className="flex items-center justify-between">
            <div>
              <span className="text-sm font-mono font-medium" style={{ color: 'var(--text-primary)' }}>{evt}</span>
              <span className="text-xs ml-2 px-2 py-0.5 rounded-full" style={{
                color: evt.startsWith('entry') ? 'var(--green)' : evt.startsWith('exit') ? 'var(--red)' : '#FF9800',
                background: evt.startsWith('entry') ? 'var(--green)15' : evt.startsWith('exit') ? 'var(--red)15' : '#FF980015',
              }}>
                {evt.startsWith('entry') ? 'Entry' : evt.startsWith('exit') ? 'Exit' : evt.startsWith('cancel') ? 'Cancel' : 'Compliance'}
              </span>
            </div>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Payload template</span>
          </div>
          <div className="mt-2 p-3 rounded font-mono text-xs" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
            Configure payload for this event type...
          </div>
        </Card>
      ))}
    </div>
  );
}

function PlaceholdersTab() {
  const categories = [
    { name: 'Signal', items: ['{{signal_type}}', '{{signal_price}}', '{{signal_time}}', '{{trigger_id}}'] },
    { name: 'Order', items: ['{{order_type}}', '{{order_side}}', '{{quantity}}', '{{limit_price}}'] },
    { name: 'Strategy', items: ['{{strategy_name}}', '{{symbol}}', '{{direction}}', '{{timeframe}}'] },
    { name: 'Portfolio', items: ['{{portfolio_name}}', '{{risk_per_trade}}', '{{position_size}}'] },
    { name: 'Indicator', items: ['{{ema_short}}', '{{ema_mid}}', '{{macd_line}}', '{{atr}}'] },
    { name: 'Meta', items: ['{{timestamp}}', '{{engine_version}}', '{{alert_id}}'] },
  ];
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {categories.map((cat) => (
        <Card key={cat.name}>
          <h4 className="text-sm font-medium mb-2" style={{ color: 'var(--text-primary)' }}>{cat.name}</h4>
          <div className="space-y-1">
            {cat.items.map((item) => (
              <div key={item} className="flex items-center justify-between text-xs font-mono py-1 px-2 rounded"
                style={{ background: 'var(--bg-input)' }}>
                <span style={{ color: 'var(--accent)' }}>{item}</span>
                <button className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Copy</button>
              </div>
            ))}
          </div>
        </Card>
      ))}
    </div>
  );
}

function DeliveryHistoryTab({ log }: { log: any[] | undefined }) {
  if (!log || log.length === 0) {
    return (
      <Card>
        <div className="text-center py-12" style={{ color: 'var(--text-muted)' }}>
          No delivery history yet
        </div>
      </Card>
    );
  }
  return (
    <Card>
      <table className="w-full text-xs">
        <thead>
          <tr style={{ color: 'var(--text-muted)' }}>
            <th className="text-left pb-2">Time</th>
            <th className="text-left pb-2">Event</th>
            <th className="text-left pb-2">Status</th>
            <th className="text-left pb-2">Latency</th>
          </tr>
        </thead>
        <tbody>
          {log.map((entry: any, i: number) => (
            <tr key={i} style={{ borderTop: '1px solid var(--border)' }}>
              <td className="py-2">{entry.timestamp || '--'}</td>
              <td>{entry.event_type || '--'}</td>
              <td>
                <span className="w-2 h-2 rounded-full inline-block mr-1"
                  style={{ background: entry.status === 'success' ? 'var(--green)' : 'var(--red)' }} />
                {entry.status_code || '--'}
              </td>
              <td>{entry.latency_ms ? `${entry.latency_ms}ms` : '--'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}

function SettingsTab({ template }: { template: any }) {
  return (
    <div className="space-y-4">
      <Card>
        <h4 className="text-sm font-medium mb-3" style={{ color: 'var(--text-primary)' }}>Template Configuration</h4>
        <div className="grid grid-cols-2 gap-4 text-xs">
          <div><span style={{ color: 'var(--text-muted)' }}>Name:</span> <span>{template.name}</span></div>
          <div><span style={{ color: 'var(--text-muted)' }}>Exchange:</span> <span>{template.exchange || template.platform || 'Custom'}</span></div>
          <div className="col-span-2"><span style={{ color: 'var(--text-muted)' }}>URL:</span> <span className="font-mono">{template.url || template.config?.url_template || 'Not configured'}</span></div>
          <div><span style={{ color: 'var(--text-muted)' }}>Created:</span> <span>{template.created_at || '--'}</span></div>
          <div><span style={{ color: 'var(--text-muted)' }}>Updated:</span> <span>{template.updated_at || '--'}</span></div>
        </div>
      </Card>
      <Card>
        <h4 className="text-sm font-medium mb-3" style={{ color: 'var(--red)' }}>Danger Zone</h4>
        <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
          Deleting this template will remove it from all portfolios that use it.
        </p>
        <button className="text-xs px-3 py-1.5 rounded"
          style={{ background: 'var(--red)15', color: 'var(--red)', border: '1px solid var(--red)30' }}>
          Delete Template
        </button>
      </Card>
    </div>
  );
}
