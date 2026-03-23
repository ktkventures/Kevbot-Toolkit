import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

export default function WebhookTemplatesV1() {
  return (
    <div>
      <PageHeader title="Webhook Templates" subtitle="Reusable payload templates for alert delivery" />
      <div className="space-y-4">
        {['TradeThePool - Market Buy', 'TradeThePool - Market Sell', 'Discord Alert'].map((name, i) => (
          <Card key={i}>
            <div className="flex items-center justify-between">
              <p className="font-medium text-sm">{name}</p>
              <div className="flex gap-2">
                <button className="px-3 py-1 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>Edit</button>
                <button className="px-3 py-1 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>Test</button>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
