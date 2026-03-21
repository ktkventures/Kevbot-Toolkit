import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

export default function PortfolioRequirements() {
  return (
    <div>
      <PageHeader title="Portfolio Requirements" subtitle="Manage prop firm rule sets" />
      <div className="space-y-4">
        {['Trade The Pool (TTP)', 'FTMO'].map((name, i) => (
          <Card key={i}>
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">{name}</p>
                <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>Built-in requirement set — 5 rules</p>
              </div>
              <div className="flex gap-2">
                <button className="px-3 py-1 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>View</button>
                <button className="px-3 py-1 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>Clone</button>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
