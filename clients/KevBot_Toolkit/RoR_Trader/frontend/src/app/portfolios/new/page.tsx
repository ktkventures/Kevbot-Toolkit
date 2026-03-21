import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import ChartPlaceholder from '@/components/ChartPlaceholder';

export default function NewPortfolio() {
  return (
    <div>
      <PageHeader title="New Portfolio" backHref="/portfolios" />
      <div className="grid grid-cols-5 gap-6">
        <div className="col-span-3">
          <Card className="mb-4">
            <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Portfolio Settings</h3>
            <p style={{ color: 'var(--text-muted)' }}>Name, starting balance, risk scaling, requirement set.</p>
          </Card>
          <Card>
            <ChartPlaceholder label="Combined equity curve + drawdown preview" height={300} />
          </Card>
        </div>
        <div className="col-span-2">
          <Card>
            <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Strategies</h3>
            <p style={{ color: 'var(--text-muted)' }}>Add strategies with risk-per-trade allocation. Live preview updates as you build.</p>
          </Card>
        </div>
      </div>
    </div>
  );
}
