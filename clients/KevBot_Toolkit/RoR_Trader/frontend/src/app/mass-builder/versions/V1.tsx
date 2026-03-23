import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import ChartPlaceholder from '@/components/ChartPlaceholder';

export default function MassBuilderV1() {
  return (
    <div>
      <PageHeader title="Mass Strategy Builder" subtitle="Bulk strategy discovery & optimization" />
      <div className="grid grid-cols-3 gap-6">
        <Card className="col-span-1">
          <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Search Configuration</h3>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Ticker, timeframe, session, confluence combinations, and search parameters.</p>
        </Card>
        <Card className="col-span-2">
          <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Results</h3>
          <ChartPlaceholder label="Mass search results — strategy cards ranked by score" height={400} />
        </Card>
      </div>
    </div>
  );
}
