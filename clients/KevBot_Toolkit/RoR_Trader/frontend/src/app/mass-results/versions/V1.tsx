import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

export default function MassResultsV1() {
  return (
    <div>
      <PageHeader title="Mass Results" subtitle="Saved mass search results" />
      <Card>
        <p style={{ color: 'var(--text-muted)' }}>Browse and compare saved mass search results. Filter by ticker, score, and date.</p>
      </Card>
    </div>
  );
}
