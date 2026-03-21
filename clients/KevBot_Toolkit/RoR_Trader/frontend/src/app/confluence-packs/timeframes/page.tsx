import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

export default function Timeframes() {
  return (
    <div>
      <PageHeader title="Timeframes" subtitle="Confluence Packs" />
      <Card>
        <p style={{ color: 'var(--text-muted)' }}>Timeframes management — configure indicator groups, interpreters, and triggers.</p>
      </Card>
    </div>
  );
}
