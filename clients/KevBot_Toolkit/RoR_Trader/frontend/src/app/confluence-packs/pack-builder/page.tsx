import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

export default function PackBuilder() {
  return (
    <div>
      <PageHeader title="Pack Builder" subtitle="Confluence Packs" />
      <Card>
        <p style={{ color: 'var(--text-muted)' }}>Pack Builder management — configure indicator groups, interpreters, and triggers.</p>
      </Card>
    </div>
  );
}
