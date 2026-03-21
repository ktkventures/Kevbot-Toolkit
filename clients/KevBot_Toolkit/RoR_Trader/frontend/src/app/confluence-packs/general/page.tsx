import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

export default function General() {
  return (
    <div>
      <PageHeader title="General" subtitle="Confluence Packs" />
      <Card>
        <p style={{ color: 'var(--text-muted)' }}>General management — configure indicator groups, interpreters, and triggers.</p>
      </Card>
    </div>
  );
}
