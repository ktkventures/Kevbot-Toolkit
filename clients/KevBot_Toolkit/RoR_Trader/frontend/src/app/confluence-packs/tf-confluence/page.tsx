import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

export default function TfConfluence() {
  return (
    <div>
      <PageHeader title="Tf Confluence" subtitle="Confluence Packs" />
      <Card>
        <p style={{ color: 'var(--text-muted)' }}>Tf Confluence management — configure indicator groups, interpreters, and triggers.</p>
      </Card>
    </div>
  );
}
