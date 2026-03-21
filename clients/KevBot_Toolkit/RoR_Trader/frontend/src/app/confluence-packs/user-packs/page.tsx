import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

export default function UserPacks() {
  return (
    <div>
      <PageHeader title="User Packs" subtitle="Confluence Packs" />
      <Card>
        <p style={{ color: 'var(--text-muted)' }}>User Packs management — configure indicator groups, interpreters, and triggers.</p>
      </Card>
    </div>
  );
}
