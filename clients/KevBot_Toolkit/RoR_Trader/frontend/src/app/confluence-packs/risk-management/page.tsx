import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

export default function RiskManagement() {
  return (
    <div>
      <PageHeader title="Risk Management" subtitle="Confluence Packs" />
      <Card>
        <p style={{ color: 'var(--text-muted)' }}>Risk Management management — configure indicator groups, interpreters, and triggers.</p>
      </Card>
    </div>
  );
}
