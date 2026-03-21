import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

export default function EditPortfolio() {
  return (
    <div>
      <PageHeader title="Edit Portfolio" backHref="/portfolios" />
      <Card>
        <p style={{ color: 'var(--text-muted)' }}>Portfolio builder in edit mode — same layout as New Portfolio with existing data pre-loaded.</p>
      </Card>
    </div>
  );
}
