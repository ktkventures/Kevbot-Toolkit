import Card from '@/components/Card';

export default function AccountSettings() {
  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Account</h1>
      <div className="max-w-2xl space-y-6">
        <Card>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">kevin@example.com</p>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Member since March 2026</p>
            </div>
            <button className="px-3 py-1.5 rounded text-xs" style={{ background: 'var(--red-muted)', color: 'var(--red)' }}>Sign Out</button>
          </div>
        </Card>
      </div>
    </div>
  );
}
