import Card from '@/components/Card';

export default function SettingsDisplayV1() {
  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Display</h1>
      <div className="max-w-2xl space-y-6">
        <Card>
          <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Preferences</h3>
          <div className="space-y-4">
            <div>
              <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Timezone</label>
              <select className="w-full px-3 py-2 rounded-lg text-sm" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}>
                <option>US/Mountain</option><option>US/Eastern</option><option>US/Pacific</option><option>UTC</option>
              </select>
            </div>
            <div>
              <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Date Format</label>
              <select className="w-full px-3 py-2 rounded-lg text-sm" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}>
                <option>YYYY-MM-DD</option><option>MM/DD/YYYY</option><option>DD/MM/YYYY</option>
              </select>
            </div>
          </div>
        </Card>
        <button className="w-full px-4 py-2.5 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: 'white' }}>Save Settings</button>
      </div>
    </div>
  );
}
