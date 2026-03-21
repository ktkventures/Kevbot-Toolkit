import Card from '@/components/Card';

export default function Settings() {
  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Settings</h1>

      <div className="max-w-2xl space-y-6">
        {/* Display settings */}
        <Card>
          <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
            Display
          </h3>
          <div className="space-y-4">
            <div>
              <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Timezone</label>
              <select
                className="w-full px-3 py-2 rounded-lg text-sm"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
              >
                <option>US/Mountain</option>
                <option>US/Eastern</option>
                <option>US/Pacific</option>
                <option>UTC</option>
              </select>
            </div>
            <div>
              <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Theme</label>
              <select
                className="w-full px-3 py-2 rounded-lg text-sm"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
              >
                <option>Dark (Default)</option>
                <option>Light</option>
                <option>Midnight Blue</option>
              </select>
            </div>
          </div>
        </Card>

        {/* Connections */}
        <Card>
          <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
            Connections
          </h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between py-2">
              <div>
                <p className="text-sm font-medium">Market Data Provider</p>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Polygon.io — Stocks Advanced</p>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full" style={{ background: 'var(--green)' }} />
                <span className="text-xs" style={{ color: 'var(--green)' }}>Connected</span>
              </div>
            </div>
            <div className="flex items-center justify-between py-2 border-t" style={{ borderColor: 'var(--border)' }}>
              <div>
                <p className="text-sm font-medium">Broker</p>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Alpaca — Paper Trading</p>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full" style={{ background: 'var(--text-muted)' }} />
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Not Connected</span>
              </div>
            </div>
          </div>
        </Card>

        {/* Account */}
        <Card>
          <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
            Account
          </h3>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm">kevin@example.com</p>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Member since March 2026</p>
            </div>
            <button className="px-3 py-1.5 rounded text-xs" style={{ background: 'var(--red-muted)', color: 'var(--red)' }}>
              Sign Out
            </button>
          </div>
        </Card>

        <button
          className="w-full px-4 py-2.5 rounded-lg text-sm font-medium"
          style={{ background: 'var(--accent)', color: 'white' }}
        >
          Save Settings
        </button>
      </div>
    </div>
  );
}
