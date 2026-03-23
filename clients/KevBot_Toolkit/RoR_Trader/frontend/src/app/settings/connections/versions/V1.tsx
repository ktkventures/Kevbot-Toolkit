import Card from '@/components/Card';

export default function SettingsConnectionsV1() {
  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Connections</h1>
      <div className="max-w-2xl space-y-6">
        <Card>
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
          <div className="flex items-center justify-between py-2 border-t mt-2 pt-4" style={{ borderColor: 'var(--border)' }}>
            <div>
              <p className="text-sm font-medium">Broker</p>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Alpaca — Paper Trading</p>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full" style={{ background: 'var(--text-muted)' }} />
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Not Connected</span>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
