import Card from '@/components/Card';

export default function AlertsV1() {
  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Alerts & Signals</h1>

      {/* Monitor status */}
      <Card className="mb-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="w-3 h-3 rounded-full" style={{ background: 'var(--green)' }} />
            <div>
              <p className="font-medium">Engine: Streaming</p>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                12 symbols | 8,432 ticks | via Polygon.io
              </p>
            </div>
          </div>
          <div className="flex gap-2">
            <button className="px-4 py-2 rounded-lg text-sm" style={{ background: 'var(--red-muted)', color: 'var(--red)' }}>
              Disable Monitoring
            </button>
          </div>
        </div>
      </Card>

      {/* Monitored strategies */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
          Monitored Strategies
        </h3>
        <div className="space-y-2">
          {[
            { name: 'SPY LONG - Mass #1', symbol: 'SPY', status: 'FLAT', alerts: 5 },
            { name: 'NVDA LONG - Mass #2', symbol: 'NVDA', status: 'IN_POSITION', alerts: 3 },
            { name: 'AAPL LONG - Mass #5', symbol: 'AAPL', status: 'FLAT', alerts: 8 },
          ].map((strat, i) => (
            <div
              key={i}
              className="flex items-center justify-between py-2 px-3 rounded-lg"
              style={{ background: 'var(--bg-input)' }}
            >
              <div className="flex items-center gap-3">
                <span className="text-sm font-medium">{strat.name}</span>
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{strat.symbol}</span>
              </div>
              <div className="flex items-center gap-3">
                <span
                  className="text-xs px-2 py-0.5 rounded"
                  style={{
                    color: strat.status === 'IN_POSITION' ? 'var(--green)' : 'var(--text-muted)',
                    background: strat.status === 'IN_POSITION' ? 'var(--green-muted)' : 'var(--bg-card)',
                  }}
                >
                  {strat.status}
                </span>
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{strat.alerts} alerts</span>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Recent alerts */}
      <Card>
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
          Recent Alerts
        </h3>
        <div className="space-y-1">
          {[
            { time: '15:52', type: 'ENTRY', strategy: 'SPY LONG - Mass #3', price: 646.75, tag: '[C]' },
            { time: '15:48', type: 'EXIT', strategy: 'NVDA LONG - Mass #2', price: 173.05, tag: '[C]' },
            { time: '15:31', type: 'ENTRY', strategy: 'AAPL LONG - Mass #5', price: 247.36, tag: '[L]' },
          ].map((alert, i) => (
            <div
              key={i}
              className="flex items-center justify-between py-2.5 border-b last:border-0"
              style={{ borderColor: 'var(--border)' }}
            >
              <div className="flex items-center gap-3">
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{alert.time}</span>
                <span
                  className="text-xs px-1.5 py-0.5 rounded font-medium"
                  style={{
                    color: alert.type === 'ENTRY' ? 'var(--green)' : 'var(--red)',
                    background: alert.type === 'ENTRY' ? 'var(--green-muted)' : 'var(--red-muted)',
                  }}
                >
                  {alert.type}
                </span>
                <span className="text-xs px-1.5 py-0.5 rounded" style={{ background: 'var(--bg-card)', color: 'var(--text-muted)' }}>
                  {alert.tag}
                </span>
                <span className="text-sm">{alert.strategy}</span>
              </div>
              <span className="text-sm font-medium">${alert.price}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
