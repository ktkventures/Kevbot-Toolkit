import Card from '@/components/Card';
import MetricCard from '@/components/MetricCard';

export default function Dashboard() {
  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Dashboard</h1>

      {/* Portfolio summary metrics */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <MetricCard label="Total P&L" value="$2,847" delta="+3.6%" positive />
        <MetricCard label="Win Rate" value="58.3%" delta="+2.1%" positive />
        <MetricCard label="Active Strategies" value="22" />
        <MetricCard label="Open Positions" value="3" />
      </div>

      {/* Placeholder charts */}
      <div className="grid grid-cols-2 gap-6 mb-6">
        <Card>
          <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>
            Portfolio Equity Curve
          </h3>
          <div
            className="rounded-lg flex items-center justify-center"
            style={{ background: 'var(--bg-input)', height: 300 }}
          >
            <span style={{ color: 'var(--text-muted)' }}>Chart placeholder — TradingView widget</span>
          </div>
        </Card>
        <Card>
          <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>
            Daily P&L
          </h3>
          <div
            className="rounded-lg flex items-center justify-center"
            style={{ background: 'var(--bg-input)', height: 300 }}
          >
            <span style={{ color: 'var(--text-muted)' }}>Chart placeholder — bar chart</span>
          </div>
        </Card>
      </div>

      {/* Recent alerts */}
      <Card>
        <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>
          Recent Alerts
        </h3>
        {['SPY LONG entry @ $647.80', 'NVDA LONG exit @ $173.05', 'AAPL LONG entry @ $247.55'].map(
          (alert, i) => (
            <div
              key={i}
              className="py-2.5 border-b last:border-0 text-sm"
              style={{ borderColor: 'var(--border)', color: 'var(--text-secondary)' }}
            >
              {alert}
            </div>
          )
        )}
      </Card>
    </div>
  );
}
